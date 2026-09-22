// workdir.js: the working directory - one folder granted once, from which the page lists every material.
//
// This is the whole of what the File System Access API buys the page, and it is chosen BY FEATURE DETECTION and never
// by the browser's name (the plan's 5.0): `'showDirectoryPicker' in window`. Where it is absent nothing here is drawn
// at all - a disabled button with a tooltip is worse than an absent one - and the panel says the one quiet line
// instead. Nothing else in the page changes shape: the pulldown, the file rows, the drop and Reload are the same
// controls doing the same things with one more place to get their bytes from.
//
// WHY A DIRECTORY HANDLE IS WORTH THE TROUBLE, in one paragraph, because it is the only reason this file exists. No
// browser hands a page a file's siblings from a file handle (5.1), so a bare PREFIX_nntc.json cannot load anywhere -
// but a DIRECTORY handle resolves a child by name with no prompt per file, and it can say whether a given file lies
// inside it. So a folder granted once turns every one of the page's problems into a lookup: the pulldown can list the
// materials, a click can open one, a bare .json dropped from that folder can find its own .dds files, and Reload can
// re-read a file the encoder has just rewritten - which a `File` snapshot cannot do at all, because the File API says
// a read of one that has changed on disk must throw.
//
// THE HANDLE IS KEPT IN IndexedDB, which is where a structured-cloneable object survives a visit; the permission is
// NOT kept with it. Chrome decides that separately, and the page asks - queryPermission on the way back in,
// requestPermission behind a click, because requestPermission needs a user activation. What each of those actually
// answers on this machine is written down in README.md, measured rather than assumed.

import { FileSet, hasSeparator } from './intake.js';
import { readDescriptorText, namedFiles, summarise } from './descriptor.js';

// ---------------------------------------------------------------------------
// Feature detection, and nothing else. Two separate questions with two separate answers: a browser could grow one
// picker without the other, and the page draws each control from its own test.
// ---------------------------------------------------------------------------
export function supported() {
    return typeof window !== 'undefined' && 'showDirectoryPicker' in window;
}

export function filePickerSupported() {
    return typeof window !== 'undefined' && 'showOpenFilePicker' in window;
}

// A dropped folder arrives as the same kind of handle as a picked one, where the browser has this.
export function dropHandlesSupported() {
    return typeof DataTransferItem !== 'undefined'
        && typeof DataTransferItem.prototype.getAsFileSystemHandle === 'function';
}

// ---------------------------------------------------------------------------
// The handle across visits: one record in one store.
//
// Every call is wrapped, and a failure is a null rather than a throw. A page in a private window, or one whose site
// data the user has cleared, has no IndexedDB to speak of, and the working directory is a convenience: losing it
// costs one click, and taking the page down over it would cost everything.
// ---------------------------------------------------------------------------
const DB_NAME = 'nntc-webgpu';
const DB_VERSION = 1;
const DB_STORE = 'handles';
const DB_KEY = 'working-directory';

function withStore(mode, run) {
    return new Promise((resolve, reject) => {
        if (typeof indexedDB === 'undefined') { reject(new Error('this browser has no IndexedDB')); return; }
        const open = indexedDB.open(DB_NAME, DB_VERSION);
        open.onupgradeneeded = () => {
            const db = open.result;
            if (!db.objectStoreNames.contains(DB_STORE)) db.createObjectStore(DB_STORE);
        };
        open.onerror = () => reject(open.error || new Error('IndexedDB would not open'));
        open.onblocked = () => reject(new Error('IndexedDB is blocked by another tab of this page'));
        open.onsuccess = () => {
            const db = open.result;
            let request = null;
            try {
                const tx = db.transaction(DB_STORE, mode);
                request = run(tx.objectStore(DB_STORE));
                tx.oncomplete = () => { db.close(); resolve(request ? request.result : undefined); };
                tx.onerror = () => { db.close(); reject(tx.error); };
                tx.onabort = () => { db.close(); reject(tx.error); };
            } catch (e) {
                db.close();
                reject(e);
            }
        };
    });
}

// True when the handle is actually in the store, because the panel says "remembered" on the strength of it and a
// line that promises the folder will be there next time had better be answering the question rather than assuming it.
export async function remember(handle) {
    try {
        await withStore('readwrite', (store) => store.put(handle, DB_KEY));
        return true;
    } catch (e) {
        return false;
    }
}

export async function recall() {
    try {
        const handle = await withStore('readonly', (store) => store.get(DB_KEY));
        // A record that is not a directory handle is not this page's: a browser that stored something else under this
        // key, or a version of this page that stored something else, should not crash the one reading it.
        return handle && handle.kind === 'directory' ? handle : null;
    } catch (e) {
        return null;
    }
}

export async function forget() {
    try { await withStore('readwrite', (store) => store.delete(DB_KEY)); } catch (e) { /* see above */ }
}

// ---------------------------------------------------------------------------
// The grant
// ---------------------------------------------------------------------------

// The dialog opens where it last was, by `id` - the one line that makes picking the same folder twice quick. `mode`
// is read: the page never writes a byte of the user's tree, and asking for less is asking for a smaller prompt.
export async function pick() {
    return await window.showDirectoryPicker({ id: 'nntc-workdir', mode: 'read' });
}

// What the browser says WITHOUT asking the user anything. The plan (5.5) records this as its one open question -
// whether a handle restored from IndexedDB already answers 'granted', which would let the list appear on load with no
// click at all - and the answer is in README.md, from this call.
export async function queryGrant(handle) {
    if (!handle || typeof handle.queryPermission !== 'function') return 'granted';
    try { return await handle.queryPermission({ mode: 'read' }); } catch (e) { return 'prompt'; }
}

// Behind a click, always: requestPermission needs a user activation, so there is no version of this that can run on
// load. Chrome's three-way prompt - this time, every visit, don't allow - is what the user sees the first time it is
// asked for a handle it has seen before.
export async function requestGrant(handle) {
    if (!handle || typeof handle.requestPermission !== 'function') return 'granted';
    return await handle.requestPermission({ mode: 'read' });
}

// ---------------------------------------------------------------------------
// Listing the folder
//
// TWO PASSES, because a folder of hundreds of materials should appear at once and fill in rather than make the user
// wait on a read of every descriptor in it. The first pass is names and byte counts, which `entries()` and
// `getFile()` give without reading a byte of content; the second reads each descriptor and rewrites its line with the
// same summary the bundled entries carry.
//
// WHICH .json FILES ARE CANDIDATES. A PREFIX_nntc.json is one by its name, and goes in the list immediately. Any
// other .json might be an older asset (docs/FORMAT.md: PREFIX.json with "format": "ntc-dds-1"), so it is held back and
// listed only if the second pass finds a descriptor in it - which is also what keeps a *_source_material.json, a
// package.json or anything else in the folder out of the pulldown. Held back rather than shown and removed: a line
// that appears and then vanishes is worse than one that appears a moment later.
// ---------------------------------------------------------------------------
export async function listEntries(dir) {
    // A granted handle enumerates the folder as it is now every time; a session folder has to be walked
    // again to do the same. Same promise either way: what this answers is what is in the folder NOW.
    if (dir.session) await dir.refresh();
    const named = [], others = [];
    for await (const [name, handle] of dir.entries()) {
        if (handle.kind !== 'file') continue;
        const lower = name.toLowerCase();
        if (!lower.endsWith('.json')) continue;
        const certain = lower.endsWith('_nntc.json');
        (certain ? named : others).push({ name: name, handle: handle, certain: certain });
    }
    const all = named.concat(others);
    for (const entry of all) {
        try {
            const file = await entry.handle.getFile();
            entry.bytes = file.size;
            entry.lastModified = file.lastModified;
        } catch (e) {
            entry.bytes = 0;
            entry.lastModified = 0;
        }
        entry.label = entryPrefix(entry) + ' - reading it...';
    }
    named.sort((a, b) => a.name.localeCompare(b.name));
    others.sort((a, b) => a.name.localeCompare(b.name));
    return { listed: named, candidates: others };
}

// The second pass over one entry: read it, and either give it the full line or say it is not a descriptor at all.
// Returns true when the entry belongs in the list.
export async function describeEntry(entry) {
    try {
        const file = await entry.handle.getFile();
        const text = await file.text();
        const read = readDescriptorText(text);
        if (!read.root) {
            // Named *_nntc.json and yet not a descriptor: its name says it is a material, so it stays in the list
            // and says why, rather than sitting for ever under a placeholder that will never be replaced.
            if (entry.certain) entry.label = entryPrefix(entry) + ' - not a descriptor (' + read.why + ')';
            return false;
        }
        entry.root = read.root;
        entry.bytes = file.size;
        entry.label = summarise(entry.name, read.root, text.length).label;
        return true;
    } catch (e) {
        // A file that cannot be read right now keeps its place in the list, but NOT under the placeholder:
        // "reading it..." promises that something is still happening, and once this pass has been and gone nothing
        // is. It says what it is instead, and names the way to try again.
        entry.label = entryPrefix(entry) + ' - could not be read; Reload lists it again';
        return entry.certain === true;
    }
}

// The name a list line is built from: the file's name with the part that is not news taken off it.
function entryPrefix(entry) {
    return entry.name.replace(/_nntc\.json$/i, '').replace(/\.json$/i, '');
}

// ---------------------------------------------------------------------------
// Reading one material out of a directory handle
//
// The refusals are fetchAsset's, word for word where the situation is the same, because they are the same situations:
// a .json that will not parse is very often one the encoder is still writing (the plan's 5.7), and main.js's
// reloadMessage matches on that wording to add "the encoder may still be writing it".
//
// EVERY FILE IS REMEMBERED BY ITS HANDLE and not by the File it produced, which is the whole reason Reload works here
// and cannot work on a dropped file: a handle is a reference to a path and re-reads what is there now.
// ---------------------------------------------------------------------------
export async function readMaterial(dir, name) {
    // A SESSION FOLDER IS WALKED AGAIN FIRST. Its `File` objects are snapshots taken by the last listing, and the
    // File API says a read of one whose file has changed must throw - so opening a material from the pulldown after
    // it has been rewritten would fail, or worse, behave in whatever way a given browser chooses. Reload already
    // re-walks before it re-reads; this gives every other route the same footing, at the cost of one enumeration.
    if (dir.session) await dir.refresh();
    const fileHandle = await getFile(dir, name, name + ' is no longer in ' + dir.name);
    const file = await fileHandle.getFile();
    const text = await file.text();
    const read = readDescriptorText(text);
    if (!read.root) {
        if (!read.parsed) throw new Error(name + ' is not valid JSON: ' + read.why);
        throw new Error(name + ' is not an NNTC descriptor (' + read.why + ')');
    }
    const set = new FileSet();
    set.add(name, new TextEncoder().encode(text), fileHandle);
    for (const named of namedFiles(read.root)) {
        if (hasSeparator(named)) {
            throw new Error(name + ' names ' + named + ' with a path in it, which is not a name this page resolves'
                            + ' against a working directory; docs/FORMAT.md says a reader takes such a name as'
                            + ' written.');
        }
        const handle = await getFile(dir, named, name + ' names ' + named + ', which is not in ' + dir.name);
        const plane = await handle.getFile();
        set.add(named, new Uint8Array(await plane.arrayBuffer()), handle);
    }
    return { name: name, root: read.root, set: set };
}

async function getFile(dir, name, complaint) {
    try {
        return await dir.getFileHandle(name);
    } catch (e) {
        if (e && (e.name === 'NotFoundError' || e.name === 'TypeMismatchError')) throw new Error(complaint);
        throw e;
    }
}

// ---------------------------------------------------------------------------
// A file the page was handed, placed inside the working directory
//
// This is what makes the owner's preferred interaction work once a folder is granted: pick or drop a bare .json and
// the material opens, because `resolve()` says where in the granted tree that file sits and `getFileHandle` fetches
// its siblings from the directory it sits in - a SUBDIRECTORY as readily as the root, which is why the path resolve()
// returns is walked rather than merely tested for null.
//
// A file from anywhere else resolves to null, and the caller says so: the browser does not let a page see the folder
// a file came from, so there is nothing else to try.
// ---------------------------------------------------------------------------
export async function containingDirectory(dir, fileHandle) {
    if (!dir || !fileHandle) return null;
    if (fileHandle.kind !== 'file') return null;
    // A SESSION FOLDER cannot resolve(): it is one folder's files and knows nothing of a tree above or below them.
    // What it can answer is whether a name is one of its own, which is the whole of what the caller wants to know -
    // the handle in hand came out of this folder to begin with, so the name is the right question.
    if (dir.session) return dir.has(fileHandle.name) ? dir : null;
    if (typeof dir.resolve !== 'function') return null;
    let path = null;
    try { path = await dir.resolve(fileHandle); } catch (e) { return null; }
    if (!path || !path.length) return null;
    let here = dir;
    for (let i = 0; i < path.length - 1; i++) {
        try { here = await here.getDirectoryHandle(path[i]); } catch (e) { return null; }
    }
    return here;
}

// The files a set is short of, fetched from a directory by name. Returns the names it could not find, so the caller
// can fall back to the message it would have written anyway.
export async function fillFrom(dir, names, set) {
    const absent = [];
    for (const name of names) {
        try {
            const handle = await dir.getFileHandle(name);
            const file = await handle.getFile();
            set.add(name, new Uint8Array(await file.arrayBuffer()), handle);
        } catch (e) {
            absent.push(name);
        }
    }
    return absent;
}

// ---------------------------------------------------------------------------
// A DROPPED FOLDER IN A BROWSER THAT HAS NO HANDLE FOR ONE
//
// Firefox and Safari have no showDirectoryPicker and no getAsFileSystemHandle, so neither the button above nor
// the handle-carrying drop exists there. What they do have, and have had since Firefox 50, is
// webkitGetAsEntry(): a dropped folder arrives as a FileSystemDirectoryEntry, its reader lists it, and every
// file in it reads. So a dropped folder works there, and the page does not say no to something it can do.
//
// It is the SAME working directory to everything above: the object below answers kind, name, entries() and
// getFileHandle() the way a FileSystemDirectoryHandle does, so listEntries, describeEntry, readMaterial and
// fillFrom take one without being told which they were given, and the pulldown, Reload and the status line are
// the controls they already were.
//
// IT RE-WALKS THE FOLDER on every listing, which is what makes Reload mean here what it means there: a
// material the encoder has just written joins the list, and a file it has just rewritten comes back as a fresh
// File rather than a stale snapshot the File API would refuse to read. The entry is live for as long as the
// page is; the one thing it is not is storable, so this folder does not survive a reload of the page and there
// is no Reopen for it. That is the whole of the difference, and the status line says it in one clause.
// ---------------------------------------------------------------------------

// The second of the two drop shapes, tested separately from the first: a browser could have either.
export function dropEntriesSupported() {
    return typeof DataTransferItem !== 'undefined'
        && typeof DataTransferItem.prototype.webkitGetAsEntry === 'function';
}

// readEntries hands back a BATCH rather than the folder, and the contract is to call it until it answers with
// an empty one. The cap is there because a page that walks fifty thousand files to fill a pulldown has stopped
// being a viewer; the folder says it was cut short rather than quietly showing part of itself.
const SESSION_LIMIT = 2000;

function readBatch(reader) {
    return new Promise((resolve, reject) => reader.readEntries(resolve, reject));
}

function entryFile(entry) {
    return new Promise((resolve, reject) => entry.file(resolve, reject));
}

// NOT WALKED HERE. listEntries() walks a session folder itself - it has to, because that is what makes Reload
// re-list - and every route to this object goes through adoptDirectory and therefore through listEntries before
// anything reads the folder. Walking it here as well simply did it twice on the way in.
export async function sessionDirectory(dirEntry) {
    return new SessionDirectory(dirEntry);
}

class SessionDirectory {
    constructor(dirEntry) {
        this.kind = 'directory';
        this.name = dirEntry.name;
        this.session = true;      // asked about CAPABILITY - can this be remembered - never about the browser
        this.entry = dirEntry;
        this.files = new Map();
        this.truncated = false;
    }

    // One folder, its own files, read afresh. No recursion: the granted kind resolves inside a tree, this is one
    // directory, and a viewer that silently walked a whole subtree on a drop would be a surprise.
    async refresh() {
        const reader = this.entry.createReader();
        const files = new Map();
        let truncated = false;
        let jsonCount = 0;
        for (;;) {
            const batch = await readBatch(reader);
            if (!batch.length) break;
            for (const entry of batch) {
                if (!entry.isFile) continue;
                // THE CAP IS ON MATERIALS, NOT ON FILES. Counting every file meant a folder could be cut between a
                // descriptor and the .dds files it names, and opening that material then said "names m1_lat1.dds,
                // which is not in NAME" about a file sitting right there. Only the .json files are counted, so a
                // descriptor that is listed always has its whole folder behind it.
                if (/\.json$/i.test(entry.name) && jsonCount >= SESSION_LIMIT) { truncated = true; break; }
                if (/\.json$/i.test(entry.name)) jsonCount++;
                try { files.set(entry.name, await entryFile(entry)); } catch (e) { /* unreadable: not listed */ }
            }
            if (truncated) break;
        }
        this.files = files;
        this.truncated = truncated;
    }

    has(name) { return this.files.has(name); }

    async *entries() {
        for (const [name, file] of this.files) yield [name, fileHandle(name, file)];
    }

    async getFileHandle(name) {
        const file = this.files.get(name);
        if (!file) {
            const e = new Error(name + ' is not in ' + this.name);
            e.name = 'NotFoundError';
            throw e;
        }
        return fileHandle(name, file);
    }
}

// getFile() is async on a real handle and every caller awaits it, so this one is too and they cannot tell.
function fileHandle(name, file) {
    return { kind: 'file', name: name, getFile: async () => file };
}
