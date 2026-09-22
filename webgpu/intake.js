// intake.js: every way a material gets into the page, ending in the SAME call.
//
// The plan's rule (section 5.6): every route - the bundled list, a select-all, a dropped group, the one-at-a-time rows,
// and later a working directory or a .zip - ends in one function that is handed a set of files by name and a
// descriptor, and either replaces what is on screen or names what is missing. Nothing renders until every file has
// passed, and a refusal leaves the previous material where it was.
//
// The name resolution is docs/FORMAT.md section 1's, exactly: a name with a path separator is taken as written - and
// from a selection of files there is nothing to take it from, so it is reported rather than guessed at - and any other
// name is a base name, looked up in the set that came with it and then in the URL directory the descriptor came from.
//
// What no browser offers, and the reason the rows exist: NO BROWSER HANDS A PAGE A FILE'S SIBLINGS FROM A FILE HANDLE.
// A bare .json therefore cannot load on its own, anywhere. What the page does instead of refusing and forgetting is
// REMEMBER: every file offered - by a selection, by a drop, in however many goes it takes and from however many
// folders - is kept in one pending set (main.js's `pending`), and the material opens the moment that set satisfies a
// descriptor. Selecting the group is then a way of filling the set in one action rather than the only way in, and a
// descriptor in one folder can meet its .dds files in another, which no single selection can do at all.

import { isDescriptorText, readDescriptorText, namedFiles } from './descriptor.js';

export function baseName(name) {
    const i = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
    return i < 0 ? name : name.substring(i + 1);
}

export function hasSeparator(name) {
    return name.indexOf('/') >= 0 || name.indexOf('\\') >= 0;
}

// A set of files the page has been handed, by base name. `sizes` keeps each one's byte count for the panel's rows,
// and `sources` where each came from: the File object itself for a file a user offered, so that a load can hand
// Reload the handles of exactly the files it used, and the URL for one that was fetched.
export class FileSet {
    constructor() {
        this.bytes = new Map();
        this.sizes = new Map();
        this.sources = new Map();
    }
    add(name, bytes, source) {
        const key = baseName(name);
        this.bytes.set(key, bytes);
        this.sizes.set(key, bytes.length);
        this.sources.set(key, source || key);
    }
    get(name) { return this.bytes.get(baseName(name)); }
    has(name) { return this.bytes.has(baseName(name)); }
    size(name) { return this.sizes.get(baseName(name)); }
    get names() { return Array.from(this.bytes.keys()); }
}

// The files a later selection or drop adds to the ones already offered. A name already in the set is REPLACED by the
// newer file: a person re-picking a file means the new one. Nothing here looks at where a file came from - the set is
// keyed by base name and by nothing else - which is what lets the two halves of a material arrive from two folders.
export function mergeSets(base, extra) {
    const set = new FileSet();
    for (const from of [base, extra]) {
        if (!from) continue;
        for (const name of from.names) set.add(name, from.get(name), from.sources.get(name));
    }
    return set;
}

// The File objects a dialog or a drop delivered, read into a FileSet. A file that cannot be read - a dropped file the
// encoder has since rewritten is the case the File API names - is reported by name with what to do about it.
//
// EITHER KIND OF THING IS ACCEPTED, and the difference between them matters later rather than here: a `File` is a
// SNAPSHOT, which the File API says must throw once the file has changed on disk, and a `FileSystemFileHandle` is a
// REFERENCE TO A PATH, which re-reads what is there now. So where the browser gives handles - showOpenFilePicker, a
// drop that carries them, the working directory - the handle is what the set remembers, and Reload keeps working
// through a re-encode rather than asking to be given the files again.
export async function readFileList(items) {
    const set = new FileSet();
    const failed = [];
    for (const item of items) {
        try {
            const file = typeof item.getFile === 'function' ? await item.getFile() : item;
            const buffer = await file.arrayBuffer();
            // What the set remembers as this name's origin, so that a load can hand Reload the handles of exactly
            // the files the descriptor used - which, once a set is assembled over several goes, is no longer the
            // same thing as 'the files of the last selection'.
            set.add(file.name, new Uint8Array(buffer), item);
        } catch (e) {
            failed.push(item && item.name ? item.name : '(a file with no name)');
        }
    }
    if (failed.length) {
        throw new Error('These files could not be read: ' + failed.join(', ')
                        + '. If they changed on disk since they were selected, select or drop them again.');
    }
    return set;
}

// The descriptors in a set: every .json whose text parses AND whose format is nntc-dds-1 or ntc-dds-1. A
// *_source_material.json parses and is not one, so it is quietly ignored, which is what makes "select everything in
// the folder" work.
export function findDescriptors(set) {
    const found = [];
    const decoder = new TextDecoder();
    for (const name of set.names) {
        if (!name.toLowerCase().endsWith('.json')) continue;
        const root = isDescriptorText(decoder.decode(set.get(name)));
        if (root) found.push({ name: name, root: root });
    }
    found.sort((a, b) => a.name.localeCompare(b.name));
    return found;
}

// What a descriptor still needs from a set: the names it can use, the ones missing, and the ones it cannot resolve at
// all because they carry a path separator.
export function resolve(root, set) {
    const wanted = namedFiles(root);
    const have = [], missing = [], unresolvable = [];
    for (const name of wanted) {
        if (hasSeparator(name)) { unresolvable.push(name); continue; }
        if (set && set.has(name)) have.push(name); else missing.push(name);
    }
    return { wanted: wanted, have: have, missing: missing, unresolvable: unresolvable };
}

// Names in a sentence: 'a', 'a and b', 'a, b and c'.
function andList(names) {
    if (names.length < 2) return names.join('');
    return names.slice(0, -1).join(', ') + ' and ' + names[names.length - 1];
}

// The ONE sentence an incomplete set gets, with the remedy - the single place this is worded, so that the message at
// the top of the panel and anything else that reports a shortfall cannot drift apart. It names the descriptor and the
// files verbatim, because the whole point of the rows is that a wrong or missing file is caught by name rather than
// rendered, and it says what the page will do with what it already has, which is the part a user cannot guess.
export function missingMessage(status, descriptorName) {
    const who = descriptorName || 'The descriptor';
    const parts = [];
    if (status.missing.length) {
        parts.push(who + ' needs ' + andList(status.missing) + '. Select or drop '
                   + (status.missing.length === 1 ? 'it' : 'them') + ' too - from any folder, in as many goes as it'
                   + ' takes: the files offered so far are kept, and the material opens as soon as the set is'
                   + ' complete.');
    }
    if (status.unresolvable.length) {
        parts.push(who + ' names ' + andList(status.unresolvable)
                   + ' with a path in it, which cannot be resolved from a selection of files.');
    }
    return parts.join(' ');
}

// A bundled material: the descriptor by URL, then every .dds it names from the descriptor's own directory, which is
// what docs/FORMAT.md says a reader does with a base name.
export async function fetchAsset(url) {
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error('cannot fetch ' + url + ' (' + response.status + ')');
    const text = await response.text();
    // The two ways a .json can fail to be a descriptor are two different sentences, because they are two different
    // situations and only one of them is the reader's mistake. A file that will not PARSE is very often a file the
    // encoder is still writing - the encode-look-encode loop of the plan's 5.7 hits it every time Reload is pressed
    // a second too early - and telling that reader it has no "format" key is untrue of a file that has no complete
    // keys at all to be missing one of. The wording here is what main.js's reloadMessage matches on to add "the
    // encoder may still be writing it; reload again when it has finished", which is the whole of the remedy.
    const read = readDescriptorText(text);
    if (!read.root) {
        if (!read.parsed) throw new Error(url + ' is not valid JSON: ' + read.why);
        throw new Error(url + ' is not an NNTC descriptor (' + read.why + ')');
    }
    const root = read.root;
    const dir = url.substring(0, url.lastIndexOf('/') + 1);
    const set = new FileSet();
    set.add(baseName(url), new TextEncoder().encode(text), url);
    for (const name of namedFiles(root)) {
        if (hasSeparator(name)) {
            throw new Error(url + ' names ' + name + ' with a path in it; a served asset names its files by base name');
        }
        const fileResponse = await fetch(dir + name, { cache: 'no-store' });
        if (!fileResponse.ok) {
            throw new Error('cannot fetch ' + dir + name + ' (' + fileResponse.status + ')');
        }
        set.add(name, new Uint8Array(await fileResponse.arrayBuffer()), dir + name);
    }
    return { name: baseName(url), root: root, set: set, url: url };
}
