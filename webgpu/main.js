// main.js: the state object, the frame loop, the keys and the mouse, and the wiring between the panel and the renderer.
//
// THE ONE RULE THAT KEEPS THE PANEL AND THE PICTURE FROM DISAGREEING (the plan's 6.3): there is one state object and
// nothing reads the DOM to learn the state. Every change - a key, a mouse drag, a control's event, a URL field, Reset,
// Reload, the selector - goes through set(), which writes the object, marks the frame dirty and calls draw(), and
// draw() assigns every readout and the overlay's two lines from the object. The native viewers' `g` struct
// (../viewer/main.cpp:175-200) is exactly this object, and its field names carry over.

import { Renderer, Z_MIN, Z_MAX, Z_SPEED, XY_SPEED, ROT_SPEED } from './renderer.js';
import { Overlay, cameraLine, fpsLine } from './overlay.js';
import { buildAsset, summarise } from './descriptor.js';
import { FileSet, mergeSets, fetchAsset, readFileList, findDescriptors, resolve, missingMessage } from './intake.js';
import { bmpBytes, pngBlob, offerDownload, shotName } from './capture.js';
import * as workdir from './workdir.js';
import * as panel from './panel.js';

const el = (id) => document.getElementById(id);

// WHAT RESET PUTS BACK, and therefore what a fresh material starts at: every field a KEY can reach.
//
// The owner's reason for it is accidents - "all the state a user could accidently change by hitting keys" - so the
// line is drawn exactly there: the camera, the geometry, the filter, anisotropy, mips, the level-1 LOD shift, the raw
// views, the renormalise flag and the shown texture. `unpack` and `colorSpace` are NOT here and are not reset: no key
// reaches either, they are about the device and the display rather than about the view, and resetting `unpack` would
// re-read and re-upload every texture behind a button whose label says nothing of the sort. `overlay` stays too, for
// the same reason - it is the strip, not the picture - and so does the strict switch beside the log, which is about
// how a failure is reported.
//
// THE NATIVE SPACE RESETS LESS THAN THIS (../viewer/main.cpp:826-828): the camera, the two debug constant vectors
// const0 and const1, and tex_shown - and nothing else. In this page's names that is the camera, the raw views, the
// renormalise flag and the shown texture. It leaves the cube, the filter, anisotropy, mips and the level-1 LOD
// shift exactly as they were, because g.lod_bias is its own flag written into const0.z at draw time rather than a
// value the reset touches. Here the owner asked for all of it, so the geometry, the filter, anisotropy, mips and
// the shift are reset too; that is a deliberate difference from the native key, not a transliteration of it.
const DEFAULTS = {
    x: 0, y: 0, z: -3.0, yaw: 0, pitch: 0,
    geom: 'quad',        // quad, cube or sphere; the key C cycles them in that order
    allTextures: false,  // one object per output texture, spread along X (the key G)
    orbit: false,        // the camera's yaw turning by itself, 30 degrees a second (the key O)
    filterMode: 2,      // trilinear
    aniso: true,        // applies in trilinear mode only
    lodBias: true,      // level 1's gradients scaled by 2^lod_bias_level1: the encoder's 1:1 mip rule
    mipsOn: true,
    texShown: 0,
    raw0: false, raw1: false, renorm: false,
    channel: 'rgb',     // which channels of an as-stored view are drawn; a single one is drawn as grey
};

// The state, with the native viewer's defaults (../viewer/main.cpp's State).
const state = Object.assign({}, DEFAULTS, {
    unpack: false,          // filled in from the adapter and the ?unpack field
    overlay: true,
    colorSpace: 'srgb',     // the native viewers' own, and the only one they can draw (the plan's 6.5)
});

let renderer = null;
let overlay = null;
let asset = null;
let dirty = true;
let halted = false;  // ?strict=1 and a WebGPU refusal: the frame loop stops and the message stays in front of you
// Where the material on screen came from, so Reload knows what to re-read (5.7).
let source = null;   // { kind: 'bundled', url }, { kind: 'files', files, descriptorName } or { kind: 'workdir', dir, name }
let bundled = [];    // the assets.json list, each with its descriptor and its pulldown line

// THE WORKING DIRECTORY (5.5), and it is ONE PULLDOWN WITH TWO GROUPS rather than a second control: the owner's
// "pulldown of JSON files with reload" for a chosen directory is the same <select> the bundled materials are in, so
// a user learns one control whose contents depend on where the page is pointed.
//
// `granted` is the handle the page is actually working from; `stored` is one recalled from IndexedDB that the
// browser has not (yet) granted, which is the "Reopen out_material" case and the only reason the two are separate.
let granted = null;         // FileSystemDirectoryHandle, or null
let stored = null;          // a handle from a previous visit awaiting the click that asks for it
let entries = [];           // what listEntries found in `granted`, in pulldown order
let remembered = false;     // the handle went into IndexedDB, so it survives this visit
let listing = 0;            // the enumeration a lazy descriptor read belongs to, so a stale one cannot write a label
let loadVerb = 'Loaded';    // 'Reloaded' for the length of a reload, so one event writes one line
let shotWanted = null;      // the url ?asset= resolved to, so ?shot can refuse a frame of something else
let describing = Promise.resolve();   // the second pass in flight, for the one caller that needs the finished list
const DIR_PREFIX = 'dir:';  // how a working-directory entry is spelled in the <select>; a bundled one is its url
let selected = null;        // the pulldown's value, kept across a rebuild that replaces every option

// THE FILES OFFERED SO FAR BUT NOT YET LOADED, and the whole of what accumulating cost. The set used to be built
// fresh inside loadFiles and dropped on the floor when it fell short, so a .json picked on its own was refused and
// forgotten, and the .dds files picked next had nothing to join - the owner hit exactly that. It lives out here
// instead, at the same altitude as `source` and `asset`, and every offer merges into it: pick the descriptor, then
// pick its planes, and the material opens on the second pick. A name already in it is replaced by the newer file,
// and it is emptied when a material loads and when one is opened from the pulldown, so that the leftovers of an
// abandoned attempt cannot quietly satisfy a later descriptor - a mistake that would be very hard to see.
let pending = new FileSet();

function set(field, value) {
    if (state[field] === value) return;
    state[field] = value;
    dirty = true;
    draw();
}

// Several fields at once, for the changes that are one user action but more than one field: `1` and `2` are a pair
// (the native viewer turns the other off, ../viewer/main.cpp:817-818) and reset is the whole set. Going through one
// place keeps the rule of 6.3 - nothing writes a field without the panel being redrawn from it afterwards.
function setAll(fields) {
    let changed = false;
    for (const key of Object.keys(fields)) {
        if (state[key] === fields[key]) continue;
        state[key] = fields[key];
        changed = true;
    }
    if (!changed) return;
    dirty = true;
    draw();
}

// Reset (key Space, and the two buttons). Deliberately NOT what reload() does and not reachable from it: this one
// throws the view away, that one keeps every bit of it. See DEFAULTS above for what "all of it" means.
function resetView() {
    setAll(DEFAULTS);
    // setAll returns early when nothing changed, and a reset pressed twice should still be seen to do nothing rather
    // than leave a control that some other path wrote out of step.
    draw();
    dirty = true;
}

function draw() {
    panel.renderShow(asset, state);
    panel.renderSample(state);
    panel.renderCamera(state);
    panel.renderDisplay(state);
    if (renderer) panel.renderDecode(renderer, asset, state);
    if (asset) panel.renderTextureSelect(asset, state);
}

// ---------------------------------------------------------------------------
// The URL fields. They mirror the native flag names (--nomips, --nobias, --noaniso, --renorm, --raw0, --raw1, --tex,
// --cube, --z, --yaw, --pitch), with ?unpack= and ?overlay= beside them, so a reader of the viewers recognises them.
// ---------------------------------------------------------------------------
const params = new URLSearchParams(location.search);
// The filter modes by name, in the order the sampler slots are in: the index IS state.filterMode.
const FILTERS = ['point', 'bilinear', 'trilinear'];
// The order C cycles them in.
const GEOMETRIES = ['quad', 'cube', 'sphere'];
// M7's rate, asked for by the owner as "30 degrees per second".
const ORBIT_DEGREES_PER_SECOND = 30;
function flag(name) { return params.has(name) && params.get(name) !== '0'; }
function num(name, dflt) {
    const value = parseFloat(params.get(name));
    return params.has(name) && isFinite(value) ? value : dflt;
}

function applyUrlFields() {
    state.z = num('z', state.z);
    state.x = num('x', state.x);
    state.y = num('y', state.y);
    state.yaw = num('yaw', state.yaw);
    state.pitch = num('pitch', state.pitch);
    state.texShown = Math.max(0, num('tex', 0) | 0);
    // ?cube keeps working and names the native --cube; ?geom= is the one that can say sphere.
    state.geom = flag('cube') ? 'cube' : state.geom;
    const geom = (params.get('geom') || '').toLowerCase();
    if (GEOMETRIES.indexOf(geom) >= 0) state.geom = geom;
    state.allTextures = flag('row');
    state.orbit = flag('orbit');
    state.mipsOn = !flag('nomips');
    state.lodBias = !flag('nobias');
    state.aniso = !flag('noaniso');
    state.renorm = flag('renorm');
    state.raw0 = flag('raw0');
    state.raw1 = flag('raw1');
    // Under ?shot the overlay is OFF unless it is asked for: the strip names the file and its format, which is
    // exactly what two otherwise identical renders differ in, so a capture that carried it could not be compared.
    state.overlay = flag('shot') ? params.get('overlay') === '1' : params.get('overlay') !== '0';
    // ?chan=rgb|r|g|b|a. An unknown value is ignored rather than refused: a link is a thing people edit by hand,
    // and the cost of a typo should be the default channel and not a page that will not open.
    const chan = (params.get('chan') || '').toLowerCase();
    if (['rgb', 'r', 'g', 'b', 'a'].indexOf(chan) >= 0) state.channel = chan;
    // ?filter=point|bilinear|trilinear, the keys P, B and T. Unknown values are ignored rather than refused, as
    // ?chan= is: a link is a thing people edit by hand.
    const filter = FILTERS.indexOf((params.get('filter') || '').toLowerCase());
    if (filter >= 0) state.filterMode = filter;
}

// ?unpack=0|1, in the shape of the native --coopvec 0|1 and --linalg 0|1: absent is automatic, 1 forces the CPU
// unpack on any device, and 0 asks for hardware BC and is refused by name on a device that has none.
function wantedUnpack() {
    if (!params.has('unpack')) return null;
    return params.get('unpack') !== '0';
}

// ?strict=1 and the checkbox beside the log: what a WebGPU refusal DOES, not whether it is noticed. See
// renderer.onError in main() for the whole of it. It is a plain variable and not a state field because Reset must
// not reach it - it is about how the page reports rather than about the view, the same line ?unpack and the colour
// space are on - and it is a `let` because the checkbox changes it LIVE. Nothing about the device depends on it, so
// a user who has just seen something odd gets more detail about the picture that is on screen rather than a fresh
// page that has forgotten it.
let strict = flag('strict');

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

// WHAT RELOAD WILL DO, said before it is pressed rather than discovered by pressing it.
//
// Reload means four different things depending on how the material on screen got here, and the differences are
// not cosmetic: whether the folder is re-listed, whether the bytes come back as they are on disk NOW, and
// whether a file rewritten since will re-read at all or throw. A user cannot be expected to know which of the
// four they are in - the routes look identical once a picture is up - so the Load group says it.
//
// The distinction that matters most is HANDLE versus FILE. A handle is a reference to a path and re-reads what
// is there now; a File is a snapshot of bytes with a size and an mtime, and the File API says a read of one
// whose file has changed must throw. So a material dropped as loose files is the one route where Reload can
// fail for the ordinary reason - the encoder wrote it again - and that is worth a sentence in advance.
function reloadNote() {
    if (!source) return '';
    if (source.kind === 'bundled') {
        return 'Reload re-fetches this material from the files bundled with this page.';
    }
    if (source.kind === 'workdir') {
        // The folder RELOAD RE-LISTS is the one the page was given, which is not always the one this material
        // was read out of: a material in a subfolder is read from there and the root is what gets enumerated.
        const where = granted ? granted.name : source.dir ? source.dir.name : null;
        const kept = source.dir && source.dir.session
            ? ' The page cannot keep this folder when you reload the page itself, so drop it again after that.' : '';
        return 'Reload re-reads this material from disk and lists ' + (where ? where : 'the folder')
               + ' again, so a material you encode in the meantime joins the pulldown.' + kept;
    }
    // The 'files' route: handles or snapshots decides whether a rewrite survives a Reload.
    const held = (source.files || []);
    const snapshots = held.filter((f) => typeof f.getFile !== 'function').length;
    if (!held.length) return '';
    if (!snapshots) {
        return 'Reload re-reads these files from disk. Give the page the folder they are in as well, and Reload'
               + ' lists that folder again too.';
    }
    return 'Reload re-reads ' + (snapshots === held.length ? 'these files' : 'some of these files')
           + ' from the copies your browser took when you dropped them. A file you have rewritten since then will'
           + ' refuse to re-read, and you drop it again. Drop the folder itself on the page instead, and Reload'
           + ' re-reads and re-lists it however often you rewrite what is in it.';
}

// Drawn wherever the answer can have changed: after a load, and after Forget drops the folder one came from.
function renderReloadNote() {
    const line = el('reload-note');
    if (!line) return;               // the markup is optional; the page works without the hint
    line.textContent = reloadNote();
    line.hidden = !line.textContent;
}
async function showAsset(name, root, set_, sourceLabel) {
    // Nothing on screen is replaced until the whole new set has parsed: buildAsset throws on any refusal, and the
    // previous material stays where it was.
    const built = buildAsset(name, root, set_.bytes, state.unpack);
    await renderer.setAsset(built);
    asset = built;
    if (state.texShown >= built.texturesOut) state.texShown = 0;
    // ONE EVENT, ONE LINE. Reload goes through the same three loaders an ordinary open does, so the news is
    // written here and the verb says which it was; a second sentence from the caller said the same thing twice in
    // the log, and the log is the record someone pastes into a bug report.
    panel.setStatus(loadVerb + ' ' + loadedName(name) + '.');
    panel.renderAsset(built, renderer, sourceLabel);
    renderReloadNote();
    draw();
    dirty = true;
}

async function loadBundled(url) {
    const got = await fetchAsset(url);
    source = { kind: 'bundled', url: url };
    // A material opened from the pulldown ends whatever was being assembled by hand: see `pending` above for why an
    // abandoned attempt's leftovers are worse than nothing.
    pending = new FileSet();
    await showAsset(got.name, got.root, got.set, 'from ' + url);
    panel.renderFileRows(got.name, resolve(got.root, got.set), got.set, false);
    return true;
}

// ---------------------------------------------------------------------------
// The working directory
// ---------------------------------------------------------------------------

// The pulldown, rebuilt whole: the working directory's group first, because it is what the owner came for, and the
// bundled group under it - never INSTEAD of it, since the samples are the reference material a user compares their
// own encode against and taking them away the moment a folder is granted would be a poor trade.
function rebuildAssetSelect() {
    const select = el('asset-select');
    select.innerHTML = '';
    if (granted) {
        const group = document.createElement('optgroup');
        group.label = 'Working directory: ' + granted.name;
        for (const entry of entries) {
            const option = document.createElement('option');
            option.value = DIR_PREFIX + entry.name;
            option.textContent = entry.label;
            group.appendChild(option);
        }
        if (!entries.length) {
            const option = document.createElement('option');
            option.disabled = true;
            option.textContent = '(no materials in this folder)';
            group.appendChild(option);
        }
        select.appendChild(group);
    }
    const shipped = document.createElement('optgroup');
    shipped.label = 'Bundled with this page';
    for (const entry of bundled) {
        const option = document.createElement('option');
        option.value = entry.url;
        option.textContent = entry.label;
        shipped.appendChild(option);
    }
    select.appendChild(shipped);
    // An entry that is no longer there - a material deleted between two enumerations is the case - leaves the
    // pulldown showing nothing rather than silently pointing at some other material.
    select.value = selected || '';
}

// The block's buttons and its one line of status, from the three things they depend on and nothing else.
function renderWorkdirPanel(note) {
    panel.renderWorkdir({
        // The BLOCK is drawn where a folder can be picked, and also where one has been dropped: a browser with no
        // picker still has a folder to name, a count to give and a Forget to offer once something is in hand.
        supported: workdir.supported() || !!granted,
        canPick: workdir.supported(),
        name: granted ? granted.name : null,
        reopen: !granted && stored ? stored.name : null,
        note: note,
    });
}

function workdirLine() {
    if (granted) {
        const count = entries.length;
        // Three states, and the parenthesis is the only thing that differs between them: kept for the next visit,
        // held for this one because the browser would not keep it, or held for this one because there is nothing
        // to keep - a dropped folder in a browser with no handle for one. The rest is true of all three.
        const kept = granted.session ? ' (this visit: a dropped folder cannot be stored, so drop it again next time)'
                   : remembered ? ' (remembered)'
                   : ' (this visit only: this browser would not store the handle)';
        const cut = granted.truncated ? ' The folder holds more files than the page has listed.' : '';
        return 'The page is reading the folder ' + granted.name + kept + '. It holds ' + count + ' material'
               + (count === 1 ? '' : 's') + ', and they are in the pulldown above. Reload re-reads the open material'
               + ' from disk and lists the folder again, so a material you encode while this page is open shows up'
               + ' there too.' + cut;
    }
    if (stored) {
        return 'The page still has the folder ' + stored.name + ' from an earlier visit. The browser may ask you'
               + ' again before the page may read it, and it can only ask that when you click something. Reopen is'
               + ' that click.';
    }
    if (!workdir.supported()) {
        return 'Drop a folder on the page. Its materials join the pulldown above, and you can then open one of them'
               + ' by picking or dropping its .json on its own, because the page reads the .dds files it names out'
               + ' of that folder. This browser has no button that asks for a folder, and it cannot store one'
               + ' either, so you drop it again on each visit. The page only reads your files. It never writes.';
    }
    return 'Press "Select working directory..." and pick one folder. Its materials join the pulldown above, and you'
           + ' can then open one of them by picking or dropping its .json on its own, because the page reads the'
           + ' .dds files it names out of that folder. Dropping a folder on the page does the same thing. The page'
           + ' only reads your files. It never writes.';
}

// The second pass over the list (workdir.js says why there are two): each descriptor read in list order, its line
// rewritten with the same summary a bundled entry carries, and the candidates that turn out not to be descriptors
// dropped. `token` is the enumeration this pass belongs to; a Reload that re-enumerates while one is still running
// bumps it, and the stale pass writes nothing.
async function describeEntries(token, candidates) {
    for (const entry of entries.slice()) {
        if (token !== listing) return;
        await workdir.describeEntry(entry);
        if (token !== listing) return;
        const option = optionFor(entry.name);
        if (option) option.textContent = entry.label;
    }
    for (const entry of candidates) {
        if (token !== listing) return;
        if (!(await workdir.describeEntry(entry))) continue;   // a *_source_material.json, say: not a material
        if (token !== listing) return;
        entries.push(entry);
        entries.sort((a, b) => a.name.localeCompare(b.name));
        rebuildAssetSelect();
    }
}

function optionFor(name) {
    const select = el('asset-select');
    for (const option of select.options) if (option.value === DIR_PREFIX + name) return option;
    return null;
}

// Enumerate the granted folder and rebuild the pulldown. Called when a directory is adopted and again on every
// Reload, which is what the plan means by "a newly encoded material appears without regranting anything".
async function refreshWorkdir() {
    if (!granted) return;
    const token = ++listing;
    let found;
    try {
        found = await workdir.listEntries(granted);
    } catch (e) {
        // The folder can go away between two visits to it - moved, renamed, deleted, or its permission taken back -
        // and the page says so rather than showing a list that no longer refers to anything.
        //
        // WHICH OF THE TWO IT WAS DECIDES WHAT IS OFFERED. A permission taken back leaves the handle perfectly
        // good, so it goes back to `stored` and Reopen asks for it again; anything else is a folder that is not
        // there, and the only way out is another one. Either way the NAME is captured before it is dropped, and
        // Forget stays reachable - the record is still in IndexedDB, and a panel that hid the one button that
        // clears it left the same error waiting on every future visit with no way to be rid of it.
        const name = granted.name;
        const wasSession = !!granted.session;
        const lapsed = e && e.name === 'NotAllowedError';
        stored = lapsed ? granted : stored;
        granted = null;
        entries = [];
        rebuildAssetSelect();
        renderWorkdirPanel(workdirLine());
        // THE REMEDY HAS TO BE A THING THIS BROWSER HAS. "Select working directory..." and Forget are drawn
        // only where there is a picker; in Firefox and Safari the way back is to drop the folder again, and
        // naming a button that is not on the page is worse than saying nothing.
        const again = wasSession || !workdir.supported()
            ? 'Drop the folder on the page again to carry on.'
            : '"Select working directory..." grants one again; Forget drops the page\'s record of it.';
        panel.setError(lapsed
            ? name + ' is no longer allowed to be read (' + (e.message || e) + '). Reopen asks the browser for it'
              + ' again; Forget drops the page\'s record of it.'
            : name + ' could not be read (' + (e.message || e) + '): the folder may have been moved, renamed or'
              + ' deleted. ' + again);
        return;
    }
    if (token !== listing) return;
    entries = found.listed;
    rebuildAssetSelect();
    renderWorkdirPanel(workdirLine());
    // NOT AWAITED. The list is on screen now; the second pass rewrites its lines as it reads them, and every
    // write it makes is guarded by the token, so a Reload that starts while it runs simply leaves it writing
    // nothing. Awaiting it made opening a folder - and every Reload after that - wait on a read of every descriptor
    // in the folder before the first material could be drawn, which is the opposite of what two passes are for.
    describing = describeEntries(token, found.candidates).then(() => {
        if (token === listing) renderWorkdirPanel(workdirLine());
    });
}

// A directory becomes THE working directory: picked, reopened, or dropped on the page. It is stored first - the
// handle is what survives the visit - then listed, and then its first material is opened, because an explicit
// "here is my folder" is a request to see what is in it.
async function adoptDirectory(handle) {
    granted = handle;
    stored = null;
    // Nothing to store for a dropped folder in a browser with no handle for one: an entry is not
    // structured-cloneable, so there is no record to keep and IndexedDB is not asked for one.
    remembered = handle.session ? false : await workdir.remember(handle);
    // WAIT FOR THE SECOND PASS BEFORE CALLING A FOLDER EMPTY. The listing is deliberately not awaited inside
    // refreshWorkdir - the list must appear at once and fill in - but a folder holding only older PREFIX.json
    // descriptors has NOTHING in the first pass, which reads names, and everything in the second, which reads
    // files. Deciding emptiness from the first pass alone said "no materials in X" and then quietly filled the
    // pulldown with the materials it had just denied. This is the one caller that needs the whole answer.
    await refreshWorkdir();
    await describing;
    if (!granted || !entries.length) {
        if (granted) {
            panel.setError('no materials in ' + granted.name + ': the page looks for a PREFIX_nntc.json (or an older'
                           + ' .json whose "format" is ntc-dds-1) beside the .dds files it names.');
        }
        return;
    }
    // The pulldown is NOT moved here. loadWorkdir moves it once the material is actually on screen, and a
    // material that refuses to open - a half-written descriptor, a .dds that is not there - would otherwise leave
    // the control naming it over a picture of the material that is still drawn.
    await openWorkdir(entries[0].name);
}

// The directory a descriptor was picked or dropped FROM, when that is the working directory or a folder inside it,
// with the files it was short of read out of it. Null when there is no folder, when the page was handed a File
// rather than a handle (a plain <input type="file"> gives no handle anywhere), or when the file is somewhere else
// entirely - resolve() answers null for a file outside the tree it is asked about, and there is nothing else to try.
async function completeFromWorkdir(chosen, set_, status) {
    if (!granted || !status.missing.length) return { dir: null, why: 'no-folder' };
    // A DROPPED FOLDER IS LOOKED UP BY NAME, and needs nothing else. It is a snapshot of one directory keyed by
    // filename, so "is this file one of yours" is a question a plain File can answer - it has a name. The handle
    // test below belongs to the GRANTED kind, where the folder is a live directory and the only way to ask where a
    // file sits inside it is to resolve a handle against it. Applying that test to both kinds is what stopped this
    // working in Firefox and Safari, which have no API that produces a handle at all: the page held the folder, the
    // file was in it, and the page gave up before it looked, saying only that the .dds files were missing.
    if (granted.session) {
        if (!granted.has(chosen.name)) return { dir: null, why: 'outside' };
        await workdir.fillFrom(granted, status.missing, set_);
        return { dir: granted, why: 'found' };
    }
    const handle = set_.sources.get(chosen.name);
    // A plain <input type="file"> gives a File and no handle, anywhere, and against a GRANTED folder a File cannot
    // be placed: it is bytes and a name, with nothing that says where it came from.
    if (!handle || handle.kind !== 'file') return { dir: null, why: 'no-handle' };
    const dir = await workdir.containingDirectory(granted, handle);
    if (!dir) return { dir: null, why: 'outside' };
    await workdir.fillFrom(dir, status.missing, set_);
    return { dir: dir, why: 'found' };
}

// The sentence added to a shortfall when a working directory could have answered it and did not. It says the one
// fact a user cannot guess - that the page genuinely cannot see the folder the file came from - so that "then why
// did it not just look next to it" has an answer, and it says which of the three reasons applied, because the
// remedy differs: grant the folder, grant THAT folder, or use a route that hands the page handles.
function outsideHint(placed, status) {
    // `granted` rather than `supported()`: a browser with no picker can still have been handed a folder, and
    // that is exactly the case where "why did it not look in my folder" most needs an answer.
    if (!status.missing.length || (!workdir.supported() && !granted) || (placed && placed.dir)) return '';
    const why = placed ? placed.why : 'no-folder';
    if (why === 'outside') {
        return ' That .json is not in ' + granted.name + ', the folder the page is reading. No browser tells a page'
             + ' which folder a file came from, so the page cannot go and look beside it. Give the page the folder'
             + ' that .json lives in instead, or hand the page its .dds files as well.';
    }
    if (why === 'no-handle') {
        return ' This way in hands the page the file and nothing else, so the page cannot tell whether the file sits'
             + ' in ' + granted.name + ', the folder it is reading. Open the material from the pulldown instead, or'
             + ' from "Open files...", or hand the page its .dds files as well.';
    }
    return ' No browser tells a page which folder a file came from, so a .json on its own cannot find the .dds files'
         + ' it names. Give the page that folder with "Select working directory..." above, or hand the page those'
         + ' .dds files as well.';
}

async function loadWorkdir(dir, name) {
    const got = await workdir.readMaterial(dir, name);
    source = { kind: 'workdir', dir: dir, name: name };
    // As for the pulldown's bundled entries: an explicit pick ends whatever was being assembled by hand.
    pending = new FileSet();
    const base = granted || dir;
    const where = dir === base ? base.name : 'a folder inside ' + base.name;
    await showAsset(got.name, got.root, got.set, 'from ' + where + ', ' + name);
    // THE PULLDOWN FOLLOWS WHAT IS ON SCREEN, and only once it IS on screen. A material can reach this function
    // without having been picked from the list - a bare .json dropped or opened out of the granted folder is the
    // case the owner asked for - and a pulldown naming the material it was last clicked on, over a picture of
    // another one, is the kind of small lie that costs an afternoon. showAsset throws on a refusal, so a material
    // that did not open does not move the control either.
    if (dir === granted && entries.some((entry) => entry.name === name)) {
        selected = DIR_PREFIX + name;
        el('asset-select').value = selected;
    }
    panel.renderFileRows(got.name, resolve(got.root, got.set), got.set, false);
    return true;
}

// `accumulate` is the whole difference between the two callers. An offer FROM THE USER joins the pending set, because
// what a person hands the page is an addition to what they have already handed it. RELOAD re-reads the loaded
// material's own file handles and merges nothing, so a stray file offered since that load cannot walk into a reload
// of something else.
//
// NOTHING ON SCREEN MOVES ON A SHORTFALL. An incomplete set writes one message and the list of what is in hand, and
// returns before showAsset - so the picture, the asset readout and the camera are exactly as they were. Someone
// trying files out never loses the thing they were looking at.
async function loadFiles(files, accumulate) {
    const offered = await readFileList(files);
    const set_ = accumulate ? mergeSets(pending, offered) : offered;
    const found = findDescriptors(set_);
    if (!found.length) {
        if (accumulate) pending = set_;
        panel.renderFileRows(null, null, set_, true);
        panel.setError('None of the files offered so far is an NNTC descriptor: the page looks for a .json whose'
                       + ' "format" is nntc-dds-1. They are kept - add the PREFIX_nntc.json of the material,'
                       + ' by itself if you like, and it opens.');
        return;
    }
    // WHICH descriptor, when the set holds more than one: the first one the set SATISFIES, and failing that the
    // first of them, so that the message names something. findDescriptors' own order is untouched and still decides
    // between two that are both complete - the first still wins.
    //
    // Taking found[0] and stopping was right while a set was one selection and gone; with a set that persists it is
    // a trap. A .json offered by mistake, or left over from a material that was never finished, sorts where it sorts
    // and would sit in front of a complete material for ever - and the only way out would be the pulldown, which is
    // not a thing anyone would guess. The rule the page states is that it opens as soon as what it holds is enough
    // for A descriptor; this is that rule.
    let chosen = found[0];
    let status = resolve(chosen.root, set_);
    if (status.missing.length || status.unresolvable.length) {
        for (const other of found) {
            const otherStatus = resolve(other.root, set_);
            if (otherStatus.missing.length || otherStatus.unresolvable.length) continue;
            chosen = other;
            status = otherStatus;
            break;
        }
    }
    // THE OWNER'S PREFERRED INTERACTION, and the whole reason a working directory is worth a grant: pick or drop a
    // bare .json out of the granted folder and the material opens, because resolve() says where in that folder the
    // file sits and getFileHandle fetches its siblings from the directory it sits in. Without a folder this cannot
    // work in any browser - a file handle has no parent accessor anywhere - so what happens instead is the rows and
    // the one message, exactly as before.
    let placed = null;
    if (status.missing.length && !status.unresolvable.length) {
        placed = await completeFromWorkdir(chosen, set_, status);
        if (placed.dir) status = resolve(chosen.root, set_);
    }
    const fromDir = placed && placed.dir ? placed.dir : null;
    if (status.missing.length || status.unresolvable.length) {
        if (accumulate) pending = set_;
        panel.renderFileRows(chosen.name, status, set_, true);
        panel.setError(missingMessage(status, chosen.name) + outsideHint(placed, status));
        return;
    }
    // The handles of the files this material USES, not of the last selection: once a set has been assembled over
    // several goes the two are different things, and Reload wants the first.
    const handles = [chosen.name].concat(status.wanted)
        .map((name) => set_.sources.get(name))
        .filter((handle) => handle && (typeof handle.arrayBuffer === 'function'
                                       || typeof handle.getFile === 'function'));
    // A material whose files were found IN THE WORKING DIRECTORY is a working-directory load and is remembered as
    // one, so Reload re-reads it through the folder's own handles and re-enumerates the folder - which is strictly
    // better than re-reading the one file the user happened to drop.
    source = fromDir ? { kind: 'workdir', dir: fromDir, name: chosen.name }
                     : { kind: 'files', files: handles, descriptorName: chosen.name };
    const extra = found.length > 1
        ? ' (' + found.length + ' descriptors offered; ' + chosen.name + ' was opened)' : '';
    // Held over showAsset and emptied only after it RETURNS. showAsset throws on a refusal - a .dds whose header
    // disagrees with the descriptor is the case - and a set that has just been refused is the one a user most
    // wants kept: offering the right file of that name replaces it and finishes the job. Emptying before the
    // call, or in a finally, would throw the other files away with the bad one.
    if (accumulate) pending = set_;
    const label = fromDir ? 'from ' + granted.name + ', ' + chosen.name + " - the .dds files it names were taken from"
                            + ' the working directory' + extra
                          : 'from files' + extra;
    await showAsset(chosen.name, chosen.root, set_, label);
    // AND THE PULLDOWN FOLLOWS, on the same rule loadWorkdir uses and for the same reason: this is the owner's
    // preferred interaction - a bare .json out of the working directory - and it ended with the right material
    // drawn under the previous one's name in the control. Only once it is on screen, and only when it is a
    // material the list actually holds.
    if (fromDir && fromDir === granted && entries.some((entry) => entry.name === chosen.name)) {
        selected = DIR_PREFIX + chosen.name;
        el('asset-select').value = selected;
    }
    // Only now, with the material actually on screen: a set whose files are all present but one of which the
    // descriptor disagrees with never gets here, and its refusal leaves the panel showing the material that IS
    // drawn rather than a row list for one that is not.
    panel.renderFileRows(chosen.name, status, set_, false);
    pending = new FileSet();
    return true;
}

// A NEW material - picked from the pulldown, opened from the file dialog, dropped on the page - versus RELOAD, which
// is the same material re-encoded. The owner drew this line and it is not arbitrary:
//
//   * a new material is a new thing to look at, so it starts where a fresh page starts: default camera, default
//     filter, no raw view, no renormalise, texture 0. Whatever the last material's keys left on does not carry over
//     and become a puzzle about the new one;
//   * Reload is the encode-look-encode loop. Its whole worth is seeing what changed FROM THE SAME ANGLE WITH THE SAME
//     SETTINGS, so it keeps every one of them.
//
// They are therefore two paths and not one with a flag: reload() below never calls resetView(), and these two always
// do. The reset comes AFTER the load, so a selection that is refused - a missing .dds, a descriptor that will not
// parse - leaves the material already on screen exactly where it was, camera included.
//
// The edge the owner asked about - picking the SAME entry from the pulldown again - is a new load by this rule,
// because an explicit pick says "start again". In practice a <select> fires no `change` event when the value has not
// changed, so re-picking the open material does nothing at all and Reload is how it is re-read; the rule and the
// browser agree, and neither of them quietly keeps a camera the user asked to be rid of.
async function openBundled(url) {
    if (await loadBundled(url)) resetView();
}

async function openFiles(files) {
    if (await loadFiles(files, true)) resetView();
}

async function openWorkdir(name) {
    if (!granted) return;
    if (await loadWorkdir(granted, name)) resetView();
}

// Reload (5.7). Bundled: re-fetch, which the copied server's no-cache header makes return new bytes. Files: re-read
// the File objects, which the File API says throws once the encoder has rewritten them - so the page says exactly
// that, and what to do about it. The camera is kept, which is the difference between a usable loop and an annoying
// one; the textures and the uniform block are rebuilt from the new descriptor either way.
//
// IT SAYS WHAT HAPPENED, because it has a caller that has to know. 'loaded' is a material on screen; 'failed' is a
// refusal already written into the status line, with the previous material left where it was; 'nothing' is no material
// to reload at all. The unpack checkbox below binds itself to the picture only on 'loaded', and the alternative -
// a reload that swallows its own failure - is what let the checkbox, the URL and the panel end up describing a
// picture that was never drawn.
async function reload() {
    if (!source) return 'nothing';
    try {
        // A working directory is RE-ENUMERATED before the material is re-read, and deliberately in that order: the
        // material may be the one the encoder has just deleted or renamed, and the list is how a user finds what it
        // became. A handle is a reference to a path, so the re-read gets the file as it is now - the one thing a
        // File snapshot cannot do, and the reason this route is the one worth having in an encode-look loop.
        // WHATEVER FOLDER IS IN HAND IS RE-LISTED, not only when the open material came out of it: the status
        // line, the README and the Help all promise that a material written since joins the pulldown on Reload, and
        // gating it on what happens to be on screen made that promise false in the obvious case - a bundled
        // material open, a new one encoded, R pressed, nothing at all happening.
        if (granted) {
            await refreshWorkdir();
            // refreshWorkdir CLEARS `granted` when the folder could not be read, and has already written the one
            // sentence that says what to do about it - Reopen for a permission taken back, another folder for one
            // that is gone. Carrying on to re-read the material from that same dead folder throws a second time,
            // and the outer catch would replace that sentence with the bare exception. Stop while the useful
            // message is still on screen.
            if (!granted) return 'failed';
        }
        const ok = source.kind === 'bundled' ? await loadBundled(source.url)
                 : source.kind === 'workdir' ? await loadWorkdir(source.dir, source.name)
                                             : await loadFiles(source.files, false);
        // loadFiles returns nothing when the set falls short of its descriptor: it has written the rows and the one
        // message, and nothing on screen has moved, which is a failure to reload like any other.
        return ok === true ? 'loaded' : 'failed';
    } catch (e) {
        panel.setError(reloadMessage(e));
        return 'failed';
    }
}

// Reload is the button pressed most often and the one whose result is least visible: a material re-read from
// disk usually looks exactly like the one it replaced. So it says so. 'failed' has already written its own red
// line and must not be painted over with a cheerful one; 'nothing' is not a failure, just nothing to do.
async function reloadAndSay() {
    loadVerb = 'Reloaded';
    let outcome;
    try { outcome = await reload(); } finally { loadVerb = 'Loaded'; }
    // 'nothing' HAS TWO CASES AND THEY ARE NOT THE SAME SENTENCE. An empty page has nothing to reload and says so.
    // A page with a picture on it but no source is the one Forget leaves behind: the material stays on screen, as
    // Forget promised, but the folder it was read from is no longer held, so there is nothing to re-read it from.
    // Saying "nothing is loaded" there contradicts both the picture and the message before it.
    if (outcome === 'nothing') {
        panel.setStatus(asset
            ? 'There is nothing to re-read this material from: the folder it came from has been let go. Open it'
              + ' again from a folder, or hand the page its files.'
            : 'Nothing is loaded to reload.');
    }
    return outcome;
}

// WHAT TO CALL THE THING THAT WAS LOADED: the name it is actually known by, and nothing else. A bundled material
// is its path, because that is what the pulldown shows and what a url would carry; one out of a folder is its file
// name and the folder it came from, because the same file name can be in two folders; one assembled from files the
// user handed over is just its file name, since there is nowhere else to point at.
function loadedName(fileName) {
    if (source && source.kind === 'bundled' && source.url) return source.url;
    const where = source && source.kind === 'workdir'
        ? (source.dir ? source.dir.name : granted ? granted.name : null) : null;
    return fileName + (where ? ' from ' + where : '');
}

// ---------------------------------------------------------------------------
// The frame (the plan's 8.1 and 8.2)
// ---------------------------------------------------------------------------

// WHAT SIZE TO CAPTURE AT. Empty means the canvas as it stands, which is what a person pressing the button expects;
// WIDTHxHEIGHT is the `--size` of the native viewers, for a comparison that must not depend on how wide the panel
// happens to be. A size that is not two numbers is a refusal and not a silent fallback: a frame captured at the
// wrong size looks fine and compares as garbage, which is the kind of failure this page exists to avoid.
function captureSize() {
    const text = (el('shot-size').value || '').trim();
    if (!text) return { width: renderer.width, height: renderer.height };
    const m = /^(\d+)\s*[xX*]\s*(\d+)$/.exec(text);
    if (!m) throw new Error('"' + text + '" is not a frame size. Write it as WIDTHxHEIGHT, for example 2560x1421,'
                            + ' or leave it empty for the canvas as it is.');
    return { width: parseInt(m[1], 10), height: parseInt(m[2], 10) };
}

// The name the material is known by, for the file and for the status line.
function capturedFrom() {
    if (source && source.kind === 'bundled' && source.url) return source.url;
    if (source && source.name) return source.name;
    if (source && source.descriptorName) return source.descriptorName;
    return asset && asset.name ? asset.name : 'frame';
}

async function saveFrame(kind) {
    if (!asset) { panel.setError('There is no material on screen to capture.'); return; }
    let size;
    try { size = captureSize(); } catch (e) { panel.setError(e.message); return; }
    try {
        const shot = await renderer.capture(state, size.width, size.height);
        const name = shotName(capturedFrom(), shot.width, shot.height, kind);
        const blob = kind === 'png' ? await pngBlob(shot)
                                    : new Blob([bmpBytes(shot)], { type: 'image/bmp' });
        offerDownload(name, blob);
        panel.setStatus('Saved ' + name + ' (' + shot.width + ' by ' + shot.height + ', no overlay).');
        dirty = true;   // the capture borrowed the uniform block; the canvas draws itself again
    } catch (e) {
        panel.setError('The frame could not be captured: ' + (e.message || e));
    }
}

// COPY LINK carries what a link CAN carry: a bundled material by name, the camera, and every mode. A material out
// of your own folder is not reachable from a url at all - the page cannot be handed a folder by one - so the link
// names no asset in that case and says so, rather than writing a path that would fail on the other end.
function viewLink() {
    const url = new URL(location.href);
    const p = url.searchParams;
    for (const key of ['asset', 'z', 'x', 'y', 'yaw', 'pitch', 'tex', 'cube', 'geom', 'row', 'orbit', 'nomips',
                       'nobias', 'noaniso', 'renorm', 'raw0', 'raw1', 'chan', 'filter', 'overlay', 'unpack',
                       'size', 'shot', 'post']) {
        p.delete(key);
    }
    if (source && source.kind === 'bundled' && source.url) p.set('asset', source.url);
    const put = (name, value, dflt) => { if (value !== dflt) p.set(name, String(value)); };
    put('z', round3(state.z), round3(DEFAULTS.z));
    put('x', round3(state.x), round3(DEFAULTS.x));
    put('y', round3(state.y), round3(DEFAULTS.y));
    put('yaw', round3(state.yaw), round3(DEFAULTS.yaw));
    put('pitch', round3(state.pitch), round3(DEFAULTS.pitch));
    put('tex', state.texShown, 0);
    if (state.geom !== DEFAULTS.geom) p.set('geom', state.geom);
    if (state.allTextures) p.set('row', '1');
    // The ORBIT is deliberately not carried. A link is a view, and a view that starts turning the moment it opens
    // is not the one that was copied - it is a different one a second later.
    
    if (!state.mipsOn) p.set('nomips', '1');
    if (!state.lodBias) p.set('nobias', '1');
    if (!state.aniso) p.set('noaniso', '1');
    if (state.renorm) p.set('renorm', '1');
    if (state.raw0) p.set('raw0', '1');
    if (state.raw1) p.set('raw1', '1');
    if (state.channel !== 'rgb') p.set('chan', state.channel);
    // THE FILTER AND THE UNPACK were both missing, and the panel, the Help and the README all said the link
    // carried every mode. The filter is a view mode like any other. The unpack is not a view mode at all - it is
    // which path the planes are bound through - but the page already writes ?unpack= into its own address bar when
    // the box is ticked, and a link copied from that address bar that quietly dropped it would reopen on the other
    // path: two pictures that this commit's own measurement puts 2 counts and 68 dB apart.
    if (state.filterMode !== DEFAULTS.filterMode) p.set('filter', FILTERS[state.filterMode] || '');
    if (params.has('unpack')) p.set('unpack', params.get('unpack'));
    if (!state.overlay) p.set('overlay', '0');
    return url.toString();
}

function round3(v) { return Math.round(v * 1000) / 1000; }

// ---------------------------------------------------------------------------
// ?shot=1 - one frame, taken and handed over, with nothing for a person to click
//
// This is `nntc_view --shot` for a browser, and it exists for the same reason: a frame whose bytes are a function
// of the url and the asset and of nothing else can be compared against the three native viewers by a script. The
// page has no way to write a file and no way to close the browser, so the two halves a native --shot does itself
// are split - the page POSTs the bytes to the server that served it, and whatever launched the browser kills it.
//
// `&post=/frame` is the POST; without it the frame is simply downloaded, which is what a person wants when they
// are driving it by hand. document.title becomes `done` either way, and `shot failed: ...` if it did not, so a
// driver watching the title learns the outcome even if the POST never arrives.
//
// THE OVERLAY IS OFF unless ?overlay=1 says otherwise, exactly as --nooverlay is the default of a native --shot:
// the strip names the file and the format, which is the one thing two renders of the same asset differ in.
async function runShot() {
    // A NAMED MATERIAL THAT WAS NOT FOUND IS A REFUSAL, not a frame of whatever opened instead. ?asset=NAME with
    // no match writes its own error line and then the page opens the first bundled material, which is right for
    // someone reading the page and wrong for a script: it would have posted a frame of m1, named after m1, under
    // a title of `done`, and the comparison would have been between two materials.
    if (params.get('asset') && !(source && source.kind === 'bundled'
                                 && source.url === shotWanted)) {
        const why = '?shot: ?asset=' + params.get('asset') + ' is not the material that opened, so no frame was'
                  + ' taken. A shot names a bundled material.';
        panel.setError(why);
        document.title = 'shot failed: ' + why;
        return;
    }
    const size = (params.get('size') || '').trim();
    const m = /^(\d+)\s*[xX*]\s*(\d+)$/.exec(size);
    const width = m ? parseInt(m[1], 10) : renderer.width;
    const height = m ? parseInt(m[2], 10) : renderer.height;
    try {
        if (size && !m) throw new Error('?size=' + size + ' is not a frame size; write it as WIDTHxHEIGHT');
        if (!asset) throw new Error('no material is loaded; ?shot needs an ?asset= this page can fetch');
        const shot = await renderer.capture(state, width, height);
        const bytes = bmpBytes(shot);
        const post = params.get('post');
        if (post) {
            // The SAME server that served the page, so there is no cross-origin question to answer and the driver
            // does not have to be told where it lives.
            const response = await fetch(post, {
                method: 'POST',
                headers: { 'Content-Type': 'image/bmp', 'X-Shot-Name': shotName(capturedFrom(), shot.width, shot.height, 'bmp') },
                body: bytes,
            });
            if (!response.ok) throw new Error('the server answered ' + response.status + ' to the frame');
        } else {
            offerDownload(shotName(capturedFrom(), shot.width, shot.height, 'bmp'),
                          new Blob([bytes], { type: 'image/bmp' }));
        }
        panel.setStatus('?shot: ' + shot.width + ' by ' + shot.height + ', ' + bytes.length + ' bytes'
                        + (post ? ' posted to ' + post + '.' : ' downloaded.'));
        dirty = true;   // as Save frame does: the canvas redraws itself at its own aspect, not the capture's
        document.title = 'done';
    } catch (e) {
        // The title is the driver's only channel when the POST is the thing that failed, so the reason goes in it.
        panel.setError('?shot failed: ' + (e.message || e));
        document.title = 'shot failed: ' + (e.message || e);
    }
}

function reloadMessage(e) {
    const text = e && e.message ? e.message : String(e);
    if (e && (e.name === 'NotReadableError' || /could not be read/.test(text))) {
        // A material held BY HANDLE re-reads whatever is at that path now, so a failure here is not "it changed":
        // it is gone, or the permission to it lapsed, or it is being written this instant. Only a material held as
        // plain Files - a drop, or an <input> - fails because the bytes moved under the snapshot.
        // A dropped folder holds Files, not handles, so "the permission lapsed" is not one of its failures.
        const byHandle = source && (source.kind === 'bundled'
            || (source.kind === 'workdir' && !(source.dir && source.dir.session)));
        return text + (byHandle
            ? ' It may have been deleted, or the browser\'s permission to the folder may have lapsed, or it is'
              + ' being written right now - Reload tries again.'
            : ' These files were rewritten or removed since they were selected; select or drop them again.');
    }
    // A bare /JSON/ used to be the last alternative here, and it matched the DESCRIPTOR MISMATCH refusal too -
    // "the JSON says m1_lat0.dds is DXGI 83 ..." - so a wrong file was answered with advice to wait for the
    // encoder. These are the shapes a HALF-WRITTEN file actually throws, including what JSON.parse says of a
    // truncated document, and nothing else.
    if (/truncated|bytes of pixel data expected|is not valid JSON|end of JSON input|in JSON at position/
            .test(text)) {
        return text + ' - the encoder may still be writing it; reload again when it has finished.';
    }
    return text;
}

// ---------------------------------------------------------------------------
// The frame loop
// ---------------------------------------------------------------------------
const pressed = new Set();
let lastTime = 0, lastRendered = 0, fps = null, frameMs = 0;

function movement(dt) {
    if (!pressed.size) return false;
    if (pressed.has('shift')) dt *= 1.0 / 3.0;
    let moved = false;
    const step = (key, field, amount) => {
        if (pressed.has(key)) { state[field] += amount; moved = true; }
    };
    step('w', 'z', Z_SPEED * dt);
    step('s', 'z', -Z_SPEED * dt);
    step('arrowleft', 'x', XY_SPEED * dt);
    step('arrowright', 'x', -XY_SPEED * dt);
    step('arrowup', 'y', XY_SPEED * dt);
    step('arrowdown', 'y', -XY_SPEED * dt);
    step('a', 'yaw', ROT_SPEED * dt);
    step('d', 'yaw', -ROT_SPEED * dt);
    step('q', 'pitch', ROT_SPEED * dt);
    step('e', 'pitch', -ROT_SPEED * dt);
    if (state.z < Z_MAX) state.z = Z_MAX;
    if (state.z > Z_MIN) state.z = Z_MIN;
    return moved;
}

function frame(now) {
    requestAnimationFrame(frame);
    if (halted) return;      // ?strict=1 stopped the page on a WebGPU refusal; the last good frame stays on screen
    const dt = lastTime ? Math.min(0.1, (now - lastTime) / 1000) : 0;
    lastTime = now;
    // The camera is the one part of the state the frame loop writes directly, so it is the one part whose panel
    // controls have to be caught up here rather than in set(). The five numbers follow a held key exactly as the
    // drawn strip's second line does.
    // THE ORBIT, and it is the camera rather than the model, so it composes with everything else: the geometry C
    // cycles to, the all-textures row, the filter and the mip controls. Thirty degrees a SECOND, scaled by the
    // frame's own delta time, so the speed is the same on a 144 Hz monitor as on a throttled background tab - and
    // dt is already clamped to 0.1 s above, so a tab that stops compositing does not leap when it comes back.
    //
    // A held yaw key still works while it turns, and adds to it: the input wins for that moment and the orbit
    // carries on from wherever it was left, which is less surprising than a key that silently stops it.
    if (state.orbit && dt > 0) {
        // A PROPER MODULO. JavaScript's % keeps the dividend's sign, and nothing else here wraps yaw - a held
        // D key or a drag can leave it at -400 - so the plain remainder would jump the readout to -39.5 on the
        // first orbit frame and then count up towards zero. The picture never moved; the number did.
        state.yaw = ((state.yaw + ORBIT_DEGREES_PER_SECOND * dt) % 360 + 360) % 360;
        dirty = true;
        panel.renderCamera(state);
    }
    if (movement(dt)) { dirty = true; panel.renderCamera(state); }
    let goingIdle = false;
    if (!dirty) {
        // Nothing is being redrawn, so the frame rate is not a number worth printing: the strip says so, once, and
        // that one extra frame is what puts the word there.
        if (fps !== null && now - lastRendered > 400) { fps = null; frameMs = 0; goingIdle = true; dirty = true; }
        else return;
    }
    if (!goingIdle && lastRendered) {
        const ms = now - lastRendered;
        // A gap of more than a second is not a frame time: the tab was in the background, where the browser stops
        // calling back, and averaging that in would print a number that means nothing. The strip starts again.
        if (ms > 1000) { frameMs = 0; fps = null; }
        else {
            frameMs = frameMs ? frameMs * 0.8 + ms * 0.2 : ms;
            fps = frameMs > 0 ? 1000 / frameMs : null;
        }
    }
    lastRendered = now;
    if (overlay && state.overlay) overlay.setLines(fpsLine(fps, frameMs), cameraLine(state));
    renderer.render(state, state.overlay ? overlay : null);
    dirty = false;
}

// ---------------------------------------------------------------------------
// Input: the keys go to the CANVAS, not to the window, so typing in the panel never yaws the model.
// ---------------------------------------------------------------------------
function bindInput(canvas) {
    const hint = el('hint');
    const focused = () => document.activeElement === canvas;
    canvas.addEventListener('focus', () => hint.classList.add('hidden'));
    canvas.addEventListener('blur', () => { hint.classList.remove('hidden'); pressed.clear(); });
    canvas.addEventListener('pointerdown', (e) => {
        canvas.focus();
        canvas.setPointerCapture(e.pointerId);
        canvas.dataset.dragging = '1';
        canvas.dataset.lastX = String(e.clientX);
        canvas.dataset.lastY = String(e.clientY);
    });
    canvas.addEventListener('pointerup', (e) => {
        canvas.dataset.dragging = '';
        if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
        if (canvas.dataset.dragging !== '1') return;
        const dx = e.clientX - Number(canvas.dataset.lastX), dy = e.clientY - Number(canvas.dataset.lastY);
        canvas.dataset.lastX = String(e.clientX);
        canvas.dataset.lastY = String(e.clientY);
        state.yaw -= dx * 0.25;
        state.pitch -= dy * 0.25;
        dirty = true;
        panel.renderCamera(state);
    });
    // { passive: false } so the page does not scroll under the wheel.
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        state.z += (e.deltaY < 0 ? 1 : -1) * 0.15;
        if (state.z < Z_MAX) state.z = Z_MAX;
        if (state.z > Z_MIN) state.z = Z_MIN;
        dirty = true;
        panel.renderCamera(state);
    }, { passive: false });

    const HELD = new Set(['w', 's', 'a', 'd', 'q', 'e', 'shift',
                          'arrowleft', 'arrowright', 'arrowup', 'arrowdown']);

    // EVERY key the native viewers bind, with the native meaning (../viewer/README.md's table and the F1 page at
    // ../viewer/main.cpp:572-600; the Direct3D 12 and Vulkan viewers have the same set). Each one goes through set()
    // or setAll(), so the control in the panel moves with it and the two can never disagree - that is the whole
    // point of 6.3 and the reason none of these writes a field by hand.
    //
    // What is NOT bound, and why, so the next reader does not think it was forgotten:
    //   * `4`, level 0 from the load-time BC4/BC5 pack. That pack exists only for an UNCOMPRESSED level 0, and it is
    //     not implemented here at all (the plan's 10.2 defers it); every bundled asset is block-compressed, where the
    //     native key does nothing either. The page's BC control goes the other way - the unpack of a block-compressed
    //     level 0 for a GPU without the feature - and it is the Decode block's checkbox, which is not what `4` means.
    //     A key bound to something adjacent would be worse than an unbound one;
    //   * `5`-`8`, the shader's spare debug constants. Nothing in nntc_view.wgsl reads them, exactly as nothing in
    //     the shipped nntc_view.hlsl does;
    //   * `K`, the Direct3D 12 decode-path switch, which exists only in a build with that option and has no web
    //     counterpart: there is one WGSL decode path here;
    //   * `Esc`, which quits a native viewer. A page cannot close itself and must not try, so it closes the help
    //     dialog and nothing else - which <dialog> does natively.
    const ACTIONS = {
        c: () => set('geom', GEOMETRIES[(GEOMETRIES.indexOf(state.geom) + 1) % GEOMETRIES.length]),
        g: () => set('allTextures', !state.allTextures),
        o: () => set('orbit', !state.orbit),
        p: () => set('filterMode', 0),
        b: () => set('filterMode', 1),
        t: () => set('filterMode', 2),
        x: () => set('aniso', !state.aniso),
        m: () => set('mipsOn', !state.mipsOn),
        l: () => set('lodBias', !state.lodBias),
        v: () => set('renorm', !state.renorm),
        // The native viewer turns the other raw view off with the same keystroke (main.cpp:817-818): the two are one
        // three-way choice, which is why the panel draws them as one radio set.
        1: () => setAll({ raw0: !state.raw0, raw1: false }),
        2: () => setAll({ raw1: !state.raw1, raw0: false }),
        // With the row on there is no single shown texture, so texShown is the OFFSET and N slides the whole
        // material along the row: object i shows texture (i + texShown) % count. Same key, same wrap, and the
        // panel's texture pulldown says which it is either way.
        n: () => { if (asset) set('texShown', (state.texShown + 1) % asset.texturesOut); },
    };

    canvas.addEventListener('keydown', (e) => {
        if (!focused()) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;   // Ctrl+R stays the browser's
        const key = e.key.toLowerCase();
        if (HELD.has(key) || key === ' ') e.preventDefault();   // arrows and Space scroll the page otherwise
        if (e.repeat) return;
        if (HELD.has(key)) { pressed.add(key); return; }
        if (key === ' ') { resetView(); return; }              // reset: the camera AND every mode (see DEFAULTS)
        if (key === 'r') {
            e.preventDefault();
            if (e.shiftKey) reloadShader(); else reloadAndSay();
            return;
        }
        // Shift is not consulted here, because the native viewers do not consult it either: they switch on the
        // virtual key, so Shift+C is C and a capital toggles the cube just as the lower case does. `R` above is the
        // one key where Shift means something, and it is the one key whose meaning this viewer changed.
        if (ACTIONS[key]) ACTIONS[key]();
    });
    canvas.addEventListener('keyup', (e) => { pressed.delete(e.key.toLowerCase()); });
    window.addEventListener('blur', () => pressed.clear());
}

async function reloadShader() {
    try {
        await renderer.loadShader('nntc_view.wgsl');
        panel.setStatus('Reloaded the shader.');
        dirty = true;
    } catch (e) {
        panel.setError(e.message);
    }
}

function bindFiles() {
    const input = el('file-input');
    // showOpenFilePicker where it exists, for the same two reasons the directory picker uses it: `id` reopens the
    // dialog where it last was, which in a loop of "open the material's folder, Ctrl+A" is most of the work; and it
    // hands back HANDLES rather than snapshots, so the files it opened can be re-read after a re-encode. The hidden
    // <input multiple> is still there and is still the whole of this control in a browser without the picker.
    el('open-files').addEventListener('click', async () => {
        if (!workdir.filePickerSupported()) { input.click(); return; }
        let handles = null;
        try {
            // THE SAME TWO EXTENSIONS THE <input> HAS ALWAYS FILTERED ON. Moving this control to
            // showOpenFilePicker dropped the filter, because the picker takes its types as an argument and the
            // input takes them as an attribute - so the dialog opened on every file in the folder and a
            // Ctrl+A picked up whatever else was lying in it. `excludeAcceptAllOption` is left false on
            // purpose: the filter is the default, and a user who wants to point at something oddly named can
            // still switch the dialog to all files.
            handles = await window.showOpenFilePicker({
                id: 'nntc-files',
                multiple: true,
                types: [{
                    description: 'NNTC material files (.json, .dds)',
                    accept: { 'application/json': ['.json'], 'application/octet-stream': ['.dds'] },
                }],
            });
        } catch (e) {
            if (e && e.name === 'AbortError') return;    // the dialog was closed: not a failure, and not a message
            input.click();                               // anything else (a policy, an embedder): the input still works
            return;
        }
        try { await openFiles(handles); } catch (err) { panel.setError(reloadMessage(err)); }
    });
    input.addEventListener('change', async () => {
        try { await openFiles(input.files); } catch (e) { panel.setError(reloadMessage(e)); }
    });
    // The whole page takes a dropped group, not just the zone - the zone is where it SAYS so, and where the highlight
    // is. One listener, on the document, so a drop on the zone is not handled twice.
    const zone = el('drop-zone');
    // A FOLDER IS NAMED ONLY WHERE ONE CAN BE TAKEN. Both drop shapes end in the same working directory, so
    // either one earns the word; a browser with neither is not promised something it would then refuse. In practice
    // every current browser has at least webkitGetAsEntry, so this nearly always says folder - which is the point
    // of asking rather than assuming, in both directions.
    if (workdir.dropHandlesSupported() || workdir.dropEntriesSupported()) {
        zone.textContent = 'or drag the files - or the folder - onto this page';
    }
    document.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('over'); });
    document.addEventListener('dragleave', (e) => { e.preventDefault(); zone.classList.remove('over'); });
    document.addEventListener('drop', async (e) => {
        e.preventDefault();
        zone.classList.remove('over');
        const transfer = e.dataTransfer;
        if (!transfer) return;
        // EVERYTHING THAT TOUCHES dataTransfer.items HAPPENS HERE, BEFORE THE FIRST await: the item list is emptied
        // the moment this handler yields, so the handle promises are started now and waited on afterwards. This is
        // the one piece of the page where the order of two lines is the whole behaviour.
        let handlePromises = null;
        let folderEntry = null;
        if (transfer.items && transfer.items.length) {
            handlePromises = [];
            for (const item of transfer.items) {
                if (item.kind !== 'file') continue;
                if (workdir.dropHandlesSupported()) handlePromises.push(item.getAsFileSystemHandle());
                const entry = typeof item.webkitGetAsEntry === 'function' ? item.webkitGetAsEntry() : null;
                if (entry && entry.isDirectory && !folderEntry) folderEntry = entry;
            }
        }
        const files = transfer.files ? Array.from(transfer.files) : [];
        try {
            // A promise that REJECTS is one item the browser would not describe, not a failed drop: the rest of the
            // group is still worth having, so each one answers for itself and a null is simply not a handle.
            const handles = handlePromises && handlePromises.length
                ? (await Promise.all(handlePromises.map((p) => p.then((h) => h, () => null)))).filter((h) => h)
                : [];
            // A DROPPED FOLDER IS A WORKING DIRECTORY, where the browser hands over a handle for it: it is the same
            // kind of object the picker returns, so it goes down the same path and is remembered the same way.
            const directory = handles.find((h) => h.kind === 'directory');
            if (directory) { await adoptDirectory(directory); return; }
            if (handles.length) { await openFiles(handles); return; }
            // NO HANDLE FOR THE FOLDER, BUT AN ENTRY FOR IT: Firefox and Safari, which have had
            // webkitGetAsEntry since Firefox 50. It becomes the working directory like any other - same pulldown,
            // same Reload, same status line - and differs in one way the line says out loud: it is not storable, so
            // it is not offered back on the next visit. A drop is a drop; what the browser allows is what happens.
            if (folderEntry && workdir.dropEntriesSupported()) {
                await adoptDirectory(await workdir.sessionDirectory(folderEntry));
                return;
            }
            if (!files.length) return;
            await openFiles(files);
        } catch (err) { panel.setError(reloadMessage(err)); }
    });
}

// ---------------------------------------------------------------------------
// The working directory's three buttons. Each is one call and then the same two redraws, because every one of them
// ends in the same place: a folder the page holds, or none, and the pulldown rebuilt from it.
// ---------------------------------------------------------------------------
function bindWorkdir() {
    el('workdir-pick').addEventListener('click', async () => {
        let handle = null;
        try {
            handle = await workdir.pick();
        } catch (e) {
            // A cancelled dialog is not a failure and gets no message. Anything else is said out loud, because a
            // button that does nothing at all is the worst answer a page can give.
            if (e && e.name === 'AbortError') return;
            panel.setError('the folder was not granted: ' + (e.message || e));
            return;
        }
        try { await adoptDirectory(handle); } catch (e) { panel.setError(reloadMessage(e)); }
    });
    // REOPEN IS A CLICK BECAUSE requestPermission NEEDS ONE. That is the whole reason this button exists: the handle
    // came back from IndexedDB on its own, and the only thing missing is a user activation to ask the browser for
    // the permission again - which is the prompt that offers "allow on every visit".
    el('workdir-reopen').addEventListener('click', async () => {
        if (!stored) return;
        const handle = stored;
        let answer;
        try {
            answer = await workdir.requestGrant(handle);
        } catch (e) {
            panel.setError('The browser refused ' + handle.name + ' (' + (e.message || e) + ').'
                           + ' Press "Select working directory..." to pick it again.');
            return;
        }
        if (answer !== 'granted') {
            panel.setError('The page was not allowed to read ' + handle.name + '. You can still hand it the files'
                           + ' themselves. Select or drop them below.');
            return;
        }
        try { await adoptDirectory(handle); } catch (e) { panel.setError(reloadMessage(e)); }
    });
    // Forget drops the page's OWN record of the folder. The permission itself belongs to the browser, which is where
    // it is taken back, and the line says so rather than implying this button revoked anything.
    el('workdir-forget').addEventListener('click', async () => {
        const name = granted ? granted.name : (stored ? stored.name : '');
        const wasSession = !!(granted && granted.session);
        await workdir.forget();
        // The open material was being re-read through this folder. Saying the folder is forgotten and then going
        // on reading it on the next Reload is the kind of half-truth this page does not tell, so the source goes
        // with it. That leaves the material on screen with nothing to re-read it from, which is what Reload says
        // if it is pressed - it does NOT fall back to the files, because the page was not given those files and
        // holding on to handles from a folder it has just let go of would be the same half-truth by another route.
        if (source && source.kind === 'workdir') source = null;
        granted = null;
        stored = null;
        entries = [];
        remembered = false;
        listing++;
        rebuildAssetSelect();
        renderWorkdirPanel(workdirLine());
        renderReloadNote();
        // A DROPPED FOLDER HAD NO RECORD AND NO PERMISSION. Telling someone the page has stopped offering it
        // next visit, and that the browser is still holding a permission they can go and revoke, is untrue of it
        // in both halves: nothing was stored, and nothing was granted.
        // Forget did what it was asked. That is a status, not a refusal.
        panel.setStatus(!name ? ''
            : wasSession
                ? 'The page has let go of ' + name + '. The material on screen stays where it is. Drop the folder'
                  + ' again whenever you want it back.'
                : 'The page has dropped its record of ' + name + ' and will not offer it next visit. The material'
                  + ' on screen stays where it is. The browser keeps its own permission for that folder, and you'
                  + ' take that back in the browser\'s site settings, not here.');
    });
}

// ---------------------------------------------------------------------------
// The panel's controls: the other direction of the two-way binding. Each listener calls set() and NOTHING ELSE, so a
// click and a keystroke end at the same line of code; panel.renderShow / renderSample / renderCamera / renderDisplay
// do the way back. No listener reads another control to work out what the state is.
// ---------------------------------------------------------------------------
function bindControls() {
    // The checkboxes, [element id, state field]. The label beside each names the key that does the same thing.
    for (const [id, field] of [['renorm-box', 'renorm'], ['aniso-box', 'aniso'],
                               ['mips-box', 'mipsOn'], ['lodbias-box', 'lodBias'],
                               ['overlay-box', 'overlay']]) {
        el(id).addEventListener('change', (e) => set(field, e.target.checked));
    }
    // The radio sets. `view` is one three-way choice over two fields, which is what keys 1 and 2 are too.
    for (const [id, fields] of [['view-decode', { raw0: false, raw1: false }],
                                ['view-raw0', { raw0: true, raw1: false }],
                                ['view-raw1', { raw0: false, raw1: true }]]) {
        el(id).addEventListener('change', () => setAll(fields));
    }
    // The log's own controls, and the strict switch beside them. Strict is NOT a state field and does not go through
    // set(): it changes what an error does, not what is drawn, and Reset leaves it alone for the same reason it
    // leaves the unpack path and the colour space alone. Turning it OFF releases a page it has already stopped,
    // because the alternative - a switch that can only ever stop you - would be a trap.
    el('strict-box').addEventListener('change', (e) => {
        strict = e.target.checked;
        const url = new URL(location.href);
        if (strict) url.searchParams.set('strict', '1'); else url.searchParams.delete('strict');
        history.replaceState(null, '', url);
        if (!strict && halted) {
            halted = false;
            panel.setFatal('');
            dirty = true;
        }
    });
    el('log-copy').addEventListener('click', async () => {
        const text = panel.logText();
        if (!text) return;
        // The clipboard API needs a permission this page may not have (and has none at all over plain http on some
        // browsers), so a refusal selects the text instead and says so: something the user can still press Ctrl+C on.
        try {
            await navigator.clipboard.writeText(text);
            panel.setCopyNote('copied');
        } catch (e) {
            const node = el('log-list');
            const range = document.createRange();
            range.selectNodeContents(node);
            const selection = window.getSelection();
            selection.removeAllRanges();
            selection.addRange(range);
            panel.setCopyNote('this browser would not let the page write the clipboard; the text is selected - '
                              + 'press Ctrl+C');
        }
    });
    el('log-clear').addEventListener('click', () => { panel.clearLog(); panel.setCopyNote(''); });

    for (const name of GEOMETRIES) {
        el('geom-' + name).addEventListener('change', () => set('geom', name));
    }
    el('row-box').addEventListener('change', (e) => set('allTextures', e.target.checked));
    el('orbit-box').addEventListener('change', (e) => set('orbit', e.target.checked));
    for (const [id, mode] of [['filter-point', 0], ['filter-bilinear', 1], ['filter-trilinear', 2]]) {
        el(id).addEventListener('change', () => set('filterMode', mode));
    }

    // The five camera numbers. `input` rather than `change`, so the picture follows the spinner and the typing; a
    // field that is mid-edit is left alone by panel.renderCamera, which is what makes that safe.
    for (const [id, field] of [['cam-x', 'x'], ['cam-y', 'y'], ['cam-z', 'z'],
                               ['cam-yaw', 'yaw'], ['cam-pitch', 'pitch']]) {
        el(id).addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            if (!isFinite(value)) return;                       // a half-typed '-' is not a camera
            set(field, field === 'z' ? Math.min(Z_MIN, Math.max(Z_MAX, value)) : value);
        });
    }

    // The canvas colour space (6.5): one configure() call, no pixel changed, and a refusal said out loud.
    el('colorspace-select').addEventListener('change', (e) => {
        const complaint = renderer.setColorSpace(e.target.value);
        panel.setColorSpaceNote(complaint);
        state.colorSpace = renderer.colorSpace;
        dirty = true;
        draw();
    });
}

// ---------------------------------------------------------------------------
// Help. A real <dialog> shown with showModal(), so the browser gives it the backdrop, the focus trap and Esc for
// free - Esc is the one native key with no web meaning, and closing this page is the only thing it can honestly do.
//
// The BUTTON is the contract, because a page does not own F1. F1 is bound as well, on the window and with
// preventDefault, which Chrome honours; a browser that does not will open its own help and the button still works.
// ---------------------------------------------------------------------------
function bindHelp(canvas) {
    const dialog = el('help-dialog');
    const open = () => { if (!dialog.open) dialog.showModal(); };
    const close = () => { if (dialog.open) dialog.close(); };
    el('help-button').addEventListener('click', open);
    el('help-close').addEventListener('click', close);
    // The owner's reset button, beside the key table where someone who has lost track of what they pressed is
    // already looking. It closes the dialog, because the point of pressing it is to see the picture again.
    el('help-reset').addEventListener('click', () => { resetView(); close(); canvas.focus(); });
    // F1, on the WINDOW and on nothing else. One listener, deliberately: a keystroke inside the open dialog bubbles
    // to the window too, so a second listener on the dialog would close it and let this one reopen it in the same
    // event. The window is also where it belongs - F1 is the one key worth answering wherever the focus is, it
    // collides with nothing that can be typed, and a user who has just clicked something in the panel is exactly the
    // user who reaches for it. Held movement keys are dropped, so nothing keeps moving behind a modal dialog.
    //
    // Esc is NOT handled here: <dialog> closes itself on Esc, which is the whole reason this is a real <dialog>, and
    // closing the page is the only honest thing the native viewers' quit key can mean in a browser.
    window.addEventListener('keydown', (e) => {
        if (e.key !== 'F1' || e.ctrlKey || e.metaKey || e.altKey) return;
        e.preventDefault();
        pressed.clear();
        if (dialog.open) close(); else open();
    });
    // The canvas has lost focus to the dialog; give it back, or the next keystroke goes to whatever the panel had.
    // AFTER the current task, not during it: the browser restores focus to the element that was focused when
    // showModal() was called, and it does that after the `close` event, so a focus() inside the handler is undone.
    dialog.addEventListener('close', () => setTimeout(() => canvas.focus(), 0));
}

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
async function main() {
    const canvas = el('canvas');
    applyUrlFields();
    renderer = new Renderer(canvas);
    try {
        await renderer.init(wantedUnpack());
    } catch (e) {
        panel.setFatal(e.message);
        return;
    }
    state.unpack = renderer.unpack;
    panel.setStrict(strict);
    // WHERE A WEBGPU REFUSAL GOES. Its validation is not a layer and cannot be switched on or off - every call is
    // checked, always - so the only choice a page has is what it does with what it is told. By default the message
    // goes in the status line, like every other refusal. ?strict=1 makes it a STOP instead: the message goes to the
    // fatal line and the frame loop halts on the last good frame, because a moving picture can carry an error line
    // past anybody's attention, and the whole lesson of the mip-extent defect is that a wrong picture under a
    // clean-looking page is the worst thing this tool can do. Off by default so an ordinary visitor is not stopped
    // by something harmless; on for anyone testing, and the flag is in the URL table in the README.
    renderer.onError = (message) => {
        panel.setError(message, 'WebGPU');
        if (!strict) return;
        halted = true;
        panel.setFatal('?strict=1: ' + message + ' Drawing has stopped on the last good frame; reload the page'
                       + ' (or drop the strict field) to carry on.', 'WebGPU');
    };
    try {
        await renderer.loadShader('nntc_view.wgsl');
        // The status is CHECKED before the body is used. A 404 has a body too - the server's HTML error page - and
        // handing that to createShaderModule got a module that compiles into nothing and a pipeline that poisons
        // the pass the scene is drawn in, with no sentence anywhere.
        const overlayResponse = await fetch('overlay.wgsl', { cache: 'no-store' });
        if (!overlayResponse.ok) {
            throw new Error('cannot fetch overlay.wgsl (' + overlayResponse.status + ')');
        }
        const overlayCode = await overlayResponse.text();
        overlay = await Overlay.create(renderer.device, renderer.colorFormat, renderer.depthFormat, overlayCode);
    } catch (e) {
        panel.setFatal(e.message);
        return;
    }

    // The canvas is sized by script: its CSS box times devicePixelRatio, 1:1, never scaled by CSS.
    const sizeCanvas = () => {
        const rect = canvas.getBoundingClientRect();
        const ratio = window.devicePixelRatio || 1;
        if (renderer.resize(Math.round(rect.width * ratio), Math.round(rect.height * ratio))) dirty = true;
    };
    new ResizeObserver(sizeCanvas).observe(canvas);
    sizeCanvas();

    // The divider between the panel and the view. KTX2 Studio's column is a fixed width; this one is dragged, because
    // the asset readout and the picture want different splits on different monitors. Nothing here resizes the canvas:
    // the panel's width changes, the flex row gives the rest to the view, and the ResizeObserver above does the work.
    {
        const splitter = el('splitter');
        const panel = el('panel');
        const PANEL_MIN = 240;                    // narrower and the readout's tables wrap into nonsense
        const VIEW_MIN = 160;                     // always leave a picture to look at
        let dragging = false;
        const widthFrom = (clientX) => {
            const max = document.documentElement.clientWidth - VIEW_MIN - splitter.offsetWidth;
            return Math.max(PANEL_MIN, Math.min(max, Math.round(clientX)));
        };
        splitter.addEventListener('pointerdown', (e) => {
            dragging = true;
            splitter.setPointerCapture(e.pointerId);
            splitter.classList.add('dragging');
            e.preventDefault();                   // or the drag selects the panel's text
        });
        splitter.addEventListener('pointermove', (e) => {
            if (!dragging) return;
            panel.style.width = widthFrom(e.clientX) + 'px';
        });
        const stop = (e) => {
            if (!dragging) return;
            dragging = false;
            splitter.classList.remove('dragging');
            if (splitter.hasPointerCapture(e.pointerId)) splitter.releasePointerCapture(e.pointerId);
        };
        splitter.addEventListener('pointerup', stop);
        splitter.addEventListener('pointercancel', stop);
        splitter.addEventListener('dblclick', () => { panel.style.width = ''; });   // back to the stylesheet's width
    }

    bindInput(canvas);
    bindFiles();
    bindWorkdir();
    // Drawn from the feature test before anything is fetched, so the block (or the one quiet line in its place) is
    // on the page from the first paint rather than appearing a moment later.
    renderWorkdirPanel(workdirLine());
    bindControls();
    bindHelp(canvas);
    // The focus goes back to the canvas, as it does from Reset and from the help dialog: a button keeps the focus
    // once it is clicked, and every key this page has goes to the canvas, so clicking Reload with the mouse left
    // the keyboard dead until the view was clicked. Reload is the button pressed most often of the three.
    el('reload').addEventListener('click', async () => { await reloadAndSay(); canvas.focus(); });
    el('reset-view').addEventListener('click', () => { resetView(); canvas.focus(); });
    el('save-bmp').addEventListener('click', async () => { await saveFrame('bmp'); canvas.focus(); });
    el('save-png').addEventListener('click', async () => { await saveFrame('png'); canvas.focus(); });
    // The clipboard needs a user gesture and a secure context, and a page served over plain http from anything but
    // localhost has neither. It says which happened and leaves the link in the status line to be copied by hand,
    // rather than a button that appears to do nothing.
    el('copy-link').addEventListener('click', async () => {
        const link = viewLink();
        try {
            await navigator.clipboard.writeText(link);
            panel.setStatus(source && source.kind === 'bundled'
                ? 'The link is on the clipboard.'
                : 'The link is on the clipboard. It carries the camera and the modes, but not the material: a'
                  + ' material from your own files or folder is not something a link can reach.');
        } catch (e) {
            panel.setError('This browser would not put the link on the clipboard (' + (e.message || e)
                           + '). Here it is: ' + link);
        }
        canvas.focus();
    });
    el('reload-shader').addEventListener('click', reloadShader);
    el('tex-select').addEventListener('change', (e) => set('texShown', parseInt(e.target.value, 10)));
    el('chan-select').addEventListener('change', (e) => set('channel', e.target.value));
    // THE BOX, THE URL AND THE PICTURE ARE BOUND TOGETHER OR NOT AT ALL. The flag has to be written before the
    // reload, because buildAsset reads it to decide what to upload - but it is the reload that makes it true, so a
    // reload that does not load puts the flag, the field and the box back where they were. Writing them first and
    // hoping was a state that described a picture nobody could see: a tick, an unreachable file, and a panel still
    // saying BC5 (GPU) under a ticked box and a ?unpack=1 url.
    //
    // With no material open there is nothing for the box to disagree with, so the choice is simply kept and the
    // next material opens that way, which is what ?unpack= does from the address bar.
    el('unpack-box').addEventListener('change', async (e) => {
        const previous = state.unpack;
        state.unpack = e.target.checked;
        const outcome = await reload();
        if (outcome === 'failed') {
            state.unpack = previous;      // panel.renderDecode assigns the box from the state, so draw() puts it back
            draw();
            return;
        }
        // A PICTURE ON SCREEN THAT COULD NOT BE RELOADED IS A REFUSAL, not a quiet success. reload() answers
        // 'nothing' both for an empty page and for a material whose source has gone (Forget), and in the second
        // case the planes were never re-uploaded: ticking the box, writing ?unpack=1 and announcing a rebind would
        // leave the box, the url and the readout describing a picture that is still bound the old way, which is
        // exactly the disagreement the comment above says was fixed.
        if (outcome === 'nothing' && asset) {
            state.unpack = previous;
            draw();
            panel.setError('There is nothing to re-read this material from, so its planes cannot be bound the'
                           + ' other way. Open it again from a folder, or hand the page its files.');
            return;
        }
        const url = new URL(location.href);
        url.searchParams.set('unpack', state.unpack ? '1' : '0');
        history.replaceState(null, '', url);
        // The reload above has already said "Reloaded NAME", which is true and is not the news: what the person
        // just changed is which path the planes are bound through. With nothing open there is nothing to rebind
        // and the choice is simply kept, which is a different sentence and not a claim about a level that is not
        // there. "Block-compressed" rather than "level 0" because the flag applies to every BC plane of both
        // levels, and a material whose level 0 is not BC is untouched by it.
        panel.setStatus(!asset
            ? (state.unpack ? 'The next material will have its block-compressed planes unpacked on the CPU.'
                            : 'The next material will have its block-compressed planes bound as they are stored.')
            : (state.unpack ? 'Block-compressed planes are now unpacked on the CPU and bound uncompressed.'
                            : 'Block-compressed planes are now bound as they are stored, for the GPU to decode.'));
        draw();
    });
    // ONE PULLDOWN, TWO SOURCES. The value says which: a working-directory entry carries the DIR_PREFIX and its
    // name in the folder, a bundled one is its url. Everything after that is the same load and the same reset.
    el('asset-select').addEventListener('change', async (e) => {
        const value = e.target.value;
        if (!value) return;
        selected = value;
        try {
            if (value.indexOf(DIR_PREFIX) === 0) await openWorkdir(value.substring(DIR_PREFIX.length));
            else await openBundled(value);
        } catch (err) { panel.setError(reloadMessage(err)); }
    });

    draw();

    // The bundled list, so the page opens on a working picture with no file interaction at all.
    try {
        const list = await (await fetch('assets/assets.json', { cache: 'no-store' })).json();
        for (const entry of list.assets) {
            const url = 'assets/' + entry.file;
            const response = await fetch(url, { cache: 'no-store' });
            if (!response.ok) continue;
            const text = await response.text();
            const root = JSON.parse(text);
            bundled.push({ url: url, label: summarise(entry.file, root, text.length).label });
        }
    } catch (e) {
        panel.setError('the bundled list could not be read: ' + e.message);
    }
    const select = el('asset-select');
    rebuildAssetSelect();
    // ?asset= takes what a person would type: the material's NAME as the pulldown shows it, its descriptor's file
    // name, or the url itself. Only the url worked before, so ?asset=m1_m4_c0_3_c1_4 fetched /m1_m4_c0_3_c1_4, got a
    // 404 and opened the page on nothing - a URL field whose obvious value is the one that fails is worse than none.
    const wanted = params.get('asset');
    let resolved = wanted;
    if (wanted) {
        const stem = (u) => u.split('/').pop().replace(/_nntc\.json$/, '').replace(/\.json$/, '');
        const hit = bundled.find((b) => b.url === wanted)
                 || bundled.find((b) => b.url.split('/').pop() === wanted)
                 || bundled.find((b) => stem(b.url) === wanted);
        if (hit) {
            resolved = hit.url;
        } else if (!wanted.includes('/') && !wanted.endsWith('.json')) {
            // A bare name that is not bundled is a mistake worth naming, not a 404 to puzzle over.
            panel.setError('?asset=' + wanted + ' is not one of the bundled materials ('
                           + bundled.map((b) => stem(b.url)).join(', ') + '); a path or a url also works.');
            resolved = null;
        }
    }
    shotWanted = resolved;
    const opening = resolved || (bundled.length ? bundled[0].url : null);
    if (opening) {
        selected = opening;
        select.value = opening;
        try {
            await loadBundled(opening);
        } catch (e) {
            panel.setError(reloadMessage(e));
        }
    }

    // THE WAY BACK IN (5.5). The handle comes out of IndexedDB by itself; the PERMISSION does not come with it. What
    // the browser answers without asking the user anything is queryPermission's business, and whether a restored
    // handle already reads 'granted' - which would remove even the one click - is the plan's one open question here.
    // It is measured rather than assumed, and what it answered on this machine is in README.md.
    //
    // A RESTORED GRANT LISTS THE FOLDER AND CHANGES NOTHING ELSE. The bundled material has already opened by now,
    // and a page that quietly replaced it with whatever sorted first in a folder would be answering a question
    // nobody asked. Picking, reopening or dropping a folder DOES open its first material, because each of those is
    // a person saying "this folder, now".
    if (workdir.supported()) {
        const handle = await workdir.recall();
        // THE USER MAY HAVE GOT THERE FIRST. The block is live from the first paint, and everything above this -
        // assets.json, each descriptor, the first material - is fetched before this line runs, which is time enough
        // to press the button or drop a folder. A record read from IndexedDB before that happened must not then
        // install itself over the folder the person actually chose, so the question is asked again here, after the
        // await, rather than assumed from before it.
        if (handle && !granted && !stored) {
            const answer = await workdir.queryGrant(handle);
            if (answer === 'granted') {
                granted = handle;
                remembered = true;
                await refreshWorkdir();
            } else {
                stored = handle;
                // 'denied' is not 'prompt': the browser has been told no for this site and Reopen will not raise a
                // dialog, so the one line that would otherwise promise it is replaced by the place the answer can
                // actually be changed.
                if (answer === 'denied') {
                    panel.setError('The browser blocks this page from reading ' + handle.name + '. Only the'
                                   + ' browser\'s own site settings can change that. You can still hand the page the'
                                   + ' files themselves. Select or drop them below.');
                }
            }
            renderWorkdirPanel(workdirLine());
        }
    }

    requestAnimationFrame(frame);
    canvas.focus();

    // The same-origin encoder hands off in-memory Files through the normal intake.
    // No uploaded data or persistent browser storage is needed for this preview.
    if (flag('encoder') && window.parent !== window) {
        window.addEventListener('message', (event) => {
            if (event.origin !== location.origin || event.source !== window.parent ||
                event.data?.type !== 'nntc-encoded-files' || !Array.isArray(event.data.files) ||
                !event.data.files.every(file => file instanceof File)) return;
            openFiles(event.data.files);
        });
        window.parent.postMessage({type:'nntc-viewer-ready'}, location.origin);
    }

    // LAST, and after the first frame is scheduled: the shot renders offscreen and does not depend on the canvas
    // having drawn, but everything it does depend on - the adapter, the shader, the asset - has happened by here,
    // and a failure before this point has already written its own message and left the title alone.
    if (flag('shot')) await runShot();
}

main();
