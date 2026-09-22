// panel.js: the left column - what was loaded, what is drawing it, and the controls.
//
// The panel now carries the status that used to be on the picture (the plan's 6.3 and 6.4): the drawn overlay keeps
// only the frame rate and the camera, and everything else - the asset's facts, the mode, the device, which BC path
// drew level 0 - is here, in two kinds of place. Mode and status that a key toggles are shown BY THE CONTROL ITSELF,
// so there is no second line of status text to keep in step; status that is not a control is a readout.
//
// The asset readout is the owner's own list: "the full configuration of the two latent textures: format, # of
// channels, total material bitrate (mip0 and all mipmaps), and average per-texture bitrate (mip0 and all mipmaps), and
// the total # of weights used for decoding". The two latents sit side by side because the question a reader has is how
// the two compare; the four rates sit together in a 2 x 2 so they cannot be misread.

import { ANISO_MAX } from './renderer.js';

const el = (id) => document.getElementById(id);

function esc(text) {
    return String(text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}
function commas(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
function bpp(x) {
    return x.toFixed(2);
}

// ---------------------------------------------------------------------------
// The one status line, the fatal line, and the LOG BEHIND THEM.
//
// The line is what a user sees when the thing they just did failed, and it is overwritten by the next one: that is
// right for a line and wrong for a record. The log is the record. Every message that reaches either line reaches it
// too, numbered and timestamped, so that a reader can tell which of three messages came first - which is nearly
// always the real one - and can select and paste the lot into a bug report. It is <pre> text and not a canvas for
// exactly that reason.
// ---------------------------------------------------------------------------
let logCount = 0;
let logLast = '';

// `source` is WHO SAID IT, and it is the first thing a reader wants: a refusal from WebGPU and a refusal from
// this page look alike once they are both sentences in a list, and they are acted on differently - one is a driver
// or a device saying no, the other is the page declining to do something. Left off, a message is the page's own.
export function logMessage(message, source) {
    const text = String(message || '').trim();
    if (!text) return;
    // The same sentence twice running is one event reported by two paths - an error scope that also throws is the
    // case - and a log that doubles every one of those is harder to read, not more complete.
    if (text === logLast) return;
    logLast = text;
    logCount++;
    const list = el('log-list');
    const time = new Date().toLocaleTimeString();
    // Appended, oldest first, because the sequence is the point and a reader follows it downwards. The pane scrolls
    // to the newest ONLY when it was already at the bottom, so it cannot jump away from someone mid-read.
    const atBottom = list.scrollHeight - list.scrollTop - list.clientHeight < 4;
    list.textContent += (logCount > 1 ? '\n' : '') + logCount + '. [' + time + '] ['
                        + (source || 'page') + '] ' + text;
    if (atBottom) list.scrollTop = list.scrollHeight;
    el('log-summary').textContent = logCount + (logCount === 1 ? ' message' : ' messages');
}

export function clearLog() {
    logCount = 0;
    logLast = '';
    el('log-list').textContent = '';
    el('log-summary').textContent = 'no messages';
}

export function logText() {
    return el('log-list').textContent;
}

// THE STATUS LINE, and there is one of it. It says what just happened - a material opened, a folder granted, a
// Reload that worked or did not - and it is the same slot whether that news is good or bad, because a reader who
// has one place to look finds the news and a reader with two places learns to watch neither.
//
// WHAT DIFFERS IS THE COLOUR AND NOTHING ELSE. A failure carries the red bar; a success is ordinary text, because
// the page doing what it was asked is not an event worth colouring. Callers say WHICH HAPPENED and never what it
// should look like: setError for a refusal, setStatus for anything else.
//
// A STATUS MUST NOT OUTLIVE WHAT IT DESCRIBES. "Reloaded m1" left on screen while you are looking at m4 has
// quietly become a lie. What keeps it honest is that every route which changes the material on screen ends in
// showAsset, and showAsset always writes this line, so news is replaced by newer news rather than expiring on a
// timer. A view control that changes nothing about what is loaded leaves the line alone, which is right: it still
// describes the last thing that actually happened.
export function setError(message, source) {
    const line = el('status-line');
    line.textContent = message || '';
    line.classList.toggle('error', !!message);
    logMessage(message, source);
}

export function setStatus(message, source) {
    const line = el('status-line');
    line.textContent = message || '';
    line.classList.remove('error');
    logMessage(message, source);
}

export function setFatal(message, source) {
    el('fatal').textContent = message || '';
    logMessage(message, source);
}

export function setStrict(on) {
    el('strict-box').checked = on;
}

export function setCopyNote(message) {
    el('log-note').textContent = message || '';
}

// The rows of 5.4: at first the descriptor alone, and as soon as one is read one row per .dds it names. A row filled
// shows the file's name and size; a row still empty is what is missing. They are both the one-at-a-time route and the
// status of the all-at-once route.
//
// `incomplete` is the difference between a list of what a LOADED material is made of and a list of what the page is
// HOLDING while a set is assembled. The heading is the whole of it, and it earns the change: the rows under an
// incomplete set are the only place a user assembling a material out of two folders can see what they have got so
// far, and 'The files this material needs' over a material that is not on screen is what read as half-loaded. The
// message at the top of the panel (main.js, missingMessage) is what says what to do; these are the detail under it.
export function renderFileRows(descriptorName, status, set, incomplete) {
    const host = el('file-rows');
    if (!set && !status) { host.innerHTML = ''; return; }
    const row = (name, have, size) => {
        const mark = have ? '<span class="ok">&#10003;</span>' : '<span class="want">needed</span>';
        const bytes = have && size !== undefined ? ' <span class="quiet">' + commas(size) + ' bytes</span>' : '';
        return '<div class="file-row">' + mark + ' <span class="name">' + esc(name) + '</span>' + bytes + '</div>';
    };
    // Files in hand with no descriptor among them yet: there is no material to list rows against, so what is shown is
    // simply what the page is holding - which is the only honest answer until a .json turns up.
    if (!status) {
        const held = ['<div class="rows-title">the files offered so far</div>'];
        for (const name of set.names) held.push(row(name, true, set.size(name)));
        host.innerHTML = held.join('');
        return;
    }
    const parts = [incomplete
        ? '<div class="rows-title">the files offered so far, and what is still wanted</div>'
        : '<div class="rows-title">The files this material needs</div>'];
    parts.push(row(descriptorName, true, set ? set.size(descriptorName) : undefined));
    for (const name of status.wanted) {
        const have = set ? set.has(name) : false;
        parts.push(row(name, have, have ? set.size(name) : undefined));
    }
    // One file too many costs nothing: a Ctrl+A over a folder brings whatever else is in it, and a set assembled over
    // several goes can pick up more, so the extras are listed and ignored rather than refused.
    if (set) {
        const known = new Set([descriptorName].concat(status.wanted));
        const others = set.names.filter((n) => !known.has(n));
        if (others.length) {
            parts.push('<div class="rows-title">other files offered, ignored</div>');
            for (const name of others) {
                parts.push('<div class="file-row quiet"><span class="name">' + esc(name) + '</span></div>');
            }
        }
    }
    host.innerHTML = parts.join('');
}

// The working directory's block (the plan's 5.5), and the ONE QUIET LINE that stands in its place where the browser
// has no File System Access API. Exactly one of the two is ever on the page, and which one is decided by a feature
// test in workdir.js and never by the browser's name: a control that cannot work is not drawn at all, because a
// disabled button with a tooltip is worse than an absent one, and a user agent string is the wrong thing to ask.
export function renderWorkdir(view) {
    el('workdir-block').hidden = !view.supported;
    el('folder-note').hidden = !!view.supported;
    if (!view.supported) return;
    // The PICKER is a separate question from the block. A browser can have a folder without having a way to ask
    // for one - a dropped folder in Firefox is exactly that - and a button that cannot work is not drawn at all.
    el('workdir-pick').hidden = !view.canPick;
    // "Change" rather than "Select" once there is one, since that is what pressing it would do.
    el('workdir-pick').textContent = view.name ? 'Change working directory...' : 'Select working directory...';
    const reopen = el('workdir-reopen');
    reopen.hidden = !view.reopen;
    if (view.reopen) reopen.textContent = 'Reopen ' + view.reopen;
    el('workdir-forget').hidden = !(view.name || view.reopen);
    el('workdir-note').textContent = view.note || '';
}

function levelCell(asset, index, what, renderer) {
    const L = asset.levels[index];
    switch (what) {
        case 'files': {
            return L.files.map((f) => esc(f.name) + ' ' + commas(f.bytes.length)).join('; ');
        }
        case 'stored':
            return L.files.map((f) => esc(f.storedName)).join(' + ');
        case 'bound': {
            const names = L.files.map((f) => f.boundShortName).join(' + ');
            // WHY, not only what. An unpacked plane on an adapter that HAS the feature was asked for, by the checkbox
            // or by ?unpack=1, and saying so is the difference between a surprise and a setting: RG8 on a card that
            // does BC in hardware reads as a fault until the page admits whose choice it was.
            let how = '';
            if (L.files.some((f) => f.unpacked)) {
                how = renderer && renderer.hasBC ? ' (unpacked, forced)' : ' (unpacked, no hardware BC)';
            } else if (L.bc) {
                how = ' (GPU)';
            }
            return esc(names + how);
        }
        case 'size':
            return L.W + ' x ' + L.H;
        case 'channels':
            return L.C + ' / ' + L.files.map((f) => f.img.channels).join(' + ');
        case 'bits':
            return L.bits.slice(0, L.C).join(' ');
        case 'mips': {
            const last = L.mipSizes[L.mipSizes.length - 1];
            return L.mips + ' (to ' + last[0] + ' x ' + last[1] + ')';
        }
        case 'dequantise':
            return esc(L.dequantiseKind || '-');
        default:
            return '';
    }
}

export function renderAsset(asset, renderer, sourceLabel) {
    const rows = [
        ['file(s) and bytes', 'files'],
        ['format as stored', 'stored'],
        ['bound as', 'bound'],
        ['size', 'size'],
        ['channels used / stored', 'channels'],
        ['bits per channel', 'bits'],
        ['mips', 'mips'],
        ['dequantise', 'dequantise'],
    ];
    const parts = [];
    parts.push('<table><tr><th>row</th><th>level 0</th><th>level 1</th></tr>');
    for (const [label, what] of rows) {
        parts.push('<tr><th>' + label + '</th><td>' + levelCell(asset, 0, what, renderer) + '</td><td>'
                   + levelCell(asset, 1, what, renderer) + '</td></tr>');
        // The material's own count, spanning both columns, straight after the files it comes out of.
        if (what === 'files') {
            parts.push('<tr><th>textures decoded</th><td colspan="2"><b>' + asset.texturesOut + '</b></td></tr>');
        }
    }
    parts.push('</table>');

    // The four rates, the encoder's own arithmetic mirrored (descriptor.js says how and why).
    const r = asset.rates;
    parts.push('<table><tr><th>bits per source pixel</th><th>at mip 0</th><th>with the whole chain</th></tr>');
    parts.push('<tr><th>the material (the set)</th><td class="num">' + bpp(r.setBase)
               + ' <span class="quiet">(level 0 ' + bpp(r.level0Base) + ' + level 1 ' + bpp(r.level1Base)
               + ')</span></td><td class="num">' + bpp(r.setAll) + '</td></tr>');
    // Named, not implied: the per-texture average means nothing without the number it is averaged over, and the
    // material's texture count is the one figure a reader cannot get from either latent's own row.
    parts.push('<tr><th>average per texture <span class="quiet">(' + asset.texturesOut + ' texture'
               + (asset.texturesOut === 1 ? '' : 's') + ' in this material)</span></th><td class="num">'
               + bpp(r.perTextureBase) + '</td><td class="num">' + bpp(r.perTextureAll) + '</td></tr>');
    if (r.boundDiffers) {
        parts.push('<tr><th>as bound on this GPU</th><td class="num">' + bpp(r.bound.setBase)
                   + '</td><td class="num">' + bpp(r.bound.setAll) + '</td></tr>');
    }
    parts.push('</table>');
    if (r.boundDiffers) {
        parts.push('<p class="quiet">The first two rows are what the FILES cost, which is what the encoder printed.'
                   + ' The last is what is bound on this GPU: the unpack trades the block format for a plain'
                   + ' uncompressed plane, so a reader here should not think the format costs what the fallback'
                   + ' costs.</p>');
    }

    // WHAT phi IS, in the notation docs/FORMAT.md uses, rather than the descriptor's own private spelling of it.
    // The descriptor writes the three groups as 'a b sc' (src/export.cpp) and both native viewers parse that; it is a
    // format token, not a thing to put in front of a reader, and the tree has two other names for the same three
    // groups besides (z0/z1 in FORMAT.md, s/c in MATHEMATICS.md). The token is kept as a trailing aside for anyone
    // matching this page against the file, and anything unrecognised falls back to printing the token alone.
    const d = asset.decoder;
    const TERM_PHI = { a: 'z1', b: 'z0', sc: 'z0*z1' };
    const groups = String(d.terms).trim().split(/\s+/);
    const known = groups.every((w) => TERM_PHI[w]);
    const phi = known ? '[' + groups.map((w) => TERM_PHI[w]).join(', ') + ']' : "'" + d.terms + "'";
    parts.push('<p>decoder: <b>' + esc(d.type) + '</b>, phi = ' + esc(phi) + ' <span class="quiet">(z0 is level 0, '
               + 'z1 is level 1' + (known ? "; the descriptor spells it '" + esc(d.terms) + "'" : '') + ')</span>, '
               + d.nin + ' features &rarr; ' + d.nout + ' outputs (' + asset.texturesOut + ' texture'
               + (asset.texturesOut === 1 ? '' : 's') + '), <b>' + d.weightCount + ' weights</b></p>');

    const padded = (asset.decodeW && asset.decodeW !== asset.srcW) || (asset.decodeH && asset.decodeH !== asset.srcH);
    parts.push('<p>source ' + asset.srcW + ' x ' + asset.srcH + ', decoded at ' + asset.decodeW + ' x '
               + asset.decodeH + (padded ? ' <span class="quiet">(padded)</span>' : '') + '</p>');
    if (asset.inputs.length) {
        const inputs = asset.inputs.map((input, i) => {
            const type = input.type ? ' ' + esc(input.type) : '';
            return i + ': ' + esc(input.file || '?') + type + (input.srgb ? ' <span class="quiet">srgb</span>' : '');
        });
        parts.push('<p>textures: ' + inputs.join(' &middot; ') + '</p>');
    }
    parts.push('<p>block ' + asset.block + ': level 1 is sampled ' + asset.lodBiasLevel1
               + ' mips coarser - its gradients are scaled by ' + Math.pow(2, asset.lodBiasLevel1) + '</p>');
    parts.push('<p class="quiet">' + esc(sourceLabel) + ', read at ' + asset.loadedAt.toLocaleTimeString()
               + (asset.older ? " - format 'ntc-dds-1', the older name of nntc-dds-1" : '') + '</p>');
    el('asset-readout').innerHTML = parts.join('');
}

// The pulldown is rebuilt when the COUNT or any LABEL differs, not on the count alone. draw() calls this every
// frame that changes anything, so it cannot simply rebuild unconditionally without throwing away a pulldown the
// user may have open; but testing the count alone left one one-texture material's names over another's - open the
// 1024 material over m1 and the readout said npot360x200.png while this still said '0 - m1.png'. The labels are
// what a reader matches against the readout, so the labels are what is compared.
export function renderTextureSelect(asset, state) {
    const select = el('tex-select');
    const labels = [];
    for (let i = 0; i < asset.texturesOut; i++) {
        const input = asset.inputs[i];
        labels.push(input && input.file ? (i + ' - ' + input.file) : String(i));
    }
    const stale = select.options.length !== labels.length
                  || labels.some((text, i) => select.options[i].textContent !== text);
    if (stale) {
        select.innerHTML = '';
        for (let i = 0; i < labels.length; i++) {
            const option = document.createElement('option');
            option.value = String(i);
            option.textContent = labels[i];
            select.appendChild(option);
        }
    }
    select.value = String(state.texShown);
}

// The mode the native strip's second line carries, as CONTROLS rather than as text. The two-way binding of 6.3 in
// one direction: every function below ASSIGNS from the state object and reads nothing back out of the DOM, so a key
// and a click leave the panel in the same place. (The other direction is main.js's listeners, which all call set().)
// There is no second line of status text under any of them on purpose: a radio set to Trilinear IS the filter
// readout, and a readout kept beside a control is a readout that can disagree with it.
export function renderShow(asset, state) {
    el('view-decode').checked = !state.raw0 && !state.raw1;
    el('view-raw0').checked = state.raw0;
    el('view-raw1').checked = state.raw1;
    el('geom-quad').checked = state.geom === 'quad';
    el('geom-cube').checked = state.geom === 'cube';
    el('geom-sphere').checked = state.geom === 'sphere';
    el('row-box').checked = state.allTextures;
    el('orbit-box').checked = state.orbit;
    el('renorm-box').checked = state.renorm;
    el('chan-select').value = state.channel;
}

export function renderSample(state) {
    el('filter-point').checked = state.filterMode === 0;
    el('filter-bilinear').checked = state.filterMode === 1;
    el('filter-trilinear').checked = state.filterMode === 2;
    el('aniso-box').checked = state.aniso;
    el('mips-box').checked = state.mipsOn;
    el('lodbias-box').checked = state.lodBias;
    // Anisotropy can be on in point and bilinear too, where it does nothing.
    el('aniso-note').textContent = state.aniso
        ? 'MaxAnisotropy ' + ANISO_MAX + (state.filterMode === 2 ? '' : ', trilinear filtering only')
        : 'off';
}

// The five numbers. A field being edited is LEFT ALONE: assigning to an <input type="number"> that has focus moves
// the caret and eats a half-typed minus sign, and the frame loop calls this every time a held key moves the camera.
export function renderCamera(state) {
    const put = (id, value, digits) => {
        const input = el(id);
        if (document.activeElement === input) return;
        const text = value.toFixed(digits);
        if (input.value !== text) input.value = text;
    };
    put('cam-x', state.x, 2);
    put('cam-y', state.y, 2);
    put('cam-z', state.z, 2);
    put('cam-yaw', state.yaw, 1);
    put('cam-pitch', state.pitch, 1);
}

export function renderDisplay(state) {
    el('colorspace-select').value = state.colorSpace;
    el('overlay-box').checked = state.overlay;
}

// A colour space the browser or the display would not take. KTX2 Studio's control says so rather than pretending;
// so does this one, and the <select> goes back to what is actually configured.
export function setColorSpaceNote(message) {
    const note = el('colorspace-note');
    note.classList.toggle('error', !!message);
    if (message) {
        note.textContent = message;
    } else {
        note.textContent = 'Display only: it relabels the canvas and changes no pixel, so the same numbers look more'
            + ' saturated on a wide-gamut screen. The native viewers have no such control, so anything but sRGB is a'
            + ' picture they cannot draw.';
    }
}

export function renderDecode(renderer, asset, state) {
    const info = renderer.info || {};
    const described = [info.vendor, info.architecture, info.device, info.description]
        .filter((x) => x).join(' ');
    const parts = [];
    parts.push('<p>adapter: <b>' + esc(described || '(the browser reports nothing about it)') + '</b></p>');
    parts.push('<p>texture-compression-bc: <b>' + (renderer.hasBC ? 'granted' : 'absent')
               + '</b> &middot; maxTextureDimension2D: <b>' + renderer.maxTextureDimension2D + '</b></p>');
    parts.push('<p>decode: <b>plain WGSL</b>, one affine layer over two hardware samples</p>');
    if (asset) {
        const L0 = asset.levels[0];
        const how = L0.files.some((f) => f.unpacked)
            ? L0.files.map((f) => f.shortName).join(' + ') + ' unpacked to '
              + L0.files.map((f) => f.boundShortName).join(' + ')
            : (L0.bc ? L0.files.map((f) => f.shortName).join(' + ') + ' (GPU)' : L0.files[0].shortName);
        parts.push('<p>level 0: <b>' + esc(how) + '</b></p>');
    }
    el('decode-readout').innerHTML = parts.join('');
    const box = el('unpack-box');
    box.checked = state.unpack;
    box.disabled = !renderer.hasBC;
    el('unpack-note').textContent = renderer.hasBC
        ? 'The same rendering path with one format changed: a bc5-rg-unorm texture and an rg8unorm texture are the'
          + ' same texture in the shader. ?unpack=1 forces it here so the fallback can be compared with the hardware'
          + ' on the same GPU; ?unpack=0 asks for hardware BC and is refused by name where there is none.'
        : 'forced: no BC on this adapter';
}
