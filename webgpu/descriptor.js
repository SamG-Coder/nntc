// descriptor.js: PREFIX_nntc.json read and refused exactly as ../viewer/main.cpp's load_asset reads and refuses it
// (lines 438-521 there), plus the bitrate arithmetic the encoder prints, mirrored rather than derived again.
//
// Every refusal here is one of that viewer's, with its sentence: the format string, exactly two textures, the level
// index, the bits, lo / hi, decoder.type, an empty `hidden`, the identity output, `terms` in the order the shader
// builds phi, nin and nout against the channel counts, nout <= 18, and the two-file spelling's cross-check of each
// file's own dxgi_format_id and channels_stored against the .dds header. A descriptor that this viewer would refuse is
// refused here, by name, and the previous material stays on screen.
//
// THE BITRATES ARE THE ENCODER'S ARITHMETIC, MIRRORED (../src/main.cpp:3237-3260), and that is deliberate. A
// block-compressed level costs WHOLE 4x4 BLOCKS - ceil(w/4) * ceil(h/4) of them at 8 bytes (BC4) or 16 (BC5) - and not
// w * h * bytes_per_texel. The two agree only when both dimensions are multiples of 4, and the encoder's own comment
// records what deriving it the other way cost: a 40x40 crop reported 18.33 bpp against the 18.66 the files actually
// take. So the four numbers the panel shows are the four the encoder's `memory` lines print, to the printed digit.

import { ddsRead, dxgiName, dxgiShortName, DXGI_BC4_UNORM, DXGI_BC5_UNORM } from './dds.js';
import { unpackImage } from './bcdec.js';

export const LOD_BIAS_LEVEL1_MAX = 8.0;

// A JSON text that parses AND carries one of the two format strings is a descriptor; anything else is not one, and the
// matcher uses that to tell a descriptor from, say, a *_source_material.json sitting in the same folder.
//
// WHY THE TWO WAYS OF NOT BEING ONE ARE KEPT APART. A text that will not PARSE and a text that parses and is not a
// descriptor fail here for different reasons, and the one sentence that used to cover both - 'is not an NNTC
// descriptor (no "format": "nntc-dds-1")' - is simply untrue of the first: a file the encoder is halfway through
// writing has no keys at all to be missing one of, and what its reader needs to be told is to wait and press Reload,
// which is the case the plan's 5.7 says Reload exists for. So the reason travels back with the result. The old
// yes-or-no call is kept beside it for findDescriptors, which wants nothing but the answer.
export function readDescriptorText(text) {
    let root;
    try {
        root = JSON.parse(text);
    } catch (e) {
        return { root: null, parsed: false, why: e.message || String(e) };
    }
    if (!root || typeof root !== 'object' || Array.isArray(root)) {
        return { root: null, parsed: true, why: 'its top level is not a JSON object' };
    }
    if (root.format !== 'nntc-dds-1' && root.format !== 'ntc-dds-1') {
        return { root: null, parsed: true, why: 'no "format": "nntc-dds-1"' };
    }
    return { root: root, parsed: true, why: '' };
}

export function isDescriptorText(text) {
    return readDescriptorText(text).root;
}

// The .dds names a descriptor asks for, in load order, so the file rows can be built before any of them has arrived.
// A name with a path separator is taken as written, exactly as docs/FORMAT.md section 1 says a reader does - and it
// cannot be resolved from a selection of files, which is reported rather than guessed at.
export function namedFiles(root) {
    const out = [];
    const texs = root.textures;
    if (!Array.isArray(texs)) return out;
    for (const t of texs) {
        if (Array.isArray(t.files)) {
            for (const e of t.files) {
                const name = typeof e === 'string' ? e : (e && typeof e.file === 'string' ? e.file : '');
                if (name) out.push(name);
            }
        } else if (typeof t.file === 'string' && t.file) {
            out.push(t.file);
        }
    }
    return out;
}

// One line for the pulldown, from the descriptor alone: `m1_m4 - 512x512, 4 textures, level 0 BC5+BC4, 612 KB`.
// The byte count is computed rather than asked for, because a .dds is its 148-byte header and then its levels tightly
// packed, so the descriptor's own mip_sizes and formats give the file's size exactly - and a disagreement with what
// the server sends is caught later by dds.js's "N bytes of pixel data expected, M present".
export function summarise(name, root, descriptorBytes) {
    const texs = Array.isArray(root.textures) ? root.textures : [];
    let bytes = descriptorBytes || 0;
    let level0 = '?';
    for (const t of texs) {
        const sizes = Array.isArray(t.mip_sizes) ? t.mip_sizes : [];
        const entries = Array.isArray(t.files) ? t.files : [t];
        const names = [];
        for (const e of entries) {
            const dxgi = typeof e === 'object' && e ? (typeof e.dxgi_format_id === 'number' ? e.dxgi_format_id : 0) : 0;
            const stored = typeof e === 'object' && e && typeof e.channels_stored === 'number' ? e.channels_stored : 0;
            const isBC = dxgi === DXGI_BC4_UNORM || dxgi === DXGI_BC5_UNORM;
            const blockBytes = dxgi === DXGI_BC4_UNORM ? 8 : 16;
            names.push(dxgiShortName(dxgi));
            bytes += 148;
            for (const [w, h] of sizes) {
                bytes += isBC ? ((w + 3) >> 2) * ((h + 3) >> 2) * blockBytes : w * h * stored;
            }
        }
        if ((numberOr(t.level, -1) | 0) === 0) level0 = names.join('+');
    }
    const decode = root.decode || {};
    const prefix = name.replace(/_?n?ntc\.json$/i, '').replace(/\.json$/i, '');
    return {
        label: prefix + ' - ' + (numberOr(decode.width, 0) | 0) + 'x' + (numberOr(decode.height, 0) | 0) + ', '
               + (numberOr(decode.textures_out, 1) | 0) + ' texture'
               + ((numberOr(decode.textures_out, 1) | 0) === 1 ? '' : 's')
               + ', level 0 ' + level0 + ', ' + Math.round(bytes / 1024) + ' KB',
        bytes: bytes,
    };
}

function fail(message) {
    throw new Error(message);
}

function numberOr(value, dflt) {
    return typeof value === 'number' ? value : dflt;
}

// The descriptor, its .dds files (a Map from the name the descriptor uses to a Uint8Array) and the unpack decision,
// turned into everything the renderer and the panel need. Throws an Error with one sentence on any refusal.
export function buildAsset(descriptorName, root, fileBytes, unpack) {
    if (root.format !== 'nntc-dds-1' && root.format !== 'ntc-dds-1') {
        fail("unknown format '" + root.format + "' (expected nntc-dds-1)");
    }
    const asset = {
        name: descriptorName,
        older: root.format === 'ntc-dds-1',
        srcW: 0, srcH: 0, decodeW: 0, decodeH: 0,
        texturesOut: 1, block: 4, lodBiasLevel1: 2,
        inputs: [], levels: [], loadedAt: new Date(),
    };
    const src = root.source;
    asset.srcW = src ? (numberOr(src.width, 0) | 0) : 0;
    asset.srcH = src ? (numberOr(src.height, 0) | 0) : 0;
    if (src && Array.isArray(src.inputs)) asset.inputs = src.inputs;
    const dec = root.decode;
    asset.decodeW = dec ? (numberOr(dec.width, 0) | 0) : 0;
    asset.decodeH = dec ? (numberOr(dec.height, 0) | 0) : 0;
    asset.texturesOut = dec ? (numberOr(dec.textures_out, 1) | 0) : 1;
    asset.block = numberOr(root.block, 4) | 0;

    const texs = root.textures;
    if (!Array.isArray(texs) || texs.length !== 2) fail('the JSON must describe two textures');

    for (let l = 0; l < 2; l++) {
        const t = texs[l];
        const L = { level: l, files: [], bits: [8, 8, 8, 8], lo: [0, 0, 0, 0], hi: [1, 1, 1, 1] };
        if ((numberOr(t.level, -1) | 0) !== l) fail('textures[' + l + '] is not level ' + l);
        L.C = numberOr(t.channels_used, 0) | 0;
        // bits_per_channel decides a shift and a divisor, so 0 or a value above 8 would be a negative shift or a
        // division by zero. An entry that is not a number at all would read as 0, so that is checked too.
        const bits = t.bits_per_channel;
        for (let c = 0; c < 4; c++) {
            L.bits[c] = Array.isArray(bits) && c < bits.length && typeof bits[c] === 'number' ? (bits[c] | 0) : 8;
        }
        for (let c = 0; c < 4; c++) {
            const loBits = l === 0 ? 1 : 4;   // level 0 is 8 bits under --l0 bc8 and 1-4 under --l0 palette; level 1 is 4-8
            if (L.bits[c] < loBits || L.bits[c] > 8) {
                fail('level ' + l + ': bits_per_channel[' + c + '] is ' + L.bits[c] + ', which is outside '
                     + loBits + '..8');
            }
        }
        // "file" names one .dds and "files" names two, which is how a block-compressed level 0 of three or four
        // channels is stored: BC5 carries two channels, so channels 0-1 are the first file and what is left is the
        // second. Each entry of "files" is an OBJECT carrying that file's own dxgi_format_id and channels_stored,
        // because the two need not agree; a BARE STRING is the older spelling with nothing to cross-check, and every
        // asset written before the two files were allowed to differ in format carries one.
        let names = [], want = [];
        if (Array.isArray(t.files)) {
            if (t.files.length !== 2) fail('level ' + l + ': "files" must name two .dds files');
            for (let i = 0; i < 2; i++) {
                const e = t.files[i];
                const isObject = e && typeof e === 'object' && !Array.isArray(e);
                if (!isObject && typeof e !== 'string') {
                    fail('level ' + l + ': "files"[' + i + '] must be a file name or an object carrying one');
                }
                const name = isObject ? e.file : e;
                if (typeof name !== 'string' || !name) fail('level ' + l + ': "files"[' + i + '] names no file');
                names.push(name);
                want.push({
                    dxgi: isObject ? (numberOr(e.dxgi_format_id, 0) | 0) : 0,     // 0 = the older spelling: nothing to cross-check
                    stored: isObject ? (numberOr(e.channels_stored, 0) | 0) : 0,
                });
            }
        } else {
            if (typeof t.file !== 'string' || !t.file) fail('level ' + l + ': no "file" and no "files"');
            names.push(t.file);
            want.push({ dxgi: numberOr(t.dxgi_format_id, 0) | 0, stored: numberOr(t.channels_stored, 0) | 0 });
        }
        for (let i = 0; i < names.length; i++) {
            const bytes = fileBytes.get(names[i]);
            if (!bytes) fail("level " + l + ": '" + names[i] + "' was not given to the reader");
            const img = ddsRead(names[i], bytes);
            if (i > 0) {
                // The two files of one level share the extent and the level count, and NOT the format.
                const first = L.files[0].img;
                if (img.W !== first.W || img.H !== first.H || img.mips !== first.mips) {
                    fail("'" + names[i] + "' does not match the first file of this level (" + first.W + 'x' + first.H
                         + ', ' + first.mips + ' level' + (first.mips === 1 ? '' : 's') + ')');
                }
                if (img.bc !== first.bc) {
                    fail("'" + names[i] + "' is " + (img.bc ? 'block-compressed' : 'uncompressed')
                         + ' where the first file of this level is not');
                }
            }
            if ((want[i].dxgi && want[i].dxgi !== img.dxgi) || (want[i].stored && want[i].stored !== img.channels)) {
                fail('level ' + l + ': the JSON says ' + names[i] + ' is DXGI ' + want[i].dxgi + ' storing '
                     + want[i].stored + ' channel(s) and the file is DXGI ' + img.dxgi + ' storing ' + img.channels);
            }
            L.files.push({ name: names[i], img: img, bytes: bytes });
        }
        const first = L.files[0].img;
        L.W = first.W; L.H = first.H; L.mips = first.mips; L.bc = first.bc;
        // The descriptor's own size and mip count against the headers: a disagreement is the "wrong file" case, and
        // the reader says which two numbers differ rather than rendering whatever it was handed.
        const wantW = numberOr(t.width, 0) | 0, wantH = numberOr(t.height, 0) | 0;
        if (wantW && wantH && (wantW !== L.W || wantH !== L.H)) {
            fail('level ' + l + ': the JSON says ' + wantW + 'x' + wantH + ' and ' + L.files[0].name + ' is '
                 + L.W + 'x' + L.H);
        }
        const wantMips = numberOr(t.mip_count, 0) | 0;
        if (wantMips && wantMips !== L.mips) {
            fail('level ' + l + ': the JSON says ' + wantMips + ' mip level(s) and ' + L.files[0].name + ' has '
                 + L.mips);
        }
        // THE BASE OF A BLOCK-COMPRESSED TEXTURE MUST BE A MULTIPLE OF 4, and ONLY the base: that is the tree's own
        // rule ("we support non-power-of-two, non-square, BUT the base mipmap must be divisible by 4 texels") and
        // Direct3D's, and WebGPU enforces it at createTexture. The LOWER levels are not checked and must not be - a
        // 360x200 base gives a perfectly ordinary 90x50 level and that case is the whole point of the chain.
        //
        // It is refused HERE, before any device call, because a stated refusal beats a validation error even now
        // that validation is surfaced: this says which file, how big it is and what to do instead. nntc_encode pads
        // the base, so an asset out of this tree cannot hit it; a hand-made or third-party .dds can.
        if (L.bc && !unpack && (L.W % 4 || L.H % 4)) {
            fail("level " + l + ": '" + L.files[0].name + "' is " + dxgiShortName(first.dxgi) + ' and its base is '
                 + L.W + 'x' + L.H + '; a block-compressed base must be a multiple of 4 on both axes (the lower mip'
                 + ' levels need not be). nntc_encode pads the base, so this file was not written by it. The CPU'
                 + ' unpack has no such rule and will load it: add ?unpack=1 to the address, or tick "Unpack BC on'
                 + ' the CPU" in the Decode block.');
        }
        L.mipSizes = Array.isArray(t.mip_sizes) ? t.mip_sizes : first.levels.map((lv) => [lv.w, lv.h]);
        // The channels the level's files hold between them: one file stores its own, two store the sum (2 + 1 for a
        // three-channel BC level 0, which is exactly the three the decoder reads).
        let storedAll = 0;
        for (const f of L.files) storedAll += f.img.channels;
        L.storedAll = storedAll;
        if (L.C < 1 || L.C > 4 || L.C > storedAll) {
            fail('level ' + l + ': ' + L.C + ' channels used of ' + storedAll + ' stored');
        }
        if (l === 1) {
            // Absent means log2(block), which is what every writer puts there. Present and not a number is refused, as
            // an out-of-range number is: it used to fall back to the default in silence, the one failure a hand edit
            // can make.
            const lbv = root.lod_bias_level1;
            if (lbv !== undefined && typeof lbv !== 'number') fail("the descriptor's lod_bias_level1 is not a number");
            const lb = lbv !== undefined ? lbv : Math.log2(asset.block > 1 ? asset.block : 1);
            if (!(lb >= 0.0 && lb <= LOD_BIAS_LEVEL1_MAX)) {
                fail("the descriptor's lod_bias_level1 is " + lb + ', which is outside [0, ' + LOD_BIAS_LEVEL1_MAX
                     + ']: it is the number of mip levels level 1 is shifted by, so it cannot be negative and cannot'
                     + ' be that large');
            }
            asset.lodBiasLevel1 = lb;
        }
        const dq = t.dequantise;
        const lo = dq ? dq.lo : null, hi = dq ? dq.hi : null;
        if (!Array.isArray(lo) || !Array.isArray(hi) || lo.length < L.C || hi.length < L.C) {
            fail('level ' + l + ': no lo / hi per channel');
        }
        for (let c = 0; c < L.C; c++) { L.lo[c] = lo[c]; L.hi[c] = hi[c]; }
        L.dequantiseKind = dq && typeof dq.kind === 'string' ? dq.kind : '';
        asset.levels.push(L);
    }

    const d = root.decoder;
    if (!d || typeof d !== 'object') fail('no decoder in the JSON');
    if (d.type !== 'bilinear') fail("decoder type '" + d.type + "': this viewer decodes the bilinear decoder only");
    if (Array.isArray(d.hidden) && d.hidden.length) fail('hidden layers are not supported');
    if (d.output !== 'identity') fail("output rule '" + d.output + "' is not the identity");
    const C0 = numberOr(d.C0, 0) | 0, C1 = numberOr(d.C1, 0) | 0;
    const nin = numberOr(d.nin, 0) | 0, nout = numberOr(d.nout, 0) | 0;
    if (C0 !== asset.levels[0].C || C1 !== asset.levels[1].C || nin < 1 || nin > 24 || nout < 3 || nout > 18
        || nout !== 3 * asset.texturesOut) {
        fail('decoder shape C0 ' + C0 + ' C1 ' + C1 + ' nin ' + nin + ' nout ' + nout
             + ' does not fit the textures (' + asset.levels[0].C + ' / ' + asset.levels[1].C + ' channels, '
             + (3 * asset.texturesOut) + " outputs) or the shader's limits (nin <= 24, nout <= 18)");
    }
    // The terms are a SEQUENCE, not a set: the shader builds phi in the order a, b, sc, so "b a sc" would name the same
    // three groups and describe a different feature vector. Anything out of that order is refused, never reordered.
    const terms = typeof d.terms === 'string' ? d.terms : '';
    let mask = 0, last = -1;
    for (const w of terms.split(/\s+/).filter((x) => x.length)) {
        const which = w === 'a' ? 0 : (w === 'b' ? 1 : (w === 'sc' ? 2 : -1));
        if (which < 0) fail("the term '" + w + "' is not one the shader builds (a, b, sc)");
        if (which <= last) {
            fail("the terms '" + terms + "' are not in the order the shader builds phi (a, then b, then sc)");
        }
        last = which;
        mask |= 1 << which;
    }
    let expect = 0;
    if (mask & 1) expect += C1;
    if (mask & 2) expect += C0;
    if (mask & 4) expect += C0 * C1;
    if (expect !== nin) fail("the terms '" + terms + "' give " + expect + ' features, the decoder has nin ' + nin);
    const layers = d.layers;
    if (!Array.isArray(layers) || layers.length !== 1) fail('expected one layer');
    const lay = layers[0];
    const Wv = lay ? lay.weights : null, bv = lay ? lay.bias : null;
    if ((numberOr(lay.rows, 0) | 0) !== nout || (numberOr(lay.cols, 0) | 0) !== nin
        || !Array.isArray(Wv) || !Array.isArray(bv) || Wv.length !== nout * nin || bv.length !== nout) {
        fail('the layer is not ' + nout + ' x ' + nin);
    }
    asset.decoder = {
        type: 'bilinear', terms: terms, mask: mask, C0: C0, C1: C1, nin: nin, nout: nout,
        weights: Wv, bias: bv, weightCount: nout * nin + nout,
    };

    // The unpack decision, per file: a block-compressed level whose blocks the page decodes on the CPU (3.3). The
    // shader does not change - a bc5-rg-unorm texture and an rg8unorm texture are the same texture_2d<f32> under the
    // same sampler - so this only decides what createTexture is told and what writeTexture is handed.
    for (const L of asset.levels) {
        for (const f of L.files) {
            f.unpacked = unpack && f.img.bc;
            f.levelBytes = f.unpacked ? unpackImage(f.img) : null;
            f.boundChannels = f.img.channels;
            f.boundFormat = f.unpacked
                ? (f.img.channels === 1 ? 'r8unorm' : 'rg8unorm')
                : (f.img.bc ? (f.img.dxgi === DXGI_BC4_UNORM ? 'bc4-r-unorm' : 'bc5-rg-unorm')
                            : (f.img.channels === 1 ? 'r8unorm' : (f.img.channels === 2 ? 'rg8unorm' : 'rgba8unorm')));
            f.storedName = dxgiName(f.img.dxgi);
            f.shortName = dxgiShortName(f.img.dxgi);
            f.boundShortName = f.unpacked ? (f.img.channels === 1 ? 'R8' : 'RG8') : f.shortName;
        }
    }
    asset.rates = computeRates(asset);
    return asset;
}

// The encoder's `memory` lines, mirrored (../src/main.cpp:3237-3260). For each stored plane m, level 0's bytes are
// ceil(w0 / 4) * ceil(h0 / 4) * block_bytes for a block format - 8 for BC4, 16 for BC5, summed over a two-file level 0
// - or w0 * h0 * bytes_per_texel for an uncompressed one; level 1's are w1 * h1 * bytes_per_texel. The base rate is
// 8 * bytes_of_plane_0 / (level 0's base texels), the chain rate the same sum over every plane, and per texture is
// each divided by textures_out.
//
// `bound` computes the same figure over the formats actually bound instead of the ones in the file, which is what the
// unpack changes: a reader on a phone should not think the format costs what the fallback costs.
function planeBytes(asset, bound) {
    const L0 = asset.levels[0], L1 = asset.levels[1];
    let l0BlockBytes = 0;
    for (const f of L0.files) {
        const isBlock = bound ? (f.boundFormat.startsWith('bc')) : f.img.bc;
        if (isBlock) l0BlockBytes += f.img.dxgi === DXGI_BC4_UNORM ? 8 : (f.img.dxgi === DXGI_BC5_UNORM ? 16 : 0);
    }
    // The uncompressed cost per texel: the encoder writes a three-channel plane as four channels, and so does the
    // unpack's target (r8unorm and rg8unorm per file, summed).
    let l0Bpt = 0;
    if (!l0BlockBytes) {
        if (bound) { for (const f of L0.files) l0Bpt += f.boundChannels; }
        else { l0Bpt = L0.C === 3 ? 4 : L0.C; }
    }
    const l1Bpt = L1.C === 3 ? 4 : L1.C;
    const planes = Math.min(L0.mipSizes.length, L1.mipSizes.length);
    let baseBytes = 0, allBytes = 0, baseL0Bytes = 0;
    for (let i = 0; i < planes; i++) {
        const w0 = L0.mipSizes[i][0] | 0, h0 = L0.mipSizes[i][1] | 0;
        const w1 = L1.mipSizes[i][0] | 0, h1 = L1.mipSizes[i][1] | 0;
        const b0 = l0BlockBytes ? ((w0 + 3) >> 2) * ((h0 + 3) >> 2) * l0BlockBytes : l0Bpt * w0 * h0;
        const b = b0 + l1Bpt * w1 * h1;
        if (i === 0) { baseBytes = b; baseL0Bytes = b0; }
        allBytes += b;
    }
    const px = (L0.mipSizes[0][0] | 0) * (L0.mipSizes[0][1] | 0);
    const setBase = 8.0 * baseBytes / px, setAll = 8.0 * allBytes / px, l0Base = 8.0 * baseL0Bytes / px;
    return {
        setBase: setBase, setAll: setAll, level0Base: l0Base, level1Base: setBase - l0Base,
        perTextureBase: setBase / asset.texturesOut, perTextureAll: setAll / asset.texturesOut,
    };
}

function computeRates(asset) {
    const stored = planeBytes(asset, false);
    const bound = planeBytes(asset, true);
    stored.bound = bound;
    stored.boundDiffers = Math.abs(bound.setBase - stored.setBase) > 1e-9
                          || Math.abs(bound.setAll - stored.setAll) > 1e-9;
    return stored;
}
