// dds.js: the DX10 .dds reader, the JavaScript twin of ../shared/dds.h.
//
// Same header walk, same bounds, same one-line refusals - including "N bytes of pixel data expected, M present", which
// is what catches a .dds the encoder is still writing (docs/FORMAT.md section 2; the encoder writes the asset before
// the PNGs, so a file caught mid-write is shorter than its header promises).
//
// What it deliberately does NOT do is touch a graphics API: the caller turns the levels into a texture with whatever it
// has, exactly as the C++ header's comment says. The levels are base-first and tightly packed, so a level is w * h * C
// bytes when the format is uncompressed and ceil(w/4) * ceil(h/4) blocks of 8 (BC4) or 16 (BC5) bytes when it is not -
// which is why `pitch` is the BLOCK row's byte count for a block format and a texel row's for the others.

export const DXGI_R8_UNORM = 61;
export const DXGI_R8G8_UNORM = 49;
export const DXGI_R8G8B8A8_UNORM = 28;
export const DXGI_BC4_UNORM = 80;
export const DXGI_BC5_UNORM = 83;

// The names the descriptor and the viewers print. A format this reader does not accept never reaches here.
export function dxgiName(id) {
    switch (id) {
        case DXGI_R8_UNORM: return 'R8_UNORM';
        case DXGI_R8G8_UNORM: return 'R8G8_UNORM';
        case DXGI_R8G8B8A8_UNORM: return 'R8G8B8A8_UNORM';
        case DXGI_BC4_UNORM: return 'BC4_UNORM';
        case DXGI_BC5_UNORM: return 'BC5_UNORM';
        default: return 'DXGI ' + id;
    }
}

// The short names the panel's rows use, as viewer/main.cpp's fmt_name prints them.
export function dxgiShortName(id) {
    switch (id) {
        case DXGI_R8_UNORM: return 'R8';
        case DXGI_R8G8_UNORM: return 'R8G8';
        case DXGI_R8G8B8A8_UNORM: return 'RGBA8';
        case DXGI_BC4_UNORM: return 'BC4';
        case DXGI_BC5_UNORM: return 'BC5';
        default: return '?';
    }
}

// Reads `bytes` (a Uint8Array of the whole file) into a description of its levels, or throws an Error whose message is
// one line naming the file and the reason - the same sentences ../shared/dds.h prints.
export function ddsRead(path, bytes) {
    if (bytes.length < 148) throw new Error("cannot read '" + path + "' (or too short)");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const u32 = (i) => view.getUint32(i * 4, true);
    if (u32(0) !== 0x20534444 || u32(1) !== 124 || u32(19) !== 32 || u32(20) !== 0x4 || u32(21) !== 0x30315844) {
        throw new Error("'" + path + "' is not a DX10 .dds");
    }
    const H = view.getInt32(3 * 4, true), W = view.getInt32(4 * 4, true), nmip = view.getInt32(7 * 4, true);
    const dxgi = view.getInt32(32 * 4, true), dim = view.getInt32(33 * 4, true), arr = view.getInt32(35 * 4, true);
    if (dim !== 3 || arr !== 1) {
        throw new Error("'" + path + "': not a 2D texture (dimension " + dim + ", array size " + arr + ')');
    }
    // The three header fields a malformed file can turn into a crash rather than an error, bounded before anything is
    // sized from them - 16384 is Direct3D 11's own texture limit, and a chain cannot be longer than the base's log2.
    if (W < 1 || W > 16384 || H < 1 || H > 16384) {
        throw new Error("'" + path + "': " + W + 'x' + H + ' is not a size this reader accepts (1..16384)');
    }
    let maxMips = 1;
    for (let n = W > H ? W : H; n > 1; n >>= 1) maxMips++;
    if (nmip < 1 || nmip > 15 || nmip > maxMips) {
        throw new Error("'" + path + "': " + nmip + ' mip levels for a ' + W + 'x' + H + ' base (1..'
                        + (maxMips < 15 ? maxMips : 15) + ')');
    }
    const bc = dxgi === DXGI_BC4_UNORM || dxgi === DXGI_BC5_UNORM;
    const blockBytes = dxgi === DXGI_BC4_UNORM ? 8 : 16;
    const C = dxgi === DXGI_R8_UNORM ? 1 : (dxgi === DXGI_R8G8_UNORM ? 2 : (dxgi === DXGI_R8G8B8A8_UNORM ? 4
            : (dxgi === DXGI_BC4_UNORM ? 1 : (dxgi === DXGI_BC5_UNORM ? 2 : 0))));
    if (!C) {
        throw new Error("'" + path + "': DXGI format " + dxgi + ' is not R8 / R8G8 / R8G8B8A8 / BC4_UNORM / BC5_UNORM');
    }
    const levels = [];
    let off = 148, w = W, h = H;
    for (let i = 0; i < nmip; i++) {
        const bx = (w + 3) >> 2, by = (h + 3) >> 2;
        const n = bc ? bx * by * blockBytes : w * h * C;
        if (off + n > bytes.length) throw new Error("'" + path + "' is truncated at level " + i);
        levels.push({ w: w, h: h, offset: off, size: n, pitch: bc ? bx * blockBytes : w * C });
        off += n;
        w = w > 1 ? w >> 1 : 1;
        h = h > 1 ? h >> 1 : 1;
    }
    if (off !== bytes.length) {
        throw new Error("'" + path + "': " + (off - 148) + ' bytes of pixel data expected, '
                        + (bytes.length - 148) + ' present');
    }
    return {
        path: path, W: W, H: H, mips: nmip, dxgi: dxgi, channels: C, bc: bc,
        blockBytes: bc ? blockBytes : 0, levels: levels, bytes: bytes,
    };
}
