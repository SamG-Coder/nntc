// bcdec.js: BC4 and BC5 decoded in plain JavaScript, for a GPU whose WebGPU has no `texture-compression-bc` - phones,
// tablets, and anything else that says so - and, through ?unpack=1, on a GPU that does have it, which is what makes
// this path testable on a development machine (docs/WEBGPU_VIEWER_PLAN.md sections 3.3 and 8.4).
//
// It is a transliteration of the standard BC4 decode: two endpoint bytes and sixteen 3-bit selectors per 4x4 block,
// LSB-first and row-major, and the palette of the Direct3D functional specification, which is the palette the encoder
// fits against (../shared/bc_pack.h's header, ../docs/FORMAT.md section 7):
//
//     r0 > r1:  r0, r1, then ((8 - i) r0 + (i - 1) r1) / 7 for i = 2..7
//     r0 <= r1: r0, r1, then ((6 - i) r0 + (i - 1) r1) / 5 for i = 2..5, then 0 and 255
//
// WHAT "EXACT" MEANS HERE, because it is easy to expect too much. The interpolants are not whole numbers, so an 8-bit
// target carries the palette ROUNDED TO THE NEAREST BYTE - half a byte, at most, from the value the format specifies
// (../viewer/README.md's bc_check table reports 0.43 / 255 worst case over a real plane). And a real GPU is not exact
// either: this tree has measured one NVIDIA part decoding a losslessly packed plane up to 4 / 255 away from its own
// bytes, because it evaluates the palette interior with six-bit weights. So an unpacked frame and a hardware-BC frame
// of the same asset differ by a few counts on a few texels, and neither is wrong. What the unpack reproduces is the
// FORMAT'S SPECIFIED VALUES; what a vendor does is within the tolerance the API gives it.
//
// Note for a reader who compares this with ../viewer/bcdec.h: that decoder's 8-bit integer path TRUNCATES the
// interpolants ((6 * a0 + a1) / 7 in C integer arithmetic), which is up to six sevenths of a byte low. This one rounds
// to nearest, which is what ../viewer/README.md's bc_check column measures and what ../tools/dds_decode.py's float
// palette rounds to - so the two agree exactly, and that is the check the plan asks for (section 8.4).

// The eight palette entries of one BC4 block, rounded to the nearest byte, into `pal`.
function bc4Palette(r0, r1, pal) {
    pal[0] = r0;
    pal[1] = r1;
    if (r0 > r1) {
        for (let i = 2; i < 8; i++) pal[i] = Math.round(((8 - i) * r0 + (i - 1) * r1) / 7);
    } else {
        for (let i = 2; i < 6; i++) pal[i] = Math.round(((6 - i) * r0 + (i - 1) * r1) / 5);
        pal[6] = 0;
        pal[7] = 255;
    }
}

// One stored level of a BC4 or BC5 .dds, decoded to an uncompressed interleaved plane: `w * h * 1` bytes for a BC4 and
// `w * h * 2` for a BC5.
//
// A BC5 block IS two BC4 blocks in order - channel 0's eight bytes then channel 1's - so the two channels are the same
// walk over the same blocks, eight bytes apart, written into the two bytes of each output texel.
//
// The blocks cover ceil(w/4) x ceil(h/4); a texel of a partial edge block that falls outside the level is dropped,
// which is how a 6x6 level decodes from 2x2 blocks (8x8 texels) cropped to 6x6.
export function unpackLevel(bytes, offset, w, h, channels) {
    const out = new Uint8Array(w * h * channels);
    const bx = (w + 3) >> 2, by = (h + 3) >> 2;
    const blockBytes = channels === 1 ? 8 : 16;
    const pal = new Uint8Array(8);
    for (let c = 0; c < channels; c++) {
        for (let byi = 0; byi < by; byi++) {
            for (let bxi = 0; bxi < bx; bxi++) {
                const block = offset + (byi * bx + bxi) * blockBytes + c * 8;
                bc4Palette(bytes[block], bytes[block + 1], pal);
                // The 48-bit selector field as two reads: the low 32 bits and the high 16.
                const lo = (bytes[block + 2] | (bytes[block + 3] << 8) | (bytes[block + 4] << 16)
                            | (bytes[block + 5] << 24)) >>> 0;
                const hi = (bytes[block + 6] | (bytes[block + 7] << 8)) >>> 0;
                for (let t = 0; t < 16; t++) {
                    const x = bxi * 4 + (t & 3), y = byi * 4 + (t >> 2);
                    if (x >= w || y >= h) continue;
                    const bit = 3 * t;
                    let k;
                    if (bit + 3 <= 32) k = (lo >>> bit) & 7;
                    else if (bit >= 32) k = (hi >>> (bit - 32)) & 7;
                    else k = ((lo >>> bit) | (hi << (32 - bit))) & 7;
                    out[(y * w + x) * channels + c] = pal[k];
                }
            }
        }
    }
    return out;
}

// Every stored level of a block-compressed .dds (dds.js's description of it), decoded. The sampler will fetch every
// one, so every one is decoded; the result is one Uint8Array per level, tightly packed at w * h * channels.
export function unpackImage(img) {
    const out = [];
    for (const level of img.levels) {
        out.push(unpackLevel(img.bytes, level.offset, level.w, level.h, img.channels));
    }
    return out;
}
