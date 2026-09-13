// bc_pack.h: the BC4 / BC5 packing of the level-0 latent, shared by the encoder, the viewer and bc_check.
//
// A BC4 block is 8 bytes covering 4x4 texels of ONE channel: two 8-bit endpoints red_0, red_1, then sixteen 3-bit
// selectors packed LSB-first (texel 0 in the lowest bits), row-major. The selector indexes an eight-value palette
// which the decoder evaluates from the endpoints:
//
//     red_0 >  red_1   p = [r0, r1, (6 r0 + r1) / 7, (5 r0 + 2 r1) / 7, ... , (r0 + 6 r1) / 7]
//     red_0 <= red_1   p = [r0, r1, (4 r0 + r1) / 5, (3 r0 + 2 r1) / 5, (2 r0 + 3 r1) / 5, (r0 + 4 r1) / 5, 0, 255]
//
// A BC5 block is 16 bytes: two BC4 blocks, the first the red channel and the second the green.
//
// THE PALETTE IS THE STANDARD ONE, k / 7. That is what the Direct3D functional specification defines, what a software
// decoder computes and what every vendor's hardware is required to return. (Some hardware evaluates the interior of
// the eight-value palette with six-bit weights, n / 64, which lands within half a byte of k / 7; fitting against that
// instead would be a vendor-specific mode, and the asset must be right everywhere, so it is not what this file does.)
//
// WHY 1-3 BITS ARE LOSSLESS. A level-0 texel byte holds its quantisation index k with the index's bits replicated over
// the byte, so the exact value of index k is k / (2^bits - 1) of full scale. With the endpoints 255 and 0 the
// eight-value palette IS the 3-bit grid: (8 - i) * 255 / 7 for i = 1..7 together with 255 gives 255, 219, 182, 146,
// 109, 73, 36, 0, which is k * 255 / 7 for k = 7..0. So a 3-bit plane packs by relabelling k as a selector and nothing
// is lost. Two bits use the six-value mode, whose palette carries 0, 85, 170 and 255 = k * 255 / 3, with the endpoints
// 85 and 170; one bit uses that mode's fixed 0 and 255.
//
// FOUR BITS AND MORE CANNOT BE EXACT: sixteen or more distinct values do not fit in one block's eight palette entries.
// The block's lowest and highest index values become the endpoints of the eight-value mode and each texel takes the
// nearest palette entry, which is what stb_dxt and most BC4 encoders do; the error is reported by the caller as a
// packing PSNR against the exact index values.
//
// The decode here is validated against iOrange's bcdec by bc_check.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// HOW THE CHANNELS ARE SPREAD OVER FILES. BC5 is the widest block format that carries independent UNORM channels, so a
// level-0 plane of more than two channels needs two of them. The split is by channel count and nothing else:
//
//     1 channel    one BC4                 4 bits per texel
//     2 channels   one BC5                 8
//     3 channels   one BC5 + one BC4      12   (channels 0-1, then channel 2 alone)
//     4 channels   two BC5s               16
//
// Three channels used to be two BC5s with a padding fourth, which spent a quarter of the plane's bytes on a constant
// nobody reads; a BC4 second file holds exactly the one channel that is left, at 12 bits per texel instead of 16. The
// rule lives here because the encoder's writer, its pack and its refinement, the viewer, bc_check and the Python
// reader all have to agree on it.
inline int bc_level0_files(int channels)
{
    return channels <= 2 ? 1 : 2;
}

inline int bc_level0_file_channels(int channels, int file)
{
    return file == 0 ? (channels == 1 ? 1 : 2) : channels - 2;
}

// The palette in float, exactly as a decoder evaluates it from the two endpoints.
inline void bc4_palette(int r0, int r1, float p[8])
{
    p[0] = (float)r0;
    p[1] = (float)r1;
    if (r0 > r1)
    {
        for (int i = 2; i < 8; i++)
            p[i] = (float)((8 - i) * r0 + (i - 1) * r1) / 7.0f;
    }
    else
    {
        for (int i = 2; i < 6; i++)
            p[i] = (float)((6 - i) * r0 + (i - 1) * r1) / 5.0f;
        p[6] = 0.0f;
        p[7] = 255.0f;
    }
}

// The endpoints and the sixteen selectors into the block's eight bytes.
inline void bc4_pack(int r0, int r1, const int sel[16], uint8_t out[8])
{
    out[0] = (uint8_t)r0;
    out[1] = (uint8_t)r1;
    uint64_t bits = 0;
    for (int t = 0; t < 16; t++)
        bits |= (uint64_t)(sel[t] & 7) << (3 * t);
    for (int i = 0; i < 6; i++)
        out[2 + i] = (uint8_t)(bits >> (8 * i));
}

// One 4x4 block of one channel, row-major, the caller having clamped the edges: sixteen bytes each holding an index of
// `bits` bits with its bits replicated over the byte. Returns the squared error in byte units against the exact index
// values, which is zero at 1-3 bits by the relabelling above.
inline double bc4_encode_block(const uint8_t v[16], int bits, uint8_t out[8])
{
    int sel[16];
    if (bits <= 3)
    {
        int r0 = 0, r1 = 0;
        if (bits == 3)
        {
            // k / 7 of full scale is palette entry 8 - k for k = 1..6, entry 0 for k = 7 and entry 1 for k = 0.
            r0 = 255;
            r1 = 0;
            for (int t = 0; t < 16; t++)
            {
                const int k = v[t] >> 5;
                sel[t] = k == 7 ? 0 : (k == 0 ? 1 : 8 - k);
            }
        }
        else if (bits == 2)
        {
            // The six-value mode with the endpoints 85 and 170: its palette holds 85, 170, 102, 119, 136, 153, 0, 255,
            // of which 0, 85, 170 and 255 are the 2-bit grid k * 255 / 3.
            r0 = 85;
            r1 = 170;
            for (int t = 0; t < 16; t++)
            {
                const int k = v[t] >> 6;
                sel[t] = k == 0 ? 6 : (k == 1 ? 0 : (k == 2 ? 1 : 7));
            }
        }
        else
        {
            // One bit: the six-value mode's own fixed 0 and 255.
            r0 = 0;
            r1 = 255;
            for (int t = 0; t < 16; t++)
                sel[t] = (v[t] >> 7) ? 7 : 6;
        }
        bc4_pack(r0, r1, sel, out);
        return 0.0;
    }

    const int levels = (1 << bits) - 1;
    int kmin = levels, kmax = 0;
    for (int t = 0; t < 16; t++)
    {
        const int k = v[t] >> (8 - bits);
        kmin = std::min(kmin, k);
        kmax = std::max(kmax, k);
    }
    // The high endpoint first, which selects the eight-value mode.
    const int r0 = (int)std::lround(kmax * 255.0 / levels);
    const int r1 = (int)std::lround(kmin * 255.0 / levels);
    if (r0 == r1)
    {
        for (int t = 0; t < 16; t++)
            sel[t] = 0;
        bc4_pack(r0, r1, sel, out);
        return 0.0;   // a flat block: palette entry 0 is the value itself
    }
    float p[8];
    bc4_palette(r0, r1, p);
    double err = 0.0;
    for (int t = 0; t < 16; t++)
    {
        const float target = (float)(v[t] >> (8 - bits)) * 255.0f / (float)levels;
        int best = 0;
        float best_d = 1e30f;
        for (int i = 0; i < 8; i++)
        {
            const float d = std::fabs(p[i] - target);
            if (d < best_d)
            {
                best_d = d;
                best = i;
            }
        }
        sel[t] = best;
        err += (double)best_d * (double)best_d;
    }
    bc4_pack(r0, r1, sel, out);
    return err;
}

// The decode of one BC4 block into sixteen float values, the palette evaluated as a decoder evaluates it.
inline void bc4_decode_block(const uint8_t blk[8], float v[16])
{
    float p[8];
    bc4_palette(blk[0], blk[1], p);
    uint64_t bits = 0;
    for (int i = 0; i < 6; i++)
        bits |= (uint64_t)blk[2 + i] << (8 * i);
    for (int t = 0; t < 16; t++)
        v[t] = p[(bits >> (3 * t)) & 7];
}

// One level of W x H texels held as `stored_C` interleaved bytes per texel: the channels c0 .. c0 + nc - 1 (nc = 1 gives
// BC4, nc = 2 gives BC5) at bits[c] per channel, into block rows of (W + 3) / 4 blocks with 8 or 16 bytes each. The
// squared error and the texel count accumulate over the texels that are inside the level AND inside the `used` channels
// of the layout, so neither a partial edge block's padding nor the padding fourth channel of a three-channel level 0
// weighs into the caller's packing PSNR - the fourth channel is a flat block of 255s and would otherwise report a
// quarter of the plane as perfect.
inline std::vector<uint8_t> bc_pack_level(const uint8_t* src, int W, int H, int stored_C, int c0, int nc,
                                          const int* bits, int used, double& se, size_t& n)
{
    const int bx = (W + 3) / 4, by = (H + 3) / 4, bpb = nc == 1 ? 8 : 16;
    std::vector<uint8_t> out((size_t)bx * by * bpb);
    for (int y = 0; y < by; y++)
        for (int x = 0; x < bx; x++)
            for (int c = 0; c < nc; c++)
            {
                uint8_t v[16];
                for (int t = 0; t < 16; t++)
                {
                    const int px = std::min(W - 1, x * 4 + t % 4), py = std::min(H - 1, y * 4 + t / 4);
                    v[t] = src[((size_t)py * W + px) * stored_C + c0 + c];
                }
                uint8_t* blk = &out[((size_t)y * bx + x) * bpb + c * 8];
                const double e = bc4_encode_block(v, bits[c0 + c], blk);
                int inside = 0;
                for (int t = 0; t < 16; t++)
                    if (x * 4 + t % 4 < W && y * 4 + t / 4 < H)
                        inside++;
                if (c0 + c < used)
                {
                    se += e * inside / 16.0;
                    n += (size_t)inside;
                }
            }
    return out;
}
