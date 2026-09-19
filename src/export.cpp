// export.cpp: the shipped asset.
//
// PREFIX_lat1.dds is an uncompressed DX10 .dds holding level 1's base plane and its whole stored chain: R8_UNORM for one
// channel, R8G8_UNORM for two, R8G8B8A8_UNORM for three (alpha padded to 255) or four - DXGI has no 24-bit format. Level
// 0 is written the same way under --bc0 0 and is block-compressed by default (the BC block below). Each texel byte of an
// uncompressed plane holds the quantisation index k in the byte's top bits, replicated downward:
//
//     for (sh = 8 - bits; sh > -bits; sh -= bits) b8 |= sh >= 0 ? (k << sh) : (k >> -sh);
//
// so 3 bits give k k k, 4 bits give k * 17 and 8 bits give k itself. Index 0 maps to byte 0 and the top index to byte 255,
// and byte >> (8 - bits) recovers k losslessly, so a UNORM sampler can be handed the byte directly while an exact reader
// still has the index. What that byte MEANS is rep(k) / 255 of the range and not k / levels - the two agree only at 1, 2,
// 4 and 8 bits - and the grid the solver fitted is the first of those (model.h), so the palette this file publishes for
// an uncompressed level 0 is lo + rep(k) / 255 * (hi - lo) and nothing has to be rounded to read it.
//
// LEVEL 0 AS BC4 / BC5 (--bc0 1, the default). Level 0 is a full-resolution plane of 1-4 channels carrying 1-4 bits
// each, so a byte per channel wastes most of what it stores. BC4 holds one channel at half a byte per texel and BC5 two
// channels at one byte, and at 1-3 bits the pack is exact (bc_pack.h derives why), so the plane costs a quarter or a
// half of what it did and decodes to the same values. The channel count decides the files:
//
//     1 channel    PREFIX_lat0.dds                      BC4_UNORM  (DXGI 80),  W H / 2 bytes per level
//     2 channels   PREFIX_lat0.dds                      BC5_UNORM  (DXGI 83),  W H bytes per level
//     3 channels   PREFIX_lat0a.dds + PREFIX_lat0b.dds  BC5_UNORM (channels 0-1) + BC4_UNORM (channel 2)
//     4 channels   PREFIX_lat0a.dds + PREFIX_lat0b.dds  two BC5_UNORM, channels 0-1 and 2-3
//
// because BC5 is the widest block format that carries independent UNORM channels: three or four channels need two of
// them, which the JSON names in "files" and which the viewer binds to two samplers. Three channels put the odd one in
// a BC4 of its own, so level 0 costs 12 bits per texel rather than the 16 a padded fourth channel used to cost; the
// second file is then 8 bytes per block holding one channel, and each file's own format and channel count are carried
// in its "files" entry. Level 1 stays uncompressed: its 4-8 bits per channel would not survive a block palette.
//
// The .dds header of a compressed level differs from the uncompressed one in two fields, per the DDS specification:
// dwFlags carries DDSD_LINEARSIZE (0x00080000) instead of DDSD_PITCH (0x8), and dwPitchOrLinearSize is the base level's
// byte size rather than a row pitch. Everything else - the DX10 header, the caps, the levels base-first and tightly
// packed - is the same file.
//
// PREFIX_nntc.json carries the sizes, the dequantisation of both levels and the decoder. The descriptor takes the _nntc
// suffix so that a prefix derived from an input can never name the input itself; -o may name it outright instead, and
// write_asset is handed whichever name it is. The dequantisation is affine and is
// applied after sampling: value = lo + sample * (hi - lo), which is why the hardware blend of quantised texels is the
// blend of the values.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "backend.h"
#include "bc_pack.h"
#include "model.h"

// Level 1's dequantisation and its inverse are model.h's, the same two functions the device kernels call; these only
// pull the channel's lo/hi out of the model.
static float level1_dequantise(const Model& m, int c, int k)
{
    return level1_value(m.lo1[(size_t)c], m.hi1[(size_t)c], m.bits1, k);
}

static int level1_quantise(const Model& m, int c, float value)
{
    return level1_index(m.lo1[(size_t)c], m.hi1[(size_t)c], m.bits1, value);
}

// The plane on the device against the indices the .dds will store. After the grid is frozen every value the solver
// writes is a grid value, so this must hold exactly, bit for bit; a single disagreement means a value reached the plane
// by some path that does not go through the grid, and the asset would then decode to something the encoder never
// measured.
bool verify_level1_on_grid(const Model& m, const std::vector<std::vector<float>>& v1, size_t& values, size_t& offgrid)
{
    values = 0;
    offgrid = 0;
    for (size_t i = 0; i < v1.size() && i < m.k1.size(); i++)
        for (size_t t = 0; t < v1[i].size(); t++)
        {
            const int c = (int)(t % (size_t)m.c1);
            if (v1[i][t] != level1_dequantise(m, c, (int)m.k1[i][t]))
                offgrid++;
            values++;
        }
    return offgrid == 0;
}

// magic + DDS_HEADER (31 dwords) + DDS_HEADER_DXT10 (5 dwords) = 148 bytes, then the levels base-first, rows (or block
// rows) tightly packed, with no padding between levels. `dxgi` and `format_name` describe the format already chosen and
// `size_field` is the base level's row pitch for an uncompressed format or its byte size for a compressed one; the two
// are distinguished by the dwFlags bit the specification attaches to each.
static bool dds_write(const std::string& name, int w, int h, int dxgi, bool compressed, uint32_t size_field,
                      const std::vector<std::vector<uint8_t>>& levels, size_t& bytes)
{
    // The header is written as dwords straight out of memory, so the file is little-endian by construction; a big-endian
    // host would have to swap every one of them, and nothing here does.
    static_assert(sizeof(uint32_t) == 4, "the .dds header is 37 little-endian dwords");
    // A single stored level is not a mip chain, so it carries neither DDSD_MIPMAPCOUNT nor DDSCAPS_MIPMAP | COMPLEX -
    // DirectXTex's own rule, and what a reader that trusts those bits expects to see.
    const bool chain = levels.size() > 1;
    uint32_t hdr[37] = { 0 };
    hdr[0] = 0x20534444u;                                        // "DDS "
    hdr[1] = 124;                                                // dwSize
    // CAPS | HEIGHT | WIDTH | PIXELFORMAT, plus MIPMAPCOUNT (0x20000) for a chain, plus PITCH (0x8) or
    // LINEARSIZE (0x00080000)
    hdr[2] = 0x00001007u | (chain ? 0x00020000u : 0u) | (compressed ? 0x00080000u : 0x8u);
    hdr[3] = (uint32_t)h;
    hdr[4] = (uint32_t)w;
    hdr[5] = size_field;                                         // the BASE level's row pitch, or its byte size
    hdr[6] = 1;                                                  // dwDepth
    hdr[7] = (uint32_t)levels.size();                            // dwMipMapCount, counting the base
    hdr[19] = 32;                                                // ddspf dwSize
    hdr[20] = 0x4u;                                              // DDPF_FOURCC
    hdr[21] = 0x30315844u;                                       // "DX10"
    hdr[27] = chain ? 0x00401008u : 0x00001000u;                 // TEXTURE, plus MIPMAP | COMPLEX for a chain
    hdr[32] = (uint32_t)dxgi;
    hdr[33] = 3;                                                 // resourceDimension TEXTURE2D
    hdr[34] = 0;                                                 // miscFlag
    hdr[35] = 1;                                                 // arraySize
    hdr[36] = 0;                                                 // miscFlags2

    FILE* f = fopen(name.c_str(), "wb");
    if (!f)
    {
        fprintf(stderr, "ERROR: cannot write '%s'\n", name.c_str());
        return false;
    }
    // Every write and the close are checked: a full disk or a short write must be an ERROR and a failed run, not an
    // asset that looks written and is not (the owner's rule for every file this tree writes).
    bool ok = fwrite(hdr, 4, 37, f) == 37;
    bytes = 148;
    for (const std::vector<uint8_t>& level : levels)
    {
        ok = ok && fwrite(level.data(), 1, level.size(), f) == level.size();
        bytes += level.size();
    }
    if (fclose(f) != 0)
        ok = false;
    if (!ok)
    {
        fprintf(stderr, "ERROR: writing '%s' failed (a short write or a failed close: is the disk full?)\n", name.c_str());
        return false;
    }
    return true;
}

// Every number the JSON carries must be finite: a nan or an inf would be written as the token `nan` or `inf`, which is
// not JSON at all, and an asset whose decoder cannot be parsed is worse than no asset. write_asset checks the whole set
// before it writes anything (all_finite below) and refuses; this is the second line of defence, so that a number that
// reached here anyway cannot produce a file that looks valid.
static std::string json_float(double x)
{
    char b[32];
    if (!std::isfinite(x))
    {
        fprintf(stderr, "ERROR: a non-finite number (%g) reached the JSON writer\n", x);
        return "0";
    }
    snprintf(b, sizeof(b), "%.9g", x);
    return b;
}

static bool all_finite(const float* v, size_t n)
{
    for (size_t i = 0; i < n; i++)
        if (!std::isfinite(v[i]))
            return false;
    return true;
}

static std::string json_floats(const float* v, size_t n)
{
    std::string s = "[";
    for (size_t i = 0; i < n; i++)
    {
        s += json_float(v[i]);
        if (i + 1 < n)
            s += ",";
    }
    return s + "]";
}

static std::string basename_of(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// A written path as the `wrote` lines print it. The prefix comes from the command line and the suffixes are appended
// with a forward slash nowhere in sight, so one run could print `e7\name_lat0.dds` and `e7/name.json` - the same
// directory spelled two ways, because -o carried the separator the caller typed and the derived names carried none.
// Every path goes through here instead, so the three lines of a run agree on the platform's separator. It is main.cpp's
// tidy_path, which the option messages are written through, under a name of its own: the two files share no header for
// a two-line helper, and the rule - normalise, then the platform's separator - is the whole of it.
static std::string display_path(const std::string& path)
{
    return std::filesystem::path(path).lexically_normal().make_preferred().string();
}

// A JSON string. This writer concatenates its text rather than building a document, so a name or a type label carrying
// a quote, a backslash or a control character would produce a file no parser accepts - and those labels now come from
// a material written by hand, where a Windows path full of backslashes is the ordinary case and not the exotic one.
// Only what JSON requires escaping is escaped; everything else, UTF-8 included, is passed through as it stands.
static std::string json_string(const std::string& s)
{
    std::string out = "\"";
    for (char c : s)
    {
        const unsigned char u = (unsigned char)c;
        if (c == '"' || c == '\\')
        {
            out += '\\';
            out += c;
        }
        else if (u < 0x20)
        {
            char b[8];
            snprintf(b, sizeof(b), "\\u%04x", (unsigned)u);
            out += b;
        }
        else
            out += c;
    }
    return out + "\"";
}

// The PNG names: one texture needs no index, a material numbers its textures.
static std::string level_png_name(const std::string& prefix, const char* kind, int textures, int texture, int level)
{
    std::string s = prefix + "_" + kind;
    if (textures > 1)
        s += "_t" + std::to_string(texture);
    return s + "_M" + std::to_string(level) + ".png";
}

// The bytes one stored level of level 0 is written as: the index replicated over its channel's byte, three channels
// padded to four with 255 (which is what the uncompressed plane stores and what the pack then sees).
static void level0_bytes(const Model& m, int plane, int stored, std::vector<uint8_t>& out)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    const size_t texels = (size_t)p.w0 * p.h0;
    out.assign(texels * (size_t)stored, 255);
    for (size_t t = 0; t < texels; t++)
        for (int c = 0; c < m.c0; c++)
            out[t * (size_t)stored + c] =
                (uint8_t)replicate_index((int)m.k0[(size_t)plane][t * (size_t)m.c0 + c], m.bits0[(size_t)c]);
}

// How the level-0 channels are spread over .dds files (bc_pack.h's one rule): one BC4 for a single channel, one BC5
// for two, BC5 + BC4 for three and two BC5s for four. Wrapped here because the writer, the report and the refinement
// all reach for it through the model.
static int level0_files(const Model& m)
{
    return bc_level0_files(m.c0);
}

static int level0_file_channels(const Model& m, int file)
{
    return bc_level0_file_channels(m.c0, file);
}

// The exact index VALUE of one level-0 channel in byte units: the value the uncompressed plane would carry, which is
// what a pack's error is measured against. At the 8 bits of --l0 bc8 that is the index itself.
static double level0_exact_byte(const Model& m, int plane, size_t texel, int c)
{
    const int bits = m.bits0[(size_t)c];
    const int levels = (1 << bits) - 1;
    const int k = (int)m.k0[(size_t)plane][texel * (size_t)m.c0 + (size_t)c];
    return (double)k * 255.0 / (double)levels;
}

// The packing error of one file's blocks of one plane: the squared error in byte units against the exact index values,
// over the texels inside the plane and inside the layout's channels, added to se and n. bc_pack_level reports the same
// quantity for its own pack but averages a partial edge block's sixteen positions and scales by the count inside, so on
// a plane whose extent is not a multiple of four the two can differ in the last digit.
static void level0_blocks_error(const Model& m, int plane, const std::vector<uint8_t>& blocks, int file, double& se,
                                size_t& n)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    const int nc = level0_file_channels(m, file), bpb = nc == 1 ? 8 : 16, bx = (p.w0 + 3) / 4;
    for (int c = 0; c < nc; c++)
    {
        const int channel = 2 * file + c;
        for (int y = 0; y < p.h0; y++)
            for (int x = 0; x < p.w0; x++)
            {
                float decoded[16];
                bc4_decode_block(&blocks[((size_t)(y / 4) * bx + x / 4) * (size_t)bpb + (size_t)c * 8], decoded);
                const double e =
                    (double)decoded[(y % 4) * 4 + x % 4] - level0_exact_byte(m, plane, (size_t)y * p.w0 + x, channel);
                se += e * e;
                n++;
            }
    }
}

void level0_pack_seed(const Model& m, int plane, std::vector<uint8_t>& ep, std::vector<uint8_t>& sel)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    const int stored = m.c0 == 3 ? 4 : m.c0;
    const int files = level0_files(m);
    const int bx = (p.w0 + 3) / 4, by = (p.h0 + 3) / 4;
    std::vector<uint8_t> bytes;
    level0_bytes(m, plane, stored, bytes);
    int bits_of[4] = { 8, 8, 8, 8 };
    for (int c = 0; c < m.c0; c++)
        bits_of[c] = m.bits0[(size_t)c];
    ep.assign((size_t)bx * by * (size_t)m.c0 * 2, 0);
    sel.assign((size_t)bx * by * (size_t)m.c0 * 16, 0);
    for (int f = 0; f < files; f++)
    {
        double se = 0.0;
        size_t n = 0;
        const int nc = level0_file_channels(m, f), bpb = nc == 1 ? 8 : 16;
        const std::vector<uint8_t> blocks =
            bc_pack_level(bytes.data(), p.w0, p.h0, stored, 2 * f, nc, bits_of, m.c0, se, n);
        for (int c = 0; c < nc; c++)
        {
            const int channel = 2 * f + c;
            for (size_t b = 0; b < (size_t)bx * by; b++)
            {
                const uint8_t* blk = &blocks[b * (size_t)bpb + (size_t)c * 8];
                const size_t slot = b * (size_t)m.c0 + (size_t)channel;
                ep[slot * 2] = blk[0];
                ep[slot * 2 + 1] = blk[1];
                uint64_t bits = 0;
                for (int i = 0; i < 6; i++)
                    bits |= (uint64_t)blk[2 + i] << (8 * i);
                for (int t = 0; t < 16; t++)
                    sel[slot * 16 + (size_t)t] = (uint8_t)((bits >> (3 * t)) & 7);
            }
        }
    }
}

void level0_pack_blocks(const Model& m, int plane, const std::vector<uint8_t>& ep, const std::vector<uint8_t>& sel,
                        std::vector<std::vector<uint8_t>>& files_out, double& se, size_t& n)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    const int files = level0_files(m);
    const int bx = (p.w0 + 3) / 4, by = (p.h0 + 3) / 4;
    files_out.assign((size_t)files, std::vector<uint8_t>());
    for (int f = 0; f < files; f++)
    {
        // Every file holds only channels the layout uses - a three-channel level 0's second file is a BC4 of channel
        // 2 alone - so there is no padding channel to invent a flat block for any more.
        const int nc = level0_file_channels(m, f), bpb = nc == 1 ? 8 : 16;
        files_out[(size_t)f].assign((size_t)bx * by * (size_t)bpb, 0);
        for (int c = 0; c < nc; c++)
        {
            const int channel = 2 * f + c;
            for (size_t b = 0; b < (size_t)bx * by; b++)
            {
                uint8_t* blk = &files_out[(size_t)f][b * (size_t)bpb + (size_t)c * 8];
                const size_t slot = b * (size_t)m.c0 + (size_t)channel;
                int selectors[16];
                for (int t = 0; t < 16; t++)
                    selectors[t] = sel[slot * 16 + (size_t)t];
                bc4_pack(ep[slot * 2], ep[slot * 2 + 1], selectors, blk);
            }
        }
        level0_blocks_error(m, plane, files_out[(size_t)f], f, se, n);
    }
}

// One stored level of level 0 as the DEQUANTISED values a consumer's sampler returns, in the plane layout the device
// holds (texels * C0 floats). Two cases, and they are the two ways the plane can be shipped:
//
//   packed = false   the index straight off the palette of the format named by `bc` (model.h, level0_value);
//   packed = true    the plane packed to BC4 / BC5 exactly as write_asset packs it and decoded again by the standard
//                    palette, which is what the file holds once the pack is lossy (4 bits and up).
//
// It is what makes the report describe the disk rather than the plane the solver chose.
void level0_plane_values(const Model& m, int plane, bool bc, bool packed, std::vector<float>& v0, double* pack_se,
                         size_t* pack_n)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    const size_t texels = (size_t)p.w0 * p.h0;
    std::vector<std::vector<float>> pal;
    if (!m.l0_bc8)
        build_level0_palette(m.bits0, m.c0, bc, pal);
    v0.assign(texels * (size_t)m.c0, 0.0f);
    if (!packed)
    {
        for (size_t t = 0; t < texels; t++)
            for (int c = 0; c < m.c0; c++)
            {
                const int k = m.k0[(size_t)plane][t * (size_t)m.c0 + c];
                // Under --l0 bc8 there is no palette: the index is an 8-bit one on the channel's own lo/hi grid, and
                // the byte a sampler returns for it is the index itself, so the two rules coincide.
                v0[t * (size_t)m.c0 + c] = m.l0_bc8
                                               ? level1_value(m.lo0[(size_t)c], m.hi0[(size_t)c], 8, k)
                                               : pal[(size_t)c][(size_t)k];
            }
        return;
    }
    const int stored = m.c0 == 3 ? 4 : m.c0;
    const int files = level0_files(m);
    std::vector<uint8_t> bytes;
    level0_bytes(m, plane, stored, bytes);
    int bits_of[4] = { 8, 8, 8, 8 };
    for (int c = 0; c < m.c0; c++)
        bits_of[c] = m.bits0[(size_t)c];
    const int bx = (p.w0 + 3) / 4;
    // The refinement's blocks are the file's blocks once it has chosen any (refine_bc.cu), so the plane a report or a
    // decode asks for is decoded from THOSE and not from a second pack of the same bytes.
    const bool refined = (size_t)plane < m.bc0_blocks.size() && !m.bc0_blocks[(size_t)plane].empty();
    for (int f = 0; f < files; f++)
    {
        double se = 0.0;
        size_t n = 0;
        const int nc = level0_file_channels(m, f), bpb = nc == 1 ? 8 : 16;
        const std::vector<uint8_t> blocks =
            refined ? m.bc0_blocks[(size_t)plane][(size_t)f]
                    : bc_pack_level(bytes.data(), p.w0, p.h0, stored, 2 * f, nc, bits_of, m.c0, se, n);
        if (refined)
            level0_blocks_error(m, plane, blocks, f, se, n);
        if (pack_se)
            *pack_se += se;
        if (pack_n)
            *pack_n += n;
        for (int c = 0; c < nc; c++)
        {
            const int channel = 2 * f + c;
            for (int y = 0; y < p.h0; y++)
                for (int x = 0; x < p.w0; x++)
                {
                    float decoded[16];
                    bc4_decode_block(&blocks[((size_t)(y / 4) * bx + x / 4) * (size_t)bpb + (size_t)c * 8], decoded);
                    // The sampler's rule on the decoded byte: value = lo + byte / 255 * (hi - lo), with lo, hi the
                    // channel's own grid under --l0 bc8 and the fixed -1, 1 of the palette otherwise.
                    const double byte = (double)decoded[(y % 4) * 4 + x % 4];
                    const double lo = m.l0_bc8 ? (double)m.lo0[(size_t)channel] : -1.0;
                    const double hi = m.l0_bc8 ? (double)m.hi0[(size_t)channel] : 1.0;
                    v0[((size_t)y * p.w0 + x) * (size_t)m.c0 + channel] = (float)(lo + byte / 255.0 * (hi - lo));
                }
        }
    }
}

void be::reconstruct_levels(be::Device* d, const Model& m, std::vector<std::vector<uint8_t>>& recon)
{
    recon.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < m.planes.size(); i++)
        be::decode_plane(d, m, (int)i, recon[i]);
}

// THE ENCODER NEVER DELETES A FILE (the owner's rule). A run with a different layout may leave files of the same prefix
// behind - a deeper chain's _recon_M7.png, the single _lat0.dds of a two-file level 0 - and they stay where they are:
// the descriptor names the files that belong to the asset, and a reader that trusts the descriptor is never misled.
// Deleting by name pattern under the user's prefix could take a file that was never ours.

bool write_level_pngs(const Model& m, const std::vector<std::vector<uint8_t>>& recon,
                      const std::vector<Image>& source_chain, const std::string& prefix)
{
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const PlaneSize& p = m.planes[i];
        const int cw = i == 0 ? m.source_width : p.w0;
        const int ch = i == 0 ? m.source_height : p.h0;
        for (int t = 0; t < m.textures; t++)
        {
            if (!image_save_bytes(level_png_name(prefix, "recon", m.textures, t, (int)i), recon[i].data(), p.w0, p.h0,
                                  m.nout, t, cw, ch))
                return false;
            if (!image_save_texture(level_png_name(prefix, "src", m.textures, t, (int)i), source_chain[i], t, cw, ch))
                return false;
        }
    }
    return true;
}

bool write_asset(const Model& m, const std::string& prefix, const std::string& json_name, bool bc0,
                 size_t sizes[3], bool quiet)
{
    const int nplanes = (int)m.planes.size();
    size_t requantised = 0;
    // A nan or an inf in the decoder or in level 1's grid would be written as a token no JSON parser accepts, so the
    // asset would be unreadable rather than wrong. The encoder refuses to write it at all.
    if (!all_finite(m.dec.w.data(), m.dec.w.size()) || !all_finite(m.dec.b.data(), m.dec.b.size()) ||
        !all_finite(m.lo1.data(), m.lo1.size()) || !all_finite(m.hi1.data(), m.hi1.size()) ||
        !all_finite(m.lo0.data(), m.lo0.size()) || !all_finite(m.hi0.data(), m.hi0.size()))
    {
        fprintf(stderr, "ERROR: the decoder or a latent's grid carries a value that is not finite; the asset is not "
                        "written\n");
        return false;
    }
    // The palette of the format THIS call writes, which is not necessarily the one the solver fitted: --bc0 both writes
    // the same solve twice, and the uncompressed twin's bytes mean lo + rep(k) / 255 * (hi - lo) where the
    // block-compressed one's mean lo + k / (2^bits - 1) * (hi - lo). Each file publishes the palette it decodes to.
    std::vector<std::vector<float>> pal0;
    build_level0_palette(m.bits0, m.c0, bc0, pal0);
    std::string js = "{\n  \"format\": \"nntc-dds-1\",\n";
    js += "  \"source\": { \"width\": " + std::to_string(m.source_width) + ", \"height\": " +
          std::to_string(m.source_height);
    if (m.padded)
        js += ", \"padded\": true, \"padded_width\": " + std::to_string(m.width) + ", \"padded_height\": " +
              std::to_string(m.height);
    // WHAT THE INPUTS WERE. Informational, and named `inputs` rather than `textures` because `textures` is the two
    // latent levels everywhere else in this format. A consumer cannot recover from the two latent textures whether an
    // input was a tangent-space normal map, or whether its deeper levels were derived in linear light, and both change
    // what the asset means; entry t describes output channels 3t..3t+2.
    if (!m.inputs.empty())
    {
        js += ", \"inputs\": [";
        for (size_t t = 0; t < m.inputs.size(); t++)
        {
            const TextureSettings& s = m.inputs[t];
            js += std::string(t ? ", " : "") + "{ \"file\": " + json_string(basename_of(s.file)) + ", \"type\": " +
                  json_string(s.type) + ", \"filter\": " + json_string(s.filter) + ", \"srgb\": " +
                  (s.srgb ? "true" : "false") + ", \"edge\": " + json_string(s.edge) + ", \"normal_map\": " +
                  (s.normal_map ? "true" : "false") + ", \"weight\": " + json_float(s.weight) + ", \"rgb_weights\": [" +
                  json_float(s.rgb_weights[0]) + "," + json_float(s.rgb_weights[1]) + "," +
                  json_float(s.rgb_weights[2]) + "] }";
        }
        js += "]";
    }
    js += " },\n";
    js += "  \"decode\": { \"width\": " + std::to_string(m.width) + ", \"height\": " + std::to_string(m.height) +
          ", \"textures_out\": " + std::to_string(m.textures) + " },\n";
    js += "  \"block\": " + std::to_string(BLOCK) + ",\n";
    js += "  \"lod_bias_level1\": " + std::to_string(LOD_BIAS_LEVEL1) + ",\n";
    js += "  \"sampling\": \"both latent textures are read through the hardware filter at the same coordinate; at a base "
          "pixel centre level 0's filtered sample is that one texel (it is at the decoded extent) and level 1 blends "
          "four, and between centres both are filtered, which is what the encoder fits; output mip m reads mip m of "
          "BOTH textures (the 1:1 rule), and level 1 has "
          "1/block of the texels per axis, so its sampler needs MipLODBias = log2(block) = the value in lod_bias_level1, "
          "else the GPU reads it log2(block) mips finer than fitted and the colour goes splotchy from mip 1 down; "
          "dequantise AFTER sampling: value = lo + sample * (hi - lo) per channel, which is the grid the encoder fitted "
          "(the byte is the quantisation index in the top bits, replicated, and the exact index is byte >> (8 - bits); "
          "a block-compressed level 0 decodes index k to k / (2 ^ bits - 1) of full scale instead, which is what its own "
          "palette array carries)\",\n";
    js += "  \"textures\": [\n";

    for (int l = 0; l < 2; l++)
    {
        const int channels = l == 0 ? m.c0 : m.c1;
        const int stored = channels == 3 ? 4 : channels;
        std::vector<std::vector<uint8_t>> levels((size_t)nplanes);
        std::string sizes_js = "[";
        for (int i = 0; i < nplanes; i++)
        {
            const PlaneSize& p = m.planes[i];
            const int w = l == 0 ? p.w0 : p.w1, h = l == 0 ? p.h0 : p.h1;
            const std::vector<uint8_t>& k = l == 0 ? m.k0[i] : m.k1[i];
            levels[i].assign((size_t)w * h * stored, 255);
            for (size_t t = 0; t < (size_t)w * h; t++)
                for (int c = 0; c < channels; c++)
                {
                    const int bits = l == 0 ? m.bits0[c] : m.bits1;
                    const int index = (int)k[t * channels + c];
                    // The zero-change assertion: the byte carries the index, and the index dequantised and quantised
                    // again by the grid this same JSON publishes must come back as itself. A level-1 index that fails
                    // it would decode to a value the solver never chose - a re-encode of the asset would move - and the
                    // encoder refuses to write the file rather than ship it.
                    if (l == 1)
                    {
                        if (level1_quantise(m, c, level1_dequantise(m, c, index)) != index)
                        {
                            fprintf(stderr, "ERROR: level-1 index %d of channel %d does not re-quantise to itself on "
                                            "the grid [%g, %g]\n",
                                    index, c, (double)m.lo1[(size_t)c], (double)m.hi1[(size_t)c]);
                            return false;
                        }
                        requantised++;
                    }
                    levels[i][t * stored + c] = (uint8_t)replicate_index(index, bits);
                }
            sizes_js += "[" + std::to_string(w) + "," + std::to_string(h) + "]";
            if (i + 1 < nplanes)
                sizes_js += ",";
        }
        sizes_js += "]";

        const PlaneSize& base = m.planes[0];
        const int bw = l == 0 ? base.w0 : base.w1, bh = l == 0 ? base.h0 : base.h1;
        const bool compress = l == 0 && bc0;
        std::string format_name, files_js, texel_js;
        int dxgi = 0, stored_out = 0;

        if (compress)
        {
            // One BC4 for a single channel, one BC5 for two, BC5 + BC4 for three and two BC5s for four (bc_pack.h).
            // bits_of[] is indexed by the absolute channel; every file holds channels the layout uses and nothing
            // else, so there is no padding channel anywhere in the compressed path. The entry's own dxgi_format /
            // channels_stored describe the FIRST file, and each file's are carried beside its name in "files".
            const int files = level0_files(m);
            int bits_of[4] = { 8, 8, 8, 8 };
            for (int c = 0; c < m.c0; c++)
                bits_of[c] = m.bits0[c];
            dxgi = level0_file_channels(m, 0) == 1 ? 80 : 83;
            format_name = level0_file_channels(m, 0) == 1 ? "BC4_UNORM" : "BC5_UNORM";
            stored_out = level0_file_channels(m, 0);

            // The blocks the refinement chose ARE the file (refine_bc.cu): they were measured through the decoder and
            // the asset must hold exactly what was measured, so they are written as they stand and only their error is
            // recomputed for the report. Without them - the palette mode, or a bc8 run with the refinement off - the
            // writer packs the plane itself, which is the same pack the refinement would have started from.
            const bool refined = !m.bc0_blocks.empty();
            double se = 0.0;
            size_t packed_texels = 0;
            size_t total = 0;
            for (int f = 0; f < files; f++)
            {
                const int nc = level0_file_channels(m, f);
                const int f_dxgi = nc == 1 ? 80 : 83;
                const std::string f_name = nc == 1 ? "BC4_UNORM" : "BC5_UNORM";
                std::vector<std::vector<uint8_t>> blocks((size_t)nplanes);
                for (int i = 0; i < nplanes; i++)
                {
                    blocks[i] = refined ? m.bc0_blocks[(size_t)i][(size_t)f]
                                        : bc_pack_level(levels[i].data(), m.planes[i].w0, m.planes[i].h0, stored,
                                                        2 * f, nc, bits_of, m.c0, se, packed_texels);
                    if (refined)
                        level0_blocks_error(m, i, blocks[i], f, se, packed_texels);
                }
                const std::string file = prefix + "_lat0" + (files == 2 ? (f == 0 ? "a" : "b") : "") + ".dds";
                const uint32_t linear = (uint32_t)blocks[0].size();   // the base level's byte size, per the DDS spec
                size_t bytes = 0;
                if (!dds_write(file, bw, bh, f_dxgi, true, linear, blocks, bytes))
                    return false;
                total += bytes;
                if (!quiet) printf("wrote %s: %s %dx%d, %d mip level%s, channel%s %d%s, %zu bytes\n", display_path(file).c_str(),
                       f_name.c_str(), bw, bh, nplanes, nplanes == 1 ? "" : "s", nc == 1 ? "" : "s", 2 * f,
                       nc == 1 ? "" : (" and " + std::to_string(2 * f + 1)).c_str(), bytes);
                // Two files carry an OBJECT each, because they no longer have to share a format: a reader has to know
                // that the second one is a BC4 of one channel without inferring it from the channel count. One file is
                // a bare name, which is what "file" has always carried. Both branches APPEND, so neither of them
                // depends on the loop around them running exactly once.
                if (files == 2)
                    files_js += std::string(f ? ", " : "") + "{ \"file\": \"" + basename_of(file) +
                                "\", \"dxgi_format\": \"" + f_name + "\", \"dxgi_format_id\": " +
                                std::to_string(f_dxgi) + ", \"channels_stored\": " + std::to_string(nc) + " }";
                else
                    files_js += "\"" + basename_of(file) + "\"";
            }
            sizes[0] = total;

            // The packing report, against the exact index values the uncompressed plane would carry.
            bool lossless = true;
            int max_bits = 0;
            for (int c = 0; c < m.c0; c++)
            {
                max_bits = m.bits0[c] > max_bits ? m.bits0[c] : max_bits;
                if (m.bits0[c] > 3)
                    lossless = false;
            }
            if (lossless && !quiet)
                printf("level 0 pack: lossless over %zu channel-texels of the chain (the endpoints are the mode's own "
                       "0 and 255, the index is the selector), packing error %g\n", packed_texels, se);
            else if (!quiet)
                printf("level 0 pack: lossy at %d bits, packing psnr %.2f dB over %zu channel-texels of the chain "
                       "(against the exact index values; a block carries eight of the %d values)\n", max_bits,
                       se > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / (se / (double)packed_texels)) : 100.0,
                       packed_texels, 1 << max_bits);   // 100 dB for an exact pack, as every other psnr here
            if (!lossless && !quiet)
                printf("              the encoder packs and decodes the plane before it measures anything, so the "
                       "report above is of these same blocks\n");

            texel_js = m.l0_bc8
                           ? std::string("an 8-bit continuous value, BC4 / BC5 encoded: one BC4 block per 4x4 texels "
                                         "of each channel (two endpoint bytes and sixteen 3-bit selectors packed "
                                         "LSB-first, row-major; a BC5 block is the two channels' BC4 blocks in "
                                         "order), the standard k / 7 palette of the Direct3D functional "
                                         "specification. The encoder solved this plane as a continuous 8-bit plane, "
                                         "encoded it as BC and then went on optimising it AS BC - every block's "
                                         "endpoints and selectors re-chosen against the decoder's own output error, "
                                         "the model refitted and the plane repacked in outer passes - so these blocks "
                                         "are the representation that was measured. A block carries eight fitted "
                                         "values along a line and there is no quantisation index to recover: "
                                         "dequantise the sampled byte, which is what the rule below says")
                           : std::string("one BC4 block per 4x4 texels of each channel (two endpoint bytes and "
                                         "sixteen 3-bit selectors packed LSB-first, row-major; a BC5 block is the "
                                         "two channels' BC4 blocks in order), the standard k / 7 palette of the "
                                         "Direct3D functional specification. At 1-3 bits the endpoints are the "
                                         "mode's own extremes and the selector is a relabelling of the quantisation "
                                         "index, so the decoded byte is exactly k * 255 / (2 ^ bits - 1) and "
                                         "byte >> (8 - bits) recovers k; at 4 bits and more a block holds eight of "
                                         "the values and the pack is lossy");
            if (files == 2)
                texel_js += m.c0 == 3 ? ". Channels 0-1 are in the first file, a BC5, and channel 2 alone in the "
                                        "second, a BC4 of 8 bytes per block"
                                      : ". Channels 0-1 are in the first file and 2-3 in the second, each a BC5";
        }
        else
        {
            stored_out = stored;   // the same rule the byte planes above were built with: three channels stored as four
            dxgi = channels == 1 ? 61 : (channels == 2 ? 49 : 28);
            format_name = channels == 1 ? "R8_UNORM" : (channels == 2 ? "R8G8_UNORM" : "R8G8B8A8_UNORM");
            const std::string file = prefix + "_lat" + std::to_string(l) + ".dds";
            size_t bytes = 0;
            if (!dds_write(file, bw, bh, dxgi, false, (uint32_t)(bw * stored_out), levels, bytes))
                return false;
            sizes[l] = bytes;
            if (!quiet) printf("wrote %s: %s %dx%d, %d mip level%s, %d channel%s of %d stored, %zu bytes\n", display_path(file).c_str(),
                   format_name.c_str(), bw, bh, nplanes, nplanes == 1 ? "" : "s", channels, channels == 1 ? "" : "s",
                   stored_out, bytes);
            files_js = "\"" + basename_of(file) + "\"";
            texel_js = m.l0_bc8 && l == 0
                           ? std::string("an 8-bit continuous value: the byte IS the index, and this plane is the one "
                                         "the solver held before the BC pack")
                           : std::string("the quantisation index k in the byte's top bits, replicated into the low "
                                         "bits (byte = k << (8 - bits) | k >> (2 bits - 8) | ...); recover k "
                                         "losslessly with byte >> (8 - bits), or dequantise the sampled byte "
                                         "directly - the palette of this plane IS lo + byte / 255 * (hi - lo), "
                                         "because that is what the encoder fitted");
        }

        // One file is named by "file" and two by "files"; a consumer that reads only "file" therefore cannot silently
        // decode half of a two-file level 0.
        const bool two_files = compress && m.c0 >= 3;
        js += "    {\n      " + std::string(two_files ? "\"files\": [" + files_js + "]" : "\"file\": " + files_js) +
              ", \"level\": " + std::to_string(l) + ", \"width\": " + std::to_string(bw) + ", \"height\": " +
              std::to_string(bh) + ",\n";
        js += "      \"dxgi_format\": \"" + format_name + "\", \"dxgi_format_id\": " + std::to_string(dxgi) +
              ", \"channels_used\": " + std::to_string(channels) + ", \"channels_stored\": " +
              std::to_string(stored_out) + ",\n";
        if (compress)
            js += "      \"bc_palette\": \"standard\",\n";
        js += "      \"bits_per_channel\": [";
        for (int c = 0; c < channels; c++)
            js += std::to_string(l == 0 ? m.bits0[c] : m.bits1) + (c + 1 < channels ? "," : "");
        js += "],\n      \"mip_count\": " + std::to_string(nplanes) + ", \"mip_sizes\": " + sizes_js + ",\n";
        js += "      \"texel\": \"" + texel_js + "\",\n";
        js += "      \"dequantise\": {";
        if (l == 0 && m.l0_bc8)
        {
            // --l0 bc8: level 0 carries no palette at all. It is a continuous plane at 8 bits on a per-channel range,
            // exactly as level 1 is, and the rule is the same one on the byte the sampler returns - which is the
            // BC-decoded byte of the block the pack chose, and not an index of anything.
            js += " \"kind\": \"range\", \"note\": \"value = lo + byte / 255 * (hi - lo) per channel, the byte as the "
                  "sampler returns it. This level was solved as a continuous 8-bit plane, BC4 / BC5-encoded, and the "
                  "pack then refined block by block against the decoder's own output error and repacked, so the byte "
                  "is a block palette entry the encoder chose and not a quantisation index; at 8 bits the index rule "
                  "lo + k / levels * (hi - lo) is the same number anyway\", \"lo\": " +
                  json_floats(m.lo0.data(), (size_t)channels) + ", \"hi\": " +
                  json_floats(m.hi0.data(), (size_t)channels) + ", \"levels\": " + std::to_string(m.levels0()) +
                  " }\n";
        }
        else if (l == 0)
        {
            // The palette is the value of each index UNDER THIS FILE'S FORMAT, so palette[k] is exactly what a sampler
            // returns dequantised, whichever of the two rules that format follows. Both statements below are
            // equalities and not approximations, which is the whole point of publishing the array.
            js += std::string(" \"kind\": \"palette\", \"note\": \"value = palette[k] with k = byte >> (8 - bits); ") +
                  (compress ? "the standard BC palette decodes index k to k / (2 ^ bits - 1) of full scale, so "
                              "lo + sample * (hi - lo) on the sampled float is the same number"
                            : "the byte is the index replicated, so palette[k] IS lo + byte / 255 * (hi - lo) and a "
                              "sampler that dequantises the sampled float lands on the same number") +
                  "\", \"palette\": [";
            for (int c = 0; c < channels; c++)
                js += json_floats(pal0[c].data(), pal0[c].size()) + (c + 1 < channels ? "," : "");
            js += "], \"lo\": [";
            for (int c = 0; c < channels; c++)
                js += json_float(pal0[c].front()) + (c + 1 < channels ? "," : "");
            js += "], \"hi\": [";
            for (int c = 0; c < channels; c++)
                js += json_float(pal0[c].back()) + (c + 1 < channels ? "," : "");
            js += "], \"levels\": [";
            for (int c = 0; c < channels; c++)
                js += std::to_string((int)pal0[c].size() - 1) + (c + 1 < channels ? "," : "");
            js += "] }\n";
        }
        else
        {
            // Level 1 is always an uncompressed UNORM8 plane, so its grid is the replicated byte over 255: at 4 and 8
            // bits that is k / levels exactly, and at 5, 6 and 7 bits it is not, which is why the rule is written on
            // the byte and `levels` is carried as the index's own range.
            js += " \"kind\": \"range\", \"note\": \"value = lo + byte / 255 * (hi - lo) per channel, the byte as the "
                  "sampler returns it; the byte is the index k = byte >> (8 - bits) replicated over the byte, which is "
                  "lo + k / levels * (hi - lo) exactly at 4 and 8 bits and within half a byte of it at 5, 6 and 7\", "
                  "\"lo\": " +
                  json_floats(m.lo1.data(), (size_t)channels) + ", \"hi\": " +
                  json_floats(m.hi1.data(), (size_t)channels) + ", \"levels\": " + std::to_string(m.levels1()) + " }\n";
        }
        js += "    }" + std::string(l == 0 ? "," : "") + "\n";
    }


    js += "  ],\n  \"decoder\": {\n";
    js += "    \"type\": \"bilinear\",\n";
    js += "    \"terms\": \"a b sc\", \"C0\": " + std::to_string(m.c0) + ", \"C1\": " + std::to_string(m.c1) + ",\n";
    js += "    \"features\": \"phi(z) in this order: [a] c_j for j < C1 (the level-1 channels), [b] s_i for i < C0 (the "
          "level-0 channels), [sc] s_i * c_j for i < C0, j < C1 (i outer, j inner); the output is W phi + b\",\n";
    js += "    \"nin\": " + std::to_string(m.dec.nin) + ", \"nout\": " + std::to_string(m.nout) +
          ", \"hidden\": [], \"activation\": \"leaky_relu\", \"leak\": 0,\n";
    js += "    \"output\": \"identity\",\n";
    js += "    \"layers\": [\n";
    js += "      { \"rows\": " + std::to_string(m.nout) + ", \"cols\": " + std::to_string(m.dec.nin) +
          ", \"weights\": " + json_floats(m.dec.w.data(), m.dec.w.size()) + ", \"bias\": " +
          json_floats(m.dec.b.data(), m.dec.b.size()) + " }\n";
    js += "    ]\n  }\n}\n";

    const std::string jname = json_name;
    FILE* jf = fopen(jname.c_str(), "wb");
    if (!jf)
    {
        fprintf(stderr, "ERROR: cannot write '%s'\n", jname.c_str());
        return false;
    }
    const bool jok = fwrite(js.data(), 1, js.size(), jf) == js.size();
    if (fclose(jf) != 0 || !jok)
    {
        fprintf(stderr, "ERROR: writing '%s' failed (a short write or a failed close: is the disk full?)\n", jname.c_str());
        return false;
    }
    sizes[2] = js.size();
    if (!quiet) printf("wrote %s: the sizes, the bits per channel, the dequantisation of both levels and the decoder's one layer\n",
           display_path(jname).c_str());
    // The grid line is progress - a check that ran and passed - and not part of the report, so --quiet drops it as it
    // drops every other running line.
    if (!quiet)
        printf("level 1 grid: %zu stored indices, every one re-quantises to itself under the published lo/hi\n", requantised);
    return true;
}
