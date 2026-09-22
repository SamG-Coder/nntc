// dds.h: the DX10 .dds reader the encoder's assets are read back with (docs/FORMAT.md section 2), shared by the
// viewers.
//
// It is the parsing half of what viewer/main.cpp's load_dds has always done, lifted out unchanged when the Vulkan
// viewer arrived and needed the same bytes: the 148-byte header and its magic, the DX10 block, the extent and mip
// bounds a malformed file could otherwise turn into a crash, the format's channel count, and the per-level walk that
// says where each mip's bytes start, how many there are and what a row of them costs. What it deliberately does NOT do
// is touch a graphics api: the caller turns the levels into a texture with whatever it has, which is the whole reason
// two viewers can share this file.
//
// The levels are base-first and tightly packed, so a level is w * h * C bytes when the format is uncompressed and
// ceil(w/4) * ceil(h/4) blocks of 8 (BC4) or 16 (BC5) bytes when it is not - which is why `pitch` is the BLOCK row's
// byte count for a block format and a texel row's for the others.
//
// For the same "no graphics api" reason it does not enforce the multiple-of-4 BASE a block-compressed texture needs: a
// .dds whose block-compressed base is 358x198 is a well-formed file with a well-defined level walk, and the rule only
// bites on the way into a texture. Each viewer refuses it by name in its own load_dds - which is where
// webgpu/descriptor.js puts the same refusal, and not in webgpu/dds.js, this file's port.
#pragma once
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

struct DdsLevel {
    int w = 0, h = 0;        // the level's extent in TEXELS, which for a block format is not its extent in blocks
    size_t offset = 0;       // where the level's bytes start inside DdsImage::bytes
    size_t size = 0;         // how many there are
    size_t pitch = 0;        // one row's bytes: a block row for BC4 / BC5, a texel row otherwise
};

struct DdsImage {
    int W = 0, H = 0, mips = 0, dxgi = 0;
    int channels = 0;        // the channels the format stores: 1 / 2 / 4 uncompressed, 1 for BC4, 2 for BC5
    bool bc = false;         // block-compressed, so the levels are blocks and not texels
    int block_bytes = 0;     // 8 for BC4, 16 for BC5, 0 when the format is uncompressed
    std::vector<DdsLevel> levels;
    std::string bytes;       // the whole file, which the levels index into
};

// The file as bytes, or an empty string. The reader keeps the whole file rather than seeking level by level because
// every caller wants the levels at once and the largest asset here is a few tens of megabytes.
static std::string dds_read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

// Reads `path` into `img`, or prints one ERROR line naming the file and the reason and returns false. Nothing is
// written into `img` that a caller should read after a failure.
static bool dds_read(const std::string& path, DdsImage& img) {
    std::string data = dds_read_file(path);
    if (data.size() < 148) { fprintf(stderr, "ERROR: cannot read '%s' (or too short)\n", path.c_str()); return false; }
    const uint32_t* h = (const uint32_t*)data.data();
    if (h[0] != 0x20534444u || h[1] != 124 || h[19] != 32 || h[20] != 0x4u || h[21] != 0x30315844u) { fprintf(stderr, "ERROR: '%s' is not a DX10 .dds\n", path.c_str()); return false; }
    const int H = (int)h[3], W = (int)h[4], nmip = (int)h[7], dxgi = (int)h[32], dim = (int)h[33], arr = (int)h[35];
    if (dim != 3 || arr != 1) { fprintf(stderr, "ERROR: '%s': not a 2D texture (dimension %d, array size %d)\n", path.c_str(), dim, arr); return false; }
    // The three header fields a malformed file can turn into a crash rather than an error: a dimension read as a signed
    // int wraps negative, and a mip count is a vector's size below. They are bounded here, before anything is sized
    // from them - 16384 is D3D11's own texture limit, and a chain cannot be longer than the base's log2.
    if (W < 1 || W > 16384 || H < 1 || H > 16384) { fprintf(stderr, "ERROR: '%s': %dx%d is not a size this reader accepts (1..16384)\n", path.c_str(), W, H); return false; }
    int max_mips = 1; for (int n = W > H ? W : H; n > 1; n >>= 1) max_mips++;
    if (nmip < 1 || nmip > 15 || nmip > max_mips) { fprintf(stderr, "ERROR: '%s': %d mip levels for a %dx%d base (1..%d)\n", path.c_str(), nmip, W, H, max_mips < 15 ? max_mips : 15); return false; }
    const bool bc = dxgi == 80 || dxgi == 83;
    const int block_bytes = dxgi == 80 ? 8 : 16;
    const int C = dxgi == 61 ? 1 : (dxgi == 49 ? 2 : (dxgi == 28 ? 4 : (dxgi == 80 ? 1 : (dxgi == 83 ? 2 : 0))));
    if (!C) { fprintf(stderr, "ERROR: '%s': DXGI format %d is not R8 / R8G8 / R8G8B8A8 / BC4_UNORM / BC5_UNORM\n", path.c_str(), dxgi); return false; }
    std::vector<DdsLevel> levels((size_t)nmip);
    size_t off = 148; int w = W, hh = H;
    for (int i = 0; i < nmip; i++) {
        const int bx = (w + 3) / 4, by = (hh + 3) / 4;
        const size_t n = bc ? (size_t)bx * by * block_bytes : (size_t)w * hh * C;
        if (off + n > data.size()) { fprintf(stderr, "ERROR: '%s' is truncated at level %d\n", path.c_str(), i); return false; }
        levels[i].w = w; levels[i].h = hh; levels[i].offset = off; levels[i].size = n;
        levels[i].pitch = bc ? (size_t)bx * block_bytes : (size_t)w * C;
        off += n; w = w > 1 ? w >> 1 : 1; hh = hh > 1 ? hh >> 1 : 1;
    }
    if (off != data.size()) { fprintf(stderr, "ERROR: '%s': %zu bytes of pixel data expected, %zu present\n", path.c_str(), off - 148, data.size() - 148); return false; }
    img.W = W; img.H = H; img.mips = nmip; img.dxgi = dxgi; img.channels = C; img.bc = bc;
    img.block_bytes = bc ? block_bytes : 0;
    img.levels = std::move(levels); img.bytes = std::move(data);
    return true;
}
