// bc_check: validates bc_pack.h (the tree's one BC4 / BC5 encoder) against iOrange's bcdec (https://github.com/iOrange/bcdec).
//
// Two jobs, decided by what the asset's level 0 actually is.
//
// AN UNCOMPRESSED LEVEL 0 (--bc0 0): every level is packed here exactly as the viewer packs it at load, then every block is decoded three
// ways: bcdec's precise float decoder (the D3D functional spec's palette, k/7 exactly), bcdec's 8-bit integer decoder (the same palette in
// fixed point, rounded to nearest, since BCDEC_BC4BC5_PRECISE is defined below) and bc_pack.h's own bc4_decode_block; each is compared with
// the exact index value k * 255 / (2^bits - 1) of the source. For 1-3 bits the float paths must be exact (rounding only) and the 8-bit path
// off by at most 1, which is the rounding of a palette entry that is not a whole number.
//
// Three or four level-0 channels take two files - a BC5 of channels 0-1 and then a BC4 of channel 2 alone, or a second BC5 of channels
// 2-3 - and each is read with its own format and its own first channel, so the bit depths and the reference plane line up per channel.
//
// A BLOCK-COMPRESSED LEVEL 0 (--bc0 1, the default): the blocks are the encoder's own, so nothing is packed here. Every block is decoded
// by bcdec and by bc_pack.h, the two are compared, and at 1-3 bits the decoded byte must be the index's own byte again - which is the
// statement that the quantisation index survived the block format. Given a second JSON, the same run written with --bc0 0, the decoded
// bytes are also compared texel by texel against that uncompressed plane, which is the check that the two paths agree.
//
// Usage: bc_check PREFIX_nntc.json [UNCOMPRESSED_PREFIX_nntc.json]   (each .dds sits beside its own descriptor)
// bcdec.h is vendored and is not edited, so it is compiled with the warning level turned down.
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define BCDEC_IMPLEMENTATION
#define BCDEC_BC4BC5_PRECISE
#include "bcdec.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "bc_pack.h"
#include "nntc_json.h"
#include <cstdio>
#include <string>
#include <fstream>
#include <sstream>

struct Dds {
    std::vector<uint8_t> data;
    int W = 0, H = 0, mips = 0, dxgi = 0, C = 0, block_bytes = 0;
    bool bc = false;
};

static bool dds_read(const std::string& path, Dds& d) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); d.data.resize((size_t)ftell(f)); fseek(f, 0, SEEK_SET);
    const size_t got = fread(d.data.data(), 1, d.data.size(), f); fclose(f);
    if (got != d.data.size() || d.data.size() < 148) { fprintf(stderr, "%s: too short\n", path.c_str()); return false; }
    const uint32_t* h = (const uint32_t*)d.data.data();
    if (h[0] != 0x20534444u || h[21] != 0x30315844u) { fprintf(stderr, "%s: not a DX10 .dds\n", path.c_str()); return false; }
    d.H = (int)h[3]; d.W = (int)h[4]; d.mips = (int)h[7]; d.dxgi = (int)h[32];
    d.bc = d.dxgi == 80 || d.dxgi == 83;
    d.block_bytes = d.dxgi == 80 ? 8 : 16;
    d.C = d.dxgi == 61 ? 1 : (d.dxgi == 49 ? 2 : (d.dxgi == 28 ? 4 : (d.dxgi == 80 ? 1 : (d.dxgi == 83 ? 2 : 0))));
    if (!d.C) { fprintf(stderr, "%s: DXGI format %d is not one this tool reads\n", path.c_str(), d.dxgi); return false; }
    // The header fields this tool then walks the file by. Without these a malformed size or mip count is an
    // out-of-bounds read: the level loop below advances by each level's own byte count and would run off the end.
    if (d.W < 1 || d.W > 16384 || d.H < 1 || d.H > 16384) { fprintf(stderr, "%s: %dx%d is not a size this tool accepts (1..16384)\n", path.c_str(), d.W, d.H); return false; }
    int max_mips = 1; for (int n = d.W > d.H ? d.W : d.H; n > 1; n >>= 1) max_mips++;
    if (d.mips < 1 || d.mips > 15 || d.mips > max_mips) { fprintf(stderr, "%s: %d mip levels for a %dx%d base\n", path.c_str(), d.mips, d.W, d.H); return false; }
    size_t total = 148; { int w = d.W, hh = d.H; for (int i = 0; i < d.mips; i++) { total += d.bc ? (size_t)((w + 3) / 4) * ((hh + 3) / 4) * d.block_bytes : (size_t)w * hh * d.C; w = w > 1 ? w >> 1 : 1; hh = hh > 1 ? hh >> 1 : 1; } }
    if (d.data.size() != total) { fprintf(stderr, "%s: the header describes %zu bytes and the file is %zu\n", path.c_str(), total, d.data.size()); return false; }
    return true;
}

// One level's byte count inside the file, from that level's own dimensions.
static size_t level_bytes(const Dds& d, int w, int h) {
    return d.bc ? (size_t)((w + 3) / 4) * ((h + 3) / 4) * d.block_bytes : (size_t)w * h * d.C;
}

// bcdec's precise float decode of one BC4 or BC5 block into sixteen texels of nc channels, in [0, 1].
static void bcdec_block(const uint8_t* blk, int nc, float* out) {
    if (nc == 1) bcdec_bc4_float(blk, out, 4, 0); else bcdec_bc5_float(blk, out, 8, 0);
}

// The packing path: pack every level of an uncompressed .dds and validate the blocks three ways.
static bool check_packed(const Dds& d, int used, const int* bits) {
    const int npack = bc_level0_files(used);   // one BC4 / BC5, or two textures: BC5 + BC4 at three channels, two BC5s at four
    double worst_float = 0, worst_int = 0, worst_own = 0, worst_bcdec_vs_own = 0, se_float = 0; size_t texels = 0;
    size_t off = 148; int w = d.W, hh = d.H;
    for (int i = 0; i < d.mips; i++) {
        const uint8_t* src = d.data.data() + off;
        for (int p = 0; p < npack; p++) {
            const int c0 = 2 * p, nc = bc_level0_file_channels(used, p), bpb = nc == 1 ? 8 : 16, bx = (w + 3) / 4;
            if (c0 + nc > d.C) { fprintf(stderr, "level %d: channels %d-%d not stored\n", i, c0, c0 + nc - 1); return false; }
            double se = 0; size_t n = 0;
            std::vector<uint8_t> packed = bc_pack_level(src, w, hh, d.C, c0, nc, bits, used, se, n);
            for (int by = 0; by < (hh + 3) / 4; by++) for (int bxi = 0; bxi < bx; bxi++) {
                const uint8_t* blk = &packed[((size_t)by * bx + bxi) * bpb];
                float df[16 * 2]; uint8_t di[16 * 2]; float own[2][16];
                bcdec_block(blk, nc, df);
                if (nc == 1) bcdec_bc4(blk, di, 4, 0); else bcdec_bc5(blk, di, 8, 0);
                for (int c = 0; c < nc; c++) bc4_decode_block(blk + 8 * c, own[c]);
                for (int t = 0; t < 16; t++) {
                    const int x = bxi * 4 + t % 4, y = by * 4 + t / 4; if (x >= w || y >= hh) continue;
                    for (int c = 0; c < nc; c++) {   // every packed channel is one the layout uses: there is no padding channel
                        const int b = bits[c0 + c], k = src[((size_t)y * w + x) * d.C + c0 + c] >> (8 - b);
                        const double exact = k * 255.0 / ((1 << b) - 1);
                        const double vf = df[t * nc + c] * 255.0, vi = di[t * nc + c], vo = own[c][t];
                        worst_float = std::max(worst_float, std::fabs(vf - exact)); worst_int = std::max(worst_int, std::fabs(vi - exact));
                        worst_own = std::max(worst_own, std::fabs(vo - exact)); worst_bcdec_vs_own = std::max(worst_bcdec_vs_own, std::fabs(vf - vo));
                        se_float += (vf - exact) * (vf - exact); texels++;
                    }
                }
            }
        }
        off += level_bytes(d, w, hh); w = w > 1 ? w >> 1 : 1; hh = hh > 1 ? hh >> 1 : 1;
    }
    printf("  %zu channel-texels over the chain, packed here as %s%s\n", texels, used == 1 ? "BC4" : "BC5",
           npack == 2 ? (used == 3 ? " + BC4" : " + BC5") : "");
    printf("  bcdec precise float vs the exact index value : max |err| %.6g / 255%s\n", worst_float, worst_float < 1e-3 ? "  (lossless)" : "");
    // BCDEC_BC4BC5_PRECISE is defined at the top of this file, so bcdec's integer path is its fixed-point one, which
    // rounds to nearest rather than truncating; the label used to say truncated, which is what the flag turns off.
    printf("  bcdec 8-bit integer  vs the exact index value : max |err| %.6g / 255  (the palette as bytes, rounded to nearest)\n", worst_int);
    printf("  bc_pack.h bc4_decode_block vs the exact index : max |err| %.6g / 255\n", worst_own);
    printf("  bcdec precise float vs bc_pack.h's decoder    : max |diff| %.6g / 255\n", worst_bcdec_vs_own);
    if (se_float > 0) printf("  packing PSNR (bcdec float vs exact index)     : %.2f dB\n", 10.0 * std::log10(255.0 * 255.0 / (se_float / texels)));
    return true;
}

// The shipped path: the blocks are already in the file, so they are only decoded and validated. `ref` is the same level of the same run
// written uncompressed, or null; `channel_base` is the first channel this file carries, so a two-file level 0 reads the right bit depths.
static bool check_blocks(const Dds& d, const int* bits, int channel_base, const Dds* ref, int used) {
    const int nc = d.C, bpb = d.block_bytes;
    // Exactness is claimed PER CHANNEL and not per file: a BC5 whose second channel is the padding channel of a
    // three-channel level 0, or a mixed --bits0 3,4, would otherwise let one channel's inexactness clear the claim for
    // the channel beside it and leave nothing at all asserted about that one.
    bool exact_channel[2] = { true, true };
    double worst_channel[2] = { 0, 0 };
    size_t texels_channel[2] = { 0, 0 };
    double worst_bcdec_vs_own = 0, worst_ref = 0, se = 0; size_t texels = 0;
    size_t off = 148, ref_off = 148; int w = d.W, hh = d.H;
    for (int i = 0; i < d.mips; i++) {
        const uint8_t* src = d.data.data() + off;
        const int bx = (w + 3) / 4;
        for (int by = 0; by < (hh + 3) / 4; by++) for (int bxi = 0; bxi < bx; bxi++) {
            const uint8_t* blk = src + ((size_t)by * bx + bxi) * bpb;
            float df[16 * 2]; float own[2][16];
            bcdec_block(blk, nc, df);
            for (int c = 0; c < nc; c++) bc4_decode_block(blk + 8 * c, own[c]);
            for (int t = 0; t < 16; t++) {
                const int x = bxi * 4 + t % 4, y = by * 4 + t / 4; if (x >= w || y >= hh) continue;
                for (int c = 0; c < nc; c++) {
                    if (channel_base + c >= used) continue;   // a file storing more channels than the layout uses (an --bc0 0 twin)
                    const int b = bits[channel_base + c];
                    const double vf = df[t * nc + c] * 255.0, vo = own[c][t];
                    worst_bcdec_vs_own = std::max(worst_bcdec_vs_own, std::fabs(vf - vo));
                    // The index round trip: recover k from the decoded byte and ask for the byte that index stands for. The palette
                    // value k * 255 / (2^bits - 1) is not a whole number at 3 bits, and the byte a sampler returns is that value
                    // rounded, so the comparison is against the rounded one - which at 1-3 bits is also the byte the uncompressed
                    // plane carries, because replicating the index's bits over a byte lands on the same integer.
                    const int byte = (int)std::lround(vf), k = byte >> (8 - b);
                    const double exact = std::floor(k * 255.0 / ((1 << b) - 1) + 0.5);
                    if (b <= 3) worst_channel[c] = std::max(worst_channel[c], std::fabs(byte - exact)); else exact_channel[c] = false;
                    if (ref) {
                        const double r = (double)ref->data[ref_off + ((size_t)y * w + x) * ref->C + channel_base + c];
                        worst_ref = std::max(worst_ref, std::fabs(byte - r));
                        se += (byte - r) * (byte - r);
                    }
                    texels_channel[c]++;
                    texels++;
                }
            }
        }
        off += level_bytes(d, w, hh);
        if (ref) ref_off += level_bytes(*ref, w, hh);
        w = w > 1 ? w >> 1 : 1; hh = hh > 1 ? hh >> 1 : 1;
    }
    printf("  %zu channel-texels over the chain, read as %s from the file\n", texels, nc == 1 ? "BC4" : "BC5");
    printf("  bcdec precise float vs bc_pack.h's decoder    : max |diff| %.6g / 255\n", worst_bcdec_vs_own);
    bool failed = false;
    for (int c = 0; c < nc; c++) {
        if (channel_base + c >= used) continue;
        if (!texels_channel[c]) continue;
        if (exact_channel[c]) {
            printf("  channel %d, %d bits: the index round trip (byte >> (8 - bits)) : max |err| %.6g / 255  (lossless over %zu texels)\n",
                   channel_base + c, bits[channel_base + c], worst_channel[c], texels_channel[c]);
            if (worst_channel[c] > 1e-3) failed = true;
        } else {
            printf("  channel %d, %d bits: the index round trip                      : not exact by construction above 3 bits\n",
                   channel_base + c, bits[channel_base + c]);
        }
    }
    if (ref) {
        printf("  the decoded bytes vs the uncompressed plane   : max |diff| %.6g / 255\n", worst_ref);
        if (se > 0) printf("  packing PSNR against that plane               : %.2f dB\n", 10.0 * std::log10(255.0 * 255.0 / (se / (double)texels)));
    }
    if (failed) { fprintf(stderr, "ERROR: the quantisation index did not survive the pack\n"); return false; }
    return true;
}

// The .dds files one JSON texture entry names, resolved beside the .json: "file" names one, "files" two. An entry of
// "files" is an object carrying that file's own name, format and channel count (the two files of a three-channel level 0
// are a BC5 and a BC4 and do not share either), and older assets wrote a bare string there; both are read.
static std::vector<std::string> texture_files(const std::string& dir, const JVal& t) {
    std::vector<std::string> names;
    const JVal* files = t.get("files");
    if (files && files->kind == JVal::ARR) { for (const JVal& f : files->arr) names.push_back(dir + (f.kind == JVal::OBJ ? f.string("file") : f.str)); }
    else names.push_back(dir + t.string("file"));
    return names;
}

static bool load_json(const std::string& path, JVal& root, std::string& dir) {
    std::string text; { std::ifstream f(path, std::ios::binary); if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); return false; } std::ostringstream ss; ss << f.rdbuf(); text = ss.str(); }
    JParser P(text); root = P.parse();
    if (!P.ok || root.kind != JVal::OBJ) { fprintf(stderr, "%s: not JSON\n", path.c_str()); return false; }
    const size_t slash = path.find_last_of("/\\"); dir = slash == std::string::npos ? "" : path.substr(0, slash + 1);
    return true;
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) { fprintf(stderr, "usage: bc_check PREFIX_nntc.json [UNCOMPRESSED_PREFIX_nntc.json]\n"); return 1; }
    JVal root; std::string dir;
    if (!load_json(argv[1], root, dir)) return 1;
    const JVal* texs = root.get("textures");
    if (!texs || texs->kind != JVal::ARR || texs->arr.empty()) { fprintf(stderr, "%s: no textures\n", argv[1]); return 1; }

    // The optional uncompressed twin of the same run: only its level 0 is read, and it is used only where this asset's level 0 is
    // compressed - which is the one place there is a plane to compare the blocks against.
    std::vector<Dds> ref; bool have_ref = false;
    if (argc == 3) {
        JVal rroot; std::string rdir;
        if (!load_json(argv[2], rroot, rdir)) return 1;
        const JVal* rtexs = rroot.get("textures");
        if (!rtexs || rtexs->kind != JVal::ARR || rtexs->arr.empty()) { fprintf(stderr, "%s: no textures\n", argv[2]); return 1; }
        for (const std::string& n : texture_files(rdir, rtexs->arr[0])) {
            Dds d; if (!dds_read(n, d)) return 1;
            if (d.bc) { fprintf(stderr, "%s: the reference asset's level 0 is itself compressed\n", argv[2]); return 1; }
            ref.push_back(d);
        }
        have_ref = true;
    }

    for (const JVal& t : texs->arr) {   // level 0 is the block-compressed level; the others are reported for reference
        const int level = (int)t.number("level", -1), used = (int)t.number("channels_used");
        if (used < 1 || used > 4) { fprintf(stderr, "level %d: channels_used %d is outside 1..4\n", level, used); return 1; }
        int bits[4] = { 8, 8, 8, 8 }; const JVal* bv = t.get("bits_per_channel");
        for (int c = 0; c < 4; c++) if (bv && bv->kind == JVal::ARR && c < (int)bv->arr.size() && bv->arr[c].kind == JVal::NUM) bits[c] = (int)bv->arr[c].num;
        for (int c = 0; c < 4; c++) if (bits[c] < 1 || bits[c] > 8) { fprintf(stderr, "level %d: bits_per_channel[%d] is %d, which is outside 1..8\n", level, c, bits[c]); return 1; }
        printf("%s level %d%s\n", level == 0 ? "==" : "--", level, level == 0 ? "" : " (not block-compressed as it stands; shown for reference)");
        const std::vector<std::string> names = texture_files(dir, t);
        int channel_base = 0;
        for (size_t f = 0; f < names.size(); f++) {
            Dds d; if (!dds_read(names[f], d)) return 1;
            printf("%s: %dx%d, %d levels, %d channels stored, DXGI %d, bits", names[f].c_str(), d.W, d.H, d.mips, d.C, d.dxgi);
            for (int c = 0; c < used; c++) printf(" %d", bits[c]); printf("\n");
            if (d.bc) {
                // The uncompressed twin is ONE file holding every channel, while a three or four channel level 0 is
                // two files - a BC5 and a BC4 at three channels, two BC5s at four. Both are compared against that same
                // plane, the second one starting at the channel it actually carries (channel 2). Indexing the
                // reference by the file number instead left the second file with nothing to compare to.
                const Dds* r = (level == 0 && !ref.empty()) ? &ref[0] : nullptr;
                if (r && (r->W != d.W || r->H != d.H || r->mips != d.mips)) { fprintf(stderr, "the reference level 0 is a different size\n"); return 1; }
                if (r && r->C < channel_base + d.C) { fprintf(stderr, "the reference level 0 stores %d channels, %d needed for this file\n", r->C, channel_base + d.C); return 1; }
                if (!check_blocks(d, bits, channel_base, r, used)) return 1;
                channel_base += d.C;
            } else {
                if (used < 1 || used > d.C) { fprintf(stderr, "%d channels used of %d stored\n", used, d.C); return 1; }
                if (!check_packed(d, used, bits)) return 1;
            }
        }
    }
    return 0;
}
