// main.cpp: the encoder's driver.
//
// One to six PNGs of the same size are one material: texture t contributes output channels 3t..3t+2, so nout = 3T. Both
// dimensions must be divisible by 4, because level 1 holds a quarter of the texels per axis and because block-compressed
// textures need it; an input that is not divisible by 4 is padded by edge replication to the next multiple, loudly.
//
// The driver loads and pads the source, builds the source chain and the two latents' plane layout, selects the CUDA
// device (there is no CPU encoding path), initialises both latents, fits the decoder and writes the asset.

// The vendored single-header libraries are compiled here, once for the whole program, with the warning level turned
// down: they are third-party code and are not edited. The implementation macros are undefined again straight away
// because stb emits its implementation outside its own include guard, so a second include would define it twice.
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "stb_image_resize2.h"
#undef STB_IMAGE_IMPLEMENTATION
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#undef STB_IMAGE_RESIZE_IMPLEMENTATION
#define NNTC_STBIW_DEFINED   // this translation unit holds stb_image_write's definitions; image.h need not declare one
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <filesystem>
#include <new>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "backend.h"
#include "image.h"
#include "model.h"
#include "nntc_json.h"
#include "options.h"

static const float LUMA_R = 0.299f, LUMA_G = 0.587f, LUMA_B = 0.114f;

// A mean squared error in [0,1] units as a PSNR: 10 log10(1 / mse), which is the 8-bit definition with the 255s
// cancelled. A zero error is reported as 100 dB rather than as an infinity.
static double mse_to_db(double mse)
{
    return mse > 0.0 ? 10.0 * std::log10(1.0 / mse) : 100.0;
}

// The worst-case line of the report: for every pixel of the base level, the largest 8-bit error any of its outputs
// carries, against the source read at the texel centres over the original extent. The distribution of that per-pixel
// maximum is what a mean squared error hides - a bilinear artefact is a handful of pixels a long way out, and it can
// leave the mean where it was - so the report quotes its median, its 90th and 99th percentiles, its maximum and the
// share of pixels above 8 (one thirty-second of the range, about where a smooth gradient's error starts to be visible).
//
// It is not a substitute for looking: it failed to separate two runs that were plainly different by eye, and the
// design notes say so. It is here because it costs nothing and a regression in it is always worth a look.
struct WorstCase
{
    int p50 = 0, p90 = 0, p99 = 0, max = 0;
    double above8 = 0.0;   // per cent of pixels
};

static WorstCase worst_case_of(const Image& src, const uint8_t* rec, int rec_w, int nc, int crop_w, int crop_h)
{
    WorstCase w;
    std::vector<int> worst;
    worst.reserve((size_t)crop_w * crop_h);
    size_t above = 0;
    for (int y = 0; y < crop_h; y++)
        for (int x = 0; x < crop_w; x++)
        {
            int peak = 0;
            for (int c = 0; c < nc; c++)
            {
                const float f = src.at(x, y)[c];
                const int a = (int)std::lround(std::min(1.0f, std::max(0.0f, f)) * 255.0f);
                const int b = rec[((size_t)y * rec_w + x) * nc + c];
                peak = std::max(peak, std::abs(a - b));
            }
            worst.push_back(peak);
            if (peak > 8)
                above++;
        }
    if (worst.empty())
        return w;
    std::sort(worst.begin(), worst.end());
    const size_t last = worst.size() - 1;
    w.p50 = worst[(size_t)(0.50 * (double)last + 0.5)];
    w.p90 = worst[(size_t)(0.90 * (double)last + 0.5)];
    w.p99 = worst[(size_t)(0.99 * (double)last + 0.5)];
    w.max = worst[last];
    w.above8 = 100.0 * (double)above / (double)worst.size();
    return w;
}

// --diag: what one channel of one stored plane holds. The mean and the standard deviation are the plain moments of the
// channel's values over the plane; `ends` is the share of them sitting exactly on an end of the stored grid, which is
// where a channel that wanted to go further than the one grid the whole chain shares ends up.
//
// A near-constant channel (sd of the order of a grid step) on a deep plane is the interesting case: it carries no
// information there, so whatever the decoder asks of it at that level cannot be delivered.
struct ChannelStats
{
    double mean = 0.0, sd = 0.0, lo = 0.0, hi = 0.0, ends = 0.0;
};

static void channel_stats(const std::vector<float>& v, int nc, int channel, float grid_lo, float grid_hi,
                          ChannelStats& s)
{
    double sum = 0.0, sum2 = 0.0;
    size_t n = 0, ends = 0;
    double lo = 1e30, hi = -1e30;
    // Half a grid step: a value is "on an end" when it is the end, and the tolerance covers the float rounding of the
    // dequantisation rather than admitting the next step in.
    const double tol = 1e-6 * (std::fabs((double)grid_hi - (double)grid_lo) + 1.0);
    for (size_t i = (size_t)channel; i < v.size(); i += (size_t)nc)
    {
        const double x = (double)v[i];
        sum += x;
        sum2 += x * x;
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        if (std::fabs(x - (double)grid_lo) <= tol || std::fabs(x - (double)grid_hi) <= tol)
            ends++;
        n++;
    }
    if (n == 0)
        return;
    s.mean = sum / (double)n;
    const double var = sum2 / (double)n - s.mean * s.mean;
    s.sd = var > 0.0 ? std::sqrt(var) : 0.0;
    s.lo = lo;
    s.hi = hi;
    s.ends = 100.0 * (double)ends / (double)n;
}

// How the banner names the level-0 files, in bc_pack.h's own rule: three channels are a BC5 of channels 0-1 and a BC4
// holding channel 2 alone, which is 12 bits per texel, and four are two BC5s at 16.
static const char* level0_files_text(int c0)
{
    return c0 == 1 ? "one BC4 file"
                   : (c0 == 2 ? "one BC5 file"
                              : (c0 == 3 ? "two files, a BC5 of channels 0-1 and a BC4 of channel 2"
                                         : "two BC5 files, channels 0-1 and 2-3"));
}

// |a - b| relative to the larger magnitude, and 0 when both are 0.
static double relative(double a, double b)
{
    const double scale = std::max(std::fabs(a), std::fabs(b));
    return scale > 0.0 ? std::fabs(a - b) / scale : 0.0;
}

// True when every output carries the same weight, which is when the weighted E and the plain mean squared error of the
// same sites are the same number.
static bool uniform_cw(const Model& m)
{
    for (int c = 1; c < m.nout; c++)
        if (m.cw[c] != m.cw[0])
            return false;
    return true;
}

// A central finite difference of the objective in one level-1 value: E is exactly quadratic in the plane, so the
// central difference IS the derivative up to rounding, and at the solution of block (b) it must be zero to within the
// proximal ridge's own pull. The two probes are written to the device as floats and the difference actually realised
// by those two floats is used as the step, so a value whose neighbourhood is coarser than h does not bias the result.
//
// The probed values are chosen by a fixed linear congruential sequence started from a constant: the check is
// reproducible run to run and there is no seed to pass.
static double finite_difference_probe(be::Device* d, const Model& m, int k, const std::vector<double>& per_site,
                                      int plane, double h, int samples, double scale)
{
    std::vector<float> cur;
    be::level1_download(d, m, plane, cur);
    if (cur.empty() || !(scale > 0.0))
        return 0.0;
    uint64_t state = 0x9E3779B97F4A7C15ull;
    double worst = 0.0;
    Objective up, down;
    for (int i = 0; i < samples; i++)
    {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        const size_t index = (size_t)((state >> 16) % (uint64_t)cur.size());
        const float base = cur[index];
        const float plus = (float)((double)base + h), minus = (float)((double)base - h);
        const double step = (double)plus - (double)minus;
        be::level1_poke(d, plane, index, plus);
        be::objective_eval(d, m, k, per_site, up);
        be::level1_poke(d, plane, index, minus);
        be::objective_eval(d, m, k, per_site, down);
        be::level1_poke(d, plane, index, base);
        if (!(step > 0.0))
            continue;
        const double derivative = (up.e_plane[(size_t)plane] - down.e_plane[(size_t)plane]) / step;
        worst = std::max(worst, std::fabs(derivative) * h / scale);
    }
    return worst;
}

// A path's own file name, for a log line that would otherwise carry the whole path twice over.
static std::string basename_of_path(const std::string& path)
{
    const size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

// A path as a message should print it. A `file` resolved against a material's directory comes out as
// `json/../img/nope.png` with the separators of two different platforms in it; the components are folded away and one
// separator is used throughout, so that what the error names is what the caller would type.
static std::string tidy_path(const std::filesystem::path& p)
{
    std::filesystem::path q = p.lexically_normal();
    return q.make_preferred().string();
}

// Whether two paths name the same file, as far as can be told before one of them exists. weakly_canonical resolves
// whatever part of the path IS on disk - which is the directory here, never the asset's .json, since that is exactly
// what has not been written yet - and normalises the rest; a filesystem error leaves the lexical form, which is still
// enough to catch the everyday spelling. Windows file names are case-insensitive, so the comparison is too, there and
// only there.
static bool same_file_path(const std::filesystem::path& a, const std::filesystem::path& b)
{
    std::error_code ec;
    std::filesystem::path x = std::filesystem::weakly_canonical(a, ec);
    if (ec)
    {
        ec.clear();
        x = a.lexically_normal();
    }
    std::filesystem::path y = std::filesystem::weakly_canonical(b, ec);
    if (ec)
        y = b.lexically_normal();
    std::string sx = x.make_preferred().string(), sy = y.make_preferred().string();
#ifdef _WIN32
    for (char& c : sx)
        c = (char)std::tolower((unsigned char)c);
    for (char& c : sy)
        c = (char)std::tolower((unsigned char)c);
#endif
    return sx == sy;
}

// The everyday command line is one line with no flags: the inputs, and at most -o. --help shows that and the layout;
// --help-advanced shows the solver and testing flags, which exist for measurement and have defaults that do not need
// touching.
static void usage()
{
    printf("nntc_encode <in0> [in1 .. in5] [-o DIR_or_PREFIX] [flags]\n");
    printf("nntc_encode <material.json> [-o DIR_or_PREFIX] [flags]\n\n");
    printf("One to six images of the same size form one material (texture t gives output channels 3t..3t+2), in any\n");
    printf("format stb_image reads: PNG, JPEG, TGA, BMP, PSD, GIF, HDR, PIC, PNM. Three channels each; an alpha\n");
    printf("channel is ignored with a warning.\n");
    printf("The asset is PREFIX_lat0.dds (or _lat0a and _lat0b), PREFIX_lat1.dds and PREFIX_nntc.json; the default\n");
    printf("PREFIX is the first input's base name (or the material's) in the current directory.\n\n");
    printf("A MATERIAL JSON names the inputs and their per-texture settings instead, and the asset then takes the\n");
    printf("JSON's own base name. It is one array in texture order, a relative \"file\" resolving against the JSON:\n\n");
    printf("  [ { \"file\": \"albedo.png\", \"type\": \"albedo\", \"filter\": \"mitchell\", \"srgb\": true },\n");
    printf("    { \"file\": \"normal.png\", \"filter\": \"mitchell\", \"normal_map\": true },\n");
    printf("    { \"file\": \"rough.png\", \"weight\": 2, \"rgb_weights\": [1, 1, 0.5] } ]\n\n");
    printf("  keys: file (required), type, filter, srgb, edge (clamp|wrap), normal_map, weight, rgb_weights.\n");
    printf("  srgb, edge and normal_map describe a DERIVED chain, which the default filter is not: a texture that\n");
    printf("  sets one of them and no filter at all is switched to 'box' and the settings row says so; setting one\n");
    printf("  beside an explicit filter of 'default' contradicts it and is refused. An unknown key is refused too.\n");
    printf("  The command line overrides a key and says which value it replaced.\n\n");
    printf("  -o DIR_or_PREFIX  a bare name with no extension (or a trailing separator) is a DIRECTORY and is\n");
    printf("                    created; a name ending in .json IS the descriptor (out/name.json beside\n");
    printf("                    out/name_lat0.dds); any other name with an extension, or an existing file, is\n");
    printf("                    the prefix itself\n");
    printf("  --no-mkdir        refuse if -o names a directory that does not exist, rather than creating it\n");
    printf("  --weights W[,W]   per-texture loss weight, ONE PER TEXTURE or none at all  (default 1)\n"
           "                    values above 128 are clamped to it, with a warning\n");
    printf("  --mip-filter F[,F]  per-texture source-chain filter, one per texture or none (default default)\n");
    printf("                    default (the iterated 2x2 box), box, mitchell, catmullrom\n");
    printf("  --c0 N            level-0 channels, 1..4 (full resolution, the selector)   (default 2)\n");
    printf("  --l0 bc8|palette  bc8 solves level 0 as a continuous 8-bit plane, BC4 / BC5- (default bc8)\n");
    printf("                    encodes it and refines the pack block by block against the\n");
    printf("                    decoder's own error; palette is the 1-4 bit index instead\n");
    printf("  --bits0 B[,B]     bits per level-0 channel, each 1..4, under --l0 palette   (default 3)\n");
    printf("                    one value stands for every channel, unlike the per-texture lists above: it is per\n");
    printf("                    level-0 CHANNEL, and how many there are is --c0's business and not the list's\n");
    printf("  --c1 N            level-1 channels, 1..4 (quarter resolution, bilinear)    (default 4)\n");
    printf("  --mips 0|1        1 stores both mip chains, 0 one level per texture        (default 1)\n");
    printf("  --bc0 0|1         1 writes level 0 as BC4 / BC5, 0 as an uncompressed plane (default 1)\n\n");
    printf("  (--bc0 both writes the run both ways; see --help-advanced)\n\n");
    printf("  --help-advanced for the solver and testing options\n");
}

static void usage_advanced()
{
    printf("nntc_encode: the solver and testing options. Every one of them has a default that measures well; they are\n");
    printf("here so an arm can be run against another.\n\n");
    printf("  --k K             fixed subtexel sites per texel, K in {0,1,2,4}           (default 4)\n");
    printf("  --rounds R        the block-round budget                                   (default 20)\n");
    printf("  --tol X           stop below this relative drop in E, twice running        (default 1e-4)\n");
    printf("  --q1-start R0     the round at which level 1's grid is frozen, 0 = at the end (default 3)\n");
    printf("  --q1-sweeps N     four-colour Gauss-Seidel sweeps per quantised block (b)  (default 4)\n");
    printf("  --q1-range R      minmax|pct: level 1's grid range per channel             (default pct)\n");
    printf("  --bits1 B         bits for every level-1 channel, 4..8                     (default 8)\n");
    printf("                    level 1 is ALWAYS stored as eight bits per channel in the .dds, so a value below 8\n");
    printf("                    only reduces how many distinct values those bits carry (the index is bit-replicated\n");
    printf("                    back into the byte) and saves no memory at all: it is an experiment, not a setting\n");
    printf("  --init pca|box    level 1's initialisation                                 (default box)\n");
    printf("  --init0 R         residual|texture|luma: level 0's initialisation          (default residual)\n");
    printf("  --init0-scope S   base|chain: where that residual is measured                (default base)\n");
    printf("  --mip-min S       halve while level 0 stays >= S per axis                  (default 8)\n");
    printf("  --rgb-weights R,G,B  per-channel weight inside a texture                   (default 1,1,1)\n");
    printf("                    ONE triple, applied to EVERY texture and normalised to mean 1 within each of them;\n");
    printf("                    a per-texture triple is a material JSON's rgb_weights, which this replaces on every\n");
    printf("                    texture that carried one, with a WARNING naming both values\n");
    printf("  --mip-weight M    sqrt|pixels|uniform: the chain's share of the decoder    (default pixels)\n");
    printf("  --mip-mix X       the chain's total share of that mass                     (default 0.5)\n");
    printf("  --ridge X         level 1's proximal ridge, relative to mean(diag H)       (default 1e-4)\n");
    printf("  --sweeps N        level-0 coordinate sweeps when the enumeration is large  (default 2)\n");
    printf("  --bc-refine N     rounds of the per-block refinement of the BC pack, 0 off  (default 4)\n");
    printf("  --bc-refine-after N  further rounds after level 1 and the decoder are refitted (default 0)\n");
    printf("  --bc-outer N      outer repack passes, each taken only if the shipped E falls (default 2)\n");
    printf("  --device N        the CUDA device                                          (default 0)\n");
    printf("  --backend B       auto|cuda|cpu|check: which backend runs the encode       (default auto)\n");
    printf("                    auto tries cuda and falls back to cpu with a WARNING naming why; cuda asked for by\n");
    printf("                    name is an ERROR wherever it cannot be used, rather than quietly something else.\n");
    printf("                    cpu is a scalar C++ port of the CUDA kernels with std::thread, for machines with no\n");
    printf("                    NVIDIA GPU: the same algorithm, several times slower. check is the per-kernel\n");
    printf("                    harness, in a build with both backends: cuda runs the encode and at every call the\n");
    printf("                    cpu backend is given the same state, run and compared; a summary per function\n");
    printf("                    ends the run, and any MISMATCH makes the exit status 1\n");
    printf("  --kcheck-only F   under check, print every call of seam function F, not only the ones not ok\n");
    printf("  --kcheck-stop     under check, end the run at the first MISMATCH\n");
    printf("  -j N              the cpu backend's worker threads, 0 = as many as the machine has (default 0)\n");
    printf("                    0 asks std::thread::hardware_concurrency(), clamped into 1..1024, and a machine that\n");
    printf("                    cannot answer counts as one; the count is resolved when the pool is built, so the\n");
    printf("                    number the report prints is the number that ran\n");
    printf("  --png 0|1         write the recon and source PNGs per level                (default 0)\n");
    printf("  --bc0 both        write the same solve twice: the asset with level 0 block-compressed,\n");
    printf("                    and the PREFIX_u twin (PREFIX_u_nntc.json) with level 0 uncompressed, so\n");
    printf("                    the two decode paths can be compared on one set of planes         (off)\n");
    printf("  --check           recompute E by brute force on the host and print both            (off)\n");
    printf("  --diag            the per-plane diagnosis table: every texture's psnr at every level and\n");
    printf("                    every latent channel's statistics and grid-end share, plane by plane   (off)\n");
    printf("  --quiet           print nothing but the WARNING and ERROR lines: no banner, no progress, no\n");
    printf("                    `wrote` lines and no report                                             (off)\n");
}

// The command line's numbers are parsed with the whole argument consumed, so that a value the shell mangled or a typed
// unit is refused rather than silently truncated: atoi("1e3") is 1 and atoi("abc") is 0, and a run that quietly used a
// budget of one round or a tolerance of zero would look like a result.
static bool parse_int(const char* s, int& out)
{
    char* end = nullptr;
    errno = 0;
    const long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX)
        return false;
    out = (int)v;
    return true;
}

// The sane range of a texture or channel weight, and the clamp onto it (see the weight block of validate_options).
// A weight of zero is allowed - it is how a caller says a texture or a channel does not count - and only a POSITIVE
// value too small for a float is raised, so that a texture the caller gave a weight to keeps one.
// The tolerances of every "did E rise" comparison (see the round loop); the gate uses the same two numbers.
static const double E_NOISE = 1e-8;
static const double E_REL = 1e-6;

static const double WEIGHT_MAX = 128.0;
static const double WEIGHT_MIN = 1e-6;
static void clamp_weight(size_t texture, const char* what, double& w)
{
    if (!(w <= WEIGHT_MAX))
    {
        printf("WARNING: texture %zu's %s %g is above the sane range and is clamped to %g\n", texture, what, w, WEIGHT_MAX);
        w = WEIGHT_MAX;
    }
    else if (w > 0.0 && w < WEIGHT_MIN)
    {
        printf("WARNING: texture %zu's %s %g is below what the objective can hold and is raised to %g\n", texture, what,
               w, WEIGHT_MIN);
        w = WEIGHT_MIN;
    }
}

static bool parse_double(const char* s, double& out)
{
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s, &end);
    if (end == s || *end != '\0' || errno == ERANGE || !std::isfinite(v))
        return false;
    out = v;
    return true;
}

static bool parse_int_list(const std::string& s, std::vector<int>& out)
{
    out.clear();
    size_t i = 0;
    while (i <= s.size())
    {
        const size_t comma = s.find(',', i);
        const std::string part = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        int v = 0;
        if (part.empty() || !parse_int(part.c_str(), v))
            return false;
        out.push_back(v);
        if (comma == std::string::npos)
            break;
        i = comma + 1;
    }
    return !out.empty();
}

static bool parse_double_list(const std::string& s, std::vector<double>& out)
{
    out.clear();
    size_t i = 0;
    while (i <= s.size())
    {
        const size_t comma = s.find(',', i);
        const std::string part = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        double v = 0.0;
        if (part.empty() || !parse_double(part.c_str(), v))
            return false;
        out.push_back(v);
        if (comma == std::string::npos)
            break;
        i = comma + 1;
    }
    return !out.empty();
}

// The same split for a list of names. An empty entry is refused rather than taken as a default, because `a,,b` is a
// typed comma and not a request for the middle texture's default.
static bool parse_string_list(const std::string& s, std::vector<std::string>& out)
{
    out.clear();
    size_t i = 0;
    while (i <= s.size())
    {
        const size_t comma = s.find(',', i);
        const std::string part = s.substr(i, comma == std::string::npos ? std::string::npos : comma - i);
        if (part.empty())
            return false;
        out.push_back(part);
        if (comma == std::string::npos)
            break;
        i = comma + 1;
    }
    return !out.empty();
}

static bool need_value(int i, int argc, const char* flag)
{
    if (i < argc)
        return true;
    fprintf(stderr, "ERROR: %s needs a value\n", flag);
    return false;
}

// One numeric flag and its value, with the argument index advanced past it.
static bool value_int(int& i, int argc, char** argv, const char* flag, int& out)
{
    if (!need_value(i + 1, argc, flag))
        return false;
    if (!parse_int(argv[++i], out))
    {
        fprintf(stderr, "ERROR: %s '%s' is not a whole number\n", flag, argv[i]);
        return false;
    }
    return true;
}

static bool value_double(int& i, int argc, char** argv, const char* flag, double& out)
{
    if (!need_value(i + 1, argc, flag))
        return false;
    if (!parse_double(argv[++i], out))
    {
        fprintf(stderr, "ERROR: %s '%s' is not a number\n", flag, argv[i]);
        return false;
    }
    return true;
}

// The four filter names the command line and the material JSON both accept. `point`, `triangle` and the cubic B-spline
// exist in the vendored resizer and are not exposed: point keeps one texel in sixty-four at 8:1 and hands the deep
// planes an aliased target, which is the opposite of what the chain is for.
static bool filter_name_ok(const std::string& name)
{
    return name == "default" || name == "box" || name == "mitchell" || name == "catmullrom";
}

// Whether a path's extension is .json, in any case. Two spellings turn on it: a positional naming a material, and -o
// naming the asset's descriptor outright.
static bool has_json_extension(const std::filesystem::path& p)
{
    std::string ext = p.extension().string();
    for (char& c : ext)
        c = (char)std::tolower((unsigned char)c);
    return ext == ".json";
}

static bool parse_options(int argc, char** argv, Options& o)
{
    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help")
        {
            usage();
            o.help = true;
            return false;
        }
        else if (a == "--help-advanced")
        {
            usage_advanced();
            o.help = true;
            return false;
        }
        else if (a == "-o")
        {
            if (!need_value(i + 1, argc, "-o"))
                return false;
            o.prefix = argv[++i];
        }
        else if (a == "--c0")
        {
            if (!value_int(i, argc, argv, "--c0", o.c0))
                return false;
        }
        else if (a == "--bits0")
        {
            if (!need_value(i + 1, argc, "--bits0") || !parse_int_list(argv[++i], o.bits0))
                return false;
            o.bits0_given = true;
        }
        else if (a == "--l0")
        {
            if (!need_value(i + 1, argc, "--l0"))
                return false;
            o.l0 = argv[++i];
        }
        else if (a == "--bc-refine")
        {
            if (!value_int(i, argc, argv, "--bc-refine", o.bc_refine))
                return false;
            o.bc_refine_given = true;
        }
        else if (a == "--bc-refine-after")
        {
            if (!value_int(i, argc, argv, "--bc-refine-after", o.bc_refine_after))
                return false;
            o.bc_refine_after_given = true;
        }
        else if (a == "--bc-outer")
        {
            if (!value_int(i, argc, argv, "--bc-outer", o.bc_outer))
                return false;
            o.bc_outer_given = true;
        }
        else if (a == "--c1")
        {
            if (!value_int(i, argc, argv, "--c1", o.c1))
                return false;
        }
        else if (a == "--bits1")
        {
            if (!value_int(i, argc, argv, "--bits1", o.bits1))
                return false;
        }
        else if (a == "--bc0")
        {
            if (!need_value(i + 1, argc, "--bc0"))
                return false;
            const std::string v = argv[++i];
            o.bc0_both = v == "both";
            if (!o.bc0_both && !parse_int(v.c_str(), o.bc0))
            {
                fprintf(stderr, "ERROR: --bc0 '%s' must be 0, 1 or both\n", v.c_str());
                return false;
            }
            if (o.bc0_both)
                o.bc0 = 1;
        }
        else if (a == "--mip-min")
        {
            if (!value_int(i, argc, argv, "--mip-min", o.mip_min))
                return false;
        }
        else if (a == "--mips")
        {
            if (!value_int(i, argc, argv, "--mips", o.mips))
                return false;
        }
        else if (a == "--k")
        {
            if (!value_int(i, argc, argv, "--k", o.k))
                return false;
        }
        else if (a == "--rounds")
        {
            if (!value_int(i, argc, argv, "--rounds", o.rounds))
                return false;
        }
        else if (a == "--tol")
        {
            if (!value_double(i, argc, argv, "--tol", o.tol))
                return false;
        }
        else if (a == "--q1-start")
        {
            if (!value_int(i, argc, argv, "--q1-start", o.q1_start))
                return false;
        }
        else if (a == "--q1-sweeps")
        {
            if (!value_int(i, argc, argv, "--q1-sweeps", o.q1_sweeps))
                return false;
        }
        else if (a == "--q1-range")
        {
            if (!need_value(i + 1, argc, "--q1-range"))
                return false;
            o.q1_range = argv[++i];
        }
        else if (a == "--init0-scope")
        {
            if (!need_value(i + 1, argc, "--init0-scope"))
                return false;
            o.init0_scope = argv[++i];
        }
        else if (a == "--init0")
        {
            if (!need_value(i + 1, argc, "--init0"))
                return false;
            o.init0 = argv[++i];
        }
        else if (a == "--init")
        {
            if (!need_value(i + 1, argc, "--init"))
                return false;
            o.init = argv[++i];
        }
        else if (a == "--weights")
        {
            // The refusal SAYS SO. This arm used to return false without a word, so `--weights abc`, an empty string, a
            // stray comma, `inf` and `1e999` all exited 1 in silence while the --mip-filter arm nine lines below named
            // its own bad value - and a caller reading a script's exit code alone had nothing to go on.
            if (!need_value(i + 1, argc, "--weights"))
                return false;
            if (!parse_double_list(argv[++i], o.weights))
            {
                fprintf(stderr, "ERROR: --weights '%s' is not a comma-separated list of numbers\n", argv[i]);
                return false;
            }
        }
        else if (a == "--mip-filter")
        {
            if (!need_value(i + 1, argc, "--mip-filter"))
                return false;
            if (!parse_string_list(argv[++i], o.mip_filter))
            {
                fprintf(stderr, "ERROR: --mip-filter '%s' is not a comma-separated list of filter names\n", argv[i]);
                return false;
            }
        }
        else if (a == "--rgb-weights")
        {
            if (!need_value(i + 1, argc, "--rgb-weights"))
                return false;
            if (!parse_double_list(argv[++i], o.rgb_weights))   // named, for the reason --weights above is
            {
                fprintf(stderr, "ERROR: --rgb-weights '%s' is not a comma-separated list of numbers\n", argv[i]);
                return false;
            }
            o.rgb_weights_given = true;
        }
        else if (a == "--mip-weight")
        {
            if (!need_value(i + 1, argc, "--mip-weight"))
                return false;
            o.mip_weight = argv[++i];
        }
        else if (a == "--mip-mix")
        {
            if (!value_double(i, argc, argv, "--mip-mix", o.mip_mix))
                return false;
        }
        else if (a == "--ridge")
        {
            if (!value_double(i, argc, argv, "--ridge", o.ridge))
                return false;
        }
        else if (a == "--sweeps")
        {
            if (!value_int(i, argc, argv, "--sweeps", o.sweeps))
                return false;
        }
        else if (a == "--device")
        {
            if (!value_int(i, argc, argv, "--device", o.device))
                return false;
            o.device_given = true;
        }
        else if (a == "--backend")
        {
            if (!need_value(i + 1, argc, "--backend"))
                return false;
            o.backend = argv[++i];
            o.backend_given = true;
        }
        else if (a == "--kcheck-only")
        {
            if (!need_value(i + 1, argc, "--kcheck-only"))
                return false;
            o.kcheck_only = argv[++i];
        }
        else if (a == "--kcheck-stop")
        {
            o.kcheck_stop = true;
        }
        else if (a == "-j")
        {
            // value_int and not atoi, for the reason every other number here has: -j 8x is a typed unit and is
            // refused, where a silent read of 8 would look like a setting.
            if (!value_int(i, argc, argv, "-j", o.threads))
                return false;
            o.threads_given = true;
        }
        else if (a == "--png")
        {
            if (!value_int(i, argc, argv, "--png", o.png))
                return false;
        }
        else if (a == "--check")
        {
            o.check = true;
        }
        else if (a == "--diag")
        {
            o.diag = true;
        }
        else if (a == "--quiet")
        {
            o.quiet = true;
        }
        else if (a == "--no-mkdir")
        {
            o.no_mkdir = true;
        }
        else if (!a.empty() && a[0] == '-')
        {
            fprintf(stderr, "ERROR: unknown flag '%s'\n", a.c_str());
            return false;
        }
        else
        {
            // A positional naming a .json is the MATERIAL rather than an image: it carries the inputs and their
            // per-texture settings, so there can be exactly one of it and it cannot stand beside image arguments.
            if (has_json_extension(a))
            {
                if (!o.json_path.empty())
                {
                    fprintf(stderr, "ERROR: two material JSONs were given ('%s' and '%s'); one run encodes one "
                                    "material\n", o.json_path.c_str(), a.c_str());
                    return false;
                }
                o.json_path = a;
            }
            else
                o.inputs.push_back(a);
        }
    }
    return true;
}

// ---- THE MATERIAL JSON
//
// `nntc_encode material.json [-o ..] [flags]` names the inputs and their per-texture settings in one file instead of on
// the command line, because the command line alone is a burden at four textures and has no per-texture spelling for
// half of what a material needs. It is one array in TEXTURE ORDER; every entry names a `file` and may carry any of the
// settings below. It is read before anything is validated and it fills the input list with resolved paths, so
// everything downstream works exactly as it does when the paths were typed.
//
// An unknown key is an ERROR rather than a shrug. A material file is written by hand, and a misspelled `normal_map`
// that silently did nothing would be found by looking at a chain months later, not by reading the log.
static bool load_material(Options& o)
{
    if (o.json_path.empty())
        return true;
    if (!o.inputs.empty())
    {
        fprintf(stderr, "ERROR: '%s' is a material and names its own inputs, so it cannot stand beside image "
                        "arguments (%zu given)\n", o.json_path.c_str(), o.inputs.size());
        return false;
    }
    o.json_path = tidy_path(o.json_path);   // every message below names it, and a material is named on a command line
    std::string text;
    {
        FILE* f = image_fopen_utf8(o.json_path, "rb");
        if (!f)
        {
            fprintf(stderr, "ERROR: cannot read the material '%s'\n", o.json_path.c_str());
            return false;
        }
        char buffer[4096];
        size_t n = 0;
        while ((n = fread(buffer, 1, sizeof(buffer), f)) > 0)
            text.append(buffer, n);
        fclose(f);
    }
    // A LEADING BYTE-ORDER MARK IS SKIPPED. Several Windows editors write one when they save UTF-8, it is not part of
    // the JSON grammar, and a parser that stops on it has nothing useful to say - the file looks right in every editor
    // the caller owns. It is three bytes and it is the caller's file, not ours, so it is dropped rather than refused.
    if (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB &&
        (unsigned char)text[2] == 0xBF)
        text.erase(0, 3);
    JParser parser(text);
    const JVal root = parser.parse();
    // WHERE it failed, not only that it did. A trailing comma and a byte-order mark used to come back as the same
    // "must be one JSON array" as a file that really is an object, and a hand-written material is exactly the file
    // where a line and a column are worth having.
    if (!parser.ok)
    {
        fprintf(stderr, "ERROR: the material '%s' does not parse: %s (line %zu, column %zu)\n", o.json_path.c_str(),
                parser.error_text, parser.error_line, parser.error_column);
        return false;
    }
    if (root.kind != JVal::ARR)
    {
        fprintf(stderr, "ERROR: the material '%s' must be one JSON array of texture entries, in texture order\n",
                o.json_path.c_str());
        return false;
    }
    if (root.arr.empty())
    {
        fprintf(stderr, "ERROR: the material '%s' is an empty array; a material is one to %d textures\n",
                o.json_path.c_str(), MAX_TEXTURES);
        return false;
    }
    if (root.arr.size() > (size_t)MAX_TEXTURES)
    {
        fprintf(stderr, "ERROR: the material '%s' has %zu entries and a material is one to %d textures\n",
                o.json_path.c_str(), root.arr.size(), MAX_TEXTURES);
        return false;
    }
    const std::filesystem::path dir = std::filesystem::path(o.json_path).parent_path();
    for (size_t t = 0; t < root.arr.size(); t++)
    {
        const JVal& entry = root.arr[t];
        if (entry.kind != JVal::OBJ)
        {
            fprintf(stderr, "ERROR: the material '%s', texture %zu: every entry is an object naming at least a "
                            "'file'\n", o.json_path.c_str(), t);
            return false;
        }
        TextureSettings s;
        std::vector<std::string> seen;
        for (const std::pair<std::string, JVal>& kv : entry.obj)
        {
            const std::string& key = kv.first;
            const JVal& v = kv.second;
            // A REPEATED KEY IS A TYPING ERROR, NOT A CHOICE. JSON does not forbid one and a last-wins reader makes
            // `{"file": "nope.png", "file": "tiny.png"}` encode tiny.png without a word, which is the same class of
            // silence an unknown key is refused for: the entry does not say what its author thinks it says.
            if (std::find(seen.begin(), seen.end(), key) != seen.end())
            {
                fprintf(stderr, "ERROR: the material '%s', texture %zu: '%s' appears twice in the one entry, so the "
                                "entry does not say which of the two values it means\n",
                        o.json_path.c_str(), t, key.c_str());
                return false;
            }
            seen.push_back(key);
            const char* want = nullptr;   // set when the key is known and its value is of the wrong shape
            if (key == "file")
            {
                if (v.kind == JVal::STR && !v.str.empty())
                    s.file = v.str;
                else
                    want = "a non-empty string";
            }
            else if (key == "type")
            {
                if (v.kind == JVal::STR)
                {
                    s.type = v.str;
                    s.src_type = SettingSource::Json;
                }
                else
                    want = "a string";
            }
            else if (key == "filter")
            {
                if (v.kind == JVal::STR)
                {
                    s.filter = v.str;
                    s.src_filter = SettingSource::Json;
                    s.json_filter = true;
                }
                else
                    want = "a string";
            }
            else if (key == "srgb")
            {
                if (v.kind == JVal::BOOL)
                {
                    s.srgb = v.b;
                    s.src_srgb = SettingSource::Json;
                    s.json_srgb = true;
                }
                else
                    want = "true or false";
            }
            else if (key == "edge")
            {
                if (v.kind == JVal::STR)
                {
                    s.edge = v.str;
                    s.src_edge = SettingSource::Json;
                    s.json_edge = true;
                }
                else
                    want = "a string";
            }
            else if (key == "normal_map")
            {
                if (v.kind == JVal::BOOL)
                {
                    s.normal_map = v.b;
                    s.src_normal_map = SettingSource::Json;
                    s.json_normal_map = true;
                }
                else
                    want = "true or false";
            }
            else if (key == "weight")
            {
                if (v.kind == JVal::NUM)
                {
                    s.weight = v.num;
                    s.src_weight = SettingSource::Json;
                    s.json_weight = true;
                }
                else
                    want = "a number";
            }
            else if (key == "rgb_weights")
            {
                bool ok = v.kind == JVal::ARR && v.arr.size() == 3;
                for (size_t c = 0; ok && c < 3; c++)
                    ok = v.arr[c].kind == JVal::NUM;
                if (ok)
                {
                    for (size_t c = 0; c < 3; c++)
                        s.rgb_weights[c] = v.arr[c].num;
                    s.src_rgb_weights = SettingSource::Json;
                    s.json_rgb_weights = true;
                }
                else
                    want = "an array of three numbers";
            }
            else
            {
                fprintf(stderr, "ERROR: the material '%s', texture %zu: '%s' is not a key of a texture entry; the keys "
                                "are file, type, filter, srgb, edge, normal_map, weight and rgb_weights\n",
                        o.json_path.c_str(), t, key.c_str());
                return false;
            }
            if (want)
            {
                fprintf(stderr, "ERROR: the material '%s', texture %zu: '%s' must be %s\n", o.json_path.c_str(), t,
                        key.c_str(), want);
                return false;
            }
        }
        if (s.file.empty())
        {
            fprintf(stderr, "ERROR: the material '%s', texture %zu: no 'file'\n", o.json_path.c_str(), t);
            return false;
        }
        if (!filter_name_ok(s.filter))
        {
            fprintf(stderr, "ERROR: the material '%s', texture %zu: 'filter' is '%s', which is not a filter name; the "
                            "names are default, box, mitchell and catmullrom\n",
                    o.json_path.c_str(), t, s.filter.c_str());
            return false;
        }
        if (s.edge != "clamp" && s.edge != "wrap")
        {
            fprintf(stderr, "ERROR: the material '%s', texture %zu: 'edge' is '%s' and must be clamp or wrap\n",
                    o.json_path.c_str(), t, s.edge.c_str());
            return false;
        }
        // A tangent-space normal map is a direction packed into a byte triple and is not a colour, so filtering it in
        // linear light would be filtering a transfer curve that was never applied. The two keys contradict each other
        // and the entry is refused rather than one of them being chosen for the caller.
        if (s.normal_map && s.srgb)
        {
            fprintf(stderr, "ERROR: the material '%s', texture %zu: 'normal_map' filters in linear light already - a "
                            "packed direction is not a colour - so it cannot also ask for 'srgb'\n",
                    o.json_path.c_str(), t);
            return false;
        }
        // A relative `file` is relative to the MATERIAL, not to the working directory: the two travel together and a
        // material that only works from one directory is a material that mostly does not work.
        std::filesystem::path path(s.file);
        if (path.is_relative() && !dir.empty())
            path = dir / path;
        s.file = tidy_path(path);   // `json/../img/x.png` is the same file and a worse thing to read in an error
        o.tex.push_back(s);
        o.inputs.push_back(s.file);
    }
    return true;
}

// What the report says about the chain a texture was fitted against, in the trailing parenthetical of its psnr line.
static std::string chain_text(const TextureSettings& s)
{
    return "filter " + s.filter + ", srgb " + (s.srgb ? "yes" : "no") + ", normal map " +
           (s.normal_map ? "yes" : "no");
}

// A per-texture list is either EMPTY - every texture takes the default - or exactly one entry per texture. Anything
// else is a typing error rather than an intention, and it is refused with both counts in the message, because the one
// thing the caller cannot see from a wrong count alone is which of the two numbers they meant.
static bool list_count_ok(const char* flag, size_t given, size_t textures)
{
    if (given == 0 || given == textures)
        return true;
    fprintf(stderr, "ERROR: %s gives %zu value%s for %zu texture%s; it takes exactly one entry per texture, or none at "
                    "all, in which case every texture takes the default\n",
            flag, given, given == 1 ? "" : "s", textures, textures == 1 ? "" : "s");
    return false;
}

// The per-texture settings as they stand once the command line has had its say over whatever the material JSON left
// behind. Everything downstream reads o.tex: the objective's channel weights, the source chain and the settings rows.
static bool merge_texture_settings(Options& o)
{
    if (o.tex.empty())
    {
        o.tex.assign(o.inputs.size(), TextureSettings());
        for (size_t t = 0; t < o.inputs.size(); t++)
            o.tex[t].file = o.inputs[t];
    }
    if (!list_count_ok("--weights", o.weights.size(), o.inputs.size()) ||
        !list_count_ok("--mip-filter", o.mip_filter.size(), o.inputs.size()))
        return false;
    for (const std::string& name : o.mip_filter)
        if (!filter_name_ok(name))
        {
            fprintf(stderr, "ERROR: --mip-filter '%s' is not a filter name; the names are default, box, mitchell and "
                            "catmullrom\n", name.c_str());
            return false;
        }
    // --rgb-weights stays ONE triple over the whole command line; a per-texture triple is a material JSON's business.
    for (size_t t = 0; t < o.tex.size(); t++)
    {
        TextureSettings& s = o.tex[t];
        const std::string name = basename_of_path(s.file);
        // THE COMMAND LINE WINS, and says so. A material that set the key and a flag that replaces it are two people
        // describing one texture, and the one whose answer was thrown away is the one worth naming. Equal values warn
        // too: what the warning reports is that the material's key was overridden, not that the number changed.
        if (!o.weights.empty())
        {
            if (s.json_weight)
                printf("WARNING: texture %zu (%s): the material sets weight %g and --weights sets %g; the command "
                       "line wins\n", t, name.c_str(), s.weight, o.weights[t]);
            s.weight = o.weights[t];
            s.src_weight = SettingSource::CommandLine;
        }
        if (!o.mip_filter.empty())
        {
            if (s.json_filter)
                printf("WARNING: texture %zu (%s): the material sets filter %s and --mip-filter sets %s; the command "
                       "line wins\n", t, name.c_str(), s.filter.c_str(), o.mip_filter[t].c_str());
            s.filter = o.mip_filter[t];
            s.src_filter = SettingSource::CommandLine;
        }
        if (o.rgb_weights_given)
        {
            if (s.json_rgb_weights)
                printf("WARNING: texture %zu (%s): the material sets rgb weights %g %g %g and --rgb-weights sets "
                       "%g %g %g; the command line wins\n", t, name.c_str(), s.rgb_weights[0], s.rgb_weights[1],
                       s.rgb_weights[2], o.rgb_weights[0], o.rgb_weights[1], o.rgb_weights[2]);
            for (int c = 0; c < 3; c++)
                s.rgb_weights[c] = o.rgb_weights[(size_t)c];
            s.src_rgb_weights = SettingSource::CommandLine;
        }
    }
    // srgb, edge and normal_map all say how a DERIVED chain is derived, and `default` derives nothing: it is the
    // iterated 2x2 box the tree has always built. A texture asks for a derivation by VALUE and not by naming a key:
    // `srgb` true, `edge` wrap, `normal_map` true. A key carrying its inert value - "srgb": false, "edge": "clamp",
    // "normal_map": false - asks for nothing, so it neither implies a filter nor contradicts one whatever the filter
    // is; the settings row still prints the value and says it came from the json. When a texture does ask, what
    // happens turns on whether anybody actually ASKED for `default`:
    //
    //   * nobody did - the filter is still the program's own default, source Default - and the key is then the only
    //     thing said about the chain, so it is honoured: the texture is switched to `box`, the resizer's own box,
    //     which re-derives every level DIRECTLY FROM THE BASE. That is the same picture as the built-in iterated box
    //     only where every level is an exact halving: wherever an odd plane is halved the two cover different
    //     footprints and the deeper planes really do differ (docs/DESIGN.md 4.2, the footprint paragraph). What the
    //     switch buys is that srgb, the edge mode and the renormalisation now have a derivation to apply to;
    //   * somebody did, in the JSON's "filter" or in --mip-filter, and then the two statements contradict each other.
    //     A contradiction is refused, and the message says when the command line was the half that caused it.
    for (size_t t = 0; t < o.tex.size(); t++)
    {
        TextureSettings& s = o.tex[t];
        if (s.filter != "default")
            continue;
        const char* key = s.srgb ? "srgb" : (s.edge == "wrap" ? "edge" : (s.normal_map ? "normal_map" : nullptr));
        if (!key)
            continue;
        if (s.src_filter == SettingSource::Default)
        {
            s.filter = "box";
            s.filter_implied_by = key;
            continue;
        }
        fprintf(stderr, "ERROR: texture %zu sets '%s', which only a derived source chain can honour, and its filter is "
                        "'default', the iterated 2x2 box, which derives nothing%s\n",
                t, key,
                s.src_filter != SettingSource::CommandLine
                    ? "; that filter was asked for by the material itself, so leave 'filter' out and it becomes 'box'"
                    // Only say something was replaced when there was something to replace: a material that named no
                    // filter of its own had nothing overridden, and the caller is looking for their own --mip-filter.
                    : (s.json_filter ? "; that filter came from --mip-filter, which replaced the material's"
                                     : "; that filter came from --mip-filter, so drop the flag and it becomes 'box'"));
        return false;
    }
    // The two conditions on a weight, now on the MERGED value so that a JSON entry is held to them exactly as a flag
    // is: a negative weight makes diag(cw) indefinite, so block (a)'s normal equations no longer minimise anything and
    // the run is silently wrong, and an all-zero set divides E by a sum of zero and reports nan.
    double total = 0.0;
    for (size_t t = 0; t < o.tex.size(); t++)
    {
        TextureSettings& s = o.tex[t];
        if (!(s.weight >= 0.0))
        {
            fprintf(stderr, "ERROR: --weights must not be negative\n");
            return false;
        }
        // AND IT MUST BE A NUMBER THE OBJECTIVE CAN CARRY. The per-output weight cw is a FLOAT, and a weight of 1e300
        // is a perfectly good double that becomes an infinity there: every product in E is then a nan, every round
        // line prints psnr 100.00, and a garbage asset is written with an exit code of 0. A weight is a RATIO between
        // textures and nothing in a material is a hundred times another texture, so the sane range is [0, WEIGHT_MAX]
        // and a value outside it is CLAMPED, not refused (the owner's call): the run goes on with the clamped value,
        // the settings rows show it, and one WARNING names the texture, the value and what it became. The low end is
        // the same story in the other direction: a positive weight so small that the float holds it as zero would
        // drop a texture the caller asked for out of E, so a positive weight below WEIGHT_MIN is raised to it.
        clamp_weight(t, "weight", s.weight);
        total += s.weight;
        for (int c = 0; c < 3; c++)
        {
            if (!(s.rgb_weights[c] >= 0.0))
            {
                fprintf(stderr, "ERROR: --rgb-weights must not be negative\n");
                return false;
            }
            clamp_weight(t, c == 0 ? "red weight" : c == 1 ? "green weight" : "blue weight", s.rgb_weights[c]);
        }
        if (!(s.rgb_weights[0] + s.rgb_weights[1] + s.rgb_weights[2] > 0.0))
        {
            fprintf(stderr, "ERROR: --rgb-weights must not be all zero\n");
            return false;
        }
    }
    if (!(total > 0.0))
    {
        fprintf(stderr, "ERROR: --weights must not be all zero: E would have nothing to measure\n");
        return false;
    }
    // A filter, srgb, an edge mode and normal_map describe how the levels BELOW the base are derived, and a run with
    // --mips 0, or a source too small for --mip-min to leave a level below the base, derives none. That is NOT refused:
    // the filter is a setting and the chain is the encoder's decision from the image's size, so a texture that gets no
    // mipmaps simply gets no mipmaps and the setting waits for a source that has some (the owner's rule).
    return true;
}

static bool validate_options(Options& o)
{
    if (o.inputs.empty() || o.inputs.size() > MAX_TEXTURES)
    {
        fprintf(stderr, "ERROR: one to %d input images are needed (%zu given)\n", MAX_TEXTURES, o.inputs.size());
        return false;
    }
    // -o is optional. Without it the asset is written beside the first input under that input's base name. With it:
    //
    //   a trailing separator, or a path that already IS a directory   a directory; the base name is written inside it
    //   a bare name with no extension and no file of that name        a DIRECTORY, created here
    //   a name ending in .json                                        THE DESCRIPTOR itself; the prefix is that name
    //                                                                 without the extension
    //   anything else (a name with an extension, or an existing file) the prefix itself, extension and all
    //
    // The middle case is the one the reviews found ambiguous - `-o out` wrote out_lat0.dds beside out/ - and it is
    // resolved towards the directory, which is what the help line has always promised. An explicit prefix that has no
    // extension can still be asked for with a trailing separator plus a name, or by giving the name an extension.
    //
    // The descriptor is PREFIX_nntc.json, beside PREFIX_lat0.dds and PREFIX_lat1.dds, and NOT PREFIX.json: a prefix
    // taken from an input's stem then cannot name the input itself, whatever extension the input carries. The .json
    // spelling of -o is the way to ask for a descriptor of one's own choosing.
    {
        // With a material JSON the asset is named after the MATERIAL, not after its first texture: `-o dir/` on
        // material.json writes dir/material_lat0.dds. A material is one thing with one name, and naming the asset
        // after whichever texture happens to be listed first would make the name an accident of the file's order.
        const std::filesystem::path first(o.json_path.empty() ? o.inputs[0] : o.json_path);
        const std::string base = first.stem().string();
        const bool prefix_given = !o.prefix.empty();
        bool bare_name_prefix = false;   // -o out/name, with no extension and no file of that name: a directory
        // With no -o the asset lands in the CURRENT DIRECTORY under the input's (or the material's) stem, the way a
        // command-line tool is expected to behave; it used to land beside the input, which surprised the owner.
        if (!prefix_given)
            o.prefix = base;
        else
        {
            const char last = o.prefix[o.prefix.size() - 1];
            std::error_code ec;
            const std::filesystem::path given(o.prefix);
            const bool named_directory = last == '/' || last == '\\';
            const bool is_directory = std::filesystem::is_directory(given, ec);
            const bool bare_name = !named_directory && !given.has_extension() &&
                                   !std::filesystem::exists(given, ec);
            bare_name_prefix = bare_name;
            if (named_directory || is_directory || bare_name)
                o.prefix = (given / base).string();
            else if (has_json_extension(given))
            {
                // The caller named the descriptor outright. It is written under exactly that name, and the .dds files
                // take the same name without the extension, so `-o out/name.json` gives out/name.json beside
                // out/name_lat0.dds. Nothing else in the file derives a descriptor name from the prefix.
                o.desc = o.prefix;
                o.prefix = (given.parent_path() / given.stem()).string();
            }
        }
        if (o.desc.empty())
            o.desc = o.prefix + "_nntc.json";
        // THE ASSET'S DESCRIPTOR MUST NOT BE THE MATERIAL ITSELF. The _nntc suffix is what keeps that from happening by
        // accident: with a material the prefix is the material's own stem in the material's own directory, and a
        // descriptor of PREFIX.json would have been written straight over the material that described the run. The
        // check stays as the backstop for the spelling that can still collide - `-o material.json`, and a material
        // that is itself named NAME_nntc.json - and it refuses rather than overwrites. The file the caller wrote by
        // hand would otherwise be gone, and the only copy of which images the material named with it.
        //
        // It is refused HERE, before the output directory is created and long before anything is loaded, so a refused
        // run leaves the tree exactly as it found it.
        if (!o.json_path.empty() && same_file_path(std::filesystem::path(o.desc),
                                                   std::filesystem::path(o.json_path)))
        {
            fprintf(stderr, "ERROR: the asset's descriptor would be written to '%s', which is the material '%s' "
                            "itself, and the material would be overwritten by it. Give -o a different directory (-o "
                            "out/), a different prefix (-o out/name) or a different descriptor (-o out/name.json).\n",
                    tidy_path(o.desc).c_str(), tidy_path(o.json_path).c_str());
            return false;
        }
        // WITHOUT -o there is no output directory to make: the prefix is the input's stem in the current directory,
        // which exists. Nothing is created and nothing is refused here, so a missing input's first complaint is the
        // one that fits: it cannot read the input.
        if (prefix_given)
        {
            const std::filesystem::path dir = std::filesystem::path(o.prefix).parent_path();
            std::error_code ec;
            // The directory is made by default, because -o out/ in a script should not need a mkdir in front of it.
            // Under --no-mkdir a directory that is not there is a mistyped -o rather than a directory to create, and
            // saying so here - before an image is read, let alone a file written - leaves the tree as the run found
            // it. A bare name is the spelling that surprises, so the refusal says which reading it took.
            if (!dir.empty() && !std::filesystem::exists(dir, ec) && o.no_mkdir)
            {
                fprintf(stderr, "ERROR: the output directory '%s' does not exist and --no-mkdir forbids creating it%s\n",
                        tidy_path(dir).c_str(),
                        bare_name_prefix ? " (a bare name is a directory; add an extension or a trailing separator "
                                           "for a prefix)"
                                         : "");
                return false;
            }
            if (!dir.empty() && !std::filesystem::exists(dir, ec) && !std::filesystem::create_directories(dir, ec))
            {
                fprintf(stderr, "ERROR: cannot create the output directory '%s'\n", tidy_path(dir).c_str());
                return false;
            }
        }
    }
    if (o.c0 < 1 || o.c0 > 4)
    {
        fprintf(stderr, "ERROR: --c0 %d is out of range (1..4)\n", o.c0);
        return false;
    }
    if (o.bc0 != 0 && o.bc0 != 1)
    {
        fprintf(stderr, "ERROR: --bc0 %d must be 0 or 1\n", o.bc0);
        return false;
    }
    if (o.l0 != "palette" && o.l0 != "bc8")
    {
        fprintf(stderr, "ERROR: --l0 '%s' must be bc8 or palette\n", o.l0.c_str());
        return false;
    }
    if (o.bc_refine < 0 || o.bc_refine_after < 0 || o.bc_outer < 0)
    {
        fprintf(stderr, "ERROR: --bc-refine, --bc-refine-after and --bc-outer count passes and cannot be negative\n");
        return false;
    }
    // The three pack flags belong to the BC pack, which only --l0 bc8 makes; naming one in the palette mode is refused
    // rather than ignored, for the same reason --bits0 is refused under bc8.
    if (o.l0 != "bc8" && (o.bc_refine_given || o.bc_refine_after_given || o.bc_outer_given))
    {
        fprintf(stderr, "ERROR: --bc-refine, --bc-refine-after and --bc-outer belong to --l0 bc8, whose pack they "
                        "refine; under --l0 palette the pack is lossless at 1-3 bits and a fixed nearest-palette "
                        "choice at 4\n");
        return false;
    }
    // --l0 bc8 fixes two of the other flags rather than quietly overriding them, because a run whose --bits0 was
    // ignored and a run whose --bc0 0 was ignored would both look like results.
    if (o.l0 == "bc8")
    {
        if (o.bits0_given)
        {
            fprintf(stderr, "ERROR: --bits0 does not apply under --l0 bc8: level 0 is a continuous plane at 8 bits, "
                            "which is what a BC4 / BC5 block carries\n");
            return false;
        }
        if (o.bc0 == 0)
        {
            fprintf(stderr, "ERROR: --bc0 0 does not apply under --l0 bc8: the whole point of the mode is the BC4 / "
                            "BC5 pack, and an uncompressed 8-bit level 0 costs twice what the palette mode does. "
                            "(--bc0 both is allowed: it writes the pre-pack plane beside the asset.)\n");
            return false;
        }
        o.bits0.assign((size_t)o.c0, 8);
    }
    if (o.bits0.size() == 1)
    {
        const int b = o.bits0[0];   // a copy: assign() may reallocate before it reads its own element
        o.bits0.assign((size_t)o.c0, b);
    }
    if ((int)o.bits0.size() != o.c0)
    {
        fprintf(stderr, "ERROR: --bits0 gives %zu values for %d level-0 channels\n", o.bits0.size(), o.c0);
        return false;
    }
    if (o.l0 != "bc8")
        for (int b : o.bits0)
            if (b < 1 || b > 4)
            {
                fprintf(stderr, "ERROR: --bits0 %d is out of range (1..4)\n", b);
                return false;
            }
    if (o.c1 < 1 || o.c1 > 4)
    {
        fprintf(stderr, "ERROR: --c1 %d is out of range (1..4)\n", o.c1);
        return false;
    }
    if (o.bits1 < 4 || o.bits1 > 8)
    {
        fprintf(stderr, "ERROR: --bits1 %d is out of range (4..8)\n", o.bits1);
        return false;
    }
    if (o.mip_min < 4)
    {
        fprintf(stderr, "ERROR: --mip-min %d is below the minimum of 4\n", o.mip_min);
        return false;
    }
    if (o.mips != 0 && o.mips != 1)
    {
        fprintf(stderr, "ERROR: --mips %d must be 0 or 1\n", o.mips);
        return false;
    }
    if (o.k != 0 && o.k != 1 && o.k != 2 && o.k != 4)
    {
        fprintf(stderr, "ERROR: --k %d must be 0, 1, 2 or 4\n", o.k);
        return false;
    }
    if (o.rounds < 1)
    {
        fprintf(stderr, "ERROR: --rounds %d must be at least 1\n", o.rounds);
        return false;
    }
    if (!(o.tol >= 0.0))
    {
        fprintf(stderr, "ERROR: --tol must not be negative\n");
        return false;
    }
    if (o.q1_start < 0)
    {
        fprintf(stderr, "ERROR: --q1-start %d must not be negative\n", o.q1_start);
        return false;
    }
    if (o.q1_sweeps < 1)
    {
        fprintf(stderr, "ERROR: --q1-sweeps %d must be at least 1\n", o.q1_sweeps);
        return false;
    }
    if (o.q1_range != "minmax" && o.q1_range != "pct")
    {
        fprintf(stderr, "ERROR: --q1-range '%s' must be minmax or pct\n", o.q1_range.c_str());
        return false;
    }
    if (o.init0 != "residual" && o.init0 != "texture" && o.init0 != "luma")
    {
        fprintf(stderr, "ERROR: --init0 '%s' must be residual, texture or luma\n", o.init0.c_str());
        return false;
    }
    if (o.init0_scope != "base" && o.init0_scope != "chain")
    {
        fprintf(stderr, "ERROR: --init0-scope '%s' must be base or chain\n", o.init0_scope.c_str());
        return false;
    }
    // --init0 luma seeds no direction at all, so there is no residual for a scope to be taken over. Naming one is
    // refused rather than ignored, for the reason every other combination in this function is: a run whose flag did
    // nothing would look like a result.
    if (o.init0 == "luma" && o.init0_scope != "base")
    {
        fprintf(stderr, "ERROR: --init0-scope belongs to --init0 residual and --init0 texture, which read a residual; "
                        "--init0 luma seeds channel 0 from the source luminance and takes no residual at all\n");
        return false;
    }
    if (o.init != "pca" && o.init != "box")
    {
        fprintf(stderr, "ERROR: --init '%s' must be pca or box\n", o.init.c_str());
        return false;
    }
    if (o.rgb_weights.size() != 3)
    {
        fprintf(stderr, "ERROR: --rgb-weights needs exactly three values\n");
        return false;
    }
    if (o.mip_weight != "sqrt" && o.mip_weight != "pixels" && o.mip_weight != "uniform")
    {
        fprintf(stderr, "ERROR: --mip-weight '%s' must be sqrt, pixels or uniform\n", o.mip_weight.c_str());
        return false;
    }
    if (!(o.mip_mix >= 0.0 && o.mip_mix <= 1.0))
    {
        fprintf(stderr, "ERROR: --mip-mix must be in [0, 1]\n");
        return false;
    }
    // The base plane's share of block (a) is 1 - mix, so a mix of exactly 1 with a chain stored takes the base out of
    // the decoder's least squares entirely: the one shared W would be fitted to the chain alone and the base level -
    // the image - would be decoded by a decoder that never saw it.
    if (o.mip_mix >= 1.0 && o.mips != 0)
    {
        fprintf(stderr, "ERROR: --mip-mix 1 leaves the base plane out of block (a) altogether; use --mips 0 to fit one "
                        "level per texture, or a mix below 1\n");
        return false;
    }
    if (!(o.ridge >= 0.0))
    {
        fprintf(stderr, "ERROR: --ridge must not be negative\n");
        return false;
    }
    if (o.sweeps < 1)
    {
        fprintf(stderr, "ERROR: --sweeps %d must be at least 1\n", o.sweeps);
        return false;
    }
    if (o.png != 0 && o.png != 1)
    {
        fprintf(stderr, "ERROR: --png %d must be 0 or 1\n", o.png);
        return false;
    }
    // The backend's NAME is checked here; whether the one named can actually run is a question about the machine and
    // is answered later, by select_backend, where the fallback lives.
    if (o.backend != "auto" && o.backend != "cuda" && o.backend != "cpu" && o.backend != "check")
    {
        fprintf(stderr, "ERROR: --backend '%s' must be auto, cuda, cpu or check\n", o.backend.c_str());
        return false;
    }
    // 0 is the default and means "as many as the machine has"; a negative count is a typed mistake, and the upper end
    // is where a thread count stops being a thread count. The clamp itself lives in the dispatcher, which is what
    // resolves the 0.
    if (o.threads < 0 || o.threads > 1024)
    {
        fprintf(stderr, "ERROR: -j %d must be 0 (as many as the machine has) or a count in 1..1024\n", o.threads);
        return false;
    }
    return merge_texture_settings(o);
}

// WHICH BACKEND RUNS, and what happens when the one asked for cannot (docs/CPU_BACKEND_PLAN.md 2.3).
//
// CUDA can be unavailable in five ways: a binary built without the CUDA arm at all, no NVIDIA device, a driver that
// will not initialise, a --device N past the end, and a device older than the build's compute capability floor. The
// last four all arrive here as device_select returning false (the last with the device's name and capability left in
// info, so the message can say which); under `auto` that is a WARNING and a fallback, and under `cuda` or `check` an
// ERROR, because a caller who demanded a GPU is owed a refusal rather than something else that ran.
//
// False means the run is over and the ERROR has already been printed. The chosen backend is handed to the dispatcher
// before anything touches a device, which is what the six handle-less seam functions read.
// Why device_select refused: a device that answered but is older than the build's floor, or no device at all.
static std::string cuda_refusal(const Options& o, const DeviceInfo& info)
{
    char why[512];
    if (info.major > 0)
        snprintf(why, sizeof(why), "CUDA device %d is '%s', compute capability %d.%d, and this encoder is compiled for "
                 "8.0 and above", o.device, info.name.c_str(), info.major, info.minor);
    else if (!info.error.empty())
        snprintf(why, sizeof(why), "%s", info.error.c_str());
    else
        snprintf(why, sizeof(why), "CUDA device %d could not be used", o.device);
    return why;
}

static bool select_backend(const Options& o, DeviceInfo& info)
{
    // Two flags that belong to the other backend. Neither is a refusal: naming a GPU index and a CPU backend in one
    // command line is a mistake worth saying out loud and not a reason to throw an encode away.
    if (o.backend == "cpu" && o.device_given)
        printf("WARNING: --device %d names a CUDA device and --backend cpu does not use one: the flag is idle\n",
               o.device);
    if (o.backend == "cuda" && o.threads_given)
        printf("WARNING: -j %d is the cpu backend's worker count and --backend cuda does not use it: the flag is "
               "idle\n", o.threads);
    if (o.backend != "check" && (!o.kcheck_only.empty() || o.kcheck_stop))
        printf("WARNING: --kcheck-only and --kcheck-stop belong to --backend check, which this run is not: the flags "
               "are idle\n");

    // The harness needs both arms at once, so it is asked for the way cuda is: by name, and refused where it cannot
    // run, never quietly replaced by something else.
    if (o.backend == "check")
    {
        if (!be::cuda_built_in())
        {
            fprintf(stderr, "ERROR: --backend check was asked for, but this build has no CUDA backend (it was "
                            "configured without a CUDA compiler), and the harness compares the two; re-configure "
                            "with a CUDA 12.8+ toolset\n");
            return false;
        }
        be::backend_select(be::Backend::Check, o.threads);
        be::check_options(o.kcheck_only, o.kcheck_stop, o.quiet);
        if (be::device_select(o.device, info))
            return true;
        fprintf(stderr, "ERROR: --backend check was asked for, but %s; the harness runs the encode on CUDA\n",
                cuda_refusal(o, info).c_str());
        return false;
    }

    if (o.backend == "cpu")
    {
        be::backend_select(be::Backend::Cpu, o.threads);
        return be::device_select(o.device, info);
    }

    // auto and cuda both try CUDA, and differ only in what they do when it is not there.
    if (!be::cuda_built_in())
    {
        if (o.backend == "cuda")
        {
            fprintf(stderr, "ERROR: --backend cuda was asked for, but this build has no CUDA backend (it was "
                            "configured without a CUDA compiler); re-configure with a CUDA 12.8+ toolset, or run "
                            "--backend cpu\n");
            return false;
        }
        printf("WARNING: this build has no CUDA backend (it was configured without a CUDA compiler), so --backend "
               "auto falls back to cpu\n");
        be::backend_select(be::Backend::Cpu, o.threads);
        return be::device_select(o.device, info);
    }

    be::backend_select(be::Backend::Cuda, o.threads);
    if (be::device_select(o.device, info))
        return true;

    const std::string why = cuda_refusal(o, info);
    info = DeviceInfo();
    if (o.backend == "cuda")
    {
        // Host memory is short for the CPU backend too, so pointing at it would be advice that cannot help.
        if (why.find("out of host memory") != std::string::npos)
            fprintf(stderr, "ERROR: --backend cuda was asked for, but %s\n", why.c_str());
        else
            fprintf(stderr, "ERROR: --backend cuda was asked for, but %s; run --backend cpu, or --backend auto which "
                            "falls back to it\n", why.c_str());
        return false;
    }
    printf("WARNING: %s, so --backend auto falls back to cpu\n", why.c_str());
    be::backend_select(be::Backend::Cpu, o.threads);
    return be::device_select(o.device, info);
}

// The banner: the layout, the per-texture settings, the planes, the device and what the loop is about to do. It is
// printed once, before the first round, and --quiet is the one thing that turns it off.
static void print_banner(const Options& o, const Model& m, const DeviceInfo& info)
{
    printf("nntc_encode: %zu texture%s, %dx%d source", o.inputs.size(), o.inputs.size() == 1 ? "" : "s", m.source_width,
           m.source_height);
    if (m.padded)
        printf(" padded to %dx%d", m.width, m.height);
    printf(", nout %d\n", m.nout);
    // THE PER-TEXTURE SETTINGS, and where each of them came from. A material can be described by a JSON, by the
    // command line, or by neither, and the three are merged before anything is loaded; printing the merged value
    // without its origin would leave the one question a reader of a log actually has - which of the two spoke -
    // answerable only by reading the command line and the JSON side by side.
    if (!o.json_path.empty())
        printf("  material %s\n", o.json_path.c_str());
    for (size_t t = 0; t < o.tex.size(); t++)
    {
        const TextureSettings& s = o.tex[t];
        printf("    t%zu %s  type '%s' (%s)  weight %g (%s)  rgb weights %g %g %g (%s)\n", t,
               basename_of_path(s.file).c_str(), s.type.c_str(), setting_source_text(s.src_type), s.weight,
               setting_source_text(s.src_weight), s.rgb_weights[0], s.rgb_weights[1], s.rgb_weights[2],
               setting_source_text(s.src_rgb_weights));
        // The filter's origin is the one that can be neither of the three sources: a box nobody typed, implied by the
        // first of srgb / edge / normal_map the texture set. Saying "default" there would name the value it is not.
        const std::string filter_src = s.filter_implied_by.empty() ? std::string(setting_source_text(s.src_filter))
                                                                   : "implied by " + s.filter_implied_by;
        printf("       filter %s (%s)  srgb %s (%s)  edge %s (%s)  normal map %s (%s)\n", s.filter.c_str(),
               filter_src.c_str(), s.srgb ? "yes" : "no", setting_source_text(s.src_srgb),
               s.edge.c_str(), setting_source_text(s.src_edge), s.normal_map ? "yes" : "no",
               setting_source_text(s.src_normal_map));
    }
    printf("  output : %s (descriptor %s), --png %d, --mip-min %d\n", o.prefix.c_str(),
           basename_of_path(o.desc).c_str(), o.png, o.mip_min);
    printf("  level 0: %d channel%s, bits", m.c0, m.c0 == 1 ? "" : "s");
    for (int c = 0; c < m.c0; c++)
        printf(" %d", m.bits0[c]);
    printf(m.l0_bc8 ? ", a continuous plane on a per-channel grid (--l0 bc8)\n" : ", palette on [-1,1]\n");
    if (m.l0_bc8)
    {
        printf("           written as %s, the standard k / 7 palette\n",
               level0_files_text(m.c0));
        printf("           solved as a CONTINUOUS plane by the same sparse least squares level 1 is solved by, on its\n");
        printf("           own per-channel lo/hi grid frozen at --q1-start %d, then BC4 / BC5-encoded after the last\n",
               o.q1_start);
        printf("           round and level 1 and the decoder refitted against the packed plane. The pack is then\n");
        printf("           REFINED block by block (--bc-refine %d): the endpoints by least squares on the decoder's\n",
               o.bc_refine);
        printf("           own error and the selectors by exact argmin, a candidate taken only when that error falls,\n");
        printf("           and %d outer repack pass%s (--bc-outer) taken only if the shipped E falls.\n", o.bc_outer,
               o.bc_outer == 1 ? "" : "es");
    }
    else if (o.bc0)
    {
        printf("           written as %s, the standard k / 7 palette (lossless at 1-3 bits)\n",
               level0_files_text(m.c0));
        printf("           fitted on pal(k) = -1 + 2 k / (2^bits - 1), which is what that palette decodes to\n");
    }
    else
    {
        printf("           written as an uncompressed plane (--bc0 0)\n");
        printf("           fitted on pal(k) = -1 + 2 rep(k) / 255, the replicated byte a UNORM sampler returns\n");
    }
    printf("  level 1: %d channel%s, %d bits, 1/%d resolution per axis (LOD bias %d)\n", m.c1, m.c1 == 1 ? "" : "s",
           m.bits1, BLOCK, LOD_BIAS_LEVEL1);
    printf("  decoder: terms 'a b sc', nin %d, nout %d, %d weights\n", m.dec.nin, m.nout,
           m.dec.nin * m.nout + m.nout);
    printf("  planes (%zu stored level%s):\n", m.planes.size(), m.planes.size() == 1 ? "" : "s");
    for (size_t i = 0; i < m.planes.size(); i++)
        printf("    M%zu  level 0 %dx%d   level 1 %dx%d\n", i, m.planes[i].w0, m.planes[i].h0, m.planes[i].w1,
               m.planes[i].h1);
    // WHICH BACKEND IS ABOUT TO RUN, on every run and not only on a CPU one, so that a log answers the question
    // outright rather than by the absence of a line. The CUDA device row below it is exactly the row it always was.
    printf("  backend: %s\n", be::backend_name(be::backend_selected()));
    if (be::backend_selected() != be::Backend::Cpu)
        printf("  device %d: %s, compute capability %d.%d\n", o.device, info.name.c_str(), info.major, info.minor);
    if (be::backend_selected() == be::Backend::Cpu)
        printf("  workers: %d of the %u this machine reports, on %s\n", be::backend_threads(),
               std::thread::hardware_concurrency(), info.name.c_str());
    else if (be::backend_selected() == be::Backend::Check)
        printf("  workers: %d of the %u this machine reports, for the cpu side of the harness\n",
               be::backend_threads(), std::thread::hardware_concurrency());
    // What the CPU backend is about to take from host memory, said BEFORE it takes it (docs/CPU_BACKEND_PLAN.md
    // section 1.11): it is a pure function of the layout, and a run that will not fit should say so up front rather
    // than in the report of a run that never got there. Only on the CPU's rows, so the CUDA banner is the one it was.
    if (be::backend_selected() != be::Backend::Cuda)
        printf("  memory : %.1f MB of host memory for the cpu backend's model\n",
               (double)be::cpu_bytes_for(m) / (1024.0 * 1024.0));
    printf("  sites  : the centre of every pixel of every plane, plus --k %d fixed subtexel positions per pixel\n", o.k);
    printf("           (both levels bilinear there, the target the source read the same way)\n");
    printf("  loop   : up to --rounds %d rounds of (a) the decoder's least squares over every site of every plane,\n",
           o.rounds);
    printf("           (b) level 1 by the ridged stencil solve (--ridge %g) on EVERY plane, %s\n", o.ridge,
           m.l0_bc8 ? "(c') level 0 by that SAME"
                    : "(c) level 0's exact");
    printf("           %s; stopping at a relative drop in E below\n",
           m.l0_bc8 ? "solve over its own plane on EVERY plane (the palette search is not run)"
                    : "per-texel search in four colour passes on EVERY plane");
    printf("           --tol %g twice running, or when nothing moves. Each plane's share of block (a) is\n", o.tol);
    printf("           --mip-weight %s / --mip-mix %g%s.\n", o.mip_weight.c_str(), o.mip_mix,
           o.mip_weight == "pixels" ? " (pixels is the default: every plane carries the same per-sample weight, which "
                                      "measured better at every level than sqrt on both ablation images)"
                                    : "");
    if (o.q1_start > 0)
    {
        printf("  grid   : level 1 is continuous until round --q1-start %d, where its per-channel lo/hi are fitted\n",
               o.q1_start);
        printf("           (--q1-range %s) over the base AND every chain plane together - the format carries one\n",
               o.q1_range.c_str());
        printf("           lo/hi pair per channel for the whole chain - and FROZEN; from then on block (b) is\n");
        printf("           --q1-sweeps %d four-colour Gauss-Seidel sweeps per plane whose per-value step is the exact\n",
               o.q1_sweeps);
        printf("           argmin over that grid, so every plane is storable from the freeze onwards and nothing is\n");
        printf("           rounded at the end.\n");
    }
    else
    {
        printf("  grid   : --q1-start 0, the control: level 1 stays continuous through the whole loop and is snapped\n");
        printf("           to its grid (--q1-range %s) once, at the end.\n", o.q1_range.c_str());
    }
    printf("           The run always ends with block (b) on every plane, then %s on every plane against\n",
           m.l0_bc8 ? "block (c')" : "a level-0 search");
    printf("           the values level 1 now holds, then a last block (a), so that W is optimal for what is stored.\n");
    if (m.l0_bc8)
        printf("           Then level 0 is packed, the pack refined block by block and repacked in outer passes,"
               " and\n           level 1 and the decoder refitted against it: the asset ships OPTIMISED BC blocks.\n");
    if (m.init0_residual)
    {
        printf("  init   : level 0 --init0 %s: each channel in turn is the leading principal direction of the\n",
               o.init0.c_str());
        printf("           residual r = out - target%s, projected onto that direction and\n",
               o.init0 == "texture" ? " OF ONE TEXTURE - the one whose own residual carries the most"
                                    : " of the whole material");
        printf("           %sscaled to fill [-1,1], the decoder refitted after each so the next sees what is left\n",
               o.init0 == "texture" ? "weighted variance - " : "");
        if (!m.l0_bc8)
            printf("           the palette mode divides the projection by its %.1f %% percentile, not its peak, and\n"
                   "           follows each seeded channel with one level-0 search and one refit, so the snap is\n"
                   "           chosen by the objective (src/init.cu 4a', docs/DESIGN.md 3.4)\n",
                   100.0 * INIT0_PERCENTILE);
        printf("           --init0-scope %s: that residual is measured %s\n", o.init0_scope.c_str(),
               o.init0_scope == "chain" ? "over EVERY stored plane, each site carrying the weight the objective "
                                          "gives it"
                                        : "over the base plane alone (the control)");
        printf("           level 1 --init %s\n", o.init.c_str());
    }
    else
    {
        printf("  init   : level 0 --init0 luma: the %ssource luminance, every further channel at zero;"
               " level 1 --init %s\n", m.l0_bc8 ? "" : "snapped ", o.init.c_str());
    }
    printf("  the 1:1 rule: plane m of BOTH latents is fitted against level m of the run's OWN source chain (the\n");
    printf("         settings rows above say how each texture's was derived), so blocks (b)\n");
    printf("         and (c) run on every plane of every round, each plane on its own dimensions, its own site set and\n");
    printf("         its own source mip. With the decoder held, the planes share nothing and are independent problems;\n");
    printf("         the only coupling between levels is W itself, which block (a) fits over every site of every plane\n");
    printf("         with the mip weights above. On the GPU the same rule is a shift of level 1's LOD by %d mips - its UV\n", LOD_BIAS_LEVEL1);
    printf("         gradients scaled by %d, or a sampler LOD bias of %d where the hardware handles one correctly (README.md) -\n",
           1 << LOD_BIAS_LEVEL1, LOD_BIAS_LEVEL1);
    printf("         without which the decoder is fed a colour plane %d mips too fine from mip 1 down.\n", LOD_BIAS_LEVEL1);
    fflush(stdout);
}

// What the encode is doing, for the one message that cannot know it otherwise: main's catch of an exception that
// escaped, a std::bad_alloc above all. A stage name, not a place in the code; set as each stage begins.
static const char* g_stage = "reading the options";

static int encode(int argc, char** argv)
{
#ifndef NDEBUG
    printf("DEBUG build\n");   // the first line out, before any argument is read, so the build type is never in doubt
#endif
#ifdef NNTC_SANITIZER_SELFTEST
    // Proof that a sanitizer build is live: one deliberate fault, which the sanitizer must stop before anything else
    // runs. Defined only on the command line of a throwaway build (-DNNTC_SANITIZER_SELFTEST=N), never in the tree's.
    {
        volatile int* a = new int[4];
        volatile int big = INT_MAX;
        if (NNTC_SANITIZER_SELFTEST == 1)
            a[4] = 1;          // one past the end of a heap array: AddressSanitizer
        if (NNTC_SANITIZER_SELFTEST == 2)
            a[-1] = 1;         // one before the start: AddressSanitizer
        if (NNTC_SANITIZER_SELFTEST == 3)
            big = big + 1;     // signed overflow: UndefinedBehaviorSanitizer
        printf("sanitizer self-test %d: the fault was NOT caught\n", NNTC_SANITIZER_SELFTEST);
        delete[] a;
    }
#endif
    if (argc < 2)
    {
        usage();
        return 1;
    }
    Options o;
    if (!parse_options(argc, argv, o))
        return o.help ? 0 : 1;   // asking for the usage is not a refusal
    // The material, if there is one, before anything is validated: it is what fills the input list.
    if (!load_material(o))
        return 1;
    if (!validate_options(o))
        return 1;

    const auto t_start = std::chrono::steady_clock::now();

    // ---- the source
    g_stage = "loading the source textures";
    std::vector<Image> tex(o.inputs.size());
    for (size_t t = 0; t < o.inputs.size(); t++)
    {
        bool alpha_ignored = false;
        // The size bound first, from the header, so an oversized file is refused before its pixels are decoded; and
        // again after the decode, for a format whose header stb cannot read on its own.
        int iw = 0, ih = 0;
        if (image_dimensions(o.inputs[t], iw, ih) && (iw > IMAGE_MAX_DIM || ih > IMAGE_MAX_DIM))
        {
            fprintf(stderr, "ERROR: '%s' is %dx%d; the encoder accepts at most %dx%d (IMAGE_MAX_DIM in src/image.h)\n",
                    o.inputs[t].c_str(), iw, ih, IMAGE_MAX_DIM, IMAGE_MAX_DIM);
            return 1;
        }
        if (!image_load_rgb(o.inputs[t], tex[t], alpha_ignored))
        {
            // stb reports its own allocation failure as a failed read; it is said as what it is.
            const char* reason = stbi_failure_reason();
            if (reason && strcmp(reason, "outofmem") == 0)
            {
                fprintf(stderr, "ERROR: out of host memory while reading '%s'\n", o.inputs[t].c_str());
                return 1;
            }
            fprintf(stderr, "ERROR: cannot read '%s'\n", o.inputs[t].c_str());
            return 1;
        }
        if (tex[t].w > IMAGE_MAX_DIM || tex[t].h > IMAGE_MAX_DIM)
        {
            fprintf(stderr, "ERROR: '%s' is %dx%d; the encoder accepts at most %dx%d (IMAGE_MAX_DIM in src/image.h)\n",
                    o.inputs[t].c_str(), tex[t].w, tex[t].h, IMAGE_MAX_DIM, IMAGE_MAX_DIM);
            return 1;
        }
        // The file carried transparency and the loader dropped it. Said once per file, on stdout beside the other
        // WARNINGS and whatever --quiet says, because what was encoded is then not the whole of what was handed over:
        // an image whose transparency mattered has silently become its colour alone. An alpha plane that is opaque
        // everywhere says nothing, because nothing was lost with it.
        if (alpha_ignored)
            printf("WARNING: %s has an alpha channel; it is ignored (the tool encodes 24-bit RGB)\n",
                   o.inputs[t].c_str());
        if (tex[t].w != tex[0].w || tex[t].h != tex[0].h)
        {
            fprintf(stderr, "ERROR: '%s' is %dx%d but '%s' is %dx%d\n", o.inputs[t].c_str(), tex[t].w, tex[t].h,
                    o.inputs[0].c_str(), tex[0].w, tex[0].h);
            return 1;
        }
    }

    Model m;
    m.textures = (int)tex.size();
    m.nout = 3 * m.textures;
    m.source_width = tex[0].w;
    m.source_height = tex[0].h;
    m.c0 = o.c0;
    m.bits0 = o.bits0;
    m.c1 = o.c1;
    m.bits1 = o.bits1;
    m.dec.nin = feature_count(m.c0, m.c1);
    m.dec.nout = m.nout;
    m.inputs = o.tex;

    // The objective's per-output weight: the texture's own weight times its channel's weight inside the texture, with
    // the three rgb weights normalised to mean 1 so that changing their balance does not change the scale of E.
    // The three rgb weights are normalised to mean 1 PER TEXTURE. One global triple normalised per texture is exactly
    // what the single global normalisation used to produce, so no run's numbers move; a material JSON's per-texture
    // triples then mean the same thing texture by texture.
    m.cw.assign((size_t)m.nout, 1.0f);
    for (int c = 0; c < m.nout; c++)
    {
        const TextureSettings& s = o.tex[(size_t)(c / 3)];
        const double rgb_sum = s.rgb_weights[0] + s.rgb_weights[1] + s.rgb_weights[2];
        m.cw[c] = (float)(s.weight * s.rgb_weights[c % 3] * 3.0 / rgb_sum);
    }

    g_stage = "building the source chain";
    const int wp = (m.source_width + BLOCK - 1) / BLOCK * BLOCK;
    const int hp = (m.source_height + BLOCK - 1) / BLOCK * BLOCK;
    m.padded = wp != m.source_width || hp != m.source_height;
    if (m.padded)
    {
        printf("WARNING: the input is %dx%d, which is not divisible by %d; it is padded by edge replication to %dx%d "
               "for this run and the JSON records both sizes. Every reported PSNR is over the original %dx%d extent. "
               "The source chain is derived from the PADDED image: a filter, and `wrap` in particular, treats the "
               "replicated edge as part of the picture, so a padded input is a testing convenience and not the shape "
               "a material should ship in.\n",
               m.source_width, m.source_height, BLOCK, wp, hp, m.source_width, m.source_height);
        for (Image& t : tex)
            image_pad(t, wp, hp);
    }
    m.width = wp;
    m.height = hp;

    Image source;
    image_pack(tex, source);
    tex.clear();

    build_plane_sizes(m, o.mip_min, o.mips != 0);
    // A chain of one - --mips 0, or a source small enough that --mip-min stops the chain at the base - gives a source
    // filter nothing to derive, and that is fine: the setting is honoured wherever there is a level below the base and
    // is simply idle where there is none (validate_options says why this is not a refusal).
    std::vector<Image> chain;
    if (!build_source_chain(source, m.planes, o.tex, chain))
        return 1;

    std::vector<double> mip_share, mip_site;
    build_mip_weights(m.planes, o.mip_weight, o.mip_mix, o.k, mip_share, mip_site);

    // Level 0's palette is the values the SHIPPED format decodes to, not the values the index nominally stands for
    // (model.h, level0_value): the standard BC palette k / (2^b - 1) when level 0 is block-compressed, the replicated
    // byte over 255 when it is an uncompressed UNORM8 plane. The solver fits what the consumer's sampler returns.
    m.bc0 = o.bc0 != 0;
    m.l0_bc8 = o.l0 == "bc8";
    // The residual seed needs a full-resolution residual buffer on the device, and it is allocated only when the init
    // asks for it: at four textures over a 2048 x 1024 base it is a hundred megabytes.
    m.init0_residual = o.init0 == "residual" || o.init0 == "texture";
    m.init0_texture = o.init0 == "texture";
    m.init0_chain = o.init0_scope == "chain";
    // Under --l0 bc8 there is no palette: level 0 is a continuous plane on a per-channel lo/hi grid at 8 bits, fitted
    // and frozen at --q1-start exactly as level 1's is, so palette0 stays empty and lo0 / hi0 carry the grid.
    if (m.l0_bc8)
    {
        m.lo0.assign((size_t)m.c0, -1.0f);
        m.hi0.assign((size_t)m.c0, 1.0f);
    }
    else
        build_level0_palette(m.bits0, m.c0, m.bc0, m.palette0);

    g_stage = "creating the model";
    // ---- the backend, and then its device. select_backend carries the fallback and has printed whichever line it
    // chose; one status for every failure, as everywhere else, so a caller tests `if errorlevel 1` and nothing finer.
    DeviceInfo info;
    if (!select_backend(o, info))
        return 1;

    // ---- the banner, the one place the run says what it is about to do. --quiet suppresses it whole, rows and
    // all; the padding warning above it and every warning below it are not part of it and are never suppressed.
    if (!o.quiet)
        print_banner(o, m, info);

    // ---- the model
    be::Device* d = be::device_create(m, chain);
    const float luma[3] = { LUMA_R, LUMA_G, LUMA_B };
    const bool init0_residual = m.init0_residual;
    g_stage = "initialising the latents";
    be::init_level0(d, m, luma, !init0_residual);
    Level1InitReport init_rep;
    be::init_level1(d, m, o.init == "pca", init_rep);

    // The box init reads source channels 0 .. C1-1 in order, so a grayscale texture (R = G = B) given first would hand
    // level 1 two or three identical channels: collinear columns in block (a) and a null direction in block (b) that
    // only rounding settles, and a result that moves by decibels on a last-bit change. So the box init takes the source
    // channels in their order with every exact copy of an earlier channel moved to the back. Where that picks the same
    // SET of channels as the plain order (no copy among the first C1, or only copies there because nothing else exists,
    // as for one grayscale texture) nothing below runs; otherwise the chosen channels' block means are formed here
    // exactly as k_init_level1_box forms them and replace the device's, on a grid fitted as the init fits it.
    std::vector<int> box_order, box_copies;   // empty unless the rule changed level 1's channels; the report reads them
    if (o.init == "box")
    {
        const int nout = m.nout;
        std::vector<int> order, copies;
        for (int s = 0; s < nout; s++)
        {
            bool copy = false;
            for (int e = 0; e < s && !copy; e++)
            {
                bool same = true;
                for (size_t i = 0; i < chain.size() && same; i++)
                {
                    const std::vector<float>& v = chain[i].v;
                    for (size_t t = 0; t < v.size() / (size_t)nout && same; t++)
                        same = v[t * nout + s] == v[t * nout + e];
                }
                copy = same;
            }
            if (copy)
                copies.push_back(s);
            else
                order.push_back(s);
        }
        order.insert(order.end(), copies.begin(), copies.end());
        const int taken = std::min(m.c1, nout);
        std::vector<int> chosen(order.begin(), order.begin() + taken);
        std::sort(chosen.begin(), chosen.end());
        bool plain = true;
        for (int c = 0; c < taken; c++)
            plain = plain && chosen[(size_t)c] == c;
        if (!plain)
        {
            for (size_t i = 0; i < m.planes.size(); i++)
            {
                const PlaneSize& pl = m.planes[i];
                const int w0 = pl.w0, h0 = pl.h0, w1 = pl.w1, h1 = pl.h1;
                std::vector<float> v1((size_t)w1 * h1 * m.c1);
                for (int y = 0; y < h1; y++)
                    for (int x = 0; x < w1; x++)
                    {
                        int xa = (int)((long long)x * w0 / w1), xb = (int)((long long)(x + 1) * w0 / w1);
                        int ya = (int)((long long)y * h0 / h1), yb = (int)((long long)(y + 1) * h0 / h1);
                        xb = std::min(w0, xb <= xa ? xa + 1 : xb);
                        yb = std::min(h0, yb <= ya ? ya + 1 : yb);
                        const size_t t1 = (size_t)y * w1 + x;
                        for (int c = 0; c < m.c1; c++)
                        {
                            if (c >= nout)
                            {
                                v1[t1 * m.c1 + c] = 0.0f;
                                continue;
                            }
                            const int sc = order[(size_t)c];
                            float acc = 0.0f;
                            int n = 0;
                            for (int sy = ya; sy < yb; sy++)
                                for (int sx = xa; sx < xb; sx++)
                                {
                                    acc += chain[i].v[((size_t)sy * w0 + sx) * nout + sc];
                                    n++;
                                }
                            const float inv = n > 0 ? 1.0f / (float)n : 0.0f;
                            v1[t1 * m.c1 + c] = 2.0f * acc * inv - 1.0f;
                        }
                    }
                be::level1_upload(d, m, (int)i, v1);
            }
            be::quantise_level1(d, m, "minmax");
            box_order.assign(order.begin(), order.begin() + taken);
            box_copies = copies;
            if (!o.quiet)
            {
                printf("level 1 init: level 1 takes source channels");
                for (int c : box_order)
                    printf(" %d", c);
                printf("; channel%s", box_copies.size() == 1 ? "" : "s");
                for (int c : box_copies)
                    printf(" %d", c);
                printf(" repeat an earlier channel exactly and are taken only after every distinct one\n");
            }
        }
    }

    // Block (a) measures its own step on the quadratic its normal equations already are, and refuses a candidate that
    // raises it: E does not rise across block (a) beyond the tolerance. The shipped ridge is tried first, so a
    // well-conditioned system gets the decoder it always got, bit for bit.
    Objective before, after, after_b, twin;
    be::objective_eval(d, m, o.k, mip_site, before);
    double ms_a = be::solve_decoder(d, m, o.k, mip_site);
    const double solve_ms = ms_a;
    be::objective_eval(d, m, o.k, mip_site, after);

    // THE RESIDUAL SEED (--init0 residual). Level 0 is still zero everywhere, and the block (a) above therefore fitted
    // the decoder to what level 1 alone can say. The residual of that decode is exactly what a full-resolution latent
    // is for, so each channel in turn is the leading principal direction of it, projected onto the plane - and each is
    // followed by a refit of the decoder, which is the deflation: the next channel's covariance is taken on a residual
    // the last channel has already been given the chance to spend.
    //
    // Channel 0 is seeded like the rest and is NOT assumed to be luminance. On a photograph the first direction comes
    // out close to luma anyway; on a normal map or a packed material it does not, and there is no reason the direction
    // a level-1 reconstruction misses most should be the one a monitor calls brightness.
    //
    // What --init0 luma does beyond channel 0 is seed nothing and let the alternation invent the channel, and a channel
    // that is zero everywhere is a dead column in block (a)'s normal equations: the more of them there are, the poorer
    // the basin the loop settles in. That is measurable - on md1-md4 --c0 4 came out BELOW --c0 3 at the same memory -
    // and it is the whole reason this seed exists.
    //
    // UNDER --l0 palette THE SEED IS FOLLOWED BY A SEARCH, and that is the second half of the palette mode's own fix.
    // The seed writes the projection SNAPPED onto the channel's fixed palette, and a snap is not a minimisation of
    // anything: it is the nearest entry to a number, chosen without the decoder, level 1 or a single site in sight.
    // The residual that channel k + 1 is then read off carries the whole of that snap's own error, which lies along
    // channel k's own direction - so the leading eigenvector comes back POINTING WHERE IT ALREADY POINTED and every
    // further channel is a copy of the first. It is measurable in the report's own e lines: on model10 at C0 2 the
    // two directions were +0.4356 +0.6500 +0.6227 and +0.4413 +0.6594 +0.6087, a cosine of 0.9995, where --l0 bc8 on
    // the same image picks +0.8924 -0.4134 -0.1808 for the second. Two collinear channels are a near-singular pair of
    // columns in block (a)'s normal equations; the decoder fits a large, nearly cancelling combination of them, and a
    // level-0 plane whose stored values are then perturbed by the BC4 / BC5 pack comes out of the decoder multiplied.
    //
    // So in the palette mode each seeded channel is followed by ONE block (c) - the exact per-texel search of
    // solve_level0.cu, which is the minimisation the snap is not - and one more block (a). The assignment the next
    // channel's residual is measured against is then the one the OBJECTIVE chose, and the direction that comes back is
    // a direction the representation is actually still missing. Both steps are minimisations, so E is monotone across
    // them exactly as it is across the rest of the seed.
    //
    // Under --l0 bc8 there is nothing to do here: the seed is continuous, no snap happens, and block (c') is not this
    // search. The bc8 path is untouched, byte for byte.
    std::vector<Init0ChannelReport> init0_rep;
    std::vector<double> init0_e_before, init0_e_refit, init0_e_after;
    double ms_c_seed = 0.0;
    if (init0_residual)
    {
        for (int c = 0; c < m.c0; c++)
        {
            Init0ChannelReport cr;
            be::init0_residual_channel(d, m, c, mip_site, cr);
            Objective seeded;
            be::objective_eval(d, m, o.k, mip_site, seeded);
            ms_a += be::solve_decoder(d, m, o.k, mip_site);
            be::objective_eval(d, m, o.k, mip_site, after);
            const double e_refit = after.e;
            if (!m.l0_bc8)
            {
                std::vector<Level0Report> l0_seed;
                ms_c_seed += be::solve_level0_all(d, m, o.k, o.sweeps, l0_seed);
                ms_a += be::solve_decoder(d, m, o.k, mip_site);
                be::objective_eval(d, m, o.k, mip_site, after);
            }
            init0_rep.push_back(cr);
            init0_e_before.push_back(seeded.e);
            init0_e_refit.push_back(e_refit);
            init0_e_after.push_back(after.e);
            // The seed's own progress: what each channel was given and what it bought. Every number of it comes back
            // in the report's `level 0 init` block, so --quiet drops it as it drops the round lines.
            if (o.quiet)
                continue;
            if (cr.texture >= 0)
                printf("level 0 init: channel %d = TEXTURE %d's leading residual direction, %.1f %% of the material's "
                       "weighted residual variance (eigenvalue %.3e of a trace of %.3e, divisor %.4f of a peak "
                       "of %.4f)\n",
                       c, cr.texture, 100.0 * cr.share, cr.eigenvalue, cr.total_variance, cr.divisor, cr.peak);
            else
                printf("level 0 init: channel %d = the residual's leading direction, %.1f %% of its variance "
                       "(eigenvalue %.3e of a trace of %.3e, divisor %.4f of a peak of %.4f)\n",
                       c, 100.0 * cr.share, cr.eigenvalue, cr.total_variance, cr.divisor, cr.peak);
            printf("              e =");
            for (int r = 0; r < cr.nout; r++)
                printf(" %+.4f", cr.dir[r]);
            printf("\n              E %.9e -> %.9e across the refit\n", seeded.e, e_refit);
            if (!m.l0_bc8)
                printf("              E %.9e -> %.9e across the search that chooses the snap and the refit after "
                       "it\n", e_refit, after.e);
        }
        if (!o.quiet)
            printf("\n");
    }
    after_b = after;

    g_stage = "encoding";
    // ---- the round loop
    //
    // Each block is the exact minimiser of E in its own variables with the other two held, so E is non-increasing
    // across every one of them and the whole sequence is monotone by construction. The loop stops when the rounds stop
    // buying anything: a relative drop below --tol twice running, or a round in which the search moved no texel and
    // level 1's correction was nothing beside the plane itself.
    //
    // Blocks (b) and (c) run on EVERY stored plane, every round. Plane m of both latents is fitted against mip m of the
    // source chain - the 1:1 rule - and with the decoder held the planes share no variable at all, so the M + 1
    // problems are independent and each is solved on its own dimensions, its own site set and its own source mip. They
    // are therefore issued together, each on its own stream, and waited for once; the per-plane timing in the report
    // says what each one cost. The mip weights stay out of both blocks: they govern only how much of the
    // one shared W each level gets, which is block (a)'s business and nothing else's.
    const size_t nplanes = m.planes.size();
    // The palette seed's per-channel searches ARE block (c) and are counted as it, so the report's block totals cover
    // every kernel the run issued.
    double ms_b = 0.0, ms_c = ms_c_seed;
    std::vector<double> ms_b_plane(nplanes, 0.0), ms_c_plane(nplanes, 0.0);
    std::vector<float> lo_before, hi_before, lo_after, hi_after, c_prev, plane_now, lo_p, hi_p;
    std::vector<double> delta_device;
    int dense_planes = 0;
    be::level1_range(d, m, 0, lo_before, hi_before);
    lo_after = lo_before;
    hi_after = hi_before;
    std::vector<Level1Report> lr(nplanes), lr0(nplanes);
    std::vector<Level0Report> l0(nplanes);
    Level1Report lr_cont;   // the BASE plane's last continuous solve: what the report's block (b) line quotes
    bool frozen = false;
    int frozen_round = 0;
    double ms_b_cg = 0.0, ms_b_sweeps = 0.0;
    double fd_worst = 0.0, dense_worst = 0.0, dense_range = 0.0;
    bool dense_done = false, fd_done = false, e_rose = false;
    const double fd_h = 1e-3;
    const int fd_samples = 20;
    const int dense_cap = 1024;
    int rounds_used = 0, tol_hits = 0;
    const char* stop_reason = "budget";
    double e_prev = after.e;

    if (!o.quiet)
        printf("\n");
    for (int r = 1; r <= o.rounds; r++)
    {
        Objective ea, eb, ec;

        // (a) the decoder, over every site of every plane.
        const double round_a = be::solve_decoder(d, m, o.k, mip_site);
        ms_a += round_a;
        be::objective_eval(d, m, o.k, mip_site, ea);

        // (b) level 1, every plane. Until the grid is frozen each plane is the continuous ridged minimiser of its own
        // values; at round --q1-start the one grid the format carries is fitted over the base and every chain plane
        // together and frozen, and every plane is snapped onto it - the one step of the whole loop that is a constraint
        // and not a minimisation, so it is the one step that may raise E - and from then on the block is the quantised
        // sweeps, which cannot.
        if (o.q1_start > 0 && r >= o.q1_start && !frozen)
        {
            be::freeze_level1_grid(d, m, o.q1_range);
            // Level 0's grid is frozen at the same round and by the same policy, and for the same reason: under
            // --l0 bc8 its plane is continuous too, and the format carries one lo/hi pair per channel for the whole
            // chain. Both snaps happen here, BEFORE block (b) assembles, so the one step of the loop that is a
            // constraint rather than a minimisation - and therefore the one step that may raise E - is the same step
            // for both latents and shows up in the same place in the round line.
            if (m.l0_bc8)
            {
                be::freeze_level0_grid(d, m, o.q1_range);
                // Both grids come back in the report, so the freeze is progress like the round lines around it and
                // --quiet drops it with them.
                if (!o.quiet)
                {
                    printf("level 0 grid frozen: round %d, --q1-range %s over the base and every chain plane, 256 "
                           "levels per channel", r, o.q1_range.c_str());
                    for (int c = 0; c < m.c0; c++)
                        printf("  ch%d [%.4f, %.4f] step %.3e", c, (double)m.lo0[(size_t)c], (double)m.hi0[(size_t)c],
                               ((double)m.hi0[(size_t)c] - (double)m.lo0[(size_t)c]) / (double)m.levels0());
                    printf("\n");
                }
            }
            frozen = true;
            frozen_round = r;
            if (!o.quiet)
            {
                printf("grid frozen: round %d, --q1-range %s over the base and every chain plane, %d levels per "
                       "channel", r, o.q1_range.c_str(), m.levels1() + 1);
                for (int c = 0; c < m.c1; c++)
                    printf("  ch%d [%.4f, %.4f] step %.3e", c, (double)m.lo1[(size_t)c], (double)m.hi1[(size_t)c],
                           ((double)m.hi1[(size_t)c] - (double)m.lo1[(size_t)c]) / (double)m.levels1());
                printf("\n");
            }
        }

        // Every plane at once, each an independent problem on its own stream. A continuous plane's movement is
        // measured as the correction beside half a grid step of its own range - below that it could not change a
        // stored value - and a quantised plane's is the sweeps' own count of stored indices that actually changed.
        long long moved1 = 0;
        double round_b = 0.0, delta_norm = 0.0, plane_norm = 0.0;
        double min_diag = 1e300;

        // The planes as the stencil will see them, kept for the dense direct twin behind --check.
        std::vector<std::vector<float>> c_prev_planes;
        if (o.check && r == 1 && !frozen)
        {
            c_prev_planes.resize(nplanes);
            for (size_t p = 0; p < nplanes; p++)
                be::level1_download(d, m, (int)p, c_prev_planes[p]);
        }

        if (frozen)
            round_b = be::sweep_level1_all(d, m, o.k, o.ridge, o.q1_sweeps, lr);
        else
        {
            round_b = be::solve_level1_all(d, m, o.k, o.ridge, lr);
            lr_cont = lr[0];
        }
        for (size_t p = 0; p < nplanes; p++)
        {
            ms_b_plane[p] += lr[p].ms_assemble + lr[p].ms_precond + lr[p].ms_cg + lr[p].ms_sweeps;
            min_diag = std::min(min_diag, lr[p].min_diag);
            moved1 += lr[p].quantised ? lr[p].moved : lr[p].moved_values;
            delta_norm += lr[p].delta2;
            plane_norm += lr[p].plane2;
        }
        if (frozen)
            ms_b_sweeps += round_b;
        else
        {
            ms_b_cg += round_b;
            lo_after.assign(lr[0].lo, lr[0].lo + m.c1);
            hi_after.assign(lr[0].hi, lr[0].hi + m.c1);
        }

        // The block (b) checks, on the first round's solve of EVERY plane: the gradient of E in the plane's own values
        // is zero at the answer, and on a plane small enough to afford it the dense direct solve of the same system
        // agrees. Every plane is checked, not only the base, because the floor halving leaves a chain plane's level 1
        // at something other than a clean quarter of its level 0 - 281 against 70, not 280 - so the stencil's
        // footprint rule is exercised there in a way the base never exercises it.
        if (o.check && r == 1 && !c_prev_planes.empty())
        {
            Objective plane_e;
            be::objective_eval(d, m, o.k, mip_site, plane_e);
            fd_done = true;
            for (size_t p = 0; p < nplanes; p++)
            {
                fd_worst = std::max(fd_worst, finite_difference_probe(d, m, o.k, mip_site, (int)p, fd_h, fd_samples,
                                                                      plane_e.e_plane[p]));
                be::level1_delta(d, m, (int)p, delta_device);
                std::vector<double> delta_host;
                if (be::solve_level1_dense_host(d, m, o.k, (int)p, lr[p].lambda, c_prev_planes[p], delta_host, dense_cap))
                {
                    double lo = 1e300, hi = -1e300;
                    for (size_t i = 0; i < delta_host.size() && i < delta_device.size(); i++)
                    {
                        lo = std::min(lo, delta_host[i]);
                        hi = std::max(hi, delta_host[i]);
                        dense_worst = std::max(dense_worst, std::fabs(delta_device[i] - delta_host[i]));
                    }
                    dense_range = std::max(dense_range, hi - lo);
                    dense_planes++;
                    dense_done = true;
                }
            }
        }
        ms_b += round_b;
        delta_norm = std::sqrt(delta_norm);
        plane_norm = std::sqrt(plane_norm);
        be::objective_eval(d, m, o.k, mip_site, eb);

        // The first round's E across block (b) is what the report quotes, so it is kept whether or not --check is on.
        if (r == 1)
            after_b = eb;

        // The device's E against its brute-force host twin, once, after the first round's block (b).
        if (o.check && r == 1)
            be::objective_check_host(d, m, o.k, mip_site, twin);

        // (c) level 0, every plane. In the palette mode that is four colour passes of the exact per-texel search,
        // each pass an exact joint step over the texels it moves. Under --l0 bc8 it is block (c') instead: the same
        // sparse least squares block (b) runs, over level 0's own plane - continuous while the grid is open, the
        // quantised four-colour sweeps once it is frozen - and the palette search is not run at all.
        long long moved0 = 0;
        double round_c = 0.0;
        if (m.l0_bc8)
        {
            round_c = frozen ? be::sweep_level0_cont_all(d, m, o.k, o.ridge, o.q1_sweeps, lr0)
                             : be::solve_level0_cont_all(d, m, o.k, o.ridge, lr0);
            for (size_t p = 0; p < nplanes; p++)
            {
                ms_c_plane[p] += lr0[p].ms_assemble + lr0[p].ms_precond + lr0[p].ms_cg + lr0[p].ms_sweeps;
                moved0 += lr0[p].quantised ? lr0[p].moved : lr0[p].moved_values;
            }
        }
        else
        {
            round_c = be::solve_level0_all(d, m, o.k, o.sweeps, l0);
            for (size_t p = 0; p < nplanes; p++)
            {
                ms_c_plane[p] += l0[p].ms;
                moved0 += l0[p].moved_total;
            }
        }
        ms_c += round_c;
        be::objective_eval(d, m, o.k, mip_site, ec);

        // Every block is a minimisation, so each of these must be non-increasing to within float rounding. The single
        // exception is block (b) of the round that freezes the grid: snapping the plane onto the grid is a constraint
        // being applied, not a step being taken, and a constraint can only cost E. Every later block (b) is the
        // quantised sweeps, which are minimisations of the same objective over the grid and so cannot raise E.
        // The comparison is relative AND carries an absolute floor. E is a float sum in [0,1] units, and on a picture the
        // representation reproduces exactly - an 8x8 of alternating columns, say - it sits at 1e-19 to 1e-17, which is
        // the rounding noise of the sum and not a measurement; a purely relative test then reads a step from 1.4e-19 to
        // 1.6e-18 as a rise and warns for nothing. There is a second, larger effect at that scale: block (a) minimises
        // E plus a ridge of 1e-9 times the mean diagonal of its normal matrix, and what it guarantees is that E does
        // not rise across it BEYOND ITS OWN TOLERANCE - a tolerance in the units of E, so once E is below about 1e-10
        // the ridge term and not E decides the step and E can rise by that much (7e-15 to 8e-12 on the same 8x8).
        // The tolerances are deliberately loose - a relative 1e-6 and an absolute E_NOISE of 1e-8, three orders of
        // magnitude under the smallest objective a real image reaches - because a check that trips on rounding is a
        // check nobody trusts; a real defect raises E by a fraction of itself, not by a billionth.
        const bool rose_a = ea.e > e_prev * (1.0 + E_REL) + E_NOISE;
        const bool rose_b = eb.e > ea.e * (1.0 + E_REL) + E_NOISE && r != frozen_round;
        const bool rose_c = ec.e > eb.e * (1.0 + E_REL) + E_NOISE;
        // Deliberately NOT an assert: this is a floating-point judgement, and an assert that can cry wolf on a
        // degenerate picture is one nobody trusts. The warning line stays in the log for a reviewer to read.
        if (rose_a || rose_b || rose_c)
        {
            e_rose = true;
            printf("WARNING: round %d raised E across block (%s), which that block's own minimisation forbids\n", r,
                   rose_a ? "a" : (rose_b ? "b" : "c"));
        }

        rounds_used = r;
        const double drop = ec.e > 0.0 ? (e_prev - ec.e) / ec.e : 0.0;
        // The round line is progress and nothing else - every number on it comes back in the report or in the asset -
        // so --quiet drops it and keeps the two lines above, which are warnings.
        if (!o.quiet)
        {
            printf("round %3d  E %.6e %.6e %.6e  psnr %.2f  sampled psnr %.2f", r, ea.e, eb.e, ec.e,
                   mse_to_db(ec.centre_mse), o.k > 0 ? mse_to_db(ec.sampled_mse) : 0.0);
            // The chain's levels, from the same pass: plane m decoded at its own texel centres against mip m of the
            // source chain, in fp. The report's mip psnr line is the same quantity rounded to 8 bits per level.
            if (nplanes > 1)
            {
                printf("  mip psnr");
                for (size_t p = 1; p < nplanes; p++)
                    printf(" M%zu %.2f", p, mse_to_db(ec.centre_plane[p]));
            }
            printf("  moved0 %lld  moved1 %lld  minDiagH %.3e  %.1f ms\n", moved0, moved1, min_diag,
                   round_a + round_b + round_c);
            fflush(stdout);
        }

        e_prev = ec.e;
        // Nothing moved anywhere: no level-0 texel of any plane, and no level-1 value of any plane - a correction
        // nothing beside the planes' own norm while they are continuous, no stored index at all once they are not.
        const bool level1_still = frozen ? moved1 == 0
                                         : delta_norm <= 1e-7 * (plane_norm > 0.0 ? plane_norm : 1.0);
        if (moved0 == 0 && level1_still)
        {
            stop_reason = "no-move";
            break;
        }
        tol_hits = drop < o.tol ? tol_hits + 1 : 0;
        if (tol_hits >= 2)
        {
            stop_reason = "tol";
            break;
        }
    }

    // ---- what the asset stores. With a frozen grid every plane is already storable, so the run ends with one more
    // block (b) per plane - the quantised sweeps - against the decoder the loop left behind; with --q1-start 0 the
    // continuous planes are fitted to their one shared grid and snapped here instead. Either way a last level-0 search
    // runs on every plane against the values level 1 actually holds, and then ONE last block (a) over every site of
    // every plane, so that W is optimal for what the asset stores rather than for what the loop last held.
    std::vector<Level1Report> lr_final = lr;
    std::vector<Level0Report> l0_final(nplanes);
    if (frozen)
    {
        const double last_b = be::sweep_level1_all(d, m, o.k, o.ridge, o.q1_sweeps, lr_final);
        ms_b += last_b;
        ms_b_sweeps += last_b;
        for (size_t p = 0; p < nplanes; p++)
            ms_b_plane[p] += lr_final[p].ms_assemble + lr_final[p].ms_sweeps;
        lr = lr_final;
    }
    else
    {
        be::quantise_level1(d, m, o.q1_range);
    }
    std::vector<Level1Report> lr0_final = lr0;
    if (m.l0_bc8)
    {
        if (frozen)
        {
            const double last_c = be::sweep_level0_cont_all(d, m, o.k, o.ridge, o.q1_sweeps, lr0_final);
            ms_c += last_c;
            for (size_t p = 0; p < nplanes; p++)
                ms_c_plane[p] += lr0_final[p].ms_assemble + lr0_final[p].ms_sweeps;
        }
        else
            be::quantise_level0(d, m, o.q1_range);
        lr0 = lr0_final;
    }
    else
    {
        ms_c += be::solve_level0_all(d, m, o.k, o.sweeps, l0_final);
        for (size_t p = 0; p < nplanes; p++)
            ms_c_plane[p] += l0_final[p].ms;
    }
    ms_a += be::solve_decoder(d, m, o.k, mip_site);
    for (size_t i = 0; i < m.planes.size(); i++)
        be::level0_download(d, m, (int)i, m.k0[i]);

    // ---- THE POST-FIT BC ENCODE (--l0 bc8), THE REFINEMENT OF IT, AND THE OUTER REPACK LOOP.
    //
    // The loop has just left a level 0 whose values are the 8-bit grid values of the indices in m.k0. That plane is
    // not what the .dds holds: BC4 gives a 4x4 block two endpoints and eight values along the line between them, so
    // the pack is a choice and it costs something. Three steps are taken here, in this order, and each is measured:
    //
    //   1. THE SEED PACK. bc_pack.h's stb_dxt-style pack - the block's lowest and highest value as the endpoints and
    //      the nearest palette entry per texel - fitted against the latent's own values. It is what the mode shipped
    //      before the refinement existed, and it is the point everything below is compared against.
    //   2. THE REFINEMENT (refine_bc.cu). Every block's endpoints and selectors are re-chosen to minimise THE
    //      DECODER'S OUTPUT ERROR over the sites that block's texels touch - the endpoints by least squares on that
    //      error, the selectors by exact argmin - and a candidate is taken only when the error measurably falls. It
    //      starts from step 1, so it cannot be worse than it.
    //   3. THE REFIT. Level 1 on its frozen grid against the packed level 0, then the decoder over every site, so that
    //      what is stored is optimal for the level 0 a consumer will actually sample.
    //
    // Then the OUTER REPACK LOOP (--bc-outer N), which is step 3, the continuous level-0 solve and steps 1-2 again:
    // with level 1 and the decoder now fitted to the DECODED plane, the continuous plane that best serves them is a
    // different plane, and packing that one may land better than packing the first. It may also not, so a pass is
    // taken only if the SHIPPED objective - E measured on the packed plane, which is what a consumer gets - strictly
    // falls; otherwise the previous pass's planes and decoder are put back and the loop stops. The result is therefore
    // never worse than the pass before it, and never worse than the refinement's own pack.
    double pack_psnr = 0.0, pack_psnr_seed = 0.0;
    double e_prepack = 0.0, e_packed = 0.0, e_refined = 0.0, e_refit = 0.0, e_ship = 0.0;
    double psnr_prepack = 0.0, psnr_packed = 0.0, psnr_refined = 0.0, psnr_refit = 0.0, psnr_ship = 0.0;
    size_t pack_texels = 0;
    double ms_pack = 0.0, ms_refine = 0.0, ms_outer = 0.0;
    int outer_taken = 0, outer_tried = 0;

    // The refit, which the post-fit step and every outer pass make: one block (b) over level 1 against the level 0 the
    // device now holds. The block (a) that completes it is issued by the caller, because the two are timed apart.
    //
    // It is the QUANTISED SWEEPS on the grid level 1 already carries, under --q1-start 0 as much as under the frozen
    // path. By the time anything below runs, level 1 has been snapped once - at the end of the loop for the frozen
    // path, by quantise_level1 just above for the control path - so a grid exists either way, and the sweeps are a
    // minimisation of E over it. Re-fitting the range here instead (which is what the control path used to do) is not
    // a minimisation of anything: it moves the grid under the level 0 it is being weighed against, so the gate that
    // says a refit cannot raise E had nothing behind it there.
    auto refit_level1 = [&]()
    {
        const double refit_b = be::sweep_level1_all(d, m, o.k, o.ridge, o.q1_sweeps, lr_final);
        ms_b += refit_b;
        ms_b_sweeps += refit_b;
        for (size_t pi = 0; pi < nplanes; pi++)
            ms_b_plane[pi] += lr_final[pi].ms_assemble + lr_final[pi].ms_sweeps;
        lr = lr_final;
        return refit_b;
    };

    if (m.l0_bc8)
    {
        Objective probe;
        be::objective_eval(d, m, o.k, mip_site, probe);
        e_prepack = probe.e;
        psnr_prepack = mse_to_db(probe.centre_mse);

        ms_pack += be::bc_pack_prepare(d, m, o.k, true, pack_psnr_seed, pack_texels);
        be::objective_eval(d, m, o.k, mip_site, probe);
        e_packed = probe.e;
        psnr_packed = mse_to_db(probe.centre_mse);
        pack_psnr = pack_psnr_seed;

        std::vector<BcRefineReport> refine_passes;
        ms_refine += be::bc_refine(d, m, o.bc_refine, refine_passes, pack_psnr, pack_texels);
        be::objective_eval(d, m, o.k, mip_site, probe);
        e_refined = probe.e;
        psnr_refined = mse_to_db(probe.centre_mse);
        refit_level1();
        ms_a += be::solve_decoder(d, m, o.k, mip_site);
        be::objective_eval(d, m, o.k, mip_site, probe);
        e_refit = probe.e;
        psnr_refit = mse_to_db(probe.centre_mse);

        // The pack, the refinement and the outer loop are the second half of the run's progress, and every figure on
        // these lines comes back in the report's `level 0 pack` block; --quiet drops them all.
        if (!o.quiet)
        {
            printf("\nlevel 0 packed: BC4 / BC5 over %zu channel-texels of the chain, packing psnr %.2f dB against "
                   "the continuous plane\n", pack_texels, pack_psnr_seed);
            printf("                E %.9e psnr %.2f before the pack, E %.9e psnr %.2f after it, E %.9e psnr %.2f "
                   "after the refit\n", e_prepack, psnr_prepack, e_packed, psnr_packed, e_refit, psnr_refit);
            printf("                (the refit is one block (b) on level 1's grid and one block (a), both against the "
                   "packed plane; everything reported below is of that plane)\n");
            if (!refine_passes.empty())
            {
                printf("level 0 refine: %d pass%s of analysis by synthesis over the 4x4 blocks (the endpoints by "
                       "least squares on\n", (int)refine_passes.size(), refine_passes.size() == 1 ? "" : "es");
                printf("                the decoder's error, the selectors by exact argmin, a candidate taken only on "
                       "a strict decrease)\n");
                for (const BcRefineReport& r : refine_passes)
                    printf("                pass %d: %lld of %lld blocks improved, the blocks' own E fell by %.6e, "
                           "%.1f ms\n", r.pass, r.improved, r.blocks, -r.delta_e, r.ms);
                printf("                E %.9e psnr %.2f after the seed pack, E %.9e psnr %.2f after the refinement\n",
                       e_packed, psnr_packed, e_refined, psnr_refined);
                printf("                packing psnr %.2f -> %.2f dB against the continuous plane: it FALLS, which is "
                       "the point -\n", pack_psnr_seed, pack_psnr);
                printf("                the refined pack minimises the decoder's output error, not its distance to a "
                       "plane nobody samples\n");
            }
        }

        // --bc-refine-after: the same refinement again, now against the level 1 and the decoder that were fitted to
        // the packed plane. The blocks are kept and only re-weighed, so this too can only lower E.
        if (o.bc_refine_after > 0)
        {
            std::vector<BcRefineReport> after_passes;
            ms_pack += be::bc_pack_prepare(d, m, o.k, false, pack_psnr, pack_texels);
            ms_refine += be::bc_refine(d, m, o.bc_refine_after, after_passes, pack_psnr, pack_texels);
            be::objective_eval(d, m, o.k, mip_site, probe);
            if (!o.quiet)
                printf("level 0 refine: %d further pass%s after the refit: E %.9e psnr %.2f -> E %.9e psnr %.2f\n",
                       o.bc_refine_after, o.bc_refine_after == 1 ? "" : "es", e_refit, psnr_refit, probe.e,
                       mse_to_db(probe.centre_mse));
            e_refit = probe.e;
            psnr_refit = mse_to_db(probe.centre_mse);
        }

        e_ship = e_refit;
        psnr_ship = psnr_refit;

        // ---- the outer repack loop
        for (int pass = 1; pass <= o.bc_outer; pass++)
        {
            outer_tried++;

            // What a rejected pass is put back to: both planes and their indices as the device holds them, the model's
            // own indices and blocks, the decoder and both grids. Memory is not the constraint here, and a restore that
            // missed one of them would ship a state nothing measured.
            std::vector<std::vector<float>> keep_v0(nplanes), keep_v1(nplanes);
            std::vector<std::vector<uint8_t>> keep_dev_k0(nplanes), keep_dev_k1(nplanes);
            for (size_t i = 0; i < nplanes; i++)
            {
                be::level0_download_values(d, m, (int)i, keep_v0[i]);
                be::level0_download(d, m, (int)i, keep_dev_k0[i]);
                be::level1_download(d, m, (int)i, keep_v1[i]);
                be::level1_download_indices(d, m, (int)i, keep_dev_k1[i]);
            }
            const std::vector<std::vector<uint8_t>> keep_k0 = m.k0, keep_k1 = m.k1;
            const std::vector<std::vector<std::vector<uint8_t>>> keep_blocks = m.bc0_blocks;
            // The refinement's own device state beside the model's blocks: a later refinement continues from the
            // endpoints and selectors the device holds, not from the .dds bytes, so the two are saved and put back
            // together and a rejected pass leaves nothing of itself anywhere.
            std::vector<std::vector<uint8_t>> keep_bc_ep, keep_bc_sel;
            be::bc_state_save(d, m, keep_bc_ep, keep_bc_sel);
            const Decoder keep_dec = m.dec;
            const std::vector<float> keep_lo0 = m.lo0, keep_hi0 = m.hi0, keep_lo1 = m.lo1, keep_hi1 = m.hi1;
            const double keep_pack_psnr = pack_psnr;
            const double keep_pack_psnr_seed = pack_psnr_seed;
            const size_t keep_pack_texels = pack_texels;
            // The report-only fields the passes also overwrite. They carry no decision, but a rejected pass that left
            // them behind would have the report describe a state the asset is not in.
            const std::vector<Level1Report> keep_lr = lr, keep_lr0 = lr0, keep_lr_final = lr_final,
                                            keep_lr0_final = lr0_final;

            // (1) level 1 and the decoder against the PACKED level 0, (2) level 0 continuous again against them.
            double ms_this = refit_level1();
            const double ms_a_pass = be::solve_decoder(d, m, o.k, mip_site);
            ms_a += ms_a_pass;
            ms_this += ms_a_pass;
            {
                // Step (2) is the CONTINUOUS solve of level 0 followed by a snap, under both --q1-start paths. The
                // frozen path runs the four-colour sweeps, which are the exact grid argmin; the control path runs the
                // unconstrained solve and snaps onto the grid level 0 already has. What the control path must NOT do
                // is re-fit the range and snap - that is not a minimisation of E, so the pass would be weighed against
                // a plane the pass itself moved the grid of.
                const double ms_c_pass = frozen ? be::sweep_level0_cont_all(d, m, o.k, o.ridge, o.q1_sweeps, lr0_final)
                                                : be::solve_level0_cont_all(d, m, o.k, o.ridge, lr0_final);
                if (!frozen)
                    be::snap_level0_on_grid(d, m);
                ms_c += ms_c_pass;
                ms_this += ms_c_pass;
                for (size_t pi = 0; pi < nplanes; pi++)
                    ms_c_plane[pi] += lr0_final[pi].ms_assemble + lr0_final[pi].ms_precond + lr0_final[pi].ms_cg +
                                      lr0_final[pi].ms_sweeps;
                lr0 = lr0_final;
            }
            for (size_t i = 0; i < nplanes; i++)
                be::level0_download(d, m, (int)i, m.k0[i]);

            // (3) pack that plane and refine the pack, exactly as the post-fit step did.
            std::vector<BcRefineReport> outer_refine;
            const double ms_seed = be::bc_pack_prepare(d, m, o.k, true, pack_psnr_seed, pack_texels);
            ms_pack += ms_seed;
            const double ms_ref = be::bc_refine(d, m, o.bc_refine, outer_refine, pack_psnr, pack_texels);
            ms_refine += ms_ref;
            ms_this += ms_seed + ms_ref;
            ms_outer += ms_this;

            // (4) the shipped objective decides. It is E of the plane the file holds, which is the only quantity this
            // loop is allowed to improve.
            be::objective_eval(d, m, o.k, mip_site, probe);
            if (probe.e < e_ship)
            {
                if (!o.quiet)
                    printf("level 0 outer : pass %d ACCEPTED: shipped E %.9e psnr %.2f -> E %.9e psnr %.2f "
                           "(%.1f ms)\n", pass, e_ship, psnr_ship, probe.e, mse_to_db(probe.centre_mse), ms_this);
                e_ship = probe.e;
                psnr_ship = mse_to_db(probe.centre_mse);
                e_refit = e_ship;
                psnr_refit = psnr_ship;
                outer_taken++;
                continue;
            }
            if (!o.quiet)
                printf("level 0 outer : pass %d rejected: shipped E %.9e -> E %.9e, so the previous pass's planes and "
                       "decoder are restored (%.1f ms)\n", pass, e_ship, probe.e, ms_this);
            for (size_t i = 0; i < nplanes; i++)
            {
                be::level0_upload(d, m, (int)i, keep_v0[i]);
                be::level0_upload_indices(d, m, (int)i, keep_dev_k0[i]);
                be::level1_upload(d, m, (int)i, keep_v1[i]);
                be::level1_upload_indices(d, m, (int)i, keep_dev_k1[i]);
            }
            m.k0 = keep_k0;
            m.k1 = keep_k1;
            m.bc0_blocks = keep_blocks;
            be::bc_state_restore(d, m, keep_bc_ep, keep_bc_sel);
            m.dec = keep_dec;
            m.lo0 = keep_lo0;
            m.hi0 = keep_hi0;
            m.lo1 = keep_lo1;
            m.hi1 = keep_hi1;
            pack_psnr = keep_pack_psnr;
            pack_psnr_seed = keep_pack_psnr_seed;
            pack_texels = keep_pack_texels;
            lr = keep_lr;
            lr0 = keep_lr0;
            lr_final = keep_lr_final;
            lr0_final = keep_lr0_final;
            be::upload_decoder_to_device(d, m);
            be::upload_grids_to_device(d, m);
            break;
        }

        // ---- one last free refit, and the line the gates read the shipped E off.
        //
        // Whatever the pack the asset ends up with - the post-fit step's, the last ACCEPTED outer pass's, or the one a
        // rejected pass's restore put back - level 1 and the decoder beside it are the ones that were fitted BEFORE
        // that pack. One block (b) on level 1's grid and one block (a) over every site fix that. Level 0 does not move
        // here, so both are minimisations of E in their own variables over a fixed plane: block (b) is exact, block (a)
        // takes the step only if its own quadratic says E does not rise beyond the tolerance, so the shipped E does not
        // rise; it is two blocks of work and it is worth about a hundredth of a decibel.
        //
        // The "from" number on this line is therefore exactly the E of the last accepted pass, or the pre-loop E when
        // none was accepted - which is what the restore has to have produced - and the "to" number is the E the report
        // prints as `E shipped`. tests/run_checks.py asserts both.
        const double ms_last_b = refit_level1();
        const double ms_last_a = be::solve_decoder(d, m, o.k, mip_site);
        ms_a += ms_last_a;   // refit_level1 has already added its own to ms_b; the line below prints the pair's total
        be::objective_eval(d, m, o.k, mip_site, probe);
        if (!o.quiet)
            printf("level 0 final : one more refit of level 1 and the decoder against the packed plane: shipped E "
                   "%.9e psnr %.2f -> E %.9e psnr %.2f (%.1f ms)\n", e_ship, psnr_ship, probe.e,
                   mse_to_db(probe.centre_mse), ms_last_b + ms_last_a);
        e_ship = probe.e;
        psnr_ship = mse_to_db(probe.centre_mse);
        e_refit = e_ship;
        psnr_refit = psnr_ship;
    }

    for (size_t i = 0; i < m.planes.size(); i++)
        be::level1_download_indices(d, m, (int)i, m.k1[i]);

    // The plane as the device holds it must be exactly the grid value of every index about to be written.
    std::vector<std::vector<float>> v1_final(m.planes.size());
    for (size_t i = 0; i < m.planes.size(); i++)
        be::level1_download(d, m, (int)i, v1_final[i]);
    size_t grid_values = 0, grid_offgrid = 0;
    const bool on_grid = verify_level1_on_grid(m, v1_final, grid_values, grid_offgrid);
    assert(on_grid);
    if (!on_grid)
    {
        fprintf(stderr, "ERROR: %zu of %zu level-1 values are not the grid value of the index beside them; the asset "
                        "would decode to something the encoder never measured\n",
                grid_offgrid, grid_values);
        be::device_destroy(d);
        return 1;   // one status for every failure, this backstop included
    }

    // ---- what the report describes: the plane on the DISK.
    //
    // At 1-3 bits a block-compressed level 0 is exact - the endpoints are the mode's own extremes and the selector is a
    // relabelling of the index - so the plane the solver holds IS the plane the file holds. At 4 bits it is not: a
    // block carries eight of the sixteen values and the pack is a nearest-palette choice, worth about a decibel. The
    // plane is therefore packed and decoded again here, before anything is measured, so that the psnr, the mip psnr,
    // the recon PNGs and dds_decode --ref all describe the same bytes a consumer will sample.
    bool packed_report = false;
    if (o.bc0 && !m.l0_bc8)   // --l0 bc8 has already packed, refitted and said so above
        for (int b : m.bits0)
            packed_report = packed_report || b > 3;
    if (packed_report)
    {
        std::vector<float> v0;
        for (size_t i = 0; i < m.planes.size(); i++)
        {
            level0_plane_values(m, (int)i, true, true, v0);
            be::level0_upload(d, m, (int)i, v0);
        }
        if (!o.quiet)
            printf("level 0 packed: the report below is of the PACKED plane, decoded back from the blocks the .dds "
                   "holds, because the pack is lossy above 3 bits\n");
    }

    Objective shipped;
    be::objective_eval(d, m, o.k, mip_site, shipped);

    std::vector<std::vector<uint8_t>> recon;
    be::reconstruct_levels(d, m, recon);

    // ---- --diag: the per-plane diagnosis table, of the state the asset is in.
    //
    // The report's mip psnr line quotes the WORST texture of each level, which says a level is poor without saying
    // which texture is poor there, and the per-plane E hides the same thing behind the sum over the outputs. Both
    // matter when one texture of a material collapses at depth while another is untouched, so this table is the whole
    // matrix: every texture at every stored level, beside what each latent channel actually holds at that level.
    //
    // It is a measurement of the planes as they stand, taken after the last refit and before the files are written, so
    // it describes the bytes the asset carries.
    if (o.diag)
    {
        printf("\ndiag: per-texture psnr at every stored level (8-bit, texel centres, each against its own level of the "
               "run's own source chain)\n");
        // WHICH CHAIN each texture was fitted against, spelled out in the header the way the report's psnr lines
        // spell it out. The table below is a matrix of numbers with no room for it on a row, and "its own level of the
        // run's own source chain" is not an answer to which filter a row was measured under.
        for (int t = 0; t < m.textures; t++)
            printf("      t%d %s: %s\n", t, basename_of_path(o.tex[(size_t)t].file).c_str(),
                   chain_text(o.tex[(size_t)t]).c_str());
        // THE RANGE OF EACH SOURCE LEVEL, which is the one place the clamp mips.cpp applies is visible. A cubic filter
        // overshoots at an edge, and the target the objective is measured against has to stay inside the byte range
        // the base itself lives in; a level whose min went below 0 or whose max went above 1 would be a target no
        // 8-bit source could hold, and every number below would be measured against it.
        printf("diag: the range of each texture's source level, over its three channels (the chain is clamped to "
               "[0,1]\n      after every resize: a level outside it would be a target the 8-bit source cannot hold)\n");
        for (size_t i = 0; i < chain.size(); i++)
        {
            printf("      M%-2zu  %5dx%-5d", i, chain[i].w, chain[i].h);
            for (int t = 0; t < m.textures; t++)
            {
                float lo = 1e30f, hi = -1e30f;
                for (size_t p = 0; p < (size_t)chain[i].w * chain[i].h; p++)
                    for (int c = 0; c < 3; c++)
                    {
                        const float v = chain[i].v[p * (size_t)chain[i].nc + (size_t)(3 * t + c)];
                        lo = std::min(lo, v);
                        hi = std::max(hi, v);
                    }
                printf("  tex%d [%.6f, %.6f]", t, (double)lo, (double)hi);
            }
            printf("\n");
        }
        printf("      plane      size   ");
        for (int t = 0; t < m.textures; t++)
            printf("      tex%d", t);
        printf("     E plane\n");
        for (size_t i = 0; i < m.planes.size(); i++)
        {
            printf("      M%-2zu  %5dx%-5d", i, m.planes[i].w0, m.planes[i].h0);
            for (int t = 0; t < m.textures; t++)
            {
                // The base is measured over the original extent, as every psnr in the report is; a chain plane has no
                // original extent of its own and is measured over the whole plane.
                const int cw = i == 0 ? m.source_width : m.planes[i].w0;
                const int ch = i == 0 ? m.source_height : m.planes[i].h0;
                printf("  %8.2f", image_psnr_texture(chain[i], recon[i].data(), m.planes[i].w0, m.nout, t, cw, ch));
            }
            printf("  %.4e\n", shipped.e_plane[i]);
        }
        printf("      (the report's mip psnr line is the worst texture of each row)\n");

        printf("diag: what each latent channel holds, plane by plane: mean / sd over the plane, the range it reaches, "
               "and\n      the share of its values on an end of the ONE grid the whole chain shares\n");
        std::vector<float> dv0, dv1;
        for (size_t i = 0; i < m.planes.size(); i++)
        {
            be::level0_download_values(d, m, (int)i, dv0);
            be::level1_download(d, m, (int)i, dv1);
            for (int c = 0; c < m.c0; c++)
            {
                ChannelStats s;
                const float glo = m.l0_bc8 ? m.lo0[(size_t)c] : -1.0f;
                const float ghi = m.l0_bc8 ? m.hi0[(size_t)c] : 1.0f;
                channel_stats(dv0, m.c0, c, glo, ghi, s);
                printf("      M%-2zu level 0 ch%d  %5dx%-5d  mean %+9.5f  sd %9.5f  [%+9.5f, %+9.5f]  grid "
                       "[%+8.4f, %+8.4f]  ends %6.2f %%\n",
                       i, c, m.planes[i].w0, m.planes[i].h0, s.mean, s.sd, s.lo, s.hi, (double)glo, (double)ghi,
                       s.ends);
            }
            for (int c = 0; c < m.c1; c++)
            {
                ChannelStats s;
                channel_stats(dv1, m.c1, c, m.lo1[(size_t)c], m.hi1[(size_t)c], s);
                printf("      M%-2zu level 1 ch%d  %5dx%-5d  mean %+9.5f  sd %9.5f  [%+9.5f, %+9.5f]  grid "
                       "[%+8.4f, %+8.4f]  ends %6.2f %%\n",
                       i, c, m.planes[i].w1, m.planes[i].h1, s.mean, s.sd, s.lo, s.hi, (double)m.lo1[(size_t)c],
                       (double)m.hi1[(size_t)c], s.ends);
            }
        }
        printf("      (a channel whose sd is a grid step or two carries nothing at that level, whatever the decoder "
               "asks of it there)\n");

        // WHICH TEXTURE USES WHICH LATENT CHANNEL. The decoder's feature order is [a: c_j][b: s_i][sc: s_i c_j], so
        // level-0 channel i reaches output r through the b column i and the C1 sc columns of that i, and level-1
        // channel j through the a column j and the C0 sc columns of that j. The figure below is the root mean square
        // of texture t's three rows over exactly those columns: how hard texture t leans on that channel.
        //
        // It is the line that says whether a channel is DEDICATED or SHARED. Three independent pictures and three
        // level-0 channels can be one picture per channel, and this table says whether that is what the run found or
        // whether every channel came out serving every texture a little. And a texture whose level-0 figures are near
        // zero is a texture with no full-resolution path at all: it is then held to what the quarter-resolution level 1
        // can carry, at EVERY plane, which is what collapses a chain while the base looks fine.
        printf("diag: how hard each texture leans on each latent channel (rms of that texture's decoder rows over the\n"
               "      columns the channel reaches: level 0 through b_i and sc_i*, level 1 through a_j and sc_*j)\n");
        printf("      texture ");
        for (int c = 0; c < m.c0; c++)
            printf("   lat0 ch%d", c);
        for (int c = 0; c < m.c1; c++)
            printf("   lat1 ch%d", c);
        printf("       bias\n");
        for (int t = 0; t < m.textures; t++)
        {
            printf("         %d    ", t);
            for (int i = 0; i < m.c0; i++)
            {
                double acc = 0.0;
                for (int r = 3 * t; r < 3 * t + 3; r++)
                {
                    const float* row = &m.dec.w[(size_t)r * m.dec.nin];
                    acc += (double)row[m.c1 + i] * (double)row[m.c1 + i];
                    for (int j = 0; j < m.c1; j++)
                    {
                        const double v = (double)row[m.c1 + m.c0 + i * m.c1 + j];
                        acc += v * v;
                    }
                }
                printf("  %9.4f", std::sqrt(acc / (3.0 * (1 + m.c1))));
            }
            for (int j = 0; j < m.c1; j++)
            {
                double acc = 0.0;
                for (int r = 3 * t; r < 3 * t + 3; r++)
                {
                    const float* row = &m.dec.w[(size_t)r * m.dec.nin];
                    acc += (double)row[j] * (double)row[j];
                    for (int i = 0; i < m.c0; i++)
                    {
                        const double v = (double)row[m.c1 + m.c0 + i * m.c1 + j];
                        acc += v * v;
                    }
                }
                printf("  %9.4f", std::sqrt(acc / (3.0 * (1 + m.c0))));
            }
            double bias = 0.0;
            for (int r = 3 * t; r < 3 * t + 3; r++)
                bias += (double)m.dec.b[(size_t)r] * (double)m.dec.b[(size_t)r];
            printf("  %9.4f\n", std::sqrt(bias / 3.0));
        }
        printf("      (a texture with no level-0 figure of any size has no full-resolution path: level 1 alone is its "
               "ceiling at every plane)\n");
        fflush(stdout);
    }

    g_stage = "writing the asset";
    // The asset first: it is the deliverable. The PNGs are a viewing aid, so failing to write one is a WARNING and the
    // run still succeeds.
    size_t sizes[3] = { 0, 0, 0 };
    if (!write_asset(m, o.prefix, o.desc, o.bc0 != 0, sizes, o.quiet))
    {
        be::device_destroy(d);
        return 1;
    }
    if (o.png && !write_level_pngs(m, recon, chain, o.prefix))
        printf("WARNING: the recon / source PNGs are incomplete (see above); the asset was written\n");
    // --bc0 both: the same solve written a second time with an uncompressed level 0, so that the block-compressed path
    // and the uncompressed one can be compared on one set of planes. The reported sizes stay those of the shipped asset.
    //
    // The twin is a different asset, not a copy: its level-0 bytes mean lo + rep(k) / 255 * (hi - lo) where the
    // block-compressed one's mean lo + k / (2^bits - 1) * (hi - lo), and at 3, 5, 6 and 7 bits those are not the same
    // numbers. So the twin gets its own recon and source PNGs, decoded from its own plane, and PREFIX_u_recon is what
    // the twin round-trips against; the difference between the two sets is the format difference itself and is printed.
    if (o.bc0_both)
    {
        std::vector<std::vector<uint8_t>> recon_u;
        std::vector<float> v0;
        for (size_t i = 0; i < m.planes.size(); i++)
        {
            level0_plane_values(m, (int)i, false, false, v0);
            be::level0_upload(d, m, (int)i, v0);
        }
        be::reconstruct_levels(d, m, recon_u);
        int worst = 0;
        for (size_t i = 0; i < recon.size() && i < recon_u.size(); i++)
            for (size_t t = 0; t < recon[i].size() && t < recon_u[i].size(); t++)
                worst = std::max(worst, std::abs((int)recon[i][t] - (int)recon_u[i][t]));
        if (!o.quiet)
            printf("--bc0 both: the uncompressed twin decodes to within %d / 255 of the block-compressed asset over "
                   "every level (the two formats' own grids differ%s)\n", worst,
                   o.png ? "; PREFIX_u_recon is the twin's own reconstruction" : "");
        size_t other[3] = { 0, 0, 0 };
        if (!write_asset(m, o.prefix + "_u", o.prefix + "_u_nntc.json", false, other, o.quiet))
        {
            be::device_destroy(d);
            return 1;
        }
        if (o.png && !write_level_pngs(m, recon_u, chain, o.prefix + "_u"))
            printf("WARNING: the twin's recon / source PNGs are incomplete (see above); the twin's asset was written\n");
    }
    const size_t device_bytes = be::device_memory(d);
    be::device_destroy(d);

    g_stage = "writing the report";
    // ---- the report
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
    // --quiet means quiet: nothing on stdout but the WARNING lines, and nothing on stderr but the ERROR lines (the
    // owner's rule). The report, like the `wrote` lines, is progress a script did not ask for; the asset and its
    // descriptor are the account of what the encode produced.
    if (!o.quiet)
    {
        printf("\nreport\n");
        // The backend again, at the top of the report, for the same reason it is in the banner: a log that is read
        // without its banner still has to say what produced the numbers under it.
        printf("  backend      %s\n", be::backend_name(be::backend_selected()));
        for (int t = 0; t < m.textures; t++)
        {
            const double psnr = image_psnr_texture(chain[0], recon[0].data(), m.planes[0].w0, m.nout, t,
                                                   m.source_width, m.source_height);
            printf("  psnr         texture %d  %.2f dB  (8-bit, texel centres, original extent, unweighted; %s)\n", t,
                   psnr, chain_text(o.tex[(size_t)t]).c_str());
        }
        printf("  centre psnr  %.2f dB  (fp, every output, padded extent: the decode E measures at the centres)\n",
               mse_to_db(shipped.centre_mse));
        if (o.k > 0)
            printf("  sampled psnr %.2f dB  (k %d, the base plane's fixed subtexel sites, fp, every output)\n",
                   mse_to_db(shipped.sampled_mse), o.k);
        else
            printf("  sampled psnr n/a      (--k 0: centres only)\n");
        if (m.planes.size() > 1)
        {
            printf("  mip psnr    ");
            for (size_t i = 1; i < m.planes.size(); i++)
            {
                double worst = 1e30;
                for (int t = 0; t < m.textures; t++)
                    worst = std::min(worst, image_psnr_texture(chain[i], recon[i].data(), m.planes[i].w0, m.nout, t,
                                                               m.planes[i].w0, m.planes[i].h0));
                printf(" M%zu %.2f", i, worst);
            }
            printf("  dB, each against the run's own source chain (the worst texture of each level)\n");
        }
        {
            const WorstCase w = worst_case_of(chain[0], recon[0].data(), m.planes[0].w0, m.nout, m.source_width,
                                              m.source_height);
            printf("  worst case   p50 %d  p90 %d  p99 %d  max %d  (8-bit, the per-pixel largest error over all %d "
                   "outputs at\n",
                   w.p50, w.p90, w.p99, w.max, m.nout);
            printf("               the base's texel centres, original extent), %.3f %% of pixels above 8\n", w.above8);
        }
        printf("  E            %.9e before block (a), %.9e after  (normalised over every site of every plane)\n",
               before.e, after.e);
        if (after.e > before.e * (1.0 + E_REL) + E_NOISE)
            printf("  WARNING:     block (a) raised E, which its own normal equations forbid\n");
        printf("  E block (b)  %.9e after block (a), %.9e after block (b)  (the whole objective, round 1)\n", after.e,
               after_b.e);
        if (e_rose)
            printf("  WARNING:     a block raised E, which its own minimisation forbids\n");
        printf("  E per plane ");
        for (size_t i = 0; i < m.planes.size(); i++)
            printf(" M%zu %.6e -> %.6e", i, after.e_plane[i], shipped.e_plane[i]);
        printf("   (each on its own plane, without its mip weight; every plane is solved every round)\n");
        printf("  E shipped    %.9e  (the same objective on the values the .dds holds, level 1 snapped to its grid)\n",
               shipped.e);
        if (frozen_round == 1)
        {
            printf("  block (b)    no continuous round ran: --q1-start 1 froze the grid before the first solve\n");
        }
        else
        {
            printf("  block (b)    iterations %d  residual %.3e  lambda %.6e  diag H min %.6e max %.6e  (the BASE "
                   "plane's last CONTINUOUS round)\n",
                   lr_cont.iterations, lr_cont.residual, lr_cont.lambda, lr_cont.min_diag, lr_cont.max_diag);
            printf("               kernels: assembly %.3f ms  preconditioner %.3f ms  CG %.3f ms\n",
                   lr_cont.ms_assemble, lr_cont.ms_precond, lr_cont.ms_cg);
        }
        if (frozen)
        {
            long long moved_final = 0;
            for (size_t i = 0; i < nplanes; i++)
                moved_final += lr_final[i].moved;
            printf("  block (b) q  the quantised sweeps: %d per plane per round from round %d, the last pass moved "
                   "%lld stored values over every plane\n",
                   lr_final[0].sweeps, frozen_round, moved_final);
            printf("               kernels (the base plane's last pass): assembly %.3f ms  sweeps %.3f ms  lambda "
                   "%.6e  diag H min %.6e max %.6e\n",
                   lr_final[0].ms_assemble, lr_final[0].ms_sweeps, lr_final[0].lambda, lr_final[0].min_diag,
                   lr_final[0].max_diag);
            printf("               block (b) over the run: %.3f ms continuous (assembly + preconditioner + CG), "
                   "%.3f ms quantised (assembly + sweeps)\n",
                   ms_b_cg, ms_b_sweeps);
            printf("  level 1 grid frozen at round %d, --q1-range %s over the base and every chain plane:",
                   frozen_round, o.q1_range.c_str());
            for (int c = 0; c < m.c1; c++)
                printf("  ch%d [%.4f, %.4f]", c, (double)m.lo1[(size_t)c], (double)m.hi1[(size_t)c]);
            printf("\n");
        }
        else
        {
            if (o.q1_start == 0)
                printf("  level 1 grid --q1-start 0: fitted over every plane and snapped once at the end, --q1-range %s:",
                       o.q1_range.c_str());
            else
                printf("  level 1 grid: the loop ended at round %d, before --q1-start %d would have frozen it, so it was "
                       "fitted over every plane and snapped once at the end, --q1-range %s:",
                       rounds_used, o.q1_start, o.q1_range.c_str());
            for (int c = 0; c < m.c1; c++)
                printf("  ch%d [%.4f, %.4f]", c, (double)m.lo1[(size_t)c], (double)m.hi1[(size_t)c]);
            printf("\n");
        }
        printf("  level 1 on grid: %zu values, every one exactly the grid value of its stored index\n", grid_values);
        if (m.l0_bc8)
        {
            if (frozen)
                printf("  level 0 grid frozen at round %d, --q1-range %s over the base and every chain plane, 256 "
                       "levels:", frozen_round, o.q1_range.c_str());
            else
                printf("  level 0 grid fitted and snapped once at the end (the loop ended before any freeze), --q1-range "
                       "%s over the base and every chain plane, 256 levels:", o.q1_range.c_str());
            for (int c = 0; c < m.c0; c++)
                printf("  ch%d [%.4f, %.4f]", c, (double)m.lo0[(size_t)c], (double)m.hi0[(size_t)c]);
            printf("\n");
            printf("  block (c')   level 0 as a continuous plane by the same least squares block (b) uses, the BASE "
                   "plane: the last pass moved %lld stored values,\n", lr0[0].moved);
            printf("               lambda %.6e  diag H min %.6e max %.6e, assembly %.3f ms  sweeps %.3f ms\n",
                   lr0[0].lambda, lr0[0].min_diag, lr0[0].max_diag, lr0[0].ms_assemble, lr0[0].ms_sweeps);
            printf("  level 0 pack BC4 / BC5 after the last round over %zu channel-texels, packing psnr %.2f dB "
                   "(the seed pack's own %.2f)\n", pack_texels, pack_psnr, pack_psnr_seed);
            printf("               E %.9e -> %.9e -> %.9e -> %.9e and psnr %.2f -> %.2f -> %.2f -> %.2f over "
                   "(continuous, seed pack,\n",
                   e_prepack, e_packed, e_refined, e_refit, psnr_prepack, psnr_packed, psnr_refined, psnr_refit);
            printf("               refined pack, refitted): the first gap is what the block format costs, the second "
                   "what analysis by\n");
            printf("               synthesis takes back (%d pass%s, %.1f ms; the seed pack itself %.1f ms), the third "
                   "what a refit of\n", o.bc_refine, o.bc_refine == 1 ? "" : "es", ms_refine, ms_pack);
            printf("               level 1 and the decoder wins\n");
            if (outer_tried > 0)
                printf("  outer repack %d of %d pass%s accepted (refit, re-solve level 0, repack, taken only if the "
                       "shipped E falls), %.1f ms\n", outer_taken, outer_tried, outer_tried == 1 ? "" : "es", ms_outer);
        }
        else
        {
        printf("  block (c)    the final search against the stored level 1, the BASE plane: moved");
        for (int c = 0; c < 4; c++)
            printf(" %lld", l0_final[0].moved[c]);
        printf(" texels in the four colour passes (%lld of %lld), %.3f ms\n", l0_final[0].moved_total,
               (long long)m.planes[0].w0 * m.planes[0].h0, l0_final[0].ms);
        if (nplanes > 1)
        {
            long long chain_moved = 0, chain_texels = 0;
            double chain_ms = 0.0;
            for (size_t i = 1; i < nplanes; i++)
            {
                chain_moved += l0_final[i].moved_total;
                chain_texels += (long long)m.planes[i].w0 * m.planes[i].h0;
                chain_ms += l0_final[i].ms;
            }
            printf("               the chain's %zu planes: moved %lld of %lld texels, %.3f ms\n", nplanes - 1,
                   chain_moved, chain_texels, chain_ms);
        }
        printf("               the argmin is %s\n",
               l0_final[0].joint ? "the joint enumeration over every palette state of the texel"
                                 : "coordinate sweeps: exact in one channel at a time, not jointly");
        }
        printf("  level 0 init --init0 %s --init0-scope %s\n", o.init0.c_str(), o.init0_scope.c_str());
        if (init0_residual)
        {
            for (size_t c = 0; c < init0_rep.size(); c++)
            {
                const Init0ChannelReport& cr = init0_rep[c];
                if (cr.texture >= 0)
                    printf("               ch%zu  texture %d, %.1f %% of the weighted residual variance (%.3e of "
                           "%.3e), divisor %.4f of a peak of %.4f, E %.6e -> %.6e", c, cr.texture,
                           100.0 * cr.share, cr.eigenvalue, cr.total_variance, cr.divisor, cr.peak,
                           init0_e_before[c], init0_e_refit[c]);
                else
                    printf("               ch%zu  %.1f %% of the residual's variance (%.3e of %.3e), divisor %.4f of "
                           "a peak of %.4f, E %.6e -> %.6e", c, 100.0 * cr.share, cr.eigenvalue,
                           cr.total_variance, cr.divisor, cr.peak, init0_e_before[c], init0_e_refit[c]);
                // The third E is the search that chooses the snap and the refit after it, which only the palette mode
                // runs; under --l0 bc8 there is no snap and the line ends at the deflation's own refit.
                if (!m.l0_bc8)
                    printf(" -> %.6e", init0_e_after[c]);
                printf("\n");
                printf("                    e =");
                for (int r = 0; r < cr.nout; r++)
                    printf(" %+.4f", cr.dir[r]);
                printf("\n");
            }
            printf("               (each channel a leading principal direction of the residual at that point - of the\n");
            printf("               whole material under --init0 residual, of the one texture whose own residual\n");
            printf("               carries the most under --init0 texture - taken over %s and used for\n",
                   o.init0_scope == "chain" ? "EVERY plane, weighted as the objective weights them"
                                            : "the BASE plane alone");
            printf("               every plane of the chain, so channel k means the same direction of the material at\n");
            printf("               every level; projected and divided over the whole chain by the largest magnitude\n");
            printf("               it reaches under --l0 bc8, whose grid is fitted afterwards, and by the %.1f %%\n",
                   100.0 * INIT0_PERCENTILE);
            printf("               percentile of that magnitude under --l0 palette, whose palette is fixed on [-1,1]\n");
            printf("               and has no such freeze. The refit after each channel is the deflation; the three E\n");
            printf("               values are the seed, that refit, and - in the palette mode - the search that\n");
            printf("               chooses the snap by the objective, with its own refit.)\n");
        }
        else
        {
            printf("               channel 0 the source luminance, every further channel at zero\n");
        }
        printf("  level 1 init --init %s", o.init.c_str());
        if (init_rep.pca)
        {
            printf(", principal values");
            for (int c = 0; c < init_rep.components; c++)
                printf(" %.3e", init_rep.eigenvalue[c]);
            printf(" of a total variance of %.3e\n", init_rep.total_variance);
            printf("               (the base plane's block means projected on its own first %d directions, each\n",
                   init_rep.components);
            printf("               divided by the largest magnitude it reaches, so it fills [-1,1] and nothing is\n");
            printf("               clipped; those divisors are");
            for (int c = 0; c < init_rep.components; c++)
                printf(" %.4f", init_rep.peak[c]);
            printf(")\n");
        }
        else if (box_order.empty())
        {
            printf(": the block mean of source channel j as channel j\n");
        }
        else
        {
            printf(": the block means of source channels");
            for (int c : box_order)
                printf(" %d", c);
            printf(" (channel%s", box_copies.size() == 1 ? "" : "s");
            for (int c : box_copies)
                printf(" %d", c);
            printf(" repeat an earlier channel exactly and were taken only after every distinct one)\n");
        }
        printf("  level 1 range");
        for (int c = 0; c < m.c1; c++)
            printf("  ch%d [%.4f, %.4f] -> [%.4f, %.4f]", c, lo_before[(size_t)c], hi_before[(size_t)c],
                   lo_after[(size_t)c], hi_after[(size_t)c]);
        printf("\n               (the BASE plane, before and after block (b); the ridge keeps this bounded when G\n");
        printf("               loses rank. The grid the format stores is fitted over every plane together.)\n");
        if (o.k == 0 && m.planes.size() == 1 && uniform_cw(m))
            printf("  E = centre   E %.17g  centre mse %.17g  relative difference %.3e\n", shipped.e,
                   shipped.centre_mse, relative(shipped.e, shipped.centre_mse));
        if (o.check)
        {
            double worst = relative(after_b.e, twin.e);
            for (size_t i = 0; i < after_b.e_plane.size(); i++)
                worst = std::max(worst, relative(after_b.e_plane[i], twin.e_plane[i]));
            printf("  E check      device %.17g  host brute force %.17g\n", after_b.e, twin.e);
            printf("               relative difference %.3e, worst over E and every plane %.3e  (the bar is 1e-9)\n",
                   relative(after_b.e, twin.e), worst);
            // With --q1-start 1 the grid is frozen before the first continuous solve, so there is no continuous block
            // (b) for either probe to stand beside and neither ran. They say so rather than printing the zero they
            // were initialised with, which would read as a pass.
            if (!fd_done)
                printf("  block (b) fd skipped: --q1-start %d froze level 1's grid before any continuous solve, so "
                       "there was no continuous block (b) to probe\n", o.q1_start);
            else
            {
                printf("  block (b) fd max |dE/dx| h / E over %d values of EACH of the %zu planes %.3e  (the bar is "
                       "1e-5;\n", fd_samples, nplanes, fd_worst);
                printf("               h %g, central, in the plane's own units, E that plane's own normalised "
                       "objective)\n", fd_h);
            }
            if (!fd_done)
                printf("  block (b) dense skipped: the same reason\n");
            else if (dense_done)
                printf("  block (b) dense max |delta device - delta host| %.3e over a range of %.6e, relative %.3e"
                       "  (the bar is 1e-6, over the %d plane%s small enough to afford the dense solve)\n",
                       dense_worst, dense_range, dense_range > 0.0 ? dense_worst / dense_range : 0.0, dense_planes,
                       dense_planes == 1 ? "" : "s");
            else
                printf("  block (b) dense skipped: every plane has more than %d unknowns\n", dense_cap);
        }
        printf("  mip weights  --mip-weight %s --mip-mix %g --k %d: share is the plane's part of the decoder's\n",
               o.mip_weight.c_str(), o.mip_mix, o.k);
        printf("               least-squares mass, w what one of its sites carries (pixels makes every w equal)\n");
        if (o.mip_weight == "pixels" && m.planes.size() > 1)
        {
            double base_px = (double)m.planes[0].w0 * m.planes[0].h0, chain_px = 0.0;
            for (size_t i = 1; i < m.planes.size(); i++)
                chain_px += (double)m.planes[i].w0 * m.planes[i].h0;
            printf("               pixels equalises w across the CHAIN; the base's share is 1 - mix under every rule,\n");
            printf("               so the base joins the chain's w exactly at --mip-mix %.6f\n",
                   chain_px / (base_px + chain_px));
        }
        for (size_t i = 0; i < m.planes.size(); i++)
            printf("    M%zu  %dx%d  sites %zu  share %.6f  w %.9e\n", i, m.planes[i].w0, m.planes[i].h0,
                   (size_t)m.planes[i].w0 * m.planes[i].h0 * (1 + o.k), mip_share[i], mip_site[i]);
        printf("  size         lat0 %zu + lat1 %zu + json %zu = %zu bytes\n", sizes[0], sizes[1], sizes[2],
               sizes[0] + sizes[1] + sizes[2]);
        {
            // The in-memory bitrate once the asset is loaded into D3D11: the .dds payloads are exactly what the GPU
            // holds, so the bytes of each level's texture summed over the actual plane sizes give the bits per source
            // pixel. Reported for the set (every texture shares the two latents) and per material texture.
            //
            // A BLOCK-COMPRESSED level counts WHOLE 4x4 BLOCKS - ceil(w/4) * ceil(h/4) of them at 8 bytes (BC4) or 16
            // (BC5) - and not w * h * bytes_per_texel. The two agree only when both dimensions are multiples of 4. A
            // 40x40 crop has a 10x10 plane at the deep end of the chain, which is nine blocks per axis and not six and a
            // quarter, and the chain figure came out 18.33 bpp against the 18.66 the files actually cost. An
            // uncompressed plane is one byte per stored channel per texel, with no such rounding.
            const int l0_block_bytes = o.bc0 != 0 ? (m.c0 == 1 ? 8 : (m.c0 == 2 ? 16 : (m.c0 == 3 ? 24 : 32))) : 0;   // BC4 / BC5 / BC5 + BC4 / two BC5s
            const double l0_bpt = (double)(m.c0 == 3 ? 4 : m.c0);   // the uncompressed plane's bytes per texel
            const double l1_bpt = (double)(m.c1 == 3 ? 4 : m.c1);
            double base_bytes = 0.0, all_bytes = 0.0, base_l0_bytes = 0.0;
            for (size_t i = 0; i < m.planes.size(); i++) {
                const PlaneSize& p = m.planes[i];
                const double b0 = l0_block_bytes
                                      ? (double)((p.w0 + 3) / 4) * ((p.h0 + 3) / 4) * l0_block_bytes
                                      : l0_bpt * p.w0 * p.h0;
                const double b = b0 + l1_bpt * p.w1 * p.h1;
                if (i == 0) { base_bytes = b; base_l0_bytes = b0; }
                all_bytes += b;
            }
            const double px = (double)m.planes[0].w0 * m.planes[0].h0;
            const double set_base = 8.0 * base_bytes / px, set_all = 8.0 * all_bytes / px;
            const double l0_base = 8.0 * base_l0_bytes / px;   // level 0 alone at the base, per source pixel
            printf("  memory       %.2f bpp for the set at the base (level 0 %.2f + level 1 %.2f), %.2f with the mip chain;\n",
                   set_base, l0_base, set_base - l0_base, set_all);
            printf("               %.2f bpp per texture at the base, %.2f with mips (%d texture%s the two latents)\n",
                   set_base / m.textures, set_all / m.textures, m.textures, m.textures == 1 ? " shares" : "s share");
        }
        printf("  rounds       %d (stopped on %s)\n", rounds_used, stop_reason);
        printf("  time         %.3f s total, (a) %.3f ms / (b) %.3f ms / (c) %.3f ms / E %.3f ms over %lld passes\n",
               seconds, ms_a, ms_b, ms_c, be::objective_total_ms(), be::objective_passes());
        if (be::backend_selected() != be::Backend::Cpu)
        {
            printf("               the three splits are the blocks' own spans over the whole run, CUDA events around the\n");
            printf("               whole block: every plane of it, on its own stream, from the first launch to the last.\n");
        }
        else
        {
            printf("               the three splits are the blocks' own spans over the whole run, the wall clock around\n");
            printf("               the whole block: every plane of it, walked one at a time, from the first to the last.\n");
        }
        printf("               The total is the wall clock of the encode, which also carries the E passes, the file\n");
        printf("               I/O and the host's own work. E pass %.3f ms (%.3f ms the first, cold), the init's\n",
               after.ms, before.ms);
        printf("               block (a) %.3f ms.\n", solve_ms);
        {
            // Which rung of the ridge ladder the run's block (a) calls ended on. On an ordinary image every call takes
            // the shipped ridge, and the line then says in one place what the E column only implies: the decoder is
            // the one the same source always produced. A reduced, none or refused is the near-singular arm, and a
            // reader chasing an odd asset should see it without re-running under a debugger.
            long long rung[4];
            be::decoder_ridge_tally(rung);
            printf("  block (a) ridge: %lld standard, %lld reduced, %lld none, %lld refused  (the rung of the ridge\n",
                   rung[0], rung[1], rung[2], rung[3]);
            printf("               ladder each call ended on; anything but all-standard is a near-singular matrix)\n");
        }
        printf("  device       %.1f MB: the source chain, both latents at every level, and block (b)'s workspace,\n",
               (double)device_bytes / (1024.0 * 1024.0));
        printf("               which every plane owns its own of so that the planes can be solved at the same time\n");
        // Where blocks (b) and (c) went, plane by plane. On CUDA each figure is that plane's own stream, from the first
        // event of a block to the last, and the planes run TOGETHER: they overlap each other, so their sum is larger
        // than the block's own span above and only their shape is worth reading. The base's figure is close to the
        // whole block because it is the plane the block waits on; the chain, a third of the texels, rides along behind
        // it. The CPU backend walks the planes one at a time, so the same figures SUM to the block instead of
        // overlapping it, and the percentages below them mean more rather than less.
        if (be::backend_selected() != be::Backend::Cpu)
        {
            printf("  per plane    (b) and (c) over the whole run, each plane's own stream, CUDA events. The planes are\n");
            printf("               issued together, so these overlap and their sum exceeds the block's own time:\n");
        }
        else
        {
            printf("  per plane    (b) and (c) over the whole run, each plane's own phases on the wall clock. The\n");
            printf("               planes are walked one at a time, so these SUM to the block's own time:\n");
        }
        for (size_t i = 0; i < nplanes; i++)
            printf("    M%zu  %dx%d  (b) %8.3f ms  (c) %8.3f ms\n", i, m.planes[i].w0, m.planes[i].h0, ms_b_plane[i],
                   ms_c_plane[i]);
        if (nplanes > 1)
        {
            double base_px = (double)m.planes[0].w0 * m.planes[0].h0, chain_px = 0.0;
            for (size_t i = 1; i < nplanes; i++)
                chain_px += (double)m.planes[i].w0 * m.planes[i].h0;
            printf("    the base's stream is %.1f %% of block (b) and %.1f %% of block (c); the chain holds %.1f %% "
                   "of the base's texels\n",
                   100.0 * ms_b_plane[0] / (ms_b > 0.0 ? ms_b : 1.0), 100.0 * ms_c_plane[0] / (ms_c > 0.0 ? ms_c : 1.0),
                   100.0 * chain_px / base_px);
        }
    }

    // --backend check: the harness's per-function summary, after every dispatched call the run made (the report's own
    // included), and a MISMATCH anywhere is the run's failure. Outside check mode this is a quiet true.
    if (!be::check_finish())
        return 1;
    return 0;
}

// Every failure the encoder foresees prints its own ERROR and returns 1. This is the net under the ones it does not: a
// std::bad_alloc from any of the host's buffers (the source, its padding and chain, the reconstructions - a large
// enough material on a small enough machine), and anything else the C++ runtime throws, on this thread or carried here
// from a CPU worker (cpu/pool.h). Without it the runtime terminates the process with no ERROR line at all. What the OS
// does to a process it kills outright is out of any program's reach.
//
// On these paths the CUDA device model and the CPU pool are left for the driver and the OS to reclaim at exit, on
// purpose: releasing them while unwinding from a failed allocation is more code that can fail, for nothing.
int main(int argc, char** argv)
{
    try
    {
        return encode(argc, argv);
    }
    catch (const std::length_error&)
    {
        fflush(stdout);
        fprintf(stderr, "ERROR: out of host memory while %s (a buffer larger than the C++ library allows): the "
                        "material is too large for this machine's memory; try fewer or smaller textures\n", g_stage);
    }
    catch (const std::bad_alloc&)
    {
        fflush(stdout);
        fprintf(stderr, "ERROR: out of host memory while %s: the material is too large for this machine's memory; try "
                        "fewer or smaller textures\n", g_stage);
    }
    catch (const std::exception& e)
    {
        fflush(stdout);
        fprintf(stderr, "ERROR: unexpected failure while %s: %s\n", g_stage, e.what());
    }
    catch (...)
    {
        fflush(stdout);
        fprintf(stderr, "ERROR: unexpected failure while %s (an unknown exception)\n", g_stage);
    }
    return 1;
}
