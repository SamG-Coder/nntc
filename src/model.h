// model.h: the representation, and the host-callable entry points of the device code.
//
// The representation is two quantised latent textures plus one affine decoder:
//
//   level 0   full resolution, C0 channels, bits0[c] bits each, on the palette the SHIPPED format decodes to:
//             pal_c(k) = -1 + 2 k / (2^b - 1) block-compressed, -1 + 2 rep(k) / 255 uncompressed (level0_value below)
//   level 1   a quarter of the resolution per axis (BLOCK = 4), C1 channels, bits1 bits each, on the per-channel grid
//             g_c(k) = lo_c + rep(k) / 255 * (hi_c - lo_c), rep(k) the index replicated over the stored byte
//   decoder   out = W phi(z) + b, with phi in the order the GPU shader builds it:
//                 [a]  c_j          j < C1      (level 1's channels, sampled bilinearly)
//                 [b]  s_i          i < C0      (level 0's channels, read at the texel)
//                 [sc] s_i c_j      i outer, j inner
//             so nin = C1 + C0 + C0 C1 and nout = 3 T.
//
// The dequantisation is affine and is applied AFTER sampling, which is the whole reason a quantised latent can be handed
// to a hardware sampler: the blend of the samples is the blend of the values.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "image.h"
// The merged per-texture settings. They reach this header because the asset's own JSON records what each input was
// and how its chain was derived (source.inputs), and because the source chain is built per texture.
#include "options.h"

// Level 1 is always 1/BLOCK of level 0's resolution per axis, so its sampler always carries MipLODBias = log2(BLOCK).
static const int BLOCK = 4;
static const int LOD_BIAS_LEVEL1 = 2;
static const int MAX_MIP_LEVELS = 12;

// The material's texture cap. It is a compile-time constant in three places that have to agree: the kernels' per-output
// arrays are sized by 3 * MAX_TEXTURES (MAX_NOUT in device.cuh), the report's per-channel direction is sized by it, and
// the viewer's decoder constant buffer carries that many rows of W and the matching bias. The kernels pay for the cap
// in registers whether a run uses it or not; encoding is seconds and the cost was measured and accepted rather than
// templated away.
static const int MAX_TEXTURES = 6;

// One stored mip level: the two latent planes that together decode output mip m (the 1:1 rule).
struct PlaneSize
{
    int w0 = 0, h0 = 0;   // level 0's plane, which is also the decoded extent of this level
    int w1 = 0, h1 = 0;   // level 1's plane
};

struct Decoder
{
    int nin = 0;
    int nout = 0;
    std::vector<float> w;   // nout x nin, row-major
    std::vector<float> b;   // nout
};

struct Model
{
    int textures = 1;         // T
    int nout = 3;             // 3 T
    int width = 0, height = 0;   // the padded extent, which is level 0's base
    int source_width = 0, source_height = 0;
    bool padded = false;

    int c0 = 1;
    std::vector<int> bits0;   // one entry per level-0 channel
    int c1 = 2;
    int bits1 = 8;

    std::vector<PlaneSize> planes;   // the base first

    bool bc0 = true;          // the shipped level-0 format, which decides the palette (level0_value)
    // --l0 bc8: level 0 is not an index on a fixed [-1,1] palette at all but a CONTINUOUS plane at 8 bits on its own
    // per-channel lo/hi grid - exactly level 1's arrangement - which is BC4 / BC5-encoded after the last round and then
    // GOES ON BEING OPTIMISED AS BC: every block refined against the decoder's own error and repacked in outer passes,
    // so the .dds holds blocks that were chosen, not a compressed copy of the continuous plane. palette0 is empty in
    // that mode and lo0 / hi0 carry the grid instead.
    bool l0_bc8 = false;
    // --init0 residual: level 0 is seeded one channel at a time from the principal direction of the residual, which
    // needs a full-resolution residual buffer on the device. The encode allocates it only when the init asks for it.
    bool init0_residual = false;
    // --init0 texture: the direction is ONE TEXTURE's own leading residual direction rather than the whole material's,
    // so a full-resolution channel is dedicated to a texture instead of being a blend of all of them.
    bool init0_texture = false;
    // --init0-scope chain: the residual covariance that direction comes from is accumulated over EVERY stored plane
    // with the objective's own per-site weight, not over the base plane alone.
    bool init0_chain = false;
    std::vector<std::vector<float>> palette0;   // per level-0 channel, 2^bits0[c] entries, for THAT format
    std::vector<float> lo0, hi0;                // per level-0 channel under --l0 bc8, shared by the whole chain
    std::vector<float> lo1, hi1;                // per level-1 channel, shared by the whole chain (the format carries one pair)

    Decoder dec;

    std::vector<float> cw;   // per output channel: weight[t] * rgb_weights[t][c % 3] * 3 / sum(rgb_weights[t])

    // What each input WAS, as the run finally merged it. Nothing in the solve reads this; it is carried so that the
    // asset's JSON can say which input was a normal map and whether its deeper levels were derived in linear light,
    // which a consumer cannot recover from the two latent textures.
    std::vector<TextureSettings> inputs;

    // The quantisation indices as they are written to the .dds, downloaded from the device after each solve.
    std::vector<std::vector<uint8_t>> k0;   // per plane, w0*h0*c0
    std::vector<std::vector<uint8_t>> k1;   // per plane, w1*h1*c1

    // THE BLOCKS THE FILE HOLDS, under --l0 bc8: per plane, per level-0 .dds file (one for 1-2 channels, two for 3-4),
    // the BC4 / BC5 blocks the refinement chose. They are the level-0 plane - k0 above is then only the nearest 8-bit
    // index to each decoded byte, which is what the uncompressed twin of --bc0 both would store. Empty means no pack
    // has been made yet and the writer packs from k0 itself, which is the palette mode's whole story.
    std::vector<std::vector<std::vector<uint8_t>>> bc0_blocks;

    int levels1() const { return (1 << bits1) - 1; }
    // Level 0's index range under --l0 bc8 is the 8-bit one, and its grid rule is then the sampled-grid rule with
    // rep(k) = k: value = lo + k / 255 * (hi - lo), which is what a BC4 block's decoded byte is read as too.
    int levels0() const { return 255; }
};

// THE GRID IS A FUNCTION OF THE SHIPPED FORMAT.
//
// The encoder must fit the values the consumer's sampler actually returns, not the values the index nominally stands
// for. A UNORM8 texel is presented to the shader as byte / 255, and the byte an uncompressed plane stores is the index
// with its bits replicated, so the value of index k on an uncompressed plane is
//
//     value(k) = lo + rep(k) / 255 * (hi - lo) ,   rep(k) = the index replicated over the byte (below)
//
// which equals lo + k / levels * (hi - lo) only at 1, 2, 4 and 8 bits, where rep(k) / 255 IS k / levels. At 3, 5, 6 and
// 7 bits the two differ by up to half a byte - 0.0034 of a [-1,1] level-0 channel at 3 bits - and fitting the second
// while shipping the first is a systematic per-entry bias in every decode.
//
// A block-compressed level 0 is the other case: BC4's palette at 1-3 bits is the standard k / (2^bits - 1) of full
// scale exactly (the endpoints are the mode's own extremes and the selector is a relabelling of the index, see
// bc_pack.h), so there the value IS lo + k / levels * (hi - lo). At 4 bits k * 17 / 255 = k / 15 and the two rules
// coincide anyway, and the pack is lossy at that depth in any case.
//
// So: uncompressed -> the replicated byte, block-compressed -> the palette. Level 1 is always uncompressed.
#ifdef __CUDACC__
#define NNTC_HOST_DEVICE __host__ __device__
#else
#define NNTC_HOST_DEVICE
#endif

// The index in the byte's top bits, replicated into the low bits: 3 bits give k k k, 4 bits give k * 17, 8 bits give k.
NNTC_HOST_DEVICE inline int replicate_index(int k, int bits)
{
    unsigned b8 = 0;
    for (int sh = 8 - bits; sh > -bits; sh -= bits)
        b8 |= sh >= 0 ? ((unsigned)k << sh) : ((unsigned)k >> -sh);
    return (int)(b8 & 0xFFu);
}

// Level 1's dequantisation and its inverse, written once for the host and the device:
//
//     value = lo + rep(k) / 255 * (hi - lo) ,   k = the index whose value is nearest (the grid is monotone)
//
// Every path that turns an index into a value - the snap, the quantised sweeps, the writer's own round trip - goes
// through this one function, because the asset's whole promise is that the value the solver measured and the value the
// index stands for are the same number. The arithmetic is double and the result is rounded to float once. The device
// spells the multiply and the add out as separately rounded operations: nvcc would otherwise contract them into one
// fused multiply-add, which the host has no counterpart for, and the two would then disagree in the last bit often
// enough to matter over a million values.
NNTC_HOST_DEVICE inline float level1_value(float lo, float hi, int bits, int k)
{
    const double span = (double)hi - (double)lo;
    const double frac = (double)replicate_index(k, bits) / 255.0;
#ifdef __CUDA_ARCH__
    return (float)__dadd_rn((double)lo, __dmul_rn(frac, span));
#else
    return (float)((double)lo + frac * span);
#endif
}

// The nearest grid VALUE, not the nearest index: rep(k) / 255 is not k / levels, so rounding the index would land one
// step off wherever the two rules disagree by more than half a step of the difference. The grid is strictly monotone,
// so the answer is within one of the rounded index and three comparisons settle it; a tie keeps the lower index, on
// both the host and the device, because the same bits have to come out of both.
NNTC_HOST_DEVICE inline int level1_index(float lo, float hi, int bits, float value)
{
    const double span = (double)hi - (double)lo;
    const int levels = (1 << bits) - 1;
    if (!(span > 0.0))
        return 0;
    double x = ((double)value - (double)lo) / span * (double)levels;
    x = x < 0.0 ? 0.0 : (x > (double)levels ? (double)levels : x);
    int k = (int)(x + 0.5);
    k = k < 0 ? 0 : (k > levels ? levels : k);
    int best = k;
    double best_d = fabs((double)value - (double)level1_value(lo, hi, bits, k));
    for (int d = -1; d <= 1; d += 2)
    {
        const int n = k + d;
        if (n < 0 || n > levels)
            continue;
        const double dist = fabs((double)value - (double)level1_value(lo, hi, bits, n));
        if (dist < best_d || (dist == best_d && n < best))
        {
            best_d = dist;
            best = n;
        }
    }
    return best;
}

// Level 0's palette: the value of index k of a channel of `bits` bits, on [-1, 1], under the format the channel is
// SHIPPED in. bc says the plane is written as BC4 / BC5, where the decoded byte is the standard palette entry
// k * 255 / (2^bits - 1) and the sampler returns k / (2^bits - 1) of full scale; otherwise the plane is an
// uncompressed UNORM8 and the sampler returns the replicated byte over 255.
inline float level0_value(int bits, int k, bool bc)
{
    const int levels = (1 << bits) - 1;
    const double frac = bc ? (double)k / (double)levels : (double)replicate_index(k, bits) / 255.0;
    return (float)(-1.0 + 2.0 * frac);
}

inline void build_level0_palette(const std::vector<int>& bits, int c0, bool bc, std::vector<std::vector<float>>& pal)
{
    pal.assign((size_t)c0, std::vector<float>());
    for (int c = 0; c < c0; c++)
    {
        const int levels = (1 << bits[(size_t)c]) - 1;
        pal[(size_t)c].resize((size_t)levels + 1);
        for (int k = 0; k <= levels; k++)
            pal[(size_t)c][(size_t)k] = level0_value(bits[(size_t)c], k, bc);
    }
}

inline int feature_count(int c0, int c1)
{
    return c1 + c0 + c0 * c1;
}

// The number of extra mip levels: halve while both of level 0's dimensions stay at or above min_size.
inline int mip_count(int w, int h, int min_size)
{
    int n = 0;
    while (n < MAX_MIP_LEVELS && w / 2 >= min_size && h / 2 >= min_size)
    {
        w /= 2;
        h /= 2;
        n++;
    }
    return n;
}

// mips.cpp: the plane layout, the source chain and the mip weights.
//
// The chain is per texture now, so it takes the merged settings. False means the vendored resizer refused a level and
// said which one, and the caller gives up rather than fitting a plane against zeros.
void build_plane_sizes(Model& m, int mip_min, bool want_mips);
bool build_source_chain(const Image& base, const std::vector<PlaneSize>& planes,
                        const std::vector<TextureSettings>& settings, std::vector<Image>& chain);
void build_mip_weights(const std::vector<PlaneSize>& planes, const std::string& kind, double mix, int k,
                       std::vector<double>& share, std::vector<double>& per_site);

// The device side. DeviceModel is opaque to the host translation units; device.cuh defines it.
struct DeviceModel;

struct DeviceInfo
{
    std::string name;
    int major = 0;
    int minor = 0;
};

bool device_select(int index, DeviceInfo& info);
DeviceModel* device_create(const Model& m, const std::vector<Image>& source_chain);
void device_destroy(DeviceModel* d);

// init.cu
//
// Level 0's initialisation. Under --init0 luma channel 0 is the source luminance (snapped to its palette in the
// palette mode) and every further channel starts at zero; under --init0 residual EVERY channel starts at zero here and
// init0_residual_channel seeds them one at a time from the residual, after the decoder has been fitted once.
void init_level0(DeviceModel* d, Model& m, const float* luma_weights, bool luma_channel0);

// Every byte the encode holds on the GPU: the source chain, both latents at every level, and block (b)'s per-plane
// workspace, which is the largest part of it and the price of solving every plane at the same time.
size_t device_memory(const DeviceModel* d);
// Level 1's initialisation. The pca init projects each texel's block mean of the 3T source channels onto the plane's
// first C1 principal directions; the box init takes the block mean of source channel j as channel j. The report carries
// the variance along each kept direction, so a layout with more level-1 channels than the source has directions shows
// up in the banner as a principal value at zero rather than as a channel that quietly does nothing.
struct Level1InitReport
{
    bool pca = false;
    int components = 0;                     // the directions actually kept: min(C1, 3T)
    double eigenvalue[4] = {};              // the variance along each, largest first (4 is the level-1 channel cap)
    double peak[4] = {};                    // the largest |value| each kept component reached before it was scaled by
                                            // it, which is the divisor that puts the channel on [-1,1] without clipping
    double total_variance = 0.0;            // the covariance's trace, so a direction can be read as a share of it
};

void init_level1(DeviceModel* d, Model& m, bool pca, Level1InitReport& rep);

// ONE LEVEL-0 CHANNEL SEEDED FROM THE RESIDUAL (--init0 residual).
//
// With the decoder as it stands, the residual r(p) = out(p) - target(p) is what the representation still gets wrong at
// the base plane's own texel centres, one vector of nout numbers per pixel. Its nout x nout covariance over the base
// has a leading eigenvector e - the one direction of the material the decode is missing most - and the channel is that
// residual projected onto it, r(p) . e, divided by the largest magnitude the projection reaches over the WHOLE chain so
// that the channel fills [-1,1] and nothing is clipped. The caller then refits the decoder, which is the deflation: what
// the new channel can explain leaves the residual through the refit rather than by a subtraction, so the next channel's
// covariance is taken on what is genuinely left.
//
// e is computed on the BASE plane and used for every plane of the chain, for the reason the level-1 pca init has: one
// decoder is shared by every stored level, so channel k has to mean the same direction of the material at every level.
// The seed's divisor in the palette mode: the percentile of |projection| the channel is scaled by, and the resolution
// of the histogram it is read off. The peak itself is one texel's extreme value and a residual's projection is
// heavy-tailed, so the peak leaves the channel's mass inside a small part of the fixed [-1,1] palette; under --l0 bc8
// the freeze refits lo/hi and the scale is a gauge, under --l0 palette there is no such freeze. init.cu's (4a') states
// the measurement. 4096 bins resolve the divisor to 0.024 % of the peak, which is finer than the tail being cut, and
// 99 % measured best of 99.9 / 99 / 98 / 95 / 90 on model10 at both C0 1 and C0 2 (docs/RESULTS.md 6.2).
static const double INIT0_PERCENTILE = 0.99;
static const int PROJ_HIST_BINS = 4096;

struct Init0ChannelReport
{
    double eigenvalue = 0.0;        // the variance along the residual's leading direction
    double total_variance = 0.0;    // the residual covariance's trace, so the eigenvalue reads as a share of it
    double share = 0.0;             // eigenvalue / total_variance
    double peak = 0.0;              // the largest |projection| over the whole chain
    double divisor = 0.0;           // what the projection was actually divided by: the peak under --l0 bc8, the
                                    // INIT0_PERCENTILE percentile of |projection| under --l0 palette
    double dir[3 * MAX_TEXTURES] = {};   // e itself, one entry per output (the cap is 3 channels of MAX_TEXTURES)
    int nout = 0;                   // how many of those entries are real
    int texture = -1;               // under --init0 texture, the texture the direction was taken from; -1 for the
                                    // joint direction of --init0 residual, which belongs to no one texture
};

// `plane_weight` is mips.cpp's per-site weight, one per stored plane: it is what --init0-scope chain accumulates the
// residual covariance with, so that the direction explains the error as the objective itself weighs it. The base-only
// scope ignores it.
void init0_residual_channel(DeviceModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep);

// Level 1's grid, fitted from the continuous plane under one of two range policies:
//
//   minmax  lo and hi are the smallest and largest value of the channel over the planes the fit sees;
//   pct     they are the 0.1 % and 99.9 % percentiles of those same values, so that a handful of outliers cannot spend
//           the whole grid on a range nothing else occupies; a value outside the range clamps onto the end of the grid.
//
// Both entry points fit the one grid the format carries - one lo/hi pair per channel for the whole chain - over the base
// AND every chain plane together, and snap every plane onto it. quantise_level1 is the round-at-the-end control
// (--q1-start 0), called once after a wholly continuous loop; freeze_level1_grid is the --q1-start R0 path, called at
// round R0, after which the quantised sweeps re-optimise every plane on the frozen grid.
void quantise_level1(DeviceModel* d, Model& m, const std::string& range_policy);
void freeze_level1_grid(DeviceModel* d, Model& m, const std::string& range_policy);

// The same two entry points for level 0 under --l0 bc8, where level 0 is a continuous plane on a per-channel grid of
// its own at 8 bits. One grid per channel for the whole chain, for the same reason level 1 has one: the JSON carries
// exactly one lo/hi pair per channel per latent texture, because the GPU dequantises with one scale and bias whatever
// mip the sampler chose. Both are refused in the palette mode, where level 0 has no such grid.
void quantise_level0(DeviceModel* d, Model& m, const std::string& range_policy);
void freeze_level0_grid(DeviceModel* d, Model& m, const std::string& range_policy);

// The snap alone, onto the grid the model already carries: no range is fitted and lo0 / hi0 do not move. The outer
// repack loop needs it under --q1-start 0, where level 0's grid was fitted once after the loop and every later pass
// has to land on THAT grid - a pass that re-fitted the range would move the grid under the pack it is being compared
// against, and the two passes' shipped E would not be two numbers on one scale.
void snap_level0_on_grid(DeviceModel* d, Model& m);

// solve_level1.cu: the count of monomials the stencil's quadratic form in the level-0 sample has, which is the number
// of fixed C1 x C1 matrices the assembly carries. Declared here because device_create allocates them.
int stencil_monomials(int c0);

// solve_decoder.cu
double solve_decoder(DeviceModel* d, Model& m, int k, const std::vector<double>& per_site);

// How the run's block (a) calls ended, counted since the process started: [0] took the shipped ridge, [1] the reduced
// one, [2] no ridge at all, [3] refused every rung and kept the decoder it came in with. The report prints the four,
// because which rung a run lands on is the difference between "the decoder it always got" and the near-singular arm,
// and nothing else in the output says so.
void decoder_ridge_tally(long long out[4]);

// solve_level1.cu: block (b), the level-1 plane of one stored level as one sparse least squares.
struct Level1Report
{
    int iterations = 0;       // conjugate-gradient iterations taken
    double residual = 0.0;    // the final relative residual |(H + lambda I) delta + grad| / |grad|
    double lambda = 0.0;      // the proximal ridge actually used: --ridge times mean(diag H)
    double min_diag = 0.0;    // the smallest and largest diagonal entry of H over the plane; a small minimum is the
    double max_diag = 0.0;    // signature of a texel with a null direction (see the file's header)
    // The plane's own kernels, its own event pair. The planes run together, so these OVERLAP: their sum is the work
    // the block did and the block's own span is how long it took.
    double ms_assemble = 0.0, ms_precond = 0.0, ms_cg = 0.0;

    // The movement probe, reduced on the device so a continuous round costs no download of the plane: values whose
    // correction exceeded half a grid step of the plane's own range, the square norms of the correction and of the
    // plane, and the plane's per-channel range after the solve (4 is the level-1 channel cap).
    long long moved_values = 0;
    double delta2 = 0.0, plane2 = 0.0;
    float lo[4] = {}, hi[4] = {};

    // The quantised path (sweep_level1) instead of the continuous one: the sweeps taken, how many stored values changed
    // their grid index over all of them, and the sweeps' own kernel time.
    bool quantised = false;
    int sweeps = 0;
    long long moved = 0;
    double ms_sweeps = 0.0;
};

// Block (b) on EVERY plane, each on its own stream and all of them waited for once. With the decoder held the planes
// share no variable, so the M + 1 problems of a round are one block of independent work.
double solve_level1_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);

// Block (b) on a frozen grid: the same assembled stencil and gradient, then `sweeps` four-colour Gauss-Seidel passes
// whose per-value step is the exact argmin over the grid. Every value it writes is a grid value, so the plane stays
// storable throughout and the .dds needs no rounding at the end.
double sweep_level1_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                        std::vector<Level1Report>& rep);

// BLOCK (c') UNDER --l0 bc8: the same two entry points over LEVEL 0's plane instead of level 1's. The unknown is then a
// continuous 8-bit plane at the decoded extent, its stencil is a 9-point one over level-0 texels, and the quadratic
// form the assembly expands is in the level-1 sample rather than the level-0 one; everything downstream - the ridge,
// the block-Jacobi preconditioner, the conjugate gradients, the movement probe and the four-colour sweeps with the
// exact per-channel grid argmin - is the same code (solve_level1.cu). The palette mode never calls either.
double solve_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep);
// Block (c')'s assembly on its own: the stencil and the gradient of level 0's plane at the plane as it stands, with the
// correction measured from it and started at zero. It is what the BC refinement minimises over (refine_bc.cu), and it
// is assembled with NO proximal ridge: the refinement's steps are exact minimisations of the objective's own quadratic
// and are accepted on its own decrease, so there is nothing for a proximal term to stabilise.
double assemble_level0_for_refine(DeviceModel* d, const Model& m, int k);

void level1_download_indices(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k1);
// The two planes and their indices written back to the device, which is what the outer repack loop's restore needs:
// a rejected pass must leave the device holding exactly the planes the accepted one did.
void level1_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v1);
void level1_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k1);
void level0_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k0);
void upload_decoder_to_device(DeviceModel* d, const Model& m);
void upload_grids_to_device(DeviceModel* d, const Model& m);
void level1_download(DeviceModel* d, const Model& m, int plane, std::vector<float>& v);
void level1_delta(DeviceModel* d, const Model& m, int plane, std::vector<double>& delta);
void level1_poke(DeviceModel* d, int plane, size_t index, float value);
void level1_range(DeviceModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi);
bool solve_level1_dense_host(DeviceModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns);

// solve_level0.cu: block (c), the level-0 plane of one stored level as one exact search per texel.
struct Level0Report
{
    long long moved[4] = { 0, 0, 0, 0 };   // texels changed by each of the four colour passes
    long long moved_total = 0;
    double ms = 0.0;                       // the four kernels together, CUDA events
    bool joint = false;                    // the argmin used: the joint enumeration, or --sweeps coordinate sweeps
    long long states = 0;                  // the joint enumeration's state count, so the banner can say what it costs
};

// Block (c) on EVERY plane, each on its own stream and all of them waited for once, for the same reason block (b) is.
double solve_level0_all(DeviceModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep);
void level0_download(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k0);

// One plane's level-0 values written back to the device, so that a report and a decode can describe a plane the solver
// did not itself choose: the plane as the block format decodes it, or the same indices on the other format's palette.
void level0_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v0);

// One plane's level-0 values as the device holds them, which under --l0 bc8 is what the grid is fitted over and what
// the continuous solve leaves behind.
void level0_download_values(DeviceModel* d, const Model& m, int plane, std::vector<float>& v0);

// objective.cu
void decode_plane(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8);

// One evaluation of the objective over every site of every plane. Every number here is normalised so that it reads
// like a mean squared error in [0,1] units: a plane's E is its weighted squared error divided by its site count times
// sum(cw), and the whole E divides by sum over planes of w_m N_m (1 + K) times sum(cw).
struct Objective
{
    double e = 0.0;                 // the whole objective, normalised
    std::vector<double> e_plane;    // the same for each plane on its own, ignoring w_m
    double centre_mse = 0.0;        // the base plane's centre sites, every output, unweighted, in fp units
    double sampled_mse = 0.0;       // the base plane's K fractional sites, every output, unweighted
    // Every plane's centre-site mean squared error on its own, unweighted, in the same fp units: plane m decoded at its
    // own texel centres against level m of the run's own source chain. The progress line reads the per-level quality
    // off this, so a round costs no second decode to report it. Entry 0 is the base and equals centre_mse.
    std::vector<double> centre_plane;
    double ms = 0.0;                // the kernel time, CUDA events around the whole pass
};

void objective_eval(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);

// Every objective pass the run has made, and how long they took together: the measurement's own share of the encode,
// which the report prints beside the three blocks because it is neither free nor part of any of them.
double objective_total_ms();
long long objective_passes();
void objective_check_host(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);

// export.cpp
//
// One stored level of level 0 as the values a consumer's sampler returns: `bc` picks the palette of the format the
// plane is shipped in, and `packed` asks for the plane actually block-compressed and decoded again rather than for the
// indices on their palette - which is what the file holds once the pack is lossy, at 4 bits and up.
// `packed` also reports what the pack cost: the squared error in byte units against the exact index values and the
// channel-texels it was measured over, which is the packing PSNR the caller prints. Both are added to, not assigned,
// so a caller can accumulate a whole chain; pass nullptr for neither.
void level0_plane_values(const Model& m, int plane, bool bc, bool packed, std::vector<float>& v0,
                         double* pack_se = nullptr, size_t* pack_n = nullptr);

// The stb_dxt-style pack of one plane's level 0 as the two endpoints and the sixteen selectors of every block of every
// channel - bc_pack.h's own pack, taken apart rather than re-derived, so the refinement starts from exactly the bytes
// the refinement starts from. ep is 2 bytes per block-channel and sel 16, block-major then channel.
void level0_pack_seed(const Model& m, int plane, std::vector<uint8_t>& ep, std::vector<uint8_t>& sel);

// The reverse: the .dds's block bytes of one plane, one entry per level-0 file, built from endpoints and selectors the
// refinement chose. The squared error in byte units against the exact index values of m.k0 and the channel-texels it
// was measured over are ADDED to se and n, which is the packing psnr the caller prints - the same definition
// bc_pack_level reports for the seed.
void level0_pack_blocks(const Model& m, int plane, const std::vector<uint8_t>& ep, const std::vector<uint8_t>& sel,
                        std::vector<std::vector<uint8_t>>& files, double& se, size_t& n);

// refine_bc.cu: the post-fit BC pack refined by analysis by synthesis. One entry per refinement pass.
struct BcRefineReport
{
    int pass = 0;
    long long blocks = 0;     // the 4x4 blocks of every plane of the chain
    long long improved = 0;   // those that accepted at least one step
    double delta_e = 0.0;     // the sum of the accepted decreases, in the objective's own quadratic units (negative)
    double ms = 0.0;          // the pass's four colour launches over every plane, CUDA events
};

double bc_pack_prepare(DeviceModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels);
double bc_refine(DeviceModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels);
void bc_blocks_from_device(DeviceModel* d, Model& m, double& pack_psnr, size_t& pack_texels);

// The refinement's own DEVICE state - every plane's endpoints and selectors - saved and put back. The outer repack loop
// restores a rejected pass, and the model's block bytes are not the whole of what a pass moved: p.bc_ep and p.bc_sel are
// what a later refinement step continues from, so a rejected pass's left there would refine blocks the asset does not
// hold. Nothing reads them after the loop as the code stands; this is what keeps that from having to stay true.
void bc_state_save(const DeviceModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel);
void bc_state_restore(DeviceModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel);
void reconstruct_levels(DeviceModel* d, const Model& m, std::vector<std::vector<uint8_t>>& recon);
bool write_level_pngs(const Model& m, const std::vector<std::vector<uint8_t>>& recon,
                      const std::vector<Image>& source_chain, const std::string& prefix);
// `quiet` is --quiet: it drops the pack lines, which are progress, and keeps the `wrote` lines, which say what the
// run actually produced and are never suppressed.
bool write_asset(const Model& m, const std::string& prefix, const std::string& json_name, bool bc0,
                 size_t sizes[3], bool quiet);

// The plane as the device holds it against the indices the .dds will store: every value must be exactly the grid value
// of its own index, which is what the quantised phase promises and what makes the writer's round trip meaningful.
bool verify_level1_on_grid(const Model& m, const std::vector<std::vector<float>>& v1, size_t& values, size_t& offgrid);
