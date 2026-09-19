// cpu/cpu_init.cpp: the initial state of both latents, and both latents' grids, on the CPU (docs/CPU_BACKEND_PLAN.md
// stage C3, the port of init.cu's ten kernels and their drivers).
//
// init.cu's header is the WHY of every rule below and is not repeated: level 0 starts at luma or at zero and is seeded
// channel by channel from the residual (init0_residual_channel); level 1 starts from the block means of the source,
// either mapped channel for channel (box) or projected onto their principal directions (pca); every grid is fitted over
// the base and every chain plane together and the planes are snapped onto it. This file is a transcription. What
// differs, and why:
//
//   * THE REDUCTIONS. k_cov_partial reduces each covariance entry over a fixed grid of SITE_BLOCKS blocks of 256
//     threads through a shared-memory tree, and the host adds the blocks' partials in index order. cov_partials
//     replays that order exactly (grid-stride lanes, the halving tree, one task per block), because the residual seed's
//     eigenvector is ill-conditioned: a last-bit difference in the covariance moved it enough to fork a whole material's
//     run by 0.6 dB. The order depends on the plane's size alone, so the result is the same bits at every -j;
//   * THE PEAK AND THE HISTOGRAM. k_proj_peak's maximum and k_proj_hist's integer counts are order-independent, so the
//     CPU forms (a maximum per chunk, a private histogram per chunk, both folded in chunk order) are exactly CUDA's
//     answer for the same projections. The histogram is folded into ONE histogram that spans the whole plane loop, as
//     init.cu memsets the device histogram once and accumulates every plane of the chain into it before reading it;
//   * THE STREAMS AND THE STAGING. Every download and upload the CUDA drivers make through d->scratch or a pinned
//     buffer is a copy between the model's own vectors here, and there is no fan-out: a CPU call's work is finished
//     when it returns.
//
// Everything else - the per-texel arithmetic of all ten kernels in the float or double it is written in (the float
// accumulators of the block means stay float), the grid's one definition (model.h's level1_index / level1_value, called
// and not copied), the nearest-palette-value walk and its lower-index tie rule, lroundf's round-half-away-from-zero, the
// fit's percentiles, the Jacobi sweep, the eigen order and the sign rule, the per-texture argmax, the percentile scan -
// is copied operation for operation, and the host-side parts of the drivers are copied as they stand.
//
// ZERO DATA RACES (section 3.4). Every kernel here is one of two shapes. A per-texel or per-value kernel's chunk c reads
// inputs no chunk of the same run writes and writes only the outputs of its own texels or values - including
// k_quantise_level1 where its input and output are the SAME buffer (the level-0 snaps), which is safe because a chunk
// reads and writes exactly its own elements and no other. A reduction's chunk c reads inputs no chunk writes and writes
// only its own partial slot, which is read after the join, on the calling thread. The host-side steps between runs are
// serial.

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_sample.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace nntc_cpu
{

namespace
{

// The histogram kernel's chunk. Larger than CPU_CHUNK because every chunk carries a private PROJ_HIST_BINS-bin
// histogram until the join, and at CPU_CHUNK a 4096x4096 base plane would hold 4096 of them (128 MB); at this size it
// holds 256 (8 MB). A compile-time constant, like CPU_CHUNK, so the chunks cover the same texels at every -j.
const size_t CPU_HIST_CHUNK = 65536;

// A per-element kernel over [0, n) in CPU_CHUNK chunks: body(i) for every element, each chunk writing only its own.
template <class Body>
void per_element(CpuModel* d, size_t n, const Body& body)
{
    d->pool->run(chunks_for(n, CPU_CHUNK), [&](int c) {
        const size_t begin = (size_t)c * CPU_CHUNK;
        const size_t end = begin + CPU_CHUNK < n ? begin + CPU_CHUNK : n;
        for (size_t i = begin; i < end; i++)
            body(i);
    });
}

// ---------------------------------------------------------------------------------------------------------------------
// The per-texel kernels, copied
// ---------------------------------------------------------------------------------------------------------------------

// k_init_level0: the nearest PALETTE VALUE to 2 luma - 1 on channel 0 (under --init0 luma) and to 0 on every other
// channel, by one walk over the at most sixteen monotone entries; a tie keeps the lower index (strict <).
void init_level0_texel(const float* src, int nout, int c0, const ChannelBits& bits, const float* palette, float lr,
                       float lg, float lb, int luma_channel0, float* v0, uint8_t* k0, size_t i)
{
    const float* s = src + i * nout;
    const float luma = lr * s[0] + lg * s[1] + lb * s[2];
    for (int c = 0; c < c0; c++)
    {
        const float value = (c == 0 && luma_channel0) ? 2.0f * luma - 1.0f : 0.0f;
        const int levels = (1 << bits.b[c]) - 1;
        int k = 0;
        float best = std::fabs(palette[(size_t)c * 16] - value);
        for (int j = 1; j <= levels; j++)
        {
            const float dist = std::fabs(palette[(size_t)c * 16 + j] - value);
            if (dist < best)
            {
                best = dist;
                k = j;
            }
        }
        v0[i * c0 + c] = palette[(size_t)c * 16 + k];
        k0[i * c0 + c] = (uint8_t)k;
    }
}

// k_init_level0_cont: the same values, continuous (--l0 bc8, whose grid does not exist until the freeze).
void init_level0_cont_texel(const float* src, int nout, int c0, float lr, float lg, float lb, int luma_channel0,
                            float* v0, size_t i)
{
    const float* s = src + i * nout;
    const float luma = lr * s[0] + lg * s[1] + lb * s[2];
    for (int c = 0; c < c0; c++)
        v0[i * c0 + c] = (c == 0 && luma_channel0) ? 2.0f * luma - 1.0f : 0.0f;
}

// The footprint of level-1 texel (x, y) in the plane: [x w0 / w1, (x+1) w0 / w1) and the matching span in y, in long
// long, widened to one pixel where the floor halving left it empty and cut at the plane's edge. Shared by
// k_init_level1_box and k_block_means, which spell it out identically in init.cu.
void footprint(int x, int y, int w0, int h0, int w1, int h1, int& xa, int& xb, int& ya, int& yb)
{
    xa = (int)((long long)x * w0 / w1);
    xb = (int)((long long)(x + 1) * w0 / w1);
    ya = (int)((long long)y * h0 / h1);
    yb = (int)((long long)(y + 1) * h0 / h1);
    if (xb <= xa)
        xb = xa + 1;
    if (yb <= ya)
        yb = ya + 1;
    xb = xb > w0 ? w0 : xb;
    yb = yb > h0 ? h0 : yb;
}

// k_init_level1_box: channel c is source channel c's mean over the footprint, mapped by 2 mean - 1, and a channel past
// the source's is 0. THE ACCUMULATOR IS FLOAT, as the kernel's is; a double one would move the last bit of every mean.
void init_level1_box_texel(const float* src, int w0, int h0, int nout, int w1, int h1, int c1, float* out, size_t i)
{
    const int x = (int)(i % (size_t)w1), y = (int)(i / (size_t)w1);
    int xa, xb, ya, yb;
    footprint(x, y, w0, h0, w1, h1, xa, xb, ya, yb);
    float acc[MAX_CHANNELS];
    for (int c = 0; c < c1; c++)
        acc[c] = 0.0f;
    int n = 0;
    for (int sy = ya; sy < yb; sy++)
        for (int sx = xa; sx < xb; sx++)
        {
            const float* s = src + ((size_t)sy * w0 + sx) * nout;
            for (int c = 0; c < c1; c++)
                acc[c] += c < nout ? s[c] : 0.0f;
            n++;
        }
    const float inv = n > 0 ? 1.0f / (float)n : 0.0f;
    for (int c = 0; c < c1; c++)
        out[i * c1 + c] = c < nout ? 2.0f * acc[c] * inv - 1.0f : 0.0f;
}

// k_block_means: all nout source channels' means over the footprint, float accumulator again.
void block_means_texel(const float* src, int w0, int h0, int nout, int w1, int h1, float* means, size_t i)
{
    const int x = (int)(i % (size_t)w1), y = (int)(i / (size_t)w1);
    int xa, xb, ya, yb;
    footprint(x, y, w0, h0, w1, h1, xa, xb, ya, yb);
    float acc[MAX_NOUT];
    for (int c = 0; c < nout; c++)
        acc[c] = 0.0f;
    int n = 0;
    for (int sy = ya; sy < yb; sy++)
        for (int sx = xa; sx < xb; sx++)
        {
            const float* s = src + ((size_t)sy * w0 + sx) * nout;
            for (int c = 0; c < nout; c++)
                acc[c] += s[c];
            n++;
        }
    const float inv = n > 0 ? 1.0f / (float)n : 0.0f;
    for (int c = 0; c < nout; c++)
        means[i * (size_t)nout + c] = acc[c] * inv;
}

// k_quantise_level1: the nearest index of the channel's grid and the value it stands for, through model.h's ONE
// definition, so that every path onto the grid lands on the same bits (and verify_level1_on_grid agrees with them).
// `in` and `out` may be the same buffer - the level-0 snaps pass the plane as both - which is safe here exactly as it
// is on the device: element i is read and then written by one thread, and no other element is touched.
void quantise_run(CpuModel* d, const float* in, size_t n, int channels, int bits, const float* lo, const float* hi,
                  float* v, uint8_t* k)
{
    per_element(d, n, [&](size_t i) {
        const int c = (int)(i % (size_t)channels);
        const int q = ::level1_index(lo[c], hi[c], bits, in[i]);
        v[i] = ::level1_value(lo[c], hi[c], bits, q);
        k[i] = (uint8_t)q;
    });
}

// k_residual: out - target at the base plane's own texel centres, in the decoder's own fp32 arithmetic and WITHOUT the
// saturate and the rounding the recon PNGs go through. It samples through cpu_sample.h, the copy of the decoder's own
// rule, and not through the objective's double twin: this is deliberately the decoder's rounding.
void residual_texel(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0, int c1,
                    int nin, int nout, const float* weights, const float* bias, float* resid, size_t idx)
{
    const int px = (int)(idx % (size_t)w0), py = (int)(idx / (size_t)w0);
    float s[MAX_CHANNELS], c[MAX_CHANNELS], phi[MAX_NIN];
    for (int i = 0; i < c0; i++)
        s[i] = v0[idx * c0 + i];
    const BilinearTap t = bilinear_tap_pixel(px, py, 0.0f, 0.0f, w0, h0, w1, h1);
    sample_bilinear(v1, w1, c1, t, c);
    build_phi(s, c0, c, c1, phi);
    for (int r = 0; r < nout; r++)
    {
        float acc = bias[r];
        const float* row = weights + (size_t)r * nin;
        for (int i = 0; i < nin; i++)
            acc += row[i] * phi[i];
        resid[idx * (size_t)nout + r] = acc - src[idx * (size_t)nout + r];
    }
}

// init.cu's ProjDir and proj_value: the direction a projection is taken along and the residual's own mean.
struct ProjDir
{
    double e[MAX_NOUT];
    double mu[MAX_NOUT];
};

inline double proj_value(const float* resid, size_t t, int nout, const ProjDir& dir)
{
    double acc = 0.0;
    for (int r = 0; r < nout; r++)
        acc += dir.e[r] * ((double)resid[t * (size_t)nout + r] - dir.mu[r]);
    return acc;
}

// k_seed_channel: the scaled projection, clamped to [-1, 1], snapped onto the channel's palette in the palette mode
// (by lroundf, which rounds a half AWAY from zero: std::lround on a float is exactly that, where std::lrint would follow
// the rounding mode) and left continuous under --l0 bc8.
void seed_texel(const float* resid, int nout, const ProjDir& dir, double inv, int c0, int channel, int levels,
                const float* palette, int palette_mode, float* v0, uint8_t* k0, size_t t)
{
    const double p = proj_value(resid, t, nout, dir) * inv;
    const float value = (float)(p < -1.0 ? -1.0 : (p > 1.0 ? 1.0 : p));
    if (palette_mode)
    {
        int k = (int)std::lround((value + 1.0f) * 0.5f * (float)levels);
        k = k < 0 ? 0 : (k > levels ? levels : k);
        v0[t * (size_t)c0 + channel] = palette[(size_t)channel * 16 + k];
        k0[t * (size_t)c0 + channel] = (uint8_t)k;
    }
    else
    {
        v0[t * (size_t)c0 + channel] = value;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The reductions
// ---------------------------------------------------------------------------------------------------------------------

// k_cov_partial's sums over `texels` rows of `nout` floats: entry e < nout is the sum of channel e, and entry
// nout + r nout + s the sum of the product of channels r and s, each product formed in double from two floats exactly
// as the kernel forms it. Summed in the kernel's order: block b's lane x takes texels b 256 + x, then every
// SITE_BLOCKS 256 further, and the 256 lanes are halved into one. Block b's sums land in partial[b * entries + e]; the
// caller folds them in block order. Returns SITE_BLOCKS.
int cov_partials(CpuModel* d, const float* data, size_t texels, int nout, std::vector<double>& partial)
{
    const int entries = nout + nout * nout;
    const int threads = 256;  // init.cu's launch
    const size_t stride = (size_t)SITE_BLOCKS * threads;
    partial.assign((size_t)SITE_BLOCKS * entries, 0.0);
    d->pool->run(SITE_BLOCKS, [&](int b) {
        std::vector<double> lane((size_t)threads);
        for (int e = 0; e < entries; e++)
        {
            const int r = e < nout ? e : (e - nout) / nout;
            const int s = e < nout ? -1 : (e - nout) % nout;
            std::fill(lane.begin(), lane.end(), 0.0);
            for (size_t row = (size_t)b * threads; row < texels; row += stride)
                for (int x = 0; x < threads && row + x < texels; x++)
                {
                    const size_t t = row + x;
                    const double a = (double)data[t * (size_t)nout + r];
                    lane[(size_t)x] += s < 0 ? a : a * (double)data[t * (size_t)nout + s];
                }
            for (int half = threads / 2; half > 0; half >>= 1)
                for (int x = 0; x < half; x++)
                    lane[(size_t)x] += lane[(size_t)(x + half)];
            partial[(size_t)b * entries + e] = lane[0];
        }
    });
    return SITE_BLOCKS;
}

// k_proj_peak: the largest |projection| over a plane. A maximum is order-independent exactly, so each chunk's own
// maximum (the kernel's `v > acc ? v : acc`, from 0, which also passes over a NaN) folded by std::max is the device's
// answer for the same projections.
double proj_peak(CpuModel* d, const float* resid, size_t texels, int nout, const ProjDir& dir)
{
    return fold_chunks(
        *d->pool, chunks_for(texels, CPU_CHUNK), 0.0,
        [&](int c) {
            const size_t begin = (size_t)c * CPU_CHUNK;
            const size_t end = begin + CPU_CHUNK < texels ? begin + CPU_CHUNK : texels;
            double acc = 0.0;
            for (size_t t = begin; t < end; t++)
            {
                const double v = std::fabs(proj_value(resid, t, nout, dir));
                acc = v > acc ? v : acc;
            }
            return acc;
        },
        [](double a, double b) { return std::max(a, b); });
}

// k_proj_hist: |projection| / peak binned uniformly on [0, 1], ADDED INTO `hist`, which the caller keeps across every
// plane of the chain. Each chunk counts into its own private histogram; the private histograms are added into `hist`
// in chunk order after the join. The counts are integers, so this is exactly the device's atomicAdd total.
void proj_hist(CpuModel* d, const float* resid, size_t texels, int nout, const ProjDir& dir, double inv_peak, int bins,
               std::vector<unsigned long long>& hist)
{
    const int chunks = chunks_for(texels, CPU_HIST_CHUNK);
    std::vector<unsigned long long> part((size_t)chunks * bins, 0ull);
    d->pool->run(chunks, [&](int c) {
        const size_t begin = (size_t)c * CPU_HIST_CHUNK;
        const size_t end = begin + CPU_HIST_CHUNK < texels ? begin + CPU_HIST_CHUNK : texels;
        unsigned long long* h = part.data() + (size_t)c * bins;
        for (size_t t = begin; t < end; t++)
        {
            const double v = std::fabs(proj_value(resid, t, nout, dir)) * inv_peak;
            int b = (int)(v * (double)bins);
            b = b < 0 ? 0 : (b >= bins ? bins - 1 : b);
            h[b] += 1ull;
        }
    });
    for (int c = 0; c < chunks; c++)
        for (int b = 0; b < bins; b++)
            hist[(size_t)b] += part[(size_t)c * bins + b];
}

// ---------------------------------------------------------------------------------------------------------------------
// The host math, copied (plan section 1.8)
// ---------------------------------------------------------------------------------------------------------------------

// init.cu's fit_range: the per-channel range of a set of planes under one policy, minmax or the 0.1 % / 99.9 %
// percentiles by a sort of the channel's values.
void fit_range(const std::vector<std::vector<float>>& cont, int c1, const std::string& policy, std::vector<float>& lo,
               std::vector<float>& hi)
{
    lo.assign((size_t)c1, 1e30f);
    hi.assign((size_t)c1, -1e30f);
    if (policy == "minmax")
    {
        for (size_t i = 0; i < cont.size(); i++)
            for (size_t t = 0; t < cont[i].size(); t++)
            {
                const int c = (int)(t % (size_t)c1);
                lo[(size_t)c] = std::min(lo[(size_t)c], cont[i][t]);
                hi[(size_t)c] = std::max(hi[(size_t)c], cont[i][t]);
            }
        return;
    }
    std::vector<float> v;
    for (int c = 0; c < c1; c++)
    {
        v.clear();
        for (size_t i = 0; i < cont.size(); i++)
            for (size_t t = (size_t)c; t < cont[i].size(); t += (size_t)c1)
                v.push_back(cont[i][t]);
        if (v.empty())
            continue;
        std::sort(v.begin(), v.end());
        const size_t last = v.size() - 1;
        const size_t a = (size_t)(0.001 * (double)last + 0.5);
        const size_t b = (size_t)(0.999 * (double)last + 0.5);
        lo[(size_t)c] = v[a];
        hi[(size_t)c] = v[b];
    }
}

// init.cu's jacobi_eigen: the cyclic Jacobi eigendecomposition of a small symmetric matrix. vec's COLUMNS are the
// eigenvectors.
void jacobi_eigen(std::vector<double>& a, int n, std::vector<double>& vec, std::vector<double>& val)
{
    vec.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        vec[(size_t)i * n + i] = 1.0;
    for (int sweep = 0; sweep < 100; sweep++)
    {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                off += a[(size_t)p * n + q] * a[(size_t)p * n + q];
        if (off <= 1e-30)
            break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
            {
                const double apq = a[(size_t)p * n + q];
                if (std::fabs(apq) <= 1e-300)
                    continue;
                const double theta = (a[(size_t)q * n + q] - a[(size_t)p * n + p]) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < n; k++)
                {
                    const double akp = a[(size_t)k * n + p], akq = a[(size_t)k * n + q];
                    a[(size_t)k * n + p] = c * akp - s * akq;
                    a[(size_t)k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++)
                {
                    const double apk = a[(size_t)p * n + k], aqk = a[(size_t)q * n + k];
                    a[(size_t)p * n + k] = c * apk - s * aqk;
                    a[(size_t)q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++)
                {
                    const double vkp = vec[(size_t)k * n + p], vkq = vec[(size_t)k * n + q];
                    vec[(size_t)k * n + p] = c * vkp - s * vkq;
                    vec[(size_t)k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    val.assign((size_t)n, 0.0);
    for (int i = 0; i < n; i++)
        val[(size_t)i] = a[(size_t)i * n + i];
}

// ---------------------------------------------------------------------------------------------------------------------
// The drivers' shared steps
// ---------------------------------------------------------------------------------------------------------------------

// init.cu's fit_and_snap_level1: fit the one per-channel grid over the planes given, write it into the Model and the
// backend, and snap every plane onto it; v1 then holds the grid values and m.k1 the indices, which agree by
// construction. The CUDA side stages each plane through d->scratch; the staging here is the same buffer.
void fit_and_snap_level1(CpuModel* d, Model& m, const std::vector<std::vector<float>>& cont,
                         const std::vector<std::vector<float>>& fit_over, const std::string& policy)
{
    std::vector<float> lo, hi;
    fit_range(fit_over, m.c1, policy, lo, hi);
    for (int c = 0; c < m.c1; c++)
        if (!(hi[(size_t)c] > lo[(size_t)c]))
        {
            lo[(size_t)c] = -1.0f;
            hi[(size_t)c] = 1.0f;
        }
    m.lo1 = lo;
    m.hi1 = hi;
    upload_grids_to_device(d, m);

    m.k1.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w1 * p.h1 * m.c1;
        pool_copy(*d->pool, cont[i].data(), d->scratch.data(), n);
        quantise_run(d, d->scratch.data(), n, m.c1, m.bits1, d->lo1.data(), d->hi1.data(), p.v1.data(),
                     p.k1.data());
        m.k1[i].resize(n);
        pool_copy(*d->pool, p.k1.data(), m.k1[i].data(), n);
    }
}

// The planes as they stand in the model (init.cu's download_planes and download_level0_planes).
void download_planes(CpuModel* d, const Model& m, std::vector<std::vector<float>>& cont)
{
    cont.assign(d->planes.size(), std::vector<float>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w1 * p.h1 * m.c1;
        cont[i].resize(n);
        pool_copy(*d->pool, p.v1.data(), cont[i].data(), n);
    }
}

void download_level0_planes(CpuModel* d, const Model& m, std::vector<std::vector<float>>& cont)
{
    cont.assign(d->planes.size(), std::vector<float>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        cont[i].resize(n);
        pool_copy(*d->pool, p.v0.data(), cont[i].data(), n);
    }
}

// init.cu's fit_and_snap_level0: level 0's grid under --l0 bc8, at 8 bits, fitted over the base and every chain plane
// together; the continuous values go back into the plane's own v0 and are snapped IN PLACE (quantise_run's aliasing).
void fit_and_snap_level0(CpuModel* d, Model& m, const std::vector<std::vector<float>>& cont, const std::string& policy)
{
    std::vector<float> lo, hi;
    fit_range(cont, m.c0, policy, lo, hi);
    for (int c = 0; c < m.c0; c++)
        if (!(hi[(size_t)c] > lo[(size_t)c]))
        {
            lo[(size_t)c] = -1.0f;
            hi[(size_t)c] = 1.0f;
        }
    m.lo0 = lo;
    m.hi0 = hi;
    upload_grids_to_device(d, m);

    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        pool_copy(*d->pool, cont[i].data(), p.v0.data(), n);
        quantise_run(d, p.v0.data(), n, m.c0, 8, d->lo0.data(), d->hi0.data(), p.v0.data(), p.k0.data());
        m.k0[i].resize(n);
        pool_copy(*d->pool, p.k0.data(), m.k0[i].data(), n);
    }
}

// One plane's residual into d->resid (init.cu's residual_pass): the decoder and both latents as the last block left
// them.
void residual_pass(CpuModel* d, const Model& m, size_t plane)
{
    const CpuPlane& p = d->planes[plane];
    const size_t n = (size_t)p.w0 * p.h0;
    const float* v0 = p.v0.data();
    const float* v1 = p.v1.data();
    const float* src = p.src.data();
    const float* weights = d->weights.data();
    const float* bias = d->bias.data();
    float* resid = d->resid.data();
    const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1, c0 = m.c0, c1 = m.c1, nin = m.dec.nin, nout = m.nout;
    per_element(d, n, [&](size_t idx) {
        residual_texel(v0, v1, src, w0, h0, w1, h1, c0, c1, nin, nout, weights, bias, resid, idx);
    });
}

}   // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Level 0
// ---------------------------------------------------------------------------------------------------------------------

void init_level0(CpuModel* d, Model& m, const float* luma_weights, bool luma_channel0)
{
    upload_grids_to_device(d, m);
    ChannelBits bits = {};
    for (int c = 0; c < m.c0; c++)
        bits.b[c] = m.bits0[c];
    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    const float lr = luma_weights[0], lg = luma_weights[1], lb = luma_weights[2];
    const int luma0 = luma_channel0 ? 1 : 0;
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0;
        const float* src = p.src.data();
        float* v0 = p.v0.data();
        uint8_t* k0 = p.k0.data();
        const int nout = m.nout, c0 = m.c0;
        if (m.l0_bc8)
        {
            per_element(d, n, [&](size_t t) { init_level0_cont_texel(src, nout, c0, lr, lg, lb, luma0, v0, t); });
            std::fill(p.k0.begin(), p.k0.begin() + (std::ptrdiff_t)(n * m.c0), (uint8_t)0);
            m.k0[i].assign(n * m.c0, 0);
        }
        else
        {
            const float* palette = d->palette.data();
            per_element(d, n, [&](size_t t) {
                init_level0_texel(src, nout, c0, bits, palette, lr, lg, lb, luma0, v0, k0, t);
            });
            m.k0[i].resize(n * m.c0);
            pool_copy(*d->pool, p.k0.data(), m.k0[i].data(), n * m.c0);
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Level 1 and the grids
// ---------------------------------------------------------------------------------------------------------------------

void quantise_level1(CpuModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_planes(d, m, cont);
    fit_and_snap_level1(d, m, cont, cont, range_policy);
}

void freeze_level1_grid(CpuModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_planes(d, m, cont);
    fit_and_snap_level1(d, m, cont, cont, range_policy);
}

void quantise_level0(CpuModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_level0_planes(d, m, cont);
    fit_and_snap_level0(d, m, cont, range_policy);
}

void snap_level0_on_grid(CpuModel* d, Model& m)
{
    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        quantise_run(d, p.v0.data(), n, m.c0, 8, d->lo0.data(), d->hi0.data(), p.v0.data(), p.k0.data());
        m.k0[i].resize(n);
        pool_copy(*d->pool, p.k0.data(), m.k0[i].data(), n);
    }
}

void freeze_level0_grid(CpuModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_level0_planes(d, m, cont);
    fit_and_snap_level0(d, m, cont, range_policy);
}

void init_level1(CpuModel* d, Model& m, bool pca, Level1InitReport& rep)
{
    rep = Level1InitReport();
    rep.pca = pca;

    std::vector<std::vector<float>> cont(d->planes.size());

    if (!pca)
    {
        for (size_t i = 0; i < d->planes.size(); i++)
        {
            const CpuPlane& p = d->planes[i];
            const size_t n = (size_t)p.w1 * p.h1;
            const float* src = p.src.data();
            float* out = d->scratch.data();
            const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1, nout = m.nout, c1 = m.c1;
            per_element(d, n, [&](size_t t) { init_level1_box_texel(src, w0, h0, nout, w1, h1, c1, out, t); });
            cont[i].resize(n * m.c1);
            pool_copy(*d->pool, d->scratch.data(), cont[i].data(), n * m.c1);
        }
        fit_and_snap_level1(d, m, cont, cont, "minmax");
        return;
    }

    // The basis: the base plane's block means, their covariance, the eigenproblem.
    const int nout = m.nout;
    const int entries = nout + nout * nout;
    std::vector<double> raw((size_t)entries, 0.0);
    auto block_means = [&](const CpuPlane& p) {
        const size_t texels = (size_t)p.w1 * p.h1;
        const float* src = p.src.data();
        float* means = d->means.data();
        const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1;
        per_element(d, texels, [&](size_t t) { block_means_texel(src, w0, h0, nout, w1, h1, means, t); });
    };
    {
        const CpuPlane& p = d->planes[0];
        const size_t texels = (size_t)p.w1 * p.h1;
        block_means(p);
        std::vector<double> partial;
        const int chunks = cov_partials(d, d->means.data(), texels, nout, partial);
        for (int e = 0; e < entries; e++)
            for (int c = 0; c < chunks; c++)
                raw[(size_t)e] += partial[(size_t)c * entries + e];
    }

    const CpuPlane& base = d->planes[0];
    const double count = (double)((size_t)base.w1 * base.h1);
    std::vector<double> mu((size_t)nout, 0.0), cov((size_t)nout * nout, 0.0);
    for (int r = 0; r < nout; r++)
        mu[(size_t)r] = count > 0.0 ? raw[(size_t)r] / count : 0.0;
    for (int r = 0; r < nout; r++)
        for (int s = 0; s < nout; s++)
            cov[(size_t)r * nout + s] =
                (count > 0.0 ? raw[(size_t)(nout + r * nout + s)] / count : 0.0) - mu[(size_t)r] * mu[(size_t)s];

    for (int r = 0; r < nout; r++)
        rep.total_variance += cov[(size_t)r * nout + r];

    std::vector<double> work = cov, vec, val;
    jacobi_eigen(work, nout, vec, val);

    std::vector<int> order((size_t)nout);
    for (int i = 0; i < nout; i++)
        order[(size_t)i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return val[(size_t)a] > val[(size_t)b]; });

    rep.components = m.c1 < nout ? m.c1 : nout;
    for (int j = 0; j < rep.components && j < MAX_CHANNELS; j++)
        rep.eigenvalue[j] = val[(size_t)order[(size_t)j]];

    // Pass one: the raw projection of every plane's block means onto that one basis, with each channel's largest
    // magnitude kept as it goes. Host code in init.cu, and serial here as there.
    std::vector<double> peak((size_t)m.c1, 0.0);
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w1 * p.h1;
        block_means(p);
        const std::vector<float>& means = d->means;

        cont[i].assign(texels * (size_t)m.c1, 0.0f);
        for (size_t t = 0; t < texels; t++)
            for (int j = 0; j < rep.components; j++)
            {
                const int col = order[(size_t)j];
                double acc = 0.0;
                for (int r = 0; r < nout; r++)
                    acc += vec[(size_t)r * nout + col] * ((double)means[t * (size_t)nout + r] - mu[(size_t)r]);
                cont[i][t * (size_t)m.c1 + j] = (float)acc;
                peak[(size_t)j] = std::max(peak[(size_t)j], std::fabs(acc));
            }
    }

    // Pass two: each channel divided by its own largest magnitude; the clamp is a guard, a no-op by construction.
    for (int j = 0; j < m.c1; j++)
    {
        rep.peak[j] = peak[(size_t)j];
        if (!(peak[(size_t)j] > 0.0))
            continue;
        const double inv = 1.0 / peak[(size_t)j];
        for (size_t i = 0; i < cont.size(); i++)
            for (size_t t = j; t < cont[i].size(); t += (size_t)m.c1)
            {
                const double v = (double)cont[i][t] * inv;
                cont[i][t] = (float)(v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v));
            }
    }

    fit_and_snap_level1(d, m, cont, cont, "minmax");
}

// ---------------------------------------------------------------------------------------------------------------------
// Level 0 seeded from the residual (--init0 residual / texture)
// ---------------------------------------------------------------------------------------------------------------------

void init0_residual_channel(CpuModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep)
{
    const int nout = m.nout;
    rep = Init0ChannelReport();
    rep.nout = nout;
    rep.texture = -1;

    // (1) and (2): the residual and its covariance, over the base plane alone or, under --init0-scope chain, over every
    // plane with the weight the objective gives that plane's sites - applied to each chunk's partial as the CUDA host
    // applies it to each block's.
    const int entries = nout + nout * nout;
    std::vector<double> raw((size_t)entries, 0.0);
    double count = 0.0;
    {
        const size_t nplanes = m.init0_chain ? d->planes.size() : 1;
        std::vector<double> partial;
        for (size_t i = 0; i < nplanes; i++)
        {
            const CpuPlane& p = d->planes[i];
            const size_t texels = (size_t)p.w0 * p.h0;
            const double w = m.init0_chain ? plane_weight[i] : 1.0;
            if (!(w > 0.0))
                continue;
            residual_pass(d, m, i);
            const int chunks = cov_partials(d, d->resid.data(), texels, nout, partial);
            for (int e = 0; e < entries; e++)
                for (int c = 0; c < chunks; c++)
                    raw[(size_t)e] += w * partial[(size_t)c * entries + e];
            count += w * (double)texels;
        }
    }

    std::vector<double> mu((size_t)nout, 0.0), cov((size_t)nout * nout, 0.0);
    for (int r = 0; r < nout; r++)
        mu[(size_t)r] = count > 0.0 ? raw[(size_t)r] / count : 0.0;
    for (int r = 0; r < nout; r++)
        for (int s = 0; s < nout; s++)
            cov[(size_t)r * nout + s] =
                (count > 0.0 ? raw[(size_t)(nout + r * nout + s)] / count : 0.0) - mu[(size_t)r] * mu[(size_t)s];

    // The plain covariance, not the cw-weighted one: init.cu records the measurement that chose it.
    for (int r = 0; r < nout; r++)
        rep.total_variance += cov[(size_t)r * nout + r];

    // (3) the direction: the leading eigenvector of the whole covariance, or under --init0 texture of the one texture's
    // 3x3 block that carries the most residual variance, every other output left at zero.
    std::vector<double> u((size_t)nout, 0.0);
    if (m.init0_texture)
    {
        int best = 0;
        double best_value = -1.0;
        std::vector<double> best_vec(3, 0.0);
        for (int t = 0; t < m.textures; t++)
        {
            std::vector<double> block(9, 0.0);
            for (int r = 0; r < 3; r++)
                for (int s = 0; s < 3; s++)
                    block[(size_t)r * 3 + s] = cov[(size_t)(3 * t + r) * nout + (3 * t + s)];
            std::vector<double> work = block, vec, val;
            jacobi_eigen(work, 3, vec, val);
            int top = 0;
            for (int i = 1; i < 3; i++)
                if (val[(size_t)i] > val[(size_t)top])
                    top = i;
            if (val[(size_t)top] > best_value)
            {
                best_value = val[(size_t)top];
                best = t;
                for (int r = 0; r < 3; r++)
                    best_vec[(size_t)r] = vec[(size_t)r * 3 + top];
            }
        }
        rep.texture = best;
        rep.eigenvalue = best_value > 0.0 ? best_value : 0.0;
        for (int r = 0; r < 3; r++)
            u[(size_t)(3 * best + r)] = best_vec[(size_t)r];
    }
    else
    {
        std::vector<double> work = cov, vec, val;
        jacobi_eigen(work, nout, vec, val);
        int top = 0;
        for (int i = 1; i < nout; i++)
            if (val[(size_t)i] > val[(size_t)top])
                top = i;
        rep.eigenvalue = val[(size_t)top];
        for (int r = 0; r < nout; r++)
            u[(size_t)r] = vec[(size_t)r * nout + top];
    }
    rep.share = rep.total_variance > 0.0 ? rep.eigenvalue / rep.total_variance : 0.0;

    ProjDir dir = {};
    for (int r = 0; r < nout; r++)
    {
        dir.e[r] = u[(size_t)r];
        dir.mu[r] = mu[(size_t)r];
    }
    // The sign: the largest component positive, which makes the reported direction readable and changes nothing else.
    int biggest = 0;
    for (int r = 1; r < nout; r++)
        if (std::fabs(dir.e[r]) > std::fabs(dir.e[biggest]))
            biggest = r;
    if (dir.e[biggest] < 0.0)
        for (int r = 0; r < nout; r++)
            dir.e[r] = -dir.e[r];
    for (int r = 0; r < nout && r < MAX_NOUT; r++)
        rep.dir[r] = dir.e[r];

    // (4a) the divisor: the largest |projection| over the WHOLE chain.
    double peak = 0.0;
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        residual_pass(d, m, i);
        peak = std::max(peak, proj_peak(d, d->resid.data(), texels, nout, dir));
    }
    rep.peak = peak;
    rep.divisor = peak;

    // (4a') in the palette mode a PERCENTILE of |projection| instead, read off a histogram on [0, peak] that spans
    // every plane of the chain, at the bin's upper edge.
    double divisor = peak;
    if (!m.l0_bc8 && peak > 0.0)
    {
        std::vector<unsigned long long> hist((size_t)PROJ_HIST_BINS, 0ull);
        unsigned long long total = 0ull;
        for (size_t i = 0; i < d->planes.size(); i++)
        {
            const CpuPlane& p = d->planes[i];
            const size_t texels = (size_t)p.w0 * p.h0;
            residual_pass(d, m, i);
            proj_hist(d, d->resid.data(), texels, nout, dir, 1.0 / peak, PROJ_HIST_BINS, hist);
            total += (unsigned long long)texels;
        }
        const unsigned long long want = (unsigned long long)(INIT0_PERCENTILE * (double)total);
        unsigned long long cum = 0ull;
        int bin = PROJ_HIST_BINS - 1;
        for (int b = 0; b < PROJ_HIST_BINS; b++)
        {
            cum += hist[(size_t)b];
            if (cum >= want)
            {
                bin = b;
                break;
            }
        }
        divisor = peak * (double)(bin + 1) / (double)PROJ_HIST_BINS;
        if (!(divisor > 0.0))
            divisor = peak;
        rep.divisor = divisor;
    }

    // (4b) the channel itself, on every plane.
    const double inv = divisor > 0.0 ? 1.0 / divisor : 0.0;
    const int levels = m.l0_bc8 ? 255 : (1 << m.bits0[(size_t)channel]) - 1;
    const int palette_mode = m.l0_bc8 ? 0 : 1;
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        residual_pass(d, m, i);
        const float* resid = d->resid.data();
        const float* palette = d->palette.data();
        float* v0 = p.v0.data();
        uint8_t* k0 = p.k0.data();
        const int c0 = m.c0;
        per_element(d, texels, [&](size_t t) {
            seed_texel(resid, nout, dir, inv, c0, channel, levels, palette, palette_mode, v0, k0, t);
        });
        // The palette mode's host mirror of the stored indices, which the writer reads; the CUDA side copies the
        // plane's whole index buffer into the vector init_level0 sized.
        if (!m.l0_bc8)
        {
            const size_t n = texels * (size_t)m.c0;
            if (m.k0[i].size() < n)
                m.k0[i].resize(n);
            pool_copy(*d->pool, p.k0.data(), m.k0[i].data(), n);
        }
    }
}

}   // namespace nntc_cpu
