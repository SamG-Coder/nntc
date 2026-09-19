// cpu/cpu_objective.cpp: the objective, the encoder's own decode of a stored level, and the brute-force twin behind
// --check, on the CPU (docs/CPU_BACKEND_PLAN.md stage C2, the port of objective.cu).
//
// objective.cu's header is the WHY of every line below and is not repeated: E is the weighted squared error of the
// decoded material over every site of every stored plane, evaluated in double throughout, and the reported numbers are
// normalised so that E reads like a mean squared error. This file is a transcription of it. What differs, and why:
//
//   * THE SUMMATION ORDER. k_objective walks a fixed grid of SITE_BLOCKS blocks of OBJ_THREADS threads and reduces each
//     block through a shared-memory tree. The CPU does not replay that tree (the plan's section 0a retired the replay:
//     CUDA and the CPU are held to agree on PSNR and by the per-kernel harness's tolerances, not on bits). Its own
//     fixed order is chunks of CPU_CHUNK consecutive sites, each summed in increasing site order, the chunk sums folded
//     in chunk order after the join. The chunk count is a function of the plane's size alone, so E is the same bits at
//     every -j - and E is the round loop's stopping criterion, so that is what makes a CPU encode repeatable;
//   * THE STREAMS. The CUDA pass issues every plane on its own stream; here the planes are walked one after another
//     and each plane's sites are spread over the pool (section 3.1 rule 6: runs do not nest);
//   * THE CLOCK. Objective::ms is the wall clock around the pass, on the calling thread, where CUDA's is two events.
//
// Everything else - the site set, the three copies of the bilinear rule, the feature order, the weights, the three sums
// and their normalisation, k_decode's fp32 arithmetic with its saturate and its round-half-away-from-zero - is copied,
// operation for operation, so that on identical inputs the only difference the harness can see is the order the double
// partial sums were added in.
//
// ZERO DATA RACES (section 3.4). k_objective's chunk c reads the planes, the source, the decoder and the output weights,
// none of which any chunk writes, and writes nothing but its own partial triple, which fold_chunks keeps in its own
// slot until the join. k_decode's chunk c writes only the bytes of its own pixels. objective_check_host is serial.

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_sample.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nntc_cpu
{

namespace
{

// objective.cu's report counters: the objective is its own line in the report's time split and this is where the CPU
// backend counts it. Touched only by objective_eval, which the host calls from one thread.
double g_objective_ms = 0.0;
long long g_objective_calls = 0;

// ---------------------------------------------------------------------------------------------------------------------
// k_decode
// ---------------------------------------------------------------------------------------------------------------------

// Pixel idx of a stored level, decoded in the decoder's own fp32 arithmetic and saturated to a byte (objective.cu's
// k_decode, copied). std::lround on a float IS lroundf: it rounds a half AWAY from zero whatever the rounding mode is,
// which is what the device's lroundf does. (std::lrint and std::nearbyint follow the rounding mode, to-even by
// default, and would move every byte that sits on a half.)
void decode_pixel(const float* v0, const float* v1, int w0, int h0, int w1, int h1, int c0, int c1, int nin, int nout,
                  const float* weights, const float* bias, uint8_t* out, size_t idx)
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
        acc = acc < 0.0f ? 0.0f : (acc > 1.0f ? 1.0f : acc);
        out[idx * nout + r] = (uint8_t)std::lround(acc * 255.0f);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// k_objective
// ---------------------------------------------------------------------------------------------------------------------

// objective.cu's obj_clamp and obj_sample, copied: the hardware's bilinear rule in DOUBLE, the third copy of that rule
// in the tree beside cpu_sample.h's fp32 one and twin_sample below. They are kept apart on purpose (plan section 5.5):
// each is the copy of a distinct original, and merging them would make the harness compare one of them with itself.
inline int obj_clamp(int i, int n)
{
    return i < 0 ? 0 : (i >= n ? n - 1 : i);
}

inline void obj_sample(const float* plane, int w, int h, int c, double u, double v, double* out)
{
    const double x = u * (double)w - 0.5, y = v * (double)h - 0.5;
    const double xf = std::floor(x), yf = std::floor(y);
    const double fx = x - xf, fy = y - yf;
    const int x0 = obj_clamp((int)xf, w), x1 = obj_clamp((int)xf + 1, w);
    const int y0 = obj_clamp((int)yf, h), y1 = obj_clamp((int)yf + 1, h);
    const float* a = plane + ((size_t)y0 * w + x0) * c;
    const float* b = plane + ((size_t)y0 * w + x1) * c;
    const float* e = plane + ((size_t)y1 * w + x0) * c;
    const float* f = plane + ((size_t)y1 * w + x1) * c;
    const double w00 = (1.0 - fx) * (1.0 - fy), w10 = fx * (1.0 - fy), w01 = (1.0 - fx) * fy, w11 = fx * fy;
    for (int k = 0; k < c; k++)
        out[k] = w00 * (double)a[k] + w10 * (double)b[k] + w01 * (double)e[k] + w11 * (double)f[k];
}

// The three sums of one plane: the cw-weighted squared error over every site, and, unweighted, over the centre sites
// alone and over the fractional sites alone.
struct Sums
{
    double weighted = 0.0, centre = 0.0, fractional = 0.0;
};

// Sites [begin, end) of one plane, in increasing order: k_objective's loop body, copied, with its grid-stride loop
// replaced by a contiguous range.
Sums objective_sites(const CpuPlane& p, int c0, int c1, int nin, int nout, const float* weights, const float* bias,
                     const float* cw, int k_count, size_t begin, size_t end)
{
    const int per_pixel = 1 + k_count;
    const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1;
    const float* v0 = p.v0.data();
    const float* v1 = p.v1.data();
    const float* src = p.src.data();
    Sums s;
    for (size_t si = begin; si < end; si++)
    {
        const size_t pixel = si / (size_t)per_pixel;
        const int j = (int)(si % (size_t)per_pixel);
        const int px = (int)(pixel % (size_t)w0), py = (int)(pixel / (size_t)w0);

        float dx = 0.0f, dy = 0.0f;
        if (j > 0)
            subtexel_offset(k_count, j - 1, px, py, dx, dy);
        const double u = ((double)px + 0.5 + (double)dx) / (double)w0;
        const double v = ((double)py + 0.5 + (double)dy) / (double)h0;

        double sv[MAX_CHANNELS], c[MAX_CHANNELS], target[MAX_NOUT];
        obj_sample(v1, w1, h1, c1, u, v, c);
        if (j == 0)
        {
            for (int i = 0; i < c0; i++)
                sv[i] = (double)v0[pixel * c0 + i];
            for (int i = 0; i < nout; i++)
                target[i] = (double)src[pixel * nout + i];
        }
        else
        {
            obj_sample(v0, w0, h0, c0, u, v, sv);
            obj_sample(src, w0, h0, nout, u, v, target);
        }

        double phi[MAX_NIN];
        int n = 0;
        for (int q = 0; q < c1; q++)
            phi[n++] = c[q];
        for (int i = 0; i < c0; i++)
            phi[n++] = sv[i];
        for (int i = 0; i < c0; i++)
            for (int q = 0; q < c1; q++)
                phi[n++] = sv[i] * c[q];

        for (int r = 0; r < nout; r++)
        {
            double acc = (double)bias[r];
            const float* row = weights + (size_t)r * nin;
            for (int i = 0; i < nin; i++)
                acc += (double)row[i] * phi[i];
            const double e = acc - target[r];
            s.weighted += (double)cw[r] * e * e;
            if (j == 0)
                s.centre += e * e;
            else
                s.fractional += e * e;
        }
    }
    return s;
}

// objective.cu's normalise, copied: the three raw sums of every plane turned into the reported numbers.
void normalise(const Model& m, int k, const std::vector<double>& per_site, const std::vector<double>& acc, Objective& r)
{
    double sum_cw = 0.0;
    for (int c = 0; c < m.nout; c++)
        sum_cw += (double)m.cw[c];

    double num = 0.0, den = 0.0;
    r.e_plane.assign(m.planes.size(), 0.0);
    r.centre_plane.assign(m.planes.size(), 0.0);
    for (size_t p = 0; p < m.planes.size(); p++)
    {
        const double pixels = (double)m.planes[p].w0 * (double)m.planes[p].h0;
        const double sites = pixels * (double)(1 + k);
        r.e_plane[p] = acc[3 * p] / (sum_cw * sites);
        r.centre_plane[p] = acc[3 * p + 1] / (pixels * (double)m.nout);
        num += per_site[p] * acc[3 * p];
        den += per_site[p] * sites;
    }
    r.e = den > 0.0 ? num / (sum_cw * den) : 0.0;

    const double base = (double)m.planes[0].w0 * (double)m.planes[0].h0 * (double)m.nout;
    r.centre_mse = acc[1] / base;
    r.sampled_mse = k > 0 ? acc[2] / (base * (double)k) : 0.0;
}

// objective.cu's twin_sample, copied: the bilinear rule written out once more, for the brute-force twin alone.
void twin_sample(const std::vector<float>& plane, int w, int h, int c, double u, double v, double* out)
{
    const double x = u * (double)w - 0.5, y = v * (double)h - 0.5;
    const double xf = std::floor(x), yf = std::floor(y);
    const double fx = x - xf, fy = y - yf;
    int x0 = (int)xf, x1 = (int)xf + 1, y0 = (int)yf, y1 = (int)yf + 1;
    x0 = x0 < 0 ? 0 : (x0 >= w ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 >= w ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 >= h ? h - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 >= h ? h - 1 : y1);
    for (int k = 0; k < c; k++)
    {
        const double a = (double)plane[((size_t)y0 * w + x0) * c + k];
        const double b = (double)plane[((size_t)y0 * w + x1) * c + k];
        const double e = (double)plane[((size_t)y1 * w + x0) * c + k];
        const double f = (double)plane[((size_t)y1 * w + x1) * c + k];
        out[k] = (1.0 - fx) * (1.0 - fy) * a + fx * (1.0 - fy) * b + (1.0 - fx) * fy * e + fx * fy * f;
    }
}

}   // namespace

double objective_total_ms()
{
    return g_objective_ms;
}

long long objective_passes()
{
    return g_objective_calls;
}

void decode_plane(CpuModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8)
{
    const CpuPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w0 * p.h0;
    const float* v0 = p.v0.data();
    const float* v1 = p.v1.data();
    const float* weights = d->weights.data();
    const float* bias = d->bias.data();
    uint8_t* out = d->rgb8.data();
    const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1, c0 = m.c0, c1 = m.c1, nin = m.dec.nin, nout = m.nout;
    d->pool->run(chunks_for(n, CPU_CHUNK), [&](int c) {
        const size_t begin = (size_t)c * CPU_CHUNK;
        const size_t end = begin + CPU_CHUNK < n ? begin + CPU_CHUNK : n;
        for (size_t idx = begin; idx < end; idx++)
            decode_pixel(v0, v1, w0, h0, w1, h1, c0, c1, nin, nout, weights, bias, out, idx);
    });
    rgb8.resize(n * m.nout);
    pool_copy(*d->pool, d->rgb8.data(), rgb8.data(), n * m.nout);
}

void objective_eval(CpuModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    const auto start = std::chrono::steady_clock::now();
    const size_t np = d->planes.size();
    for (size_t pi = 0; pi < np; pi++)
    {
        const CpuPlane& p = d->planes[pi];
        const size_t sites = (size_t)p.w0 * p.h0 * (size_t)(1 + k);
        const Sums s = fold_chunks(
            *d->pool, chunks_for(sites, CPU_CHUNK), Sums(),
            [&](int c) {
                const size_t begin = (size_t)c * CPU_CHUNK;
                const size_t end = begin + CPU_CHUNK < sites ? begin + CPU_CHUNK : sites;
                return objective_sites(p, m.c0, m.c1, m.dec.nin, m.nout, d->weights.data(), d->bias.data(),
                                       d->cw.data(), k, begin, end);
            },
            [](Sums acc, const Sums& part) {
                acc.weighted += part.weighted;
                acc.centre += part.centre;
                acc.fractional += part.fractional;
                return acc;
            });
        // d->obj is k_obj_reduce's output, the plane's three sums; the CUDA side's per-block partials have no
        // counterpart, because the chunk partials never leave fold_chunks.
        d->obj[3 * pi] = s.weighted;
        d->obj[3 * pi + 1] = s.centre;
        d->obj[3 * pi + 2] = s.fractional;
    }
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    std::vector<double> acc(d->obj.begin(), d->obj.begin() + (std::ptrdiff_t)(3 * np));
    normalise(m, k, per_site, acc, r);
    r.ms = ms;
    g_objective_ms += ms;
    g_objective_calls++;
}

// objective.cu's objective_check_host, copied (plan section 1.8): the slowest, plainest thing that can compute E, and
// deliberately NOT the pass above - every pixel and every site of every plane in scan order, each plane sampled by hand
// through twin_sample and the decoder read from the host Model rather than the backend's copy of it. That independence
// is what makes it a witness, and a copy keeps it (section 1.4). The CUDA original downloads the three planes first;
// here they are the model's own vectors, so the three downloads are simply the plane's buffers.
void objective_check_host(CpuModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    const size_t np = d->planes.size();
    std::vector<double> acc(3 * np, 0.0);
    const int c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;

    for (size_t p = 0; p < np; p++)
    {
        const CpuPlane& dp = d->planes[p];
        const std::vector<float>& v0 = dp.v0;
        const std::vector<float>& v1 = dp.v1;
        const std::vector<float>& src = dp.src;

        for (int py = 0; py < dp.h0; py++)
            for (int px = 0; px < dp.w0; px++)
                for (int j = 0; j <= k; j++)
                {
                    float dx = 0.0f, dy = 0.0f;
                    if (j > 0)
                        subtexel_offset(k, j - 1, px, py, dx, dy);
                    const double u = ((double)px + 0.5 + (double)dx) / (double)dp.w0;
                    const double v = ((double)py + 0.5 + (double)dy) / (double)dp.h0;

                    double s[MAX_CHANNELS], c[MAX_CHANNELS], target[MAX_NOUT];
                    twin_sample(v1, dp.w1, dp.h1, c1, u, v, c);
                    if (j == 0)
                    {
                        const size_t idx = (size_t)py * dp.w0 + px;
                        for (int i = 0; i < c0; i++)
                            s[i] = (double)v0[idx * c0 + i];
                        for (int i = 0; i < nout; i++)
                            target[i] = (double)src[idx * nout + i];
                    }
                    else
                    {
                        twin_sample(v0, dp.w0, dp.h0, c0, u, v, s);
                        twin_sample(src, dp.w0, dp.h0, nout, u, v, target);
                    }

                    double phi[MAX_NIN];
                    int n = 0;
                    for (int q = 0; q < c1; q++)
                        phi[n++] = c[q];
                    for (int i = 0; i < c0; i++)
                        phi[n++] = s[i];
                    for (int i = 0; i < c0; i++)
                        for (int q = 0; q < c1; q++)
                            phi[n++] = s[i] * c[q];

                    for (int row = 0; row < nout; row++)
                    {
                        double y = (double)m.dec.b[row];
                        for (int i = 0; i < nin; i++)
                            y += (double)m.dec.w[(size_t)row * nin + i] * phi[i];
                        const double e = y - target[row];
                        acc[3 * p] += (double)m.cw[row] * e * e;
                        acc[3 * p + (j == 0 ? 1 : 2)] += e * e;
                    }
                }
    }
    normalise(m, k, per_site, acc, r);
    r.ms = 0.0;
}

}   // namespace nntc_cpu
