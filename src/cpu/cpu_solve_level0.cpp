// cpu/cpu_solve_level0.cpp: block (c), level 0 as one exact palette search per texel, on the CPU
// (docs/CPU_BACKEND_PLAN.md stage C6, the port of solve_level0.cu).
//
// solve_level0.cu's header is the WHY of every line below and is not repeated: with level 1 and the decoder held, the
// objective over the sites one level-0 texel can reach is an exact quadratic in that texel's own C0 values, assembled in
// double from the affine structure of the decoder; its argmin over the palette is a joint enumeration when the bits sum
// to at most 8 and --sweeps coordinate sweeps otherwise; ties keep the current state; and the texels are moved in four
// colour passes, (tx & 1, ty & 1), each of which is an exact joint minimisation over its texels because no site can
// touch two texels of one colour. This file is a transcription. What differs, and why:
//
//   * THE RACE FIX (plan section 3.4). k_level0_search (solve_level0.cu:126-141) accumulates a fractional site's `beta`
//     from EVERY tap that is not this texel, and only afterwards discards the site when `alpha` - this texel's own weight
//     in it - is not positive. A site of pixel (tx + 1, ty) whose bilinear pair spans tx + 1 and tx + 2 has no tap on
//     (tx, ty), so its alpha is zero, but its beta has already read v0 at (tx + 2, ty): a texel of THIS colour, which
//     another texel's worker of the same pass may be writing. On the device that is a stale read of a number that is
//     then thrown away; under the C++ memory model it is a data race. So here the four taps are walked twice: once for
//     alpha, then the site is discarded if alpha is not positive, and only then once more for beta. Nothing computed
//     changes - alpha's additions happen in the same q order and are the same, beta's happen in the same q order and are
//     the same for every site that survives, and a site that does not survive never read its beta - and after the hoist
//     every surviving site has (tx, ty) among its taps, so each of its other taps is within one texel of (tx, ty) in
//     each axis (the clamp moves an index by at most the step that took it out of range), which is a texel of another
//     colour: the race is gone, not narrowed;
//   * THE SHADOWED LOOP VARIABLES. The kernel's pixel loop is `for (int dy ...) for (int dx ...)` and its site loop then
//     declares `float dx, dy` for the site's subtexel offset, shadowing the pair it is inside (solve_level0.cu:98-108).
//     Nothing after the inner declaration reads the outer pair, so the kernel is correct; a transcription that kept the
//     names would be one misplaced line from reading the wrong one. Here they are (ox, oy) for the neighbour pixel and
//     (sdx, sdy) for the subtexel offset;
//   * THE COUNTERS. CUDA counts the moved texels of each colour pass with an atomicAdd on an unsigned; here each chunk
//     counts its own and the chunk counts are folded in chunk order after the join. An integer sum, so the same number;
//   * THE CLOCK. Each plane's Level0Report::ms is the wall clock around its four passes, where CUDA's is the plane's
//     pair of events around the same four launches.
//
// Everything else is copied operation for operation: search_energy, the site set (the centre site read nearest with
// alpha 1 and beta 0, then the K fractional sites of each of the nine pixels around the texel), the tap-merge rule (a
// tap the clamp pulled onto this texel ADDS its weight to alpha, so a texel appearing twice carries the sum), p and Q's
// columns in float, the A0 / A1 / A2 accumulation in double, the argmin starting from the current state with strict
// `<`, the enumeration's mixed-radix state order and its skip of the current state, the sweeps' channel order and their
// early break on a sweep that moved nothing, and the write of v0 and k0 only for a texel that moved.
//
// ZERO DATA RACES (section 3.4). A colour pass is one pool run; a chunk covers a contiguous range of that colour's
// texels and writes v0 and k0 of those texels only. It reads v1, the source, the decoder, cw and the palette, which no
// chunk of the run writes; its own texel's k0, which only it writes; and, after the hoist above, v0 only at taps within
// one texel of its own texel in each axis and not equal to it - texels of another colour, written by an earlier run or
// not at all. The four runs are the barrier the colours need between them: pass q + 1 reads what pass q wrote.

#include "cpu/cpu_solve_level0.h"

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_sample.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace nntc_cpu
{

namespace
{

double since_ms(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// solve_level0.cu's search_energy, copied: E(x) = A0 + 2 x.A1 + x.A2.x, with A2 held in full.
inline double search_energy(double a0, const double* a1, const double* a2, const float* x, int c0)
{
    double e = a0;
    for (int i = 0; i < c0; i++)
    {
        e += 2.0 * (double)x[i] * a1[i];
        for (int j = 0; j < c0; j++)
            e += (double)x[i] * (double)x[j] * a2[i * c0 + j];
    }
    return e;
}

// k_level0_search for texel (tx, ty): the exact quadratic over the sites it can reach, then the argmin over the
// palette. v0 is read at the texel's neighbours and written at the texel itself, in place, as the kernel does it.
// Returns 1 when the texel moved.
unsigned int level0_search_texel(float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0,
                                 int c1, int nin, int nout, const float* weights, const float* bias, const float* cw,
                                 const float* palette, const ChannelBits& bits, int k_count, bool joint, int sweeps,
                                 uint8_t* k0, int tx, int ty)
{
    const size_t t = (size_t)ty * w0 + tx;

    double a0 = 0.0, a1[MAX_CHANNELS], a2[MAX_CHANNELS * MAX_CHANNELS];
    for (int i = 0; i < c0; i++)
        a1[i] = 0.0;
    for (int i = 0; i < c0 * c0; i++)
        a2[i] = 0.0;

    // (ox, oy) is the neighbour pixel's offset, the kernel's shadowed outer (dx, dy); see the file header.
    for (int oy = -1; oy <= 1; oy++)
        for (int ox = -1; ox <= 1; ox++)
        {
            const int px = tx + ox, py = ty + oy;
            if (px < 0 || py < 0 || px >= w0 || py >= h0)
                continue;
            const int first = (ox == 0 && oy == 0) ? 0 : 1;   // only this texel's own pixel carries a centre site
            for (int j = first; j <= k_count; j++)
            {
                // (sdx, sdy) is the site's subtexel offset, the kernel's shadowing inner (dx, dy).
                float sdx, sdy;
                site_delta(k_count, j, px, py, sdx, sdy);

                float alpha = 0.0f, beta[MAX_CHANNELS], cbar[MAX_CHANNELS], target[MAX_NOUT];
                const BilinearTap t1 = bilinear_tap_pixel(px, py, sdx, sdy, w0, h0, w1, h1);
                sample_bilinear(v1, w1, c1, t1, cbar);
                for (int i = 0; i < c0; i++)
                    beta[i] = 0.0f;

                if (j == 0)
                {
                    // The centre site reads level 0 nearest, so the texel IS the sample.
                    alpha = 1.0f;
                    for (int r = 0; r < nout; r++)
                        target[r] = src[t * nout + r];
                }
                else
                {
                    const BilinearTap t0 = bilinear_tap_pixel(px, py, sdx, sdy, w0, h0, w0, h0);
                    sample_bilinear(src, w0, nout, t0, target);
                    const int cx[4] = { t0.x0, t0.x1, t0.x0, t0.x1 };
                    const int cy[4] = { t0.y0, t0.y0, t0.y1, t0.y1 };
                    const float cwt[4] = { (1.0f - t0.fx) * (1.0f - t0.fy), t0.fx * (1.0f - t0.fy),
                                           (1.0f - t0.fx) * t0.fy, t0.fx * t0.fy };
                    // The race fix: alpha over the four taps first, the site discarded if this texel carries no
                    // weight in it, and only then beta from the other taps - each loop in the kernel's q order.
                    for (int q = 0; q < 4; q++)
                        if (cx[q] == tx && cy[q] == ty)
                            alpha += cwt[q];
                    if (!(alpha > 0.0f))
                        continue;   // this site cannot see the texel, so it is the same constant in every state
                    for (int q = 0; q < 4; q++)
                        if (!(cx[q] == tx && cy[q] == ty))
                        {
                            const float* nv = v0 + ((size_t)cy[q] * w0 + cx[q]) * c0;
                            for (int i = 0; i < c0; i++)
                                beta[i] += cwt[q] * nv[i];
                        }
                }

                // p = the decode with this texel's channels at zero, and Q's columns at this site's level-1 sample.
                float qcol[MAX_NOUT * MAX_CHANNELS], resid[MAX_NOUT];
                for (int r = 0; r < nout; r++)
                {
                    const float* row = weights + (size_t)r * nin;
                    float out = bias[r];
                    for (int q = 0; q < c1; q++)
                        out += row[q] * cbar[q];
                    for (int i = 0; i < c0; i++)
                    {
                        float slope = row[c1 + i];
                        for (int q = 0; q < c1; q++)
                            slope += row[c1 + c0 + i * c1 + q] * cbar[q];
                        qcol[r * c0 + i] = slope;
                        out += slope * beta[i];
                    }
                    resid[r] = out - target[r];
                }

                for (int i = 0; i < nout; i++)
                    a0 += (double)cw[i] * (double)resid[i] * (double)resid[i];
                for (int i = 0; i < c0; i++)
                {
                    double g = 0.0;
                    for (int r = 0; r < nout; r++)
                        g += (double)cw[r] * (double)qcol[r * c0 + i] * (double)resid[r];
                    a1[i] += (double)alpha * g;
                    for (int j2 = 0; j2 < c0; j2++)
                    {
                        double s = 0.0;
                        for (int r = 0; r < nout; r++)
                            s += (double)cw[r] * (double)qcol[r * c0 + i] * (double)qcol[r * c0 + j2];
                        a2[i * c0 + j2] += (double)alpha * (double)alpha * s;
                    }
                }
            }
        }

    // The argmin over the palette, starting from the current state so that a tie keeps it.
    int cur[MAX_CHANNELS], best[MAX_CHANNELS], levels[MAX_CHANNELS];
    float x[MAX_CHANNELS];
    for (int i = 0; i < c0; i++)
    {
        levels[i] = (1 << bits.b[i]) - 1;
        cur[i] = (int)k0[t * c0 + i];
        best[i] = cur[i];
        x[i] = palette[(size_t)i * 16 + cur[i]];
    }
    double best_e = search_energy(a0, a1, a2, x, c0);

    if (joint)
    {
        long long total = 1;
        for (int i = 0; i < c0; i++)
            total *= (long long)(levels[i] + 1);
        for (long long n = 0; n < total; n++)
        {
            long long rest = n;
            bool same = true;
            int state[MAX_CHANNELS];
            for (int i = 0; i < c0; i++)
            {
                state[i] = (int)(rest % (long long)(levels[i] + 1));
                rest /= (long long)(levels[i] + 1);
                if (state[i] != cur[i])
                    same = false;
                x[i] = palette[(size_t)i * 16 + state[i]];
            }
            if (same)
                continue;
            const double e = search_energy(a0, a1, a2, x, c0);
            if (e < best_e)
            {
                best_e = e;
                for (int i = 0; i < c0; i++)
                    best[i] = state[i];
            }
        }
    }
    else
    {
        for (int sw = 0; sw < sweeps; sw++)
        {
            bool changed = false;
            for (int i = 0; i < c0; i++)
            {
                for (int q = 0; q < c0; q++)
                    x[q] = palette[(size_t)q * 16 + best[q]];
                const int hold = best[i];
                for (int kk = 0; kk <= levels[i]; kk++)
                {
                    if (kk == best[i])
                        continue;
                    x[i] = palette[(size_t)i * 16 + kk];
                    const double e = search_energy(a0, a1, a2, x, c0);
                    if (e < best_e)
                    {
                        best_e = e;
                        best[i] = kk;
                    }
                }
                if (best[i] != hold)
                    changed = true;
            }
            if (!changed)
                break;   // a sweep that moves nothing cannot be followed by one that does
        }
    }

    bool moved_here = false;
    for (int i = 0; i < c0; i++)
        moved_here = moved_here || best[i] != cur[i];
    if (!moved_here)
        return 0;
    for (int i = 0; i < c0; i++)
    {
        v0[t * c0 + i] = palette[(size_t)i * 16 + best[i]];
        k0[t * c0 + i] = (uint8_t)best[i];
    }
    return 1;
}

ChannelBits level0_bits(const Model& m)
{
    ChannelBits bits = {};
    for (int c = 0; c < m.c0; c++)
        bits.b[c] = m.bits0[(size_t)c];
    return bits;
}

}   // namespace

bool level0_search_joint(const Model& m)
{
    int total_bits = 0;
    for (int c = 0; c < m.c0; c++)
        total_bits += m.bits0[(size_t)c];
    return total_bits <= 8;
}

unsigned int level0_search_pass(CpuModel* d, const Model& m, int k, int sweeps, bool joint, int plane, int colour)
{
    CpuPlane& p = d->planes[(size_t)plane];
    const int cx = colour & 1, cy = colour >> 1;
    const int cw0 = (p.w0 - cx + 1) / 2, ch0 = (p.h0 - cy + 1) / 2;
    const size_t count = (size_t)cw0 * (size_t)ch0;
    if (count == 0)
        return 0;
    const ChannelBits bits = level0_bits(m);
    float* v0 = p.v0.data();
    uint8_t* k0 = p.k0.data();
    const float* v1 = p.v1.data();
    const float* src = p.src.data();
    const float* weights = d->weights.data();
    const float* bias = d->bias.data();
    const float* cw = d->cw.data();
    const float* palette = d->palette.data();
    return fold_chunks(
        *d->pool, chunks_for(count, CPU_CHUNK), 0u,
        [&](int ch) {
            unsigned int part = 0;
            const size_t begin = (size_t)ch * CPU_CHUNK;
            const size_t end = begin + CPU_CHUNK < count ? begin + CPU_CHUNK : count;
            for (size_t ci = begin; ci < end; ci++)
            {
                const int tx = cx + 2 * (int)(ci % (size_t)cw0);
                const int ty = cy + 2 * (int)(ci / (size_t)cw0);
                part += level0_search_texel(v0, v1, src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout,
                                            weights, bias, cw, palette, bits, k, joint, sweeps, k0, tx, ty);
            }
            return part;
        },
        [](unsigned int acc, unsigned int part) { return acc + part; });
}

// Block (c) on every plane: four colour passes each, the planes one after another (the pool's rule 6), with the
// branch and the state count decided once for the call.
double solve_level0_all(CpuModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep)
{
    const auto start = std::chrono::steady_clock::now();
    const size_t np = d->planes.size();
    rep.assign(np, Level0Report());
    long long states = 1;
    for (int c = 0; c < m.c0; c++)
        states *= (long long)1 << m.bits0[(size_t)c];
    const bool joint = level0_search_joint(m);

    for (size_t i = 0; i < np; i++)
    {
        CpuPlane& p = d->planes[i];
        const auto t0 = std::chrono::steady_clock::now();
        for (int colour = 0; colour < 4; colour++)
        {
            p.counters[(size_t)(1 + colour)] = level0_search_pass(d, m, k, sweeps, joint, (int)i, colour);
            rep[i].moved[colour] = (long long)p.counters[(size_t)(1 + colour)];
            rep[i].moved_total += rep[i].moved[colour];
        }
        rep[i].ms = since_ms(t0);
        rep[i].joint = joint;
        rep[i].states = states;
    }
    return since_ms(start);
}

}   // namespace nntc_cpu
