// cpu/cpu_solve_decoder.cpp: block (a), the decoder as one global least squares, on the CPU (docs/CPU_BACKEND_PLAN.md
// stage C4, the port of solve_decoder.cu).
//
// solve_decoder.cu's header is the WHY of every line below and is not repeated: the normal equations of the decoder
// over every site of every stored plane, accumulated in double from fp32 features, each plane's sum scaled by its own
// site weight, and the small dense system solved on the host by Gauss-Jordan behind the ridge ladder that keeps E from
// rising across the block. This file is a transcription. What differs, and why:
//
//   * THE ACCUMULATION KEEPS k_ls_accumulate's SHAPE, because here the shape is the cheapest honest form and not a
//     replay. The CUDA kernel already works in panels: a block stages LS_TILE sites' features and targets, then sums
//     each normal-equation entry over that tile as one serial double dot product, and adds the tile sum into its own
//     accumulator. The CPU form is the same two phases with the barriers deleted - chunk b of SITE_BLOCKS walks tiles
//     b, b + SITE_BLOCKS, ... exactly as block b does and writes its accumulator into the same slot of ls_partial - so
//     the tile boundary is the summation order, as the plan's table 5.2 says it must be, and the chunk count is a
//     constant, so the sums are the same bits at every -j. A site past the plane's end still contributes a row of
//     zeros rather than being skipped: that is the kernel's rule and it costs one partial tile per plane. LS_STRIDE's
//     one float of padding is a shared-memory bank fix and is not carried: it moves addresses, not additions;
//   * k_ls_reduce is the serial second stage it already was, entry by entry, the planes in index order and inside each
//     plane the chunks in index order;
//   * THE CLOCK. The block's time is the wall clock around the accumulation, where CUDA's is the base plane's pair of
//     events around the same work.
//
// Everything on the host side - the trace and the mean diagonal, the flat x of the incoming decoder, the normaliser and
// its refusal to fail open, qform and its rounding bound, the four rungs (three ridges and the refusal), the float round
// trip before judging, the non-finite refusal, the acceptance bar, the tally, the constant-fit fallback - is copied as
// it stands (section 1.8), so that given the same normal equations the CPU backend takes the same rung and ships the
// same decoder. The per-kernel harness holds it to exactly that.
//
// ZERO DATA RACES (section 3.4). Chunk b of a plane's accumulation reads the plane's two latents and its source, which
// nothing in the run writes, stages into a tile and an accumulator of its own, and writes only column b of the plane's
// block of ls_partial - one slot per entry, which no other chunk writes. The reduction and the solve are serial, after
// the join.

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_sample.h"

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nntc_cpu
{

namespace
{

// k_ls_accumulate's block width, the sites one tile stages. It is the summation order of every entry (one serial dot
// product per tile, then the tile sums added in tile order), not a tuning knob.
const int LS_TILE = 256;

// solve_decoder.cu's tri_index, copied: entry (i, j), i <= j, of the upper triangle of the mm x mm matrix A.
inline int tri_index(int i, int j, int mm)
{
    return i * mm - i * (i - 1) / 2 + (j - i);
}

// solve_decoder.cu's gauss_jordan, copied: partial pivoting on [A | rhs^T], an exact zero the only pivot refused.
bool gauss_jordan(std::vector<double>& g, int m, int nrhs)
{
    const int cols = m + nrhs;
    for (int k = 0; k < m; k++)
    {
        int pivot = k;
        double best = std::fabs(g[(size_t)k * cols + k]);
        for (int i = k + 1; i < m; i++)
        {
            const double a = std::fabs(g[(size_t)i * cols + k]);
            if (a > best)
            {
                best = a;
                pivot = i;
            }
        }
        if (best == 0.0)
            return false;
        if (pivot != k)
            for (int j = 0; j < cols; j++)
                std::swap(g[(size_t)k * cols + j], g[(size_t)pivot * cols + j]);
        const double inv = 1.0 / g[(size_t)k * cols + k];
        for (int j = 0; j < cols; j++)
            g[(size_t)k * cols + j] *= inv;
        for (int i = 0; i < m; i++)
        {
            if (i == k)
                continue;
            const double f = g[(size_t)i * cols + k];
            if (f == 0.0)
                continue;
            for (int j = 0; j < cols; j++)
                g[(size_t)i * cols + j] -= f * g[(size_t)k * cols + j];
        }
    }
    return true;
}

// The ridge ladder's constants and the tally, copied: the shipped ridge first, then smaller ones, then none; a
// candidate refused only when it raises E by more than Q's own rounding can resolve.
const double RIDGE_LADDER[] = { 1e-9, 1e-12, 0.0 };
const double LS_ACCEPT_RISE = 1e-12;
const size_t RIDGE_REFUSED = sizeof(RIDGE_LADDER) / sizeof(RIDGE_LADDER[0]);

// Which rung each call of this run ended on: the CPU backend's own count, touched only by solve_decoder, which the
// host calls from one thread.
long long g_ridge_tally[RIDGE_REFUSED + 1] = { 0 };

// One plane's share of the normal equations: chunk b of SITE_BLOCKS walks the tiles b, b + SITE_BLOCKS, ... of LS_TILE
// sites, stages each tile's features, its constant one and its targets entry-major, sums every entry over the tile in
// site order and adds the tile sum into its own accumulator; the accumulator lands in column b of `partial`
// (entries x SITE_BLOCKS, entry-major), which is k_ls_accumulate's layout.
void ls_accumulate(CpuModel* d, const CpuPlane& p, const Model& m, int k_count, double* partial)
{
    const int nin = m.dec.nin, nout = m.nout, mm = nin + 1;
    const int tri = mm * (mm + 1) / 2;
    const int entries = tri + nout * mm;
    const int per_pixel = 1 + k_count;
    const size_t sites = (size_t)p.w0 * p.h0 * per_pixel;
    const size_t tiles = (sites + LS_TILE - 1) / LS_TILE;

    // The two staged rows each entry is the dot product of: the triangle's (i, j), walked row by row as the kernel
    // walks it, then the right-hand side's (target c, feature i). Rows 0..mm-1 of a tile are the features and the one,
    // rows mm..mm+nout-1 the targets.
    std::vector<int> row_a((size_t)entries), row_b((size_t)entries);
    for (int e = 0; e < entries; e++)
    {
        if (e < tri)
        {
            int i = 0, rest = e;
            while (rest >= mm - i)
                rest -= mm - i++;
            row_a[(size_t)e] = i;
            row_b[(size_t)e] = i + rest;
        }
        else
        {
            const int e2 = e - tri;
            row_a[(size_t)e] = mm + e2 / mm;
            row_b[(size_t)e] = e2 % mm;
        }
    }

    const float* v0 = p.v0.data();
    const float* v1 = p.v1.data();
    const float* src = p.src.data();
    d->pool->run(SITE_BLOCKS, [&](int b) {
        std::vector<float> tile((size_t)(mm + nout) * LS_TILE);
        std::vector<double> acc((size_t)entries, 0.0);
        for (size_t t = (size_t)b; t < tiles; t += SITE_BLOCKS)
        {
            for (int s = 0; s < LS_TILE; s++)
            {
                const size_t si = t * LS_TILE + (size_t)s;
                float phi[MAX_NIN];
                for (int i = 0; i < nin; i++)
                    phi[i] = 0.0f;
                float target[MAX_NOUT];
                for (int c = 0; c < nout; c++)
                    target[c] = 0.0f;
                float one = 0.0f;
                if (si < sites)
                {
                    const size_t pixel = si / (size_t)per_pixel;
                    const int j = (int)(si % (size_t)per_pixel);
                    const int px = (int)(pixel % (size_t)p.w0), py = (int)(pixel / (size_t)p.w0);
                    site_row(v0, v1, src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, nout, k_count, j, px, py, phi, target);
                    one = 1.0f;
                }
                for (int i = 0; i < nin; i++)
                    tile[(size_t)i * LS_TILE + (size_t)s] = phi[i];
                tile[(size_t)nin * LS_TILE + (size_t)s] = one;   // a site outside the plane contributes a row of zeros
                for (int c = 0; c < nout; c++)
                    tile[(size_t)(mm + c) * LS_TILE + (size_t)s] = target[c];
            }
            for (int e = 0; e < entries; e++)
            {
                const float* a = tile.data() + (size_t)row_a[(size_t)e] * LS_TILE;
                const float* bb = tile.data() + (size_t)row_b[(size_t)e] * LS_TILE;
                double sum = 0.0;
                for (int s = 0; s < LS_TILE; s++)
                    sum += (double)a[s] * (double)bb[s];
                acc[(size_t)e] += sum;
            }
        }
        for (int e = 0; e < entries; e++)
            partial[(size_t)e * SITE_BLOCKS + (size_t)b] = acc[(size_t)e];
    });
}

}   // namespace

void decoder_ridge_tally(long long out[4])
{
    for (size_t i = 0; i <= RIDGE_REFUSED; i++)
        out[i] = g_ridge_tally[i];
}

double solve_decoder(CpuModel* d, Model& m, int k, const std::vector<double>& per_site)
{
    const int nin = m.dec.nin, nout = m.nout, mm = nin + 1;
    const int tri = mm * (mm + 1) / 2;
    const int entries = tri + nout * mm;
    const size_t planes = d->planes.size();

    // The accumulation, plane by plane (the pool's rule 6: runs do not nest), each plane into its own block of the
    // partials, as the CUDA planes each write their own.
    for (size_t i = 0; i < planes; i++)
        d->ls_omega[i] = per_site[i];
    const auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < planes; i++)
        ls_accumulate(d, d->planes[i], m, k, d->ls_partial.data() + i * (size_t)entries * SITE_BLOCKS);
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    // k_ls_reduce: entry by entry, the planes in index order and inside each plane the chunks in index order, each
    // plane's own sum scaled by its own site weight.
    for (int e = 0; e < entries; e++)
    {
        double total = 0.0;
        for (size_t p = 0; p < planes; p++)
        {
            const double* row = d->ls_partial.data() + (p * (size_t)entries + (size_t)e) * SITE_BLOCKS;
            double sum = 0.0;
            for (int b = 0; b < SITE_BLOCKS; b++)
                sum += row[b];
            total += d->ls_omega[p] * sum;
        }
        d->reduction[(size_t)e] = total;
    }
    const std::vector<double> acc(d->reduction.begin(), d->reduction.begin() + entries);

    // ---- from here to the end, solve_decoder.cu's host side, copied as it stands.
    double trace = 0.0;
    for (int i = 0; i < mm; i++)
        trace += acc[(size_t)tri_index(i, i, mm)];
    const double mean_diag = trace > 0.0 ? trace / mm : 1.0;

    // The decoder the caller arrives with, as the flat x = [W_c ; b_c] the quadratic below is written in.
    const bool had_previous = !m.dec.w.empty() && !m.dec.b.empty();
    std::vector<double> x_prev((size_t)nout * mm, 0.0);
    if (had_previous)
        for (int c = 0; c < nout; c++)
        {
            for (int i = 0; i < nin; i++)
                x_prev[(size_t)c * mm + i] = (double)m.dec.w[(size_t)c * nin + i];
            x_prev[(size_t)c * mm + nin] = (double)m.dec.b[c];
        }

    // Q's normaliser, in the round loop's own units; a zero one refuses rather than making every candidate look free.
    double sum_cw = 0.0;
    for (int c = 0; c < nout; c++)
        sum_cw += (double)m.cw[c];
    double den = 0.0;
    for (size_t i = 0; i < planes; i++)
        den += per_site[i] * (double)d->planes[i].w0 * (double)d->planes[i].h0 * (double)(1 + k);
    const bool norm_ok = sum_cw * den > 0.0;
    const double inv_norm = norm_ok ? 1.0 / (sum_cw * den) : 0.0;

    // Q(x) = sum_c cw[c] ( x_c^T A x_c - 2 x_c . rhs_c ), and beside it its own rounding bound, in the loop's units.
    auto qform = [&](const std::vector<double>& x, double& err) {
        double q = 0.0, e = 0.0;
        for (int c = 0; c < nout; c++)
        {
            double qc = 0.0, ec = 0.0;
            for (int i = 0; i < mm; i++)
            {
                for (int j = 0; j < mm; j++)
                {
                    const double a = i <= j ? acc[(size_t)tri_index(i, j, mm)] : acc[(size_t)tri_index(j, i, mm)];
                    qc += x[(size_t)c * mm + i] * a * x[(size_t)c * mm + j];
                    ec += std::fabs(x[(size_t)c * mm + i]) * std::fabs(a) * std::fabs(x[(size_t)c * mm + j]);
                }
                qc -= 2.0 * x[(size_t)c * mm + i] * acc[(size_t)tri + (size_t)c * mm + i];
                ec += 2.0 * std::fabs(x[(size_t)c * mm + i]) * std::fabs(acc[(size_t)tri + (size_t)c * mm + i]);
            }
            q += (double)m.cw[c] * qc;
            e += (double)m.cw[c] * ec;
        }
        err = DBL_EPSILON * (double)mm * e * inv_norm;
        return q;
    };

    double err_prev = 0.0;
    const double q_prev = had_previous ? qform(x_prev, err_prev) : 0.0;
    const int cols = mm + nout;
    std::vector<double> g((size_t)mm * cols, 0.0);
    std::vector<double> x_new((size_t)nout * mm, 0.0);
    bool accepted = false;
    size_t taken = 0;

    for (size_t step = 0; step < sizeof(RIDGE_LADDER) / sizeof(RIDGE_LADDER[0]) && !accepted; step++)
    {
        const double ridge = RIDGE_LADDER[step] * mean_diag;

        // The triangle mirrored into the augmented system [A + ridge I | rhs^T].
        std::fill(g.begin(), g.end(), 0.0);
        for (int i = 0; i < mm; i++)
        {
            for (int j = i; j < mm; j++)
            {
                const double a = acc[(size_t)tri_index(i, j, mm)];
                g[(size_t)i * cols + j] = a;
                g[(size_t)j * cols + i] = a;
            }
            g[(size_t)i * cols + i] += ridge;
            for (int c = 0; c < nout; c++)
                g[(size_t)i * cols + mm + c] = acc[(size_t)tri + (size_t)c * mm + i];
        }
        if (!gauss_jordan(g, mm, nout))
            continue;

        // Rounded to the float the asset carries BEFORE it is judged.
        bool finite = true;
        for (int c = 0; c < nout; c++)
            for (int i = 0; i < mm; i++)
            {
                const double v = (double)(float)g[(size_t)i * cols + mm + c];
                x_new[(size_t)c * mm + i] = v;
                finite = finite && std::isfinite(v);
            }
        if (!finite)
            continue;
        double err_cand = 0.0;
        const double q_cand = qform(x_new, err_cand);
        if (!had_previous ||
            (norm_ok && (q_cand - q_prev) * inv_norm + err_prev + err_cand <= LS_ACCEPT_RISE))
        {
            accepted = true;
            taken = step;
        }
    }

    g_ridge_tally[accepted ? taken : RIDGE_REFUSED]++;

    if (accepted)
    {
        m.dec.w.assign((size_t)nout * nin, 0.0f);
        m.dec.b.assign((size_t)nout, 0.0f);
        for (int c = 0; c < nout; c++)
        {
            for (int i = 0; i < nin; i++)
                m.dec.w[(size_t)c * nin + i] = (float)x_new[(size_t)c * mm + i];
            m.dec.b[c] = (float)x_new[(size_t)c * mm + nin];
        }
    }
    else if (!had_previous)
    {
        // Every rung refused with nothing to fall back on: the best constant fit, the weighted mean target.
        m.dec.w.assign((size_t)nout * nin, 0.0f);
        m.dec.b.assign((size_t)nout, 0.0f);
        const double count = acc[(size_t)tri_index(nin, nin, mm)];
        for (int c = 0; c < nout; c++)
            m.dec.b[c] = count > 0.0 ? (float)(acc[(size_t)tri + (size_t)c * mm + nin] / count) : 0.0f;
    }
    // Otherwise m.dec is left exactly as the caller had it: no candidate lowered E, so the block takes no step.

    upload_decoder_to_device(d, m);
    return ms;
}

}   // namespace nntc_cpu
