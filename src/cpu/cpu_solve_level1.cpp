// cpu/cpu_solve_level1.cpp: block (b), level 1 as one sparse weighted least squares per plane, and block (c'), level 0
// as the same thing under --l0 bc8, on the CPU (docs/CPU_BACKEND_PLAN.md stage C5, the port of solve_level1.cu).
//
// solve_level1.cu's header is the WHY of every line below and is not repeated: with the other latent and the decoder
// held, E is an exact quadratic in the unknown plane; its 9-point block stencil is assembled by a GATHER, one texel at a
// time, from monomial sums expanded once against fixed matrices; the proximal step (H + lambda I) d = -grad, lambda =
// --ridge times the mean diagonal, is solved by conjugate gradients preconditioned by the block Jacobi of the stencil;
// the plane takes the correction and the movement probe measures it; and once the grid is frozen the same stencil is
// walked by four-colour Gauss-Seidel sweeps whose per-value step is the exact argmin over the grid. This file is a
// transcription. What differs, and why:
//
//   * THE REDUCTIONS. The CUDA arm reduces every inner product, the diagonal's three figures, the range and the
//     movement probe over a fixed grid of REDUCE_BLOCKS blocks through shared-memory trees. The CPU does not replay
//     those trees (the plan's section 0a): every reduction here is summed over chunks of CPU_CHUNK consecutive TEXELS -
//     each chunk's values in increasing index order - and the chunk partials are folded in chunk order after the join.
//     The chunk count is a function of the plane's size alone, so every inner product, and therefore every CG
//     iteration, is the same bits at every -j. They are not CUDA's bits, which is why the iteration count and the
//     residual can differ from CUDA's by a probe on a plane whose residual crosses its bar within a rounding of it;
//   * THE KERNEL BOUNDARIES. One CG iteration is thirteen launches on the device. Here it is three passes over the
//     plane - the mat-vec with the curvature's inner product; the two updates with the residual's norm, the
//     preconditioner and the second inner product; the new direction - with the three scalar steps (alpha, beta and
//     the residual) on the calling thread between them. Every value is the same arithmetic on the same operands in
//     the same order as in the kernel that computes it on the device; the passes only decide which of them one
//     pool run does, and a pass reads nothing another texel of the same pass writes (below);
//   * THE PLANES. The CUDA arm issues every plane on its own stream and runs the iteration in lockstep, dropping a
//     plane at the probe where its residual is under the bar. Here each plane runs its whole pipeline in turn (the
//     pool's rule 6: runs do not nest), and a plane stops at exactly the probe where the device would have dropped it,
//     so the iteration count and the correction it lands on are the device's own rule. CG_PROBE stays 4, the cap 200
//     and the bar 1e-10: a plane that overshoots its stop by up to three iterations on the device overshoots by the
//     same three here, because the correction it lands on is what the asset carries;
//   * THE CLOCKS. Each Level1Report's times are the wall clock around the same stretches the device's events bracket.
//
// Everything else is copied operation for operation: the monomial values and the form matrices, footprint_range, both
// assemblies (the gather, the four-tap dedup-merge that SUMS a clamped duplicate's weights, stencil_slot's -1 for the
// directions the pair's other texel owns, the `if (s == 0.0) continue` of the expansion, block (c')'s centre site as a
// single tap of weight 1), the block inverse and its identity fallback with the preconditioner STORED AS FLOAT, the
// mat-vec's plain in-bounds tests with no clamp (a different rule from the sampler's clamp-to-edge, deliberately not
// unified), alpha's zero on non-positive curvature and beta's on a zero rz, the movement probe's count kept as a double,
// the sweeps' four colours in order and their Gauss-Seidel over one texel's channels through model.h's level1_index and
// level1_value (called, not copied), and the dense direct twin solve_level1_dense_host, with its two downloads deleted
// and its own tap indexing kept (it is the independent witness, so it does not call cpu_sample.h).
//
// ZERO DATA RACES (section 3.4, where each kernel's row carries the argument). In short: the assemblies and every CG
// pass write only the entries of their own texels and read only what no chunk of the same run writes; the reductions
// return their partials by value into fold_chunks' own slots; and a colour pass of the sweeps writes only texels of its
// colour, every one of whose stencil neighbours is at Chebyshev distance 1 and therefore of another colour.

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"
#include "cpu/cpu_sample.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

namespace nntc_cpu
{

namespace
{

// solve_level1.cu's constants, copied: how often the stop is looked at, the iteration cap and the relative-residual
// bar. CG_PROBE is not a CPU tuning knob - the residual is computed every iteration here and could be tested every
// iteration - but the device tests it every fourth, and the iteration it stops at decides the correction (above).
const int CG_PROBE = 4;
const int CG_MAX_ITERATIONS = 200;
const double CG_TOLERANCE = 1e-10;

// 1 + C0 + C0 (C0 + 1) / 2 at the channel cap of 4.
const int MAX_MONOMIALS = 15;
static_assert(MAX_MONOMIALS == 1 + MAX_CHANNELS + MAX_CHANNELS * (MAX_CHANNELS + 1) / 2,
              "MAX_MONOMIALS must hold the whole quadratic form at the level-0 channel cap");

double since_ms(std::chrono::steady_clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// A per-texel pass over [0, n) in CPU_CHUNK chunks: body(t) for every texel, each chunk writing only its own texels.
template <class Body>
void per_texel(CpuModel* d, size_t n, const Body& body)
{
    d->pool->run(chunks_for(n, CPU_CHUNK), [&](int c) {
        const size_t begin = (size_t)c * CPU_CHUNK;
        const size_t end = begin + CPU_CHUNK < n ? begin + CPU_CHUNK : n;
        for (size_t t = begin; t < end; t++)
            body(t);
    });
}

// Two inner products carried through one pass.
struct Pair
{
    double a = 0.0, b = 0.0;
};

// ---------------------------------------------------------------------------------------------------------------------
// The host-side matrices, copied
// ---------------------------------------------------------------------------------------------------------------------

// The monomials of the form, in the one order the matrices and the sums both use: the constant, s_i, then s_i s_j.
inline void stencil_monomial_values(const float* s, int c0, double* mono)
{
    int n = 0;
    mono[n++] = 1.0;
    for (int i = 0; i < c0; i++)
        mono[n++] = (double)s[i];
    for (int i = 0; i < c0; i++)
        for (int j = i; j < c0; j++)
            mono[n++] = (double)s[i] * (double)s[j];
}

// X^T C X = P_0 + sum_i y_i P_i + sum_{i <= j} y_i y_j P_ij for X(y) = X0 + sum_i y_i Y_i.
void build_form_matrices(const Model& m, const std::vector<double>& x0, const std::vector<double>& y, int terms,
                         int dim, std::vector<double>& p)
{
    const int nout = m.nout;
    const int nmono = 1 + terms + terms * (terms + 1) / 2;
    p.assign((size_t)nmono * dim * dim, 0.0);

    auto product = [&](const double* x, const double* z, double* out)
    {
        for (int u = 0; u < dim; u++)
            for (int v = 0; v < dim; v++)
            {
                double s = 0.0;
                for (int r = 0; r < nout; r++)
                    s += (double)m.cw[(size_t)r] * x[(size_t)r * dim + u] * z[(size_t)r * dim + v];
                out[u * dim + v] += s;
            }
    };

    int slot = 0;
    product(x0.data(), x0.data(), p.data());
    slot++;
    for (int i = 0; i < terms; i++, slot++)
    {
        const double* y_i = y.data() + (size_t)i * nout * dim;
        product(y_i, x0.data(), p.data() + (size_t)slot * dim * dim);
        product(x0.data(), y_i, p.data() + (size_t)slot * dim * dim);
    }
    for (int i = 0; i < terms; i++)
        for (int j = i; j < terms; j++, slot++)
        {
            const double* y_i = y.data() + (size_t)i * nout * dim;
            const double* y_j = y.data() + (size_t)j * nout * dim;
            product(y_i, y_j, p.data() + (size_t)slot * dim * dim);
            if (i != j)
                product(y_j, y_i, p.data() + (size_t)slot * dim * dim);
        }
}

// Block (b)'s matrices: A and the sc-blocks M_i of the decoder, G(s) = A + sum_i s_i M_i.
void build_monomial_matrices(const Model& m, std::vector<double>& p)
{
    const int c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;
    std::vector<double> a((size_t)nout * c1), mi((size_t)MAX_CHANNELS * nout * c1, 0.0);
    for (int r = 0; r < nout; r++)
    {
        const float* row = m.dec.w.data() + (size_t)r * nin;
        for (int q = 0; q < c1; q++)
            a[(size_t)r * c1 + q] = (double)row[q];
        for (int i = 0; i < c0; i++)
            for (int q = 0; q < c1; q++)
                mi[((size_t)i * nout + r) * c1 + q] = (double)row[c1 + c0 + i * c1 + q];
    }
    build_form_matrices(m, a, mi, c0, c1, p);
}

// Block (c')'s matrices: B and the sc-columns N_q of the decoder, Q(c) = B + sum_q c_q N_q.
void build_monomial_matrices_level0(const Model& m, std::vector<double>& p)
{
    const int c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;
    std::vector<double> b((size_t)nout * c0), nq((size_t)MAX_CHANNELS * nout * c0, 0.0);
    for (int r = 0; r < nout; r++)
    {
        const float* row = m.dec.w.data() + (size_t)r * nin;
        for (int i = 0; i < c0; i++)
            b[(size_t)r * c0 + i] = (double)row[c1 + i];
        for (int q = 0; q < c1; q++)
            for (int i = 0; i < c0; i++)
                nq[((size_t)q * nout + r) * c0 + i] = (double)row[c1 + c0 + i * c1 + q];
    }
    build_form_matrices(m, b, nq, c1, c0, p);
}

// The pixels whose bilinear footprint can reach texel ti along one axis: (ti - 0.5) r - 1 <= p < (ti + 1.5) r.
inline void footprint_range(int ti, int n0, int n1, int& lo, int& hi)
{
    const double r = (double)n0 / (double)n1;
    lo = (int)std::ceil(((double)ti - 0.5) * r - 1.0);
    hi = (int)std::ceil(((double)ti + 1.5) * r) - 1;
    lo = lo < 0 ? 0 : lo;
    hi = hi > n0 - 1 ? n0 - 1 : hi;
}

// ---------------------------------------------------------------------------------------------------------------------
// The two assemblies, copied: one texel each, gathering every site whose footprint can reach it
// ---------------------------------------------------------------------------------------------------------------------

// The five owned blocks of one texel, expanded from its monomial sums: block = sum over monomials of its weighted sum
// times that monomial's matrix, a zero sum skipped.
inline void expand_blocks(const double* sums, int nmono, const double* mono_p, int c, double* out_h)
{
    for (int slot = 0; slot < STENCIL_BLOCKS; slot++)
    {
        double* blk = out_h + (size_t)slot * c * c;
        for (int i = 0; i < c * c; i++)
            blk[i] = 0.0;
        for (int i = 0; i < nmono; i++)
        {
            const double s = sums[(size_t)slot * nmono + i];
            if (s == 0.0)
                continue;
            const double* mat = mono_p + (size_t)i * c * c;
            for (int e = 0; e < c * c; e++)
                blk[e] += s * mat[e];
        }
    }
}

// k_stencil_assemble: level-1 texel ti's five blocks and its gradient.
void stencil_assemble_texel(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0,
                            int c1, int nin, int nout, const float* weights, const float* bias, const float* cw,
                            const double* mono_p, int nmono, int k_count, double* stencil, double* grad, size_t ti)
{
    const int tx = (int)(ti % (size_t)w1), ty = (int)(ti / (size_t)w1);

    double sums[STENCIL_BLOCKS * MAX_MONOMIALS];
    double g[MAX_CHANNELS];
    for (int i = 0; i < STENCIL_BLOCKS * nmono; i++)
        sums[i] = 0.0;
    for (int i = 0; i < c1; i++)
        g[i] = 0.0;

    int px_lo, px_hi, py_lo, py_hi;
    footprint_range(tx, w0, w1, px_lo, px_hi);
    footprint_range(ty, h0, h1, py_lo, py_hi);

    for (int py = py_lo; py <= py_hi; py++)
        for (int px = px_lo; px <= px_hi; px++)
            for (int j = 0; j <= k_count; j++)
            {
                float dx, dy;
                site_delta(k_count, j, px, py, dx, dy);
                const BilinearTap tp = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w1, h1);

                // The four corners merged by texel index, so a clamped duplicate adds its weight to the one texel.
                int nx[4], ny[4];
                float tw[4];
                int taps = 0;
                const int cx[4] = { tp.x0, tp.x1, tp.x0, tp.x1 };
                const int cy[4] = { tp.y0, tp.y0, tp.y1, tp.y1 };
                const float cwt[4] = { (1.0f - tp.fx) * (1.0f - tp.fy), tp.fx * (1.0f - tp.fy),
                                       (1.0f - tp.fx) * tp.fy, tp.fx * tp.fy };
                for (int q = 0; q < 4; q++)
                {
                    int found = -1;
                    for (int e = 0; e < taps; e++)
                        if (nx[e] == cx[q] && ny[e] == cy[q])
                            found = e;
                    if (found >= 0)
                        tw[found] += cwt[q];
                    else
                    {
                        nx[taps] = cx[q];
                        ny[taps] = cy[q];
                        tw[taps] = cwt[q];
                        taps++;
                    }
                }
                int self = -1;
                for (int e = 0; e < taps; e++)
                    if (nx[e] == tx && ny[e] == ty)
                        self = e;
                if (self < 0)
                    continue;
                const float w_self = tw[self];

                // The site's level-0 sample, its level-1 sample under the current plane, and its target.
                float sbar[MAX_CHANNELS], cbar[MAX_CHANNELS], target[MAX_NOUT];
                sample_bilinear(v1, w1, c1, tp, cbar);
                if (j == 0)
                {
                    const size_t pi = (size_t)py * w0 + px;
                    for (int i = 0; i < c0; i++)
                        sbar[i] = v0[pi * c0 + i];
                    for (int r = 0; r < nout; r++)
                        target[r] = src[pi * nout + r];
                }
                else
                {
                    const BilinearTap t0 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w0, h0);
                    sample_bilinear(v0, w0, c0, t0, sbar);
                    sample_bilinear(src, w0, nout, t0, target);
                }

                // G = A + sum_i sbar_i M_i, and the residual of the current plane r = b + B sbar + G cbar - target.
                float gmat[MAX_NOUT * MAX_CHANNELS], res[MAX_NOUT];
                for (int r = 0; r < nout; r++)
                {
                    const float* row = weights + (size_t)r * nin;
                    float out = bias[r];
                    for (int i = 0; i < c0; i++)
                        out += row[c1 + i] * sbar[i];
                    for (int q = 0; q < c1; q++)
                    {
                        float gg = row[q];
                        for (int i = 0; i < c0; i++)
                            gg += sbar[i] * row[c1 + c0 + i * c1 + q];
                        gmat[r * c1 + q] = gg;
                        out += gg * cbar[q];
                    }
                    res[r] = out - target[r];
                }

                // G^T diag(cw) r, formed per site, and the monomials that stand for G^T diag(cw) G.
                double mono[MAX_MONOMIALS];
                stencil_monomial_values(sbar, c0, mono);
                for (int a = 0; a < c1; a++)
                {
                    double acc = 0.0;
                    for (int r = 0; r < nout; r++)
                        acc += (double)cw[r] * (double)gmat[r * c1 + a] * (double)res[r];
                    g[a] += (double)w_self * acc;
                }
                for (int e = 0; e < taps; e++)
                {
                    const int slot = stencil_slot(nx[e] - tx, ny[e] - ty);
                    if (slot < 0)
                        continue;
                    const double scale = (double)w_self * (double)tw[e];
                    double* dst = sums + (size_t)slot * nmono;
                    for (int i = 0; i < nmono; i++)
                        dst[i] += scale * mono[i];
                }
            }

    expand_blocks(sums, nmono, mono_p, c1, stencil + ti * (size_t)(STENCIL_BLOCKS * c1 * c1));
    for (int i = 0; i < c1; i++)
        grad[ti * (size_t)c1 + i] = g[i];
}

// k_stencil_assemble0: level-0 texel ti's five blocks and its gradient, block (c') under --l0 bc8. The one structural
// difference from the level-1 assembly is the centre site, where level 0 is read NEAREST: one tap, this texel, weight 1.
void stencil_assemble0_texel(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0,
                             int c1, int nin, int nout, const float* weights, const float* bias, const float* cw,
                             const double* mono_p, int nmono, int k_count, double* stencil, double* grad, size_t ti)
{
    const int tx = (int)(ti % (size_t)w0), ty = (int)(ti / (size_t)w0);

    double sums[STENCIL_BLOCKS * MAX_MONOMIALS];
    double g[MAX_CHANNELS];
    for (int i = 0; i < STENCIL_BLOCKS * nmono; i++)
        sums[i] = 0.0;
    for (int i = 0; i < c0; i++)
        g[i] = 0.0;

    int px_lo, px_hi, py_lo, py_hi;
    footprint_range(tx, w0, w0, px_lo, px_hi);
    footprint_range(ty, h0, h0, py_lo, py_hi);

    for (int py = py_lo; py <= py_hi; py++)
        for (int px = px_lo; px <= px_hi; px++)
            for (int j = 0; j <= k_count; j++)
            {
                float dx, dy;
                site_delta(k_count, j, px, py, dx, dy);

                int nx[4], ny[4];
                float tw[4];
                int taps = 0;
                float sbar[MAX_CHANNELS], cbar[MAX_CHANNELS], target[MAX_NOUT];
                const BilinearTap tp = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w1, h1);
                sample_bilinear(v1, w1, c1, tp, cbar);
                if (j == 0)
                {
                    const size_t pi = (size_t)py * w0 + px;
                    nx[0] = px;
                    ny[0] = py;
                    tw[0] = 1.0f;
                    taps = 1;
                    for (int i = 0; i < c0; i++)
                        sbar[i] = v0[pi * c0 + i];
                    for (int r = 0; r < nout; r++)
                        target[r] = src[pi * nout + r];
                }
                else
                {
                    const BilinearTap t0 = bilinear_tap_pixel(px, py, dx, dy, w0, h0, w0, h0);
                    sample_bilinear(v0, w0, c0, t0, sbar);
                    sample_bilinear(src, w0, nout, t0, target);
                    const int cx[4] = { t0.x0, t0.x1, t0.x0, t0.x1 };
                    const int cy[4] = { t0.y0, t0.y0, t0.y1, t0.y1 };
                    const float cwt[4] = { (1.0f - t0.fx) * (1.0f - t0.fy), t0.fx * (1.0f - t0.fy),
                                           (1.0f - t0.fx) * t0.fy, t0.fx * t0.fy };
                    for (int q = 0; q < 4; q++)
                    {
                        int found = -1;
                        for (int e = 0; e < taps; e++)
                            if (nx[e] == cx[q] && ny[e] == cy[q])
                                found = e;
                        if (found >= 0)
                            tw[found] += cwt[q];
                        else
                        {
                            nx[taps] = cx[q];
                            ny[taps] = cy[q];
                            tw[taps] = cwt[q];
                            taps++;
                        }
                    }
                }
                int self = -1;
                for (int e = 0; e < taps; e++)
                    if (nx[e] == tx && ny[e] == ty)
                        self = e;
                if (self < 0)
                    continue;
                const float w_self = tw[self];

                // Q = B + sum_q cbar_q N_q, and the residual of the current plane r = b + A cbar + Q sbar - target.
                float qmat[MAX_NOUT * MAX_CHANNELS], res[MAX_NOUT];
                for (int r = 0; r < nout; r++)
                {
                    const float* row = weights + (size_t)r * nin;
                    float out = bias[r];
                    for (int q = 0; q < c1; q++)
                        out += row[q] * cbar[q];
                    for (int i = 0; i < c0; i++)
                    {
                        float qq = row[c1 + i];
                        for (int q = 0; q < c1; q++)
                            qq += cbar[q] * row[c1 + c0 + i * c1 + q];
                        qmat[r * c0 + i] = qq;
                        out += qq * sbar[i];
                    }
                    res[r] = out - target[r];
                }

                // Q^T diag(cw) r per site, and the monomials of the level-1 sample that stand for Q^T diag(cw) Q.
                double mono[MAX_MONOMIALS];
                stencil_monomial_values(cbar, c1, mono);
                for (int a = 0; a < c0; a++)
                {
                    double acc = 0.0;
                    for (int r = 0; r < nout; r++)
                        acc += (double)cw[r] * (double)qmat[r * c0 + a] * (double)res[r];
                    g[a] += (double)w_self * acc;
                }
                for (int e = 0; e < taps; e++)
                {
                    const int slot = stencil_slot(nx[e] - tx, ny[e] - ty);
                    if (slot < 0)
                        continue;
                    const double scale = (double)w_self * (double)tw[e];
                    double* dst = sums + (size_t)slot * nmono;
                    for (int i = 0; i < nmono; i++)
                        dst[i] += scale * mono[i];
                }
            }

    expand_blocks(sums, nmono, mono_p, c0, stencil + ti * (size_t)(STENCIL_BLOCKS * c0 * c0));
    for (int i = 0; i < c0; i++)
        grad[ti * (size_t)c0 + i] = g[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// The preconditioner and the mat-vec, copied
// ---------------------------------------------------------------------------------------------------------------------

// solve_level1.cu's invert_block (a __device__ function there, this backend's own copy here): Gauss-Jordan with
// partial pivoting on [a | I] for n <= MAX_CHANNELS, in double; a pivot that is not positive refuses.
bool invert_block(double* a, int n, double* inv)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i * n + j] = i == j ? 1.0 : 0.0;
    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double best = std::fabs(a[k * n + k]);
        for (int i = k + 1; i < n; i++)
            if (std::fabs(a[i * n + k]) > best)
            {
                best = std::fabs(a[i * n + k]);
                pivot = i;
            }
        if (!(best > 0.0))
            return false;
        if (pivot != k)
            for (int j = 0; j < n; j++)
            {
                double t = a[k * n + j];
                a[k * n + j] = a[pivot * n + j];
                a[pivot * n + j] = t;
                t = inv[k * n + j];
                inv[k * n + j] = inv[pivot * n + j];
                inv[pivot * n + j] = t;
            }
        const double s = 1.0 / a[k * n + k];
        for (int j = 0; j < n; j++)
        {
            a[k * n + j] *= s;
            inv[k * n + j] *= s;
        }
        for (int i = 0; i < n; i++)
        {
            if (i == k)
                continue;
            const double f = a[i * n + k];
            if (f == 0.0)
                continue;
            for (int j = 0; j < n; j++)
            {
                a[i * n + j] -= f * a[k * n + j];
                inv[i * n + j] -= f * inv[k * n + j];
            }
        }
    }
    return true;
}

// k_precondition: (H[t][t] + lambda I)^-1 in double, KEPT AS FLOATS - it changes only how fast the iteration converges,
// not what it converges to - and the identity where the block is singular even with the ridge.
void precondition_texel(const double* stencil, int c, double lambda, float* minv, size_t t)
{
    const double* dg = stencil + t * (size_t)(STENCIL_BLOCKS * c * c);
    double a[MAX_CHANNELS * MAX_CHANNELS], inv[MAX_CHANNELS * MAX_CHANNELS];
    for (int i = 0; i < c; i++)
        for (int j = 0; j < c; j++)
            a[i * c + j] = dg[i * c + j] + (i == j ? lambda : 0.0);
    if (!invert_block(a, c, inv))
        for (int i = 0; i < c; i++)
            for (int j = 0; j < c; j++)
                inv[i * c + j] = i == j ? 1.0 : 0.0;
    float* o = minv + t * (size_t)(c * c);
    for (int i = 0; i < c * c; i++)
        o[i] = (float)inv[i];
}

// k_apply_precond: z_t = Minv_t r_t.
inline void apply_precond_texel(const float* minv, const double* r, int c, double* z, size_t t)
{
    const float* a = minv + t * (size_t)(c * c);
    const double* x = r + t * (size_t)c;
    double* o = z + t * (size_t)c;
    for (int i = 0; i < c; i++)
    {
        double acc = 0.0;
        for (int j = 0; j < c; j++)
            acc += (double)a[i * c + j] * x[j];
        o[i] = acc;
    }
}

// k_matvec: y_t = (H[t][t] + lambda I) x_t + sum over the eight neighbours of H[t][n] x_n, the four directions the texel
// does not own read as the transpose of the neighbour's block. The neighbours are tested for being inside the plane and
// NOT clamped: a texel on the edge simply has fewer of them.
inline void matvec_texel(const double* stencil, int w, int h, int c, double lambda, const double* x, double* y, size_t t)
{
    const int tx = (int)(t % (size_t)w), ty = (int)(t / (size_t)w);
    const size_t bstride = (size_t)(STENCIL_BLOCKS * c * c);

    double acc[MAX_CHANNELS];
    const double* own = stencil + t * bstride;
    const double* xt = x + t * (size_t)c;
    for (int i = 0; i < c; i++)
    {
        double s = lambda * xt[i];
        for (int j = 0; j < c; j++)
            s += own[i * c + j] * xt[j];
        acc[i] = s;
    }
    for (int slot = 1; slot < STENCIL_BLOCKS; slot++)
    {
        int dx, dy;
        stencil_offset(slot, dx, dy);
        const int fx = tx + dx, fy = ty + dy;
        if (fx >= 0 && fx < w && fy >= 0 && fy < h)
        {
            const double* blk = own + (size_t)slot * c * c;
            const double* xn = x + ((size_t)fy * w + fx) * (size_t)c;
            for (int i = 0; i < c; i++)
                for (int j = 0; j < c; j++)
                    acc[i] += blk[i * c + j] * xn[j];
        }
        const int bx = tx - dx, by = ty - dy;
        if (bx >= 0 && bx < w && by >= 0 && by < h)
        {
            const size_t nt = (size_t)by * w + bx;
            const double* blk = stencil + nt * bstride + (size_t)slot * c * c;
            const double* xn = x + nt * (size_t)c;
            for (int i = 0; i < c; i++)
                for (int j = 0; j < c; j++)
                    acc[i] += blk[j * c + i] * xn[j];
        }
    }
    double* o = y + t * (size_t)c;
    for (int i = 0; i < c; i++)
        o[i] = acc[i];
}

// ---------------------------------------------------------------------------------------------------------------------
// The quantised phase, copied: one texel of one colour
// ---------------------------------------------------------------------------------------------------------------------

// k_quant_sweep for texel (tx, ty): the neighbours' share of (H d)[t] once, then the channels IN TURN, each one's exact
// grid argmin using the values the earlier channels of the same texel have just taken. Returns the indices it changed.
unsigned int quant_sweep_texel(const double* stencil, const double* grad, const float* prev, int w, int h, int c,
                               double lambda, const float* lo, const float* hi, int bits, double* delta, float* v,
                               uint8_t* k, int tx, int ty)
{
    const size_t t = (size_t)ty * w + tx;
    const size_t bstride = (size_t)(STENCIL_BLOCKS * c * c);
    const double* own = stencil + t * bstride;

    double nb[MAX_CHANNELS];
    for (int i = 0; i < c; i++)
        nb[i] = 0.0;
    for (int slot = 1; slot < STENCIL_BLOCKS; slot++)
    {
        int dx, dy;
        stencil_offset(slot, dx, dy);
        const int fx = tx + dx, fy = ty + dy;
        if (fx >= 0 && fx < w && fy >= 0 && fy < h)
        {
            const double* blk = own + (size_t)slot * c * c;
            const double* xn = delta + ((size_t)fy * w + fx) * (size_t)c;
            for (int i = 0; i < c; i++)
                for (int j = 0; j < c; j++)
                    nb[i] += blk[i * c + j] * xn[j];
        }
        const int bx = tx - dx, by = ty - dy;
        if (bx >= 0 && bx < w && by >= 0 && by < h)
        {
            const size_t nt = (size_t)by * w + bx;
            const double* blk = stencil + nt * bstride + (size_t)slot * c * c;
            const double* xn = delta + nt * (size_t)c;
            for (int i = 0; i < c; i++)
                for (int j = 0; j < c; j++)
                    nb[i] += blk[j * c + i] * xn[j];
        }
    }

    unsigned int changed = 0;
    for (int q = 0; q < c; q++)
    {
        const size_t idx = t * (size_t)c + (size_t)q;
        const float l = lo[q], span = hi[q] - lo[q];
        const double a = own[q * c + q] + lambda;
        if (!(a > 0.0) || !(span > 0.0f))
            continue;
        double rest = nb[q];
        for (int j = 0; j < c; j++)
            if (j != q)
                rest += own[q * c + j] * delta[t * (size_t)c + (size_t)j];
        const double b = grad[idx] + rest;
        const double want = (double)prev[idx] - b / a;

        const int kk = level1_index(l, hi[q], bits, (float)want);
        const float value = level1_value(l, hi[q], bits, kk);
        if ((int)k[idx] != kk)
        {
            k[idx] = (uint8_t)kk;
            changed++;
        }
        v[idx] = value;
        delta[idx] = (double)value - (double)prev[idx];
    }
    return changed;
}

// ---------------------------------------------------------------------------------------------------------------------
// The drivers
// ---------------------------------------------------------------------------------------------------------------------

// assemble_all for one plane: the gather over every texel, then the diagonal's sum, smallest and largest and the
// proximal ridge from its mean (k_diag_partial and k_reduce_diag), into the plane's scalar row.
void assemble_plane(CpuModel* d, const Model& m, CpuPlane& p, const PlaneWork& w, int k, double ridge, bool level0,
                    Level1Report& rep)
{
    const size_t texels = w.texels();
    const int c = w.c;
    auto t0 = std::chrono::steady_clock::now();
    const float* v0 = p.v0.data();
    const float* v1 = p.v1.data();
    const float* src = p.src.data();
    const float* weights = d->weights.data();
    const float* bias = d->bias.data();
    const float* cw = d->cw.data();
    if (level0)
        per_texel(d, texels, [&](size_t t) {
            stencil_assemble0_texel(v0, v1, src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout, weights, bias,
                                    cw, w.mono, w.nmono, k, w.stencil, w.grad, t);
        });
    else
        per_texel(d, texels, [&](size_t t) {
            stencil_assemble_texel(v0, v1, src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout, weights, bias,
                                   cw, w.mono, w.nmono, k, w.stencil, w.grad, t);
        });
    rep.ms_assemble = since_ms(t0);

    t0 = std::chrono::steady_clock::now();
    struct Diag
    {
        double sum = 0.0, lo = 1e300, hi = -1e300;
    };
    const double* stencil = w.stencil;
    const Diag dg = fold_chunks(
        *d->pool, chunks_for(texels, CPU_CHUNK), Diag(),
        [&](int ch) {
            Diag part;
            const size_t begin = (size_t)ch * CPU_CHUNK;
            const size_t end = begin + CPU_CHUNK < texels ? begin + CPU_CHUNK : texels;
            for (size_t t = begin; t < end; t++)
            {
                const double* dd = stencil + t * (size_t)(STENCIL_BLOCKS * c * c);
                for (int a = 0; a < c; a++)
                {
                    const double x = dd[a * c + a];
                    part.sum += x;
                    part.lo = x < part.lo ? x : part.lo;
                    part.hi = x > part.hi ? x : part.hi;
                }
            }
            return part;
        },
        [](Diag acc, const Diag& part) {
            acc.sum += part.sum;
            acc.lo = part.lo < acc.lo ? part.lo : acc.lo;
            acc.hi = part.hi > acc.hi ? part.hi : acc.hi;
            return acc;
        });
    double* s = p.scalars.data();
    const size_t unknowns = w.values();
    s[SC_DIAG_SUM] = dg.sum;
    s[SC_DIAG_MIN] = dg.lo;
    s[SC_DIAG_MAX] = dg.hi;
    s[SC_LAMBDA] = unknowns > 0 ? ridge * dg.sum / (double)unknowns : 0.0;
    rep.ms_precond = since_ms(t0);
}

// The monomial matrices of this block, rebuilt from the decoder the caller holds - once a call, before any plane reads
// them - into the model's own copy, which is what the assembly reads.
void build_block_matrices(CpuModel* d, const Model& m, bool level0)
{
    std::vector<double> mono;
    if (level0)
        build_monomial_matrices_level0(m, mono);
    else
        build_monomial_matrices(m, mono);
    std::vector<double>& dst = level0 ? d->mono0 : d->mono;
    std::copy(mono.begin(), mono.begin() + (std::ptrdiff_t)std::min(mono.size(), dst.size()), dst.begin());
}

// read_scalars, copied: what the report takes out of one plane's scalar row.
void read_scalars(const CpuPlane& p, int channels, Level1Report& rep)
{
    const double* s = p.scalars.data();
    rep.lambda = s[SC_LAMBDA];
    rep.min_diag = s[SC_DIAG_MIN];
    rep.max_diag = s[SC_DIAG_MAX];
    rep.residual = s[SC_RESID];
    rep.moved_values = (long long)s[SC_MOVED];
    rep.delta2 = s[SC_DELTA2];
    rep.plane2 = s[SC_PLANE2];
    for (int c = 0; c < channels; c++)
    {
        rep.lo[c] = (float)s[SC_LO + c];
        rep.hi[c] = (float)s[SC_HI + c];
    }
}

// One plane's preconditioned conjugate gradients on (H + lambda I) delta = -grad from delta = 0, then the correction
// and the movement probe - solve_plane_all's per-plane work, in the device's order.
void solve_plane(CpuModel* d, CpuPlane& p, const PlaneWork& w, Level1Report& rep)
{
    const size_t texels = w.texels();
    const int c = w.c;
    const int chunks = chunks_for(texels, CPU_CHUNK);
    double* s = p.scalars.data();
    const double lambda = s[SC_LAMBDA];
    double* x = w.cg_x;
    double* r = w.cg_r;
    double* z = w.cg_z;
    double* dir = w.cg_p;
    double* ap = w.cg_ap;
    float* minv = w.precond;
    const double* grad = w.grad;
    const double* stencil = w.stencil;
    // The values of chunk ch: its texels' channels, in index order.
    auto value_range = [&](int ch, size_t& begin, size_t& end) {
        const size_t tb = (size_t)ch * CPU_CHUNK;
        const size_t te = tb + CPU_CHUNK < texels ? tb + CPU_CHUNK : texels;
        begin = tb * (size_t)c;
        end = te * (size_t)c;
    };
    auto add = [](Pair acc, const Pair& part) {
        acc.a += part.a;
        acc.b += part.b;
        return acc;
    };

    auto t0 = std::chrono::steady_clock::now();
    // k_precondition, then the start of the iteration: delta = 0, r = -grad, z = Minv r, p = z, and |r|^2 and r.z.
    const Pair start = fold_chunks(*d->pool, chunks, Pair(), [&](int ch) {
        size_t begin, end;
        value_range(ch, begin, end);
        for (size_t t = begin / (size_t)c; t < end / (size_t)c; t++)
            precondition_texel(stencil, c, lambda, minv, t);
        for (size_t i = begin; i < end; i++)
        {
            x[i] = 0.0;
            r[i] = -grad[i];
        }
        for (size_t t = begin / (size_t)c; t < end / (size_t)c; t++)
            apply_precond_texel(minv, r, c, z, t);
        Pair part;
        for (size_t i = begin; i < end; i++)
        {
            dir[i] = z[i];
            part.a += r[i] * r[i];
        }
        for (size_t i = begin; i < end; i++)
            part.b += r[i] * z[i];
        return part;
    }, add);
    rep.ms_precond += since_ms(t0);
    s[SC_RHS2] = start.a;
    s[SC_RZ] = start.b;

    t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < CG_MAX_ITERATIONS; it++)
    {
        // k_matvec and the curvature p . Ap.
        const double pap = fold_chunks(*d->pool, chunks, 0.0, [&](int ch) {
            size_t begin, end;
            value_range(ch, begin, end);
            for (size_t t = begin / (size_t)c; t < end / (size_t)c; t++)
                matvec_texel(stencil, w.w, w.h, c, lambda, dir, ap, t);
            double part = 0.0;
            for (size_t i = begin; i < end; i++)
                part += dir[i] * ap[i];
            return part;
        }, [](double acc, double part) { return acc + part; });
        s[SC_PAP] = pap;

        // k_cg_alpha: no step along a direction of non-positive curvature.
        s[SC_ALPHA] = s[SC_PAP] > 0.0 ? s[SC_RZ] / s[SC_PAP] : 0.0;
        const double alpha = s[SC_ALPHA];

        // The two k_axpy, |r|^2, k_apply_precond and r . z.
        const Pair upd = fold_chunks(*d->pool, chunks, Pair(), [&](int ch) {
            size_t begin, end;
            value_range(ch, begin, end);
            const double plus = 1.0, minus = -1.0;
            for (size_t i = begin; i < end; i++)
                x[i] += plus * alpha * dir[i];
            for (size_t i = begin; i < end; i++)
                r[i] += minus * alpha * ap[i];
            Pair part;
            for (size_t i = begin; i < end; i++)
                part.a += r[i] * r[i];
            for (size_t t = begin / (size_t)c; t < end / (size_t)c; t++)
                apply_precond_texel(minv, r, c, z, t);
            for (size_t i = begin; i < end; i++)
                part.b += r[i] * z[i];
            return part;
        }, add);
        s[SC_RR] = upd.a;
        s[SC_RZ_NEW] = upd.b;

        // k_cg_beta, which also writes the relative residual the probe reads.
        const double rz = s[SC_RZ], rz_new = s[SC_RZ_NEW];
        s[SC_BETA] = rz != 0.0 ? rz_new / rz : 0.0;
        s[SC_RZ] = rz_new;
        const double rhs2 = s[SC_RHS2], rr = s[SC_RR];
        s[SC_RESID] = rhs2 > 0.0 ? std::sqrt((rr > 0.0 ? rr : 0.0) / rhs2) : 0.0;
        const double beta = s[SC_BETA];

        // k_xpby: p = z + beta p.
        per_texel(d, texels * (size_t)c, [&](size_t i) { dir[i] = z[i] + beta * dir[i]; });
        rep.iterations = it + 1;

        // The probe, every CG_PROBE iterations, with the device's own test: a plane stays while its residual is at or
        // above the bar, so a NaN residual leaves exactly as it does there.
        if ((it + 1) % CG_PROBE != 0)
            continue;
        if (!(s[SC_RESID] >= CG_TOLERANCE))
            break;
    }

    // k_add_correction, then the range the probe measures the correction against (k_range_partial / k_reduce_range).
    float* v = w.v;
    struct Range
    {
        double lo[MAX_CHANNELS], hi[MAX_CHANNELS];
    };
    Range init;
    for (int q = 0; q < MAX_CHANNELS; q++)
    {
        init.lo[q] = 1e300;
        init.hi[q] = -1e300;
    }
    const Range range = fold_chunks(*d->pool, chunks, init, [&](int ch) {
        size_t begin, end;
        value_range(ch, begin, end);
        for (size_t i = begin; i < end; i++)
            v[i] = (float)((double)v[i] + x[i]);
        Range part = init;
        for (int q = 0; q < c; q++)
            for (size_t i = begin + (size_t)q; i < end; i += (size_t)c)
            {
                const double val = (double)v[i];
                part.lo[q] = val < part.lo[q] ? val : part.lo[q];
                part.hi[q] = val > part.hi[q] ? val : part.hi[q];
            }
        return part;
    }, [&](Range acc, const Range& part) {
        for (int q = 0; q < c; q++)
        {
            acc.lo[q] = part.lo[q] < acc.lo[q] ? part.lo[q] : acc.lo[q];
            acc.hi[q] = part.hi[q] > acc.hi[q] ? part.hi[q] : acc.hi[q];
        }
        return acc;
    });
    for (int q = 0; q < c; q++)
    {
        s[SC_LO + q] = range.lo[q];
        s[SC_HI + q] = range.hi[q];
    }

    // k_movement_partial: values that moved by more than half a grid step of the plane's own range (counted as a
    // double, as the device counts it), and the square norms of the correction and of the corrected plane.
    struct Movement
    {
        double moved = 0.0, d2 = 0.0, p2 = 0.0;
    };
    const Movement mv = fold_chunks(*d->pool, chunks, Movement(), [&](int ch) {
        size_t begin, end;
        value_range(ch, begin, end);
        Movement part;
        for (size_t i = begin; i < end; i++)
        {
            const int q = (int)(i % (size_t)c);
            const double step = (s[SC_HI + q] - s[SC_LO + q]) / (double)w.levels;
            if (std::fabs(x[i]) > 0.5 * step)
                part.moved += 1.0;
            part.d2 += x[i] * x[i];
            part.p2 += (double)v[i] * (double)v[i];
        }
        return part;
    }, [](Movement acc, const Movement& part) {
        acc.moved += part.moved;
        acc.d2 += part.d2;
        acc.p2 += part.p2;
        return acc;
    });
    s[SC_MOVED] = mv.moved;
    s[SC_DELTA2] = mv.d2;
    s[SC_PLANE2] = mv.p2;
    rep.ms_cg = since_ms(t0);
}

// ONE BLOCK OVER ONE LATENT, continuous: level 1 (block (b)) or, under --l0 bc8, level 0 (block (c')).
double solve_plane_all(CpuModel* d, const Model& m, int k, double ridge, bool level0, std::vector<Level1Report>& rep)
{
    const auto start = std::chrono::steady_clock::now();
    const size_t np = d->planes.size();
    const int nc = level0 ? m.c0 : m.c1;
    rep.assign(np, Level1Report());
    build_block_matrices(d, m, level0);
    for (size_t i = 0; i < np; i++)
    {
        CpuPlane& p = d->planes[i];
        const PlaneWork w = plane_work(d, m, p, level0);
        assemble_plane(d, m, p, w, k, ridge, level0, rep[i]);
        solve_plane(d, p, w, rep[i]);
        rep[i].quantised = false;
        read_scalars(p, nc, rep[i]);
    }
    return since_ms(start);
}

// The same block on a frozen grid: the same assembly, then `sweeps` four-colour Gauss-Seidel passes. The colours run
// in order, each one a pool run of its own: pass q + 1 reads what pass q wrote, so the runs are the barrier between
// them, and inside a pass no texel reads a value another texel of the pass writes.
double sweep_plane_all(CpuModel* d, const Model& m, int k, double ridge, int sweeps, bool level0,
                       std::vector<Level1Report>& rep)
{
    const auto start = std::chrono::steady_clock::now();
    const size_t np = d->planes.size();
    const int nc = level0 ? m.c0 : m.c1;
    rep.assign(np, Level1Report());
    build_block_matrices(d, m, level0);
    for (size_t i = 0; i < np; i++)
    {
        CpuPlane& p = d->planes[i];
        const PlaneWork w = plane_work(d, m, p, level0);
        assemble_plane(d, m, p, w, k, ridge, level0, rep[i]);

        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = w.values();
        // The correction is measured from the plane the stencil was assembled at, and starts at zero.
        per_texel(d, n, [&](size_t e) {
            w.prev[e] = w.v[e];
            w.cg_x[e] = 0.0;
        });
        p.counters[0] = 0;
        const double lambda = p.scalars[SC_LAMBDA];
        for (int sw = 0; sw < sweeps; sw++)
            for (int colour = 0; colour < 4; colour++)
            {
                const int cx = colour & 1, cy = colour >> 1;
                const int cw1 = (w.w - cx + 1) / 2, ch1 = (w.h - cy + 1) / 2;
                const size_t count = (size_t)cw1 * (size_t)ch1;
                if (count == 0)
                    continue;
                const unsigned int changed = fold_chunks(
                    *d->pool, chunks_for(count, CPU_CHUNK), 0u,
                    [&](int ch) {
                        unsigned int part = 0;
                        const size_t begin = (size_t)ch * CPU_CHUNK;
                        const size_t end = begin + CPU_CHUNK < count ? begin + CPU_CHUNK : count;
                        for (size_t ci = begin; ci < end; ci++)
                        {
                            const int tx = cx + 2 * (int)(ci % (size_t)cw1);
                            const int ty = cy + 2 * (int)(ci / (size_t)cw1);
                            part += quant_sweep_texel(w.stencil, w.grad, w.prev, w.w, w.h, nc, lambda, w.lo, w.hi,
                                                      w.bits, w.cg_x, w.v, w.k, tx, ty);
                        }
                        return part;
                    },
                    [](unsigned int acc, unsigned int part) { return acc + part; });
                p.counters[0] += changed;
            }
        rep[i].quantised = true;
        rep[i].sweeps = sweeps;
        rep[i].ms_sweeps = since_ms(t0);
        read_scalars(p, nc, rep[i]);
        // The scalar row persists between blocks, so the fields only the continuous path writes are cleared rather
        // than reported as this block's (solve_level1.cu says why).
        rep[i].iterations = 0;
        rep[i].residual = 0.0;
        rep[i].moved_values = 0;
        rep[i].delta2 = 0.0;
        rep[i].plane2 = 0.0;
        rep[i].moved = (long long)p.counters[0];
    }
    return since_ms(start);
}

}   // namespace

// The four public entry points, which say which latent each block solves and share everything else.
double solve_level1_all(CpuModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return solve_plane_all(d, m, k, ridge, false, rep);
}

double sweep_level1_all(CpuModel* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep)
{
    return sweep_plane_all(d, m, k, ridge, sweeps, false, rep);
}

double solve_level0_cont_all(CpuModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return solve_plane_all(d, m, k, ridge, true, rep);
}

double sweep_level0_cont_all(CpuModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep)
{
    return sweep_plane_all(d, m, k, ridge, sweeps, true, rep);
}

// Block (c')'s assembly alone, for the BC refinement: the stencil and the gradient at a ZERO ridge, the plane as it
// stands kept as the point the correction is measured from, and the correction at zero.
double assemble_level0_for_refine(CpuModel* d, const Model& m, int k)
{
    const auto start = std::chrono::steady_clock::now();
    build_block_matrices(d, m, true);
    for (CpuPlane& p : d->planes)
    {
        const PlaneWork w = plane_work(d, m, p, true);
        Level1Report scratch;
        assemble_plane(d, m, p, w, k, 0.0, true, scratch);
        per_texel(d, w.values(), [&](size_t e) {
            w.prev[e] = w.v[e];
            w.cg_x[e] = 0.0;
        });
    }
    return since_ms(start);
}

// ---------------------------------------------------------------------------------------------------------------------
// The dense direct twin behind --check, copied (plan section 1.8)
// ---------------------------------------------------------------------------------------------------------------------
//
// The same H and gradient built by a plain double loop over every site, SCATTERED into a dense matrix, and solved by
// Gauss-Jordan: it re-derives its own tap indexing on purpose, so it shares nothing with the assembly above but the
// table of subtexel offsets. The CUDA original downloads level 0 and the source first; here they are the plane's own
// vectors, read in place, so those two downloads are simply gone.
bool solve_level1_dense_host(CpuModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns)
{
    const CpuPlane& p = d->planes[(size_t)plane];
    const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1, c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;
    const int nunk = w1 * h1 * c1;
    if (nunk > max_unknowns)
        return false;

    const std::vector<float>& v0 = p.v0;
    const std::vector<float>& src = p.src;

    std::vector<double> hm((size_t)nunk * nunk, 0.0), gv((size_t)nunk, 0.0);

    for (int py = 0; py < h0; py++)
        for (int px = 0; px < w0; px++)
            for (int j = 0; j <= k; j++)
            {
                double dx = 0.0, dy = 0.0;
                if (j > 0)
                {
                    float fx = 0.0f, fy = 0.0f;
                    subtexel_offset(k, j - 1, px, py, fx, fy);
                    dx = (double)fx;
                    dy = (double)fy;
                }
                const double u = ((double)px + 0.5 + dx) / (double)w0;
                const double v = ((double)py + 0.5 + dy) / (double)h0;

                // Level 1's four taps at this position, merged by texel index.
                const double gx = u * (double)w1 - 0.5, gy = v * (double)h1 - 0.5;
                const int ix = (int)std::floor(gx), iy = (int)std::floor(gy);
                const double ax = gx - (double)ix, ay = gy - (double)iy;
                const int xs[2] = { ix < 0 ? 0 : (ix > w1 - 1 ? w1 - 1 : ix),
                                    ix + 1 < 0 ? 0 : (ix + 1 > w1 - 1 ? w1 - 1 : ix + 1) };
                const int ys[2] = { iy < 0 ? 0 : (iy > h1 - 1 ? h1 - 1 : iy),
                                    iy + 1 < 0 ? 0 : (iy + 1 > h1 - 1 ? h1 - 1 : iy + 1) };
                const double wx[2] = { 1.0 - ax, ax }, wy[2] = { 1.0 - ay, ay };
                int tid[4];
                double tws[4];
                int taps = 0;
                for (int b = 0; b < 2; b++)
                    for (int a = 0; a < 2; a++)
                    {
                        const int id = ys[b] * w1 + xs[a];
                        const double weight = wx[a] * wy[b];
                        int found = -1;
                        for (int e = 0; e < taps; e++)
                            if (tid[e] == id)
                                found = e;
                        if (found >= 0)
                            tws[found] += weight;
                        else
                        {
                            tid[taps] = id;
                            tws[taps] = weight;
                            taps++;
                        }
                    }

                // The level-0 sample, the current level-1 sample and the target at this position.
                double sbar[MAX_CHANNELS] = { 0, 0, 0, 0 };
                double target[MAX_NOUT];
                double cbar[MAX_CHANNELS] = { 0, 0, 0, 0 };
                for (int e = 0; e < taps; e++)
                    for (int q = 0; q < c1; q++)
                        cbar[q] += tws[e] * (double)c_prev[(size_t)tid[e] * c1 + q];
                if (j == 0)
                {
                    const size_t pi = (size_t)py * w0 + px;
                    for (int i = 0; i < c0; i++)
                        sbar[i] = (double)v0[pi * c0 + i];
                    for (int r = 0; r < nout; r++)
                        target[r] = (double)src[pi * nout + r];
                }
                else
                {
                    const double bx = u * (double)w0 - 0.5, by = v * (double)h0 - 0.5;
                    const int jx = (int)std::floor(bx), jy = (int)std::floor(by);
                    const double bfx = bx - (double)jx, bfy = by - (double)jy;
                    const int qx[2] = { jx < 0 ? 0 : (jx > w0 - 1 ? w0 - 1 : jx),
                                        jx + 1 < 0 ? 0 : (jx + 1 > w0 - 1 ? w0 - 1 : jx + 1) };
                    const int qy[2] = { jy < 0 ? 0 : (jy > h0 - 1 ? h0 - 1 : jy),
                                        jy + 1 < 0 ? 0 : (jy + 1 > h0 - 1 ? h0 - 1 : jy + 1) };
                    const double bwx[2] = { 1.0 - bfx, bfx }, bwy[2] = { 1.0 - bfy, bfy };
                    for (int r = 0; r < nout; r++)
                        target[r] = 0.0;
                    for (int b = 0; b < 2; b++)
                        for (int a = 0; a < 2; a++)
                        {
                            const size_t pi = (size_t)qy[b] * w0 + qx[a];
                            const double weight = bwx[a] * bwy[b];
                            for (int i = 0; i < c0; i++)
                                sbar[i] += weight * (double)v0[pi * c0 + i];
                            for (int r = 0; r < nout; r++)
                                target[r] += weight * (double)src[pi * nout + r];
                        }
                }

                // G and the residual of the current plane.
                std::vector<double> gmat((size_t)nout * c1);
                std::vector<double> res((size_t)nout);
                for (int r = 0; r < nout; r++)
                {
                    const float* row = m.dec.w.data() + (size_t)r * nin;
                    double out = (double)m.dec.b[(size_t)r];
                    for (int i = 0; i < c0; i++)
                        out += (double)row[c1 + i] * sbar[i];
                    for (int q = 0; q < c1; q++)
                    {
                        double gg = (double)row[q];
                        for (int i = 0; i < c0; i++)
                            gg += sbar[i] * (double)row[c1 + c0 + i * c1 + q];
                        gmat[(size_t)r * c1 + q] = gg;
                        out += gg * cbar[q];
                    }
                    res[(size_t)r] = out - target[r];
                }

                for (int e = 0; e < taps; e++)
                {
                    for (int a = 0; a < c1; a++)
                    {
                        double ga = 0.0;
                        for (int r = 0; r < nout; r++)
                            ga += (double)m.cw[(size_t)r] * gmat[(size_t)r * c1 + a] * res[(size_t)r];
                        gv[(size_t)tid[e] * c1 + a] += tws[e] * ga;
                    }
                    for (int f = 0; f < taps; f++)
                        for (int a = 0; a < c1; a++)
                            for (int b = 0; b < c1; b++)
                            {
                                double s = 0.0;
                                for (int r = 0; r < nout; r++)
                                    s += (double)m.cw[(size_t)r] * gmat[(size_t)r * c1 + a] *
                                         gmat[(size_t)r * c1 + b];
                                hm[((size_t)tid[e] * c1 + a) * nunk + (size_t)tid[f] * c1 + b] +=
                                    tws[e] * tws[f] * s;
                            }
                }
            }

    // (H + lambda I) delta = -grad, by Gauss-Jordan with partial pivoting.
    const int cols = nunk + 1;
    std::vector<double> aug((size_t)nunk * cols, 0.0);
    for (int i = 0; i < nunk; i++)
    {
        for (int j = 0; j < nunk; j++)
            aug[(size_t)i * cols + j] = hm[(size_t)i * nunk + j];
        aug[(size_t)i * cols + i] += lambda;
        aug[(size_t)i * cols + nunk] = -gv[(size_t)i];
    }
    for (int kk = 0; kk < nunk; kk++)
    {
        int pivot = kk;
        double best = std::fabs(aug[(size_t)kk * cols + kk]);
        for (int i = kk + 1; i < nunk; i++)
            if (std::fabs(aug[(size_t)i * cols + kk]) > best)
            {
                best = std::fabs(aug[(size_t)i * cols + kk]);
                pivot = i;
            }
        if (!(best > 0.0))
            return false;
        if (pivot != kk)
            for (int j = 0; j < cols; j++)
                std::swap(aug[(size_t)kk * cols + j], aug[(size_t)pivot * cols + j]);
        const double s = 1.0 / aug[(size_t)kk * cols + kk];
        for (int j = 0; j < cols; j++)
            aug[(size_t)kk * cols + j] *= s;
        for (int i = 0; i < nunk; i++)
        {
            if (i == kk)
                continue;
            const double f = aug[(size_t)i * cols + kk];
            if (f == 0.0)
                continue;
            for (int j = 0; j < cols; j++)
                aug[(size_t)i * cols + j] -= f * aug[(size_t)kk * cols + j];
        }
    }
    delta.assign((size_t)nunk, 0.0);
    for (int i = 0; i < nunk; i++)
        delta[(size_t)i] = aug[(size_t)i * cols + nunk];
    return true;
}

}   // namespace nntc_cpu
