// solve_level1.cu: block (b), level 1 as one sparse weighted least squares per plane - and, under --l0 bc8, block (c'),
// level 0 as the same thing over its own plane.
//
// THE TWO BLOCKS THIS FILE CARRIES. Block (b) holds level 0 and the decoder and solves for level 1; block (c') holds
// level 1 and the decoder and solves for level 0 as a CONTINUOUS 8-bit plane instead of searching a palette. They are
// the same problem with the roles of the two latents exchanged, so everything below the assembly - the ridge, the
// block-Jacobi preconditioner, the conjugate gradients, the movement probe and the four-colour sweeps with the exact
// per-channel grid argmin - is ONE body of code reading one description of the problem (PlaneWork, device.cuh). Only
// the assembly differs, because only the assembly knows which latent is the unknown, and there are two of them below:
// k_stencil_assemble for level 1 and k_stencil_assemble0 for level 0, each opening with its own derivation.
//
// The level-1 derivation follows; the level-0 one is at k_stencil_assemble0.
//
// Hold level 0 and the decoder fixed and ask for the level-1 plane that minimises the objective. The decoder is affine
// in its features and its features are affine in the level-1 sample, so with level 0 fixed the objective is an exact
// quadratic in the plane's values, and the plane's exact minimiser is the solution of one linear system.
//
// The decoder's columns follow the shader's feature order [a: c_j][b: s_i][sc: s_i c_j], so W splits into
//
//     A     = the a-block,   nout x C1        (column j is d out / d c_j at s = 0)
//     B     = the b-block,   nout x C0
//     M_i   = the sc-rows of selector channel i, nout x C1
//
// and at a site whose level-0 sample is sbar and whose level-1 taps are the texels t with weights w_t,
//
//     out   = m(sbar) + G(sbar) . sum_t w_t c_t ,   m(sbar) = b + B sbar ,   G(sbar) = A + sum_i sbar_i M_i .
//
// The tap weights are accumulated BY TEXEL INDEX, so an edge site whose two clamped taps are the same texel gives that
// texel the sum of both weights - the same rule the hardware's clamp-to-edge addressing produces, and the rule the
// objective is measured under.
//
// Writing the plane as one unknown vector of all its texels' C1 values, the weighted normal equations are
//
//     H[t][t'] += omega w_t w_t' G^T diag(cw) G        (a C1 x C1 block per texel pair)
//     grad[t]  += omega w_t      G^T diag(cw) r ,      r = out(c_prev) - target
//
// with omega = 1 for every site of the plane being solved: the mip weights govern only how much of the shared decoder
// each plane gets (block (a)), and a plane's own level-1 solve is independent of every other plane's.
//
// The step is PROXIMAL: the unknown is the correction d from the current plane, minimising E(c_prev + d) + lambda |d|^2,
// so grad is evaluated at c_prev and the system is
//
//     (H + lambda I) d = -grad ,   c = c_prev + d ,   lambda = ridge * mean(diag H) .
//
// The ridge is not cosmetic. With one texture (nout = 3) and C1 = 4 every site's G^T diag(cw) G has rank at most 3 in a
// four-dimensional space, so a texel whose footprint is flat in sbar has a genuine null direction; without the ridge the
// solve wanders along it and blows up the channel's range, which costs every other texel a quantisation step. The
// smallest and largest diagonal entry of H are reported for exactly that reason: a degenerate layout shows up there.
//
// THE STENCIL. A site reads four level-1 texels, so two texels couple only if they are within one texel of each other
// in both axes: a 9-point block stencil. H is symmetric with H[t][t'] = H[t'][t]^T, so each texel stores five blocks -
// its own diagonal and the blocks to the right, down-left, down and down-right - and the other four directions are read
// as the transpose of the neighbour's block. That is 5 C1^2 numbers per texel, accumulated and kept in double: 160
// bytes at C1 = 2, 640 at C1 = 4. The extra word matters where the layout is degenerate, because the error a solve can
// carry is its residual times the condition number, and rounding H itself to floats is an error of that same kind - it
// would put the answer four or five digits away from the direct solution of the same system rather than at the residual
// the iteration reports.
//
// The assembly is a GATHER: one thread per level-1 texel walks every site whose bilinear footprint can touch it (the
// pixels of its own footprint plus a ring, and for each pixel the centre site and the K fractional sites), recomputes
// sbar, G and the tap weights for that site and accumulates into its OWN blocks. No atomics, no contention, and every
// texel's blocks are finished when the kernel returns. A site is recomputed by each of the texels it touches, which is
// the price of not needing atomics.
//
// THE SOLVER. Conjugate gradients preconditioned by the block Jacobi of the stencil: each texel's diagonal block plus
// lambda I is inverted once in double and stored as floats, and the iteration needs nothing but the stencil mat-vec,
// two vector updates and three dot products, each a deterministic two-stage reduction in double. The stencil is a hat
// function Gram matrix rather than a Laplacian, so its condition number does not grow with the resolution and the
// iteration count stays in the tens. It stops at a relative residual |(H + lambda I) d + grad| / |grad| below 1e-10, or
// at 200 iterations; both numbers are reported.
//
// The stop is far below the accuracy asked of the correction itself because a small residual is not by itself a small
// error: the error is bounded by the residual times the condition number, and where the ridge is the only thing
// bounding a direction that number is the largest diagonal of H over lambda, of the order of 1e5. A stop at 1e-6 would
// leave a correction only five digits from the direct solution of the same system - the same order as the tolerance the
// correction is held to - and the margin would come and go with the last bits of the decoder. The extra iterations are
// a handful and the iteration is not where the time goes.
//
// The iteration's five vectors are double for the same reason the stencil is: when the ridge is the only thing bounding
// a direction, the system's condition number is the ratio of the largest diagonal of H to lambda, which is easily 1e5,
// and a small residual then permits an error that large times the residual. The preconditioner alone stays float, since
// it changes only how fast the iteration converges and not what it converges to.

#include <algorithm>
#include <cmath>
#include <vector>

#include "device.cuh"
#include "sample.cuh"

static const int ASM_THREADS = 128;
static const int CG_THREADS = 256;

// How often the host looks at the conjugate gradients' residual. Every inner product the iteration itself needs is
// computed and consumed on the device, so an iteration costs no synchronisation at all; the stop, which is a host
// decision, needs one. Looking every fourth iteration therefore replaces three synchronisations per iteration with a
// quarter of one, at the price of overshooting the stop by at most three iterations - a few tenths of a millisecond on
// the base plane, against the milliseconds a synchronisation costs on every plane of every round.
static const int CG_PROBE = 4;

// The assembly's dependence on a site is a quadratic form in that site's level-0 sample (see the assembly below), so
// the stencil is carried by one fixed C1 x C1 matrix per monomial of that form: the constant, the C0 linear terms and
// the C0 (C0 + 1) / 2 quadratic ones.
int stencil_monomials(int c0)
{
    return 1 + c0 + c0 * (c0 + 1) / 2;
}

// ---------------------------------------------------------------------------------------------------------------
// The assembly
// ---------------------------------------------------------------------------------------------------------------
//
// A site contributes w_t w_t' G^T diag(cw) G to the block of every pair of taps it touches, so a texel gathers over the
// sites of its own footprint. Each site is touched by four texels and is therefore visited four times over the plane,
// and what is recomputed is G^T diag(cw) G - a C1 x C1 matrix, nout multiply-adds per entry, in double.
//
// It does not have to be. G is AFFINE in the site's level-0 sample, so G^T diag(cw) G is a QUADRATIC FORM in it:
//
//     G(s)    = A + sum_i s_i M_i
//     G^T C G = P_0 + sum_i s_i P_i + sum_{i <= j} s_i s_j P_ij
//
//     P_0  = A^T C A ,   P_i = M_i^T C A + A^T C M_i ,   P_ii = M_i^T C M_i ,   P_ij = M_i^T C M_j + M_j^T C M_i
//
// that is 1 + C0 + C0 (C0 + 1) / 2 fixed C1 x C1 matrices - six at C0 = 2, fifteen at the channel cap of 4 - which
// depend on the decoder alone and are built once per round on the host. What varies from site to site is then that many
// SCALARS, the monomials of the sample, so the whole gather reduces to accumulating one weighted sum per monomial per
// stencil slot and expanding them against the matrices once, at the end, per texel. The double arithmetic per site per
// touching texel falls from one C1 x C1 matrix product plus its scatter to a handful of multiply-adds per slot, and the
// thread's accumulator falls from five C1 x C1 blocks to five rows of monomials - which is the difference between
// spilling to memory and staying in registers.
//
// The gradient has no such structure: its residual is not affine in anything the decoder fixes, so G^T diag(cw) r is
// still formed per site. It is C1 rows against nout, not C1 x C1, so it was never the cost.

// 1 + C0 + C0 (C0 + 1) / 2 at the level-0 channel cap of MAX_CHANNELS. The accumulator is sized for the cap and only
// the first stencil_monomials(C0) entries of each slot's row are touched, so a layout narrower than the cap costs the
// registers it does not use and nothing else.
static const int MAX_MONOMIALS = 15;
static_assert(MAX_MONOMIALS == 1 + MAX_CHANNELS + MAX_CHANNELS * (MAX_CHANNELS + 1) / 2,
              "MAX_MONOMIALS must hold the whole quadratic form at the level-0 channel cap");

// The monomials of the form, in the one order the host's matrices and the kernel's sums both use: the constant, then
// s_i, then s_i s_j for i <= j.
__host__ __device__ inline void stencil_monomial_values(const float* s, int c0, double* mono)
{
    int n = 0;
    mono[n++] = 1.0;
    for (int i = 0; i < c0; i++)
        mono[n++] = (double)s[i];
    for (int i = 0; i < c0; i++)
        for (int j = i; j < c0; j++)
            mono[n++] = (double)s[i] * (double)s[j];
}

// The matrices of a form X(y) = X0 + sum_i y_i Y_i, given as `dim`-column nout x dim matrices, weighted by cw:
//
//     X^T C X = P_0 + sum_i y_i P_i + sum_{i <= j} y_i y_j P_ij
//     P_0 = X0^T C X0 ,  P_i = Y_i^T C X0 + X0^T C Y_i ,  P_ii = Y_i^T C Y_i ,  P_ij = Y_i^T C Y_j + Y_j^T C Y_i
//
// in the monomial order stencil_monomial_values writes. Written once because the two assemblies differ only in what
// stands in for X0, Y and y: block (b) takes X0 = A, Y_i = M_i and y = the level-0 sample, giving C1 x C1 matrices;
// block (c') takes X0 = B, Y_q = N_q and y = the level-1 sample, giving C0 x C0 ones. Both are built on the host
// whenever the decoder changes, which is once a round.
static void build_form_matrices(const Model& m, const std::vector<double>& x0, const std::vector<double>& y, int terms,
                                int dim, std::vector<double>& p)
{
    const int nout = m.nout;
    const int nmono = 1 + terms + terms * (terms + 1) / 2;
    p.assign((size_t)nmono * dim * dim, 0.0);

    // X^T C Y for two nout x dim matrices, accumulated into one nmono slot.
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

// Block (b)'s matrices, from the decoder's a-block A (columns 0..C1-1 of each row of W) and its sc-blocks M_i
// (columns C1 + C0 + i C1 .. + C1 - 1): G(s) = A + sum_i s_i M_i, so nmono = stencil_monomials(C0) C1 x C1 matrices.
static void build_monomial_matrices(const Model& m, std::vector<double>& p)
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

// Block (c')'s matrices, from the decoder's b-block B (columns C1..C1+C0-1 of each row of W) and the sc-columns of one
// level-1 channel N_q (column i of N_q is W[:, C1 + C0 + i C1 + q]): Q(c) = B + sum_q c_q N_q, so this is
// stencil_monomials(C1) matrices of C0 x C0. The same algebra as above with the two latents exchanged, which is why it
// is the same builder and not a second copy of it.
static void build_monomial_matrices_level0(const Model& m, std::vector<double>& p)
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

// The pixels of the plane whose bilinear level-1 footprint can reach texel index ti along one axis. A site at pixel p
// with subtexel offset dx addresses x = (p + 0.5 + dx) * n1 / n0 - 0.5, and its two clamped taps are floor(x) and
// floor(x) + 1 pulled into the plane; since x never leaves [-0.5, n1 - 0.5], a tap lands on ti exactly when floor(x) is
// ti - 1 or ti, that is when x is in [ti - 1, ti + 1). Writing r = n0 / n1 and using |dx| <= 1/2 on both sides, that is
//
//     (ti - 0.5) r - 1 <= p < (ti + 1.5) r ,
//
// and the clamp only ever pulls a tap further inside, so a pixel outside that band cannot reach ti from either end.
__host__ __device__ inline void footprint_range(int ti, int n0, int n1, int& lo, int& hi)
{
    const double r = (double)n0 / (double)n1;
    lo = (int)ceil(((double)ti - 0.5) * r - 1.0);
    hi = (int)ceil(((double)ti + 1.5) * r) - 1;
    lo = lo < 0 ? 0 : lo;
    hi = hi > n0 - 1 ? n0 - 1 : hi;
}

__global__ void k_stencil_assemble(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1,
                                   int c0, int c1, int nin, int nout, const float* weights, const float* bias,
                                   const float* cw, const double* mono_p, int nmono, int k_count, double* stencil,
                                   double* grad)
{
    const size_t ti = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= (size_t)w1 * h1)
        return;
    const int tx = (int)(ti % (size_t)w1), ty = (int)(ti / (size_t)w1);

    // Five stencil slots, each a weighted sum of the monomials rather than of whole matrices.
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

    // The five blocks, expanded once: block = sum over monomials of its weighted sum times that monomial's matrix.
    double* out_h = stencil + ti * (size_t)(STENCIL_BLOCKS * c1 * c1);
    for (int slot = 0; slot < STENCIL_BLOCKS; slot++)
    {
        double* blk = out_h + (size_t)slot * c1 * c1;
        for (int i = 0; i < c1 * c1; i++)
            blk[i] = 0.0;
        for (int i = 0; i < nmono; i++)
        {
            const double s = sums[(size_t)slot * nmono + i];
            if (s == 0.0)
                continue;
            const double* mat = mono_p + (size_t)i * c1 * c1;
            for (int e = 0; e < c1 * c1; e++)
                blk[e] += s * mat[e];
        }
    }
    for (int i = 0; i < c1; i++)
        grad[ti * (size_t)c1 + i] = g[i];
}

// ---------------------------------------------------------------------------------------------------------------
// Block (c'): the same assembly with the two latents exchanged
// ---------------------------------------------------------------------------------------------------------------
//
// Under --l0 bc8 level 0 is not an index on a fixed palette but a continuous plane of C0 channels at 8 bits, so it is
// solved exactly as level 1 is. Hold level 1 and the decoder and the objective is again an exact quadratic in the
// plane's values.
//
// THE ALGEBRA. The decoder's features are affine in the level-0 sample for a fixed level-1 one, so at a site whose
// level-1 sample is c and whose level-0 taps are the texels t with weights w_t,
//
//     out = n(c) + Q(c) . sum_t w_t x_t ,   n(c) = b + A c ,   Q(c) = B + sum_q c_q N_q      (nout x C0)
//
// where A is the a-block of W, B the b-block and N_q the sc-columns of level-1 channel q - that is, column i of N_q is
// W[:, C1 + C0 + i C1 + q]. This is exactly the G of block (b) with the two latents swapped: there the output was
// affine in the level-1 sample with a coefficient matrix quadratic in the level-0 one, here it is affine in the
// level-0 sample with a coefficient matrix quadratic in the level-1 one. So the weighted normal equations are the same
// shape,
//
//     H[t][t'] += w_t w_t' Q^T diag(cw) Q        (a C0 x C0 block per texel pair)
//     grad[t]  += w_t      Q^T diag(cw) r ,      r = out(x_prev) - target
//
// the step is the same proximal one, (H + lambda I) d = -grad with lambda = --ridge * mean(diag H), and Q^T diag(cw) Q
// is again a quadratic form - in the level-1 sample this time - so the gather accumulates stencil_monomials(C1)
// scalars per slot and expands them against the fixed C0 x C0 matrices once per texel.
//
// THE SITES OF ONE LEVEL-0 TEXEL. Level 0 is at the decoded extent, so a texel IS a pixel and the site set is the one
// block (c)'s palette search already enumerates (solve_level0.cu):
//
//   - its own CENTRE site, where level 0 is read NEAREST: a single tap, this texel, weight 1. That is the one place
//     this assembly differs from block (b)'s, where the unknown is read bilinearly at the centre too;
//   - the K fractional sites of each pixel of the 3x3 neighbourhood, where level 0 is read bilinearly at the SAME
//     resolution, so the two taps either side of the position in each axis are this texel and its neighbours, and a
//     tap the clamp pulled onto this texel adds its weight to it - the hardware's rule.
//
// A texel therefore couples only to texels within one in both axes: a 9-point stencil again, five owned blocks again,
// and the four colours of (tx & 1, ty & 1) are exactly independent again. footprint_range(ti, n, n) gives
// [ti - 1, ti + 1] by its own arithmetic at a ratio of 1, so the same range function states the neighbourhood.

__global__ void k_stencil_assemble0(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1,
                                    int c0, int c1, int nin, int nout, const float* weights, const float* bias,
                                    const float* cw, const double* mono_p, int nmono, int k_count, double* stencil,
                                    double* grad)
{
    const size_t ti = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= (size_t)w0 * h0)
        return;
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

                // The level-0 taps of this site, merged by texel index, and the site's target. The centre site reads
                // level 0 nearest, so its one tap is the pixel itself; a fractional site reads it bilinearly.
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

    double* out_h = stencil + ti * (size_t)(STENCIL_BLOCKS * c0 * c0);
    for (int slot = 0; slot < STENCIL_BLOCKS; slot++)
    {
        double* blk = out_h + (size_t)slot * c0 * c0;
        for (int i = 0; i < c0 * c0; i++)
            blk[i] = 0.0;
        for (int i = 0; i < nmono; i++)
        {
            const double s = sums[(size_t)slot * nmono + i];
            if (s == 0.0)
                continue;
            const double* mat = mono_p + (size_t)i * c0 * c0;
            for (int e = 0; e < c0 * c0; e++)
                blk[e] += s * mat[e];
        }
    }
    for (int i = 0; i < c0; i++)
        grad[ti * (size_t)c0 + i] = g[i];
}

// ---------------------------------------------------------------------------------------------------------------
// The reductions
// ---------------------------------------------------------------------------------------------------------------
//
// Every reduction is two kernels: a fixed grid of REDUCE_BLOCKS blocks writes one partial each into a row of the
// plane's partial buffer, and one single-threaded kernel walks those partials in index order and writes the answer
// into the plane's scalar row. The result therefore does not depend on how the blocks happened to be scheduled, and -
// this is the point - it never leaves the device. The kernel that needs it reads it where it lies.

__global__ void k_dot_partial(const double* a, const double* b, size_t n, double* partial)
{
    __shared__ double sh[CG_THREADS];
    double acc = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        acc += a[i] * b[i];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int half = CG_THREADS / 2; half > 0; half >>= 1)
    {
        if ((int)threadIdx.x < half)
            sh[threadIdx.x] += sh[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[(size_t)ROW_GENERAL * REDUCE_BLOCKS + blockIdx.x] = sh[0];
}

// The second stage of a plain sum: `rows` consecutive rows of partials, each into its own scalar.
__global__ void k_reduce_sums(const double* partial, int first_row, int rows, double* scalars, int first_slot)
{
    for (int r = 0; r < rows; r++)
    {
        const double* row = partial + (size_t)(first_row + r) * REDUCE_BLOCKS;
        double s = 0.0;
        for (int i = 0; i < REDUCE_BLOCKS; i++)
            s += row[i];
        scalars[first_slot + r] = s;
    }
}

// The diagonal entries of the diagonal blocks: their sum, smallest and largest, which is what the ridge and the
// degeneracy report are built from.
__global__ void k_diag_partial(const double* stencil, size_t texels, int c1, double* partial)
{
    __shared__ double ss[CG_THREADS], smin[CG_THREADS], smax[CG_THREADS];
    double sum = 0.0, lo = 1e300, hi = -1e300;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < texels; t += (size_t)gridDim.x * blockDim.x)
    {
        const double* d = stencil + t * (size_t)(STENCIL_BLOCKS * c1 * c1);
        for (int a = 0; a < c1; a++)
        {
            const double x = d[a * c1 + a];
            sum += x;
            lo = x < lo ? x : lo;
            hi = x > hi ? x : hi;
        }
    }
    ss[threadIdx.x] = sum;
    smin[threadIdx.x] = lo;
    smax[threadIdx.x] = hi;
    __syncthreads();
    for (int half = CG_THREADS / 2; half > 0; half >>= 1)
    {
        if ((int)threadIdx.x < half)
        {
            ss[threadIdx.x] += ss[threadIdx.x + half];
            smin[threadIdx.x] = smin[threadIdx.x] < smin[threadIdx.x + half] ? smin[threadIdx.x]
                                                                            : smin[threadIdx.x + half];
            smax[threadIdx.x] = smax[threadIdx.x] > smax[threadIdx.x + half] ? smax[threadIdx.x]
                                                                            : smax[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partial[(size_t)ROW_GENERAL * REDUCE_BLOCKS + blockIdx.x] = ss[0];
        partial[(size_t)(ROW_GENERAL + 1) * REDUCE_BLOCKS + blockIdx.x] = smin[0];
        partial[(size_t)(ROW_GENERAL + 2) * REDUCE_BLOCKS + blockIdx.x] = smax[0];
    }
}

// The diagonal's three figures and, from its mean, the proximal ridge lambda = --ridge * mean(diag H). lambda stays on
// the device: the preconditioner, the mat-vec and the quantised sweeps all read it from here, and the host sees it only
// in the copy of the scalar row it takes once per block.
__global__ void k_reduce_diag(const double* partial, size_t unknowns, double ridge, double* scalars)
{
    double sum = 0.0, lo = 1e300, hi = -1e300;
    for (int i = 0; i < REDUCE_BLOCKS; i++)
    {
        sum += partial[(size_t)ROW_GENERAL * REDUCE_BLOCKS + i];
        const double a = partial[(size_t)(ROW_GENERAL + 1) * REDUCE_BLOCKS + i];
        const double b = partial[(size_t)(ROW_GENERAL + 2) * REDUCE_BLOCKS + i];
        lo = a < lo ? a : lo;
        hi = b > hi ? b : hi;
    }
    scalars[SC_DIAG_SUM] = sum;
    scalars[SC_DIAG_MIN] = lo;
    scalars[SC_DIAG_MAX] = hi;
    scalars[SC_LAMBDA] = unknowns > 0 ? ridge * sum / (double)unknowns : 0.0;
}

// ---------------------------------------------------------------------------------------------------------------
// The preconditioner and the mat-vec
// ---------------------------------------------------------------------------------------------------------------

// Gauss-Jordan with partial pivoting on [a | I] for n <= MAX_CHANNELS, in double.
__device__ inline bool invert_block(double* a, int n, double* inv)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            inv[i * n + j] = i == j ? 1.0 : 0.0;
    for (int k = 0; k < n; k++)
    {
        int pivot = k;
        double best = fabs(a[k * n + k]);
        for (int i = k + 1; i < n; i++)
            if (fabs(a[i * n + k]) > best)
            {
                best = fabs(a[i * n + k]);
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

// The block Jacobi preconditioner: (H[t][t] + lambda I)^-1 per texel, inverted in double, kept as floats. A texel whose
// block is singular even with the ridge (which the ridge makes impossible for a positive semi-definite block) falls
// back to the identity, so the iteration degrades rather than producing nonsense.
__global__ void k_precondition(const double* stencil, size_t texels, int c1, const double* scalars, float* minv)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= texels)
        return;
    const double lambda = scalars[SC_LAMBDA];
    const double* d = stencil + t * (size_t)(STENCIL_BLOCKS * c1 * c1);
    double a[MAX_CHANNELS * MAX_CHANNELS], inv[MAX_CHANNELS * MAX_CHANNELS];
    for (int i = 0; i < c1; i++)
        for (int j = 0; j < c1; j++)
            a[i * c1 + j] = d[i * c1 + j] + (i == j ? lambda : 0.0);
    if (!invert_block(a, c1, inv))
        for (int i = 0; i < c1; i++)
            for (int j = 0; j < c1; j++)
                inv[i * c1 + j] = i == j ? 1.0 : 0.0;
    float* o = minv + t * (size_t)(c1 * c1);
    for (int i = 0; i < c1 * c1; i++)
        o[i] = (float)inv[i];
}

__global__ void k_apply_precond(const float* minv, const double* r, size_t texels, int c1, double* z)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= texels)
        return;
    const float* a = minv + t * (size_t)(c1 * c1);
    const double* x = r + t * (size_t)c1;
    double* o = z + t * (size_t)c1;
    for (int i = 0; i < c1; i++)
    {
        double acc = 0.0;
        for (int j = 0; j < c1; j++)
            acc += (double)a[i * c1 + j] * x[j];
        o[i] = acc;
    }
}

// y_t = (H[t][t] + lambda I) x_t + sum over the eight neighbours n of H[t][n] x_n, where the four directions the texel
// does not own are read as the transpose of the neighbour's own block in the same slot.
__global__ void k_matvec(const double* stencil, int w1, int h1, int c1, const double* scalars, const double* x,
                         double* y)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (size_t)w1 * h1)
        return;
    const int tx = (int)(t % (size_t)w1), ty = (int)(t / (size_t)w1);
    const size_t bstride = (size_t)(STENCIL_BLOCKS * c1 * c1);
    const double lambda = scalars[SC_LAMBDA];

    double acc[MAX_CHANNELS];
    const double* own = stencil + t * bstride;
    const double* xt = x + t * (size_t)c1;
    for (int i = 0; i < c1; i++)
    {
        double s = lambda * xt[i];
        for (int j = 0; j < c1; j++)
            s += own[i * c1 + j] * xt[j];
        acc[i] = s;
    }
    for (int slot = 1; slot < STENCIL_BLOCKS; slot++)
    {
        int dx, dy;
        stencil_offset(slot, dx, dy);
        const int fx = tx + dx, fy = ty + dy;
        if (fx >= 0 && fx < w1 && fy >= 0 && fy < h1)
        {
            const double* blk = own + (size_t)slot * c1 * c1;
            const double* xn = x + ((size_t)fy * w1 + fx) * (size_t)c1;
            for (int i = 0; i < c1; i++)
                for (int j = 0; j < c1; j++)
                    acc[i] += blk[i * c1 + j] * xn[j];
        }
        const int bx = tx - dx, by = ty - dy;
        if (bx >= 0 && bx < w1 && by >= 0 && by < h1)
        {
            const size_t nt = (size_t)by * w1 + bx;
            const double* blk = stencil + nt * bstride + (size_t)slot * c1 * c1;
            const double* xn = x + nt * (size_t)c1;
            for (int i = 0; i < c1; i++)
                for (int j = 0; j < c1; j++)
                    acc[i] += blk[j * c1 + i] * xn[j];
        }
    }
    double* o = y + t * (size_t)c1;
    for (int i = 0; i < c1; i++)
        o[i] = acc[i];
}

__global__ void k_negate(const double* a, size_t n, double* out)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = -a[i];
}

// y += sign * scalars[slot] * x. The coefficient is a device address rather than a value, which is what lets a whole
// iteration be issued without ever waiting to learn what it is.
__global__ void k_axpy(double* y, const double* scalars, int slot, double sign, const double* x, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        y[i] += sign * scalars[slot] * x[i];
}

// p = z + beta p
__global__ void k_xpby(double* p, const double* z, const double* scalars, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        p[i] = z[i] + scalars[SC_BETA] * p[i];
}

// The two coefficients of one conjugate-gradient iteration, from the inner products the reductions have just left in
// the scalar row. A curvature that is not positive means the direction carries no descent, so the step is zero and the
// iteration stands still until the host's next look at the residual ends it.
__global__ void k_cg_alpha(double* scalars)
{
    const double pap = scalars[SC_PAP];
    scalars[SC_ALPHA] = pap > 0.0 ? scalars[SC_RZ] / pap : 0.0;
}

__global__ void k_cg_beta(double* scalars)
{
    const double rz = scalars[SC_RZ], rz_new = scalars[SC_RZ_NEW];
    scalars[SC_BETA] = rz != 0.0 ? rz_new / rz : 0.0;
    scalars[SC_RZ] = rz_new;
    const double rhs2 = scalars[SC_RHS2], rr = scalars[SC_RR];
    scalars[SC_RESID] = rhs2 > 0.0 ? sqrt((rr > 0.0 ? rr : 0.0) / rhs2) : 0.0;
}

// The plane takes the correction: c = c_prev + delta, back in the plane's own float storage.
__global__ void k_add_correction(float* v1, const double* delta, size_t n)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        v1[i] = (float)((double)v1[i] + delta[i]);
}

// ---------------------------------------------------------------------------------------------------------------
// The movement probe
// ---------------------------------------------------------------------------------------------------------------
//
// While level 1 is continuous the loop's no-move test cannot count stored indices, because there are none yet. What it
// asks instead is whether the correction was big enough to have changed one: a value that moved by less than half a
// grid step of the plane's own range could not have. That needs the plane's per-channel range, which is a reduction,
// and then the count and the two norms, which are another - both on the device, so a continuous round costs the host
// no download of the plane.

__global__ void k_range_partial(const float* v1, size_t texels, int c1, double* partial)
{
    __shared__ double slo[CG_THREADS], shi[CG_THREADS];
    for (int c = 0; c < c1; c++)
    {
        double lo = 1e300, hi = -1e300;
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < texels; i += (size_t)gridDim.x * blockDim.x)
        {
            const double x = (double)v1[i * (size_t)c1 + c];
            lo = x < lo ? x : lo;
            hi = x > hi ? x : hi;
        }
        slo[threadIdx.x] = lo;
        shi[threadIdx.x] = hi;
        __syncthreads();
        for (int half = CG_THREADS / 2; half > 0; half >>= 1)
        {
            if ((int)threadIdx.x < half)
            {
                slo[threadIdx.x] = slo[threadIdx.x] < slo[threadIdx.x + half] ? slo[threadIdx.x]
                                                                             : slo[threadIdx.x + half];
                shi[threadIdx.x] = shi[threadIdx.x] > shi[threadIdx.x + half] ? shi[threadIdx.x]
                                                                             : shi[threadIdx.x + half];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0)
        {
            partial[(size_t)(ROW_LO + c) * REDUCE_BLOCKS + blockIdx.x] = slo[0];
            partial[(size_t)(ROW_HI + c) * REDUCE_BLOCKS + blockIdx.x] = shi[0];
        }
        __syncthreads();
    }
}

__global__ void k_reduce_range(const double* partial, int c1, double* scalars)
{
    for (int c = 0; c < c1; c++)
    {
        double lo = 1e300, hi = -1e300;
        for (int i = 0; i < REDUCE_BLOCKS; i++)
        {
            const double a = partial[(size_t)(ROW_LO + c) * REDUCE_BLOCKS + i];
            const double b = partial[(size_t)(ROW_HI + c) * REDUCE_BLOCKS + i];
            lo = a < lo ? a : lo;
            hi = b > hi ? b : hi;
        }
        scalars[SC_LO + c] = lo;
        scalars[SC_HI + c] = hi;
    }
}

// Values that moved by more than half a grid step of the plane's own range, and the square norms of the correction and
// of the plane.
__global__ void k_movement_partial(const float* v1, const double* delta, size_t texels, int c1, int levels,
                                   const double* scalars, double* partial)
{
    __shared__ double smoved[CG_THREADS], sdelta[CG_THREADS], splane[CG_THREADS];
    double moved = 0.0, d2 = 0.0, p2 = 0.0;
    const size_t n = texels * (size_t)c1;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    {
        const int c = (int)(i % (size_t)c1);
        const double step = (scalars[SC_HI + c] - scalars[SC_LO + c]) / (double)levels;
        if (fabs(delta[i]) > 0.5 * step)
            moved += 1.0;
        d2 += delta[i] * delta[i];
        p2 += (double)v1[i] * (double)v1[i];
    }
    smoved[threadIdx.x] = moved;
    sdelta[threadIdx.x] = d2;
    splane[threadIdx.x] = p2;
    __syncthreads();
    for (int half = CG_THREADS / 2; half > 0; half >>= 1)
    {
        if ((int)threadIdx.x < half)
        {
            smoved[threadIdx.x] += smoved[threadIdx.x + half];
            sdelta[threadIdx.x] += sdelta[threadIdx.x + half];
            splane[threadIdx.x] += splane[threadIdx.x + half];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        partial[(size_t)ROW_GENERAL * REDUCE_BLOCKS + blockIdx.x] = smoved[0];
        partial[(size_t)(ROW_GENERAL + 1) * REDUCE_BLOCKS + blockIdx.x] = sdelta[0];
        partial[(size_t)(ROW_GENERAL + 2) * REDUCE_BLOCKS + blockIdx.x] = splane[0];
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The quantised phase: four-colour Gauss-Seidel on the frozen grid
// ---------------------------------------------------------------------------------------------------------------
//
// Once the grid is frozen, level 1's values are not free: each one must be one of the levels + 1 points of its own
// channel's grid. A continuous solve followed by a rounding is then no longer the minimiser of anything - it is the
// minimiser of the UNCONSTRAINED problem pushed to the nearest storable point, and at 4 bits a half step is a thirtieth
// of the channel's range, which is not a rounding error. The constrained problem has its own exact block step.
//
// With level 0 and the decoder held, E is exactly quadratic in the plane, so with the correction d measured from the
// plane as the stencil saw it (c_prev), and the same proximal ridge the continuous solve uses,
//
//     f(d) = E(c_prev + d) - E(c_prev) + lambda |d|^2 = 2 g.d + d^T H d + lambda |d|^2 .
//
// In ONE value x = d[t][q] with every other value held, that is a parabola
//
//     f = a x^2 + 2 b x + const ,   a = H[t][t][q][q] + lambda ,   b = g[t][q] + (H d)[t][q] - H[t][t][q][q] x
//
// - b is the gradient component at the current d with x's own term taken back out - whose vertex is x* = -b / a. Since
// a > 0 the parabola falls towards the vertex from both sides, so its minimiser over a UNIFORM grid is simply the grid
// point nearest that vertex:
//
//     index = clamp(round((c_prev + x* - lo) / step), 0, levels) ,   step = (hi - lo) / levels .
//
// One divide and one round per value: exact, and not a search over a shortlist.
//
// THE FOUR COLOURS. The stencil couples only texels within one texel in both axes, and two texels of one
// (tx & 1, ty & 1) class are two apart, so within a colour pass no texel can see another one move: the pass is one
// exact joint step in all of its texels at once, and it may write in place. The C1 channels of ONE texel do couple,
// through the off-diagonal of its own diagonal block, so a thread walks them in turn and each channel's step uses the
// values the earlier channels have just taken - Gauss-Seidel inside the texel, exact in each coordinate.
//
// THE GRADIENT. The neighbours' share of (H d)[t] is computed once per texel per pass, because no neighbour can move
// during the pass; the texel's own block supplies the rest and is recomputed per channel as the channels move. Nothing
// is carried incrementally between passes: an incremental residual would have to be corrected by every neighbour's
// write and would drift across the sweeps, while recomputing it costs the nine C1 x C1 blocks one mat-vec row reads -
// the same reads a single CG iteration makes, and there are a handful of sweeps against tens of iterations.
//
// WHY E CANNOT RISE. f falls at every coordinate step and f(0) = 0, so f(d) <= 0 at every moment and therefore
// E(c_prev + d) <= E(c_prev) - lambda |d|^2. That is why the ridge can stay in the quantised phase. And every value the
// kernel writes is a grid value, so the plane is storable after every pass, not only at the end.

__global__ void k_quant_sweep(const double* stencil, const double* grad, const float* prev, int w1, int h1, int c1,
                              const double* scalars, const float* lo, const float* hi, int bits, int colour_x, int colour_y,
                              double* delta, float* v1, uint8_t* k1, unsigned int* moved)
{
    // One thread per texel OF THIS COLOUR: the pass's texels are the sub-grid (colour_x + 2 i, colour_y + 2 j).
    const int cw1 = (w1 - colour_x + 1) / 2, ch1 = (h1 - colour_y + 1) / 2;
    const size_t ci = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= (size_t)cw1 * ch1)
        return;
    const int tx = colour_x + 2 * (int)(ci % (size_t)cw1);
    const int ty = colour_y + 2 * (int)(ci / (size_t)cw1);
    const size_t t = (size_t)ty * w1 + tx;
    const double lambda = scalars[SC_LAMBDA];
    const size_t bstride = (size_t)(STENCIL_BLOCKS * c1 * c1);
    const double* own = stencil + t * bstride;

    // The neighbours' share of (H d)[t], fixed for the whole pass. The four directions the texel does not own are read
    // as the transpose of the neighbour's own block, exactly as the mat-vec reads them.
    double nb[MAX_CHANNELS];
    for (int i = 0; i < c1; i++)
        nb[i] = 0.0;
    for (int slot = 1; slot < STENCIL_BLOCKS; slot++)
    {
        int dx, dy;
        stencil_offset(slot, dx, dy);
        const int fx = tx + dx, fy = ty + dy;
        if (fx >= 0 && fx < w1 && fy >= 0 && fy < h1)
        {
            const double* blk = own + (size_t)slot * c1 * c1;
            const double* xn = delta + ((size_t)fy * w1 + fx) * (size_t)c1;
            for (int i = 0; i < c1; i++)
                for (int j = 0; j < c1; j++)
                    nb[i] += blk[i * c1 + j] * xn[j];
        }
        const int bx = tx - dx, by = ty - dy;
        if (bx >= 0 && bx < w1 && by >= 0 && by < h1)
        {
            const size_t nt = (size_t)by * w1 + bx;
            const double* blk = stencil + nt * bstride + (size_t)slot * c1 * c1;
            const double* xn = delta + nt * (size_t)c1;
            for (int i = 0; i < c1; i++)
                for (int j = 0; j < c1; j++)
                    nb[i] += blk[j * c1 + i] * xn[j];
        }
    }

    unsigned int changed = 0;
    for (int q = 0; q < c1; q++)
    {
        const size_t idx = t * (size_t)c1 + (size_t)q;
        const float l = lo[q], span = hi[q] - lo[q];
        const double a = own[q * c1 + q] + lambda;
        if (!(a > 0.0) || !(span > 0.0f))
            continue;
        double rest = nb[q];
        for (int j = 0; j < c1; j++)
            if (j != q)
                rest += own[q * c1 + j] * delta[t * (size_t)c1 + (size_t)j];
        const double b = grad[idx] + rest;
        const double want = (double)prev[idx] - b / a;

        // The nearest grid point to the vertex, and the value it stands for, through model.h's one definition: a value
        // written here and a value written by the snap are the same bits for the same index.
        const int kk = level1_index(l, hi[q], bits, (float)want);
        const float value = level1_value(l, hi[q], bits, kk);
        if ((int)k1[idx] != kk)
        {
            k1[idx] = (uint8_t)kk;
            changed++;
        }
        v1[idx] = value;
        delta[idx] = (double)value - (double)prev[idx];
    }
    if (changed)
        atomicAdd(moved, changed);
}

// ---------------------------------------------------------------------------------------------------------------
// The host driver
// ---------------------------------------------------------------------------------------------------------------
//
// Block (b) is issued for EVERY plane at once, each on its own stream, and waited for exactly once. The planes share no
// variable and no buffer, so the only ordering the streams need is at the two ends of the block; in between the base
// plane's work keeps the device busy while the deep planes - a few thousand texels each - cost little more than their
// launches. Nothing inside the block travels to the host: the ridge, the conjugate gradients' inner products, the
// movement probe and the sweeps' moved count are computed and consumed on the device, and the host reads one pinned
// copy of each plane's scalar row after the block's single synchronisation.

static int grid_for(size_t n, int threads)
{
    return (int)((n + threads - 1) / threads);
}

static void device_dot(const DevPlane& p, const double* a, const double* b, size_t n, int slot)
{
    k_dot_partial<<<REDUCE_BLOCKS, CG_THREADS, 0, p.stream>>>(a, b, n, p.partial);
    CUDA_CHECK(cudaGetLastError());
    k_reduce_sums<<<1, 1, 0, p.stream>>>(p.partial, ROW_GENERAL, 1, p.scalars, slot);
    CUDA_CHECK(cudaGetLastError());
}

void level1_download(DeviceModel* d, const Model& m, int plane, std::vector<float>& v)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w1 * p.h1 * m.c1;
    v.resize(n);
    CUDA_CHECK(cudaMemcpy(v.data(), p.v1, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// One value of one plane, for the finite-difference probe. Through the master stream and a fan-out like every other
// upload: an objective pass on the plane streams follows it immediately and has to see the value that was poked.
void level1_poke(DeviceModel* d, int plane, size_t index, float value)
{
    device_upload_fan_out(d, d->planes[(size_t)plane].v1 + index, &value, sizeof(float));
}

void level1_delta(DeviceModel* d, const Model& m, int plane, std::vector<double>& delta)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w1 * p.h1 * m.c1;
    delta.resize(n);
    CUDA_CHECK(cudaMemcpy(delta.data(), p.cg_x, n * sizeof(double), cudaMemcpyDeviceToHost));
}

void level1_range(DeviceModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi)
{
    std::vector<float> v;
    level1_download(d, m, plane, v);
    lo.assign((size_t)m.c1, 1e30f);
    hi.assign((size_t)m.c1, -1e30f);
    for (size_t i = 0; i < v.size(); i++)
    {
        const int c = (int)(i % (size_t)m.c1);
        lo[(size_t)c] = v[i] < lo[(size_t)c] ? v[i] : lo[(size_t)c];
        hi[(size_t)c] = v[i] > hi[(size_t)c] ? v[i] : hi[(size_t)c];
    }
}

// The assembly and the ridge of every plane, shared by the continuous solve and the quantised sweeps: the gather kernel
// for the stencil and the gradient at the plane as it stands, then the diagonal's sum, smallest and largest, which give
// the ridge and the degeneracy report. Both paths minimise the same quadratic from the same point; they differ only in
// whether the answer is allowed off the grid.
// `level0` picks which latent is the unknown: false is block (b) over level 1, true is block (c') over level 0 under
// --l0 bc8. Only the monomial matrices and the assembly kernel differ; the ridge is read off the same diagonal.
static void assemble_all(DeviceModel* d, const Model& m, int k, double ridge, bool level0)
{
    // The decoder changes once a round, so the monomial matrices are rebuilt and uploaded once a block, before any
    // plane looks at them.
    std::vector<double> mono;
    if (level0)
        build_monomial_matrices_level0(m, mono);
    else
        build_monomial_matrices(m, mono);
    // Through the master stream and then fanned out (device_upload_fan_out), so that the copy is ORDERED against every
    // plane stream that is about to read it: an async copy from a pageable host pointer blocks the HOST, but nothing
    // in it synchronises the non-blocking plane streams, which could still be running the previous round's kernels.
    device_upload_fan_out(d, level0 ? d->mono0 : d->mono, mono.data(), mono.size() * sizeof(double));
    for (DevPlane& p : d->planes)
    {
        const PlaneWork w = plane_work(d, m, p, level0);
        const size_t texels = w.texels();
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_START], p.stream));
        if (level0)
            k_stencil_assemble0<<<grid_for(texels, ASM_THREADS), ASM_THREADS, 0, p.stream>>>(
                p.v0, p.v1, p.src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout, d->weights, d->bias, d->cw,
                w.mono, w.nmono, k, w.stencil, w.grad);
        else
            k_stencil_assemble<<<grid_for(texels, ASM_THREADS), ASM_THREADS, 0, p.stream>>>(
                p.v0, p.v1, p.src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin, m.nout, d->weights, d->bias, d->cw,
                w.mono, w.nmono, k, w.stencil, w.grad);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_ASSEMBLED], p.stream));
        k_diag_partial<<<REDUCE_BLOCKS, CG_THREADS, 0, p.stream>>>(w.stencil, texels, w.c, p.partial);
        CUDA_CHECK(cudaGetLastError());
        k_reduce_diag<<<1, 1, 0, p.stream>>>(p.partial, w.values(), ridge, p.scalars);
        CUDA_CHECK(cudaGetLastError());
    }
}

// What the host reads out of one plane's scalar row once the block has been waited for.
static void read_scalars(const DeviceModel* d, size_t i, int channels, Level1Report& rep)
{
    const double* s = d->host_scalars + i * (size_t)SC_COUNT;
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

// ONE BLOCK OVER ONE LATENT. `level0` picks which: false is block (b) over level 1, true is block (c') over level 0
// under --l0 bc8. Every plane is issued at once, each on its own stream, and waited for exactly once.
static double solve_plane_all(DeviceModel* d, const Model& m, int k, double ridge, bool level0,
                              std::vector<Level1Report>& rep)
{
    const size_t np = d->planes.size();
    const int nc = level0 ? m.c0 : m.c1;
    rep.assign(np, Level1Report());
    device_block_begin(d);
    assemble_all(d, m, k, ridge, level0);

    // ---- the preconditioner and the start of the iteration, per plane
    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        const PlaneWork w = plane_work(d, m, p, level0);
        const size_t texels = w.texels(), n = w.values();
        k_precondition<<<grid_for(texels, CG_THREADS), CG_THREADS, 0, p.stream>>>(w.stencil, texels, nc, p.scalars,
                                                                                  w.precond);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_PRECOND], p.stream));

        // Preconditioned conjugate gradients on (H + lambda I) delta = -grad, started at delta = 0.
        CUDA_CHECK(cudaMemsetAsync(w.cg_x, 0, n * sizeof(double), p.stream));
        k_negate<<<grid_for(n, CG_THREADS), CG_THREADS, 0, p.stream>>>(w.grad, n, w.cg_r);
        CUDA_CHECK(cudaGetLastError());
        k_apply_precond<<<grid_for(texels, CG_THREADS), CG_THREADS, 0, p.stream>>>(w.precond, w.cg_r, texels, nc,
                                                                                   w.cg_z);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(w.cg_p, w.cg_z, n * sizeof(double), cudaMemcpyDeviceToDevice, p.stream));
        device_dot(p, w.cg_r, w.cg_r, n, SC_RHS2);
        device_dot(p, w.cg_r, w.cg_z, n, SC_RZ);
    }

    // ---- the iteration, every plane in lockstep. A plane whose residual has fallen below the tolerance is dropped at
    // the next probe and the others carry on; the probe is the only synchronisation the loop makes. A plane whose
    // right-hand side is zero has nothing to do and leaves at the first probe with a residual of zero.
    const double tolerance = 1e-10;
    std::vector<int> active;
    for (size_t i = 0; i < np; i++)
        active.push_back((int)i);
    for (int it = 0; it < 200 && !active.empty(); it++)
    {
        for (int idx : active)
        {
            DevPlane& p = d->planes[(size_t)idx];
            const PlaneWork w = plane_work(d, m, p, level0);
            const size_t texels = w.texels(), n = w.values();
            const int vblocks = grid_for(n, CG_THREADS), tblocks = grid_for(texels, CG_THREADS);
            k_matvec<<<tblocks, CG_THREADS, 0, p.stream>>>(w.stencil, w.w, w.h, nc, p.scalars, w.cg_p, w.cg_ap);
            CUDA_CHECK(cudaGetLastError());
            device_dot(p, w.cg_p, w.cg_ap, n, SC_PAP);
            k_cg_alpha<<<1, 1, 0, p.stream>>>(p.scalars);
            CUDA_CHECK(cudaGetLastError());
            k_axpy<<<vblocks, CG_THREADS, 0, p.stream>>>(w.cg_x, p.scalars, SC_ALPHA, 1.0, w.cg_p, n);
            CUDA_CHECK(cudaGetLastError());
            k_axpy<<<vblocks, CG_THREADS, 0, p.stream>>>(w.cg_r, p.scalars, SC_ALPHA, -1.0, w.cg_ap, n);
            CUDA_CHECK(cudaGetLastError());
            device_dot(p, w.cg_r, w.cg_r, n, SC_RR);
            k_apply_precond<<<tblocks, CG_THREADS, 0, p.stream>>>(w.precond, w.cg_r, texels, nc, w.cg_z);
            CUDA_CHECK(cudaGetLastError());
            device_dot(p, w.cg_r, w.cg_z, n, SC_RZ_NEW);
            k_cg_beta<<<1, 1, 0, p.stream>>>(p.scalars);
            CUDA_CHECK(cudaGetLastError());
            k_xpby<<<vblocks, CG_THREADS, 0, p.stream>>>(w.cg_p, w.cg_z, p.scalars, n);
            CUDA_CHECK(cudaGetLastError());
            rep[(size_t)idx].iterations = it + 1;
        }
        if ((it + 1) % CG_PROBE != 0)
            continue;
        for (int idx : active)
        {
            const DevPlane& p = d->planes[(size_t)idx];
            CUDA_CHECK(cudaMemcpyAsync(d->host_scalars + (size_t)idx * SC_COUNT, p.scalars,
                                       (size_t)SC_COUNT * sizeof(double), cudaMemcpyDeviceToHost, p.stream));
        }
        for (int idx : active)
            CUDA_CHECK(cudaStreamSynchronize(d->planes[(size_t)idx].stream));
        std::vector<int> still;
        for (int idx : active)
            if (d->host_scalars[(size_t)idx * SC_COUNT + SC_RESID] >= tolerance)
                still.push_back(idx);
        active.swap(still);
    }

    // ---- the plane takes the correction, and the movement probe
    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        const PlaneWork w = plane_work(d, m, p, level0);
        const size_t texels = w.texels(), n = w.values();
        k_add_correction<<<grid_for(n, CG_THREADS), CG_THREADS, 0, p.stream>>>(w.v, w.cg_x, n);
        CUDA_CHECK(cudaGetLastError());
        k_range_partial<<<REDUCE_BLOCKS, CG_THREADS, 0, p.stream>>>(w.v, texels, nc, p.partial);
        CUDA_CHECK(cudaGetLastError());
        k_reduce_range<<<1, 1, 0, p.stream>>>(p.partial, nc, p.scalars);
        CUDA_CHECK(cudaGetLastError());
        k_movement_partial<<<REDUCE_BLOCKS, CG_THREADS, 0, p.stream>>>(w.v, w.cg_x, texels, nc, w.levels, p.scalars,
                                                                       p.partial);
        CUDA_CHECK(cudaGetLastError());
        k_reduce_sums<<<1, 1, 0, p.stream>>>(p.partial, ROW_GENERAL, 3, p.scalars, SC_MOVED);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_END], p.stream));
        CUDA_CHECK(cudaMemcpyAsync(d->host_scalars + i * (size_t)SC_COUNT, p.scalars,
                                   (size_t)SC_COUNT * sizeof(double), cudaMemcpyDeviceToHost, p.stream));
    }

    const double ms = device_block_end(d);
    for (size_t i = 0; i < np; i++)
    {
        const DevPlane& p = d->planes[i];
        rep[i].quantised = false;
        rep[i].ms_assemble = device_elapsed(p, EV_B_START, EV_B_ASSEMBLED);
        rep[i].ms_precond = device_elapsed(p, EV_B_ASSEMBLED, EV_B_PRECOND);
        rep[i].ms_cg = device_elapsed(p, EV_B_PRECOND, EV_B_END);
        read_scalars(d, i, nc, rep[i]);
    }
    return ms;
}

// The same block on a frozen grid: the same assembled stencil and gradient, then `sweeps` four-colour Gauss-Seidel
// passes whose per-value step is the exact argmin over that grid. Every value written is a grid value, so the plane
// stays storable throughout.
static double sweep_plane_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps, bool level0,
                              std::vector<Level1Report>& rep)
{
    const size_t np = d->planes.size();
    const int nc = level0 ? m.c0 : m.c1;
    rep.assign(np, Level1Report());
    device_block_begin(d);
    assemble_all(d, m, k, ridge, level0);

    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        const PlaneWork w = plane_work(d, m, p, level0);
        const size_t n = w.values();

        // The correction is measured from the plane the stencil was assembled at, and starts at zero.
        CUDA_CHECK(cudaMemcpyAsync(w.prev, w.v, n * sizeof(float), cudaMemcpyDeviceToDevice, p.stream));
        CUDA_CHECK(cudaMemsetAsync(w.cg_x, 0, n * sizeof(double), p.stream));
        CUDA_CHECK(cudaMemsetAsync(p.counters, 0, sizeof(unsigned int), p.stream));
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_PRECOND], p.stream));
        for (int s = 0; s < sweeps; s++)
            for (int colour = 0; colour < 4; colour++)
            {
                const int cx = colour & 1, cy = colour >> 1;
                const size_t count = (size_t)((w.w - cx + 1) / 2) * (size_t)((w.h - cy + 1) / 2);
                if (count == 0)
                    continue;
                k_quant_sweep<<<grid_for(count, CG_THREADS), CG_THREADS, 0, p.stream>>>(
                    w.stencil, w.grad, w.prev, w.w, w.h, nc, p.scalars, w.lo, w.hi, w.bits, cx, cy, w.cg_x, w.v, w.k,
                    p.counters);
                CUDA_CHECK(cudaGetLastError());
            }
        CUDA_CHECK(cudaEventRecord(p.ev[EV_B_END], p.stream));
        CUDA_CHECK(cudaMemcpyAsync(d->host_scalars + i * (size_t)SC_COUNT, p.scalars,
                                   (size_t)SC_COUNT * sizeof(double), cudaMemcpyDeviceToHost, p.stream));
        CUDA_CHECK(cudaMemcpyAsync(d->host_counters + i * 5, p.counters, sizeof(unsigned int), cudaMemcpyDeviceToHost,
                                   p.stream));
    }

    const double ms = device_block_end(d);
    for (size_t i = 0; i < np; i++)
    {
        const DevPlane& p = d->planes[i];
        rep[i].quantised = true;
        rep[i].sweeps = sweeps;
        rep[i].ms_assemble = device_elapsed(p, EV_B_START, EV_B_ASSEMBLED);
        rep[i].ms_sweeps = device_elapsed(p, EV_B_PRECOND, EV_B_END);
        read_scalars(d, i, nc, rep[i]);
        // The scalar row is the plane's own and persists between blocks, so the fields only the CONTINUOUS path writes
        // still hold whatever the last continuous round left there. The quantised path has no iteration, no residual
        // and no correction to measure - it counts the stored indices it changed - so they are cleared rather than
        // reported as this block's.
        rep[i].iterations = 0;
        rep[i].residual = 0.0;
        rep[i].moved_values = 0;
        rep[i].delta2 = 0.0;
        rep[i].plane2 = 0.0;
        rep[i].moved = (long long)d->host_counters[i * 5];
    }
    return ms;
}

// The four public entry points, which say which latent each block solves and share everything else.
double solve_level1_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return solve_plane_all(d, m, k, ridge, false, rep);
}

double sweep_level1_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                        std::vector<Level1Report>& rep)
{
    return sweep_plane_all(d, m, k, ridge, sweeps, false, rep);
}

double solve_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return solve_plane_all(d, m, k, ridge, true, rep);
}

double sweep_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep)
{
    return sweep_plane_all(d, m, k, ridge, sweeps, true, rep);
}

// BLOCK (c')'s ASSEMBLY ALONE, for the BC refinement (refine_bc.cu). The refinement minimises the very quadratic this
// assembly builds - E in level 0's values with level 1 and the decoder held - but over one 4x4 block at a time and
// under the block format's constraint, so it needs the stencil and the gradient and nothing that follows them. The
// correction is measured from the plane as it stands here and starts at zero, exactly as the quantised sweeps measure
// theirs.
//
// The ridge is ZERO. It exists to bound a direction the data does not, in an iteration that would otherwise wander
// along it; the refinement takes only exact minimisations of the objective's own quadratic and accepts them only when
// that quadratic falls, so a proximal term would bias the acceptance against a move the objective wants.
double assemble_level0_for_refine(DeviceModel* d, const Model& m, int k)
{
    device_block_begin(d);
    assemble_all(d, m, k, 0.0, true);
    for (DevPlane& p : d->planes)
    {
        const PlaneWork w = plane_work(d, m, p, true);
        const size_t n = w.values();
        CUDA_CHECK(cudaMemcpyAsync(w.prev, w.v, n * sizeof(float), cudaMemcpyDeviceToDevice, p.stream));
        CUDA_CHECK(cudaMemsetAsync(w.cg_x, 0, n * sizeof(double), p.stream));
    }
    return device_block_end(d);
}

// The plane's grid indices as the .dds will store them.
void level1_download_indices(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k1)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w1 * p.h1 * m.c1;
    k1.resize(n);
    CUDA_CHECK(cudaMemcpy(k1.data(), p.k1, n, cudaMemcpyDeviceToHost));
}

// The plane and its indices written back, which the outer repack loop's restore needs: a rejected pass leaves the
// device holding exactly the planes the accepted one did.
// The decoder and both latents' grids written back, for the outer repack loop's restore. device.cuh's uploads are
// device code's own; these are the host-callable spellings of them, because main.cpp sees DeviceModel only as a name.
void upload_decoder_to_device(DeviceModel* d, const Model& m)
{
    device_upload_decoder(d, m);
}

void upload_grids_to_device(DeviceModel* d, const Model& m)
{
    device_upload_palettes(d, m);
}

void level1_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v1)
{
    DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w1 * p.h1 * m.c1;
    if (v1.size() != n)
        return;
    device_upload_fan_out(d, p.v1, v1.data(), n * sizeof(float));
}

void level1_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k1)
{
    DevPlane& p = d->planes[(size_t)plane];
    const size_t n = (size_t)p.w1 * p.h1 * m.c1;
    if (k1.size() != n)
        return;
    device_upload_fan_out(d, p.k1, k1.data(), n);
}

// ---------------------------------------------------------------------------------------------------------------
// The dense direct twin behind --check
// ---------------------------------------------------------------------------------------------------------------
//
// The same H and the same gradient, built on the host by a plain double loop over every site of the plane - scattering
// into a dense matrix instead of gathering into a stencil, so the loop structure, the indexing and the ownership rule
// of the kernel above are all re-derived rather than reused - and then solved exactly by Gauss-Jordan. It shares
// nothing with the device path but the table of subtexel offsets, which is the definition of the site set and not part
// of what is being checked. Only small planes can afford it: the matrix is (texels * C1)^2 doubles.

bool solve_level1_dense_host(DeviceModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns)
{
    const DevPlane& p = d->planes[(size_t)plane];
    const int w0 = p.w0, h0 = p.h0, w1 = p.w1, h1 = p.h1, c0 = m.c0, c1 = m.c1, nout = m.nout, nin = m.dec.nin;
    const int nunk = w1 * h1 * c1;
    if (nunk > max_unknowns)
        return false;

    std::vector<float> v0((size_t)w0 * h0 * c0), src((size_t)w0 * h0 * nout);
    CUDA_CHECK(cudaMemcpy(v0.data(), p.v0, v0.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(src.data(), p.src, src.size() * sizeof(float), cudaMemcpyDeviceToHost));

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
