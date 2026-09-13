// refine_bc.cu: the post-fit BC pack of level 0, refined by ANALYSIS BY SYNTHESIS.
//
// THE GAP THIS CLOSES. Under --l0 bc8 the loop leaves a continuous level-0 plane at 8 bits, and the plane the .dds holds
// is that plane packed into BC4 / BC5: two endpoints and eight values along the line between them per 4x4 block per
// channel. The pack bc_pack.h makes is the stb_dxt-style one - the block's lowest and highest value become the
// endpoints and every texel takes the nearest palette entry - and it is fitted against the LATENT's own values, which
// is not the quantity anyone cares about. Measured on eight runs (docs/RESULTS.md section 6.1) it costs 0.4 to 4.6 dB
// of the decoded image, and refitting level 1 and the decoder afterwards wins back essentially none of it.
//
// So this file does what a BC1-7 or ASTC encoder does: it chooses each block's endpoints and selectors to minimise THE
// DECODER'S OUTPUT ERROR over the sites that block's texels touch, and it takes a candidate only when that error
// measurably falls. The pack it starts from is exactly the stb_dxt-style one, so the result is never worse than what
// the seed pack shipped on its own; when nothing improves, the bytes are the same bytes.
//
// ---------------------------------------------------------------------------------------------------------------
// THE BLOCK OBJECTIVE
// ---------------------------------------------------------------------------------------------------------------
//
// With level 1 and the decoder held, the objective is an EXACT quadratic in level 0's values - that is the whole
// premise of block (c') (solve_level1.cu), whose assembly already builds it:
//
//     E(x_prev + d) = E(x_prev) + 2 g . d + d^T H d
//
// with g the gradient at the plane the stencil was assembled at and H the 9-point stencil, H[t][t'] = sum over the
// sites both texels touch of w_t w_t' Q^T diag(cw) Q. Both are exactly the sums the brief's
//
//     out(site) = p(site) + sum_t alpha_t(site) Q(site) x_t
//
// produces: the centre site of each of the block's sixteen texels, and every fractional site whose bilinear level-0
// taps reach one of them, with the neighbouring blocks' texels held fixed. Nothing has to be re-enumerated here.
//
// Restricted to one 4x4 block B, with every texel outside B held at the value it currently has, that is a quadratic in
// the block's 16 C0 values alone:
//
//     E_blk(d_B) = const + 2 A1 . d_B + d_B^T A2 d_B
//     A2[(t,i)][(t',j)] = H[t][t'][i][j]                      for t, t' in B          (dense inside the block:
//                                                                                      fractional sites couple
//                                                                                      neighbouring texels)
//     A1[(t,i)]         = g[t][i] + sum_{t' not in B} (H[t][t'] d_{t'})[i]            (the neighbours' current share)
//
// A2 is built once per block from the five stencil blocks each texel owns plus the transposes of its neighbours', which
// is the same reading of H the mat-vec and the quantised sweeps make. Everything is double, as the stencil is.
//
// ---------------------------------------------------------------------------------------------------------------
// THE BC CONSTRAINT
// ---------------------------------------------------------------------------------------------------------------
//
// Each channel of the block is one BC4 block: two 8-bit endpoints r0 > r1 (the EIGHT-value mode; the six-value mode is
// not used here and is not searched - it trades two interpolants for a hard 0 and 255, which a latent on a fitted
// lo/hi grid has no use for) and one 3-bit selector per texel, so the stored byte of texel t is
//
//     byte_t = a(sel_t) r0 + (1 - a(sel_t)) r1 ,   a = 1, 0, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7 for sel = 0..7
//
// - bc_pack.h's palette, written as the affine weight on the high endpoint - and the value the sampler returns for it
// is the plane's own dequantisation, value = lo_c + byte / 255 * (hi_c - lo_c). The palette is evaluated in float
// exactly as bc_pack.h's decoder evaluates it, so a value weighed here is the value the file decodes to.
//
// ---------------------------------------------------------------------------------------------------------------
// THE SEARCH
// ---------------------------------------------------------------------------------------------------------------
//
// Per block, per round (--bc-refine N, default 4):
//
//   (i)  THE ENDPOINTS, per channel, by least squares. byte_t is affine in (r0, r1) at fixed selectors, so with
//        p_t = s a(sel_t) and q_t = s (1 - a(sel_t)), s = (hi - lo) / 255, a move (dr0, dr1) changes the block's values
//        by d_t = p_t dr0 + q_t dr1 and
//
//            dE = 2 (b.p) dr0 + 2 (b.q) dr1 + (p A2 p) dr0^2 + 2 (p A2 q) dr0 dr1 + (q A2 q) dr1^2 ,  b = A1 + A2 d
//
//        - five scalars, from which the exact minimiser is one 2x2 solve and every candidate costs three multiplies.
//        The vertex is rounded onto the 8-bit grid and its +-1 neighbours in both endpoints are tried with it (nine
//        candidates, which is what catches the rounding), each required to keep r0 > r1.
//
//   (ii) THE SELECTORS, one texel-channel at a time, as an exact 8-way argmin: with every other value held, E_blk in
//        one value x is a parabola, so the eight palette entries are simply evaluated and the smallest taken. Two
//        passes over the block's 16 C0 selectors, Gauss-Seidel (each step sees the previous ones).
//
// A step is taken ONLY IF IT STRICTLY LOWERS E_blk, and the deciding number is always evaluated in the arithmetic the
// plane is actually written in: the endpoint search above is the ideal quadratic, and the candidate it picks is then
// weighed again against the float-rounded palette values before it is accepted. Since every accepted step lowers the
// same quadratic and the rest of the plane is held, E over the plane cannot rise.
//
// THE FOUR COLOURS. Two blocks couple when a fractional site's bilinear taps span their border, which reaches one texel
// past a block edge - so blocks that are two apart in either axis share no site. Colouring the BLOCK grid by
// (gx & 1, gy & 1) and running the four colours as four kernel launches therefore makes every block of a pass see a
// frozen neighbourhood, exactly as block (b)'s four-colour sweeps do over texels.
//
// One CUDA block per 4x4 block, 64 threads (the sixteen texels times the level-0 channel cap), A2 and the working
// vectors in shared memory: 9.5 KB at C0 2, 35 KB at C0 4.

#include <algorithm>
#include <cmath>
#include <vector>

#include "bc_pack.h"
#include "device.cuh"

// Sixteen texels times the level-0 channel cap: the unknowns of one block, and the thread count.
static const int REFINE_THREADS = 16 * MAX_CHANNELS;

// bc_pack.h's eight-value palette (r0 > r1), as the affine weight on the high endpoint: p = a r0 + (1 - a) r1.
__host__ __device__ inline double bc4_weight8(int sel)
{
    return sel == 0 ? 1.0 : (sel == 1 ? 0.0 : (double)(8 - sel) / 7.0);
}

// The same palette entry as a decoder computes it, in float and in bc_pack.h's own spelling, so that a value weighed
// here is the value the block decodes to.
__host__ __device__ inline float bc4_entry(int r0, int r1, int sel)
{
    if (sel == 0)
        return (float)r0;
    if (sel == 1)
        return (float)r1;
    return (float)((8 - sel) * r0 + (sel - 1) * r1) / 7.0f;
}

// The plane value a decoded byte stands for, in the one arithmetic export.cpp's level0_plane_values uses. The device
// spells the multiply and the add out as separately rounded double operations, exactly as model.h's level1_value does:
// nvcc would otherwise contract them into one fused multiply-add that the host has no counterpart for, and the two
// would then disagree by an ulp of a double on the value every acceptance test is weighed in.
__host__ __device__ inline float bc0_value(float lo, float hi, float byte)
{
    const double frac = (double)byte / 255.0;
    const double span = (double)hi - (double)lo;
#ifdef __CUDA_ARCH__
    return (float)__dadd_rn((double)lo, __dmul_rn(frac, span));
#else
    return (float)((double)lo + frac * span);
#endif
}

// One block-wide sum, over a fixed tree so that the answer does not depend on how the warps were scheduled. Every
// thread must call it; the leading barrier is what keeps a thread from overwriting the previous sum before its
// neighbours have read it.
__device__ inline double refine_reduce(double* red, double v, int tid)
{
    __syncthreads();
    red[tid] = v;
    __syncthreads();
    for (int s = REFINE_THREADS / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// The plane, the indices and the correction that follow from a set of endpoints and selectors. Run after the seed pack
// is uploaded and after every refinement, so the three device copies of the packed plane never disagree: v0 is what the
// objective and the next block read, delta is what the refinement measures from the plane the stencil was assembled at,
// and k0 is the nearest 8-bit index to the decoded byte - which is what an uncompressed twin or a report would show,
// the file itself holding the block.
__global__ void k_bc_decode(const uint8_t* ep, const uint8_t* sel, const float* lo0, const float* hi0, int w0, int h0,
                            int c0, int bx, const float* prev, float* v0, uint8_t* k0, double* delta)
{
    const size_t ti = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= (size_t)w0 * h0)
        return;
    const int px = (int)(ti % (size_t)w0), py = (int)(ti / (size_t)w0);
    const size_t gi = (size_t)(py / 4) * bx + (px / 4);
    const int t = (py % 4) * 4 + (px % 4);
    for (int c = 0; c < c0; c++)
    {
        const size_t b = (gi * (size_t)c0 + (size_t)c);
        const float byte = bc4_entry(ep[b * 2], ep[b * 2 + 1], sel[b * 16 + (size_t)t]);
        const float value = bc0_value(lo0[c], hi0[c], byte);
        const size_t idx = ti * (size_t)c0 + (size_t)c;
        v0[idx] = value;
        k0[idx] = (uint8_t)(byte < 0.0f ? 0 : (byte > 255.0f ? 255 : (int)(byte + 0.5f)));
        delta[idx] = (double)value - (double)prev[idx];
    }
}

// One refinement round over the blocks of one colour: the endpoints of each channel by least squares, then two
// Gauss-Seidel passes over the block's selectors, every step taken only on a strict decrease of E_blk.
__global__ void k_bc_refine(const double* stencil, const double* grad, const float* prev, const float* lo0,
                            const float* hi0, int w0, int h0, int c0, int bx, int colour_x, int colour_y, uint8_t* ep,
                            uint8_t* sel, double* delta, float* v0, uint8_t* k0, double* stats)
{
    const int cbx = (bx - colour_x + 1) / 2;
    const int gx = colour_x + 2 * (int)(blockIdx.x % (unsigned)cbx);
    const int gy = colour_y + 2 * (int)(blockIdx.x / (unsigned)cbx);
    const int tid = (int)threadIdx.x;
    const int n = 16 * c0;

    extern __shared__ double sm[];
    double* a2 = sm;              // n x n, the block's own quadratic
    double* a1 = a2 + n * n;      // n, the gradient with the neighbours' share folded in
    double* dv = a1 + n;          // n, the block's correction from the plane the stencil saw
    double* uv = dv + n;          // n, a2 . dv, kept up to date as values move
    double* red = uv + n;         // REFINE_THREADS, the reduction's scratch
    double* dlt = red + REFINE_THREADS;   // 16, one endpoint candidate's change per texel

    __shared__ int s_ep[2 * MAX_CHANNELS];
    __shared__ unsigned char s_sel[16 * MAX_CHANNELS];
    __shared__ double s_num[2];
    __shared__ int s_int[2];

    for (int i = tid; i < n * n; i += REFINE_THREADS)
        a2[i] = 0.0;
    for (int i = tid; i < n; i += REFINE_THREADS)
    {
        a1[i] = 0.0;
        dv[i] = 0.0;
        uv[i] = 0.0;
    }
    __syncthreads();

    // ---- the block's quadratic, one thread per texel. A texel outside the plane (a partial edge block) keeps the
    // zero rows it was given: it carries no site, so no choice about it can change E, and its selector therefore never
    // moves - which is right, since the file's decoder never reads it either.
    const int px = gx * 4 + (tid & 3), py = gy * 4 + (tid >> 2);
    const bool inside = tid < 16 && px < w0 && py < h0;
    const size_t ti = inside ? (size_t)py * w0 + px : 0;
    if (inside)
    {
        const size_t bstride = (size_t)(STENCIL_BLOCKS * c0 * c0);
        const double* own = stencil + ti * bstride;
        double acc[MAX_CHANNELS];
        for (int i = 0; i < c0; i++)
            acc[i] = grad[ti * (size_t)c0 + (size_t)i];
        for (int slot = 0; slot < STENCIL_BLOCKS; slot++)
        {
            int dx, dy;
            stencil_offset(slot, dx, dy);
            const int fx = px + dx, fy = py + dy;
            if (fx >= 0 && fx < w0 && fy >= 0 && fy < h0)
            {
                const double* blk = own + (size_t)slot * c0 * c0;
                const int lx = fx - gx * 4, ly = fy - gy * 4;
                if (lx >= 0 && lx < 4 && ly >= 0 && ly < 4)
                {
                    // Inside the block: the pair belongs to A2, and its transpose with it. No two threads write the
                    // same entry - the negative of a slot offset is never itself a slot offset, bar the diagonal.
                    const int ft = ly * 4 + lx;
                    for (int i = 0; i < c0; i++)
                        for (int j = 0; j < c0; j++)
                        {
                            a2[(size_t)(tid * c0 + i) * n + (ft * c0 + j)] += blk[i * c0 + j];
                            if (slot != 0)
                                a2[(size_t)(ft * c0 + j) * n + (tid * c0 + i)] += blk[i * c0 + j];
                        }
                }
                else
                {
                    const double* xn = delta + ((size_t)fy * w0 + fx) * (size_t)c0;
                    for (int i = 0; i < c0; i++)
                        for (int j = 0; j < c0; j++)
                            acc[i] += blk[i * c0 + j] * xn[j];
                }
            }
            if (slot == 0)
                continue;
            // The four directions the texel does not own, read as the transpose of the neighbour's block. A neighbour
            // inside the block is skipped: that pair is the neighbour's own forward one and is already in A2.
            const int ax = px - dx, ay = py - dy;
            const int lx = ax - gx * 4, ly = ay - gy * 4;
            if (ax < 0 || ax >= w0 || ay < 0 || ay >= h0 || (lx >= 0 && lx < 4 && ly >= 0 && ly < 4))
                continue;
            const size_t nt = (size_t)ay * w0 + ax;
            const double* blk = stencil + nt * bstride + (size_t)slot * c0 * c0;
            const double* xn = delta + nt * (size_t)c0;
            for (int i = 0; i < c0; i++)
                for (int j = 0; j < c0; j++)
                    acc[i] += blk[j * c0 + i] * xn[j];
        }
        for (int i = 0; i < c0; i++)
        {
            a1[tid * c0 + i] = acc[i];
            dv[tid * c0 + i] = (double)v0[ti * (size_t)c0 + (size_t)i] - (double)prev[ti * (size_t)c0 + (size_t)i];
        }
    }
    __syncthreads();

    if (tid < n)
    {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += a2[(size_t)tid * n + j] * dv[j];
        uv[tid] = s;
    }
    const size_t gi = (size_t)gy * bx + gx;
    if (tid < c0)
    {
        s_ep[2 * tid] = ep[(gi * (size_t)c0 + (size_t)tid) * 2];
        s_ep[2 * tid + 1] = ep[(gi * (size_t)c0 + (size_t)tid) * 2 + 1];
    }
    if (tid < n)
        s_sel[tid] = sel[gi * (size_t)c0 * 16 + (size_t)tid];
    __syncthreads();

    double taken = 0.0;   // thread 0's running total of the decreases this block accepted

    // ---- (i) the endpoints of each channel, by least squares on E_blk
    for (int c = 0; c < c0; c++)
    {
        const double step = ((double)hi0[c] - (double)lo0[c]) / 255.0;
        const int idx = tid < 16 ? tid * c0 + c : 0;
        double p = 0.0, q = 0.0, b = 0.0;
        if (tid < 16)
        {
            const double a = bc4_weight8(s_sel[c * 16 + tid]);
            p = step * a;
            q = step * (1.0 - a);
            b = a1[idx] + uv[idx];
        }
        double ap = 0.0, aq = 0.0;
        if (tid < 16)
            for (int j = 0; j < 16; j++)
            {
                const double e = a2[(size_t)idx * n + (j * c0 + c)];
                const double aj = bc4_weight8(s_sel[c * 16 + j]);
                ap += e * step * aj;
                aq += e * step * (1.0 - aj);
            }
        const double bp = refine_reduce(red, tid < 16 ? b * p : 0.0, tid);
        const double bq = refine_reduce(red, tid < 16 ? b * q : 0.0, tid);
        const double pap = refine_reduce(red, tid < 16 ? p * ap : 0.0, tid);
        const double paq = refine_reduce(red, tid < 16 ? p * aq : 0.0, tid);
        const double qaq = refine_reduce(red, tid < 16 ? q * aq : 0.0, tid);

        if (tid == 0)
        {
            // The vertex of the 2x2 problem, rounded onto the 8-bit grid, and the nine candidates around it. The
            // guard on the determinant is RELATIVE to the diagonal it came from: on a rank-1 block (every texel on
            // one selector, so p and q are parallel) the exact determinant is zero and what survives is cancellation
            // noise of order eps * pap * qaq, which an absolute det > 0 would accept and divide by. The vertex is only
            // the search's proposal either way - the nine candidates are clamped and re-evaluated exactly - so
            // refusing it there costs nothing and keeps a meaningless quotient out of the arithmetic.
            const int r0 = s_ep[2 * c], r1 = s_ep[2 * c + 1];
            const double det = pap * qaq - paq * paq;
            double d0 = 0.0, d1 = 0.0;
            if (det > 1e-12 * pap * qaq)
            {
                d0 = (paq * bq - qaq * bp) / det;
                d1 = (paq * bp - pap * bq) / det;
            }
            const double v0r = (double)r0 + d0, v1r = (double)r1 + d1;
            const int c0r = (int)floor(v0r + 0.5), c1r = (int)floor(v1r + 0.5);
            double best = 0.0;
            int best0 = r0, best1 = r1;
            for (int e0 = -1; e0 <= 1; e0++)
                for (int e1 = -1; e1 <= 1; e1++)
                {
                    const int a0 = c0r + e0 < 0 ? 0 : (c0r + e0 > 255 ? 255 : c0r + e0);
                    const int a1i = c1r + e1 < 0 ? 0 : (c1r + e1 > 255 ? 255 : c1r + e1);
                    if (a0 <= a1i)
                        continue;   // the eight-value mode, which is the only one this search uses
                    const double f0 = (double)(a0 - r0), f1 = (double)(a1i - r1);
                    const double de = 2.0 * (bp * f0 + bq * f1) + pap * f0 * f0 + 2.0 * paq * f0 * f1 + qaq * f1 * f1;
                    if (de < best)
                    {
                        best = de;
                        best0 = a0;
                        best1 = a1i;
                    }
                }
            s_int[0] = best0;
            s_int[1] = best1;
        }
        __syncthreads();

        const int new0 = s_int[0], new1 = s_int[1];
        if (new0 == s_ep[2 * c] && new1 == s_ep[2 * c + 1])
            continue;

        // The candidate weighed again in the arithmetic the plane is written in: the float palette and the float
        // dequantisation, which is what the file decodes to. The ideal quadratic above was the search; this is the
        // acceptance, and it is what makes E non-increasing.
        if (tid < 16)
        {
            const float now = bc0_value(lo0[c], hi0[c], bc4_entry(s_ep[2 * c], s_ep[2 * c + 1], s_sel[c * 16 + tid]));
            const float next = bc0_value(lo0[c], hi0[c], bc4_entry(new0, new1, s_sel[c * 16 + tid]));
            dlt[tid] = (double)next - (double)now;
        }
        __syncthreads();
        double contrib = 0.0;
        if (tid < 16)
        {
            double quad = 0.0;
            for (int j = 0; j < 16; j++)
                quad += a2[(size_t)idx * n + (j * c0 + c)] * dlt[j];
            contrib = dlt[tid] * (2.0 * (a1[idx] + uv[idx]) + quad);
        }
        const double exact = refine_reduce(red, contrib, tid);
        if (exact < 0.0)
        {
            if (tid < n)
            {
                double add = 0.0;
                for (int j = 0; j < 16; j++)
                    add += a2[(size_t)tid * n + (j * c0 + c)] * dlt[j];
                uv[tid] += add;
            }
            __syncthreads();
            if (tid == 0)
            {
                s_ep[2 * c] = new0;
                s_ep[2 * c + 1] = new1;
                for (int j = 0; j < 16; j++)
                    dv[j * c0 + c] += dlt[j];
                taken += exact;
            }
        }
        __syncthreads();
    }

    // ---- (ii) the selectors, two Gauss-Seidel passes of exact 8-way argmins
    for (int pass = 0; pass < 2; pass++)
        for (int c = 0; c < c0; c++)
            for (int t = 0; t < 16; t++)
            {
                const int i = t * c0 + c;
                if (tid == 0)
                {
                    const double b = a1[i] + uv[i], a = a2[(size_t)i * n + i];
                    const int r0 = s_ep[2 * c], r1 = s_ep[2 * c + 1];
                    const double now = (double)bc0_value(lo0[c], hi0[c], bc4_entry(r0, r1, s_sel[c * 16 + t]));
                    double best = 0.0, move = 0.0;
                    int which = s_sel[c * 16 + t];
                    for (int k = 0; k < 8; k++)
                    {
                        const double d = (double)bc0_value(lo0[c], hi0[c], bc4_entry(r0, r1, k)) - now;
                        const double de = 2.0 * b * d + a * d * d;
                        if (de < best)
                        {
                            best = de;
                            move = d;
                            which = k;
                        }
                    }
                    s_num[0] = best;
                    s_num[1] = move;
                    s_int[0] = which;
                }
                __syncthreads();
                if (s_num[0] < 0.0)
                {
                    const double move = s_num[1];
                    if (tid < n)
                        uv[tid] += a2[(size_t)tid * n + i] * move;
                    if (tid == 0)
                    {
                        s_sel[c * 16 + t] = (unsigned char)s_int[0];
                        dv[i] += move;
                        taken += s_num[0];
                    }
                }
                __syncthreads();
            }

    // ---- what the block leaves behind
    if (tid < c0)
    {
        ep[(gi * (size_t)c0 + (size_t)tid) * 2] = (uint8_t)s_ep[2 * tid];
        ep[(gi * (size_t)c0 + (size_t)tid) * 2 + 1] = (uint8_t)s_ep[2 * tid + 1];
    }
    if (tid < n)
        sel[gi * (size_t)c0 * 16 + (size_t)tid] = s_sel[tid];
    if (inside)
        for (int c = 0; c < c0; c++)
        {
            const float byte = bc4_entry(s_ep[2 * c], s_ep[2 * c + 1], s_sel[c * 16 + tid]);
            const float value = bc0_value(lo0[c], hi0[c], byte);
            const size_t idx = ti * (size_t)c0 + (size_t)c;
            v0[idx] = value;
            k0[idx] = (uint8_t)(byte < 0.0f ? 0 : (byte > 255.0f ? 255 : (int)(byte + 0.5f)));
            delta[idx] = (double)value - (double)prev[idx];
        }
    // The pass's two figures, for the log. This is the one atomic in the encoder that adds doubles, so its last digits
    // depend on the order the blocks finished in - and nothing reads it: every decision above was taken from this
    // block's own quadratic in its own shared memory, and the four colours fix which neighbours it saw, so the blocks
    // the file holds are the same blocks every time (docs/DESIGN.md section 6).
    if (tid == 0 && taken < 0.0)
    {
        atomicAdd(stats, taken);
        atomicAdd(stats + 1, 1.0);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// The host driver
// ---------------------------------------------------------------------------------------------------------------

static int refine_grid(size_t n, int threads)
{
    return (int)((n + threads - 1) / threads);
}

// The endpoints and selectors of one plane, as bc_pack.h packs them and as the .dds stores them, held per block per
// channel while the refinement works on them: ep is two bytes per block-channel and sel sixteen.
static size_t plane_blocks(const Model& m, int plane)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    return (size_t)((p.w0 + 3) / 4) * (size_t)((p.h0 + 3) / 4);
}

// The pack the refinement starts from or continues, uploaded and decoded onto the device, with the stencil of block
// (c') assembled at the plane the loop left. `seed` true takes bc_pack.h's stb_dxt-style pack of m.k0 - which is
// exactly the seed pack of main.cpp's step 1 - and false keeps the endpoints and selectors the device already holds, re-assembling
// against level 1 and the decoder as they now stand.
double bc_pack_prepare(DeviceModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels)
{
    double ms = assemble_level0_for_refine(d, m, k);
    const size_t np = m.planes.size();
    if (seed)
    {
        std::vector<uint8_t> ep, sel;
        for (size_t i = 0; i < np; i++)
        {
            level0_pack_seed(m, (int)i, ep, sel);
            DevPlane& p = d->planes[i];
            CUDA_CHECK(cudaMemcpyAsync(p.bc_ep, ep.data(), ep.size(), cudaMemcpyHostToDevice, d->master));
            CUDA_CHECK(cudaMemcpyAsync(p.bc_sel, sel.data(), sel.size(), cudaMemcpyHostToDevice, d->master));
        }
        CUDA_CHECK(cudaStreamSynchronize(d->master));
        device_fan_out(d);
    }
    device_block_begin(d);
    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        k_bc_decode<<<refine_grid(texels, 256), 256, 0, p.stream>>>(p.bc_ep, p.bc_sel, d->lo0, d->hi0, p.w0, p.h0, m.c0,
                                                                     (p.w0 + 3) / 4, p.q0_prev, p.v0, p.k0, p.cg0_x);
        CUDA_CHECK(cudaGetLastError());
    }
    ms += device_block_end(d);
    bc_blocks_from_device(d, m, pack_psnr, pack_texels);
    return ms;
}

// `passes` rounds of the refinement over every plane, four colour launches each, one report entry per pass, and the
// blocks it chose written back into the model so that the file holds exactly what was measured.
double bc_refine(DeviceModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels)
{
    rep.clear();
    if (passes <= 0)
        return 0.0;
    const size_t np = m.planes.size();
    const int n = 16 * m.c0;
    const size_t shared = ((size_t)n * n + 3 * (size_t)n + REFINE_THREADS + 16) * sizeof(double);
    double ms = 0.0;
    for (int pass = 0; pass < passes; pass++)
    {
        device_block_begin(d);
        for (size_t i = 0; i < np; i++)
        {
            DevPlane& p = d->planes[i];
            CUDA_CHECK(cudaMemsetAsync(p.bc_stats, 0, 2 * sizeof(double), p.stream));
            const int bx = (p.w0 + 3) / 4, by = (p.h0 + 3) / 4;
            for (int colour = 0; colour < 4; colour++)
            {
                const int cx = colour & 1, cy = colour >> 1;
                const size_t count = (size_t)((bx - cx + 1) / 2) * (size_t)((by - cy + 1) / 2);
                if (count == 0)
                    continue;
                k_bc_refine<<<(int)count, REFINE_THREADS, shared, p.stream>>>(
                    p.stencil0, p.grad0, p.q0_prev, d->lo0, d->hi0, p.w0, p.h0, m.c0, bx, cx, cy, p.bc_ep, p.bc_sel,
                    p.cg0_x, p.v0, p.k0, p.bc_stats);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        const double pass_ms = device_block_end(d);
        ms += pass_ms;

        BcRefineReport r;
        r.pass = pass + 1;
        r.ms = pass_ms;
        for (size_t i = 0; i < np; i++)
        {
            double stats[2] = { 0.0, 0.0 };
            CUDA_CHECK(cudaMemcpy(stats, d->planes[i].bc_stats, 2 * sizeof(double), cudaMemcpyDeviceToHost));
            r.delta_e += stats[0];
            r.improved += (long long)stats[1];
            r.blocks += (long long)plane_blocks(m, (int)i);
        }
        rep.push_back(r);
    }
    bc_blocks_from_device(d, m, pack_psnr, pack_texels);
    return ms;
}

// The endpoints and selectors the device holds, turned back into the .dds's blocks and into the model, with the pack's
// own error against the continuous plane the loop left (m.k0's exact index values, in byte units) measured on the way.
void bc_blocks_from_device(DeviceModel* d, Model& m, double& pack_psnr, size_t& pack_texels)
{
    const size_t np = m.planes.size();
    m.bc0_blocks.assign(np, std::vector<std::vector<uint8_t>>());
    double se = 0.0;
    size_t n = 0;
    std::vector<uint8_t> ep, sel;
    for (size_t i = 0; i < np; i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t blocks = plane_blocks(m, (int)i);
        ep.assign(blocks * (size_t)m.c0 * 2, 0);
        sel.assign(blocks * (size_t)m.c0 * 16, 0);
        CUDA_CHECK(cudaMemcpy(ep.data(), p.bc_ep, ep.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sel.data(), p.bc_sel, sel.size(), cudaMemcpyDeviceToHost));
        level0_pack_blocks(m, (int)i, ep, sel, m.bc0_blocks[i], se, n);
    }
    pack_texels = n;
    pack_psnr = n > 0 && se > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / (se / (double)n)) : 100.0;
}

// The refinement's device state, saved and put back, so that a rejected outer pass leaves nothing of itself behind on
// the device either. m.bc0_blocks carries the same choice in the .dds's own packing, but a later refinement continues
// from p.bc_ep / p.bc_sel and not from the block bytes, so the two have to be restored together.
void bc_state_save(const DeviceModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel)
{
    const size_t np = m.planes.size();
    ep.assign(np, std::vector<uint8_t>());
    sel.assign(np, std::vector<uint8_t>());
    for (size_t i = 0; i < np; i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t blocks = plane_blocks(m, (int)i);
        ep[i].assign(blocks * (size_t)m.c0 * 2, 0);
        sel[i].assign(blocks * (size_t)m.c0 * 16, 0);
        CUDA_CHECK(cudaMemcpy(ep[i].data(), p.bc_ep, ep[i].size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(sel[i].data(), p.bc_sel, sel[i].size(), cudaMemcpyDeviceToHost));
    }
}

void bc_state_restore(DeviceModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel)
{
    const size_t np = m.planes.size();
    if (ep.size() != np || sel.size() != np)
        return;
    for (size_t i = 0; i < np; i++)
    {
        DevPlane& p = d->planes[i];
        CUDA_CHECK(cudaMemcpyAsync(p.bc_ep, ep[i].data(), ep[i].size(), cudaMemcpyHostToDevice, d->master));
        CUDA_CHECK(cudaMemcpyAsync(p.bc_sel, sel[i].data(), sel[i].size(), cudaMemcpyHostToDevice, d->master));
    }
    CUDA_CHECK(cudaStreamSynchronize(d->master));
    device_fan_out(d);   // the plane streams are non-blocking: they must see the restored bytes before their next kernel
}
