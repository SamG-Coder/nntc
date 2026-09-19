// cpu/cpu_refine_bc.cpp: the post-fit BC pack of level 0, refined by analysis by synthesis, on the CPU
// (docs/CPU_BACKEND_PLAN.md stage C7, the port of refine_bc.cu).
//
// refine_bc.cu's header is the WHY of every line below and is not repeated: under --l0 bc8 the plane the .dds holds is
// the continuous level 0 packed into BC4 / BC5, and each 4x4 block's endpoints and selectors are chosen to minimise the
// decoder's output error over the sites the block's texels touch - the exact quadratic block (c')'s assembly builds,
// restricted to the block with every texel outside it held - by a least-squares step on each channel's endpoints (the
// vertex rounded onto the 8-bit grid and its nine neighbours tried) and two Gauss-Seidel passes of exact 8-way selector
// argmins, every step taken ONLY on a strict decrease weighed in the arithmetic the file decodes in. The blocks move in
// four colour passes over the BLOCK grid, because blocks two apart share no site. This file is a transcription. What
// differs, and why:
//
//   * ONE BLOCK IS ONE SERIAL ROUTINE. On the device a block is 64 threads with its quadratic in shared memory, and
//     every decision is thread 0's while the other 63 wait at a barrier: the threads exist to make the assembly and the
//     sums parallel. Here the routine walks the same steps on one thread - each of the device's per-thread loops becomes
//     a loop over the thread index in increasing order - and the blocks of a colour are what is spread over the pool.
//     Every barrier and every shared array is gone; the arithmetic of each value is the kernel's;
//   * THE BLOCK SUMS ARE PLAIN SERIAL FOLDS. refine_reduce is a 64-slot halving tree; the five endpoint sums (b.p, b.q,
//     p A p, p A q, q A q) and the endpoint candidate's exact change are its six uses. Section 0a of the plan retired
//     the replay section 5.6 asked for: each is summed here over the block's sixteen texels in increasing order, from
//     0.0. So the three decisions those sums feed - the relative determinant guard, the nine-candidate argmin and the
//     strict-decrease acceptance - can land differently from CUDA's where a sum sits within a rounding of its
//     threshold, and the per-kernel harness counts such a case as a FLIP. The selector stage has no reduction on the
//     device either (its decision is thread 0's own loop), so it is the kernel's arithmetic exactly;
//   * THE PASS FIGURES. The device adds each improving block's decrease and a count of one into the plane's two
//     doubles with atomicAdd, the one order-dependent sum in the encoder (docs/DESIGN.md section 6); here each chunk
//     sums its own blocks in block order and the chunks are folded in chunk order after the join, the colours in order.
//     Nothing reads the figures but the log, so the difference is not a decision; the CPU's are the stricter of the two;
//   * THE DELTA ALIAS, NAMED. k_bc_decode and k_bc_refine write their `delta` into cg0_x, level 0's conjugate-gradient
//     solution vector, and read their `prev` from q0_prev: the correction from the plane block (c')'s assembly was built
//     at. The CPU model has the same two buffers under the same names (cpu_model.h), and the calls below pass them by
//     those names so that the alias is written down here and not inherited silently;
//   * THE CLOCKS are the wall clock around the same stretches the device's events bracket.
//
// Everything else is copied: bc4_weight8, bc4_entry and bc0_value (its host half, which spells the multiply and the add
// as two separately rounded double operations - the __dadd_rn / __dmul_rn of the device half - and relies on the build
// not contracting them, as model.h's level1_value does), refine_grid's arithmetic in chunks_for, plane_blocks
// (cpu_model.h), the edge rule (a texel outside the plane keeps a zero row, so its selector never moves), the eight-value
// mode's `continue` on a0 <= a1, the float round trip `(int)(byte + 0.5f)` of the decoded byte, and the host drivers.
//
// ZERO DATA RACES (section 3.4). A colour is one pool run and a chunk covers a contiguous range of that colour's
// blocks. A block writes its own endpoints and selectors and, for its own sixteen texels, v0, k0 and the correction;
// it reads the stencil, the gradient and the plane the correction is measured from, which no chunk of the run writes,
// its own texels' v0, and the correction of every texel one step outside its edge (a stencil slot's offset is at most
// one texel in each axis, forward or read backwards as a transpose). Such a texel is in a block one step away on the
// block grid, which is of another colour: two blocks of one colour are two blocks, eight texels, apart in some axis. So
// no block reads what another block of its run writes; the four runs are the barrier the colours need between them.
// The pass figures are returned by value into the pool's own slots and folded after the join. k_bc_decode writes each
// texel's own three outputs from its block's bytes, which it only reads.

#include "cpu/cpu_refine_bc.h"

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"

#include <chrono>
#include <cmath>
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

// The blocks one chunk of a refinement colour covers. A block is a whole 4x4 problem - its (16 C0)^2 quadratic
// gathered from five stencil blocks per texel, nine endpoint candidates and two selector sweeps per channel - so
// CPU_CHUNK blocks to a chunk would put a 1024x1024 plane's colour (16384 blocks) in four chunks and leave most of the
// pool idle. Like CPU_CHUNK it is a constant and never a function of the thread count (cpu/pool.h, rule 1), so the pass
// figures, the one thing folded across chunks, are the same bits at every -j.
const size_t CPU_BLOCK_CHUNK = 64;

// refine_bc.cu's bc4_weight8, copied: bc_pack.h's eight-value palette (r0 > r1) as the affine weight on the high
// endpoint, p = a r0 + (1 - a) r1.
inline double bc4_weight8(int sel)
{
    return sel == 0 ? 1.0 : (sel == 1 ? 0.0 : (double)(8 - sel) / 7.0);
}

// refine_bc.cu's bc4_entry, copied: the same palette entry in float and in bc_pack.h's own spelling.
inline float bc4_entry(int r0, int r1, int sel)
{
    if (sel == 0)
        return (float)r0;
    if (sel == 1)
        return (float)r1;
    return (float)((8 - sel) * r0 + (sel - 1) * r1) / 7.0f;
}

// refine_bc.cu's bc0_value, its host half: the plane value a decoded byte stands for, the multiply and the add rounded
// separately in double, as export.cpp's level0_plane_values computes it.
inline float bc0_value(float lo, float hi, float byte)
{
    const double frac = (double)byte / 255.0;
    const double span = (double)hi - (double)lo;
    return (float)((double)lo + frac * span);
}

// The byte's nearest 8-bit index, the kernel's clamp and its float round trip.
inline uint8_t bc0_index(float byte)
{
    return (uint8_t)(byte < 0.0f ? 0 : (byte > 255.0f ? 255 : (int)(byte + 0.5f)));
}

// k_bc_decode for texel ti: the plane value, the index and the correction its block's bytes give it.
void bc_decode_texel(const uint8_t* ep, const uint8_t* sel, const float* lo0, const float* hi0, int w0, int c0, int bx,
                     const float* prev, float* v0, uint8_t* k0, double* delta, size_t ti)
{
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
        k0[idx] = bc0_index(byte);
        delta[idx] = (double)value - (double)prev[idx];
    }
}

// k_bc_refine for block (gx, gy), as one serial routine. `work` is the block's scratch - the device's shared memory -
// of at least n * n + 3 n + 16 doubles, n = 16 C0. Returns the sum of the decreases the block accepted (zero or
// negative), which is what the device's thread 0 calls `taken`.
double bc_refine_block(const double* stencil, const double* grad, const float* prev, const float* lo0, const float* hi0,
                       int w0, int h0, int c0, int bx, int gx, int gy, uint8_t* ep, uint8_t* sel, double* delta,
                       float* v0, uint8_t* k0, double* work)
{
    const int n = 16 * c0;
    double* a2 = work;            // n x n, the block's own quadratic
    double* a1 = a2 + n * n;      // n, the gradient with the neighbours' share folded in
    double* dv = a1 + n;          // n, the block's correction from the plane the stencil saw
    double* uv = dv + n;          // n, a2 . dv, kept up to date as values move
    double* dlt = uv + n;         // 16, one endpoint candidate's change per texel
    int s_ep[2 * MAX_CHANNELS];
    unsigned char s_sel[16 * MAX_CHANNELS];

    for (int i = 0; i < n * n; i++)
        a2[i] = 0.0;
    for (int i = 0; i < n; i++)
    {
        a1[i] = 0.0;
        dv[i] = 0.0;
        uv[i] = 0.0;
    }

    // ---- the block's quadratic, texel by texel (the device's thread per texel). A texel outside the plane keeps the
    // zero rows it was given, so its selector never moves. Every entry of a2 receives at most one addition, onto its
    // zero - the negative of a slot offset is never itself a slot offset, bar the diagonal - so the order the texels
    // are walked in cannot show in it.
    const size_t bstride = (size_t)(STENCIL_BLOCKS * c0 * c0);
    for (int tid = 0; tid < 16; tid++)
    {
        const int px = gx * 4 + (tid & 3), py = gy * 4 + (tid >> 2);
        if (!(px < w0 && py < h0))
            continue;
        const size_t ti = (size_t)py * w0 + px;
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
                    // Inside the block: the pair belongs to A2, and its transpose with it.
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

    for (int r = 0; r < n; r++)
    {
        double s = 0.0;
        for (int j = 0; j < n; j++)
            s += a2[(size_t)r * n + j] * dv[j];
        uv[r] = s;
    }
    const size_t gi = (size_t)gy * bx + gx;
    for (int c = 0; c < c0; c++)
    {
        s_ep[2 * c] = ep[(gi * (size_t)c0 + (size_t)c) * 2];
        s_ep[2 * c + 1] = ep[(gi * (size_t)c0 + (size_t)c) * 2 + 1];
    }
    for (int i = 0; i < n; i++)
        s_sel[i] = sel[gi * (size_t)c0 * 16 + (size_t)i];

    double taken = 0.0;   // the running total of the decreases this block accepted

    // ---- (i) the endpoints of each channel, by least squares on E_blk
    for (int c = 0; c < c0; c++)
    {
        const double step = ((double)hi0[c] - (double)lo0[c]) / 255.0;
        // The five sums refine_reduce forms on the device, here serial folds over the sixteen texels (file header).
        double bp = 0.0, bq = 0.0, pap = 0.0, paq = 0.0, qaq = 0.0;
        for (int tid = 0; tid < 16; tid++)
        {
            const int idx = tid * c0 + c;
            const double a = bc4_weight8(s_sel[c * 16 + tid]);
            const double p = step * a;
            const double q = step * (1.0 - a);
            const double b = a1[idx] + uv[idx];
            double ap = 0.0, aq = 0.0;
            for (int j = 0; j < 16; j++)
            {
                const double e = a2[(size_t)idx * n + (j * c0 + c)];
                const double aj = bc4_weight8(s_sel[c * 16 + j]);
                ap += e * step * aj;
                aq += e * step * (1.0 - aj);
            }
            bp += b * p;
            bq += b * q;
            pap += p * ap;
            paq += p * aq;
            qaq += q * aq;
        }

        // The vertex of the 2x2 problem, rounded onto the 8-bit grid, and the nine candidates around it, behind the
        // determinant guard RELATIVE to the diagonal it came from (refine_bc.cu says why).
        const int r0 = s_ep[2 * c], r1 = s_ep[2 * c + 1];
        const double det = pap * qaq - paq * paq;
        double d0 = 0.0, d1 = 0.0;
        if (det > 1e-12 * pap * qaq)
        {
            d0 = (paq * bq - qaq * bp) / det;
            d1 = (paq * bp - pap * bq) / det;
        }
        const double v0r = (double)r0 + d0, v1r = (double)r1 + d1;
        const int c0r = (int)std::floor(v0r + 0.5), c1r = (int)std::floor(v1r + 0.5);
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

        const int new0 = best0, new1 = best1;
        if (new0 == s_ep[2 * c] && new1 == s_ep[2 * c + 1])
            continue;

        // The candidate weighed again in the arithmetic the plane is written in: the acceptance.
        for (int tid = 0; tid < 16; tid++)
        {
            const float now = bc0_value(lo0[c], hi0[c], bc4_entry(s_ep[2 * c], s_ep[2 * c + 1], s_sel[c * 16 + tid]));
            const float next = bc0_value(lo0[c], hi0[c], bc4_entry(new0, new1, s_sel[c * 16 + tid]));
            dlt[tid] = (double)next - (double)now;
        }
        double exact = 0.0;
        for (int tid = 0; tid < 16; tid++)
        {
            const int idx = tid * c0 + c;
            double quad = 0.0;
            for (int j = 0; j < 16; j++)
                quad += a2[(size_t)idx * n + (j * c0 + c)] * dlt[j];
            exact += dlt[tid] * (2.0 * (a1[idx] + uv[idx]) + quad);
        }
        if (exact < 0.0)
        {
            for (int r = 0; r < n; r++)
            {
                double add = 0.0;
                for (int j = 0; j < 16; j++)
                    add += a2[(size_t)r * n + (j * c0 + c)] * dlt[j];
                uv[r] += add;
            }
            s_ep[2 * c] = new0;
            s_ep[2 * c + 1] = new1;
            for (int j = 0; j < 16; j++)
                dv[j * c0 + c] += dlt[j];
            taken += exact;
        }
    }

    // ---- (ii) the selectors, two Gauss-Seidel passes of exact 8-way argmins
    for (int pass = 0; pass < 2; pass++)
        for (int c = 0; c < c0; c++)
            for (int t = 0; t < 16; t++)
            {
                const int i = t * c0 + c;
                const double b = a1[i] + uv[i], a = a2[(size_t)i * n + i];
                const int r0 = s_ep[2 * c], r1 = s_ep[2 * c + 1];
                const double now = (double)bc0_value(lo0[c], hi0[c], bc4_entry(r0, r1, s_sel[c * 16 + t]));
                double best = 0.0, move = 0.0;
                int which = s_sel[c * 16 + t];
                for (int kk = 0; kk < 8; kk++)
                {
                    const double d = (double)bc0_value(lo0[c], hi0[c], bc4_entry(r0, r1, kk)) - now;
                    const double de = 2.0 * b * d + a * d * d;
                    if (de < best)
                    {
                        best = de;
                        move = d;
                        which = kk;
                    }
                }
                if (best < 0.0)
                {
                    for (int r = 0; r < n; r++)
                        uv[r] += a2[(size_t)r * n + i] * move;
                    s_sel[c * 16 + t] = (unsigned char)which;
                    dv[i] += move;
                    taken += best;
                }
            }

    // ---- what the block leaves behind
    for (int c = 0; c < c0; c++)
    {
        ep[(gi * (size_t)c0 + (size_t)c) * 2] = (uint8_t)s_ep[2 * c];
        ep[(gi * (size_t)c0 + (size_t)c) * 2 + 1] = (uint8_t)s_ep[2 * c + 1];
    }
    for (int i = 0; i < n; i++)
        sel[gi * (size_t)c0 * 16 + (size_t)i] = s_sel[i];
    for (int tid = 0; tid < 16; tid++)
    {
        const int px = gx * 4 + (tid & 3), py = gy * 4 + (tid >> 2);
        if (!(px < w0 && py < h0))
            continue;
        const size_t ti = (size_t)py * w0 + px;
        for (int c = 0; c < c0; c++)
        {
            const float byte = bc4_entry(s_ep[2 * c], s_ep[2 * c + 1], s_sel[c * 16 + tid]);
            const float value = bc0_value(lo0[c], hi0[c], byte);
            const size_t idx = ti * (size_t)c0 + (size_t)c;
            v0[idx] = value;
            k0[idx] = bc0_index(byte);
            delta[idx] = (double)value - (double)prev[idx];
        }
    }
    return taken;
}

// The pass figures of a run of blocks: the accepted decreases of the blocks that improved, and their count.
struct Taken
{
    double taken = 0.0, improved = 0.0;
};

}   // namespace

void bc_refine_colour(CpuModel* d, const Model& m, int plane, int colour, double& taken, double& improved)
{
    CpuPlane& p = d->planes[(size_t)plane];
    const int bx = (p.w0 + 3) / 4, by = (p.h0 + 3) / 4;
    const int cx = colour & 1, cy = colour >> 1;
    const int cbx = (bx - cx + 1) / 2;
    const size_t count = (size_t)cbx * (size_t)((by - cy + 1) / 2);
    if (count == 0)
        return;
    const int c0 = m.c0;
    const size_t scratch = (size_t)(16 * c0) * (size_t)(16 * c0) + 3 * (size_t)(16 * c0) + 16;
    // The level-0 workspace under its own names: the stencil and gradient block (c')'s assembly left, the plane it was
    // assembled at (q0_prev), and the correction from it (cg0_x, the conjugate-gradient solution vector's buffer).
    const double* stencil = p.stencil0.data();
    const double* grad = p.grad0.data();
    const float* prev = p.q0_prev.data();
    double* delta = p.cg0_x.data();
    const Taken sum = fold_chunks(
        *d->pool, chunks_for(count, CPU_BLOCK_CHUNK), Taken(),
        [&](int ch) {
            std::vector<double> work(scratch);
            Taken part;
            const size_t begin = (size_t)ch * CPU_BLOCK_CHUNK;
            const size_t end = begin + CPU_BLOCK_CHUNK < count ? begin + CPU_BLOCK_CHUNK : count;
            for (size_t b = begin; b < end; b++)
            {
                const int gx = cx + 2 * (int)(b % (size_t)cbx);
                const int gy = cy + 2 * (int)(b / (size_t)cbx);
                const double t = bc_refine_block(stencil, grad, prev, d->lo0.data(), d->hi0.data(), p.w0, p.h0, c0, bx,
                                                 gx, gy, p.bc_ep.data(), p.bc_sel.data(), delta, p.v0.data(),
                                                 p.k0.data(), work.data());
                if (t < 0.0)
                {
                    part.taken += t;
                    part.improved += 1.0;
                }
            }
            return part;
        },
        [](Taken acc, const Taken& part) {
            acc.taken += part.taken;
            acc.improved += part.improved;
            return acc;
        });
    taken += sum.taken;
    improved += sum.improved;
}

// The pack the refinement starts from or continues, decoded into the plane, with block (c')'s stencil assembled at the
// plane the loop left (refine_bc.cu's bc_pack_prepare): `seed` takes bc_pack.h's stb_dxt-style pack of m.k0 through
// export.cpp's level0_pack_seed, the host code the CUDA driver calls; otherwise the endpoints and selectors the model
// holds are kept and only re-assembled against.
double bc_pack_prepare(CpuModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels)
{
    double ms = assemble_level0_for_refine(d, m, k);
    const size_t np = m.planes.size();
    if (seed)
    {
        std::vector<uint8_t> ep, sel;
        for (size_t i = 0; i < np; i++)
        {
            ::level0_pack_seed(m, (int)i, ep, sel);
            CpuPlane& p = d->planes[i];
            if (ep.size() > p.bc_ep.size() || sel.size() > p.bc_sel.size())
                continue;
            pool_copy(*d->pool, ep.data(), p.bc_ep.data(), ep.size());
            pool_copy(*d->pool, sel.data(), p.bc_sel.data(), sel.size());
        }
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < np; i++)
    {
        CpuPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        const uint8_t* ep = p.bc_ep.data();
        const uint8_t* sel = p.bc_sel.data();
        const float* lo0 = d->lo0.data();
        const float* hi0 = d->hi0.data();
        const float* prev = p.q0_prev.data();
        float* v0 = p.v0.data();
        uint8_t* k0 = p.k0.data();
        double* delta = p.cg0_x.data();
        const int w0 = p.w0, c0 = m.c0, bx = (p.w0 + 3) / 4;
        d->pool->run(chunks_for(texels, CPU_CHUNK), [&](int ch) {
            const size_t begin = (size_t)ch * CPU_CHUNK;
            const size_t end = begin + CPU_CHUNK < texels ? begin + CPU_CHUNK : texels;
            for (size_t ti = begin; ti < end; ti++)
                bc_decode_texel(ep, sel, lo0, hi0, w0, c0, bx, prev, v0, k0, delta, ti);
        });
    }
    ms += since_ms(t0);
    bc_blocks_from_device(d, m, pack_psnr, pack_texels);
    return ms;
}

// `passes` rounds of the refinement over every plane, four colour runs each, one report entry per pass, and the blocks
// it chose written back into the model (refine_bc.cu's bc_refine).
double bc_refine(CpuModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels)
{
    rep.clear();
    if (passes <= 0)
        return 0.0;
    const size_t np = m.planes.size();
    double ms = 0.0;
    for (int pass = 0; pass < passes; pass++)
    {
        const auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < np; i++)
        {
            CpuPlane& p = d->planes[i];
            double taken = 0.0, improved = 0.0;
            for (int colour = 0; colour < 4; colour++)
                bc_refine_colour(d, m, (int)i, colour, taken, improved);
            p.bc_stats[0] = taken;
            p.bc_stats[1] = improved;
        }
        const double pass_ms = since_ms(t0);
        ms += pass_ms;

        BcRefineReport r;
        r.pass = pass + 1;
        r.ms = pass_ms;
        for (size_t i = 0; i < np; i++)
        {
            r.delta_e += d->planes[i].bc_stats[0];
            r.improved += (long long)d->planes[i].bc_stats[1];
            r.blocks += (long long)plane_blocks(m, (int)i);
        }
        rep.push_back(r);
    }
    bc_blocks_from_device(d, m, pack_psnr, pack_texels);
    return ms;
}

// The endpoints and selectors the model holds, turned into the .dds's blocks and into the Model, with the pack's own
// error against the continuous plane measured on the way (refine_bc.cu's bc_blocks_from_device; the packing is
// export.cpp's level0_pack_blocks, the host code the CUDA driver calls).
void bc_blocks_from_device(CpuModel* d, Model& m, double& pack_psnr, size_t& pack_texels)
{
    const size_t np = m.planes.size();
    m.bc0_blocks.assign(np, std::vector<std::vector<uint8_t>>());
    double se = 0.0;
    size_t n = 0;
    std::vector<uint8_t> ep, sel;
    for (size_t i = 0; i < np; i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t blocks = plane_blocks(m, (int)i);
        ep.assign(blocks * (size_t)m.c0 * 2, 0);
        sel.assign(blocks * (size_t)m.c0 * 16, 0);
        if (p.bc_ep.size() >= ep.size() && p.bc_sel.size() >= sel.size())
        {
            pool_copy(*d->pool, p.bc_ep.data(), ep.data(), ep.size());
            pool_copy(*d->pool, p.bc_sel.data(), sel.data(), sel.size());
        }
        ::level0_pack_blocks(m, (int)i, ep, sel, m.bc0_blocks[i], se, n);
    }
    pack_texels = n;
    pack_psnr = n > 0 && se > 0.0 ? 10.0 * std::log10(255.0 * 255.0 / (se / (double)n)) : 100.0;
}

}   // namespace nntc_cpu
