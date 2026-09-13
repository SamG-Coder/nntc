// init.cu: the device context and the initial state of both latents.
//
// LEVEL 0 (--init0):
//
//   luma      the control, and what this file did alone: s_0(texel) = 2 luma(src_m at that texel) - 1, taken to the
//             NEAREST VALUE of the channel's palette - pal(k) = -1 + 2 rep(k) / 255 uncompressed, -1 + 2k / (2^b - 1)
//             block-compressed, whichever the shipped format decodes to. Any further channel starts at the palette
//             value nearest 0, and the decoder's least squares is left to give it a direction.
//
//             It cannot. A channel that is zero everywhere is a constant feature: block (a) sees a dead column, hands
//             it whatever the ridge leaves, and the alternation then has to invent the channel out of nothing. On the
//             four-texture material md1-md4 that is why --c0 4 measured WORSE than --c0 3 at the same memory - more
//             capacity, a poorer basin.
//
//   residual  the default: EVERY channel starts at zero here, and init0_residual_channel below seeds them one at a
//             time from the residual, each followed by a refit of the decoder. Channel 0 is not assumed to be luma:
//             the first direction is whatever the level-1-only reconstruction misses most at full resolution, which on
//             a photograph comes out close to luma and on a normal map does not.
//
//             The projection is divided by its own PEAK under --l0 bc8 and by a PERCENTILE of itself under
//             --l0 palette, and the reason is the grid: bc8's lo/hi is fitted to the plane the solve produces, so the
//             seed's scale is a gauge the freeze removes, while the palette is fixed on [-1,1] and a seed that uses a
//             fiftieth of it lands on the entries around zero with its detail already gone. The palette mode also
//             follows each seeded channel with one exact level-0 search and one more refit, so the assignment the next
//             channel's residual is read against is chosen by the objective and not by a blind snap; without it every
//             further channel re-picks the first one's direction. init0_residual_channel's (4a') states the first
//             measurement and main.cpp's seeding loop the second.
//
//             "Zero" is exact under --l0 bc8 and only nearly true under --l0 palette, and the difference decides what
//             may be claimed for the seed. Under bc8 an unseeded channel IS 0.0f: its column is one block (a) can only
//             see as zero, so writing the seeded channel over it cannot move E and the whole seed is monotone in E,
//             write by write and refit by refit. Under palette no index stands for 0 at any depth the mode allows
//             (levels = 2^b - 1 is odd at every one of them), so an unseeded channel is a small non-zero constant that
//             block (a) has spent weight on, and the write itself CAN raise E - only the refit after it is a
//             minimisation. The stronger claim is bc8's and is stated as bc8's wherever it is made.
//
// Level 1 starts from the block mean of the source: the average of all 3T source channels over a level-1 texel's
// footprint in the plane. Two ways of turning those 3T numbers into C1 channels:
//
//   box   channel j is source channel j's own block mean, mapped from [0,1] to [-1,1] by v = 2 mean - 1, and a channel
//         beyond the source's 3T starts at 0 - which is a channel with no direction at all, and the decoder can only
//         give it one through the level-0 cross terms.
//   pca   the block means are projected onto the first C1 principal directions of their own covariance over the plane.
//         Every kept channel then carries variance by construction, largest first, and the fourth channel of a single
//         RGB texture holds the smallest of the three real directions rather than nothing. Each component is divided by
//         the largest magnitude it actually takes over the chain, so the channel arrives filling [-1,1] exactly and
//         NOTHING is clipped; the clamp that follows is a no-op kept only as a guard against a rounding step at the
//         extreme value itself.
//
//         Scaling by the standard deviation instead - the obvious choice, and the one this file started with - clips.
//         A principal component's values are not bounded by two standard deviations: on a single texture the dominant
//         direction carries most of the variance and its tails run several deviations out, so dividing by sigma and
//         clamping flattens every texel in those tails onto +-1 and throws away exactly the contrast the first
//         component was chosen to carry. Dividing by max |value| cannot clip by construction and keeps the same
//         common scale across the channels, since every channel then spans [-1,1].
//
// The principal directions are computed on the BASE plane and used for every plane of the chain. One decoder is shared
// by every stored level, so channel j has to mean the same direction of the material at every level; a per-plane
// eigenbasis would have each level's channel 2 pointing somewhere else and the one shared W could fit none of them.
//
// The per-channel grid is then fitted over the base and every plane together (the asset format carries one lo/hi pair
// per channel for the whole chain) and the plane is snapped onto it.

#include <algorithm>
#include <cmath>
#include <vector>

#include "device.cuh"
#include "sample.cuh"

// ---------------------------------------------------------------------------------------------------------------
// The device context
// ---------------------------------------------------------------------------------------------------------------

bool device_select(int index, DeviceInfo& info)
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0)
        return false;
    if (index < 0 || index >= count)
        return false;
    if (cudaSetDevice(index) != cudaSuccess)
        return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, index) != cudaSuccess)
        return false;
    // The compiled floor is compute capability 8.0 (CMakeLists.txt). A pre-Ampere device would otherwise fail at the
    // first launch with "no kernel image is available", which says nothing about why, so it is refused here by name.
    if (prop.major < 8)
    {
        fprintf(stderr, "ERROR: device %d is '%s', compute capability %d.%d; this encoder is compiled for 8.0 and "
                        "above\n", index, prop.name, prop.major, prop.minor);
        return false;
    }
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    return true;
}

// Every cudaMalloc of the run goes through this, so the report can name the device memory the encode holds.
template <typename T>
static void device_alloc(DeviceModel* d, T** p, size_t bytes)
{
    CUDA_CHECK(cudaMalloc(p, bytes));
    d->device_bytes += bytes;
}

DeviceModel* device_create(const Model& m, const std::vector<Image>& source_chain)
{
    DeviceModel* d = new DeviceModel();
    d->nout = m.nout;
    d->nin = m.dec.nin;
    d->c0 = m.c0;
    d->c1 = m.c1;
    d->l0_bc8 = m.l0_bc8;
    d->mono0_count = stencil_monomials(m.c1);
    d->planes.resize(m.planes.size());
    CUDA_CHECK(cudaStreamCreateWithFlags(&d->master, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&d->master_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&d->block_start));
    CUDA_CHECK(cudaEventCreate(&d->block_end));

    const size_t bsq = (size_t)m.c1 * m.c1;
    size_t scratch = 0, rgb8 = 0, texels1 = 0;
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const PlaneSize& p = m.planes[i];
        DevPlane& dp = d->planes[i];
        dp.w0 = p.w0;
        dp.h0 = p.h0;
        dp.w1 = p.w1;
        dp.h1 = p.h1;
        const size_t n0 = (size_t)p.w0 * p.h0, n1 = (size_t)p.w1 * p.h1;
        device_alloc(d, &dp.src, n0 * m.nout * sizeof(float));
        device_alloc(d, &dp.v0, n0 * m.c0 * sizeof(float));
        device_alloc(d, &dp.v1, n1 * m.c1 * sizeof(float));
        device_alloc(d, &dp.k0, n0 * m.c0);
        device_alloc(d, &dp.k1, n1 * m.c1);
        // The one upload that does NOT go through the master stream and a fan-out: it happens while the planes are
        // still being created, so there are no streams to fan out to yet, and it is a synchronous copy that has landed
        // before this plane's own stream exists. Every upload after this point is device_upload_fan_out's business.
        CUDA_CHECK(cudaMemcpy(dp.src, source_chain[i].v.data(), n0 * m.nout * sizeof(float), cudaMemcpyHostToDevice));

        // Block (b)'s workspace, one per plane: the planes are independent problems and are solved at the same time on
        // their own streams, so they cannot share a scratch buffer. It is allocated once, so a round costs none.
        device_alloc(d, &dp.stencil, n1 * STENCIL_BLOCKS * bsq * sizeof(double));
        device_alloc(d, &dp.grad, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.precond, n1 * bsq * sizeof(float));
        device_alloc(d, &dp.cg_x, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.cg_r, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.cg_z, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.cg_p, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.cg_ap, n1 * m.c1 * sizeof(double));
        device_alloc(d, &dp.q1_prev, n1 * m.c1 * sizeof(float));

        // Block (c')'s workspace, under --l0 bc8 only. It is the same five buffers over a plane with BLOCK^2 times as
        // many texels, so it is the largest allocation the encode makes and it is not made at all in the palette mode.
        if (m.l0_bc8)
        {
            const size_t b0 = (size_t)m.c0 * m.c0;
            device_alloc(d, &dp.stencil0, n0 * STENCIL_BLOCKS * b0 * sizeof(double));
            device_alloc(d, &dp.grad0, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.precond0, n0 * b0 * sizeof(float));
            device_alloc(d, &dp.cg0_x, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.cg0_r, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.cg0_z, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.cg0_p, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.cg0_ap, n0 * m.c0 * sizeof(double));
            device_alloc(d, &dp.q0_prev, n0 * m.c0 * sizeof(float));

            // The pack's own state: one block per 4x4 texels per axis, two endpoints and sixteen selectors per channel
            // of each. A quarter of a byte per texel-channel beside the plane itself.
            const size_t blocks = (size_t)((p.w0 + 3) / 4) * (size_t)((p.h0 + 3) / 4);
            device_alloc(d, &dp.bc_ep, blocks * (size_t)m.c0 * 2);
            device_alloc(d, &dp.bc_sel, blocks * (size_t)m.c0 * 16);
            device_alloc(d, &dp.bc_stats, 2 * sizeof(double));
        }
        device_alloc(d, &dp.partial, (size_t)PARTIAL_ROWS * REDUCE_BLOCKS * sizeof(double));
        device_alloc(d, &dp.scalars, (size_t)SC_COUNT * sizeof(double));
        device_alloc(d, &dp.counters, 5 * sizeof(unsigned int));
        CUDA_CHECK(cudaStreamCreateWithFlags(&dp.stream, cudaStreamNonBlocking));
        for (int e = 0; e < EV_COUNT; e++)
            CUDA_CHECK(e == EV_DONE ? cudaEventCreateWithFlags(&dp.ev[e], cudaEventDisableTiming)
                                    : cudaEventCreate(&dp.ev[e]));

        scratch = std::max(scratch, n1 * m.c1);
        rgb8 = std::max(rgb8, n0 * m.nout);
        texels1 = std::max(texels1, n1);
    }
    device_alloc(d, &d->weights, (size_t)m.dec.nin * m.nout * sizeof(float));
    device_alloc(d, &d->bias, (size_t)m.nout * sizeof(float));
    // The decoder starts at zero rather than at whatever the allocator handed back: the objective is evaluated once
    // before the first block (a), and that pass has to measure the zero decoder - a defined state, whose E is the
    // weighted mean square of the source itself - and not uninitialised memory.
    CUDA_CHECK(cudaMemset(d->weights, 0, (size_t)m.dec.nin * m.nout * sizeof(float)));
    CUDA_CHECK(cudaMemset(d->bias, 0, (size_t)m.nout * sizeof(float)));
    device_alloc(d, &d->palette, (size_t)MAX_CHANNELS * 16 * sizeof(float));
    device_alloc(d, &d->lo1, (size_t)MAX_CHANNELS * sizeof(float));
    device_alloc(d, &d->hi1, (size_t)MAX_CHANNELS * sizeof(float));
    if (m.l0_bc8)
    {
        device_alloc(d, &d->lo0, (size_t)MAX_CHANNELS * sizeof(float));
        device_alloc(d, &d->hi0, (size_t)MAX_CHANNELS * sizeof(float));
        device_alloc(d, &d->mono0, (size_t)d->mono0_count * m.c0 * m.c0 * sizeof(double));
    }
    device_alloc(d, &d->cw, (size_t)m.nout * sizeof(float));
    device_alloc(d, &d->mono, (size_t)stencil_monomials(m.c0) * bsq * sizeof(double));
    const int mm = m.dec.nin + 1;
    // Block (a) accumulates only the upper triangle of the symmetric normal matrix, beside the nout right-hand sides.
    // Each of those entries is reduced by the fixed two-stage rule: every one of a plane's SITE_BLOCKS blocks holds one
    // partial and the final pass adds the planes and the blocks by index, so the matrix does not depend on the order
    // the blocks ran in. The objective's three sums per plane are reduced the same way.
    const size_t entries = (size_t)(mm * (mm + 1) / 2 + m.nout * mm);
    device_alloc(d, &d->reduction, entries * sizeof(double));
    device_alloc(d, &d->ls_partial, m.planes.size() * entries * (size_t)SITE_BLOCKS * sizeof(double));
    device_alloc(d, &d->ls_omega, m.planes.size() * sizeof(double));
    device_alloc(d, &d->obj, 3 * m.planes.size() * sizeof(double));
    device_alloc(d, &d->obj_partial, 3 * m.planes.size() * (size_t)SITE_BLOCKS * sizeof(double));
    device_alloc(d, &d->scratch, scratch * sizeof(float));
    device_alloc(d, &d->rgb8, rgb8);

    // The pca init's block means and its covariance sums, the latter one partial per block per entry.
    device_alloc(d, &d->means, texels1 * (size_t)m.nout * sizeof(float));
    device_alloc(d, &d->cov, (size_t)(m.nout + m.nout * m.nout) * (size_t)SITE_BLOCKS * sizeof(double));

    // --init0 residual's own buffer: the widest plane's residual, which is as large as that plane's source. The
    // covariance of the residual and the peak of its projection both reduce into d->cov above, whose
    // nout + nout^2 rows of SITE_BLOCKS partials cover the one row a peak needs.
    if (m.init0_residual)
        device_alloc(d, &d->resid, rgb8 * sizeof(float));

    // Pinned staging, so the one copy a block makes of what the host reports does not go through a bounce buffer.
    CUDA_CHECK(cudaMallocHost(&d->host_scalars, m.planes.size() * (size_t)SC_COUNT * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&d->host_counters, m.planes.size() * 5 * sizeof(unsigned int)));

    d->scratch_floats = scratch;
    d->rgb8_bytes = rgb8;
    device_upload_weights(d, m);
    return d;
}

size_t device_memory(const DeviceModel* d)
{
    return d ? d->device_bytes : 0;
}

void device_destroy(DeviceModel* d)
{
    if (!d)
        return;
    for (DevPlane& p : d->planes)
    {
        cudaFree(p.src);
        cudaFree(p.v0);
        cudaFree(p.v1);
        cudaFree(p.k0);
        cudaFree(p.k1);
        cudaFree(p.stencil);
        cudaFree(p.grad);
        cudaFree(p.precond);
        cudaFree(p.cg_x);
        cudaFree(p.cg_r);
        cudaFree(p.cg_z);
        cudaFree(p.cg_p);
        cudaFree(p.cg_ap);
        cudaFree(p.q1_prev);
        cudaFree(p.stencil0);
        cudaFree(p.grad0);
        cudaFree(p.precond0);
        cudaFree(p.cg0_x);
        cudaFree(p.cg0_r);
        cudaFree(p.cg0_z);
        cudaFree(p.cg0_p);
        cudaFree(p.cg0_ap);
        cudaFree(p.q0_prev);
        cudaFree(p.bc_ep);
        cudaFree(p.bc_sel);
        cudaFree(p.bc_stats);
        cudaFree(p.partial);
        cudaFree(p.scalars);
        cudaFree(p.counters);
        for (int e = 0; e < EV_COUNT; e++)
            cudaEventDestroy(p.ev[e]);
        cudaStreamDestroy(p.stream);
    }
    cudaFree(d->weights);
    cudaFree(d->bias);
    cudaFree(d->palette);
    cudaFree(d->lo1);
    cudaFree(d->hi1);
    cudaFree(d->lo0);
    cudaFree(d->hi0);
    cudaFree(d->mono0);
    cudaFree(d->cw);
    cudaFree(d->mono);
    cudaFree(d->reduction);
    cudaFree(d->ls_partial);
    cudaFree(d->ls_omega);
    cudaFree(d->obj);
    cudaFree(d->obj_partial);
    cudaFree(d->scratch);
    cudaFree(d->rgb8);
    cudaFree(d->means);
    cudaFree(d->cov);
    cudaFree(d->resid);
    cudaFreeHost(d->host_scalars);
    cudaFreeHost(d->host_counters);
    cudaEventDestroy(d->master_done);
    cudaEventDestroy(d->block_start);
    cudaEventDestroy(d->block_end);
    cudaStreamDestroy(d->master);
    delete d;
}

// ---------------------------------------------------------------------------------------------------------------
// Level 0
// ---------------------------------------------------------------------------------------------------------------

__global__ void k_init_level0(const float* src, int w, int h, int nout, int c0, ChannelBits bits, const float* palette,
                              float lr, float lg, float lb, int luma_channel0, float* v0, uint8_t* k0)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)w * h)
        return;
    const float* s = src + i * nout;
    const float luma = lr * s[0] + lg * s[1] + lb * s[2];
    for (int c = 0; c < c0; c++)
    {
        const float value = (c == 0 && luma_channel0) ? 2.0f * luma - 1.0f : 0.0f;
        const int levels = (1 << bits.b[c]) - 1;
        // The nearest PALETTE VALUE, not the nearest index. The palette is what the shipped format decodes to, and
        // (value + 1) / 2 * levels assumes that is k / levels, which it is not at 3 bits on an uncompressed plane. The
        // palette is monotone and at most sixteen entries, so the nearest value is one walk and no search; a tie keeps
        // the lower index, as every other argmin in the tree does.
        //
        // AND IT IS NOT ENOUGH TO MAKE AN UNSEEDED CHANNEL ZERO, which is worth stating where the code is. levels is
        // 2^bits - 1 and therefore odd at every depth the palette mode allows, so value 0 sits exactly between two
        // entries and no index stands for it: the best this can do is a seventh of the span at 3 bits, a fifteenth at
        // 4. The residual seed's claim that writing a channel over a column block (a) has so far seen as constant cannot
        // move E holds only where that constant is ZERO - which is --l0 bc8, where an unseeded channel really is 0.0f
        // (k_init_level0_cont below). Under --l0 palette the write can and does raise E, and the refit that follows
        // takes it back; docs/DESIGN.md 3.4 and tests/run_checks.py say so rather than claiming the stronger thing.
        int k = 0;
        float best = fabsf(palette[(size_t)c * 16] - value);
        for (int j = 1; j <= levels; j++)
        {
            const float dist = fabsf(palette[(size_t)c * 16 + j] - value);
            if (dist < best)
            {
                best = dist;
                k = j;
            }
        }
        v0[i * c0 + c] = palette[(size_t)c * 16 + k];
        k0[i * c0 + c] = (uint8_t)k;
    }
    (void)nout;
}

// --l0 bc8's init: the same values, but CONTINUOUS. There is no palette to snap onto - level 0's grid does not exist
// until the freeze fits it - so under --init0 luma the plane starts at s_0 = 2 luma - 1 exactly and the further
// channels at 0, and the first block (c') moves them off it. The stored indices are undefined until the freeze, which
// is the same arrangement level 1 has under --q1-start.
__global__ void k_init_level0_cont(const float* src, int w, int h, int nout, int c0, float lr, float lg, float lb,
                                   int luma_channel0, float* v0)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)w * h)
        return;
    const float* s = src + i * nout;
    const float luma = lr * s[0] + lg * s[1] + lb * s[2];
    for (int c = 0; c < c0; c++)
        v0[i * c0 + c] = (c == 0 && luma_channel0) ? 2.0f * luma - 1.0f : 0.0f;
}

void init_level0(DeviceModel* d, Model& m, const float* luma_weights, bool luma_channel0)
{
    device_upload_palettes(d, m);
    ChannelBits bits = {};
    for (int c = 0; c < m.c0; c++)
        bits.b[c] = m.bits0[c];
    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0;
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        // On the master stream, like every other step that writes a plane the plane's own stream will later read: the
        // plane streams are non-blocking and are ordered against neither the legacy stream nor this one until a fan-out.
        if (m.l0_bc8)
        {
            k_init_level0_cont<<<blocks, threads, 0, d->master>>>(p.src, p.w0, p.h0, m.nout, m.c0, luma_weights[0],
                                                                  luma_weights[1], luma_weights[2],
                                                                  luma_channel0 ? 1 : 0, p.v0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemsetAsync(p.k0, 0, n * m.c0, d->master));
            m.k0[i].assign(n * m.c0, 0);
        }
        else
        {
            k_init_level0<<<blocks, threads, 0, d->master>>>(p.src, p.w0, p.h0, m.nout, m.c0, bits, d->palette,
                                                             luma_weights[0], luma_weights[1], luma_weights[2],
                                                             luma_channel0 ? 1 : 0, p.v0, p.k0);
            CUDA_CHECK(cudaGetLastError());
            m.k0[i].resize(n * m.c0);
            CUDA_CHECK(cudaMemcpyAsync(m.k0[i].data(), p.k0, n * m.c0, cudaMemcpyDeviceToHost, d->master));
        }
        CUDA_CHECK(cudaStreamSynchronize(d->master));
    }
    device_fan_out(d);
}

// ---------------------------------------------------------------------------------------------------------------
// Level 1
// ---------------------------------------------------------------------------------------------------------------

// The block mean: level-1 texel (x, y) of a w1 x h1 plane covers the source pixels [x w0 / w1, (x+1) w0 / w1) in x and
// the matching span in y, so that the footprints tile the plane exactly whatever the floor halving has done to the
// two chains. The mean of source channel j over that footprint, mapped by v = 2 mean - 1.
__global__ void k_init_level1_box(const float* src, int w0, int h0, int nout, int w1, int h1, int c1, float* out)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)w1 * h1)
        return;
    const int x = (int)(i % (size_t)w1), y = (int)(i / (size_t)w1);
    int xa = (int)((long long)x * w0 / w1), xb = (int)((long long)(x + 1) * w0 / w1);
    int ya = (int)((long long)y * h0 / h1), yb = (int)((long long)(y + 1) * h0 / h1);
    if (xb <= xa)
        xb = xa + 1;
    if (yb <= ya)
        yb = ya + 1;
    xb = xb > w0 ? w0 : xb;
    yb = yb > h0 ? h0 : yb;
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

// The snap: the nearest index of the channel's grid, and the value that index stands for, both through model.h's one
// definition so that every path onto the grid lands on the same bits.
__global__ void k_quantise_level1(const float* in, size_t n, int c1, int bits, const float* lo, const float* hi,
                                  float* v1, uint8_t* k1)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const int c = (int)(i % (size_t)c1);
    const int k = level1_index(lo[c], hi[c], bits, in[i]);
    v1[i] = level1_value(lo[c], hi[c], bits, k);
    k1[i] = (uint8_t)k;
}

// The per-channel range of a set of planes under one policy.
//
//   minmax  the smallest and largest value the channel takes. Nothing is ever clamped, and one outlying texel can
//           stretch the grid so far that every other texel loses steps to a range it does not occupy.
//   pct     the 0.1 % and 99.9 % percentiles of the same values, by a sort of the channel's values (a level-1 plane is
//           a sixteenth of the image, so the sort costs milliseconds and needs no histogram). Values outside the range
//           clamp onto the end of the grid: a texel that was an outlier is worse off and every other texel is better
//           off by the width the outlier was spending.
static void fit_range(const std::vector<std::vector<float>>& cont, int c1, const std::string& policy,
                      std::vector<float>& lo, std::vector<float>& hi)
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

// Fit the one per-channel grid the asset format carries over the planes given, freeze it, and snap every plane onto it.
// The continuous values are the planes as they stand on the device; after this call v1 holds the dequantised grid values
// and m.k1 holds the indices the .dds stores, so the two agree by construction.
static void fit_and_snap_level1(DeviceModel* d, Model& m, const std::vector<std::vector<float>>& cont,
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
    device_upload_palettes(d, m);

    m.k1.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w1 * p.h1 * m.c1;
        CUDA_CHECK(cudaMemcpyAsync(d->scratch, cont[i].data(), n * sizeof(float), cudaMemcpyHostToDevice, d->master));
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        k_quantise_level1<<<blocks, threads, 0, d->master>>>(d->scratch, n, m.c1, m.bits1, d->lo1, d->hi1, p.v1, p.k1);
        CUDA_CHECK(cudaGetLastError());
        m.k1[i].resize(n);
        CUDA_CHECK(cudaMemcpyAsync(m.k1[i].data(), p.k1, n, cudaMemcpyDeviceToHost, d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));   // the staging buffer is the caller's, and is reused next round
    }
    device_fan_out(d);   // every plane's stream must see the snapped plane before its next kernel reads it
}

// The planes as they stand on the device.
static void download_planes(DeviceModel* d, const Model& m, std::vector<std::vector<float>>& cont)
{
    cont.assign(d->planes.size(), std::vector<float>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w1 * p.h1 * m.c1;
        cont[i].resize(n);
        CUDA_CHECK(cudaMemcpy(cont[i].data(), p.v1, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

// The round-at-the-end control (--q1-start 0): every plane fits the grid and every plane is snapped onto it, once,
// after the loop has finished with a continuous level 1.
void quantise_level1(DeviceModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_planes(d, m, cont);
    fit_and_snap_level1(d, m, cont, cont, range_policy);
}

// The --q1-start R0 path: the grid is fitted over the BASE AND EVERY CHAIN PLANE TOGETHER and every plane is snapped
// onto it.
//
// One grid for the whole chain is the format's own constraint - the JSON carries exactly one lo/hi pair per channel per
// latent texture, because the GPU dequantises the sampled value with one scale and bias whatever mip the sampler chose.
// So the fit has to see every plane: each plane is solved against its own level of the source chain, and a deeper plane's
// values are not bounded by the base's (a level whose material is flatter pulls the channel one way, a level whose
// block means are more extreme pushes it the other), and a range fitted on the base alone would clamp those planes onto
// the ends of the grid for the rest of the run. Every value of every plane enters the percentile with the same weight:
// the base has the most values and therefore the most say, which is the right balance because it also carries the most
// error, and the chain still moves the range where it genuinely reaches past the base.
//
// Every plane is snapped, the base included: E is exactly quadratic in the level-1 values, so assembling the quantised
// sweeps at the snapped plane rather than at the continuous one gives the same H and the same minimiser, and the sweeps
// then re-optimise every value on the frozen grid. From here on every plane is storable at every moment.
void freeze_level1_grid(DeviceModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_planes(d, m, cont);
    fit_and_snap_level1(d, m, cont, cont, range_policy);
}

// ---------------------------------------------------------------------------------------------------------------
// Level 0's grid under --l0 bc8
// ---------------------------------------------------------------------------------------------------------------
//
// The same two entry points level 1 has, over level 0's plane at 8 bits. The snap is the same kernel and the same
// definition (model.h's level1_value / level1_index at bits = 8, where rep(k) = k, so value = lo + k / 255 * (hi - lo)
// and the index is one round), the range policy is the same --q1-range, and the fit sees the base AND every chain
// plane together for the same reason level 1's does: the JSON carries one lo/hi pair per channel for the whole chain.
//
// The plane can be much larger than d->scratch, which is sized for the widest LEVEL-1 plane, so the staging goes
// through the plane's own v0 rather than through the shared buffer: the continuous values are uploaded back into it
// and the snap reads them in place, which is safe because the kernel reads and writes one value each.
static void fit_and_snap_level0(DeviceModel* d, Model& m, const std::vector<std::vector<float>>& cont,
                                const std::string& policy)
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
    device_upload_palettes(d, m);

    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        CUDA_CHECK(cudaMemcpyAsync(p.v0, cont[i].data(), n * sizeof(float), cudaMemcpyHostToDevice, d->master));
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        k_quantise_level1<<<blocks, threads, 0, d->master>>>(p.v0, n, m.c0, 8, d->lo0, d->hi0, p.v0, p.k0);
        CUDA_CHECK(cudaGetLastError());
        m.k0[i].resize(n);
        CUDA_CHECK(cudaMemcpyAsync(m.k0[i].data(), p.k0, n, cudaMemcpyDeviceToHost, d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));   // the staging vector is the caller's
    }
    device_fan_out(d);
}

static void download_level0_planes(DeviceModel* d, const Model& m, std::vector<std::vector<float>>& cont)
{
    cont.assign(d->planes.size(), std::vector<float>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        cont[i].resize(n);
        CUDA_CHECK(cudaMemcpy(cont[i].data(), p.v0, n * sizeof(float), cudaMemcpyDeviceToHost));
    }
}

void quantise_level0(DeviceModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_level0_planes(d, m, cont);
    fit_and_snap_level0(d, m, cont, range_policy);
}

// The snap with the grid held: every plane onto the lo0 / hi0 the model already carries, no fit. Same kernel and same
// definition as the snap above, so a value lands on the same bits either way; only the range is not recomputed - and,
// with no range to fit, the plane never has to reach the host: the kernel reads p.v0 and writes it back in place.
void snap_level0_on_grid(DeviceModel* d, Model& m)
{
    m.k0.assign(m.planes.size(), std::vector<uint8_t>());
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t n = (size_t)p.w0 * p.h0 * m.c0;
        const int threads = 256;
        const int blocks = (int)((n + threads - 1) / threads);
        k_quantise_level1<<<blocks, threads, 0, d->master>>>(p.v0, n, m.c0, 8, d->lo0, d->hi0, p.v0, p.k0);
        CUDA_CHECK(cudaGetLastError());
        m.k0[i].resize(n);
        CUDA_CHECK(cudaMemcpyAsync(m.k0[i].data(), p.k0, n, cudaMemcpyDeviceToHost, d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));   // the staging vector is this function's
    }
    device_fan_out(d);
}

void freeze_level0_grid(DeviceModel* d, Model& m, const std::string& range_policy)
{
    std::vector<std::vector<float>> cont;
    download_level0_planes(d, m, cont);
    fit_and_snap_level0(d, m, cont, range_policy);
}

// The same footprint, but keeping all nout channels of the mean rather than mapping C1 of them: the raw material the
// principal directions are found in.
__global__ void k_block_means(const float* src, int w0, int h0, int nout, int w1, int h1, float* means)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)w1 * h1)
        return;
    const int x = (int)(i % (size_t)w1), y = (int)(i / (size_t)w1);
    int xa = (int)((long long)x * w0 / w1), xb = (int)((long long)(x + 1) * w0 / w1);
    int ya = (int)((long long)y * h0 / h1), yb = (int)((long long)(y + 1) * h0 / h1);
    if (xb <= xa)
        xb = xa + 1;
    if (yb <= ya)
        yb = ya + 1;
    xb = xb > w0 ? w0 : xb;
    yb = yb > h0 ? h0 : yb;
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

// The sums the covariance is built from, one entry per block row: entry e < nout is sum of channel e, and entry
// nout + r * nout + s is the sum of the product of channels r and s. Each block reduces its own stride of the texels
// for ONE entry in shared memory, so no thread carries the whole nout x nout matrix, and writes that one partial into
// its own slot; the host adds the SITE_BLOCKS partials of an entry in index order.
__global__ void k_cov_partial(const float* means, size_t texels, int nout, double* cov)
{
    extern __shared__ double sh_cov[];
    const int e = (int)blockIdx.y;
    const int r = e < nout ? e : (e - nout) / nout;
    const int s = e < nout ? -1 : (e - nout) % nout;
    double acc = 0.0;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < texels; t += (size_t)gridDim.x * blockDim.x)
    {
        const double a = (double)means[t * (size_t)nout + r];
        acc += s < 0 ? a : a * (double)means[t * (size_t)nout + s];
    }
    sh_cov[threadIdx.x] = acc;
    __syncthreads();
    for (int half = (int)blockDim.x / 2; half > 0; half >>= 1)
    {
        if ((int)threadIdx.x < half)
            sh_cov[threadIdx.x] += sh_cov[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        cov[(size_t)e * SITE_BLOCKS + blockIdx.x] = sh_cov[0];
}

// The cyclic Jacobi eigendecomposition of a small symmetric matrix: rotate away the largest off-diagonal entry over and
// over until the off-diagonal mass is negligible. It is the plainest exact method there is for a matrix this size
// (3T <= MAX_NOUT, which is 18 at the six-texture cap) and it returns an orthonormal basis, which is what the
// projection needs.
static void jacobi_eigen(std::vector<double>& a, int n, std::vector<double>& vec, std::vector<double>& val)
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
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
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

void init_level1(DeviceModel* d, Model& m, bool pca, Level1InitReport& rep)
{
    const int threads = 256;
    rep = Level1InitReport();
    rep.pca = pca;

    std::vector<std::vector<float>> cont(d->planes.size());

    if (!pca)
    {
        // Pass one: the continuous block means of every plane, kept on the host while the shared grid is fitted.
        for (size_t i = 0; i < d->planes.size(); i++)
        {
            const DevPlane& p = d->planes[i];
            const size_t n = (size_t)p.w1 * p.h1;
            const int blocks = (int)((n + threads - 1) / threads);
            k_init_level1_box<<<blocks, threads, 0, d->master>>>(p.src, p.w0, p.h0, m.nout, p.w1, p.h1, m.c1,
                                                                 d->scratch);
            CUDA_CHECK(cudaGetLastError());
            cont[i].resize(n * m.c1);
            CUDA_CHECK(cudaMemcpyAsync(cont[i].data(), d->scratch, n * m.c1 * sizeof(float), cudaMemcpyDeviceToHost,
                                       d->master));
            CUDA_CHECK(cudaStreamSynchronize(d->master));
        }
        // The init's grid is the plain min/max of the initial values, over every plane: nothing is clipped before the
        // solve has decided what the plane holds, and the range policy applies at the fit that freezes the grid.
        fit_and_snap_level1(d, m, cont, cont, "minmax");
        return;
    }

    // The basis: the base plane's block means, their covariance as a device reduction, the eigenproblem on the host.
    const int nout = m.nout;
    const int entries = nout + nout * nout;
    std::vector<double> raw((size_t)entries, 0.0);
    {
        const DevPlane& p = d->planes[0];
        const size_t texels = (size_t)p.w1 * p.h1;
        k_block_means<<<(int)((texels + threads - 1) / threads), threads, 0, d->master>>>(p.src, p.w0, p.h0, nout,
                                                                                         p.w1, p.h1, d->means);
        CUDA_CHECK(cudaGetLastError());
        dim3 grid((unsigned int)SITE_BLOCKS, (unsigned int)entries);
        k_cov_partial<<<grid, threads, threads * sizeof(double), d->master>>>(d->means, texels, nout, d->cov);
        CUDA_CHECK(cudaGetLastError());
        std::vector<double> partial((size_t)entries * SITE_BLOCKS, 0.0);
        CUDA_CHECK(cudaMemcpyAsync(partial.data(), d->cov, partial.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                   d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));
        for (int e = 0; e < entries; e++)
            for (int b = 0; b < SITE_BLOCKS; b++)
                raw[(size_t)e] += partial[(size_t)e * SITE_BLOCKS + b];
    }

    const DevPlane& base = d->planes[0];
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

    // The directions in order of the variance they carry. vec's COLUMNS are the eigenvectors.
    std::vector<int> order((size_t)nout);
    for (int i = 0; i < nout; i++)
        order[(size_t)i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) { return val[(size_t)a] > val[(size_t)b]; });

    rep.components = m.c1 < nout ? m.c1 : nout;
    for (int j = 0; j < rep.components && j < MAX_CHANNELS; j++)
        rep.eigenvalue[j] = val[(size_t)order[(size_t)j]];

    // Pass one: the raw projection of every plane's block means onto that one basis, unscaled, with the largest
    // magnitude each channel reaches kept as it goes. A direction with no variance (more level-1 channels than the
    // source has directions) stays a channel of zeros, which is what the box init would have produced for it anyway.
    std::vector<float> means;
    std::vector<double> peak((size_t)m.c1, 0.0);
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w1 * p.h1;
        k_block_means<<<(int)((texels + threads - 1) / threads), threads, 0, d->master>>>(p.src, p.w0, p.h0, nout,
                                                                                         p.w1, p.h1, d->means);
        CUDA_CHECK(cudaGetLastError());
        means.resize(texels * (size_t)nout);
        CUDA_CHECK(cudaMemcpyAsync(means.data(), d->means, means.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));

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

    // Pass two: each channel divided by its own largest magnitude, so it spans [-1,1] and no value is clipped. The
    // clamp is a no-op by construction and is kept only so that a value rounded a hair past 1 by the division cannot
    // leave the range level 0's palette lives in.
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

// ---------------------------------------------------------------------------------------------------------------
// Level 0 seeded from the residual (--init0 residual)
// ---------------------------------------------------------------------------------------------------------------
//
// PROJECT, REFIT, DEFLATE, REPEAT. With the decoder as it stands, the residual
//
//     r(p) = out(p) - target(p)
//
// at the base plane's own texel centres is what the representation still gets wrong, one vector of nout numbers per
// pixel. Its nout x nout covariance over the plane has a leading eigenvector e - the single direction of the material
// the decode is missing most - and the channel is that residual projected onto it,
//
//     s_k(p) = r(p) . e / D ,   D = max |r(p) . e| over every plane of the chain under --l0 bc8,
//                                   the 99 % percentile of that same magnitude under --l0 palette,
//
// which under bc8 fills [-1,1] and cannot clip, the same scaling rule the level-1 pca init settled on and for the same
// reason: a principal component's tails run several standard deviations out, so dividing by sigma would flatten
// exactly the contrast the direction was chosen to carry. The palette mode cannot use the peak, and (4a') says why.
//
// The caller then refits the decoder. THAT is the deflation. What the new channel can explain leaves the residual
// because block (a) is free to use the channel and does, not because anything was subtracted from r: the next channel's
// covariance is taken on a residual that has already been given the chance to spend this one. A subtraction would
// deflate along e whether or not the decoder could use that direction at this resolution, and the decoder, not the
// covariance, is what decides.
//
// e is computed on the BASE plane and used for every plane of the chain - one decoder is shared by every stored level,
// so channel k has to mean the same direction of the material at every level, exactly as the level-1 pca init argues -
// while the residual, the projection and the peak are each plane's own, taken against its own source mip.
//
// Channel 0 is seeded the same way as the rest, from the residual of a decode with level 0 still at zero: the first
// direction is whatever the level-1 reconstruction misses most at full resolution. On a photograph that lands close to
// luminance; on a normal map or a packed material it does not, and assuming luma there is assuming the answer.

// The residual itself, in the decoder's own arithmetic and WITHOUT the saturate and the rounding to bytes the recon
// PNGs go through: the objective is measured on the unclamped output, so this is the error the next block (a) sees.
__global__ void k_residual(const float* v0, const float* v1, const float* src, int w0, int h0, int w1, int h1, int c0,
                           int c1, int nin, int nout, const float* weights, const float* bias, float* resid)
{
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)w0 * h0)
        return;
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

// The direction a projection is taken along, small enough to be passed to a kernel by value: the eigenvector and the
// residual's own mean, which is subtracted because the covariance the eigenvector came from is of the centred residual
// (a constant offset is the decoder's bias's business, not a channel's).
struct ProjDir
{
    double e[MAX_NOUT];
    double mu[MAX_NOUT];
};

__device__ inline double proj_value(const float* resid, size_t t, int nout, const ProjDir& dir)
{
    double acc = 0.0;
    for (int r = 0; r < nout; r++)
        acc += dir.e[r] * ((double)resid[t * (size_t)nout + r] - dir.mu[r]);
    return acc;
}

// The largest |projection| a plane reaches, on the same fixed grid every other reduction of this file uses: each block
// walks its own stride and writes one partial, and the host takes the maximum of the partials in index order. A maximum
// is order-independent exactly, so this one could not drift with the scheduling anyway; it is written this way to stay
// the one shape.
__global__ void k_proj_peak(const float* resid, size_t texels, int nout, ProjDir dir, double* partial)
{
    extern __shared__ double sh_peak[];
    double acc = 0.0;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < texels; t += (size_t)gridDim.x * blockDim.x)
    {
        const double v = fabs(proj_value(resid, t, nout, dir));
        acc = v > acc ? v : acc;
    }
    sh_peak[threadIdx.x] = acc;
    __syncthreads();
    for (int half = (int)blockDim.x / 2; half > 0; half >>= 1)
    {
        if ((int)threadIdx.x < half)
            sh_peak[threadIdx.x] = sh_peak[threadIdx.x] > sh_peak[threadIdx.x + half] ? sh_peak[threadIdx.x]
                                                                                     : sh_peak[threadIdx.x + half];
        __syncthreads();
    }
    if (threadIdx.x == 0)
        partial[blockIdx.x] = sh_peak[0];
}

// The distribution of |projection| over a plane, binned uniformly on [0, peak] - what the percentile divisor of the
// palette mode is read off. The bins are counts and are accumulated over every plane of the chain, so the percentile is
// the chain's and channel k is on one scale at every level exactly as the peak would have put it. atomicAdd on a
// 64-bit counter is order-independent on integers, so this reduction does not drift with the scheduling either.
__global__ void k_proj_hist(const float* resid, size_t texels, int nout, ProjDir dir, double inv_peak, int bins,
                            unsigned long long* hist)
{
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < texels; t += (size_t)gridDim.x * blockDim.x)
    {
        const double v = fabs(proj_value(resid, t, nout, dir)) * inv_peak;   // in [0,1] by the definition of the peak
        int b = (int)(v * (double)bins);
        b = b < 0 ? 0 : (b >= bins ? bins - 1 : b);
        atomicAdd(hist + b, 1ull);
    }
}

// The channel written: the scaled projection, snapped onto the channel's palette in the palette mode and left
// continuous under --l0 bc8, where level 0's grid does not exist until the freeze fits it. THE CLAMP IS LIVE OR NOT
// DEPENDING ON THE DIVISOR: under --l0 bc8, where the divisor IS the largest magnitude the projection reaches, it is a
// no-op by construction and is kept only so that a value carried a hair past 1 by the division cannot leave the range
// the palette lives in; under --l0 palette the divisor is a percentile of that magnitude and the clamp is what
// saturates the hundredth of the texels above it, which is the point of choosing the percentile (4a').
__global__ void k_seed_channel(const float* resid, size_t texels, int nout, ProjDir dir, double inv, int c0,
                               int channel, int levels, const float* palette, int palette_mode, float* v0, uint8_t* k0)
{
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= texels)
        return;
    const double p = proj_value(resid, t, nout, dir) * inv;
    const float value = (float)(p < -1.0 ? -1.0 : (p > 1.0 ? 1.0 : p));
    if (palette_mode)
    {
        int k = (int)lroundf((value + 1.0f) * 0.5f * (float)levels);
        k = k < 0 ? 0 : (k > levels ? levels : k);
        v0[t * (size_t)c0 + channel] = palette[(size_t)channel * 16 + k];
        k0[t * (size_t)c0 + channel] = (uint8_t)k;
    }
    else
    {
        v0[t * (size_t)c0 + channel] = value;
    }
}

// One plane's residual into d->resid, on the master stream: the plane streams fan into it at the end of every block, so
// the decoder and both latents are what the last block left, and the next kernel here reads the buffer where it lies.
static void residual_pass(DeviceModel* d, const Model& m, size_t plane)
{
    const DevPlane& p = d->planes[plane];
    const size_t n = (size_t)p.w0 * p.h0;
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads);
    k_residual<<<blocks, threads, 0, d->master>>>(p.v0, p.v1, p.src, p.w0, p.h0, p.w1, p.h1, m.c0, m.c1, m.dec.nin,
                                                  m.nout, d->weights, d->bias, d->resid);
    CUDA_CHECK(cudaGetLastError());
}

void init0_residual_channel(DeviceModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep)
{
    const int threads = 256;
    const int nout = m.nout;
    rep = Init0ChannelReport();
    rep.nout = nout;
    rep.texture = -1;

    // (1) and (2): the residual and its covariance, by the same device reduction the level-1 pca init takes over the
    // block means - the buffer it reduces is nout floats per element either way.
    //
    // THE SCOPE (--init0-scope). The direction is one direction for the whole chain, but it does not follow that it
    // should be read off the BASE PLANE ALONE, which is what this used to do. Every plane is fitted by the same shared
    // decoder, and a texture the base does not need a full-resolution channel for can be exactly the texture a deep
    // plane cannot decode without one: level 1 is a quarter of the resolution at every level, so a texture level 1
    // carries at the base has no reason to stay within level 1's reach four levels down. Reading the covariance off
    // the base alone makes that texture's need invisible at the one moment level 0's channels are handed out.
    //
    // `chain` therefore accumulates every plane's residual with the weight THE OBJECTIVE gives that plane's sites
    // (mips.cpp's per-site weight, which is exactly what block (a) will weigh them by), so the direction is the one
    // that explains the most of the error as the run actually measures it. `base` is the control and is the old
    // behaviour exactly.
    const int entries = nout + nout * nout;
    std::vector<double> raw((size_t)entries, 0.0);
    double count = 0.0;
    {
        const size_t nplanes = m.init0_chain ? d->planes.size() : 1;
        std::vector<double> partial((size_t)entries * SITE_BLOCKS, 0.0);
        for (size_t i = 0; i < nplanes; i++)
        {
            const DevPlane& p = d->planes[i];
            const size_t texels = (size_t)p.w0 * p.h0;
            const double w = m.init0_chain ? plane_weight[i] : 1.0;
            if (!(w > 0.0))
                continue;
            residual_pass(d, m, i);
            dim3 grid((unsigned int)SITE_BLOCKS, (unsigned int)entries);
            k_cov_partial<<<grid, threads, threads * sizeof(double), d->master>>>(d->resid, texels, nout, d->cov);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(partial.data(), d->cov, partial.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                       d->master));
            CUDA_CHECK(cudaStreamSynchronize(d->master));
            for (int e = 0; e < entries; e++)
                for (int b = 0; b < SITE_BLOCKS; b++)
                    raw[(size_t)e] += w * partial[(size_t)e * SITE_BLOCKS + b];
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

    // THE COVARIANCE IS THE PLAIN ONE, NOT THE cw-WEIGHTED ONE, and that is a measured choice rather than an oversight.
    // The obvious argument says otherwise: the channel is a scalar s = (r - mu) . e, what the decoder can then take off
    // the error is a rank-one approximation of the residual weighed by cw, and the best such approximation is the
    // leading eigenvector u of the whitened covariance diag(sqrt(cw)) C diag(sqrt(cw)) projected along
    // e = diag(sqrt(cw)) u. That was implemented and measured, and it LOSES: on model10 with --rgb-weights 9,11,1 the
    // shipped E went from 1.314974e-05 to 1.346578e-05 and the psnr from 44.97 to 44.23 dB, and on the three-picture
    // material it was better on two layouts of four and worse on the other two. A seed is not the answer, only the
    // point the alternation starts from, and a direction that is better by the objective's own lights at the first
    // step is not therefore in a better basin. The whitening is a no-op at the default weights in any case, since cw
    // is then all ones.
    for (int r = 0; r < nout; r++)
        rep.total_variance += cov[(size_t)r * nout + r];

    // (3) the direction, by the same host Jacobi sweep.
    //
    // --init0 residual takes the leading eigenvector of the whole nout x nout covariance. --init0 texture takes the
    // leading eigenvector of ONE TEXTURE's own 3x3 block instead - the texture whose block carries the most weighted
    // residual variance - and leaves every other output at zero.
    //
    // WHY THE SECOND EXISTS. With several unrelated pictures in one material the joint covariance's leading direction
    // is a BLEND of all of them: no picture's residual dominates, the top eigenvector mixes them, and the channel that
    // results is a linear combination no picture owns. The refit deflates it, the next direction is another blend, and
    // the run ends with every channel serving every texture a little and none of them serving one texture well - which
    // is a poorer basin than the obvious one, where each full-resolution channel carries one picture and the decoder's
    // s_i c_j products route it. Restricting the direction to one texture makes that assignment the thing the seed
    // chooses, and the choice is by the largest remaining weighted residual, so the channels go where the error is.
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
            u[(size_t)r] = vec[(size_t)r * nout + top];   // vec's COLUMNS are the eigenvectors
    }
    rep.share = rep.total_variance > 0.0 ? rep.eigenvalue / rep.total_variance : 0.0;

    ProjDir dir = {};
    for (int r = 0; r < nout; r++)
    {
        dir.e[r] = u[(size_t)r];
        dir.mu[r] = mu[(size_t)r];
    }
    // An eigenvector is defined up to its sign, and Jacobi's choice of one carries no meaning. Fixing the sign so that
    // the largest component is positive makes the reported direction readable - a luma-like direction then comes out
    // positive rather than negative half the time - and changes nothing else: the decoder's own column absorbs it.
    int biggest = 0;
    for (int r = 1; r < nout; r++)
        if (std::fabs(dir.e[r]) > std::fabs(dir.e[biggest]))
            biggest = r;
    if (dir.e[biggest] < 0.0)
        for (int r = 0; r < nout; r++)
            dir.e[r] = -dir.e[r];
    // Every output, up to the cap the report's own array is sized by (MAX_NOUT, which is model.h's dir[3 *
    // MAX_TEXTURES]). The bound used to be the literal 12 of the four-texture cap, which silently truncated a
    // six-texture material's direction at output 11 and left the last six printing as +0.0000.
    for (int r = 0; r < nout && r < MAX_NOUT; r++)
        rep.dir[r] = dir.e[r];

    // (4a) the divisor: the largest |projection| over the WHOLE chain, so channel k is on one scale at every level, as
    // its direction is. The peak reduces into d->cov, whose first row of SITE_BLOCKS partials is free again.
    double peak = 0.0;
    std::vector<double> partial((size_t)SITE_BLOCKS, 0.0);
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        residual_pass(d, m, i);
        k_proj_peak<<<SITE_BLOCKS, threads, threads * sizeof(double), d->master>>>(d->resid, texels, nout, dir, d->cov);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(partial.data(), d->cov, partial.size() * sizeof(double), cudaMemcpyDeviceToHost,
                                   d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));
        for (int b = 0; b < SITE_BLOCKS; b++)
            peak = std::max(peak, partial[(size_t)b]);
    }
    rep.peak = peak;
    rep.divisor = peak;

    // (4a') THE DIVISOR IS NOT THE PEAK IN THE PALETTE MODE, and the difference is worth several decibels.
    //
    // The peak is an extreme value of one texel of one plane, and a residual's projection is heavy-tailed: on model10
    // the peak is 1.06 while the standard deviation along e is 0.064, so dividing by the peak leaves a channel whose
    // mass sits inside a fiftieth of [-1,1].
    //
    // UNDER --l0 bc8 THAT COSTS NOTHING. Level 0's grid does not exist yet; --q1-start fits lo/hi to the plane the
    // solve actually produced and the stored 8-bit index spans that range whatever scale the seed arrived on. The
    // scale is a gauge the freeze removes.
    //
    // UNDER --l0 palette THERE IS NO SUCH FREEZE. The palette is fixed on [-1,1] - pal(k) = -1 + 2k / (2^b - 1) - so a
    // seed whose mass is a fiftieth of that range lands on the two or three entries around zero and arrives with its
    // detail already gone. The refit that follows fits a large decoder weight to the small channel that is left, the
    // search then moves texels against that weight, and the run settles in a basin where level 0's alphabet is barely
    // used and the decoder's gain on it is correspondingly high - which is also why the BC4 / BC5 pack of a 4-bit
    // plane, whose error is in the plane's own units, comes out multiplied in the output.
    //
    // So in the palette mode the divisor is a PERCENTILE of |projection| instead: the smallest value at or below which
    // INIT0_PERCENTILE of the chain's texels lie, read off a 4096-bin histogram on [0, peak] and taken at the bin's
    // upper edge, so it is never below the true percentile. The clamp in k_seed_channel, which the peak made a no-op,
    // is then live and saturates the tail - a hundredth of the texels at the 99 % INIT0_PERCENTILE the measurement
    // settled on, against the whole plane's contrast that the peak was flattening. The histogram reduces into d->cov,
    // whose nout + nout^2 rows of SITE_BLOCKS doubles cover 4096 eight-byte counters at every layout the encoder
    // allows, exactly as the peak's partials do.
    double divisor = peak;
    if (!m.l0_bc8 && peak > 0.0)
    {
        std::vector<unsigned long long> hist((size_t)PROJ_HIST_BINS, 0ull);
        unsigned long long* dh = (unsigned long long*)d->cov;
        CUDA_CHECK(cudaMemsetAsync(dh, 0, hist.size() * sizeof(unsigned long long), d->master));
        unsigned long long total = 0ull;
        for (size_t i = 0; i < d->planes.size(); i++)
        {
            const DevPlane& p = d->planes[i];
            const size_t texels = (size_t)p.w0 * p.h0;
            residual_pass(d, m, i);
            k_proj_hist<<<SITE_BLOCKS, threads, 0, d->master>>>(d->resid, texels, nout, dir, 1.0 / peak,
                                                                PROJ_HIST_BINS, dh);
            CUDA_CHECK(cudaGetLastError());
            total += (unsigned long long)texels;
        }
        CUDA_CHECK(cudaMemcpyAsync(hist.data(), dh, hist.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                                   d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));
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
            divisor = peak;   // every texel in the first bin: a projection that carries nothing, left on the peak
        rep.divisor = divisor;
    }

    // (4b) the channel itself, on every plane. A residual with no variance at all along e leaves the channel at zero,
    // which is where --init0 luma would have left it anyway.
    const double inv = divisor > 0.0 ? 1.0 / divisor : 0.0;
    const int levels = m.l0_bc8 ? 255 : (1 << m.bits0[(size_t)channel]) - 1;
    for (size_t i = 0; i < d->planes.size(); i++)
    {
        const DevPlane& p = d->planes[i];
        const size_t texels = (size_t)p.w0 * p.h0;
        residual_pass(d, m, i);
        const int blocks = (int)((texels + threads - 1) / threads);
        k_seed_channel<<<blocks, threads, 0, d->master>>>(d->resid, texels, nout, dir, inv, m.c0, channel, levels,
                                                          d->palette, m.l0_bc8 ? 0 : 1, p.v0, p.k0);
        CUDA_CHECK(cudaGetLastError());
        // The palette mode's host mirror of the stored indices, which the writer reads: under --l0 bc8 there is no
        // index yet at all, and m.k0 stays at the zeros the freeze will overwrite.
        if (!m.l0_bc8)
            CUDA_CHECK(cudaMemcpyAsync(m.k0[i].data(), p.k0, texels * (size_t)m.c0, cudaMemcpyDeviceToHost,
                                       d->master));
        CUDA_CHECK(cudaStreamSynchronize(d->master));
    }
    device_fan_out(d);   // every plane's stream must see the seeded channel before its next kernel reads it
}
