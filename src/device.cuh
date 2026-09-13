// device.cuh: the device-resident model.
//
// Everything the solver touches per texel lives on the GPU: the run's own source chain, both latents' planes as
// dequantised float values (what every sampling rule reads) and as quantisation indices (what the .dds stores), and the
// decoder's weights. The host holds only the sizes, the palettes and the report.
//
// A failed CUDA call ends the process, because there is no recovering a half-built device model. It ends it with the
// one failure status the program has, so a caller that tests for non-zero sees the same 1 an argument error gives it.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#include "model.h"

#define CUDA_CHECK(call)                                                                                    \
    do                                                                                                      \
    {                                                                                                       \
        const cudaError_t err_ = (call);                                                                    \
        if (err_ != cudaSuccess)                                                                            \
        {                                                                                                   \
            fprintf(stderr, "ERROR: CUDA %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                      \
                    cudaGetErrorString(err_));                                                              \
            exit(EXIT_FAILURE);                                                                             \
        }                                                                                                   \
    } while (0)

static const int MAX_CHANNELS = 4;    // the per-level channel cap the shader and the .dds formats share
static const int MAX_NIN = 24;        // the shader's |phi| cap
static const int MAX_NOUT = 3 * MAX_TEXTURES;   // the shader's output cap: 3 channels of MAX_TEXTURES textures

// The bit count of each level-0 channel, packed small enough to be passed to a kernel by value.
struct ChannelBits
{
    int b[MAX_CHANNELS];
};

// The fixed grid of a reduction: every block writes one partial and one final block adds the partials in index order,
// so a reduced value does not depend on how the blocks happened to be scheduled.
static const int REDUCE_BLOCKS = 256;

// The blocks of the 9-point stencil a texel OWNS: its own diagonal and the four directions right, down-left, down and
// down-right. The other four are read as the transpose of the neighbour's block in the same slot, which is what makes
// H symmetric by construction rather than by arithmetic. Here because the allocation of the workspace needs it and the
// assembly (solve_level1.cu) defines what goes in it.
static const int STENCIL_BLOCKS = 5;

// Those four directions, in the order they are stored. The definition sits here rather than beside the assembly
// because two translation units read the stencil - solve_level1.cu, which builds and solves it, and refine_bc.cu,
// which reads one 4x4 block's worth of it - and the device code of the two is compiled separately, so the one
// definition has to be in a header they share.
__host__ __device__ inline void stencil_offset(int slot, int& dx, int& dy)
{
    const int ox[STENCIL_BLOCKS] = { 0, 1, -1, 0, 1 };
    const int oy[STENCIL_BLOCKS] = { 0, 0, 1, 1, 1 };
    dx = ox[slot];
    dy = oy[slot];
}

// The slot a texel offset belongs to, or -1 when the offset is owned by the other texel of the pair.
__host__ __device__ inline int stencil_slot(int dx, int dy)
{
    for (int s = 0; s < STENCIL_BLOCKS; s++)
    {
        int ox, oy;
        stencil_offset(s, ox, oy);
        if (ox == dx && oy == dy)
            return s;
    }
    return -1;
}

// The same rule for the two reductions that run over SITES rather than over texels - the decoder's normal equations
// (solve_decoder.cu) and the objective (objective.cu) - and for the covariance of the pca init (init.cu). Their grids
// are fixed too and a block walks its share of the work in a strided loop, so the partial a block writes is a fixed
// subset added in a fixed order whatever the plane's size is, and the final pass adds the partials by index. A wider
// grid than REDUCE_BLOCKS, because these passes carry more of the run's arithmetic and want the occupancy; the final
// in-order pass is a few thousand adds and costs nothing.
static const int SITE_BLOCKS = 512;

// The rows of a plane's partial buffer: three for the stencil diagonal (its sum, smallest and largest), three for the
// movement probe (the count, the correction's square norm and the plane's), and one per channel for each end of the
// plane's range.
static const int PARTIAL_ROWS = 3 + 2 * MAX_CHANNELS;

// The rows a reduction writes its partials into: three general-purpose rows (a dot product, or the diagonal's three
// figures, or the movement probe's three), then one row per channel for each end of the plane's range.
static const int ROW_GENERAL = 0;
static const int ROW_LO = 3;
static const int ROW_HI = 3 + MAX_CHANNELS;

// A plane's device-resident scalars. Everything a round of block (b) needs from a reduction lives here and is read by
// the next kernel where it lies, so neither the conjugate gradients nor the sweeps ever wait for the host: the whole
// row is copied to pinned memory once per block and the host reads it after the block's single synchronisation.
enum PlaneScalar
{
    SC_DIAG_SUM = 0,   // sum, smallest and largest of the diagonal of H, and the ridge they give
    SC_DIAG_MIN,
    SC_DIAG_MAX,
    SC_LAMBDA,
    SC_RHS2,           // |rhs|^2, and the conjugate gradients' running inner products
    SC_RZ,
    SC_RZ_NEW,
    SC_PAP,
    SC_RR,
    SC_RESID,          // |r| / |rhs|, the only one the host looks at, and only once every CG_PROBE iterations
    SC_ALPHA,          // the step and the conjugacy coefficient, computed on the device from the inner products above
    SC_BETA,
    SC_MOVED,          // the movement probe: values that moved by more than half a grid step, and the two norms
    SC_DELTA2,
    SC_PLANE2,
    SC_LO,                              // the plane's range, one pair per channel
    SC_HI = SC_LO + MAX_CHANNELS,
    SC_COUNT = SC_HI + MAX_CHANNELS
};

// The events a plane's timing is read from, recorded on its own stream and created once for the whole run.
enum PlaneEvent
{
    EV_A_START = 0,   // block (a): around the plane's share of the normal equations
    EV_A_END,
    EV_B_START,       // block (b): before the assembly, after it, after the preconditioner, after the CG or the sweeps
    EV_B_ASSEMBLED,
    EV_B_PRECOND,
    EV_B_END,
    EV_C_START,       // block (c): around the plane's four colour passes
    EV_C_END,
    EV_DONE,          // the fan in: what the master stream waits for
    EV_COUNT
};

// THE WORKSPACE OF ONE SPARSE LEAST SQUARES OVER ONE LATENT PLANE.
//
// Block (b) solves level 1 of one stored level as a 9-point stencil system in that plane's values, and under
// `--l0 bc8` block (c') solves level 0 of the same stored level as the very same kind of system in ITS values
// (solve_level1.cu carries both assemblies and the one solver they share). The two differ in which plane is the
// unknown, how wide it is, how many channels it has, which quadratic form the assembly expands and which grid its
// quantised sweeps step on - and in nothing else - so everything downstream of the assembly reads the problem through
// this one description.
//
// The reductions' partials, the scalar row, the counters, the stream and the events are NOT here: they belong to the
// stored level and are shared, because blocks (b) and (c') are separate blocks of the round and never run at once.
struct PlaneWork
{
    int w = 0, h = 0, c = 0;     // the unknown plane's extent and channel count
    int bits = 8;                // its stored depth: the grid its quantised sweeps step on
    int levels = 255;            // (1 << bits) - 1, the index's own range
    float* v = nullptr;          // the plane's dequantised values, w * h * c
    uint8_t* k = nullptr;        // the indices the .dds stores, w * h * c
    const float* lo = nullptr;   // the frozen per-channel grid on the device, one entry per channel
    const float* hi = nullptr;
    const double* mono = nullptr;   // the assembly's monomial matrices, nmono of them, each c x c
    int nmono = 0;

    double* stencil = nullptr;   // texels * 5 * c * c
    double* grad = nullptr;      // texels * c
    float* precond = nullptr;    // texels * c * c
    double* cg_x = nullptr;      // the correction delta
    double* cg_r = nullptr;
    double* cg_z = nullptr;
    double* cg_p = nullptr;
    double* cg_ap = nullptr;
    float* prev = nullptr;       // the plane as it stood when the stencil was assembled

    size_t texels() const { return (size_t)w * h; }
    size_t values() const { return texels() * (size_t)c; }
};

// One stored level on the device.
//
// A plane owns its own block (b) workspace and its own stream. With the decoder held, the planes share no variable at
// all, so the M + 1 problems of a round are independent and are issued together: the base keeps the device busy while
// the deep planes, whose work is a few thousand texels, cost little more than their launches.
struct DevPlane
{
    float* src = nullptr;    // w0 * h0 * nout, the run's own source chain at this level
    float* v0 = nullptr;     // w0 * h0 * c0, level 0's dequantised values
    float* v1 = nullptr;     // w1 * h1 * c1, level 1's dequantised values
    uint8_t* k0 = nullptr;   // w0 * h0 * c0, level 0's palette indices (its grid indices under --l0 bc8)
    uint8_t* k1 = nullptr;   // w1 * h1 * c1, level 1's grid indices
    int w0 = 0, h0 = 0, w1 = 0, h1 = 0;

    // Block (b)'s workspace: the 9-point stencil (five C1 x C1 blocks per texel) and its gradient, the block-Jacobi
    // inverses, the five conjugate-gradient vectors, the plane as the stencil saw it, and the reductions' partials.
    double* stencil = nullptr;   // texels * 5 * C1 * C1
    double* grad = nullptr;      // texels * C1
    float* precond = nullptr;    // texels * C1 * C1
    double* cg_x = nullptr;      // the correction delta
    double* cg_r = nullptr;
    double* cg_z = nullptr;
    double* cg_p = nullptr;
    double* cg_ap = nullptr;
    float* q1_prev = nullptr;    // the plane as it stood when the stencil was assembled
    double* partial = nullptr;   // PARTIAL_ROWS * REDUCE_BLOCKS
    double* scalars = nullptr;   // SC_COUNT
    unsigned int* counters = nullptr;   // level 1's moved count, then block (c)'s four colour passes

    // Block (c')'s workspace under --l0 bc8, and nothing at all otherwise: the same five buffers over level 0's own
    // plane, which has sixteen times the texels of level 1's and therefore cannot borrow them. Block (c)'s palette
    // search allocates none of this, so the palette mode pays nothing for a mode it does not run.
    double* stencil0 = nullptr;   // w0 * h0 * 5 * C0 * C0
    double* grad0 = nullptr;      // w0 * h0 * C0
    float* precond0 = nullptr;
    double* cg0_x = nullptr;
    double* cg0_r = nullptr;
    double* cg0_z = nullptr;
    double* cg0_p = nullptr;
    double* cg0_ap = nullptr;
    float* q0_prev = nullptr;

    // The BC refinement's own state (refine_bc.cu), allocated with block (c')'s workspace and only under --l0 bc8: the
    // two endpoints and the sixteen selectors of every 4x4 block of every channel, which together ARE the shipped
    // level-0 plane, and the two figures a refinement pass reduces into (the decrease it accepted and the blocks that
    // accepted one).
    uint8_t* bc_ep = nullptr;    // blocks * C0 * 2
    uint8_t* bc_sel = nullptr;   // blocks * C0 * 16
    double* bc_stats = nullptr;  // 2

    cudaStream_t stream = nullptr;
    cudaEvent_t ev[EV_COUNT] = {};
};

struct DeviceModel
{
    std::vector<DevPlane> planes;
    float* weights = nullptr;   // nout * nin, row-major
    float* bias = nullptr;      // nout
    float* palette = nullptr;   // MAX_CHANNELS * 16, level 0's palette per channel (2^bits0[c] <= 16 entries)
    float* lo1 = nullptr;       // MAX_CHANNELS
    float* hi1 = nullptr;       // MAX_CHANNELS
    float* lo0 = nullptr;       // MAX_CHANNELS, level 0's own grid under --l0 bc8 (unused in the palette mode)
    float* hi0 = nullptr;       // MAX_CHANNELS
    float* cw = nullptr;        // nout, the per-output weight of the objective
    double* mono = nullptr;     // the stencil's monomial matrices, mono_count(c0) * C1 * C1 (solve_level1.cu)
    double* mono0 = nullptr;    // the same for block (c')'s stencil, whose form is in the level-1 sample instead:
                                //   mono_count(c1) * C0 * C0, allocated only under --l0 bc8
    double* reduction = nullptr;// the least-squares result: one entry per upper-triangle entry of A and per rhs entry
    double* ls_partial = nullptr;   // its partials: planes * entries * SITE_BLOCKS
    double* ls_omega = nullptr;     // the per-plane site weight the final pass applies: planes
    double* obj = nullptr;      // the objective's three sums per plane
    double* obj_partial = nullptr;  // their partials: planes * 3 * SITE_BLOCKS

    // The host's own stream, and what the planes' streams wait for when a step of the round has to see something the
    // master stream produced (and the other way round at the end of a block).
    cudaStream_t master = nullptr;
    cudaEvent_t master_done = nullptr;
    cudaEvent_t block_start = nullptr, block_end = nullptr;   // the span of one block, on the master stream

    // Pinned staging for everything the host reads once per block: a plane's scalar row and its counters.
    double* host_scalars = nullptr;      // planes * SC_COUNT
    unsigned int* host_counters = nullptr;   // planes * 5

    // The pca init's workspace: one block mean of all nout source channels per level-1 texel, and the plane's mean and
    // covariance of those means accumulated in double (nout + nout * nout entries).
    float* means = nullptr;
    double* cov = nullptr;

    // --init0 residual's workspace: one plane's full-resolution residual, nout floats per pixel of the widest plane.
    // Allocated only when the init asks for it, because at four textures and a 2048 x 1024 base it is the same size as
    // the base's source itself.
    float* resid = nullptr;

    float* scratch = nullptr;   // the largest plane's worth of floats, for the continuous level-1 init
    uint8_t* rgb8 = nullptr;    // the largest level's decoded bytes
    size_t scratch_floats = 0;
    size_t rgb8_bytes = 0;
    size_t device_bytes = 0;    // every allocation above, for the report
    int nout = 0, nin = 0, c0 = 0, c1 = 0;
    bool l0_bc8 = false;        // level 0 is a continuous 8-bit plane solved like level 1, not a palette search
    int mono0_count = 0;        // stencil_monomials(c1), the length of mono0 in C0 x C0 matrices
};

// The two problems one stored level carries, as the one description everything below the assembly reads. Level 1 is
// block (b)'s unknown at C1 channels of bits1 on the grid the format carries; level 0 is block (c')'s unknown under
// --l0 bc8, at C0 channels of 8 bits on its own grid, over a plane at the decoded extent.
inline PlaneWork plane_work(const DeviceModel* d, const Model& m, DevPlane& p, bool level0)
{
    PlaneWork k;
    if (level0)
    {
        k.w = p.w0;
        k.h = p.h0;
        k.c = m.c0;
        k.bits = 8;
        k.v = p.v0;
        k.k = p.k0;
        k.lo = d->lo0;
        k.hi = d->hi0;
        k.mono = d->mono0;
        k.nmono = d->mono0_count;
        k.stencil = p.stencil0;
        k.grad = p.grad0;
        k.precond = p.precond0;
        k.cg_x = p.cg0_x;
        k.cg_r = p.cg0_r;
        k.cg_z = p.cg0_z;
        k.cg_p = p.cg0_p;
        k.cg_ap = p.cg0_ap;
        k.prev = p.q0_prev;
    }
    else
    {
        k.w = p.w1;
        k.h = p.h1;
        k.c = m.c1;
        k.bits = m.bits1;
        k.v = p.v1;
        k.k = p.k1;
        k.lo = d->lo1;
        k.hi = d->hi1;
        k.mono = d->mono;
        k.nmono = stencil_monomials(m.c0);
        k.stencil = p.stencil;
        k.grad = p.grad;
        k.precond = p.precond;
        k.cg_x = p.cg_x;
        k.cg_r = p.cg_r;
        k.cg_z = p.cg_z;
        k.cg_p = p.cg_p;
        k.cg_ap = p.cg_ap;
        k.prev = p.q1_prev;
    }
    k.levels = (1 << k.bits) - 1;
    return k;
}

// A step issued on the master stream that every plane's stream must see before its own work (an upload, a memset).
inline void device_fan_out(DeviceModel* d)
{
    CUDA_CHECK(cudaEventRecord(d->master_done, d->master));
    for (DevPlane& p : d->planes)
        CUDA_CHECK(cudaStreamWaitEvent(p.stream, d->master_done, 0));
}

// The end of a block: the master stream waits for every plane, so one synchronisation of it waits for them all.
inline void device_fan_in(DeviceModel* d)
{
    for (DevPlane& p : d->planes)
    {
        CUDA_CHECK(cudaEventRecord(p.ev[EV_DONE], p.stream));
        CUDA_CHECK(cudaStreamWaitEvent(d->master, p.ev[EV_DONE], 0));
    }
}

// One block of the round, timed end to end on the master stream. The planes' own event pairs measure each plane's own
// work, which OVERLAPS: their sum is the work done, the span below is the time it took.
inline void device_block_begin(DeviceModel* d)
{
    CUDA_CHECK(cudaEventRecord(d->block_start, d->master));
    device_fan_out(d);
}

inline double device_block_end(DeviceModel* d)
{
    device_fan_in(d);
    CUDA_CHECK(cudaEventRecord(d->block_end, d->master));
    CUDA_CHECK(cudaStreamSynchronize(d->master));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, d->block_start, d->block_end));
    return (double)ms;
}

// The milliseconds between two of a plane's events, which are only meaningful once the block has been waited for.
inline double device_elapsed(const DevPlane& p, PlaneEvent a, PlaneEvent b)
{
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, p.ev[a], p.ev[b]));
    return (double)ms;
}

// EVERY host-to-device upload the plane streams then read goes through this one pattern: the copy is issued on the
// master stream and the planes are made to wait for it. A plain cudaMemcpy would be synchronous with respect to the
// HOST, which is not the same thing at all - the plane streams are cudaStreamNonBlocking, so they are ordered against
// neither the legacy default stream nor the master one, and a kernel still in flight from the previous round could read
// the new weights, or the next round's kernel could be issued before the copy lands. The host synchronisation after the
// copy is what lets the caller's staging vector go out of scope; it costs one wait per block, not per plane.
inline void device_upload_fan_out(DeviceModel* d, void* dst, const void* src, size_t bytes)
{
    if (bytes == 0)
        return;
    CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, d->master));
    CUDA_CHECK(cudaStreamSynchronize(d->master));
    device_fan_out(d);
}

inline void device_upload_palettes(DeviceModel* d, const Model& m)
{
    // Level 0's palette, which --l0 bc8 does not have: there the plane is continuous on lo0 / hi0 and no kernel reads
    // this buffer, so it stays at the zeros it was uploaded with.
    std::vector<float> pal((size_t)MAX_CHANNELS * 16, 0.0f);
    for (int c = 0; c < m.c0 && c < (int)m.palette0.size(); c++)
        for (size_t k = 0; k < m.palette0[(size_t)c].size() && k < 16; k++)
            pal[(size_t)c * 16 + k] = m.palette0[(size_t)c][k];
    device_upload_fan_out(d, d->palette, pal.data(), pal.size() * sizeof(float));

    std::vector<float> lo(MAX_CHANNELS, 0.0f), hi(MAX_CHANNELS, 0.0f);
    for (int c = 0; c < m.c1 && c < (int)m.lo1.size(); c++)
    {
        lo[c] = m.lo1[c];
        hi[c] = m.hi1[c];
    }
    device_upload_fan_out(d, d->lo1, lo.data(), lo.size() * sizeof(float));
    device_upload_fan_out(d, d->hi1, hi.data(), hi.size() * sizeof(float));

    // Level 0's own grid, which exists only under --l0 bc8 and is empty until the freeze fits it.
    if (d->lo0 && d->hi0)
    {
        std::vector<float> lo0(MAX_CHANNELS, 0.0f), hi0(MAX_CHANNELS, 0.0f);
        for (int c = 0; c < m.c0 && c < (int)m.lo0.size(); c++)
        {
            lo0[c] = m.lo0[c];
            hi0[c] = m.hi0[c];
        }
        device_upload_fan_out(d, d->lo0, lo0.data(), lo0.size() * sizeof(float));
        device_upload_fan_out(d, d->hi0, hi0.data(), hi0.size() * sizeof(float));
    }
}

inline void device_upload_weights(DeviceModel* d, const Model& m)
{
    device_upload_fan_out(d, d->cw, m.cw.data(), m.cw.size() * sizeof(float));
}

inline void device_upload_decoder(DeviceModel* d, const Model& m)
{
    device_upload_fan_out(d, d->weights, m.dec.w.data(), m.dec.w.size() * sizeof(float));
    device_upload_fan_out(d, d->bias, m.dec.b.data(), m.dec.b.size() * sizeof(float));
}
