// cpu/cpu_constants.h: device.cuh's CUDA-free half, COPIED.
//
// device.cuh opens with #include <cuda_runtime.h>, so a translation unit built with no CUDA toolkit cannot include it,
// and it is one of the nine files that are not edited (docs/CPU_BACKEND_PLAN.md decision 1, hashed by the gate's
// kernel_hash_check). So what the CPU backend needs from it is copied here, value for value and in the same order:
// the caps, ChannelBits, the fixed-grid constants, the stencil geometry and the scalar-row slots. The copy cannot drift
// from its original without one of two alarms going off - the hash arm if device.cuh moves, and the per-kernel harness
// (src/backend_check.cpp) if a copied value disagrees with what the CUDA side computes with it.
//
// Nothing here is a CUDA idiom: the two stencil functions lose their __host__ __device__ and nothing else.

#pragma once

#include <cstddef>

#include "model.h"

namespace nntc_cpu
{

static const int MAX_CHANNELS = 4;              // the per-level channel cap the shader and the .dds formats share
static const int MAX_NIN = 24;                  // the shader's |phi| cap
static const int MAX_NOUT = 3 * MAX_TEXTURES;   // the shader's output cap: 3 channels of MAX_TEXTURES textures

// The bit count of each level-0 channel, packed small enough to be passed by value.
struct ChannelBits
{
    int b[MAX_CHANNELS];
};

// The CUDA side's fixed reduction grids. The CPU backend does NOT replay them (section 0a of the plan retired that):
// its reductions use their own fixed chunk count, below. They are carried because the device's workspace is sized by
// them and the CPU backend allocates the same buffers, so that device_memory reports the same number under the same
// name on both backends (section 1.10).
static const int REDUCE_BLOCKS = 256;
static const int SITE_BLOCKS = 512;

// The blocks of the 9-point stencil a texel OWNS: its own diagonal and the four directions right, down-left, down and
// down-right. The other four are read as the transpose of the neighbour's block in the same slot, which is what makes
// H symmetric by construction rather than by arithmetic.
static const int STENCIL_BLOCKS = 5;

inline void stencil_offset(int slot, int& dx, int& dy)
{
    const int ox[STENCIL_BLOCKS] = { 0, 1, -1, 0, 1 };
    const int oy[STENCIL_BLOCKS] = { 0, 0, 1, 1, 1 };
    dx = ox[slot];
    dy = oy[slot];
}

// The slot a texel offset belongs to, or -1 when the offset is owned by the other texel of the pair.
inline int stencil_slot(int dx, int dy)
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

// The rows of a plane's partial buffer: three for the stencil diagonal (its sum, smallest and largest), three for the
// movement probe (the count, the correction's square norm and the plane's), and one per channel for each end of the
// plane's range.
static const int PARTIAL_ROWS = 3 + 2 * MAX_CHANNELS;

static const int ROW_GENERAL = 0;
static const int ROW_LO = 3;
static const int ROW_HI = 3 + MAX_CHANNELS;

// A plane's scalars, in device.cuh's slot order.
enum PlaneScalar
{
    SC_DIAG_SUM = 0,
    SC_DIAG_MIN,
    SC_DIAG_MAX,
    SC_LAMBDA,
    SC_RHS2,
    SC_RZ,
    SC_RZ_NEW,
    SC_PAP,
    SC_RR,
    SC_RESID,
    SC_ALPHA,
    SC_BETA,
    SC_MOVED,
    SC_DELTA2,
    SC_PLANE2,
    SC_LO,
    SC_HI = SC_LO + MAX_CHANNELS,
    SC_COUNT = SC_HI + MAX_CHANNELS
};

// ---------------------------------------------------------------------------------------------------------------------
// The CPU backend's own constants. These are NOT copies: they have no CUDA counterpart.
// ---------------------------------------------------------------------------------------------------------------------

// The elements one chunk of a one-thread-per-element kernel covers. It is a compile-time constant and never a function
// of the thread count (the pool's rule 1, cpu/pool.h), so chunk c covers the same elements at every -j on every run,
// and a reduction folded over chunks in index order gives the same bits at every -j.
static const size_t CPU_CHUNK = 4096;

// The same for a plain copy, which does a byte's worth of work per element and wants a larger chunk before the wake of
// a worker is worth its cost. Also a constant, for the same reason.
static const size_t CPU_COPY_CHUNK = 65536;

// The chunk count of n elements at `per` elements a chunk: a function of the size alone. Zero elements is zero chunks,
// which Pool::run treats as nothing to do.
inline int chunks_for(size_t n, size_t per)
{
    return (int)((n + per - 1) / per);
}

}   // namespace nntc_cpu
