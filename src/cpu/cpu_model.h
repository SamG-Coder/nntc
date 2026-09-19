// cpu/cpu_model.h: CpuModel, the plain-array twin of DeviceModel.
//
// device.cuh's DeviceModel holds every per-texel buffer of the encode in device memory; CpuModel holds the same buffers
// in host memory, with the same names, the same element types and the same sizes, so that a CPU kernel is a transcription
// of a CUDA kernel with the indexing unchanged, and so that device_memory reports the same number under the same name on
// both backends (docs/CPU_BACKEND_PLAN.md section 1.10). What it does NOT carry is everything that exists only because a
// GPU is asynchronous: the streams, the events, the pinned staging and the fan-out and fan-in machinery. On the CPU a
// call's work is finished when the call returns.
//
// Every buffer is a std::vector, allocated through cpu_model.cpp's one allocator (section 1.11): a failed allocation
// prints an ERROR naming the buffer and the bytes and exits 1, like every other failure in this tree. Vectors are
// value-initialised, so a CPU model starts at zeros where cudaMalloc leaves whatever the allocator handed back; the CUDA
// side is correct only because it writes before it reads, and the CPU side is then correct AND deterministic.
//
// PlaneWork and plane_work are copied from device.cuh (plan section 1.8), with plane_work's stencil_monomials call
// resolving to the CPU backend's own copy - which is the one subtle point of section 1.7: device.cuh calls it
// unqualified at global scope and that global is the CUDA arm's.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "cpu/cpu_constants.h"
#include "cpu/pool.h"
#include "image.h"
#include "model.h"

namespace nntc_cpu
{

int stencil_monomials(int c0);

// The workspace of one sparse least squares over one latent plane (device.cuh's PlaneWork, copied). Raw pointers into
// CpuModel's vectors: a kernel reads the problem through this one description whichever latent is the unknown.
struct PlaneWork
{
    int w = 0, h = 0, c = 0;
    int bits = 8;
    int levels = 255;
    float* v = nullptr;
    uint8_t* k = nullptr;
    const float* lo = nullptr;
    const float* hi = nullptr;
    const double* mono = nullptr;
    int nmono = 0;

    double* stencil = nullptr;
    double* grad = nullptr;
    float* precond = nullptr;
    double* cg_x = nullptr;
    double* cg_r = nullptr;
    double* cg_z = nullptr;
    double* cg_p = nullptr;
    double* cg_ap = nullptr;
    float* prev = nullptr;

    size_t texels() const { return (size_t)w * h; }
    size_t values() const { return texels() * (size_t)c; }
};

// One stored level (device.cuh's DevPlane, less the stream and the events).
struct CpuPlane
{
    std::vector<float> src;       // w0 * h0 * nout, the run's own source chain at this level
    std::vector<float> v0;        // w0 * h0 * c0
    std::vector<float> v1;        // w1 * h1 * c1
    std::vector<uint8_t> k0;      // w0 * h0 * c0
    std::vector<uint8_t> k1;      // w1 * h1 * c1
    int w0 = 0, h0 = 0, w1 = 0, h1 = 0;

    // Block (b)'s workspace.
    std::vector<double> stencil;  // texels * 5 * C1 * C1
    std::vector<double> grad;     // texels * C1
    std::vector<float> precond;   // texels * C1 * C1
    std::vector<double> cg_x;     // the correction delta
    std::vector<double> cg_r;
    std::vector<double> cg_z;
    std::vector<double> cg_p;
    std::vector<double> cg_ap;
    std::vector<float> q1_prev;
    std::vector<double> partial;  // PARTIAL_ROWS * REDUCE_BLOCKS
    std::vector<double> scalars;  // SC_COUNT
    std::vector<unsigned int> counters;   // 5

    // Block (c')'s workspace and the BC refinement's state, under --l0 bc8 only.
    std::vector<double> stencil0;
    std::vector<double> grad0;
    std::vector<float> precond0;
    std::vector<double> cg0_x;
    std::vector<double> cg0_r;
    std::vector<double> cg0_z;
    std::vector<double> cg0_p;
    std::vector<double> cg0_ap;
    std::vector<float> q0_prev;
    std::vector<uint8_t> bc_ep;    // blocks * C0 * 2
    std::vector<uint8_t> bc_sel;   // blocks * C0 * 16
    std::vector<double> bc_stats;  // 2
};

struct CpuModel
{
    std::vector<CpuPlane> planes;
    std::vector<float> weights;     // nout * nin, row-major
    std::vector<float> bias;        // nout
    std::vector<float> palette;     // MAX_CHANNELS * 16
    std::vector<float> lo1, hi1;    // MAX_CHANNELS
    std::vector<float> lo0, hi0;    // MAX_CHANNELS, under --l0 bc8 only
    std::vector<float> cw;          // nout
    std::vector<double> mono;       // stencil_monomials(c0) * C1 * C1
    std::vector<double> mono0;      // stencil_monomials(c1) * C0 * C0, under --l0 bc8 only
    std::vector<double> reduction;
    std::vector<double> ls_partial;
    std::vector<double> ls_omega;
    std::vector<double> obj;
    std::vector<double> obj_partial;
    std::vector<float> means;
    std::vector<double> cov;
    std::vector<float> resid;       // under --init0 residual only
    std::vector<float> scratch;
    std::vector<uint8_t> rgb8;

    size_t scratch_floats = 0;
    size_t rgb8_bytes = 0;
    size_t bytes = 0;               // every allocation above, for the report: device_memory's number
    int nout = 0, nin = 0, c0 = 0, c1 = 0;
    bool l0_bc8 = false;
    int mono0_count = 0;

    // Built once, in device_create, and destroyed with the model: the workers park between runs (cpu/pool.h).
    std::unique_ptr<Pool> pool;
};

// The two problems one stored level carries (device.cuh's plane_work, copied).
inline PlaneWork plane_work(CpuModel* d, const Model& m, CpuPlane& p, bool level0)
{
    PlaneWork k;
    if (level0)
    {
        k.w = p.w0;
        k.h = p.h0;
        k.c = m.c0;
        k.bits = 8;
        k.v = p.v0.data();
        k.k = p.k0.data();
        k.lo = d->lo0.data();
        k.hi = d->hi0.data();
        k.mono = d->mono0.data();
        k.nmono = d->mono0_count;
        k.stencil = p.stencil0.data();
        k.grad = p.grad0.data();
        k.precond = p.precond0.data();
        k.cg_x = p.cg0_x.data();
        k.cg_r = p.cg0_r.data();
        k.cg_z = p.cg0_z.data();
        k.cg_p = p.cg0_p.data();
        k.cg_ap = p.cg0_ap.data();
        k.prev = p.q0_prev.data();
    }
    else
    {
        k.w = p.w1;
        k.h = p.h1;
        k.c = m.c1;
        k.bits = m.bits1;
        k.v = p.v1.data();
        k.k = p.k1.data();
        k.lo = d->lo1.data();
        k.hi = d->hi1.data();
        k.mono = d->mono.data();
        k.nmono = stencil_monomials(m.c0);   // nntc_cpu::stencil_monomials, the CPU backend's own copy
        k.stencil = p.stencil.data();
        k.grad = p.grad.data();
        k.precond = p.precond.data();
        k.cg_x = p.cg_x.data();
        k.cg_r = p.cg_r.data();
        k.cg_z = p.cg_z.data();
        k.cg_p = p.cg_p.data();
        k.cg_ap = p.cg_ap.data();
        k.prev = p.q1_prev.data();
    }
    k.levels = (1 << k.bits) - 1;
    return k;
}

// The BC refinement's block count of one plane (refine_bc.cu's plane_blocks, copied): one 4x4 block per four texels
// per axis, a partial edge block counted whole.
inline size_t plane_blocks(const Model& m, int plane)
{
    const PlaneSize& p = m.planes[(size_t)plane];
    return (size_t)((p.w0 + 3) / 4) * (size_t)((p.h0 + 3) / 4);
}

// The bytes device_create will allocate for this model, computed without allocating anything: the same walk over the
// same buffer list, counting instead of allocating, so the number cannot disagree with device_memory afterwards. It is
// what lets the banner say how much host memory a CPU encode is about to take before it takes it (section 1.11).
size_t bytes_for(const Model& m);

// The lifecycle. `threads` is -j as the dispatcher resolved it.
CpuModel* model_create(const Model& m, const std::vector<Image>& source_chain, int threads);
void model_destroy(CpuModel* d);

// The CPU backend's copy of a range of elements, over the pool in fixed CPU_COPY_CHUNK chunks. Each chunk writes only
// its own slice of dst, so the chunks are disjoint by construction.
template <class T>
void pool_copy(Pool& pool, const T* src, T* dst, size_t n)
{
    pool.run(chunks_for(n, CPU_COPY_CHUNK), [&](int c) {
        const size_t begin = (size_t)c * CPU_COPY_CHUNK;
        const size_t end = begin + CPU_COPY_CHUNK < n ? begin + CPU_COPY_CHUNK : n;
        std::copy(src + begin, src + end, dst + begin);
    });
}

}   // namespace nntc_cpu
