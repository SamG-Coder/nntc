// cpu/cpu_model.cpp: CpuModel's lifecycle, its memory accounting and its one allocator, and the transfer functions.
//
// THE BUFFER LIST IS init.cu's device_create, buffer for buffer. That is deliberate and it is what makes device_memory
// the same number on both backends (docs/CPU_BACKEND_PLAN.md section 1.10): the report's `device ... MB` line is a fact
// a user needs whichever backend is holding the encode. It is walked by ONE function, layout(), which either counts or
// allocates, so bytes_for (the banner's figure, before anything is allocated) and device_memory (the report's, after)
// cannot disagree. The per-kernel harness holds the two backends' figures equal on every layout it runs.
//
// THE TRANSFERS are the seam's downloads and uploads: on the CPU a download is a copy out of the model and an upload a
// copy into it. Each one reproduces its CUDA twin's contract exactly, including the quiet ones - an upload whose size is
// not the plane's is ignored, as the CUDA side ignores it, and a download resizes its vector to the plane. They are
// also the harness's instrument (section 4.1): the harness puts identical inputs into both backends and reads both
// outputs back through them, which is why they are written, and checked, before any kernel.
//
// The copies go through the pool in fixed CPU_COPY_CHUNK chunks, each writing only its own slice of the destination,
// so a copy is threaded without any two threads touching one element.

#include "cpu/cpu_backend.h"
#include "cpu/cpu_model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <string>

namespace nntc_cpu
{

// solve_level1.cu's stencil_monomials, copied (plan section 1.8): the count of monomials the stencil's quadratic form
// in the level-0 sample has - the constant, the C0 linear terms and the C0 (C0 + 1) / 2 quadratic ones. It lives here
// rather than beside the assembly because the allocation needs it first; the harness asserts it equals the CUDA
// arm's for every channel count at start-up.
int stencil_monomials(int c0)
{
    return 1 + c0 + c0 * (c0 + 1) / 2;
}

namespace
{

// Every allocation of the CPU backend goes through here (section 1.11). A std::vector that cannot be sized throws, and
// main's catch would say only which stage ran out; here the failure reads like every other one in the tree, names what
// could not be had and how much, and exits 1.
[[noreturn]] void allocation_failed(size_t bytes, const std::string& what)
{
    std::fprintf(stderr, "ERROR: cannot allocate %zu bytes for %s: no encode\n", bytes, what.c_str());
    std::exit(EXIT_FAILURE);
}

// Counts every buffer, and allocates it too when `allocate` is set. One visitor, so the count and the allocation are
// the same walk.
struct Layout
{
    bool allocate = false;
    size_t bytes = 0;

    template <class T>
    void operator()(std::vector<T>& buf, size_t n, const std::string& what)
    {
        bytes += n * sizeof(T);
        if (!allocate)
            return;
        try
        {
            buf.assign(n, T());
        }
        catch (const std::bad_alloc&)
        {
            allocation_failed(n * sizeof(T), what);
        }
        catch (const std::length_error&)
        {
            allocation_failed(n * sizeof(T), what);
        }
    }
};

std::string plane_name(size_t i, const char* what)
{
    return "plane " + std::to_string(i) + "'s " + what;
}

// init.cu's device_create, as a list. Every device_alloc there is one call here, with the same count and the same
// element type; the streams, the events and the pinned host staging have no counterpart, and the pinned buffers are not
// in the CUDA figure either (they are cudaMallocHost, not device_alloc).
void layout(const Model& m, CpuModel& d, Layout& L)
{
    const size_t bsq = (size_t)m.c1 * m.c1;
    size_t scratch = 0, rgb8 = 0, texels1 = 0;
    d.planes.resize(m.planes.size());
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        const PlaneSize& p = m.planes[i];
        CpuPlane& dp = d.planes[i];
        dp.w0 = p.w0;
        dp.h0 = p.h0;
        dp.w1 = p.w1;
        dp.h1 = p.h1;
        const size_t n0 = (size_t)p.w0 * p.h0, n1 = (size_t)p.w1 * p.h1;
        L(dp.src, n0 * m.nout, plane_name(i, "source"));
        L(dp.v0, n0 * m.c0, plane_name(i, "level-0 values"));
        L(dp.v1, n1 * m.c1, plane_name(i, "level-1 values"));
        L(dp.k0, n0 * m.c0, plane_name(i, "level-0 indices"));
        L(dp.k1, n1 * m.c1, plane_name(i, "level-1 indices"));

        L(dp.stencil, n1 * STENCIL_BLOCKS * bsq, plane_name(i, "level-1 stencil"));
        L(dp.grad, n1 * m.c1, plane_name(i, "level-1 gradient"));
        L(dp.precond, n1 * bsq, plane_name(i, "level-1 preconditioner"));
        L(dp.cg_x, n1 * m.c1, plane_name(i, "level-1 correction"));
        L(dp.cg_r, n1 * m.c1, plane_name(i, "level-1 CG residual"));
        L(dp.cg_z, n1 * m.c1, plane_name(i, "level-1 CG z"));
        L(dp.cg_p, n1 * m.c1, plane_name(i, "level-1 CG direction"));
        L(dp.cg_ap, n1 * m.c1, plane_name(i, "level-1 CG product"));
        L(dp.q1_prev, n1 * m.c1, plane_name(i, "level-1 previous plane"));

        if (m.l0_bc8)
        {
            const size_t b0 = (size_t)m.c0 * m.c0;
            L(dp.stencil0, n0 * STENCIL_BLOCKS * b0, plane_name(i, "level-0 stencil"));
            L(dp.grad0, n0 * m.c0, plane_name(i, "level-0 gradient"));
            L(dp.precond0, n0 * b0, plane_name(i, "level-0 preconditioner"));
            L(dp.cg0_x, n0 * m.c0, plane_name(i, "level-0 correction"));
            L(dp.cg0_r, n0 * m.c0, plane_name(i, "level-0 CG residual"));
            L(dp.cg0_z, n0 * m.c0, plane_name(i, "level-0 CG z"));
            L(dp.cg0_p, n0 * m.c0, plane_name(i, "level-0 CG direction"));
            L(dp.cg0_ap, n0 * m.c0, plane_name(i, "level-0 CG product"));
            L(dp.q0_prev, n0 * m.c0, plane_name(i, "level-0 previous plane"));
            const size_t blocks = (size_t)((p.w0 + 3) / 4) * (size_t)((p.h0 + 3) / 4);
            L(dp.bc_ep, blocks * (size_t)m.c0 * 2, plane_name(i, "BC endpoints"));
            L(dp.bc_sel, blocks * (size_t)m.c0 * 16, plane_name(i, "BC selectors"));
            L(dp.bc_stats, 2, plane_name(i, "BC refinement figures"));
        }
        L(dp.partial, (size_t)PARTIAL_ROWS * REDUCE_BLOCKS, plane_name(i, "reduction partials"));
        L(dp.scalars, (size_t)SC_COUNT, plane_name(i, "scalar row"));
        L(dp.counters, 5, plane_name(i, "counters"));

        scratch = std::max(scratch, n1 * m.c1);
        rgb8 = std::max(rgb8, n0 * m.nout);
        texels1 = std::max(texels1, n1);
    }
    L(d.weights, (size_t)m.dec.nin * m.nout, "the decoder's weights");
    L(d.bias, (size_t)m.nout, "the decoder's bias");
    L(d.palette, (size_t)MAX_CHANNELS * 16, "level 0's palette");
    L(d.lo1, (size_t)MAX_CHANNELS, "level 1's grid");
    L(d.hi1, (size_t)MAX_CHANNELS, "level 1's grid");
    if (m.l0_bc8)
    {
        L(d.lo0, (size_t)MAX_CHANNELS, "level 0's grid");
        L(d.hi0, (size_t)MAX_CHANNELS, "level 0's grid");
        L(d.mono0, (size_t)stencil_monomials(m.c1) * m.c0 * m.c0, "block (c')'s monomial matrices");
    }
    L(d.cw, (size_t)m.nout, "the objective's output weights");
    L(d.mono, (size_t)stencil_monomials(m.c0) * bsq, "block (b)'s monomial matrices");
    const int mm = m.dec.nin + 1;
    const size_t entries = (size_t)(mm * (mm + 1) / 2 + m.nout * mm);
    L(d.reduction, entries, "the normal equations");
    L(d.ls_partial, m.planes.size() * entries * (size_t)SITE_BLOCKS, "the normal equations' partials");
    L(d.ls_omega, m.planes.size(), "the per-plane site weights");
    L(d.obj, 3 * m.planes.size(), "the objective's sums");
    L(d.obj_partial, 3 * m.planes.size() * (size_t)SITE_BLOCKS, "the objective's partials");
    L(d.scratch, scratch, "the level-1 init's scratch");
    L(d.rgb8, rgb8, "the decoded bytes");
    L(d.means, texels1 * (size_t)m.nout, "the pca init's block means");
    L(d.cov, (size_t)(m.nout + m.nout * m.nout) * (size_t)SITE_BLOCKS, "the pca init's covariance");
    if (m.init0_residual)
        L(d.resid, rgb8, "the residual");
    d.scratch_floats = scratch;
    d.rgb8_bytes = rgb8;
}

// Refusals for the handful of calls whose CUDA twin would write past a buffer or copy from one that does not exist.
// The CUDA side ends such a call with a CUDA error and status 1; this ends it with a sentence and status 1.
[[noreturn]] void transfer_refused(const char* what)
{
    std::fprintf(stderr, "ERROR: the CPU backend's %s: no encode\n", what);
    std::exit(EXIT_FAILURE);
}

// A download: the vector becomes the plane's buffer, element for element.
template <class T>
void download(CpuModel* d, const std::vector<T>& buf, std::vector<T>& out)
{
    out.resize(buf.size());
    pool_copy(*d->pool, buf.data(), out.data(), buf.size());
}

// An upload of a whole plane: a vector of any other size is ignored, which is exactly what the CUDA twins do.
template <class T>
void upload(CpuModel* d, const std::vector<T>& in, std::vector<T>& buf)
{
    if (in.size() != buf.size())
        return;
    pool_copy(*d->pool, in.data(), buf.data(), in.size());
}

}   // namespace

size_t bytes_for(const Model& m)
{
    CpuModel scratch;
    Layout L;
    layout(m, scratch, L);
    return L.bytes;
}

CpuModel* model_create(const Model& m, const std::vector<Image>& source_chain, int threads)
{
    CpuModel* d = new CpuModel();
    d->nout = m.nout;
    d->nin = m.dec.nin;
    d->c0 = m.c0;
    d->c1 = m.c1;
    d->l0_bc8 = m.l0_bc8;
    d->mono0_count = stencil_monomials(m.c1);
    d->pool.reset(new Pool(threads));

    Layout L;
    L.allocate = true;
    layout(m, *d, L);
    d->bytes = L.bytes;

    // The source chain, the one upload the CUDA side makes before the planes' streams exist; here it is a copy like
    // any other. The decoder starts at the zeros the allocation left, which is the state the CUDA side memsets it to.
    for (size_t i = 0; i < m.planes.size(); i++)
    {
        CpuPlane& p = d->planes[i];
        if (source_chain[i].v.size() != p.src.size())
            transfer_refused("source upload was handed a chain level of the wrong size");
        pool_copy(*d->pool, source_chain[i].v.data(), p.src.data(), p.src.size());
    }
    if (m.cw.size() > d->cw.size())
        transfer_refused("output-weight upload was handed more weights than the model has outputs");
    std::copy(m.cw.begin(), m.cw.end(), d->cw.begin());
    return d;
}

void model_destroy(CpuModel* d)
{
    delete d;
}

// ---------------------------------------------------------------------------------------------------------------------
// The seam's lifecycle functions over the model
// ---------------------------------------------------------------------------------------------------------------------

CpuModel* device_create(const Model& m, const std::vector<Image>& source_chain, int threads)
{
    return model_create(m, source_chain, threads);
}

void device_destroy(CpuModel* d)
{
    model_destroy(d);
}

size_t device_memory(const CpuModel* d)
{
    return d ? d->bytes : 0;
}

// ---------------------------------------------------------------------------------------------------------------------
// The transfers
// ---------------------------------------------------------------------------------------------------------------------

void level1_download(CpuModel* d, const Model&, int plane, std::vector<float>& v)
{
    download(d, d->planes[(size_t)plane].v1, v);
}

void level1_download_indices(CpuModel* d, const Model&, int plane, std::vector<uint8_t>& k1)
{
    download(d, d->planes[(size_t)plane].k1, k1);
}

void level1_upload(CpuModel* d, const Model&, int plane, const std::vector<float>& v1)
{
    upload(d, v1, d->planes[(size_t)plane].v1);
}

void level1_upload_indices(CpuModel* d, const Model&, int plane, const std::vector<uint8_t>& k1)
{
    upload(d, k1, d->planes[(size_t)plane].k1);
}

void level0_upload_indices(CpuModel* d, const Model&, int plane, const std::vector<uint8_t>& k0)
{
    upload(d, k0, d->planes[(size_t)plane].k0);
}

void level0_upload(CpuModel* d, const Model&, int plane, const std::vector<float>& v0)
{
    upload(d, v0, d->planes[(size_t)plane].v0);
}

void level0_download_values(CpuModel* d, const Model&, int plane, std::vector<float>& v0)
{
    download(d, d->planes[(size_t)plane].v0, v0);
}

void level0_download(CpuModel* d, const Model&, int plane, std::vector<uint8_t>& k0)
{
    download(d, d->planes[(size_t)plane].k0, k0);
}

// Level 1's correction, the continuous solve's whole output (and the dense twin's comparand).
void level1_delta(CpuModel* d, const Model&, int plane, std::vector<double>& delta)
{
    download(d, d->planes[(size_t)plane].cg_x, delta);
}

// One value of one plane, for the finite-difference probe. The CUDA twin does not bound the index; a CPU write past
// the plane would be undefined behaviour rather than a CUDA error, so it is refused instead.
void level1_poke(CpuModel* d, int plane, size_t index, float value)
{
    std::vector<float>& v1 = d->planes[(size_t)plane].v1;
    if (index >= v1.size())
        transfer_refused("finite-difference probe poked past the end of a level-1 plane");
    v1[index] = value;
}

// The plane's per-channel range, as solve_level1.cu computes it on the host after a download: start at +-1e30 and keep
// v wherever v < lo (v > hi). Here it is a reduction under the pool's rule 4 - each chunk the same comparisons over its
// own elements, folded in chunk order - and it gives the serial loop's answer exactly: a strict comparison keeps the
// FIRST extreme in element order, the fold keeps the earliest chunk's on a tie, and a NaN is skipped by both.
void level1_range(CpuModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi)
{
    struct Range
    {
        float lo[MAX_CHANNELS], hi[MAX_CHANNELS];
    };
    Range init;
    for (int c = 0; c < MAX_CHANNELS; c++)
    {
        init.lo[c] = 1e30f;
        init.hi[c] = -1e30f;
    }
    const std::vector<float>& v = d->planes[(size_t)plane].v1;
    const size_t n = v.size();
    const int c1 = m.c1;
    const Range r = fold_chunks(
        *d->pool, chunks_for(n, CPU_CHUNK), init,
        [&](int c) {
            Range part = init;
            const size_t begin = (size_t)c * CPU_CHUNK;
            const size_t end = begin + CPU_CHUNK < n ? begin + CPU_CHUNK : n;
            for (size_t i = begin; i < end; i++)
            {
                const int ch = (int)(i % (size_t)c1);
                part.lo[ch] = v[i] < part.lo[ch] ? v[i] : part.lo[ch];
                part.hi[ch] = v[i] > part.hi[ch] ? v[i] : part.hi[ch];
            }
            return part;
        },
        [&](Range acc, const Range& part) {
            for (int ch = 0; ch < c1; ch++)
            {
                acc.lo[ch] = part.lo[ch] < acc.lo[ch] ? part.lo[ch] : acc.lo[ch];
                acc.hi[ch] = part.hi[ch] > acc.hi[ch] ? part.hi[ch] : acc.hi[ch];
            }
            return acc;
        });
    lo.assign((size_t)c1, 1e30f);
    hi.assign((size_t)c1, -1e30f);
    for (int ch = 0; ch < c1; ch++)
    {
        lo[(size_t)ch] = r.lo[ch];
        hi[(size_t)ch] = r.hi[ch];
    }
}

// The decoder written back (device.cuh's device_upload_decoder): as many floats as the model carries. More than the
// buffer holds would be a write past it on the CUDA side, and is refused here.
void upload_decoder_to_device(CpuModel* d, const Model& m)
{
    if (m.dec.w.size() > d->weights.size() || m.dec.b.size() > d->bias.size())
        transfer_refused("decoder upload was handed more weights than the model was built for");
    std::copy(m.dec.w.begin(), m.dec.w.end(), d->weights.begin());
    std::copy(m.dec.b.begin(), m.dec.b.end(), d->bias.begin());
}

// Both latents' grids written back (device.cuh's device_upload_palettes), with its zero padding: the palette is
// MAX_CHANNELS rows of 16, the unused entries zero, and level 0's own grid exists only under --l0 bc8.
void upload_grids_to_device(CpuModel* d, const Model& m)
{
    std::fill(d->palette.begin(), d->palette.end(), 0.0f);
    for (int c = 0; c < m.c0 && c < (int)m.palette0.size(); c++)
        for (size_t k = 0; k < m.palette0[(size_t)c].size() && k < 16; k++)
            d->palette[(size_t)c * 16 + k] = m.palette0[(size_t)c][k];

    std::fill(d->lo1.begin(), d->lo1.end(), 0.0f);
    std::fill(d->hi1.begin(), d->hi1.end(), 0.0f);
    for (int c = 0; c < m.c1 && c < (int)m.lo1.size(); c++)
    {
        d->lo1[(size_t)c] = m.lo1[(size_t)c];
        d->hi1[(size_t)c] = m.hi1[(size_t)c];
    }

    if (!d->lo0.empty() && !d->hi0.empty())
    {
        std::fill(d->lo0.begin(), d->lo0.end(), 0.0f);
        std::fill(d->hi0.begin(), d->hi0.end(), 0.0f);
        for (int c = 0; c < m.c0 && c < (int)m.lo0.size(); c++)
        {
            d->lo0[(size_t)c] = m.lo0[(size_t)c];
            d->hi0[(size_t)c] = m.hi0[(size_t)c];
        }
    }
}

// The BC refinement's state saved and put back (refine_bc.cu's bc_state_save / bc_state_restore). They exist only
// under --l0 bc8; the CUDA twins would copy from a null buffer outside it and end on a CUDA error, so the CPU side
// refuses in words instead.
void bc_state_save(const CpuModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel)
{
    const size_t np = m.planes.size();
    ep.assign(np, std::vector<uint8_t>());
    sel.assign(np, std::vector<uint8_t>());
    for (size_t i = 0; i < np; i++)
    {
        const CpuPlane& p = d->planes[i];
        const size_t blocks = plane_blocks(m, (int)i);
        ep[i].assign(blocks * (size_t)m.c0 * 2, 0);
        sel[i].assign(blocks * (size_t)m.c0 * 16, 0);
        if (p.bc_ep.size() < ep[i].size() || p.bc_sel.size() < sel[i].size())
            transfer_refused("BC state save was called on a model with no BC state (it exists under --l0 bc8 only)");
        pool_copy(*d->pool, p.bc_ep.data(), ep[i].data(), ep[i].size());
        pool_copy(*d->pool, p.bc_sel.data(), sel[i].data(), sel[i].size());
    }
}

void bc_state_restore(CpuModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel)
{
    const size_t np = m.planes.size();
    if (ep.size() != np || sel.size() != np)
        return;
    for (size_t i = 0; i < np; i++)
    {
        CpuPlane& p = d->planes[i];
        if (ep[i].size() > p.bc_ep.size() || sel[i].size() > p.bc_sel.size())
            transfer_refused("BC state restore was handed more blocks than the plane holds");
        pool_copy(*d->pool, ep[i].data(), p.bc_ep.data(), ep[i].size());
        pool_copy(*d->pool, sel[i].data(), p.bc_sel.data(), sel[i].size());
    }
}

}   // namespace nntc_cpu
