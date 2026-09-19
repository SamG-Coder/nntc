// backend_check_probe.cu: the per-kernel harness's reads of the CUDA arm's workspaces (see backend_check_probe.h).
//
// Compiled by nvcc because it reads DeviceModel, which device.cuh defines and which only a CUDA translation unit can
// include; it defines no kernel and launches none. Every read first waits for the whole device, so that it sees what
// the seam call before it finished writing whatever stream wrote it: the seam's blocks end on a synchronisation of the
// master stream, and this makes the read independent of that detail.

#include "backend_check_probe.h"

#include "device.cuh"

namespace nntc_probe
{

void normal_equations(DeviceModel* d, const Model& m, std::vector<double>& out)
{
    const int mm = m.dec.nin + 1;
    const size_t entries = (size_t)(mm * (mm + 1) / 2 + m.nout * mm);
    out.assign(entries, 0.0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(out.data(), d->reduction, entries * sizeof(double), cudaMemcpyDeviceToHost));
}

namespace
{

template <class T>
void download(const T* src, size_t n, std::vector<T>& out)
{
    out.assign(n, T());
    if (src && n)
        CUDA_CHECK(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost));
    else
        out.clear();
}

}   // namespace

void workspace(DeviceModel* d, const Model& m, int plane, bool level0, Workspace& w)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    const PlaneWork pw = plane_work(d, m, d->planes[(size_t)plane], level0);
    const size_t texels = pw.texels(), c = (size_t)pw.c;
    download(pw.stencil, texels * STENCIL_BLOCKS * c * c, w.stencil);
    download(pw.grad, texels * c, w.grad);
    download(pw.precond, texels * c * c, w.precond);
    download(pw.cg_x, texels * c, w.cg_x);
    download(pw.prev, texels * c, w.prev);
}

void monomials(DeviceModel* d, const Model& m, bool level0, std::vector<double>& out)
{
    CUDA_CHECK(cudaDeviceSynchronize());
    if (level0)
        download(d->mono0, (size_t)d->mono0_count * m.c0 * m.c0, out);
    else
        download(d->mono, (size_t)stencil_monomials(m.c0) * m.c1 * m.c1, out);
}

}   // namespace nntc_probe
