// backend.cpp: pure routing.
//
// It knows that the arms exist and nothing whatever about either of them: no CUDA header, no CPU header beyond the one
// that declares the twins, no state that is not the choice itself. Every body below has the same shape - unwrap the
// handle, call this arm or that one - and the shape is deliberately mechanical, because a reader checking that 45
// forwarders are right should be reading 45 identical things.
//
// The #ifdef NNTC_CUDA pairs live here and in the CMake source list, and nowhere else in the host code.
//
// --backend check is a third arm in this file's eyes and nothing more: a branch per forwarder that hands the WHOLE handle
// to the per-kernel harness (backend_check.cpp), which holds both arms' models and runs both. It exists only where both
// arms are built, so its branches sit inside the same #ifdef NNTC_CUDA as the CUDA arm's.

#include "backend.h"

#ifdef NNTC_CUDA
#include "backend_check.h"
#include "cuda_backend.h"
#endif
#include "cpu/cpu_backend.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace be
{

// Six of the seam's functions - device_select, device_create, stencil_monomials, decoder_ridge_tally,
// objective_total_ms and objective_passes - take no handle, so the choice has to live somewhere. It lives here, it is
// written once from the parsed command line before anything touches a device, and it is read exactly six times a run.
// Everything else routes on d->backend, which is the copy device_create put in the handle.
static Backend g_backend = Backend::Cuda;
static int g_threads = 0;

void backend_select(Backend b, int threads)
{
    g_backend = b;
    g_threads = threads;
}

Backend backend_selected()
{
    return g_backend;
}

// 0 means "as many as the machine has", resolved here rather than at parse time so that the number in the report is
// the number the pool will actually be built with. A hardware_concurrency() that cannot answer reports 0, and one
// worker is the honest reading of that; the upper clamp is there because a thread count is a typed number.
int backend_threads()
{
    int t = g_threads;
    if (t <= 0)
        t = (int)std::thread::hardware_concurrency();
    if (t < 1)
        t = 1;
    if (t > 1024)
        t = 1024;
    return t;
}

const char* backend_name(Backend b)
{
    return b == Backend::Cuda ? "cuda" : (b == Backend::Cpu ? "cpu" : "check");
}

size_t cpu_bytes_for(const Model& m)
{
    return nntc_cpu::bytes_for(m);
}

// The harness's two entry points outside the seam. In a build without CUDA there is no harness - --backend check is
// refused before a device exists - so the options have nowhere to go and the finish has nothing to report.
void check_options(const std::string& only, bool stop_at_mismatch, bool quiet)
{
#ifdef NNTC_CUDA
    nntc_check::options(only, stop_at_mismatch, quiet);
#else
    (void)only;
    (void)stop_at_mismatch;
    (void)quiet;
#endif
}

bool check_finish()
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Check)
        return nntc_check::finish();
#endif
    return true;
}

bool cuda_built_in()
{
#ifdef NNTC_CUDA
    return true;
#else
    return false;
#endif
}

// The two lifecycle functions are the only ones that are not a plain forward: this one allocates the handle the host
// carries and fills whichever arm's pointer (both, under check), and its partner frees both. Everything between them
// reads d->backend.
Device* device_create(const Model& m, const std::vector<Image>& source_chain)
{
    Device* d = new Device();
    d->backend = g_backend;
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda)
    {
        d->cuda = nntc_cuda::device_create(m, source_chain);
        return d;
    }
    if (d->backend == Backend::Check)
    {
        nntc_check::device_create(d, m, source_chain, backend_threads());
        return d;
    }
#endif
    d->cpu = nntc_cpu::device_create(m, source_chain, backend_threads());
    return d;
}

// The host destroys the device on several paths, including the early returns of the write failures, so a second call
// on a handle already freed would be a real possibility if this did not simply ignore a null.
void device_destroy(Device* d)
{
    if (!d)
        return;
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda)
        nntc_cuda::device_destroy(d->cuda);
    if (d->backend == Backend::Check)
        nntc_check::device_destroy(d);
#endif
    if (d->backend == Backend::Cpu)
        nntc_cpu::device_destroy(d->cpu);
    delete d;
}

bool device_select(int index, DeviceInfo& info)
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Cuda) { return nntc_cuda::device_select(index, info); }
    if (g_backend == Backend::Check) { return nntc_check::device_select(index, info); }
#endif
    return nntc_cpu::device_select(index, info);
}

size_t device_memory(const Device* d)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::device_memory(d->cuda); }
    if (d->backend == Backend::Check) { return nntc_check::device_memory(d); }
#endif
    return nntc_cpu::device_memory(d->cpu);
}

void init_level0(Device* d, Model& m, const float* luma_weights, bool luma_channel0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::init_level0(d->cuda, m, luma_weights, luma_channel0); return; }
    if (d->backend == Backend::Check) { nntc_check::init_level0(d, m, luma_weights, luma_channel0); return; }
#endif
    nntc_cpu::init_level0(d->cpu, m, luma_weights, luma_channel0);
}

void init_level1(Device* d, Model& m, bool pca, Level1InitReport& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::init_level1(d->cuda, m, pca, rep); return; }
    if (d->backend == Backend::Check) { nntc_check::init_level1(d, m, pca, rep); return; }
#endif
    nntc_cpu::init_level1(d->cpu, m, pca, rep);
}

void init0_residual_channel(Device* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda)
    {
        nntc_cuda::init0_residual_channel(d->cuda, m, channel, plane_weight, rep); return;
    }
    if (d->backend == Backend::Check) { nntc_check::init0_residual_channel(d, m, channel, plane_weight, rep); return; }
#endif
    nntc_cpu::init0_residual_channel(d->cpu, m, channel, plane_weight, rep);
}

void quantise_level1(Device* d, Model& m, const std::string& range_policy)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::quantise_level1(d->cuda, m, range_policy); return; }
    if (d->backend == Backend::Check) { nntc_check::quantise_level1(d, m, range_policy); return; }
#endif
    nntc_cpu::quantise_level1(d->cpu, m, range_policy);
}

void freeze_level1_grid(Device* d, Model& m, const std::string& range_policy)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::freeze_level1_grid(d->cuda, m, range_policy); return; }
    if (d->backend == Backend::Check) { nntc_check::freeze_level1_grid(d, m, range_policy); return; }
#endif
    nntc_cpu::freeze_level1_grid(d->cpu, m, range_policy);
}

void quantise_level0(Device* d, Model& m, const std::string& range_policy)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::quantise_level0(d->cuda, m, range_policy); return; }
    if (d->backend == Backend::Check) { nntc_check::quantise_level0(d, m, range_policy); return; }
#endif
    nntc_cpu::quantise_level0(d->cpu, m, range_policy);
}

void freeze_level0_grid(Device* d, Model& m, const std::string& range_policy)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::freeze_level0_grid(d->cuda, m, range_policy); return; }
    if (d->backend == Backend::Check) { nntc_check::freeze_level0_grid(d, m, range_policy); return; }
#endif
    nntc_cpu::freeze_level0_grid(d->cpu, m, range_policy);
}

void snap_level0_on_grid(Device* d, Model& m)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::snap_level0_on_grid(d->cuda, m); return; }
    if (d->backend == Backend::Check) { nntc_check::snap_level0_on_grid(d, m); return; }
#endif
    nntc_cpu::snap_level0_on_grid(d->cpu, m);
}

int stencil_monomials(int c0)
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Cuda) { return nntc_cuda::stencil_monomials(c0); }
    if (g_backend == Backend::Check) { return nntc_check::stencil_monomials(c0); }
#endif
    return nntc_cpu::stencil_monomials(c0);
}

double solve_decoder(Device* d, Model& m, int k, const std::vector<double>& per_site)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::solve_decoder(d->cuda, m, k, per_site); }
    if (d->backend == Backend::Check) { return nntc_check::solve_decoder(d, m, k, per_site); }
#endif
    return nntc_cpu::solve_decoder(d->cpu, m, k, per_site);
}

void decoder_ridge_tally(long long out[4])
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Cuda) { nntc_cuda::decoder_ridge_tally(out); return; }
    if (g_backend == Backend::Check) { nntc_check::decoder_ridge_tally(out); return; }
#endif
    nntc_cpu::decoder_ridge_tally(out);
}

double solve_level1_all(Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::solve_level1_all(d->cuda, m, k, ridge, rep); }
    if (d->backend == Backend::Check) { return nntc_check::solve_level1_all(d, m, k, ridge, rep); }
#endif
    return nntc_cpu::solve_level1_all(d->cpu, m, k, ridge, rep);
}

double sweep_level1_all(Device* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::sweep_level1_all(d->cuda, m, k, ridge, sweeps, rep); }
    if (d->backend == Backend::Check) { return nntc_check::sweep_level1_all(d, m, k, ridge, sweeps, rep); }
#endif
    return nntc_cpu::sweep_level1_all(d->cpu, m, k, ridge, sweeps, rep);
}

double solve_level0_cont_all(Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::solve_level0_cont_all(d->cuda, m, k, ridge, rep); }
    if (d->backend == Backend::Check) { return nntc_check::solve_level0_cont_all(d, m, k, ridge, rep); }
#endif
    return nntc_cpu::solve_level0_cont_all(d->cpu, m, k, ridge, rep);
}

double sweep_level0_cont_all(Device* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::sweep_level0_cont_all(d->cuda, m, k, ridge, sweeps, rep); }
    if (d->backend == Backend::Check) { return nntc_check::sweep_level0_cont_all(d, m, k, ridge, sweeps, rep); }
#endif
    return nntc_cpu::sweep_level0_cont_all(d->cpu, m, k, ridge, sweeps, rep);
}

double assemble_level0_for_refine(Device* d, const Model& m, int k)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::assemble_level0_for_refine(d->cuda, m, k); }
    if (d->backend == Backend::Check) { return nntc_check::assemble_level0_for_refine(d, m, k); }
#endif
    return nntc_cpu::assemble_level0_for_refine(d->cpu, m, k);
}

void level1_download_indices(Device* d, const Model& m, int plane, std::vector<uint8_t>& k1)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_download_indices(d->cuda, m, plane, k1); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_download_indices(d, m, plane, k1); return; }
#endif
    nntc_cpu::level1_download_indices(d->cpu, m, plane, k1);
}

void level1_upload(Device* d, const Model& m, int plane, const std::vector<float>& v1)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_upload(d->cuda, m, plane, v1); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_upload(d, m, plane, v1); return; }
#endif
    nntc_cpu::level1_upload(d->cpu, m, plane, v1);
}

void level1_upload_indices(Device* d, const Model& m, int plane, const std::vector<uint8_t>& k1)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_upload_indices(d->cuda, m, plane, k1); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_upload_indices(d, m, plane, k1); return; }
#endif
    nntc_cpu::level1_upload_indices(d->cpu, m, plane, k1);
}

void level0_upload_indices(Device* d, const Model& m, int plane, const std::vector<uint8_t>& k0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level0_upload_indices(d->cuda, m, plane, k0); return; }
    if (d->backend == Backend::Check) { nntc_check::level0_upload_indices(d, m, plane, k0); return; }
#endif
    nntc_cpu::level0_upload_indices(d->cpu, m, plane, k0);
}

void upload_decoder_to_device(Device* d, const Model& m)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::upload_decoder_to_device(d->cuda, m); return; }
    if (d->backend == Backend::Check) { nntc_check::upload_decoder_to_device(d, m); return; }
#endif
    nntc_cpu::upload_decoder_to_device(d->cpu, m);
}

void upload_grids_to_device(Device* d, const Model& m)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::upload_grids_to_device(d->cuda, m); return; }
    if (d->backend == Backend::Check) { nntc_check::upload_grids_to_device(d, m); return; }
#endif
    nntc_cpu::upload_grids_to_device(d->cpu, m);
}

void level1_download(Device* d, const Model& m, int plane, std::vector<float>& v)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_download(d->cuda, m, plane, v); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_download(d, m, plane, v); return; }
#endif
    nntc_cpu::level1_download(d->cpu, m, plane, v);
}

void level1_delta(Device* d, const Model& m, int plane, std::vector<double>& delta)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_delta(d->cuda, m, plane, delta); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_delta(d, m, plane, delta); return; }
#endif
    nntc_cpu::level1_delta(d->cpu, m, plane, delta);
}

void level1_poke(Device* d, int plane, size_t index, float value)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_poke(d->cuda, plane, index, value); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_poke(d, plane, index, value); return; }
#endif
    nntc_cpu::level1_poke(d->cpu, plane, index, value);
}

void level1_range(Device* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level1_range(d->cuda, m, plane, lo, hi); return; }
    if (d->backend == Backend::Check) { nntc_check::level1_range(d, m, plane, lo, hi); return; }
#endif
    nntc_cpu::level1_range(d->cpu, m, plane, lo, hi);
}

bool solve_level1_dense_host(Device* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda)
    {
        return nntc_cuda::solve_level1_dense_host(d->cuda, m, k, plane, lambda, c_prev, delta, max_unknowns);
    }
    if (d->backend == Backend::Check)
    {
        return nntc_check::solve_level1_dense_host(d, m, k, plane, lambda, c_prev, delta, max_unknowns);
    }
#endif
    return nntc_cpu::solve_level1_dense_host(d->cpu, m, k, plane, lambda, c_prev, delta, max_unknowns);
}

double solve_level0_all(Device* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::solve_level0_all(d->cuda, m, k, sweeps, rep); }
    if (d->backend == Backend::Check) { return nntc_check::solve_level0_all(d, m, k, sweeps, rep); }
#endif
    return nntc_cpu::solve_level0_all(d->cpu, m, k, sweeps, rep);
}

void level0_download(Device* d, const Model& m, int plane, std::vector<uint8_t>& k0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level0_download(d->cuda, m, plane, k0); return; }
    if (d->backend == Backend::Check) { nntc_check::level0_download(d, m, plane, k0); return; }
#endif
    nntc_cpu::level0_download(d->cpu, m, plane, k0);
}

void level0_upload(Device* d, const Model& m, int plane, const std::vector<float>& v0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level0_upload(d->cuda, m, plane, v0); return; }
    if (d->backend == Backend::Check) { nntc_check::level0_upload(d, m, plane, v0); return; }
#endif
    nntc_cpu::level0_upload(d->cpu, m, plane, v0);
}

void level0_download_values(Device* d, const Model& m, int plane, std::vector<float>& v0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::level0_download_values(d->cuda, m, plane, v0); return; }
    if (d->backend == Backend::Check) { nntc_check::level0_download_values(d, m, plane, v0); return; }
#endif
    nntc_cpu::level0_download_values(d->cpu, m, plane, v0);
}

void decode_plane(Device* d, const Model& m, int plane, std::vector<uint8_t>& rgb8)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::decode_plane(d->cuda, m, plane, rgb8); return; }
    if (d->backend == Backend::Check) { nntc_check::decode_plane(d, m, plane, rgb8); return; }
#endif
    nntc_cpu::decode_plane(d->cpu, m, plane, rgb8);
}

void objective_eval(Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::objective_eval(d->cuda, m, k, per_site, r); return; }
    if (d->backend == Backend::Check) { nntc_check::objective_eval(d, m, k, per_site, r); return; }
#endif
    nntc_cpu::objective_eval(d->cpu, m, k, per_site, r);
}

double objective_total_ms()
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Cuda) { return nntc_cuda::objective_total_ms(); }
    if (g_backend == Backend::Check) { return nntc_check::objective_total_ms(); }
#endif
    return nntc_cpu::objective_total_ms();
}

long long objective_passes()
{
#ifdef NNTC_CUDA
    if (g_backend == Backend::Cuda) { return nntc_cuda::objective_passes(); }
    if (g_backend == Backend::Check) { return nntc_check::objective_passes(); }
#endif
    return nntc_cpu::objective_passes();
}

void objective_check_host(Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::objective_check_host(d->cuda, m, k, per_site, r); return; }
    if (d->backend == Backend::Check) { nntc_check::objective_check_host(d, m, k, per_site, r); return; }
#endif
    nntc_cpu::objective_check_host(d->cpu, m, k, per_site, r);
}

double bc_pack_prepare(Device* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda)
    {
        return nntc_cuda::bc_pack_prepare(d->cuda, m, k, seed, pack_psnr, pack_texels);
    }
    if (d->backend == Backend::Check) { return nntc_check::bc_pack_prepare(d, m, k, seed, pack_psnr, pack_texels); }
#endif
    return nntc_cpu::bc_pack_prepare(d->cpu, m, k, seed, pack_psnr, pack_texels);
}

double bc_refine(Device* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { return nntc_cuda::bc_refine(d->cuda, m, passes, rep, pack_psnr, pack_texels); }
    if (d->backend == Backend::Check) { return nntc_check::bc_refine(d, m, passes, rep, pack_psnr, pack_texels); }
#endif
    return nntc_cpu::bc_refine(d->cpu, m, passes, rep, pack_psnr, pack_texels);
}

void bc_blocks_from_device(Device* d, Model& m, double& pack_psnr, size_t& pack_texels)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::bc_blocks_from_device(d->cuda, m, pack_psnr, pack_texels); return; }
    if (d->backend == Backend::Check) { nntc_check::bc_blocks_from_device(d, m, pack_psnr, pack_texels); return; }
#endif
    nntc_cpu::bc_blocks_from_device(d->cpu, m, pack_psnr, pack_texels);
}

void bc_state_save(const Device* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::bc_state_save(d->cuda, m, ep, sel); return; }
    if (d->backend == Backend::Check) { nntc_check::bc_state_save(d, m, ep, sel); return; }
#endif
    nntc_cpu::bc_state_save(d->cpu, m, ep, sel);
}

void bc_state_restore(Device* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::bc_state_restore(d->cuda, m, ep, sel); return; }
    if (d->backend == Backend::Check) { nntc_check::bc_state_restore(d, m, ep, sel); return; }
#endif
    nntc_cpu::bc_state_restore(d->cpu, m, ep, sel);
}

}   // namespace be
