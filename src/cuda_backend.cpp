// cuda_backend.cpp: the CUDA arm, and the ONLY host file in the tree that names a function the .cu files define.
//
// Every body is one line. model.h's global declarations ARE the CUDA declarations, so this file forwards to them at
// global scope and adds nothing at all: no logic, no state, no translation. The kernels are not touched, not renamed
// and not wrapped (docs/CPU_BACKEND_PLAN.md decision 1); this file exists so that nothing else in the host code has to
// know they are there, and so that a configure without nvcc simply leaves one source file out.
//
// It is host C++ and includes no CUDA header, so it is compiled by the C++ compiler like any other .cpp; what makes it
// conditional is the CMake source list it sits in, beside the six .cu files it forwards to.

#include "cuda_backend.h"

#include "model.h"

namespace nntc_cuda
{

bool device_select(int index, DeviceInfo& info) { return ::device_select(index, info); }
DeviceModel* device_create(const Model& m, const std::vector<Image>& source_chain)
{
    return ::device_create(m, source_chain);
}
void device_destroy(DeviceModel* d) { ::device_destroy(d); }
size_t device_memory(const DeviceModel* d) { return ::device_memory(d); }
void init_level0(DeviceModel* d, Model& m, const float* luma_weights, bool luma_channel0)
{
    ::init_level0(d, m, luma_weights, luma_channel0);
}
void init_level1(DeviceModel* d, Model& m, bool pca, Level1InitReport& rep) { ::init_level1(d, m, pca, rep); }
void init0_residual_channel(DeviceModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep)
{
    ::init0_residual_channel(d, m, channel, plane_weight, rep);
}
void quantise_level1(DeviceModel* d, Model& m, const std::string& range_policy)
{
    ::quantise_level1(d, m, range_policy);
}
void freeze_level1_grid(DeviceModel* d, Model& m, const std::string& range_policy)
{
    ::freeze_level1_grid(d, m, range_policy);
}
void quantise_level0(DeviceModel* d, Model& m, const std::string& range_policy)
{
    ::quantise_level0(d, m, range_policy);
}
void freeze_level0_grid(DeviceModel* d, Model& m, const std::string& range_policy)
{
    ::freeze_level0_grid(d, m, range_policy);
}
void snap_level0_on_grid(DeviceModel* d, Model& m) { ::snap_level0_on_grid(d, m); }
int stencil_monomials(int c0) { return ::stencil_monomials(c0); }
double solve_decoder(DeviceModel* d, Model& m, int k, const std::vector<double>& per_site)
{
    return ::solve_decoder(d, m, k, per_site);
}
void decoder_ridge_tally(long long out[4]) { ::decoder_ridge_tally(out); }
double solve_level1_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return ::solve_level1_all(d, m, k, ridge, rep);
}
double sweep_level1_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep)
{
    return ::sweep_level1_all(d, m, k, ridge, sweeps, rep);
}
double solve_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep)
{
    return ::solve_level0_cont_all(d, m, k, ridge, rep);
}
double sweep_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep)
{
    return ::sweep_level0_cont_all(d, m, k, ridge, sweeps, rep);
}
double assemble_level0_for_refine(DeviceModel* d, const Model& m, int k)
{
    return ::assemble_level0_for_refine(d, m, k);
}
void level1_download_indices(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k1)
{
    ::level1_download_indices(d, m, plane, k1);
}
void level1_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v1)
{
    ::level1_upload(d, m, plane, v1);
}
void level1_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k1)
{
    ::level1_upload_indices(d, m, plane, k1);
}
void level0_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k0)
{
    ::level0_upload_indices(d, m, plane, k0);
}
void upload_decoder_to_device(DeviceModel* d, const Model& m) { ::upload_decoder_to_device(d, m); }
void upload_grids_to_device(DeviceModel* d, const Model& m) { ::upload_grids_to_device(d, m); }
void level1_download(DeviceModel* d, const Model& m, int plane, std::vector<float>& v)
{
    ::level1_download(d, m, plane, v);
}
void level1_delta(DeviceModel* d, const Model& m, int plane, std::vector<double>& delta)
{
    ::level1_delta(d, m, plane, delta);
}
void level1_poke(DeviceModel* d, int plane, size_t index, float value) { ::level1_poke(d, plane, index, value); }
void level1_range(DeviceModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi)
{
    ::level1_range(d, m, plane, lo, hi);
}
bool solve_level1_dense_host(DeviceModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns)
{
    return ::solve_level1_dense_host(d, m, k, plane, lambda, c_prev, delta, max_unknowns);
}
double solve_level0_all(DeviceModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep)
{
    return ::solve_level0_all(d, m, k, sweeps, rep);
}
void level0_download(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k0)
{
    ::level0_download(d, m, plane, k0);
}
void level0_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v0)
{
    ::level0_upload(d, m, plane, v0);
}
void level0_download_values(DeviceModel* d, const Model& m, int plane, std::vector<float>& v0)
{
    ::level0_download_values(d, m, plane, v0);
}
void decode_plane(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8)
{
    ::decode_plane(d, m, plane, rgb8);
}
void objective_eval(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    ::objective_eval(d, m, k, per_site, r);
}
double objective_total_ms() { return ::objective_total_ms(); }
long long objective_passes() { return ::objective_passes(); }
void objective_check_host(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r)
{
    ::objective_check_host(d, m, k, per_site, r);
}
double bc_pack_prepare(DeviceModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels)
{
    return ::bc_pack_prepare(d, m, k, seed, pack_psnr, pack_texels);
}
double bc_refine(DeviceModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels)
{
    return ::bc_refine(d, m, passes, rep, pack_psnr, pack_texels);
}
void bc_blocks_from_device(DeviceModel* d, Model& m, double& pack_psnr, size_t& pack_texels)
{
    ::bc_blocks_from_device(d, m, pack_psnr, pack_texels);
}
void bc_state_save(const DeviceModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel)
{
    ::bc_state_save(d, m, ep, sel);
}
void bc_state_restore(DeviceModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel)
{
    ::bc_state_restore(d, m, ep, sel);
}

}   // namespace nntc_cuda
