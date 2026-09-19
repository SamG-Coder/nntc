// cpu/cpu_backend.h: the CPU arm of the dispatcher, declared.
//
// The same 45 seam functions again, over the CPU backend's own model handle. The bodies arrive one stage at a time
// (docs/CPU_BACKEND_PLAN.md step C): stage C1 brought the lifecycle, the memory accounting, stencil_monomials and every
// transfer (cpu_model.cpp), stage C2 the objective and the decode (cpu_objective.cpp), stage C3 both latents'
// initialisations and grids (cpu_init.cpp), stage C4 block (a) (cpu_solve_decoder.cpp), stage C5 blocks (b) and (c')
// (cpu_solve_level1.cpp), stage C6 block (c) (cpu_solve_level0.cpp), stage C7 the BC refinement (cpu_refine_bc.cpp);
// every other KERNEL still refuses by name and exits 1 (cpu_backend.cpp)
// until its stage lands. implemented() says which is which, so that the per-kernel harness (src/backend_check.cpp)
// can report a function as `not implemented` and carry on rather than calling a refusal.
//
// Unlike the CUDA arm, this one is compiled in EVERY configure. That is what makes the CPU-only build possible, and the
// CPU-only build is what proves the dispatch complete: a call site that still names a global would fail to link there,
// because without nvcc nothing defines the global.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "image.h"
#include "model.h"

namespace nntc_cpu
{

// The plain-array twin of DeviceModel (cpu/cpu_model.h). It is opaque here for the same reason DeviceModel is opaque
// in model.h: the host carries the pointer and never looks inside it.
struct CpuModel;

// Whether this backend implements the seam function of that name yet. A name that is not a seam function is false.
bool implemented(const char* seam);

// The bytes device_create will allocate for this model, before it allocates them: the banner's figure (section 1.11).
size_t bytes_for(const Model& m);

bool device_select(int index, DeviceInfo& info);
// `threads` is -j as the dispatcher resolved it: the pool is built with it, once, here.
CpuModel* device_create(const Model& m, const std::vector<Image>& source_chain, int threads);
void device_destroy(CpuModel* d);
size_t device_memory(const CpuModel* d);
void init_level0(CpuModel* d, Model& m, const float* luma_weights, bool luma_channel0);
void init_level1(CpuModel* d, Model& m, bool pca, Level1InitReport& rep);
void init0_residual_channel(CpuModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep);
void quantise_level1(CpuModel* d, Model& m, const std::string& range_policy);
void freeze_level1_grid(CpuModel* d, Model& m, const std::string& range_policy);
void quantise_level0(CpuModel* d, Model& m, const std::string& range_policy);
void freeze_level0_grid(CpuModel* d, Model& m, const std::string& range_policy);
void snap_level0_on_grid(CpuModel* d, Model& m);
int stencil_monomials(int c0);
double solve_decoder(CpuModel* d, Model& m, int k, const std::vector<double>& per_site);
void decoder_ridge_tally(long long out[4]);
double solve_level1_all(CpuModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level1_all(CpuModel* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep);
double solve_level0_cont_all(CpuModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level0_cont_all(CpuModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep);
double assemble_level0_for_refine(CpuModel* d, const Model& m, int k);
void level1_download_indices(CpuModel* d, const Model& m, int plane, std::vector<uint8_t>& k1);
void level1_upload(CpuModel* d, const Model& m, int plane, const std::vector<float>& v1);
void level1_upload_indices(CpuModel* d, const Model& m, int plane, const std::vector<uint8_t>& k1);
void level0_upload_indices(CpuModel* d, const Model& m, int plane, const std::vector<uint8_t>& k0);
void upload_decoder_to_device(CpuModel* d, const Model& m);
void upload_grids_to_device(CpuModel* d, const Model& m);
void level1_download(CpuModel* d, const Model& m, int plane, std::vector<float>& v);
void level1_delta(CpuModel* d, const Model& m, int plane, std::vector<double>& delta);
void level1_poke(CpuModel* d, int plane, size_t index, float value);
void level1_range(CpuModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi);
bool solve_level1_dense_host(CpuModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns);
double solve_level0_all(CpuModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep);
void level0_download(CpuModel* d, const Model& m, int plane, std::vector<uint8_t>& k0);
void level0_upload(CpuModel* d, const Model& m, int plane, const std::vector<float>& v0);
void level0_download_values(CpuModel* d, const Model& m, int plane, std::vector<float>& v0);
void decode_plane(CpuModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8);
void objective_eval(CpuModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double objective_total_ms();
long long objective_passes();
void objective_check_host(CpuModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double bc_pack_prepare(CpuModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels);
double bc_refine(CpuModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels);
void bc_blocks_from_device(CpuModel* d, Model& m, double& pack_psnr, size_t& pack_texels);
void bc_state_save(const CpuModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel);
void bc_state_restore(CpuModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel);

}   // namespace nntc_cpu
