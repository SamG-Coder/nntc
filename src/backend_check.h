// backend_check.h: the per-kernel comparison harness, --backend check (docs/CPU_BACKEND_PLAN.md section 4).
//
// One function per seam function, over the dispatcher's own handle: the harness needs BOTH arms' models, and the
// handle carries both under check. backend.cpp routes to these exactly as it routes to either arm. Compiled only in a
// build that carries both arms, which is the only build in which check can be selected.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backend.h"

namespace nntc_check
{

void options(const std::string& only, bool stop_at_mismatch, bool quiet);
bool finish();

bool device_select(int index, DeviceInfo& info);
void device_create(be::Device* d, const Model& m, const std::vector<Image>& source_chain, int threads);
void device_destroy(be::Device* d);
size_t device_memory(const be::Device* d);

void init_level0(be::Device* d, Model& m, const float* luma_weights, bool luma_channel0);
void init_level1(be::Device* d, Model& m, bool pca, Level1InitReport& rep);
void init0_residual_channel(be::Device* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep);
void quantise_level1(be::Device* d, Model& m, const std::string& range_policy);
void freeze_level1_grid(be::Device* d, Model& m, const std::string& range_policy);
void quantise_level0(be::Device* d, Model& m, const std::string& range_policy);
void freeze_level0_grid(be::Device* d, Model& m, const std::string& range_policy);
void snap_level0_on_grid(be::Device* d, Model& m);
int stencil_monomials(int c0);
double solve_decoder(be::Device* d, Model& m, int k, const std::vector<double>& per_site);
void decoder_ridge_tally(long long out[4]);
double solve_level1_all(be::Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level1_all(be::Device* d, const Model& m, int k, double ridge, int sweeps,
                        std::vector<Level1Report>& rep);
double solve_level0_cont_all(be::Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level0_cont_all(be::Device* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep);
double assemble_level0_for_refine(be::Device* d, const Model& m, int k);
void level1_download_indices(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& k1);
void level1_upload(be::Device* d, const Model& m, int plane, const std::vector<float>& v1);
void level1_upload_indices(be::Device* d, const Model& m, int plane, const std::vector<uint8_t>& k1);
void level0_upload_indices(be::Device* d, const Model& m, int plane, const std::vector<uint8_t>& k0);
void upload_decoder_to_device(be::Device* d, const Model& m);
void upload_grids_to_device(be::Device* d, const Model& m);
void level1_download(be::Device* d, const Model& m, int plane, std::vector<float>& v);
void level1_delta(be::Device* d, const Model& m, int plane, std::vector<double>& delta);
void level1_poke(be::Device* d, int plane, size_t index, float value);
void level1_range(be::Device* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi);
bool solve_level1_dense_host(be::Device* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns);
double solve_level0_all(be::Device* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep);
void level0_download(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& k0);
void level0_upload(be::Device* d, const Model& m, int plane, const std::vector<float>& v0);
void level0_download_values(be::Device* d, const Model& m, int plane, std::vector<float>& v0);
void decode_plane(be::Device* d, const Model& m, int plane, std::vector<uint8_t>& rgb8);
void objective_eval(be::Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double objective_total_ms();
long long objective_passes();
void objective_check_host(be::Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double bc_pack_prepare(be::Device* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels);
double bc_refine(be::Device* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels);
void bc_blocks_from_device(be::Device* d, Model& m, double& pack_psnr, size_t& pack_texels);
void bc_state_save(const be::Device* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel);
void bc_state_restore(be::Device* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel);

}   // namespace nntc_check
