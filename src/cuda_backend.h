// cuda_backend.h: the CUDA arm of the dispatcher, declared.
//
// One declaration per seam function, with model.h's own signatures unchanged - DeviceModel* stays DeviceModel*, because
// this arm IS the .cu files' interface and there is nothing to translate. The bodies in cuda_backend.cpp are one line
// each. This header and that file are compiled only when the configure found nvcc.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "image.h"
#include "model.h"

namespace nntc_cuda
{

bool device_select(int index, DeviceInfo& info);
DeviceModel* device_create(const Model& m, const std::vector<Image>& source_chain);
void device_destroy(DeviceModel* d);
size_t device_memory(const DeviceModel* d);

void init_level0(DeviceModel* d, Model& m, const float* luma_weights, bool luma_channel0);
void init_level1(DeviceModel* d, Model& m, bool pca, Level1InitReport& rep);
void init0_residual_channel(DeviceModel* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep);

void quantise_level1(DeviceModel* d, Model& m, const std::string& range_policy);
void freeze_level1_grid(DeviceModel* d, Model& m, const std::string& range_policy);
void quantise_level0(DeviceModel* d, Model& m, const std::string& range_policy);
void freeze_level0_grid(DeviceModel* d, Model& m, const std::string& range_policy);
void snap_level0_on_grid(DeviceModel* d, Model& m);

int stencil_monomials(int c0);

double solve_decoder(DeviceModel* d, Model& m, int k, const std::vector<double>& per_site);
void decoder_ridge_tally(long long out[4]);

double solve_level1_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level1_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                        std::vector<Level1Report>& rep);
double solve_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level0_cont_all(DeviceModel* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep);
double assemble_level0_for_refine(DeviceModel* d, const Model& m, int k);

void level1_download_indices(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k1);
void level1_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v1);
void level1_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k1);
void level0_upload_indices(DeviceModel* d, const Model& m, int plane, const std::vector<uint8_t>& k0);
void upload_decoder_to_device(DeviceModel* d, const Model& m);
void upload_grids_to_device(DeviceModel* d, const Model& m);
void level1_download(DeviceModel* d, const Model& m, int plane, std::vector<float>& v);
void level1_delta(DeviceModel* d, const Model& m, int plane, std::vector<double>& delta);
void level1_poke(DeviceModel* d, int plane, size_t index, float value);
void level1_range(DeviceModel* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi);
bool solve_level1_dense_host(DeviceModel* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns);

double solve_level0_all(DeviceModel* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep);
void level0_download(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& k0);
void level0_upload(DeviceModel* d, const Model& m, int plane, const std::vector<float>& v0);
void level0_download_values(DeviceModel* d, const Model& m, int plane, std::vector<float>& v0);

void decode_plane(DeviceModel* d, const Model& m, int plane, std::vector<uint8_t>& rgb8);
void objective_eval(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double objective_total_ms();
long long objective_passes();
void objective_check_host(DeviceModel* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);

double bc_pack_prepare(DeviceModel* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels);
double bc_refine(DeviceModel* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels);
void bc_blocks_from_device(DeviceModel* d, Model& m, double& pack_psnr, size_t& pack_texels);
void bc_state_save(const DeviceModel* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel);
void bc_state_restore(DeviceModel* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel);

}   // namespace nntc_cuda
