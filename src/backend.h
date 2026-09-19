// backend.h: the encoder's device interface, once, for whichever backend is running it.
//
// model.h declares 46 free functions at global scope and the six .cu files DEFINE them there. Those files are frozen
// (docs/CPU_BACKEND_PLAN.md decision 1: they are byte-identical before and after this work and the hash says so), so a
// second implementation cannot take the same names. The host therefore stops calling the global ones and calls these
// instead: one namespace, one forwarder per seam function, and each arm's own names behind it.
//
// The handle is be::Device rather than DeviceModel*, and it carries ONE TYPED POINTER PER ARM. Each forwarder unwraps
// the pointer its branch needs and the compiler checks the type, so a wrong-branch unwrap is a compile error and not a
// cast that happens to link.
//
// What this buys, and it is the reason the design is safe: in a build configured WITHOUT CUDA the six .cu files are not
// compiled and the global symbols init_level0, solve_decoder, objective_eval and the rest do not exist. model.h still
// declares them, so a call site that was missed still compiles - but nothing defines it, and the LINK FAILS naming the
// symbol. A CPU-only build that links clean is a machine-checked proof that every call site goes through here.
//
// A third arm (HIP, Metal) is a new enumerator, a new pointer, a new file and a new branch: nothing here is shaped
// around there being exactly two.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "image.h"
#include "model.h"

namespace nntc_cpu
{
struct CpuModel;
}

namespace be
{

// Which implementation runs the encode. An enum and not a bool, and --backend takes a name and not a flag, so that a
// third arm is an enumerator beside these rather than a change of shape.
//
// Check is not a third implementation but the per-kernel comparison harness (docs/CPU_BACKEND_PLAN.md section 4,
// src/backend_check.cpp): the CUDA arm runs the encode and the CPU arm is mirrored and compared at every call. It
// exists only in a build that carries both arms.
enum class Backend
{
    Cuda,
    Cpu,
    Check
};

// The handle the host carries in place of DeviceModel*.
struct Device
{
    Backend backend = Backend::Cuda;
    DeviceModel* cuda = nullptr;          // non-null under Cuda and under Check
    nntc_cpu::CpuModel* cpu = nullptr;    // non-null under Cpu and under Check
};

// Six of the seam's functions take no handle to read a backend from, so the choice is also held here. It is set once,
// from the parsed command line, before anything touches a device; every function that DOES take a Device* reads
// d->backend and never this.
void backend_select(Backend b, int threads);
Backend backend_selected();
int backend_threads();
const char* backend_name(Backend b);

// Whether this build carries the CUDA arm at all, which is a property of the configure and not of the machine.
bool cuda_built_in();

// The host bytes the CPU arm will allocate for this model, before it allocates any: what the banner can warn with
// (plan section 1.11), and by construction the number device_memory reports afterwards.
size_t cpu_bytes_for(const Model& m);

// The harness's own two entry points (--backend check). check_options restricts the per-call lines to one seam function
// (--kcheck-only NAME, empty for all), stops the run at the first MISMATCH (--kcheck-stop), and under --quiet drops the
// summary table (never a MISMATCH line or the ERROR); check_finish prints the per-function summary and returns false
// when any call was a MISMATCH. Outside check mode check_finish is a quiet true.
void check_options(const std::string& only, bool stop_at_mismatch, bool quiet);
bool check_finish();

// ---------------------------------------------------------------------------------------------------------------------
// The 46. Every one is model.h's declaration with DeviceModel* replaced by Device*; no other parameter and no return
// type moves, so a call site's edit is the be:: prefix and nothing else.
// ---------------------------------------------------------------------------------------------------------------------

bool device_select(int index, DeviceInfo& info);
Device* device_create(const Model& m, const std::vector<Image>& source_chain);
void device_destroy(Device* d);
size_t device_memory(const Device* d);

void init_level0(Device* d, Model& m, const float* luma_weights, bool luma_channel0);
void init_level1(Device* d, Model& m, bool pca, Level1InitReport& rep);
void init0_residual_channel(Device* d, Model& m, int channel, const std::vector<double>& plane_weight,
                            Init0ChannelReport& rep);

void quantise_level1(Device* d, Model& m, const std::string& range_policy);
void freeze_level1_grid(Device* d, Model& m, const std::string& range_policy);
void quantise_level0(Device* d, Model& m, const std::string& range_policy);
void freeze_level0_grid(Device* d, Model& m, const std::string& range_policy);
void snap_level0_on_grid(Device* d, Model& m);

int stencil_monomials(int c0);

double solve_decoder(Device* d, Model& m, int k, const std::vector<double>& per_site);
void decoder_ridge_tally(long long out[4]);

double solve_level1_all(Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level1_all(Device* d, const Model& m, int k, double ridge, int sweeps, std::vector<Level1Report>& rep);
double solve_level0_cont_all(Device* d, const Model& m, int k, double ridge, std::vector<Level1Report>& rep);
double sweep_level0_cont_all(Device* d, const Model& m, int k, double ridge, int sweeps,
                             std::vector<Level1Report>& rep);
double assemble_level0_for_refine(Device* d, const Model& m, int k);

void level1_download_indices(Device* d, const Model& m, int plane, std::vector<uint8_t>& k1);
void level1_upload(Device* d, const Model& m, int plane, const std::vector<float>& v1);
void level1_upload_indices(Device* d, const Model& m, int plane, const std::vector<uint8_t>& k1);
void level0_upload_indices(Device* d, const Model& m, int plane, const std::vector<uint8_t>& k0);
void upload_decoder_to_device(Device* d, const Model& m);
void upload_grids_to_device(Device* d, const Model& m);
void level1_download(Device* d, const Model& m, int plane, std::vector<float>& v);
void level1_delta(Device* d, const Model& m, int plane, std::vector<double>& delta);
void level1_poke(Device* d, int plane, size_t index, float value);
void level1_range(Device* d, const Model& m, int plane, std::vector<float>& lo, std::vector<float>& hi);
bool solve_level1_dense_host(Device* d, const Model& m, int k, int plane, double lambda,
                             const std::vector<float>& c_prev, std::vector<double>& delta, int max_unknowns);

double solve_level0_all(Device* d, const Model& m, int k, int sweeps, std::vector<Level0Report>& rep);
void level0_download(Device* d, const Model& m, int plane, std::vector<uint8_t>& k0);
void level0_upload(Device* d, const Model& m, int plane, const std::vector<float>& v0);
void level0_download_values(Device* d, const Model& m, int plane, std::vector<float>& v0);

void decode_plane(Device* d, const Model& m, int plane, std::vector<uint8_t>& rgb8);
void objective_eval(Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);
double objective_total_ms();
long long objective_passes();
void objective_check_host(Device* d, const Model& m, int k, const std::vector<double>& per_site, Objective& r);

double bc_pack_prepare(Device* d, Model& m, int k, bool seed, double& pack_psnr, size_t& pack_texels);
double bc_refine(Device* d, Model& m, int passes, std::vector<BcRefineReport>& rep, double& pack_psnr,
                 size_t& pack_texels);
void bc_blocks_from_device(Device* d, Model& m, double& pack_psnr, size_t& pack_texels);
void bc_state_save(const Device* d, const Model& m, std::vector<std::vector<uint8_t>>& ep,
                   std::vector<std::vector<uint8_t>>& sel);
void bc_state_restore(Device* d, const Model& m, const std::vector<std::vector<uint8_t>>& ep,
                      const std::vector<std::vector<uint8_t>>& sel);

// Declared here rather than edited in model.h, so that model.h itself takes no change at all. It is defined in
// export.cpp exactly as before, on the host, and is a consumer of the seam rather than a part of it: it walks the
// planes and calls decode_plane.
void reconstruct_levels(Device* d, const Model& m, std::vector<std::vector<uint8_t>>& recon);

}   // namespace be
