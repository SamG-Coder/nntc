// cpu/cpu_backend.cpp: the CPU arm's device selection and its record of what it implements.
//
// docs/CPU_BACKEND_PLAN.md step C filled the CPU arm in one stage at a time. Stage C1 brought the model, its lifecycle,
// stencil_monomials and every transfer (cpu_model.cpp) and the device selection below; each later stage moved the
// kernels it ported into a file of their own (stage C2: cpu_objective.cpp, stage C3: cpu_init.cpp, stage C4:
// cpu_solve_decoder.cpp, stage C5: cpu_solve_level1.cpp, stage C6: cpu_solve_level0.cpp, stage C7: cpu_refine_bc.cpp)
// and deleted its refusals here. Until then each entry point refused by name and exited 1, so that the flag surface
// stayed whole and testable at every stage; since stage C7 every seam function has a body and none refuses.
//
// implemented() is the one list of what this arm can run. The per-kernel harness reads it so that it can say
// `not implemented` and carry on, where calling the entry point would have ended the run; it now names every seam
// function, and a seam function added later is added here in the same commit as its body.

#include "cpu/cpu_backend.h"

#include <cstring>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fstream>
#endif

namespace nntc_cpu
{

bool implemented(const char* seam)
{
    static const char* const done[] = {
        // stage C1: the lifecycle, the monomial count the allocation needs, and every transfer
        "device_select", "device_create", "device_destroy", "device_memory", "stencil_monomials",
        "level1_download", "level1_download_indices", "level1_upload", "level1_upload_indices",
        "level0_upload_indices", "level0_upload", "level0_download_values", "level0_download", "level1_delta",
        "level1_poke", "level1_range", "upload_decoder_to_device", "upload_grids_to_device",
        "bc_state_save", "bc_state_restore",
        // stage C2: the objective, its brute-force twin, the decode, and the report's two objective counters
        // (cpu_objective.cpp)
        "objective_eval", "objective_check_host", "decode_plane", "objective_total_ms", "objective_passes",
        // stage C3: both latents' initialisations and the five grid entry points (cpu_init.cpp)
        "init_level0", "init_level1", "init0_residual_channel", "quantise_level1", "freeze_level1_grid",
        "quantise_level0", "freeze_level0_grid", "snap_level0_on_grid",
        // stage C4: block (a), the decoder's least squares and its ridge ladder (cpu_solve_decoder.cpp)
        "solve_decoder", "decoder_ridge_tally",
        // stage C5: blocks (b) and (c'), continuous and quantised, the BC refinement's assembly and the dense direct
        // twin (cpu_solve_level1.cpp)
        "solve_level1_all", "sweep_level1_all", "solve_level0_cont_all", "sweep_level0_cont_all",
        "assemble_level0_for_refine", "solve_level1_dense_host",
        // stage C6: block (c), the four-colour exact palette search (cpu_solve_level0.cpp)
        "solve_level0_all",
        // stage C7: the BC pack's preparation, the refinement and the blocks written back (cpu_refine_bc.cpp)
        "bc_pack_prepare", "bc_refine", "bc_blocks_from_device",
    };
    for (const char* name : done)
        if (std::strcmp(name, seam) == 0)
            return true;
    return false;
}

// The processor's own name, read from the platform, for the banner's workers row. Nothing decides anything on it; an
// unanswered question is a plain description rather than an error, because a CPU encode does not need to know.
static std::string platform_cpu_name()
{
    std::string name;
#ifdef _WIN32
    char buf[256] = {};
    DWORD size = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0", "ProcessorNameString",
                     RRF_RT_REG_SZ, nullptr, buf, &size) == ERROR_SUCCESS)
        name = buf;
#else
    // /proc/cpuinfo names an x86 processor on a `model name` line; an Arm kernel often has none, and then the machine
    // is simply not named.
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (name.empty() && std::getline(f, line))
        if (line.compare(0, 10, "model name") == 0)
        {
            const size_t colon = line.find(':');
            if (colon != std::string::npos)
                name = line.substr(colon + 1);
        }
#endif
    const size_t first = name.find_first_not_of(" \t");
    const size_t last = name.find_last_not_of(" \t\r\n");
    name = first == std::string::npos ? std::string() : name.substr(first, last - first + 1);
    return name.empty() ? std::string("an unnamed processor") : name;
}

// There is always a CPU. The index is the CUDA device's and is idle here (the argument parser says so when it was
// named); the name is the platform's, and there is no compute capability to invent.
bool device_select(int, DeviceInfo& info)
{
    info.name = platform_cpu_name();
    info.major = 0;
    info.minor = 0;
    return true;
}

}   // namespace nntc_cpu
