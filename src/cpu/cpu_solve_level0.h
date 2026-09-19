// cpu/cpu_solve_level0.h: block (c)'s colour pass on the CPU, for the per-kernel harness.
//
// solve_level0_all (cpu_backend.h) is the seam function and runs the four colour passes of every plane. The harness
// (src/backend_check.cpp) also wants ONE pass on its own: a colour pass writes only the texels of its colour, so CUDA's
// state after pass q is recoverable from the state before the call and the state after it, and the CPU pass can then be
// run from exactly CUDA's input to that pass and compared with exactly its output. That is what "k0 and v0 after each of
// the four colour passes" means in docs/CPU_BACKEND_PLAN.md stage C6, and these two functions are what it needs.

#pragma once

#include "cpu/cpu_model.h"
#include "model.h"

namespace nntc_cpu
{

// The argmin's branch: the joint enumeration when the level-0 bits sum to at most 8, --sweeps coordinate sweeps
// otherwise. A property of the layout, decided once per call of the driver (solve_level0.cu's solve_level0_all), never
// per plane and never from the data.
bool level0_search_joint(const Model& m);

// k_level0_search over the texels of one colour (colour & 1, colour >> 1) of one plane, in place. Returns the texels it
// moved.
unsigned int level0_search_pass(CpuModel* d, const Model& m, int k, int sweeps, bool joint, int plane, int colour);

}   // namespace nntc_cpu
