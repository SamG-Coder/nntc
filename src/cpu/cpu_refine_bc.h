// cpu/cpu_refine_bc.h: one colour of the BC refinement on the CPU, for the per-kernel harness.
//
// bc_refine (cpu_backend.h) is the seam function and runs whole passes over every plane. The harness
// (src/backend_check.cpp) also wants ONE colour on its own: a block of one colour writes only its own endpoints,
// selectors and texels, so CUDA's state after colour q of a pass is recoverable from the state before the pass and the
// state after it, and the CPU colour can then be run from exactly CUDA's input to it and compared with exactly its
// output (docs/CPU_BACKEND_PLAN.md stage C7). The level-0 workspace it reads - the stencil, the gradient and the plane
// the correction is measured from - is the one bc_pack_prepare assembled.

#pragma once

#include "cpu/cpu_model.h"
#include "model.h"

namespace nntc_cpu
{

// k_bc_refine over the blocks of one colour (colour & 1, colour >> 1) of one plane, in place. `taken` gains the sum of
// the accepted decreases of the blocks that improved and `improved` their count, each folded in block order.
void bc_refine_colour(CpuModel* d, const Model& m, int plane, int colour, double& taken, double& improved);

}   // namespace nntc_cpu
