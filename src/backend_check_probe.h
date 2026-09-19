// backend_check_probe.h: what the per-kernel harness reads out of the CUDA arm's workspaces (src/backend_check_probe.cu).
//
// The seam downloads what the host needs and nothing more, so the CUDA arm's intermediate results - the normal
// equations block (a) solves, the stencils, gradients and preconditioners blocks (b) and (c') assemble - have no
// seam function that returns them. The harness still wants them, because a
// disagreement found there names the accumulation, where one found in the decoder only says that something upstream of
// it moved. These reads live in a file of their own rather than in the kernel file that owns the buffer, which is not
// edited (docs/CPU_BACKEND_PLAN.md decision 1): they are downloads of device memory the kernels have finished writing,
// they launch nothing and write nothing, and the encode cannot tell whether they ran. Built only with the harness.

#pragma once

#include <vector>

#include "model.h"

namespace nntc_probe
{

// Block (a)'s normal equations as the last solve_decoder left them in d->reduction: the upper triangle of A, then the
// nout right-hand sides, (nin + 1)(nin + 2) / 2 + nout (nin + 1) doubles.
void normal_equations(DeviceModel* d, const Model& m, std::vector<double>& out);

// One plane's sparse least-squares workspace over one latent (device.cuh's PlaneWork): level 1's for block (b), or
// under --l0 bc8 level 0's for block (c'). Empty vectors where the workspace does not exist (level 0's outside bc8).
struct Workspace
{
    std::vector<double> stencil;   // texels * 5 * c * c
    std::vector<double> grad;      // texels * c
    std::vector<float> precond;    // texels * c * c
    std::vector<double> cg_x;      // the correction
    std::vector<float> prev;       // the plane as the quantised sweeps' stencil saw it
};
void workspace(DeviceModel* d, const Model& m, int plane, bool level0, Workspace& w);

// The monomial matrices the last assembly over that latent read (d->mono, or d->mono0 under --l0 bc8).
void monomials(DeviceModel* d, const Model& m, bool level0, std::vector<double>& out);

}   // namespace nntc_probe
