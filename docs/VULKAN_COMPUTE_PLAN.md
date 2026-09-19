# VULKAN_COMPUTE_PLAN.md -- a Vulkan compute backend for the encoder: feasibility and porting plan

A study, not code. The owner's question: "how difficult would it be to also support Vulkan compute in the encoder as
an option? I think we could port the CUDA kernels to GLSL, and add a new backend." This document answers it in the
shape `docs/CPU_BACKEND_PLAN.md` gave the CPU backend: what the backend must provide, what the hardware must have,
how each kernel maps, how the result is verified, a staged plan with estimates, and a bottom line. Every claim about
this tree carries a file and line; every claim about hardware or drivers carries a URL or says it could not be
checked. Nothing here has been built or measured; where a number is an estimate it says so.

**The short answer.** Feasible, and less of a translation problem than the CPU port was: the kernels use no
warp-level primitive at all (section 1.3), only shared memory, block barriers and three integer atomics, so every
kernel has a direct GLSL compute shape and the fixed reduction trees replay verbatim. The difficulty is not the
kernels. It is (1) **fp64**: the encoder is fp64-accumulation-bound, so the backend is only worth running on a GPU
whose Vulkan driver reports `shaderFloat64` and whose hardware runs doubles at a usable rate -- NVIDIA and AMD do,
Apple and Qualcomm do not, Intel is mixed (section 3.1); (2) **the plumbing**: a Vulkan compute program is an order
of magnitude more host code than a CUDA one for the same 37 launches, and the viewer's Vulkan code is written as one
file of statics and cannot be linked to as it stands (sections 2.6 and 4.5); (3) **the floating-point fine print**:
Vulkan does not promise correctly rounded division or square root, does not forbid contraction unless the shader
asks, and lets a driver flush fp32 denormals, so a Vulkan arm can be held to the CPU and CUDA arms to a tolerance
but not, on every device, to the bit (section 3.4). Estimated effort **8 to 11 weeks** for one implementer who knows
the tree, against the CPU port's measured 7 to 9 (section 7). The payoff is an encoder that runs at GPU speed on the
AMD machines in the owner's fleet, which the CPU backend serves at 8 to 14 times the RTX 5090's time
(`docs/CPU_PORT_LESSONS.md` section 8); on the Snapdragon laptop and on Apple hardware it changes nothing, because
those GPUs have no fp64 (section 3.1).

---

## 0. What this document rests on

* `docs/CPU_BACKEND_PLAN.md`: the dispatcher (`src/backend.{h,cpp}`, `src/cuda_backend.{h,cpp}`), the seam of 46
  functions (its section 1.3), the per-kernel harness (`--backend check`, its section 4), the stage plan (section 6),
  the gate arms (section 7) and its 1.8a, which says the seam was built for N arms and names what a third would cost.
* `docs/CPU_PORT_LESSONS.md`: judge distributions not cases (section 2), the `NNTC_CUDA_FMAD` experiment (section 3),
  the one reduction whose order had to be replayed because it feeds an ill-conditioned eigenproblem (section 4), the
  determinism rules (section 5), the testing layers (section 6), the owner's working rules (section 11).
* The kernel-side files, all nine hash-pinned in `tests/kernel_hashes.txt:1-9`: `src/init.cu`, `src/solve_decoder.cu`,
  `src/solve_level1.cu`, `src/solve_level0.cu`, `src/refine_bc.cu`, `src/objective.cu`, `src/device.cuh`,
  `src/sample.cuh`, `src/model.h`. **None of them changes under this plan.** The CPU twins in `src/cpu/` show every
  kernel's semantics in plain C++ and are the second reference a GLSL transcription is checked against.
* The Vulkan viewer, `viewer_vk/main.cpp` (3680 lines), `viewer_vk/vk_check.h`, `viewer_vk/platform*.{h,cpp}`,
  its `README.md` and `docs/VULKAN_VIEWER_PLAN.md`; `CMakeLists.txt:280-437`, the Vulkan and shaderc detection.
* `README.md` (Building, lines 55-177; Limits, 492-503), `docs/DESIGN.md` section 6 (determinism, lines 997-1047).

---

## 1. Inventory: what the backend must provide

### 1.1 The seam, unchanged

`src/backend.h:86-150` declares the 46 seam functions over `be::Device`, plus `reconstruct_levels`; `src/backend.cpp`
routes each on `d->backend` (or on the file-static `g_backend` for the six that take no handle, `backend.cpp:29-34`).
The Vulkan arm implements the same 46 in a namespace of its own (`nntc_vk`), exactly as `nntc_cuda`
(`src/cuda_backend.h:17-82`) and `nntc_cpu` (`src/cpu/cpu_backend.h:26-92`) do. `src/backend.h:17-18` already says
so: "A third arm (HIP, Metal) is a new enumerator, a new pointer, a new file and a new branch."

Grouped as the CPU plan grouped them (its section 1.3), with what the Vulkan arm does for each:

| group | functions | Vulkan arm |
|---|---|---|
| lifecycle | `device_select`, `device_create`, `device_destroy`, `device_memory` | instance, physical device, feature and limit check, one logical device, one compute queue, the buffers, the pipelines; section 4.3 |
| inits and grids | `init_level0`, `init_level1`, `init0_residual_channel`, `quantise_level1`, `freeze_level1_grid`, `quantise_level0`, `freeze_level0_grid`, `snap_level0_on_grid`, `stencil_monomials` | ten shaders and the host drivers of `init.cu` transcribed; `fit_range`, `jacobi_eigen` copied (as the CPU arm copied them, plan 1.8) |
| block (a) | `solve_decoder`, `decoder_ridge_tally` | two shaders; the ridge ladder and `gauss_jordan` copied |
| blocks (b), (c') | `solve_level1_all`, `sweep_level1_all`, `solve_level0_cont_all`, `sweep_level0_cont_all`, `assemble_level0_for_refine`, `solve_level1_dense_host` | nineteen shaders, the CG driver as command buffers; the dense twin copied |
| block (c) | `solve_level0_all` | one shader, four dispatches per plane |
| objective and decode | `objective_eval`, `objective_check_host`, `decode_plane`, `objective_total_ms`, `objective_passes` | three shaders; `normalise`, `twin_sample`, `objective_check_host` copied |
| BC refinement | `bc_pack_prepare`, `bc_refine`, `bc_blocks_from_device`, `bc_state_save`, `bc_state_restore` | two shaders, the hardest one being `k_bc_refine` |
| transfers | the fourteen of plan 1.3 | staging buffer copies both ways |

### 1.2 The 37 kernels, by file, and what each needs from the hardware

"fp64" says where double arithmetic is, not merely where a `double` is declared. "shared" is static plus dynamic
shared memory per workgroup. Block sizes are the launch constants; grids are as launched.

**`src/init.cu` (ten kernels, lines 389-1131)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_init_level0` | 389-428 | 256, ceil(texels/256) | none (fp32 palette walk) | none | none | strict `<` tie rule, 414-423 |
| `k_init_level0_cont` | 434-444 | 256 | none | none | none | |
| `k_init_level1_box` | 491-520 | 256 | none; **`long long` footprint arithmetic** 497-498 | none | none | float accumulator, 505-516 |
| `k_quantise_level1` | 524-534 | 256 | **yes**: `level1_index` / `level1_value` are double with `__dadd_rn` / `__dmul_rn` (`model.h:159-199`) | none | none | in and out alias at the level-0 call sites, 692, 732 |
| `k_block_means` | 750-779 | 256 | none; `long long` 756-757 | none | none | float accumulator |
| `k_cov_partial` | 785-807 | 256, **2-D grid (512, entries)** 901, 1183 | **yes**, the accumulation and the tree | 2048 B dynamic (`threads * sizeof(double)`) | none | **the reduction whose order the CPU port had to replay** (`docs/CPU_PORT_LESSONS.md` section 4) |
| `k_residual` | 1024-1045 | 256 | none (the decoder's fp32) | none | none | |
| `k_proj_peak` | 1068-1088 | 256, 512 | **yes**, projection and max tree | 2048 B dynamic | none | `ProjDir` (2 x 18 doubles = 288 bytes) passed **by value**, 1050-1054 |
| `k_proj_hist` | 1094-1104 | 256, 512 | yes, the projection | none | **`atomicAdd` on `unsigned long long`** 1102 | 4096 bins accumulated over every plane of the chain, 1344-1357 |
| `k_seed_channel` | 1112-1131 | 256 | yes, the projection | none | none | `lroundf` 1122 |

Host: `device_select` 76-129, `device_create` 151-296 (every allocation, the streams, the events, the pinned staging
277-290), `device_destroy` 313-383, `fit_range` 544-576, `jacobi_eigen` 813-859, `init_level1` 861-987,
`init0_residual_channel` 1146-1400, the grid entry points 628-746.

**`src/solve_decoder.cu` (two kernels)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_ls_accumulate` | 87-161 | 256, **512** (`SITE_BLOCKS`) | **yes, the whole walk**: every entry is a 256-term double dot product of staged floats, 152-155, into `acc[LS_SLOTS]` | **dynamic, `(mm + nout) * 257 * 4` bytes**: 18,504 at one texture `--c0 2 --c1 4`; 27,756 at four textures; 41,120 at five textures `--c0 4 --c1 4`; **44,204 at the caps** (`static_assert` at 84-85 against 48 KB) | none | two `__syncthreads` per 256-site tile, 125 and 131; the tile boundary is the summation order |
| `k_ls_reduce` | 165-180 | 64, ceil(entries/64) | yes | none | none | planes in index order, then 512 blocks in index order, 171-178 |

Host: `gauss_jordan` 184-223, the ridge ladder 242-430, `g_ridge_tally` 258.

**`src/objective.cu` (three kernels)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_decode` | 53-75 | 256 | none | none | none | `lroundf(acc * 255)` 73 |
| `k_objective` | 123-211 | 256, 512 | **everything**: three bilinear samples per site in double (`obj_sample`, 104-118), the features, `nout x nin` double multiply-adds per site, the three trees | `3 * 256 * 8` = 6144 B static | none | E is the loop's stop criterion, so this kernel's order decides the asset (134-136) |
| `k_obj_reduce` | 214-221 | **3**, 1 | yes | none | none | |

Host: `normalise` 224-250, `objective_check_host` 317-380 (already serial host code apart from three downloads).

**`src/solve_level0.cu` (one kernel)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_level0_search` | 75-270 | **128**, ceil(colour texels/128), four launches per plane | **yes**: `a0`, `a1[4]`, `a2[16]` accumulate in double from fp32 `qcol` / `resid`, 168-183; `search_energy` double, 63-73 | none | `atomicAdd(moved, 1u)` 268 | `long long` in the joint enumeration 201-212 (at most 256 states); the shadowed `dx, dy` at 98-99 and 107; the four-tap merge 131-146; `joint` decided once per call, 288 |

**`src/solve_level1.cu` (nineteen kernels)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_stencil_assemble` | 259-399 | **128**, ceil(texels/128) | **yes**: `sums[5 * 15]` and `g[4]` doubles per thread, `mono[15]`, the expansion against `mono_p` 380-396 | none | none | 75-double accumulator per thread: register pressure is the port's performance risk (section 5.2) |
| `k_stencil_assemble0` | 440-581 | 128 | yes, the same | none | none | centre site a single tap of weight 1, 476-487 |
| `k_dot_partial` | 592-608 | 256, **256** (`REDUCE_BLOCKS`) | yes | 2048 B | none | three per CG iteration |
| `k_reduce_sums` | 611-621 | **1**, 1 | yes | none | none | |
| `k_diag_partial` | 625-662 | 256, 256 | yes: sum, min, max trees | 6144 B (three arrays) | none | |
| `k_reduce_diag` | 667-682 | 1, 1 | yes, writes `SC_LAMBDA` 681 | none | none | a double division |
| `k_precondition` (+ `invert_block` 689-737) | 742-760 | 256, ceil(texels/256) | yes, 4x4 Gauss-Jordan in double, stored float | none | none | |
| `k_apply_precond` | 762-777 | 256 | yes (float x double) | none | none | |
| `k_matvec` | 781-828 | 256 | yes: 9 blocks x c1^2 double multiply-adds per texel | none | none | in-bounds tests, no clamp |
| `k_negate`, `k_axpy`, `k_xpby`, `k_add_correction` | 830-878 | 256 | yes, element-wise | none | none | `k_axpy` reads its coefficient from the device scalar row, 843 |
| `k_cg_alpha`, `k_cg_beta` | 857-870 | 1, 1 | yes: two divisions and a **`sqrt`** 869 | none | none | |
| `k_range_partial` | 890-923 | 256, 256 | yes, min/max trees per channel | 4096 B | none | `2 * c1` partial rows |
| `k_reduce_range` | 925-940 | 1, 1 | yes | none | none | |
| `k_movement_partial` | 944-979 | 256, 256 | yes: the count accumulated **as a double** 955, and two norms | 6144 B | none | |
| `k_quant_sweep` | 1023-1098 | 256, ceil(colour texels/256), four per sweep | **yes**: the neighbour term, `b / a` 1082, `level1_index` / `level1_value` | none | `atomicAdd(moved, changed)` 1097 | Gauss-Seidel over the texel's own channels |

Host: `stencil_monomials` 104-107, `build_form_matrices` and its two callers 164-240, `footprint_range` 250-257,
`assemble_all` 1167-1200, `solve_plane_all` 1222-1338 (the CG loop: 13 launches per iteration per active plane,
1263-1287; the host looks at `SC_RESID` every `CG_PROBE = 4` iterations, 1288-1302), `sweep_plane_all` 1343-1403,
`solve_level1_dense_host` 1502-1688.

**`src/refine_bc.cu` (two kernels)**

| kernel | lines | block, grid | fp64 | shared | atomics | notes |
|---|---|---|---|---|---|---|
| `k_bc_decode` | 155-174 | 256, ceil(texels/256) | yes: `bc0_value` is double with `__dadd_rn` / `__dmul_rn` 122-131 | none | none | `(int)(byte + 0.5f)` 171 |
| `k_bc_refine` | 178-481 | **64** (`REFINE_THREADS`), one workgroup per 4x4 block of one colour, four launches per plane per pass | **everything** | **dynamic `(n^2 + 3n + 64 + 16) * 8` bytes with `n = 16 c0`** (545): 3072 at `--c0 1`, 9600 at 2, 20,224 at 3, **34,944 at 4**, plus 120 bytes static (196-199) | **two `atomicAdd` on `double`**, 478-479, the report's `delta_e` and `improved` | `refine_reduce` is a 64-slot tree behind two barriers, 136-148, called five times per channel (318-322) plus once per accepted endpoint (387); every decision is thread 0's (324-362, 416-437); about 460 barriers per block per pass at `--c0 4` |

Host: `bc_pack_prepare` 504-533, `bc_refine` 537-585, `bc_blocks_from_device` 589-608, `bc_state_save` / `restore`
613-644.

### 1.3 What the kernels do NOT use, verified by search

A search of `src/*.cu` and `src/*.cuh` for `__shfl`, `warpSize`, `__ballot`, `__any`, `__all_sync`, `__syncwarp`,
`__activemask`, `cooperative_groups`, `tex2D`, `cudaTextureObject`, `surf2D`, `__ldg`, `cudaMallocManaged`,
`atomicCAS`, `atomicMax`, `atomicMin`, `__launch_bounds__`, `cudaFuncSetAttribute`, `__threadfence`,
`cudaLaunchKernel` and a kernel launch inside a `__global__` body finds **nothing**. So:

* **no warp-level primitive and no assumption about the warp width anywhere.** Every block-level reduction is a
  shared-memory halving tree under `__syncthreads`, which works at any subgroup size. The only warp-shaped constant is
  `LS_STRIDE = 257` (`solve_decoder.cu:47-55`), a bank-conflict pad for 32 banks that changes addressing and no
  arithmetic; the comment there says removing the conflict "changed nothing measurable";
* **no texture objects, no dynamic parallelism, no managed memory, no cooperative launch**;
* **atomics: three integer `atomicAdd` on `unsigned int` (`solve_level0.cu:268`, `solve_level1.cu:1097`) and
  `unsigned long long` (`init.cu:1102`), and the two `atomicAdd` on `double` in `refine_bc.cu:478-479`** that
  `docs/DESIGN.md:1016-1020` names as the one order-dependent number in the encoder, a report figure no decision reads;
* **shared memory: nine kernels use a `double[256]`-shaped tree (2 to 6 KB), `k_ls_accumulate` stages up to 44,204
  bytes and `k_bc_refine` up to about 35 KB** (1.2). Nothing asks for more than the 48 KB nvcc allows without an
  attribute, and the `static_assert` at `solve_decoder.cu:84-85` pins that;
* **fixed reduction orders**: the fixed grids `SITE_BLOCKS = 512` and `REDUCE_BLOCKS = 256` (`device.cuh:90, 129`),
  the grid-stride walks, the 256-slot trees and the index-order second passes (`docs/DESIGN.md:1010-1014`). A GLSL
  transcription keeps every one of them as written; nothing about Vulkan makes that harder than it was in CUDA.

### 1.4 Streams, events, copies and pinned memory

`device.cuh:269-270` gives every plane its own non-blocking stream and `EV_COUNT = 9` events; `device.cuh:295-301`
gives the model a master stream, three more events and two pinned host buffers (`host_scalars`, `host_counters`).
The protocol is fan-out / fan-in through events (`device.cuh:378-393`), a block timed by two events on the master
stream (`device.cuh:397-411`), per-plane phase timings read from the plane's events (`device.cuh:414-419`,
`solve_level1.cu:1332-1334`), and every upload issued on the master stream, waited for on the host, then fanned out
(`device.cuh:427-434`). Downloads are `cudaMemcpy` or `cudaMemcpyAsync` + `cudaStreamSynchronize` (103 such calls
across the kernel files and the probe). The host reads the device scalar row through the pinned copy after the
block's one synchronisation (`solve_level1.cu:1203-1218, 1293-1302`).

Section 2.5 maps each of these onto queues, command buffers, barriers, fences, timestamp queries and host-visible
memory.

---

## 2. The Vulkan and GLSL mapping

### 2.1 Workgroups, shared memory, barriers

| CUDA | GLSL / Vulkan | note |
|---|---|---|
| `<<<blocks, 256>>>` | `layout(local_size_x = 256) in;`, `vkCmdDispatch(blocks, 1, 1)` | 256 is above the Vulkan 1.0 minimum `maxComputeWorkGroupInvocations` of 128 (spec, Required Limits, https://docs.vulkan.org/spec/latest/chapters/limits.html); every device of section 3.6 reports 1024, and Vulkan 1.4 raised the minimum to 256 (https://docs.vulkan.org/spec/latest/appendices/versions.html). Checked at `device_select` anyway (4.3) |
| `blockIdx.x`, `threadIdx.x`, `gridDim.x`, `blockDim.x` | `gl_WorkGroupID.x`, `gl_LocalInvocationID.x`, `gl_NumWorkGroups.x`, `gl_WorkGroupSize.x` | the 2-D grid of `k_cov_partial` (`init.cu:901`) is `gl_WorkGroupID.y` |
| `__shared__ double sh[256]`, `extern __shared__` with a launch size | `shared double sh[256]`; dynamic shared memory does not exist in GLSL, so the two dynamic users (`k_ls_accumulate`, `k_bc_refine`) are compiled with **specialization constants** for `mm + nout` and `n = 16 c0` (`layout(constant_id = N)`) that size the `shared` arrays; one pipeline per distinct size, built at `device_create` for this layout | the `static_assert` at `solve_decoder.cu:84-85` becomes a check of `maxComputeSharedMemorySize` at pipeline build |
| `__syncthreads()` | `barrier()` | in Vulkan GLSL `barrier()` is `OpControlBarrier(Workgroup, Workgroup, AcquireRelease | WorkgroupMemory)`, so it orders shared memory as `__syncthreads` does (GL_KHR_vulkan_glsl, https://raw.githubusercontent.com/KhronosGroup/GLSL/main/extensions/khr/GL_KHR_vulkan_glsl.txt). Write `memoryBarrierShared(); barrier();` anyway: it costs nothing and reads as intended |
| the halving tree `for (half = 128; half > 0; half >>= 1) { if (tid < half) sh[tid] += sh[tid + half]; __syncthreads(); }` | verbatim | no subgroup operation replaces it, on purpose: `subgroupAdd` on floats reduces in an implementation-defined order (SPIR-V `OpGroupNonUniformFAdd`: "the method used to perform the group operation ... is implementation defined", https://github.com/llvm/llvm-project/commit/731b140a52b0d9b5af8702bb6d0ba3ca3c24c0dd; the Vulkan precision table lists only the group min and max as "correct result", https://docs.vulkan.org/spec/latest/appendices/spirvenv.html), which is exactly the scheduler dependence `docs/DESIGN.md:1006-1008` forbids |
| `atomicAdd(unsigned*)` | `atomicAdd(uint)` on a buffer variable, core GLSL (https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html) | |
| `atomicAdd(unsigned long long*)` | `GL_EXT_shader_atomic_int64` needs `shaderBufferInt64Atomics` (72.95 % of reports, https://vulkan.gpuinfo.org/listfeaturescore12.php; false on Apple and Intel Iris Xe) | **avoided**: 5.2 makes the bins 32-bit, exactly |
| `atomicAdd(double*)` | `GL_EXT_shader_atomic_float` needs `shaderBufferFloat64AtomicAdd`: **7.66 % of reports**, true on NVIDIA RTX 40/50, false on every AMD, Intel Windows, Adreno and Apple report (https://vulkan.gpuinfo.org/listfeaturesextensions.php) | **avoided**: 5.2's per-slot fold |

**Subgroup size.** Defaults across all reports: 32 in 58.1 %, 64 in 33.0 %, 8 in 7.2 %, 16 in 5.7 %, 128 in 2.0 %
(https://vulkan.gpuinfo.org/displaycoreproperty.php?core=1.1&name=subgroupSize&platform=all). NVIDIA 32; AMD 64
default, 32 to 64 (RX 7900 XTX / 9070 XT reports); Intel 32 default, 8 to 32; Adreno X1-85 64, 64 to 128 (Turnip 128);
Apple 32; lavapipe 8 (the reports of section 3.6). **No kernel in this tree depends on any of it** (1.3). The
`if (tid < half)` trees, the `tid < 16` / `tid < n` / `tid == 0` role assignments in `k_bc_refine` and the `tid`-strided
entry walk of `k_ls_accumulate` are all workgroup-relative and are correct at every subgroup width; only their speed
varies. `VK_EXT_subgroup_size_control` (core 1.3) is therefore not needed.

### 2.2 Buffers: device addresses, not descriptors

The kernels take raw pointers into some thirty buffers per plane plus twenty per model (`init.cu:177-273`), and the
drivers pass sub-ranges (`d->ls_partial + i * entries * SITE_BLOCKS`, `solve_decoder.cu:288`; `p.counters + 1 +
colour`, `solve_level0.cu:306`; `d->planes[plane].v1 + index`, `solve_level1.cu:1136`). The natural Vulkan spelling is
**`GL_EXT_buffer_reference`** over `bufferDeviceAddress` (core Vulkan 1.2, required in 1.3, 98.12 % of reports,
https://vulkan.gpuinfo.org/listfeaturescore12.php): a `buffer_reference` type per element type (`FloatBuf`,
`DoubleBuf`, `ByteBuf`, `UintBuf`), and a 64-bit address in the parameter block for every pointer argument. Then a
driver line like `k_ls_accumulate<<<...>>>(p.v0, p.v1, p.src, ..., d->ls_partial + offset)` transcribes to filling a
struct of addresses and ints and dispatching -- the host code stays line-for-line with the CUDA drivers, no descriptor
set is rebuilt per launch, and `maxStorageBufferRange` (4.8) no longer applies to the addressed buffers. The
extension "requires 64-bit integers" only for address arithmetic in the shader
(https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_buffer_reference.txt); the kernels index
with 32-bit offsets into typed references, so `shaderInt64` is used only where a sub-range is formed on the host, which
is host code.

The alternative -- one descriptor set per kernel with an SSBO binding per pointer, rebuilt or offset per dispatch -- is
what most Vulkan compute code does and is portable to devices without `bufferDeviceAddress`; there are essentially none
that have fp64 and lack it. **Recommended: device addresses**, with `VK_KHR_8bit_storage`'s `storageBuffer8BitAccess`
(96.19 %) for the `uint8_t` planes `k0`, `k1`, `bc_ep`, `bc_sel`, `rgb8`; without 8-bit storage those would need packed
`uint` words and read-modify-write atomics for the per-texel byte stores, which is a rewrite.

### 2.3 Kernel arguments: a parameter block per dispatch

Every launch passes its arguments by value. `maxPushConstantsSize` is 128 bytes in Vulkan 1.0 to 1.3 (256 in 1.4;
NVIDIA and AMD report 256, section 3.6). With eight-byte addresses `k_level0_search`'s ten pointers, fourteen ints and
`ChannelBits` are about 160 bytes, and `k_proj_peak`'s `ProjDir` alone is 288 (`init.cu:1050-1054`). So: **one
`std430` parameter struct per kernel**, written into a per-dispatch slot of a host-visible ring buffer, and its
address in a 16-byte push constant (`layout(push_constant)` is `std430` by default, GL_KHR_vulkan_glsl). One
convention for all 37 kernels, no descriptor sets at all beyond an empty layout, and the parameter ring is the only
descriptor-free host-to-shader path that is portable and has no 128-byte cliff. Specialization constants are used
only for the two shared-array sizes of 2.1 and for the workgroup size (`local_size_x_id`).

### 2.4 Floating point: `precise`, and what Vulkan does and does not promise

CUDA's defaults in this tree: fp32 add, multiply, division and square root correctly rounded (`-prec-div=true`,
`-prec-sqrt=true`), denormals preserved (`-ftz=false`), contraction **off** (`NNTC_CUDA_FMAD=OFF` compiles
`-fmad=false`, `CMakeLists.txt:171-183`); fp64 IEEE throughout, unaffected by those flags
(https://docs.nvidia.com/cuda/floating-point/index.html, sections 4.3 and 4.4). Vulkan promises less, in five ways
that each need a decision in the shaders:

1. **Contraction is allowed unless forbidden.** Without `NoContraction`, the Vulkan SPIR-V environment assumes
   `AllowContract`, `AllowReassoc`, `AllowRecip` and `AllowTransform`, and "implementations may reorder or combine
   operations" (https://docs.vulkan.org/spec/latest/appendices/spirvenv.html). GLSL's `precise` qualifier forbids
   re-association and forbids turning `a * b + c` into one fma on everything that feeds a `precise` variable
   (https://docs.vulkan.org/glsl/latest/chapters/variables.html). glslang has no global switch for it; the port
   therefore **declares every floating-point accumulator, every stored result and every value a comparison reads as
   `precise`**, which in these kernels is nearly everything. That is the shader-side twin of `-fmad=false`, and it is
   also how `level1_value`'s and `bc0_value`'s `__dadd_rn` / `__dmul_rn` (`model.h:163-164`, `refine_bc.cu:126-127`)
   are spelled: two `precise` operations. Verified in stage V2 by disassembling one shader for `NoContraction`.
   (A known glslang bug loses `NoContraction` through struct constructors, https://github.com/KhronosGroup/glslang/issues/2709;
   the kernels use none.)
2. **Division is 2.5 ULP, not correctly rounded.** The precision table gives `OpFDiv` 2.5 ULP for fp32 and says of
   fp64 only that "the precision of double-precision instructions is at least that of single precision"; it does not
   promise IEEE rounding for doubles at all (same URL; an open request for such a guarantee is
   https://github.com/KhronosGroup/Vulkan-Docs/issues/2816). Where the kernels divide: fp32 in `bilinear_tap_pixel`
   (`sample.cuh:46-47`, only when `w != w0`, that is level 1's taps), `1.0f / (float)n` in the box inits
   (`init.cu:517, 776`); fp64 in `obj_sample`'s `u, v` (`objective.cu:146-147`), `k_reduce_diag`'s lambda (681),
   `k_cg_alpha` / `k_cg_beta` (860, 866), `k_quant_sweep`'s `b / a` (1082), `footprint_range` (252), the refinement's
   vertex (337-338) and `step` (299), `bc4_entry`'s `/ 7.0f` (115), `bc0_value`'s `/ 255.0` (124). In practice
   NVIDIA and AMD compile fp32 and fp64 division to sequences that are correctly rounded or within an ULP, but the
   spec does not say so and neither does any vendor document this study found; **so the harness measures it** (6.1).
   A last-bit difference in `fx` moves a bilinear weight, which the CPU port already showed the arithmetic tolerates
   (its identical-bits rows held only because `-fmad=false` made the floats identical; a 1-ULP `fx` would have put them
   at the `BAR_ENTRY_DIRECT` level instead).
3. **Square root** is 2 ULP for fp32 in the same table; the one device `sqrt` is `k_cg_beta`'s residual (869), which
   only the host stop test reads against 1e-10 -- at worst a probe's overshoot, which the harness counts as a flip.
4. **Denormals may be flushed** unless `DenormPreserve` is declared, and declaring it needs
   `shaderDenormPreserveFloat32` / `Float64` (`VK_KHR_shader_float_controls`, core 1.2). The RX 9070 XT reports both
   true; **the RTX 5090 reports both FALSE** (the two reports of 3.6), meaning NVIDIA does not let a shader ask, not
   that it flushes -- its actual default is not documented in anything this study found. Rounding mode: both report
   `shaderRoundingModeRTEFloat64` true, so `RoundingModeRTE` can be declared for fp64 on both.
5. **GLSL `round()` is implementation-defined at .5** ("will round in a direction chosen by the implementation,
   presumably the direction that is fastest", https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html);
   `lroundf` is half away from zero. Section 5.1 spells every rounding out.

One more fact worth knowing and not needing: `VK_KHR_shader_fma` (`OpFmaKHR`, correctly rounded fused
multiply-add; the RTX 5090 reports `shaderFmaFloat64`) exists on new drivers; the port has no use for a *fused* op,
since the reference is unfused.

### 2.5 Streams, events and pinned memory

| CUDA (`device.cuh`) | Vulkan |
|---|---|
| one non-blocking stream per plane, a master stream, fan-out / fan-in by events (`269-270, 295-297, 378-393`) | **one compute queue, one command buffer per block**: the planes' dispatches recorded in sequence, a `vkCmdPipelineBarrier2` (compute -> compute, shader write -> shader read) only between dependent dispatches; independent planes' dispatches sit between the same two barriers and may overlap on the device (the spec orders commands in a queue only with respect to explicit synchronisation). The deep planes then run behind the base's dispatches rather than beside them, which the CPU plan measured as immaterial: the base is 86 to 96 % of the work (`docs/CPU_BACKEND_PLAN.md` 3.1 rule 6) |
| `cudaStreamSynchronize(master)` once per block (`device_block_end`, 403-411) | `vkQueueSubmit2` with a fence, `vkWaitForFences` |
| the CG loop's probe every four iterations (`solve_level1.cu:1288-1302`) | one command buffer per four iterations, submitted, fenced; the scalar row read from a host-visible buffer. If V1's measurement shows a submission costs more than a CUDA event wait, `CG_PROBE` stays 4 regardless (decision 6 of the CPU plan) and the cost is accepted |
| `cudaEventRecord` pairs for timing (`EV_A_START` ... `EV_DONE`, 167-180; `device_elapsed`, 414-419) | `vkCmdWriteTimestamp2` into a query pool, `timestampPeriod` (1 ns NVIDIA, 10 ns AMD, 52 ns Intel and Adreno, section 3.6), read after the fence; `timestampComputeAndGraphics` is true on every device of 3.6 |
| `cudaMemcpyAsync` host-to-device on the master stream, then fan-out (427-434) | write into the mapped staging ring, `vkCmdCopyBuffer` at the head of the next command buffer, a transfer -> compute barrier |
| `cudaMemcpy` device-to-host (the downloads) | `vkCmdCopyBuffer` to a host-visible, host-cached readback buffer, fence, `memcpy` out; or, on the host-visible device-local heaps every discrete card of 3.6 exposes, a direct mapped read of the scalar rows |
| `cudaMallocHost` pinned `host_scalars`, `host_counters` (`init.cu:277-290`) | a `HOST_VISIBLE | HOST_COHERENT` buffer, persistently mapped; the same two arrays |
| `cudaMemsetAsync` (`solve_level0.cu:295`, `solve_level1.cu:1243, 1360-1361`) | `vkCmdFillBuffer` |
| `cudaMemcpyAsync` device-to-device (`solve_level1.cu:1249, 1359, 1445`) | `vkCmdCopyBuffer` |

Multiple queues are available -- NVIDIA exposes 8 compute-only families' worth, AMD 4 to 8, Intel 4, Adreno 1
(section 3.6) -- and timeline semaphores are core 1.2 (99.88 %). A queue per plane would reproduce the CUDA overlap
exactly; it is **deliberately not in the plan** until V5 measures a need, for the CPU plan's reason: one queue is one
ordering a reader can compute, and the base plane is the block.

### 2.6 The shader toolchain

Three ways to get SPIR-V from source, and the tree already has one of them working:

* **GLSL through `glslc` at build time, embedded** (recommended). The Vulkan SDK ships `glslc`, and CMake's
  `FindVulkan` exposes it as `Vulkan::glslc` beside the `shaderc_combined` component the viewer already asks for
  (https://cmake.org/cmake/help/latest/module/FindVulkan.html); Debian's `glslc` package is the same binary. The 37
  `.comp` files compile at build time into `uint32_t` arrays under the build directory, `--target-env=vulkan1.3`,
  **`-O0`** (the viewer's no-optimisation choice, which the Qualcomm compiler needed, `viewer_vk/README.md:322-324`,
  and which keeps the SPIR-V close to the GLSL for reading), and the encoder has no run-time compiler dependency and no
  shader file beside the executable. A hash of the GLSL sources can then join `tests/kernel_hashes.txt`'s arm, as a
  second pinned set.
* **GLSL through libshaderc at run time** -- the viewer's path (`viewer_vk/main.cpp:2139-2167`, `CMakeLists.txt:291-332`
  with its Linux shared-library preference). Same GLSL, same result; the cost is a run-time dependency and the "which
  copy did it read" failure the viewer already fixed once (`CMakeLists.txt:427-430`). Kept as the fallback where
  `glslc` is absent.
* **Slang** (Khronos-hosted since 2024-11, Apache 2.0, SPIR-V among its targets,
  https://www.khronos.org/news/press/khronos-group-launches-slang-initiative-hosting-open-source-compiler-contributed-by-nvidia;
  out of beta in SDK 1.4.304.1, https://vulkan.lunarg.com/doc/view/1.4.304.1/mac/release_notes.html). Its C++-like
  syntax, generics and operator overloading would make the transcription shorter and would make an fp64-free
  double-float variant (3.2) writable without rewriting every expression. Against it: a third language in a tree that
  already has HLSL and GLSL, a compiler whose `precise` and contraction story would have to be verified on its own,
  and no existing build integration.
* **HLSL through DXC** also emits `OpTypeFloat 64` for `double` and has specialization constants
  (https://raw.githubusercontent.com/microsoft/DirectXShaderCompiler/main/docs/SPIR-V.rst), and the Direct3D viewer's
  shader is HLSL; but the Vulkan viewer's is GLSL, and the compute kernels are closer to GLSL's C-like style.

**Recommendation: GLSL, `glslc` at build time, shaderc at run time as the fallback**, which is the toolchain the tree
already knows and the one whose contraction semantics (`precise`) this study could verify.

---

## 3. The hard questions

Every device figure below is from a vulkan.gpuinfo.org report read on 2026-09-19 unless another source is named; a
report is one machine and one driver, and the site sometimes lists a device both ways under different drivers. Where
that mattered it is said.

### 3.1 fp64 in Vulkan compute, vendor by vendor

`shaderFloat64` is reported by **31.4 % of all reports** across every platform
(https://vulkan.gpuinfo.org/listfeaturescore10.php) -- the figure includes phones. On the parts that matter here:

| vendor / part | `shaderFloat64` | source |
|---|---|---|
| NVIDIA GeForce RTX 40 and 50, Windows and Linux (proprietary and NVK) | **true** on every listed model; RTX 5090 driver 616.92, Vulkan 1.4.351 | https://vulkan.gpuinfo.org/displayreport.php?id=51761 |
| AMD RX 6000 / 7000 / 9000, Windows proprietary and RADV | **true**; RX 9070 XT Windows 2.0.406 and RADV 26.2.3 | https://vulkan.gpuinfo.org/displayreport.php?id=51731, https://vulkan.gpuinfo.org/displayreport.php?id=51892 |
| Intel Arc A-series (Alchemist), Windows | **false** (A770, report 51124) | https://vulkan.gpuinfo.org/displayreport.php?id=51124 |
| Intel Iris Xe (Gen12), UHD 7xx, Gen11, Arrow Lake Arc 130T/140T, Windows | **false** | https://vulkan.gpuinfo.org/listdevicescoverage.php?feature=shaderFloat64&platform=windows&option=not |
| Intel Arc B570/B580 (Battlemage), Lunar Lake Arc 130V/140V, Windows | **true** (B580, report 51772) | https://vulkan.gpuinfo.org/displayreport.php?id=51772 |
| Intel under Mesa ANV | false on DG2, Tiger Lake, Alder Lake, Raptor Lake, Ice Lake, DG1; true on Meteor Lake, Lunar Lake, Battlemage, Gen8/9 | https://vulkan.gpuinfo.org/listdevicescoverage.php?feature=shaderFloat64&platform=linux |
| Intel's own statement | 11th-gen integrated GPUs and Arc discrete GPUs "do not support" fp64 in shaders; no plan to add it | https://www.intel.com/content/www/us/en/support/articles/000089817/graphics.html |
| Qualcomm Adreno X1-85 (Snapdragon X Elite), Windows driver 512.886, Vulkan 1.4.295 | **false**; also false on X1-45, X2-xx, Adreno 7xx/8xx, and under Turnip | https://vulkan.gpuinfo.org/displayreport.php?id=51583, https://vulkan.gpuinfo.org/displayreport.php?id=48865 |
| Apple via MoltenVK (M1, M4) | **false**; Metal has no `double` (MoltenVK issue 1106 shows VkFFT forcing the feature and failing) | https://vulkan.gpuinfo.org/displayreport.php?id=51417, https://github.com/KhronosGroup/MoltenVK/issues/1106 |
| Mesa lavapipe (Mesa 26.2.2, LLVM 22) | **true**, with 64-bit int atomics | https://vulkan.gpuinfo.org/displayreport.php?id=51884 |
| Mesa dozen (Vulkan on D3D12), Windows host reports | true (Vulkan 1.2), 64-bit atomics false; not verified inside WSL | https://vulkan.gpuinfo.org/displayreport.php?id=39721 |

Mesa carries a debug-only software fp64 for Intel (`INTEL_DEBUG=soft64`, https://docs.mesa3d.org/envvars.html); the
ANV "true" on Meteor Lake and Lunar Lake was not traced to hardware or emulation by this study (not verified).
Chips and Cheese did not benchmark fp64 on the A770 "because the Arc A770 doesn't support it"
(https://chipsandcheese.com/p/microbenchmarking-intels-arc-a770).

**For the owner's fleet:** the RTX 5090 (Windows) and both AMD machines can run the arm; the Snapdragon laptop cannot;
the Intel laptop's iGPU, if it is Gen12 or Alchemist-class, cannot -- and its AMD GPU is the target there anyway;
lavapipe can, everywhere, at software speed.

### 3.2 What if fp64 is unavailable

The encoder is fp64-accumulation-bound (`docs/CPU_BACKEND_PLAN.md` 1.2: "the encode is fp64-accumulation-bound";
27 of 37 kernels carry double arithmetic, section 1.2), and the doubles are not decoration: `docs/RESULTS.md` 1.3
measured and rejected fp32 accumulation for block (a) (a relative error in the normal matrix ten times larger than
the ridge, `docs/CPU_BACKEND_PLAN.md` section 9), and `solve_level1.cu:53-57, 79-82` says why the stencil and the CG
vectors are double (condition numbers of 1e5). The options on a device without `shaderFloat64`:

1. **Refuse the device** (this plan's default). `--backend vulkan` by name is an ERROR naming the missing feature;
   `auto` falls back to the CPU with a WARNING. Faithful, zero shader work, and on the owner's fleet it changes only
   the message the Snapdragon prints.
2. **Double-float ("df64") emulation**: each double as a pair of floats, add and multiply as 10 to 20 fp32
   operations with error-free transforms. About 48 bits of significand, not 53; **not** the CUDA arithmetic, so the
   harness's exact rows become tolerances and the path-dependence lesson (`docs/CPU_PORT_LESSONS.md` section 2) applies
   in full: a different, not a worse, optimiser. It needs correctly rounded fp32 add and multiply (which Vulkan
   promises) and, for the exact product, either a true fused multiply-add (`OpFmaKHR`, `VK_KHR_shader_fma`, on new
   drivers only, section 2.4) or Dekker's split (more operations). In GLSL, with no operator overloading, every
   double expression in 2,400 lines of shader becomes a function call, so it is a second set of shaders, not a
   `#define` (Slang, 2.6, would make it a type). The Snapdragon X Elite is the only fleet machine that would gain,
   and how fast it would be is unknown (not measured; not estimated here).
3. **Compensated fp32** (Kahan) on the sums with fp32 products: the products of two floats are not exact in fp32,
   so this does not reproduce the double accumulation of `k_ls_accumulate:154` and is a different algorithm.
4. **Split the work**: the fp32 feature evaluation on the device, the fp64 accumulation on the host. The accumulation
   is the cost (1.2), so this is the CPU backend with a GPU pre-pass; not worth its complexity.

`docs/CPU_PORT_LESSONS.md` section 4's rule -- replay CUDA's exact order and precision where a sum feeds something
ill-conditioned -- cannot be honoured without fp64. **Recommendation: 1; 2 only as a separately funded experiment if the
Snapdragon ever needs to encode.**

### 3.3 fp64 throughput

Peak figures from published specification tables; the encoder achieved about 330 GFLOP/s fp64 on the RTX 5090 across
an encode (`docs/CPU_BACKEND_PLAN.md` 1.2), a fifth of that card's 1.64 TFLOPS fp64 peak.

| part | fp32 peak | fp64 peak | ratio | source |
|---|---|---|---|---|
| RTX 5090 (GB202) | 104.8 TFLOPS | 1.64 TFLOPS | 1:64 | https://en.wikipedia.org/wiki/GeForce_RTX_50_series |
| RTX 4090 (AD102) | 82.6 | 1.29 | 1:64 | https://en.wikipedia.org/wiki/GeForce_40_series |
| RX 7900 XTX (RDNA3) | 61.4 (dual-issue) | 0.96 | 1:64 of the dual-issue figure | https://en.wikipedia.org/wiki/Radeon_RX_7000_series |
| RX 9070 XT (RDNA4) | -- | 0.61 base / 0.76 boost | 1:64 | https://en.wikipedia.org/wiki/RDNA_4 (a third-party database says 1.52; the two conflict, and the Wikipedia row is used) |
| RX 6900 XT (RDNA2) | -- | -- | not verified by this study | -- |
| Intel Arc A770 | -- | none exposed | -- | https://en.wikipedia.org/wiki/Intel_Arc |
| Intel Arc B580 | inconsistent in the source | 1.7 to 2.1 TFLOPS listed | not derivable | https://en.wikipedia.org/wiki/Intel_Arc (the FP32 figure in that table looks like an FP16 one; not verified) |
| Adreno X1-85 | -- | none exposed | -- | section 3.1 |

So a consumer AMD card has **roughly half to two thirds of the RTX 5090's fp64 peak**, and an NVIDIA card under Vulkan
has the same units it has under CUDA. Section 8.1 turns that into an estimate.

### 3.4 Contraction, rounding, denormals: what CPU/CUDA/Vulkan agreement can mean

Section 2.4 has the mechanics; the consequence for the three-way comparison is this. The CPU port reached
byte-identical encodes on 19 of 21 corpus cases because `-fmad=false` made the fp32 features the same floats and
the replayed trees made the fp64 sums the same doubles (`docs/CPU_PORT_LESSONS.md` sections 3, 4, 6). A Vulkan arm
can be brought to the same place on a given driver **only if** that driver's division is correctly rounded (not
promised), its fp32 add and multiply are RTE (promised only when `RoundingModeRTE` is declared and the feature is
reported; both cards of 3.6 report it), denormals are handled the same way (declarable on AMD, not on NVIDIA), and
`precise` reaches every operation (verifiable in the SPIR-V). The honest expectation, to be replaced by V2's
measurement: **on NVIDIA and AMD, identical bits on the fp64 reductions and on block (c)'s search; the fp32 features
identical or one ULP apart depending on the division; end-to-end encodes byte-identical to the CPU arm on most corpus
cases, and where not, within the owner's distribution bars** (`docs/CPU_PORT_LESSONS.md` section 2: about 0.35 dB per
texture, 0.5 dB per mip level is "close enough"). Bit identity between arms is not a goal and must not be stated as
one to a helper agent (the same section).

### 3.5 The platforms

* **Windows, RTX 5090**: native NVIDIA Vulkan 1.4; the CUDA-against-Vulkan machine.
* **WSL 2**: NVIDIA's WSL driver exposes CUDA, DirectX and DirectML ("NVIDIA driver support for WSL 2 includes not
  only CUDA but also DirectX and Direct ML support", https://docs.nvidia.com/cuda/wsl-user-guide/index.html); the
  guide says nothing of Vulkan, and the viewer's own experience is that WSL's Vulkan is Mesa's dozen layer over D3D12
  (`docs/VULKAN_VIEWER_PLAN.md:1012`), "upstream-flagged non-conformant", Vulkan 1.2
  (https://github.com/ggml-org/llama.cpp/discussions/26729; a request for 1.3 is open, https://github.com/microsoft/wslg/issues/1340).
  Dozen reports `shaderFloat64` true on a Windows host (3.1); whether it does inside WSL, and whether its fp64 is
  D3D12's native doubles, is **not verified**. The dependable Linux device for the gate is lavapipe: fp64, Vulkan 1.4,
  software speed.
* **Windows, Intel CPU + AMD GPU**: AMD's proprietary Windows driver -- fp64 true, **32 KB shared memory, 2 GB
  `maxMemoryAllocationSize`** (3.6), which are the two limits that bite (5.2, 4.8).
* **Kubuntu, AMD**: RADV -- fp64 true, 64 KB shared memory; the second compiler for the `precise` rule.
* **Snapdragon X Elite**: Adreno X1-85, native ICD Vulkan 1.3/1.4, no fp64: the refusal path only.
* **lavapipe**: fp64 true, subgroup 8, 32 KB shared, the spec-minimum-shaped device for the fallback paths.

### 3.6 Limits, per device (the two cards the owner has or is likely to meet, plus the classes)

| limit or feature | RTX 5090, Windows 616.92 | RX 9070 XT, Windows 2.0.406 | spec minimum (1.0 / 1.4) | others |
|---|---|---|---|---|
| `maxComputeSharedMemorySize` | 49,152 | **32,768** | 16,384 / 16,384 | RADV 65,536; Intel A770/B580/MTL 49,152; Intel 140V and Iris Xe 32,768; Adreno, Apple, lavapipe 32,768. Distribution: 32,768 most common (4,314 reports), then 49,152 (3,268), 65,536 (1,648), 16,384 (193) (https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxComputeSharedMemorySize&platform=all) |
| `maxComputeWorkGroupInvocations` | 1,024 | 1,024 | 128 / 256 | 1,024 everywhere in the reports read; Turnip 2,048 |
| `maxComputeWorkGroupCount[0]` | 2,147,483,647 | 4,294,967,295 | 65,535 | lavapipe not read (not verified); 5.1 slices anyway |
| `maxPushConstantsSize` | 256 | 256 | 128 / 256 | |
| `maxStorageBufferRange` | 4,294,967,295 | 4,294,967,295 | 2^27 | 4,294,967,295 in 5,518 reports, **134,217,728 in 1,613** (https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxStorageBufferRange&platform=all) |
| `maxMemoryAllocationSize` | 2^64 - 1 (unbounded) | **2,147,483,648** | 2^30 | about 4 GB (4,292,870,144) in 1,546 reports, 2 GB in 1,344, 1 GB in 1,040 (https://vulkan.gpuinfo.org/displaycoreproperty.php?core=1.1&name=maxMemoryAllocationSize&platform=all) |
| `subgroupSize` | 32 | 64 | -- | 2.1 |
| `timestampPeriod` (ns) | 1 | 10 | -- | Intel and Adreno 52.08; lavapipe 1 |
| `shaderFloat64` | true | true | -- | 3.1 |
| `shaderInt64` | true | true | -- | false on Iris Xe and Apple |
| `bufferDeviceAddress`, `storageBuffer8BitAccess` | true, true | true, true | required in 1.3 / 96.19 % | |
| `shaderBufferFloat64AtomicAdd` | true | **false** | -- | 7.66 % overall |
| `shaderDenormPreserveFloat32` / `Float64` | **false / false** | true / true | -- | Intel true; Adreno and Apple false for fp64 |
| `shaderRoundingModeRTEFloat64` | true | true | -- | Adreno and Apple false |
| `shaderFmaFloat64` (`VK_KHR_shader_fma`) | true | not read | -- | on NVIDIA 580+, AMD 2.0.388+ and RADV 26.x, ANV 26.x, MoltenVK |
| compute queues | 16 graphics+compute, 8 compute-only, 2 transfer | 8 graphics+compute, 8 compute+transfer, 1 transfer | 1 | RADV 1 + 4; Intel 1 + 4; Adreno 4 + 1; lavapipe 1 |
| host-visible device-local heap | yes, 33.7 GB heap | yes, 17.1 GB heap | -- | Intel discrete yes; Adreno unified |

Sources for the two columns: https://vulkan.gpuinfo.org/displayreport.php?id=51761 and
https://vulkan.gpuinfo.org/displayreport.php?id=51731; the spec's tables at
https://docs.vulkan.org/spec/latest/chapters/limits.html and
https://docs.vulkan.org/spec/latest/appendices/versions.html. **Two consequences for the AMD Windows machine**:
block (a)'s tile needs the half-tile form above four textures (5.2), and a level-0 stencil over 2 GB -- 2048 x 2048 at
`--c0 4`, 4096 x 4096 at `--c0 2` -- is refused unless the split of 4.8 is built.

### 3.7 Debugging on the device

There is no sanitizer for device code. The substitutes: `VK_LAYER_KHRONOS_validation` (present on every SDK install
and in the distributions' `vulkan-validationlayers` packages the README's tables already name, `README.md:138-141`),
its GPU-assisted mode for out-of-bounds buffer accesses, `debugPrintfEXT` for a printed value from a shader
(`VK_KHR_shader_non_semantic_info`, in the same layer), and the harness, which locates a wrong value to a kernel and
a plane on identical inputs -- the instrument the CPU port found worth more than any other
(`docs/CPU_PORT_LESSONS.md` section 6). `RenderDoc` captures compute dispatches and reads back buffers, which is the
inspector when a number is wrong and the harness has said where.

### 3.8 Precedent

NVIDIA's own texture-compression SDK runs its compressor in **CUDA only** ("GPU for NTC compression: Minimum: NVIDIA
Turing") and uses Vulkan and D3D12 for decompression (https://github.com/NVIDIA-RTX/RTXNTC). VkFFT does run fp64 --
"single, double, half and quad (double-double) precision support" -- through Vulkan (https://github.com/DTolm/VkFFT),
so fp64 least-squares-class work in Vulkan compute is done in the wild, on the vendors that have fp64; ncnn stays at
fp32/fp16/int8 (https://github.com/Tencent/ncnn/wiki/vulkan-notes). A 2024 paper compares native and emulated double
precision in Vulkan GLSL (https://arxiv.org/abs/2408.09699), which is the reference to read before any df64 attempt.

---

## 4. Architecture

### 4.1 The files

New, beside the two existing arms (the shape `docs/CPU_BACKEND_PLAN.md` 1.6 and 1.8a laid down):

```
src/vk/vk_backend.h          namespace nntc_vk: the 46 seam declarations over an opaque VkModel*, plus
                             bytes_for(m) for the banner, and implemented(seam) for the harness
src/vk/vk_backend.cpp        the 46 bodies that are pure routing or transfers; the refusals while stages land
src/vk/vk_context.{h,cpp}    instance, layers, physical-device enumeration and the feature/limit check,
                             logical device, the compute queue, the command pool, fences, timestamp pool
src/vk/vk_memory.{h,cpp}     a few large VkDeviceMemory blocks, sub-allocated bump-style; staging ring;
                             VkBuffer + device address per model buffer; the byte total for device_memory
src/vk/vk_pipeline.{h,cpp}   one VkPipeline per kernel from embedded SPIR-V, one pipeline layout, the
                             push-constant / parameter-UBO convention, a dispatch helper with barriers
src/vk/vk_model.h            VkModel: DeviceModel's twin with VkBuffer handles in place of pointers,
                             PlaneWork copied as the CPU arm copied it (cpu_model.h)
src/vk/vk_init.cpp           init.cu's drivers; fit_range and jacobi_eigen copied
src/vk/vk_solve_decoder.cpp  block (a)'s driver; the ridge ladder, tri_index and gauss_jordan copied
src/vk/vk_solve_level1.cpp   blocks (b) and (c'): assemble_all, the CG loop as command buffers, the sweeps,
                             the form and monomial matrices copied, solve_level1_dense_host copied
src/vk/vk_solve_level0.cpp   block (c)'s driver
src/vk/vk_refine_bc.cpp      the refinement's drivers; the BC palette helpers copied
src/vk/vk_objective.cpp      the objective's driver, normalise and twin_sample copied, objective_check_host copied
src/vk/vk_probe.{h,cpp}      the harness's reads and writes of the arm's workspaces (the twin of
                             backend_check_probe.cu, section 6.1)
src/vk/shaders/nntc.glsl     the shared header: the constants of device.cuh:78-165, sample.cuh entire,
                             model.h's replicate_index / level1_value / level1_index, the buffer_reference
                             types, the parameter block layouts
src/vk/shaders/*.comp        one file per kernel, 37 files, named k_<kernel>.comp
```

About 5,500 to 7,000 lines of host C++ and 2,200 to 2,800 lines of GLSL (section 5.4). The duplication the CPU
plan counted at about 1,160 lines (its 1.8) recurs: the same host math (the ladder, `gauss_jordan`, `jacobi_eigen`,
`fit_range`, the form matrices, `bc0_value`, the two brute-force twins) is **copied a third time**, for the same
reason and under the same guard (the hash arm keeps the originals still; the harness holds the copies to the bit).
Whether to lift that shared host math into one file all three arms include is the one place this plan would revisit
decision 1's spirit, and it is recorded as a decision for the owner (section 7.5), not taken here.

### 4.2 The dispatcher changes

* `be::Backend` gains `Vulkan` (`src/backend.h:44-49`); `be::Device` gains `nntc_vk::VkModel* vk` (`backend.h:52-57`);
  `backend_name` gains `"vulkan"` (`backend.cpp:62-65`); `cuda_built_in()` gets a sibling `vulkan_built_in()`
  (`backend.cpp:94-101`).
* Every forwarder in `backend.cpp` gains one branch, `if (d->backend == Backend::Vulkan) { ... nntc_vk::...; }`, inside
  `#ifdef NNTC_VULKAN`. That is 46 mechanical edits in the file whose comment says it should read as identical things
  (`backend.cpp:5-6`). `docs/CPU_BACKEND_PLAN.md` 1.8a said that at a fourth arm the forwarders "should become a table
  of function pointers filled in by each arm at registration"; with `Check` already a third branch in the same
  `if` chain, the Vulkan arm is the fourth, and the table is the right time. **Recommended: do the table in stage V0**
  (section 7.1), because the harness needs it anyway (6.1). It is a change to `backend.cpp` and the three arm
  headers only; no kernel file is touched.
* `main.cpp:1514-1589` (`select_backend`): `"vulkan"` joins the accepted names at `main.cpp:1476`; the fallback
  order under `auto` is discussed in 4.4. The usage line at `main.cpp:332` grows by one word.
* `print_banner` (`main.cpp:1593-1690`): the `backend:` line already prints whichever name (1663); the device row
  at 1664-1665 is CUDA-shaped (`compute capability %d.%d`) and gets a Vulkan sibling: `device N: <name> (<kind>,
  Vulkan X.Y, driver <version>)`, the shape the viewer prints at `viewer_vk/main.cpp:609-611` minus the queue family,
  plus the driver version the viewer never prints. The CUDA banner stays byte-identical.
* The report's two backend-conditional paragraphs (`main.cpp:3274-3283, 3308-3317`) say "CUDA events" under every
  backend that is not `cpu`; under `vulkan` they should say "timestamp queries", and the per-plane overlap sentence
  depends on the queue design (2.5): with one queue the planes do not overlap and the CPU wording ("SUM to the block")
  is the true one.
* `CMakeLists.txt`: a `NNTC_VULKAN_SOURCES` list beside `NNTC_CUDA_SOURCES` (102-116), a probe beside the CUDA one, a
  `NNTC_VULKAN` define, and one word on the `nntc_encode backends:` line (242) that `tests/run_checks.py:254` parses.
  Section 4.6.

### 4.3 The Vulkan arm's lifecycle

`device_select(index, info)`:

1. `vkEnumerateInstanceVersion`; a loader below the floor is `info.error = "the Vulkan loader is older than 1.x"` and
   `false` (the viewer's text at `viewer_vk/main.cpp:384-395` is the model).
2. Create the instance with no surface extension (the viewer's `create_instance(false)`, `main.cpp:383-448`, is
   exactly this; `VK_LAYER_KHRONOS_validation` when present and `NDEBUG` is unset, as `main.cpp:320-324` does).
3. Enumerate physical devices. `--device N` is an index into that enumeration; without it, the first device that
   passes the check below, replaced by the first discrete GPU that passes (the viewer's rule, `main.cpp:553-559`).
4. **The check**, every item of which is a refusal with a reason in `info.error`, in the tree's plain words:
   * `apiVersion` at or above the floor (4.7 decides 1.2 or 1.3);
   * a queue family with `VK_QUEUE_COMPUTE_BIT` (the viewer demands graphics, `main.cpp:452-466, 477-480`; the encoder
     must not, or a compute-only device and a headless server part are refused for nothing);
   * `shaderFloat64` (section 3.1); `bufferDeviceAddress`; `storageBuffer8BitAccess`; `shaderInt64` if the address
     arithmetic path of 2.2 needs it;
   * `maxComputeWorkGroupInvocations >= 256` and `maxComputeWorkGroupSize[0] >= 256`;
   * `maxComputeSharedMemorySize` against **this layout's** need: `(mm + nout) * 257 * 4` for block (a)
     (`solve_decoder.cu:281`) or its half-tile variant (5.2), and `(n^2 + 3n + 80) * 8 + 120` for the refinement at
     `n = 16 c0` under `--l0 bc8` (`refine_bc.cu:545`); `bytes_for(m)` can compute both before anything is allocated;
   * `maxStorageBufferRange` and `maxMemoryAllocationSize` against the largest buffer of this model (4.8);
   * `timestampComputeAndGraphics` or the queue family's `timestampValidBits > 0`, else the report's ms columns are
     zero and the banner says so (a WARNING, not a refusal).
5. Fill `DeviceInfo` (`model.h:255-261`): `name` from `deviceName`; `major`/`minor` from the API version so the
   existing `cuda_refusal` shape at `main.cpp:1501-1512` does not print "compute capability" for a Vulkan device
   (a Vulkan-specific text is needed there, one more branch).

`device_create` allocates `VkModel` exactly as `init.cu:151-296` allocates `DeviceModel`, buffer for buffer, so
`device_memory` reports the same byte total under the same name (plan 1.10; the CPU arm did the same,
`docs/CPU_BACKEND_PLAN.md` C1 notes). `device_destroy` waits for the queue to idle and frees everything; a failure to
free is a WARNING, as `init.cu:303-311` does.

### 4.4 `--backend vulkan`, and what `auto` does

* `--backend vulkan` by name: the check of 4.3 passes or the run ends with an ERROR naming the reason and the remedy
  (`run --backend cpu, or --backend auto`), the exact policy `main.cpp:1576-1585` applies to `cuda`.
* `--backend auto`: **recommended order `cuda`, then `vulkan`, then `cpu`**, each step falling through with the
  existing WARNING shape (`main.cpp:1586`). CUDA first because it is the reference and, on the one machine that has
  both, at least as fast (8.1). Vulkan second only when the device passes 4.3's whole check, including a **rate
  gate**: a device whose `shaderFloat64` is a software emulation (section 3.1: Intel Gen11, Xe-LP and Alchemist under
  Mesa) would pass the feature check and then encode slower than the CPU backend. The device cannot say "emulated",
  so the gate is a list -- refuse Vulkan under `auto` for a vendor ID / device ID class known to emulate, with the
  WARNING saying so -- or a measurement: time one `k_ls_accumulate` dispatch on a synthetic 256 x 256 plane at start-up
  and fall back when it is slower than a fixed bar. **Recommended: the list, because a start-up benchmark is a source
  of run-to-run variance in the banner, and the list is three entries.** Owner's decision (7.5).
* **Until the corpus of section 6.3 has been measured on the AMD machines, `auto` should not choose `vulkan` at
  all** (stage V8 turns it on). The CPU arm was made the fallback only after its cross-backend numbers were in
  (`docs/CPU_BACKEND_PLAN.md` 7.4).
* `--device N` means the CUDA index under `cuda` and the Vulkan enumeration index under `vulkan`; under `auto` it is
  read by whichever arm ran, and the banner says which enumeration. The two enumerations are different lists (CUDA's
  has only NVIDIA devices), so a `--device 1` that meant the Radeon under Vulkan and a nonexistent device under CUDA is
  possible; the WARNING that falls back must name the index it tried.
* `-j` stays the CPU arm's and warns as `main.cpp:1521-1523` does.

### 4.5 Sharing with the viewer, without entangling them

The viewer's reusable Vulkan plumbing is about 650 to 700 lines of `viewer_vk/main.cpp`: memory-type helpers
(330-348), instance and debug messenger (318-448), queue-family and device probes (452-499, 689-699, needing a compute
variant), `pick_physical_device` (535-624), `create_device` (847-967, minus swapchain, BC and anisotropy), the command
pool, barrier, one-shot submit and `create_buffer` (969-1048), the shaderc path (2131-2240), the timestamp pattern
(3103-3158) and part of the teardown (3430-3499). **Every one of them reads file-scope statics** (`g_instance`,
`g_phys`, `g_device`, `g_queue`, `g_queue_family`, `g_command_pool`, at 218-245) and is `static` itself;
`docs/VULKAN_VIEWER_PLAN.md:1005-1008` chose "shared source, no library" for the viewer, and there is no target to
link against. `VK_CHECK` (`viewer_vk/vk_check.h:36-44`) calls `exit(1)`, which suits a viewer and does not suit a
backend that must return `false` from `device_select` so that `auto` can fall back.

Three options:

1. **Copy.** The encoder's `src/vk/vk_context.cpp` transcribes those functions with the statics replaced by a context
   struct and `VK_CHECK` replaced by a returning variant. About 700 lines duplicated between two programs that share
   nothing else, with the viewer untouched. This is what the CPU port did with the kernel-side host math, for the
   same reason: a copy cannot change what the original computes.
2. **Extract into `shared/`.** Move the context, memory and shader-compile helpers into `shared/vk_context.{h,cpp}`
   taking a context struct, and make the viewer call them. Touches the viewer, whose gate pins its start-up lines
   (`tests/run_checks.py:3720-3729`) and whose byte-identical screenshot arms would have to be re-run on every device
   the viewer has been on; the encoder's needs (compute queue, no surface, returning errors, a compute pipeline path)
   are different enough that the shared code would grow two modes.
3. **Extract later, once the encoder's version exists and has been gated**, and only if the two copies have been a
   maintenance cost. The CPU plan's rule for its own duplication (1.8): "if the drift ever bites twice, revisit
   deliberately rather than loosen a tolerance".

**Recommended: 1 now, 3 as the standing rule.** What the encoder takes from the viewer regardless: the messages and
their wording (`ERROR: no Vulkan device: ...`, which `tests/run_checks.py:630-631, 3704-3710` uses as the skip
signal), the device-list print shape (`main.cpp:547-551`, parsed at `run_checks.py:3740-3742`), the validation-layer
convention (Debug only, present or not, `main.cpp:320-324, 412-416`), the no-optimisation shaderc setting the Qualcomm
compiler needed (`viewer_vk/README.md:322-324`), and the CMake detection block (`CMakeLists.txt:291-366`), which
moves out of the viewer-only `else()` at 331 so that both targets read the same `Vulkan_FOUND`, `NNTC_SHADERC` and
`NNTC_VK_HEADER_VERSION`.

### 4.6 CMake: detection and optionality

* `find_package(Vulkan)` already runs unconditionally at `CMakeLists.txt:291`. The encoder's arm needs the loader and
  the headers; it does **not** need shaderc if the shaders are compiled at build time (2.6), which is why the
  detection of the loader must be separated from the detection of shaderc that the viewer's `else()` chain at
  321-331 folds together.
* `FindVulkan` exposes `Vulkan::glslc` / `Vulkan_GLSLC_EXECUTABLE` where the SDK or a distribution's `glslc` package
  is installed. A build-time rule compiles each `src/vk/shaders/k_*.comp` into `.spv` under the build directory and a
  generated `vk_shaders.inc` of `uint32_t` arrays. With no `glslc` and a shaderc found, fall back to the viewer's
  run-time compile of the tracked GLSL; with neither, the arm is compiled out and the configure prints a **WARNING**
  in the file's own rule (`CMakeLists.txt:15-18`: anything left out is a WARNING).
* `NNTC_VULKAN` is defined in the same block that adds the sources, so define and arm cannot disagree, exactly as
  `NNTC_CUDA` (157-163). The `nntc_encode backends:` line (235-242) says `cpu`, `cpu and cuda`, `cpu and vulkan` or
  `cpu, cuda and vulkan`; `tests/run_checks.py:254` reads it.
* The three `-ffp-contract=off` / MSVC notes (186-205) are host-side and unaffected; the shader-side counterpart is
  the `precise` rule of 2.4.
* A Vulkan-only configure (no nvcc) must link clean for the same reason the CPU-only one must (`CMakeLists.txt:44-47`):
  it is the proof that no host code names a CUDA symbol. It already does, since the Vulkan arm adds none.

### 4.7 Minimum Vulkan version and required features

Two defensible floors:

* **Vulkan 1.3**, the viewer's floor (`viewer_vk/main.cpp:82-84, 401`): one story in the README, `synchronization2`'s
  `vkCmdPipelineBarrier2` and `vkQueueSubmit2` in core, `VK_KHR_subgroup_size_control` in core (not needed here),
  and every driver in the owner's fleet except the Windows Dozen layer reports it (`viewer_vk/README.md:316-321`).
* **Vulkan 1.2**, the smallest that has `bufferDeviceAddress`, `storageBuffer8BitAccess`, the float controls,
  `shaderInt64`'s promotion and timeline semaphores in core. It admits a few more drivers and costs the 1.0 barrier
  API's verbosity.

**Recommended: 1.3**, for uniformity with the viewer and because every device that has fp64 and matters here reports
it (section 3). The required features are then: `shaderFloat64`, `bufferDeviceAddress`, `storageBuffer8BitAccess`,
and optionally `shaderInt64` (2.2), `shaderDenormPreserveFloat32` / `Float64` and `shaderRoundingModeRTEFloat64`
through the float controls (3.4), `timestampComputeAndGraphics` (report timings), `VK_EXT_memory_budget` (the OOM
message's free-memory figure, `device.cuh:20-35`'s Vulkan twin). No extension outside core 1.3 is *required*.

### 4.8 Memory: the buffer sizes, and the limits they meet

The largest buffer the encoder allocates is block (c')'s stencil, `n0 * 5 * c0^2 * 8` bytes per plane
(`init.cu:204`): 671 MB at 2048 x 2048 `--c0 2`, **2.7 GB at 2048 x 2048 `--c0 4`**, 2.7 GB at 4096 x 4096 `--c0 2`,
**10.7 GB at 4096 x 4096 `--c0 4`**; level 1's stencil is a sixteenth of that (`init.cu:189`). CUDA's `cudaMalloc`
takes any size the card has. Vulkan has two ceilings a driver may set well below the card's memory:
`maxStorageBufferRange`, the most a shader may address through one binding (spec minimum 128 MB), and
`maxMemoryAllocationSize`, the most one `vkAllocateMemory` may return (spec minimum 1 GB); section 3.6 has the
per-vendor values. Three consequences:

* buffers must be addressed through `bufferDeviceAddress` (2.2), not bound descriptors, or the 128 MB minimum range
  would refuse a 2048 x 2048 material outright on a driver that reports the minimum;
* a buffer larger than `maxMemoryAllocationSize` cannot exist. Either the arm **refuses the layout on that device**
  with an ERROR naming the buffer and the limit and pointing at `--backend cpu` or `cuda` (the `CUDA_OOM_ADVICE`
  shape, `device.cuh:38-39`), or the stencil is split into row bands in separate allocations with a per-band base
  address in the shader (one more indirection in `k_stencil_assemble0`, `k_matvec`, `k_quant_sweep`, `k_bc_refine`,
  `k_diag_partial`, `k_precondition`: six kernels). **Recommended: refuse first, split later if the AMD machines
  need it**; the 4 GB class covers every material in `docs/RESULTS.md` except the 4096 x 4096 `--c0 4` case;
* `maxMemoryAllocationCount` (spec minimum 4096) against the roughly thirty buffers per plane and twenty per model
  (`init.cu:177-273`): under 500 at the deepest chain, but the arm should sub-allocate a few large blocks anyway, so
  that `device_memory`'s figure and the driver's are the same number.

The pinned staging (`init.cu:277-290`) becomes one host-visible, host-coherent buffer (host-cached when offered, the
viewer's readback choice at `viewer_vk/main.cpp:3215-3226`); uploads go through a persistently mapped staging ring
and `vkCmdCopyBuffer`.

---

## 5. Kernel changes

The rule is the CPU port's (`docs/CPU_PORT_LESSONS.md` section 4): a port does the same computation; only the
compiler's floating-point behaviour should differ. So the default for every kernel is a line-for-line transcription
with the same block size, the same grid, the same shared arrays and the same barriers. The exceptions below are each
forced by a Vulkan limit or by a Vulkan feature that is not universal, and each is written so that the arithmetic --
the operands and the order of their additions -- is unchanged.

### 5.1 Nearly line for line (28 of 37)

`k_init_level0`, `k_init_level0_cont`, `k_quantise_level1`, `k_residual`, `k_seed_channel`, `k_decode`, `k_objective`,
`k_obj_reduce`, `k_cov_partial`, `k_proj_peak`, `k_ls_reduce`, `k_stencil_assemble`, `k_stencil_assemble0`,
`k_dot_partial`, `k_reduce_sums`, `k_diag_partial`, `k_reduce_diag`, `k_precondition`, `k_apply_precond`, `k_matvec`,
`k_negate`, `k_axpy`, `k_xpby`, `k_cg_alpha`, `k_cg_beta`, `k_add_correction`, `k_range_partial`, `k_reduce_range`,
`k_movement_partial`, `k_quant_sweep`, `k_bc_decode`, `k_level0_search`.

What "line for line" costs in GLSL, kernel by kernel where it is more than syntax:

* **Pointers become buffer references** (2.2). `sample_bilinear(const float* plane, ...)` (`sample.cuh:61-71`)
  becomes a function over a `buffer_reference` type and an element index; `plane + ((size_t)t.y0 * w + t.x0) * c`
  becomes an index into `plane.v[]`. `site_row`, `build_phi`, `bilinear_tap_pixel`, `subtexel_offset`, `site_delta`
  (`sample.cuh:43-178`) transcribe as they stand, with `floorf` -> `floor` (exact) and `(int)` -> `int()`
  (truncation toward zero in both, GLSL 4.60 section 5.4.1).
* **`lroundf` has no GLSL equivalent.** GLSL `round()` is implementation-defined at exactly .5 (section 3.4). Every
  `lroundf` argument in the tree is non-negative -- `acc * 255.0f` in [0, 255] at `objective.cu:73`,
  `(value + 1) * 0.5 * levels` in [0, levels] at `init.cu:1122` -- so `floor(x + 0.5)` is `lroundf(x)` exactly there;
  `refine_bc.cu:341, 171` already spell it that way. The GLSL header defines one `lround_nonneg` and every call site
  uses it, with a comment naming the precondition.
* **`size_t` and `long long` become 32-bit.** Element indices fit `uint` under the 4 GB allocation ceiling of 4.8
  (2^29 doubles). `k_init_level1_box` and `k_block_means` multiply a coordinate by a width in `long long`
  (`init.cu:497-498, 756-757`); with inputs refused above 16384 on either axis (`docs/CPU_PORT_LESSONS.md` section 10),
  the product is under 2^28 and an `int` gives the identical quotient. `k_level0_search`'s `long long total` is at most
  256 (`solve_level0.cu:201-204`, `total_bits <= 8`). `shaderInt64` is therefore **not required by any kernel body**;
  only the pointer path of 2.2 may want it.
* **`ProjDir` by value** (`init.cu:1050-1054`, 288 bytes) exceeds the 128-byte push-constant guarantee; it goes in the
  per-dispatch parameter block (2.3). No arithmetic changes.
* **`k_cov_partial`'s 2-D grid** is `gl_WorkGroupID.y` for the entry (`init.cu:788`); `gl_NumWorkGroups.x` stands
  for `gridDim.x` in the stride. Identical order.
* **`k_quantise_level1` in place**: `in == v1` at the level-0 call sites (`init.cu:692, 732`). One invocation reads
  and writes one element; fine in GLSL as long as the same address is used for both references (aliasing through two
  `buffer_reference` variables is allowed; the compiler may not assume they differ unless told, and `restrict` is not
  used).
* **`k_objective`** is the fp64 workhorse and transcribes as written: `obj_sample` in double, the three trees over
  `sh[3 * 256]`. The `precise` rule (2.4) applies to every double expression in it, because E's summation order and
  rounding decide the asset (`objective.cu:134-136`).
* **`k_stencil_assemble` / `k_stencil_assemble0`**: `sums[75]` doubles plus `g`, `mono`, `gmat[72]` floats per
  invocation. In CUDA this is a register-pressure kernel already (`solve_level1.cu:127-130` says the accumulator was
  redesigned to stay in registers). A Vulkan driver's compiler may spill it differently; that is a performance risk
  (7.4, risk 5), not a correctness one, and it is measured in stage V5 against the CUDA time.
* **`k_level0_search`**: the shadowed `dx, dy` (`solve_level0.cu:98-99, 107`) must be renamed in GLSL, which forbids the
  shadowing anyway in that position; the CPU port renamed them `ox, oy` / `sdx, sdy` (`docs/CPU_BACKEND_PLAN.md` C6).
  The race the CPU port hoisted (its 3.4) is a GPU idiom, harmless here, and the GLSL keeps the kernel's order.
* **The per-element grids** exceed 65,535 workgroups at large planes: 16384 x 16384 / 256 is 1,048,576, and the
  four-colour sweeps and the refinement reach 262,144 per colour at 4096 x 4096. `maxComputeWorkGroupCount[0]` is
  guaranteed only to 65,535 (3.6). Where a driver reports the minimum the dispatch is split into slices of 65,535 with
  the slice base in the parameter block, or a grid-stride loop is added to the per-element kernels. Neither changes
  what any element computes; the fixed-grid kernels (512 and 256 workgroups) are unaffected.

### 5.2 Restructured, and why

**`k_ls_accumulate`** (`solve_decoder.cu:87-161`). Its staging tile needs `(mm + nout) * 257 * 4` bytes of shared
memory, 44,204 at the caps. A device with 32 KB of shared memory (section 3.6 names which) refuses the layouts above
about four textures at `--c0 4 --c1 4`. **The half-tile form**: stage 128 sites, walk the entries adding
`a[p] * b[p]` for `p` in 0..127 into the same `double sum`, stage the next 128 sites of the same 256-site chunk, and
continue the same `sum` for `p` in 128..255 before `acc[slot] += sum`. The 256 products are added in the same order
into the same accumulator, so **the arithmetic is bit-identical to the kernel's**; only the number of barriers per
chunk doubles. Selected at pipeline build from `maxComputeSharedMemorySize`, and the harness holds both forms to the
same bar. `LS_STRIDE` may stay 257 or become 256 -- addressing only.

**`k_bc_refine`** (`refine_bc.cu:178-481`). About 35 KB of shared memory at `--c0 4`; 20 KB at `--c0 3`. On a 32 KB
device `--l0 bc8 --c0 4` cannot run this kernel as written. Options, in order of preference:

1. **refuse the combination on that device** (an ERROR naming `--c0 4`, the refinement and the limit, and pointing at
   `--c0 3`, `--bc-refine 0` or another backend). Every desktop vendor of section 3.6 offers 48 KB or more, so this
   bites only on Adreno and MoltenVK class devices, which the fp64 requirement already excludes;
2. keep `a2` (the `n x n` block, 32 KB of the 35) in a per-workgroup slice of a global scratch buffer and the rest in
   shared memory, dispatching in slices of a few thousand workgroups so the scratch stays under 200 MB. Same
   arithmetic; slower; more host code.

**Recommended: 1.** The two `atomicAdd` on `double` (478-479) are replaced by **one slot per workgroup**: each block
writes its `taken` and a 0/1 flag into `stats_blocks[gi]`, and a one-workgroup second pass (or the host, on the
downloaded array) adds them in block-index order. This removes the need for `VK_EXT_shader_atomic_float`'s fp64 add,
which is not universal (3.3), and makes the report's `delta_e` deterministic where CUDA's is not
(`docs/DESIGN.md:1016-1020`); the CPU arm did the same fold (`docs/CPU_BACKEND_PLAN.md` 3.4's audit row). Nothing
downstream reads the figure. The 64-slot `refine_reduce` tree and its barriers stay exactly as written; GLSL's
`barrier()` is the block barrier (2.1).

**`k_proj_hist`** (`init.cu:1094-1104`). `atomicAdd` on a 64-bit counter. The counts are texels of one plane per bin,
and their total over the chain is under 2^29 at the 16384 limit, so **32-bit `atomicAdd` on `uint` bins is exact and
identical**, and the 4096-bin array becomes `uint[4096]` on the device with the host summing into its existing
`unsigned long long` vector (`init.cu:1344-1373`). `shaderBufferInt64Atomics` is then not needed. The buffer aliasing
(`init.cu:1345` reuses `d->cov` as the histogram) becomes a separate small buffer.

**The per-kernel launch parameters.** Every kernel takes its arguments by value (`k_level0_search` has ten pointers,
fourteen ints and a `ChannelBits`, `solve_level0.cu:75-79`). In GLSL they become one `std430` parameter block per
dispatch (2.3), filled by the driver exactly where the CUDA driver fills the launch. Nothing computed changes.

### 5.3 What should run on the host instead

Nothing that runs on the device today. The three one-invocation kernels (`k_reduce_sums`, `k_reduce_diag`,
`k_reduce_range`, `k_cg_alpha`, `k_cg_beta`, `k_obj_reduce`, `k_ls_reduce` at 64) exist so that the CG loop never
waits for the host (`solve_level1.cu:94-99, 1104-1109`); moving them to the host would add a fence wait per
iteration and change nothing numerically. They stay as one-workgroup dispatches. What is already host code stays host
code: the ridge ladder, `gauss_jordan`, `jacobi_eigen`, `fit_range`, the form matrices, the two brute-force twins, the
percentile scan, the BC pack seed and the block writer (`export.cpp`).

### 5.4 Estimated GLSL and host line counts

Device code today, kernel bodies plus their `__device__` helpers, by file (counted from the line ranges of 1.2):
`init.cu` about 220, `solve_decoder.cu` about 90, `objective.cu` about 135, `solve_level0.cu` about 210,
`solve_level1.cu` about 700, `refine_bc.cu` about 355, `sample.cuh` 178 and the constants of `device.cuh` about 90:
**about 1,980 lines of CUDA device code**. GLSL is longer for the same content -- no pointers, a parameter block per
kernel, explicit `precise`, no `size_t`, one file per entry point with its own preamble -- so the estimate is:

| file | GLSL lines (estimate) |
|---|---|
| `nntc.glsl` (constants, sampling rule, grid rule, reference types, parameter layouts) | 350 to 400 |
| `init.cu`'s ten | 350 to 420 |
| `solve_decoder.cu`'s two | 140 to 170 |
| `objective.cu`'s three | 190 to 230 |
| `solve_level0.cu`'s one | 250 to 300 |
| `solve_level1.cu`'s nineteen | 750 to 900 |
| `refine_bc.cu`'s two | 380 to 450 |
| **total** | **2,400 to 2,900** |

Host: the plumbing of 4.1 (context, memory, pipelines, dispatch, probe) 1,600 to 2,200 lines; the 46 seam bodies and
the drivers transcribed from the six files' host halves (about 1,900 lines of CUDA host code today) 2,200 to 2,800;
the copied host math about 1,160 (plan 1.8); the harness generalisation of 6.1 in `src/backend_check.cpp` (1965 lines
today) 400 to 700 changed lines. **About 5,500 to 7,000 new lines of C++**, against the CPU arm's 5,200 in `src/cpu/`
plus 1,965 of harness.

---

## 6. Verification, modelled on the CPU port

### 6.1 The per-kernel harness, generalised to a reference arm and a candidate arm

`--backend check` today runs the CUDA arm as the trajectory, mirrors its pre-call state into the CPU model through
the transfer functions, runs the CPU side on identical inputs and diffs (`docs/CPU_BACKEND_PLAN.md` 4.1;
`src/backend_check.cpp`). Its probe, `src/backend_check_probe.cu:15-58`, downloads the CUDA arm's normal equations,
workspaces (`stencil`, `grad`, `precond`, `cg_x`, `prev`) and monomial matrices; the CPU side is written into
directly. The harness is hard-wired to those two arms and to CUDA as the reference.

The Vulkan port needs it in two configurations that do not exist today:

* **CUDA against Vulkan**, on the RTX 5090: the same hardware fp64 units under two compilers, the purest comparison
  the port can have;
* **CPU against Vulkan**, on every machine that has no NVIDIA card -- which is every machine the backend exists for.
  There the **CPU arm is the reference**, which the CPU port's own results justify: byte-identical to CUDA on 19 of 21
  corpus cases (`docs/CPU_PORT_LESSONS.md` section 6), and every kernel within 1e-13 on identical inputs.

So stage V0 (7.1) makes the harness **arm-agnostic**: a reference arm and a candidate arm named on the command line
(`--backend check --check-ref cpu --check-arm vulkan`; the default pair stays `cuda`/`cpu` so today's gate lines are
unchanged), each arm supplying (a) its 46 seam functions through the function table of 4.2, (b) a probe with
`read_workspace`, `write_workspace`, `read_normal_equations`, `read_monomials` (the CUDA one exists; the CPU one is a
memcpy; the Vulkan one is `vk_probe.cpp`, a download or upload through the staging ring after a queue-idle wait, the
twin of `backend_check_probe.cu:20, 41, 53`). The mirroring, the colour-by-colour and pass-by-pass splitting of
block (c) and the refinement (`docs/CPU_BACKEND_PLAN.md` C6 and C7 notes), the tolerances and the flip counting are
untouched. Estimated 400 to 700 changed lines in `backend_check.cpp`, no kernel file touched.

**The bars.** The CPU port's bars were set by measurement (`docs/CPU_BACKEND_PLAN.md` C1 and C3 notes): transfers and
integers exact; an index one step off a FLIP; `BAR_ENTRY_DIRECT` 1e-6 of the array's scale for kernels with no solve
between input and output; `BAR_ENTRY` 1e-5 downstream of a solve; a summed scalar 1e-9 relative. **The same bars are
the starting point for Vulkan and are expected to hold**, with one difference the CPU port did not have: section 3.4's
division and square-root precision means a Vulkan arm may sit a few ULP from the reference **even where the CPU arm
sat at zero**, on a device whose driver does not round division exactly. The harness therefore prints, as it does for
`lambda` and the CG residual, the worst difference per quantity, and stage V2 measures on the 5090 whether the
"identical bits" rows of the CPU port's tables (the snaps, the inits, the assembly, block (c)'s search) are still
identical under Vulkan on NVIDIA. If they are, the Vulkan arm on NVIDIA is held to the same rows; if a row is a few
ULP off, the cause is read from the SPIR-V (a contracted multiply-add the `precise` rule missed, or a division) before
the bar is loosened, and the measurement is written beside the bar.

### 6.2 What the harness holds exact, and where Vulkan can legitimately differ

| quantity class | CPU port | Vulkan arm, expected |
|---|---|---|
| integers: `k0`, `k1`, BC endpoints and selectors, moved counts, the ridge rung, iterations | exact | exact on identical inputs, **except** where a double division decides a rounding (`k_quant_sweep:1082`'s `b / a` then `level1_index`) or a comparison sits within an ULP; counted as FLIPs, expected to be zero on NVIDIA and AMD |
| fp64 reductions on identical inputs (the CG chain, the dots, the diagonal, `k_ls_reduce`) | identical bits | identical bits when the shader is `precise` and no division is involved; the trees are replayed verbatim |
| fp32 features (`site_row`) | identical with `-fmad=false` | identical with `precise` **and** an exactly rounded fp32 division in `bilinear_tap_pixel` (`sample.cuh:46-47`), which Vulkan does not guarantee (3.4); measured per device |
| `E` and its companions | 1e-14 (summation order) | the same order, so the same or better; the division `u = (px + 0.5 + dx) / w0` (`objective.cu:146-147`) is the one place a non-exact double division would show, at 1e-16 |
| the refinement's `delta_e` | printed, not asserted | deterministic under the per-slot fold of 5.2 |

### 6.3 The gate arms

Every arm follows the pattern the Vulkan viewer's arms already follow (`tests/run_checks.py:583-598, 630-631,
658-664, 3690, 3708`): skipped **and recorded as skipped** when the backend is not built or the machine has no
usable device, never reported as run.

* **`--backend vulkan` on the whole gate.** `run_checks.py --backend cpu` exists (`run_checks.py:101-107, 4637-4643`)
  and appends `--backend cpu` to every `nntc_encode` line; `--backend vulkan` is the same one-word change, writing
  under `out_vulkan/`. It buys the fifteen byte-for-byte arms (including `determinism_check`: two Vulkan encodes of one
  image byte-identical, which the design of 1.3 predicts), the structural arms (E monotone, the round trips,
  `dds_decode.py`'s agreement, the CG residual, the dense twin, the finite-difference gradient, the pack lossless at 2
  and 3 bits, the repack's restore), the refusals and the tree scan, with no new assertion written. Its running time
  is measured in stage V8; the CPU gate took 144 s (`docs/CPU_BACKEND_PLAN.md` stage D notes).
* **The cross-backend arm** (`cross_backend_checks`, `run_checks.py:4386-4450`, hard-coded to `('cuda', 'cpu')` at
  4444) gains the pair `(reference, vulkan)` where the reference is `cuda` when it runs and `cpu` otherwise. Its bars
  are the owner's, set on the CPU port's corpus and kept as they are: any texture worse than -4 dB FAIL, any mip level
  worse than -6 dB FAIL, a mean signed difference over at least five textures below -0.5 dB FAIL and between -0.2 and
  -0.5 a WARNING; byte identity, round counts, stop reasons and ridge tallies **reported, never asserted**
  (`docs/CPU_BACKEND_PLAN.md` 7.4 "what was built"). The one-white-pixel dot keeps its no-bar treatment.
* **A Vulkan device-refusal arm** in the shape of `device_refusal_checks` (`run_checks.py:3203-3221`): `--backend
  vulkan` on a machine without a usable device is an ERROR and exit 1; `auto` is a WARNING and a CPU encode; a refused
  feature (no `shaderFloat64`) prints its name.
* **A validation-layer arm**, which the viewer's gate does not have (the agent's reading of `run_checks.py`: the words
  `validation` and `VK_LAYER` do not occur under `tests/`): the Debug gate's Vulkan encodes of `tests/tiny.png` in
  both level-0 modes require **zero** `validation:` lines on stderr; a Release gate records the arm as skipped because
  the layer is compiled out (`viewer_vk/main.cpp:320-324`'s rule). GPU-assisted validation (bounds checking of every
  buffer access) on the same two encodes once per stage, by hand, because it is slow.
* **The kernel-hash arm is unchanged and is the guarantee** that the port did not touch a CUDA file.
* **Determinism across runs**: `determinism_check` under `--backend vulkan`. **Across devices**: not promised, as
  `docs/DESIGN.md:1025-1030` does not promise it across platforms; the cross-backend arm's table, run on each machine,
  is the record.

### 6.4 The corpus, and the machines

The CPU port's layers (`docs/CPU_PORT_LESSONS.md` section 6): the per-kernel harness, the 21-case corpus including the
4096 x 4096 five-texture material, the wide set (grayscale, palette with alpha, RGBA, non-square, 64 px to 3000 x
1480), the 24 random torture materials at three layouts from `tools/crop_set.py`, the 540-case synthetic sweep, the
determinism arms, ASan and UBSan. The Vulkan arm reuses them all except the sanitizers, which cannot see device code;
their place is taken by the validation layer, GPU-assisted validation and `debugPrintfEXT` (3.7).

| machine | Vulkan device | what it proves |
|---|---|---|
| RTX 5090, Windows | the NVIDIA driver, native | CUDA against Vulkan on one part: the harness at its tightest, the speed comparison of 8.1 |
| RTX 5090, WSL 2 | **no native NVIDIA Vulkan** (3.5); lavapipe is available | the Linux build of the arm, the validation layer under Linux, lavapipe as a software reference on `tests/tiny.png` and the dot images -- correctness only, at software speed |
| Intel CPU + AMD GPU, Windows | AMD's Windows driver, fp64 native | the first machine on which `vulkan` beats `cpu`: the cross-backend arm with the CPU as reference, and the speed figure for 8.1 |
| Kubuntu AMD | RADV | the same under Mesa; the `precise` and denormal behaviour of a second compiler (3.4) |
| Snapdragon X Elite (Adreno X1-85) | native ICD, Vulkan 1.3 (`viewer_vk/README.md:316-321`), **no fp64 expected** (3.1) | the refusal arm and the `auto` fallback; nothing else, unless the owner takes the emulation decision of 7.5 |
| lavapipe, anywhere | software | a device that reports the spec minimums for several limits (3.6): the 65,535-workgroup slicing and the shared-memory fallbacks get exercised here and nowhere else |

Measured numbers from these runs go into this document's section 8 in place of its estimates, in the tree's rule
(`docs/CPU_PORT_LESSONS.md` section 11: numbers come from logs, never estimates).

---

## 7. The staged plan

Working days for one implementer who knows the tree, as `docs/CPU_BACKEND_PLAN.md` section 6 counted them. Every
stage ends at a harness run or a gate run, and every stage keeps the nine hashed files byte-identical. The order is
the CPU port's, because it worked: the dispatcher, then the transfers and the harness, then the kernels in the order
their inputs become available, then the whole encode, then the machines.

### 7.1 The stages

**V0 -- the arm's slot, the table, and the harness made arm-agnostic. 4 to 6 days.** `Backend::Vulkan`, the `vk`
pointer, `vulkan_built_in`, the forwarders turned into the per-arm function table of 4.2, `--backend vulkan` parsed
and refused everywhere with the right message, the CMake list and probe with the arm still empty, and the harness's
reference/candidate generalisation of 6.1 with the CPU arm as a candidate of itself (`--check-ref cuda --check-arm
cpu` must reproduce today's summaries exactly). *Done when* both gates are green, every asset byte-identical to
`v1.3.4b-disclosure`'s over the CPU plan's thirty command lines, the `nntc_encode backends:` line reads as before on
a tree without Vulkan sources, and the kernel hashes did not move.

**V1 -- the plumbing and the transfers. 6 to 8 days.** `vk_context`, `vk_memory`, `vk_pipeline`, `vk_model`, the
shader build rule, `device_select` with the whole check of 4.3, `device_create` with the buffer list of `init.cu`,
the fourteen transfers, `vk_probe`. *Done when* `--backend check --check-arm vulkan` on `tests/tiny.png` reports every
transfer `ok` and every kernel `not implemented`, `device_memory` equals the CUDA figure to the byte, the Debug build
runs with zero validation messages, and the same holds on lavapipe under WSL. Two measurements before any kernel:
the cost of a dispatch plus barrier plus fence round trip on the 5090 (against CUDA's launch and event), and the time
of one empty command buffer submission, which together decide whether the CG loop of V5 needs anything beyond one
queue.

**V2 -- the objective and the decode. 3 to 4 days.** `k_objective`, `k_obj_reduce`, `k_decode`; the copied
`normalise`, `twin_sample`, `objective_check_host`. *Done when* `E` and its companions are within the scalar bar on
every plane of `tiny.png` and one material, `k_decode` exact, the Vulkan objective within 1e-9 of the Vulkan twin, and
the **first measurement of 6.2**: whether the fp32 features and the fp64 sums are bit-identical to the CPU arm's on
NVIDIA under `precise`. This is the stage that answers the "how close can Vulkan be" question with a number.

**V3 -- the initialisations. 3 to 4 days.** The ten shaders of `init.cu` with `k_proj_hist`'s 32-bit bins, the
parameter block for `ProjDir`, the copied `fit_range` and `jacobi_eigen`. *Done when* the harness reports every init
kernel within its bar on the CPU port's 23 command lines, including the residual seed's eigenvector on the five-texture
material that was the CPU port's one outlier (`docs/CPU_PORT_LESSONS.md` section 4) -- the covariance tree is
replayed verbatim here, so that case should agree to the 1e-14 the CPU arm reached after its fix.

**V4 -- block (a). 3 to 4 days.** `k_ls_accumulate` in both its full-tile and half-tile forms, `k_ls_reduce`, the
copied ladder. *Done when* the normal equations are within `BAR_ASSEMBLED` and the rung and tally exact on the CPU
port's 17 lines and the three dot images, in both tile forms on the 5090 (the half-tile form forced by a flag), and
the first speed figure: `k_ls_accumulate`'s dispatch time against the CUDA kernel's event time on the same card.

**V5 -- blocks (b) and (c'). 7 to 9 days.** The two assemblies, the diagonal and lambda, the preconditioner, the
CG loop as command buffers (one per probe interval of four iterations, 2.5), the correction, the range and movement
probes, the sweeps, the copied dense twin. *Done when* the CPU port's stage-C5 table holds for Vulkan: monomials,
stencil, gradient and preconditioner within `BAR_ASSEMBLED`, `iterations` identical, the correction within `BAR_ENTRY`,
the sweeps' indices and moved counts identical, and the three block (b) arms on the Vulkan arm's own terms. Plus the
register-pressure measurement of the assembly (5.1) against CUDA's, which is this stage's risk.

**V6 -- block (c). 2 to 3 days.** `k_level0_search` with the renamed offsets and the sliced dispatch. *Done when* `k0`,
`v0` and the per-colour moved counts are identical after every colour pass on the nine palette lines of the CPU
port's stage C6. `--l0 palette` then encodes end to end on Vulkan, and stays tested on every later change
(the owner's standing rule for the palette mode).

**V7 -- the refinement. 5 to 6 days.** `k_bc_decode`, `k_bc_refine` with the per-slot stats and the shared-memory
refusal of 5.2, the drivers. *Done when* endpoints, selectors, `v0`, `k0` and the correction are identical after every
pass and every colour on the CPU port's eleven lines, the `--l0 bc8` gate arms pass under `--backend vulkan`, and the
whole gate runs `--backend vulkan` on the 5090. **The Vulkan arm encodes end to end.**

**V8 -- the machines, the corpus, the bars, `auto`, the write-up. 7 to 9 days.** The 21-case corpus, the wide set, the
torture materials and the synthetic sweep on the 5090 (CUDA against Vulkan) and on the two AMD machines (CPU against
Vulkan); the cross-backend arm's bars confirmed or re-measured and written beside the arm; the refusal and fallback
arms on the Snapdragon; lavapipe under WSL for the minimum-limits paths; the gate's running time under `--backend
vulkan`; then, and only then, `auto` learns to choose `vulkan` (4.4); `README.md`'s Building and Limits sections and
this document's section 8 replaced with the measured numbers.

### 7.2 Totals

| stage | days | the CPU port's counterpart (measured) |
|---|---|---|
| V0 | 4 to 6 | A + B: 4 to 5 |
| V1 | 6 to 8 | C1: 3 to 4 |
| V2 | 3 to 4 | C2: 2 to 3 |
| V3 | 3 to 4 | C3: 3 to 4 |
| V4 | 3 to 4 | C4: 2 to 3 |
| V5 | 7 to 9 | C5: 5 to 7 |
| V6 | 2 to 3 | C6: 2 to 3 |
| V7 | 5 to 6 | C7: 4 to 5 |
| V8 | 7 to 9 | D: 4 to 5 |
| **total** | **40 to 53 days, 8 to 11 weeks** | 29 to 39 days, 7 to 9 weeks |

Why more than the CPU port for a port whose kernels translate more directly: V1's plumbing is about three times the
thread pool and the plain-array model (a Vulkan program has to own memory types, descriptors or addresses,
pipelines, command buffers, barriers, fences and queries before its first dispatch), V0 carries the harness
generalisation the CPU port did not need, and V8 has four machines and a software device where the CPU port had one
desktop and owed the others. Debugging is also slower: a wrong index on the GPU is a validation message or a wrong
number, not a sanitizer report with a line.

### 7.3 What is out of scope

* **Changing any of the nine hashed files.** Not a kernel, not a header, not `model.h`.
* **Any restructuring for speed** before the faithful port is measured and gated: no scatter assemblies, no
  fp32 accumulation, no subgroup reductions in place of the shared-memory trees, no cooperative vectors. The CPU plan's
  section 9 lists the same temptations with the numbers that answered them; every one applies here.
* **An fp64-free variant** (double-float emulation) of the kernels. Section 3.2 says what it would cost and what it
  would not give; it is a decision for the owner (7.5), not part of this plan.
* **A second queue per plane, timeline semaphores, async transfer queues.** Section 2.5 designs for one queue and
  says when a second would earn its place; that is a measurement in V5, not a plan.
* **Sharing the viewer's Vulkan code by extraction** (4.5, option 2).
* **HIP, Metal, OpenCL, SYCL.** Section 8.3 compares HIP because the owner has named it; nothing here plans it.
* **Making `vulkan` the default anywhere.** `auto` prefers `cuda`; `vulkan` joins the fallback chain only after V8.

### 7.4 Risks, and how each is retired

| # | risk | how it is retired | fallback |
|---|---|---|---|
| 1 | **fp64 is missing or emulated on the device**, so the arm is refused or is slower than the CPU | 4.3's feature check refuses; 4.4's list keeps `auto` off emulating parts; section 3.1 says which parts those are | the CPU backend, which is exactly what those machines run today |
| 2 | **the Vulkan compiler contracts or reassociates** where `-fmad=false` did not, so the harness's identical-bits rows become tolerances | `precise` on every floating-point result (2.4), checked by disassembling one shader (`spirv-dis`) for `NoContraction` in V2; the harness then measures | the bars of the CPU port, which already admit a last-bit difference everywhere (its 0a); the arm is still a correct port |
| 3 | **division and sqrt are not exactly rounded** on some driver (3.4) | measured per device in V2 and V5 through the harness's per-quantity worst difference; expected zero effect on decisions | a FLIP count in the harness; the PSNR arm is what judges the asset |
| 4 | **denormals flushed** on a driver that defaults to flush-to-zero for fp32 (3.4) | request `DenormPreserve` through the float controls where the device offers it; where it does not, the harness measures whether anything in this encoder's arithmetic reaches a denormal (the CPU plan found `E` at 1e-19 on one synthetic image, so it can) | record the device as "flushes fp32 denormals" in the banner and accept the tolerance |
| 5 | **register pressure in the assemblies** makes Vulkan much slower than CUDA on the same card | measured in V5 against the CUDA event times; `spirv-opt` and the driver's own tuning tried first | the assembly kernels are 23 % of the encode (plan 1.2); a 2x loss there is a 1.25x loss overall, which still leaves the AMD machines far ahead of the CPU |
| 6 | **shared memory below 48 KB** on a device (3.6) | the half-tile block (a) form, bit-identical; the refinement refused at `--c0 4` with a message | none needed beyond the message |
| 7 | **a buffer over `maxMemoryAllocationSize`** (4.8) | refused with the buffer's size, the limit and the remedy | the row-band split, if a real material on a real AMD card needs it |
| 8 | **a grid over 65,535 workgroups** on a driver reporting the minimum | the sliced dispatch of 5.1, exercised on lavapipe in every gate | none needed |
| 9 | **the harness's mirroring through Vulkan transfers is itself wrong** | the transfers are the first thing checked in V1, round-tripping every buffer exactly, before a kernel exists (the CPU port's rule) | the cross-backend arm, which shares no code with the harness |
| 10 | **a driver bug** on one vendor (wrong code for a `double` shader, a barrier miscompiled) | the harness locates it to a kernel and an input the day it appears; the viewer's history has an allowed Intel LOD difference and no bug (`docs/VULKAN_VIEWER_PLAN.md:1052`) | report upstream; refuse that driver version by name in 4.3's check, as the viewer refuses old Intel drivers |
| 11 | **the gate under `--backend vulkan` is too slow** or the AMD machines are not at the desk | measured in V8; the CPU plan's rule: a named subset with the rest recorded as skipped | -- |
| 12 | **three copies of the host math drift** (4.1) | the hash arm holds the originals; the harness holds every copy to the bit | 7.5's decision to lift them into one file, taken deliberately if it bites |
| 13 | **the two Vulkan programs in the tree diverge in their plumbing** (4.5) | they share the messages, the conventions and the CMake block, and nothing else, on purpose | extract when a fix has had to be made twice |
| 14 | **WSL has no native NVIDIA Vulkan** (3.5), so the Linux build is gated on software only | lavapipe covers correctness and the minimum-limits paths; the Kubuntu box covers a real Linux driver | -- |

### 7.5 Decisions for the owner

1. **The fp64 policy.** Refuse a device without `shaderFloat64` (this plan's default), or fund an fp64-free variant
   (3.2). The refusal keeps the port faithful and costs nothing on the machines that matter; the variant is a second
   set of shaders, a second arithmetic, and a GPU that is not in the fleet's interest except the Snapdragon.
2. **The rate gate under `auto`** (4.4): a vendor list, or a start-up measurement, or no gate (let a slow device
   run and say so in the report).
3. **The Vulkan floor**: 1.3 (recommended, the viewer's) or 1.2.
4. **The shader toolchain**: GLSL compiled by `glslc` at build time and embedded (recommended), GLSL compiled by
   shaderc at run time from beside the executable (the viewer's way), or Slang (2.6).
5. **Sharing with the viewer**: copy now, extract later (recommended), or extract now.
6. **The third copy of the host math**, or one shared file the three arms include -- the one place this plan would
   touch the spirit of decision 1 without touching a hashed file (the copies live outside them).
7. **Whether the refinement's report figure may change**: the per-slot fold of 5.2 makes `delta_e` deterministic and
   drops the fp64 atomic; its printed digits will differ from CUDA's beyond the 15th place, as the CPU arm's already
   do. Nothing reads it.
8. **Large buffers**: refuse over `maxMemoryAllocationSize`, or split.
9. **Whether to start at all**, against the alternatives of 8.3.

---

## 8. The bottom line

### 8.1 Difficulty, and expected speed

**Difficulty: moderate, and lower than the CPU port's per kernel; higher than the CPU port's overall.** The kernels
are already written in the shape Vulkan compute wants -- fixed workgroups, shared-memory trees, block barriers, no
warp intrinsics, no textures, no dynamic parallelism (1.3) -- so 28 of 37 transcribe line for line (5.1) and the
remaining nine change only their storage or their launch parameters, never their arithmetic (5.2). What is harder
than the CPU port is everything around the kernels: about 2,000 lines of Vulkan plumbing before the first dispatch,
a shader toolchain in the build, the `precise` discipline, the harness generalised to two reference arms, and a
verification that has to visit four machines and a software device instead of one desktop. **8 to 11 weeks** for one
implementer who knows the tree (7.2), and the CPU port's own estimate went up, not down, when its plan met its code.

**Speed, estimated, not measured** -- every figure below is reasoned from the kernels' arithmetic, the CPU plan's
measurement that the encode is fp64-bound at about 330 GFLOP/s on the 5090 (`docs/CPU_BACKEND_PLAN.md` 1.2), the
peaks of 3.3 and the CPU backend's measured 8 to 14x (`docs/CPU_PORT_LESSONS.md` section 8). To be replaced by V4,
V5 and V8's logs:

| machine | backend | estimate | why |
|---|---|---|---|
| RTX 5090 | CUDA | 1.0 (the reference) | measured, `docs/RESULTS.md` |
| RTX 5090 | Vulkan | **0.7 to 1.0x of CUDA's speed** | the same fp64 units and the same memory; the losses, if any, are the driver compiler's register allocation on the assemblies (5.1), `precise` forbidding optimisations `-fmad=false` also forbade (measured at 4 to 23 % on CUDA, plan 9a), and a fence per probe interval in place of an event wait |
| RTX 5090 | CPU, 32 threads | 0.07 to 0.12x | measured 8 to 14x slower |
| AMD RDNA3 / RDNA4 desktop (Windows or RADV) | Vulkan | **0.3 to 0.7x of the 5090 on CUDA**, that is **3 to 8x faster than the CPU backend on a 32-thread machine**, more on a laptop CPU | fp64 peak half to two thirds of the 5090's (3.3), the same fp64-bound kernels; the AMD Windows driver's 32 KB shared memory forces block (a)'s half-tile form, which doubles that kernel's barriers and touches nothing else |
| AMD integrated (the Ryzen's Radeon, `viewer_vk/README.md:464-477`) | Vulkan | not estimated | fp64 exists; the rate on an iGPU sharing system memory with the host is unknown to this study |
| Intel Arc B-series, Lunar Lake | Vulkan | not estimated | fp64 reported; the ratio could not be derived (3.3); no fleet machine |
| Intel Arc A-series, Gen11/Gen12 iGPU, Adreno X1-85, Apple | Vulkan | **refused**, `auto` runs the CPU | no `shaderFloat64` (3.1) |
| lavapipe | Vulkan | slower than the CPU backend, for testing only | a software Vulkan device running the same fp64 arithmetic through a JIT, with the barrier and dispatch overheads on top |

### 8.2 Is it worth it

**For the AMD machines, yes, if encoding on them matters.** Today those boxes encode at 8 to 14 times the 5090's time
(`README.md:497-498`); a Vulkan arm would bring them within a small factor of it, on the estimate above. It also
makes the second GPU implementation of the method, which is the argument `docs/CPU_BACKEND_PLAN.md`'s opening gave
for the CPU arm ("a second implementation is the cheapest evidence the method is a method"): a third, on a different
vendor's floating-point units, is stronger evidence still, and the harness would measure the agreement to the bit
where it holds.

**For the Snapdragon laptop, no**: no fp64, so nothing changes there without the emulation of 3.2, which is a
different project. **For Apple, no**, for the same reason. **For Intel**, only the newest parts.

**For the RTX 5090, no gain**: CUDA is the reference and at least as fast; the Vulkan arm on NVIDIA exists for the
harness (CUDA against Vulkan on one card is the port's cleanest comparison) and as a cross-check, not for encoding.

### 8.3 Against the alternatives

* **HIP for AMD** (the owner's earlier mention, `docs/CPU_BACKEND_PLAN.md` 1.8a, `docs/CPU_PORT_LESSONS.md` section
  12). HIP's kernel language is CUDA's with the runtime calls renamed, so the six `.cu` files would port nearly
  mechanically -- but they are frozen, so a HIP arm is a **parallel set of kernel files** anyway (1.8a says exactly
  this), about the same 2,000 lines of device code as the GLSL, plus a host arm. What HIP buys over Vulkan: warp-level
  and shared-memory semantics identical to CUDA's (irrelevant here, 1.3), `__dadd_rn`-class intrinsics and IEEE
  division by default (relevant: 2.4's fine print disappears), fp64 atomics, dynamic shared memory, and a compiler
  whose `-ffp-contract=off` is a flag rather than a qualifier on every line. What it costs: **AMD only**, a ROCm
  toolchain the fleet does not have installed, ROCm's Windows support and its list of supported consumer GPUs, which
  has been narrower than the Vulkan driver's (not verified by this study for the current release), and a third
  toolchain in the build. Vulkan's arm runs on NVIDIA too, which is what makes CUDA-against-Vulkan on one card
  possible, and it runs wherever a Vulkan loader is, which is every machine the viewers already run on.
* **The CPU backend as it is.** Zero cost. The owner accepted its speed for the purpose it serves (`docs/CPU_PORT_LESSONS.md`
  section 8: "the CPU backend exists so that the encoder works without an NVIDIA GPU, not to compete with one"). If
  the AMD machines only ever need to run the gate and encode the occasional asset, this is the answer.
* **Faster CPU backend** (the scatter assembly's measured 4.4x on 23 % of the encode, threading the host's 18 %,
  `docs/CPU_BACKEND_PLAN.md` section 9). Perhaps 1.5 to 2x overall, on every machine including the Snapdragon and
  Apple, for a fraction of this plan's effort -- **the alternative most worth weighing against it**, because it helps
  the machines Vulkan cannot.
* **OpenCL / SYCL / Metal**: not considered beyond noting that Metal has no fp64 and OpenCL fp64 support is the same
  hardware question with an older toolchain.

**Recommendation.** If the goal is the AMD boxes encoding at GPU speed, the Vulkan arm is the right shape and this
plan is how to build it; start with V0 through V2 (about three weeks), whose first measurement -- the 5090 under
Vulkan against CUDA, and the harness's exact rows -- decides whether to continue. If the goal is a faster encoder
everywhere, the CPU backend's two parked optimisations come first. Either way the nine hashed files do not change.

---

## 9. References

**In this tree**

* `docs/CPU_BACKEND_PLAN.md` -- the design this one follows: 0a (PSNR not bits), 1.3 (the seam), 1.7 (the dispatcher),
  1.8 (the copies), 1.8a (a third arm), 3.2 and 3.4 (reductions and the race audit), 4 (the harness), 6 (the stages and
  their measurements), 7 (the gate arms and bars), 8 (risks), 9 (out of scope)
* `docs/CPU_PORT_LESSONS.md` -- sections 2 (distributions), 3 (`NNTC_CUDA_FMAD`), 4 (the covariance replay), 5
  (determinism rules), 6 (testing layers and results), 8 (speed), 11 (working rules), 12 (open items, HIP)
* `src/backend.h`, `src/backend.cpp`, `src/cuda_backend.h`, `src/cpu/cpu_backend.h`, `src/backend_check.h`,
  `src/backend_check_probe.cu` -- the dispatcher, the two arms' interfaces, the harness and its probe
* `src/device.cuh` (`REDUCE_BLOCKS`, `SITE_BLOCKS`, `PlaneWork`, `DevPlane`, `DeviceModel`, the fan-out protocol),
  `src/sample.cuh` (the sampling rule), `src/model.h` (`level1_value`, `level1_index`, the seam's declarations) --
  hashed, unedited
* `src/init.cu`, `src/solve_decoder.cu`, `src/objective.cu`, `src/solve_level0.cu`, `src/solve_level1.cu`,
  `src/refine_bc.cu` -- the 37 kernels of section 1.2, hashed, unedited
* `viewer_vk/main.cpp`, `viewer_vk/vk_check.h`, `viewer_vk/README.md`, `docs/VULKAN_VIEWER_PLAN.md`,
  `CMakeLists.txt:280-437` -- the tree's Vulkan plumbing and its detection
* `tests/run_checks.py` -- the gate's Vulkan arms (583-683, 3649-3710, 4386-4450), the backend flag (101-107,
  4637-4643), the forbidden-pattern scan (37-49, 63-64)
* `docs/DESIGN.md` section 6 -- determinism, and the one order-dependent double atomic

**Outside the tree**, each cited where it is used in sections 2 and 3: the Vulkan specification
(https://docs.vulkan.org/spec/latest/appendices/spirvenv.html, https://docs.vulkan.org/spec/latest/chapters/limits.html,
https://docs.vulkan.org/spec/latest/appendices/versions.html, https://docs.vulkan.org/spec/latest/chapters/features.html),
the GLSL specification for Vulkan (https://docs.vulkan.org/glsl/latest/chapters/variables.html,
https://docs.vulkan.org/glsl/latest/chapters/builtinfunctions.html), GL_KHR_vulkan_glsl, GL_EXT_buffer_reference,
GL_EXT_shader_atomic_float, GL_EXT_shader_atomic_int64 (the KhronosGroup/GLSL repository), the vulkan.gpuinfo.org
feature, limit and device-report pages named in 3.1 and 3.6, Intel's fp64 support article, the CUDA floating-point
guide (https://docs.nvidia.com/cuda/floating-point/index.html), the CUDA on WSL user guide, CMake's FindVulkan, the
Khronos Slang announcement, and the RTXNTC and VkFFT repositories.

**Not verified by this study, stated as such where it appears**: whether Vulkan drivers round fp32 and fp64 division
exactly (2.4); NVIDIA's default fp32 denormal handling under Vulkan (2.4); whether dozen exposes working fp64 inside
WSL (3.5); whether Meteor Lake and Lunar Lake's `shaderFloat64` under Mesa is hardware or emulation (3.1); the fp64
ratio of Intel Battlemage and RDNA2 (3.3); lavapipe's `maxComputeWorkGroupCount` (3.6); ROCm's current Windows and
consumer-GPU support (8.3); the exact sentence of the Metal Shading Language specification on `double` (3.1, taken
from the MoltenVK report and issue instead).
## 10. Review findings (2026-09-19), not yet folded into the sections above

An independent review found the plan's shape right (a third arm behind the seam, faithful transcription, the pinned
CUDA files untouched, fp64 required, the V0-V2 go/no-go) and ready to execute only with these changes. They are
recorded here as found; the sections above have not been revised yet.

1. **The BC refinement at `--c0 4` does not fit AMD Windows.** `k_bc_refine` needs `(n^2 + 3n + 80) * 8` = 34,944 bytes
   of shared memory at `--c0 4` (`refine_bc.cu:545`); the RX 9070 XT on Windows reports 32,768. 5.2 says the refusal
   bites only on Adreno and MoltenVK, contradicting 3.6. Make option 2 (`a2` in a global scratch slice) the planned
   path, or make the refusal an owner decision and shrink the corpus to match; budget it in V7.
2. **The 2 GB allocation limit** on AMD Windows refuses the 4096x4096 materials even at `--c0 2` (a 2.7 GB stencil,
   `init.cu:204`). 4.8 also omits `maxBufferSize`, which device addresses do not lift (2 GiB AMD Windows, about 4 GiB
   RADV; above 4 GB only NVIDIA). Plan the row-band split or drop those cases on AMD; add `maxBufferSize` to 4.3.
3. **An AMD Windows Vulkan device is already on the owner's main machine**: the integrated Radeon, Vulkan `--device 1`
   (`viewer_vk/README.md`). Add it to 6.4 and to every stage's done-when; it also gives CUDA against Vulkan-AMD on one box.
4. **Several "identical bits" criteria cannot be promised.** RADV lowers fp32 `a/b` to `a * rcp(b)`; the CG step's
   `alpha`/`beta` are fp64 divisions and its residual an fp64 `sqrt` (`solve_level1.cu:860, 866, 869`), and Vulkan
   promises fp64 only "at least single precision". Make done-when bars plus flip counts, per device; keep "identical"
   as a measured outcome. Replace the device divisions whose bits the host must reproduce (`k / 255.0` in
   `level1_value` / `bc0_value`) with an uploaded table of the values.
5. **GLSL cannot declare DenormPreserve or RoundingModeRTE**: it needs `GL_EXT_spirv_intrinsics`
   (`spirv_execution_mode`) or SPIR-V post-processing. ACO flushes fp32 denormals unless asked; NVIDIA reports neither
   DenormPreserve32 nor FlushToZero32. Name the mechanism in 2.6 and test it in V2.
6. **Check `precise` on every module, not one**: a build or gate step that disassembles all 37 SPIR-V modules and
   fails on any float `OpFAdd`/`OpFSub`/`OpFMul` without `NoContraction` (glslang issues 2709 and 1425).
7. **Device loss and the Windows 2-second TDR are missing.** Add a risk row; report `VK_ERROR_DEVICE_LOST` as one
   ERROR suggesting `--backend cpu`; measure the longest dispatch per stage; split dispatches (launch geometry only).
8. **`auto` guards the wrong devices.** ANV already reports no fp64 on the Intel parts named; the slow device that
   passes every check is lavapipe. Under `auto`, always refuse `VK_PHYSICAL_DEVICE_TYPE_CPU`; make integrated GPUs an
   owner decision; drop the vendor-list rate gate.
9. **Kernel counts do not reconcile**: 37 kernels is right (10 + 2 + 3 + 1 + 19 + 2 `__global__`), but the
   "28 of 37" list names 32, `k_init_level1_box` and `k_block_means` appear in no list, 5.2 restructures three
   kernels not nine, and 5.3's "three one-invocation kernels" lists seven. The summary also omits the two fp64 atomics.
10. **fp64 throughput is unresolved.** Sources disagree for RDNA3 (1:64 against 1:32 of the dual-issue rate); consumer
    AMD may roughly match the 5090's fp64 rather than reach half to two thirds of it. The CPU "8-14x the 5090" figure
    was measured on the 5090 box's own 32-thread CPU. Mark both unresolved; add a one-kernel fp64 FMA benchmark to V1.
11. **The V0 function-table refactor** stretches the precedent: `backend.h` prescribes a new enumerator, pointer, file
    and branch per arm. Add the branches as prescribed, or make the table an explicit owner decision.
12. **The half-tile block (a) reasoning is correct** (same summation order; the fp64 products of two floats are
    exact), but the write-up must state that each thread keeps `sum[LS_SLOTS]` across the mid-chunk barrier and that
    threads 128-255 hold their `phi`/`target` meanwhile; it is three barriers per chunk, and 22,188 bytes at the caps.
13. **Smaller gaps:** a single-queue pipeline barrier is global, so per-plane overlap needs interleaved recording;
    36-bit timestamps on some Intel queues (mask by `timestampValidBits`); dozen in WSL 2 (Vulkan 1.2, not conformant,
    128 MB storage range, often unpackaged) and NVIDIA's WSL driver ships no Vulkan ICD, so the 5090 is not reachable
    through Vulkan inside WSL 2; "parameter-UBO" in 4.1 against device addresses in 2.3; the shaderc fallback in 4.6
    contradicts "no shader file beside the executable" (embed the sources); state the audit result that no kernel
    communicates through global memory within a workgroup across a barrier.
