# CPU_BACKEND_PLAN.md -- an optional scalar CPU backend for the encoder

A design for a second implementation of the encoder's device interface: plain portable C++17 and `std::thread`, no
intrinsics, selected with `--backend cpu`. It is a plan, not code.

**The hard constraint, first, and it is harder than the first draft's.** *The CUDA kernels are not modified.* Every
file in `src/` that carries a kernel -- `init.cu`, `solve_decoder.cu`, `solve_level1.cu`, `solve_level0.cu`,
`refine_bc.cu`, `objective.cu` -- and the two headers they own -- `device.cuh`, `sample.cuh` -- is **byte-identical
before and after the whole project**, and that is verified by hash rather than by judgement (section 2.5). No namespace
wrap, no move of a definition out of them, no extraction of the host math they contain, no reformatting. If the port
appears to need an edit to one of those eight files, that is a finding to bring to the owner, not a change to make.

**Why a second backend at all.** Three reasons, in order of weight:

1. **the encoder is the one program in this tree that needs an NVIDIA card.** Both viewers run on AMD, Intel,
   Qualcomm and Apple silicon; `tools/dds_decode.py` decodes with no GPU at all; `bc_check` is plain C++. The encoder
   is why `README.md` has to say "the release gate needs the encoder and does not run on such a machine". A CPU
   backend deletes that sentence, and with it the rule that the Arm laptop and the Kubuntu box can only ever be
   viewing machines. **Section 7.1 argues that this, and not the encoding, is the larger deliverable;**
2. **a second implementation is the cheapest evidence the method is a method** and not an artefact of one vendor's
   arithmetic. The tree already has three independent decoders (two viewers and a Python one) and exactly one encoder;
3. **it is a reference the CUDA path can be held against.** `objective_check_host` already exists for this reason and
   is trusted to 5e-15 (`docs/RESULTS.md` 1.3). A whole backend is the same idea at the scale of the whole encode.

**What it is not.** It is not a fast encoder, and it is not the default. The measurement below says a threaded scalar
CPU encode is two to three times slower than an RTX 5090 on the *base* plane, which makes it useful and not
competitive -- and section 3.2 records a cost the deep planes carry that that measurement did not.

---

## 0. What changed since the first draft, and why

A reader of the earlier version should start here. The owner's decisions after reading it changed the architecture,
not the goal, and a review of the draft found nine things wrong with it that survive the change.

**What the owner decided, and what it deleted.**

* **The kernels are frozen.** The first draft's step A wrapped every `.cu` file in a namespace (its class N), moved
  about 350 lines of host math out of them into a new `src/host_math.cpp` (class M), split `sample.cuh` into
  `sample.h` (class I), and split `device.cuh`'s CUDA-free constants into `device_constants.h`. **None of that
  happens.** The eight kernel-side files are untouched and hashed.
* **So the symbol collision is solved on the host side instead.** The dispatcher of section 1.7 is a new pair of
  files and a `be::` prefix on 97 call sites. Nothing inside a `.cu` file learns that a second backend exists.
* **So the class-F extraction is gone, and with it two of the review's findings.** The draft's one non-mechanical
  change was lifting block (a)'s ridge ladder out of `solve_decoder.cu` into shared code, and the review's objection
  was that it was guarded only by a throwaway comparison script and a tally arm. Under the new rule the CPU backend
  **copies** the ladder instead. The extraction and its risk disappear; the duplication that replaces it is counted
  and justified in section 1.8, and the instrument that guards it is a real one (section 4) rather than a throwaway.
* **Zero data races, stated as an absolute.** Not "benign races documented". Section 3.4 is a per-kernel audit, a
  rule for what a fix may be, and a permanent ThreadSanitizer arm.
* **`--backend cuda` asked for by name is an error when CUDA is unavailable**, where the draft had only a refusal at
  parse time in a build that lacked the backend. Section 2.3 gives the four cases and their messages.

**What the review found that still applies, and where each is answered.**

| # | finding | where |
|---|---|---|
| 1 | the stage structure was not testable: stages 2 to 5 claimed tests that need a full encode | **section 4**, a per-kernel comparison harness, written early and the backbone of every stage after it |
| 2 | the round loop can stop at different rounds on the two backends, and `--tol 0` does not prevent it | 1.9, 7.4 |
| 3 | `k_bc_refine`'s 64-slot reduction tree is not a serial left fold and must be replayed, not deleted | 5.6, and risk 7 |
| 4 | MSVC on Arm64 and fp contraction is an untested assumption the whole Arm story rests on | 1.5, 2.4, risk 4 |
| 5 | the byte-identity corpus that guards step A was too narrow | 2.4 |
| 6 | six report fields have no CPU counterpart, and one printed paragraph is false on a serial backend | 1.10 |
| 7 | host memory and out-of-memory behaviour was not stated | 1.11, risk 12 |
| 8 | `-j1` is the adversarial thread count, not `-j7` | 3.1, 7.3 |
| 9 | the fixed-grid reduction replay is disproportionate on deep planes and the 2-3x estimate does not carry it | 1.2, 3.2, risk 11 |
| 10 | `--check`'s brute-force twins get duplicated too | 1.8 |
| 11 | running the gate twice overwrites the first run's assets -- and the far more important consequence | 7.1 |
| 12 | twelve factual slips about this tree | fixed in place throughout |
| 13 | `stb_image_resize2.h` has per-ISA paths, so an x86 and an Arm CPU encode may differ in the SOURCE chain | 1.12 |

**What is unchanged.** The seam and its line references, the fixed-grid replay rule and its index formula, keeping
`CG_PROBE` at 4 with the 200-iteration cap and the 1e-10 bar, the four-colour independence analysis, the intra-texel
Gauss-Seidel over channels, the per-kernel rounding traps, the three distinct bilinear rules, fp32 FMA contraction as
the irreducible difference, the out-of-scope list with its measured numbers, and the degenerate-image handling in the
cross-backend arm. Section numbering is the draft's with one shift: the harness is the new section 4, so the old
sections 4 to 9 are now 5 to 10 under the same titles.

**And the estimate went up, not down.** Section 6 gives the arithmetic. The draft said 6 to 8 weeks with the
extraction and the namespace wrap in it; without them but with the harness, the race audit, the ladder copy and the BC
tree replay, it is **7 to 9**.

---

## 0a. The owner's amendment after step B: gate on PSNR, not on bits

**This section overrides the sections it names.** The owner, after step B: "I would just gate the CUDA vs. CPU steps
based off PSNR. Trying to make floating point code match between widely different platforms is ultimately a fool's
errand. Even if you think you've got it, there will be edge cases." That settles the question the earlier drafts
left open, and it removes work rather than adding it.

**Two different things were sharing the word "match", and only one of them survives.**

1. **CUDA versus CPU: PSNR, not bits.** No part of the CPU backend is required to reproduce CUDA's floating-point
   results bit for bit. The cross-backend gate is the decibel tolerance of section 7.4 and nothing stricter. This
   supersedes:
   * **section 3.2**'s rule that the CPU side replays CUDA's fixed-grid two-stage reduction trees (`b*256 + t`,
     then `+ B*256`). The CPU backend uses its own reduction order, chosen for simplicity and speed, subject only to
     point 2 below. This also retires **risk 17 / section 1.2's caveat** about that replay costing ~65k additions to
     reduce ~4k values on a deep mip plane: the replay is gone, so the cost is gone.
   * **section 5.6**'s requirement to replay `k_bc_refine`'s 64-slot `refine_reduce` tree. A plain serial fold over
     the block's sixteen terms is the CPU form. The discrete decisions it feeds (the determinant guard, the
     nine-candidate argmin, the strict-decrease acceptance) may then land differently from CUDA's in a
     near-tie, and that is accepted: the PSNR gate is what judges the result.
   * **section 9a**'s `-fmad=false` proposal, whose main purpose was bit-exactness between backends. Not needed
     under PSNR gating, so the CUDA build stays exactly as tested. The owner has said it is acceptable to turn it
     off by default if it brings the results closer; it stays AVAILABLE for the one case where it would help -
     fp32 rounding noise making the harness's tolerances awkward to set - with a build already at
     `out_fmad/build` to measure it.
   * **section 4.2**'s "exact bits" bar for the harness (see point 3).

2. **Within the CPU backend: determinism is still REQUIRED, and it is not the same thing.** The same build on the
   same machine, run twice, or at `-j1` and at `-j32`, produces byte-identical files. That is not cross-platform
   floating-point matching; it is not letting the thread scheduler choose the answer. It needs only a fixed chunk
   count per reduction, independent of the thread count, folded in chunk-index order after the join - no CUDA tree.
   The gate's fifteen byte-for-byte arms and section 7.3's thread-count arm depend on it, and section 3.1's pool
   contract (static partition, no work stealing) is unchanged.

3. **The per-kernel harness stays, with tolerances in place of exact bits.** Its value was never proving bit
   equality; it is locating a porting bug. End-to-end PSNR 0.3 dB low says something is wrong but not where. The
   harness feeds each kernel IDENTICAL inputs on both sides, so there is no accumulated drift, and a transcription
   mistake - a wrong index, a loop bound, a tie-break, a swapped colour - shows up as a disagreement many orders of
   magnitude outside rounding. Each compared output gets a tolerance appropriate to its type (for example a relative
   bar near 1e-5 on fp32 values, tighter on fp64 accumulations, exact on integer indices and counts where the
   inputs are identical and no near-tie is involved), set by measuring a known-good kernel pair rather than guessed.
   Where a discrete decision can legitimately flip on a near-tie, the harness reports it as a flip rather than a
   failure and counts it.

**What does not change:** decision 1 (the kernel files are not modified), the dispatcher, zero data races, the
fallback rules, the order of work, and the cross-backend corpus of section 7.4 with its handling of the degenerate
one-white-pixel image, which gets no PSNR bar for exactly the reason the owner gives: its path depends on a
floating-point comparison.

## 1. Recommended architecture

### 1.1 The decisions this plan does not re-open

Ten of them, taken by the owner. Every section below is downstream of one.

1. **The CUDA kernels are not modified.** The eight files named at the top of this document are byte-identical before
   and after, verified by hash (2.5). This is the paramount constraint and it outranks every other line here.
2. **A dispatcher routes to either C++ or CUDA; the encoder is mostly none the wiser, and the CUDA side, already well
   tested, stays as stable as practical.** Section 1.7 is the mechanism.
3. **Minimal surgery outside new files.** The edits are: the device-interface call sites in `main.cpp` (96 of them)
   and the one in `export.cpp`; the `--backend` and `-j` flags; the auto-fallback logic; the report and banner lines
   that name the device; and `CMakeLists.txt`. No logic moves, nothing is rewritten, no "while we are here" changes.
4. **All CPU code goes in new `.cpp` / `.h` files.** Anything the CPU backend needs that lives in a kernel file today
   is **copied**, not extracted. Section 1.8 counts the duplication and justifies it.
5. **Scalar portable C++17 plus `std::thread`.** No intrinsics, no SIMD, no per-ISA `#ifdef`. Measured: SSE 4.1 was
   worth about 1.00x on the dominant kernel while threading was worth 14 to 16x, and the fleet includes a
   Windows-on-Arm Snapdragon where SSE does not exist at all.
6. **A faithful port.** Same math, same accept and reject rules, same defaults, same iteration and probe cadences --
   including `CG_PROBE = 4`, which has no reason to exist on a CPU and is kept anyway (5.4).
7. **Zero data races.** None, not "none that matter". Section 3.4.
8. **Backend selection and fallback.** The CPU backend is always built and always available; `--backend auto` is the
   default and falls back with a WARNING; `--backend cuda` on a machine that cannot run it is an ERROR. Section 2.3.
9. **Cross-backend acceptance is PSNR-close, not bit-exact**, on the tier `docs/DESIGN.md` section 6 already defines
   and measured. **Within** the CPU backend, byte-identical repeat encodes and byte-identical results at any thread
   count are required. Sections 7.3 and 7.4.
10. **Order of work.** (A) the dispatcher and the flags, no CPU encoder at all, ending with the CUDA encoder
    byte-identical to `v1.1.28-woa-tested`; (B) the CUDA regression, signed off as its own event; (C) the CPU
    backend; (D) compare and hold the tolerances. Section 6.

### 1.2 What the encoder spends its time on

From the prior feasibility study (its microbenchmarks and arithmetic are in the gitignored `out_cpu_study/`; the two
encode logs quoted are `mb_log.txt` and `me_log.txt`). These are facts to build on, not to re-derive.

| fact | number |
|---|---|
| the encode is fp64-accumulation-bound | fp64 is 1/64 rate on the 5090's part |
| an RTX 5090 across an encode | about 330 GFLOP/s fp64 |
| a 16-core Ryzen 9950X | 200 to 280 GFLOP/s fp64 |
| so a threaded scalar CPU encode is | about 2 to 3 times slower than the 5090 |
| a 1024x1024 four-texture material at the default layout | about 7 to 11 s against the 5090's measured 3.202 s |
| a 2048x2048 five-texture material | about 30 to 45 s against the 5090's measured 14.537 s |
| SSE 4.1 against scalar, on the dominant kernel | about 1.00x |
| threading, same kernel | 14 to 16x |

**Read the 2 to 3x with the caveat the review attached to it.** Those microbenchmarks are base-plane loops: large
arrays, every chunk full, the fixed-grid replay amortised over thousands of elements per slot. A deep mip plane is a
few thousand texels, and section 3.2's rule still spends 256 chunks and a 256-slot tree on it -- roughly 65,000
additions to reduce 4,000 numbers, plus half a megabyte of scratch touched, several `Pool::run` calls per conjugate
gradient iteration, up to 200 iterations, times six or seven chain planes. **The 2 to 3x does not carry that**, and
nothing measured yet says what it costs. Measuring it is an early task (stage C1), not an end-of-project surprise, and
section 3.2 gives the one permitted remedy.

Where the time goes, which is what decides what has to be threaded and what does not:

| phase | share |
|---|---|
| block (c'), the level-0 stencil assembly | 23 % |
| block (b), level 1 | 14 % |
| the objective | 14 % |
| the BC refinement and the outer repack | 15 % |
| single-threaded host I/O and the mip build | 18 % |
| everything else | 16 % |

Two readings of that table matter. First, **the four threaded phases are 66 % of the encode**, so Amdahl's law caps a
16-thread port at about 2.6x over a single-threaded one unless the host's 18 % is threaded too -- and threading it is
out of scope (section 9). Second, **no single kernel dominates**: the port cannot be staged as "do the hot one and
stop". All six kernel files have to be ported before the backend runs a whole encode, which is exactly why section 4
exists: the harness lets each one be *verified* the moment it is written, months before the backend runs.

### 1.3 The seam, spelled out

`src/model.h` declares an opaque `struct DeviceModel` and 46 free functions against it. `src/main.cpp` (2983 lines),
`src/mips.cpp` (169) and `src/export.cpp` (811) contain **zero** occurrences of the string `cuda` in code -- verified,
not assumed; the five uppercase `CUDA` strings in `main.cpp` are a file comment, the `--device` usage line, the
`no CUDA device` refusal and two report lines, and `main.cpp` calls no CUDA API at all. So that header is already the
line the port is drawn along. The whole of the work is: write a second set of bodies for these 46 functions.

There are **97 call sites**: 96 in `main.cpp` on 95 lines, and exactly one in `export.cpp` --
`reconstruct_levels` (`src/export.cpp:437-442`), which is itself declared in `model.h` but defined on the host and
calls `decode_plane` in a loop. `mips.cpp` has none.

Grouped by what they do, with the CPU implementation's difficulty. *Trivial* means a memcpy, an accessor or ten lines
of arithmetic; *mechanical* means a loop nest transcribed from a kernel with no judgement required; *needs thought*
means a CUDA idiom with no direct CPU equivalent, or a decision the plan has to take.

| group | functions | count | difficulty |
|---|---|---|---|
| lifecycle | `device_select`, `device_create`, `device_destroy`, `device_memory` | 4 | trivial, except that `device_create` also builds the thread pool and must report the same byte total under the same name |
| level-0 and level-1 initialisation | `init_level0`, `init_level1`, `init0_residual_channel`, `stencil_monomials` | 4 | **needs thought**: `init_level1`'s pca path and `init0_residual_channel` are the two places where a fixed-grid covariance reduction feeds a host eigensolve, and the residual seed has a 4096-bin histogram behind an integer atomic that accumulates over **every plane of the chain**, not per plane |
| the grids | `quantise_level1`, `freeze_level1_grid`, `quantise_level0`, `freeze_level0_grid`, `snap_level0_on_grid` | 5 | mechanical; all five are one kernel (`k_quantise_level1`) plus the already-host `fit_range` |
| block (a), the decoder | `solve_decoder`, `decoder_ridge_tally` | 2 | **needs thought** for the accumulation (one staged-tile kernel); the ridge ladder is a 160-line copy (1.8) |
| blocks (b) and (c') | `solve_level1_all`, `sweep_level1_all`, `solve_level0_cont_all`, `sweep_level0_cont_all`, `assemble_level0_for_refine`, `solve_level1_dense_host` | 6 | **needs thought**: the largest single piece. `solve_level1_dense_host` is already pure host C++ apart from two downloads and is nearly free |
| block (c), level 0 as a palette index | `solve_level0_all` | 1 | mechanical, but the four-colour ordering is load-bearing and it carries the plan's one race fix |
| the objective and the decode | `objective_eval`, `objective_check_host`, `decode_plane`, `objective_total_ms`, `objective_passes` | 5 | mechanical; `objective_check_host` is already a serial host brute force and ports by deleting two downloads |
| the BC refinement | `bc_pack_prepare`, `bc_refine`, `bc_blocks_from_device`, `bc_state_save`, `bc_state_restore` | 5 | **needs thought**: `k_bc_refine` is the hardest kernel in the tree |
| transfers and probes | `level1_download`, `level1_download_indices`, `level1_upload`, `level1_upload_indices`, `level1_delta`, `level1_poke`, `level1_range`, `level0_download`, `level0_download_values`, `level0_upload`, `level0_upload_indices`, `upload_decoder_to_device`, `upload_grids_to_device`, `reconstruct_levels` | 14 | **trivial**: on a CPU backend a download is a `std::copy` and an upload is the same copy the other way. `level1_range` is already host code; `reconstruct_levels` lives in `export.cpp` and is a consumer of the seam rather than part of it |

46 functions; 19 of them trivial, 12 mechanical, 15 needing thought. The two host-only groups of `src/model.h` --
`build_plane_sizes` / `build_source_chain` / `build_mip_weights` in `mips.cpp`, and `level0_plane_values` /
`level0_pack_seed` / `level0_pack_blocks` / `write_level_pngs` / `write_asset` / `verify_level1_on_grid` in
`export.cpp` -- are not in the seam at all and are not touched.

The fourteen transfer functions are also the harness's instrument: they are how identical inputs are put into both
backends and how both backends' outputs are read back out (section 4). They are written first for that reason.

### 1.4 What the CPU backend shares, and what it must copy

**Shared, with no change to anything.** `src/model.h` is plain C++17: its CUDA-ness is one macro,
`NNTC_HOST_DEVICE`, defined empty when `__CUDACC__` is not set. So `replicate_index`, `level1_value`, `level1_index`,
`level0_value`, `build_level0_palette`, `feature_count`, `mip_count`, `Model`, `PlaneSize`, every report struct and
every declaration in the seam are read by the CPU backend directly, from the one definition. That matters most for
`level1_value` and `level1_index`, which are the grid itself and which `verify_level1_on_grid` depends on.
`src/image.h` and `src/options.h` are the same.

**Copied, because it cannot be shared.** `src/device.cuh` and `src/sample.cuh` both `#include <cuda_runtime.h>` on
their first lines. A build with no CUDA toolkit has no such header, so a CPU translation unit cannot include either
file, and under decision 1 neither file may be edited to make it possible. Everything the CPU backend needs from them
is therefore copied into `src/cpu/`. Section 1.8 counts it.

**The same is true of the host math inside the kernel files.** `gauss_jordan` (`solve_decoder.cu:184`),
`jacobi_eigen` (`init.cu:744`), `fit_range` (`init.cu:475`), `build_form_matrices` (`solve_level1.cu:164`) and its two
callers, `stencil_monomials`, `stencil_monomial_values`, `footprint_range`, `normalise` (`objective.cu:224`),
`twin_sample`, `bc4_weight8` / `bc4_entry` / `bc0_value`, `refine_grid`, `plane_blocks`, `level1_range`: all of it is
plain C++ compiled by nvcc today only because of where it happens to sit, and all of it is copied rather than moved.

**The brute-force twins are still the reference, and they are copied too.** `objective_check_host`
(`objective.cu:317-380`, 64 lines) is a serial host evaluation of the objective; `solve_level1_dense_host`
(`solve_level1.cu:1502-1688`, 187 lines) is a dense direct solve of block (b) with its own independently re-derived
tap indexing. Both are gated into the release checks today. They are the CPU backend's first two milestones -- and
copying them does **not** weaken them, because what makes each one a witness is that its derivation is independent of
the kernels it checks, and that independence is a property of the code, not of how many copies of it exist. Each
backend's copy is checked against that backend's own kernels, exactly as today.

### 1.5 The one thing that cannot be inherited

`src/model.h:level1_value` and `refine_bc.cu:bc0_value` are deliberately written with separately rounded **double**
operations -- `__dadd_rn` / `__dmul_rn` under `__CUDA_ARCH__`, plain operators and `-ffp-contract=off` on the host --
so that one function agrees host-against-device bit for bit, which is what `verify_level1_on_grid` depends on. **The
CPU backend gets the host branch for free, and must be built with contraction off to keep it.**

**This is an untested assumption on Arm, and the whole Arm story rests on it.** `CMakeLists.txt:128-137` puts
`-ffp-contract=off` on the **gcc and clang branch only**; the MSVC branch (118-127) sets `/W4` and nothing about
floating point, and relies on MSVC's default `/fp:precise` not contracting. That reliance is well founded on x64,
where a double-precision FMA needs `/arch:AVX2` to be emitted at all. It has never been checked on **MSVC ARM64**,
where `fmadd` is a baseline instruction and the compiler has every reason to use it. If it contracts, `level1_value`
moves a last bit, `verify_level1_on_grid` refuses to write, and the encoder on the Snapdragon laptop -- the machine
this backend most exists for -- fails on a correct asset.

**So step A carries a two-line check** (2.4): compile a sweep of `level1_value` / `level1_index` and assert the round
trip, host only, on every toolchain, and diff the sweep's hex floats between the x86 and the Arm builds. The remedy if
it fires is one line: `/fp:contract-` on the MSVC branch, which is the exact counterpart of the gcc option already
there. It is two lines of work and it retires a risk with a high consequence; it is moved into step A for that reason.

The other half of the same fact is less comfortable and is stated here rather than buried: **the device contracts fp32
multiply-adds by default** (`CMakeLists.txt` says so: "fp32 with the default FMA contraction and deliberately no
`--use_fast_math`"), and the host does not. So every fp32 feature `site_row` builds will differ from the device's in
the last bit somewhere, and `E` will not be bit-identical between the backends however carefully the summation order
is reproduced. That is the irreducible difference, and it is exactly why decision 9 says decibels rather than bytes.
It is also the line section 4 draws between the quantities the harness holds to the bit and the quantities it holds to
a tolerance.

### 1.6 File list

New in step A (the dispatcher and the CUDA arm), about 550 lines:

```
src/backend.h              namespace be: the Device handle, the Backend enum, and the 46 seam declarations
src/backend.cpp            pure routing: the 46 forwarders, the file-static selected backend, backend_select(),
                           and the declaration of reconstruct_levels that export.cpp now uses. It names no
                           backend's internals and includes no CUDA header
src/cuda_backend.h         namespace nntc_cuda: the 46 declarations of the CUDA arm
src/cuda_backend.cpp       the CUDA arm: 46 one-line wrappers over the existing global functions the .cu files
                           define, and nothing else. Compiled only when nvcc was found
```

New in step C, about 5000 lines:

```
src/cpu/pool.h             the thread pool and its contract (3.1). Header only, about 120 lines
src/cpu/cpu_constants.h    device.cuh's CUDA-free half, copied: the caps, ChannelBits, the fixed-grid constants,
                           the stencil geometry, the scalar-row slots
src/cpu/cpu_sample.h       sample.cuh, copied, with <cuda_runtime.h> -> <cmath> and NNTC_HOST_DEVICE dropped
src/cpu/cpu_model.h        CpuModel: the plain-array twin of DeviceModel, PlaneWork and plane_work copied
src/cpu/cpu_model.cpp      lifecycle, memory accounting, the allocator, and all fourteen transfer functions
src/cpu/cpu_init.cpp       init.cu's ten kernels and their drivers, plus fit_range and jacobi_eigen copied
src/cpu/cpu_solve_decoder.cpp   block (a): the staged accumulation, the two-stage reduction, and the ridge ladder,
                                gauss_jordan and tri_index copied
src/cpu/cpu_solve_level1.cpp    blocks (b) and (c'): both assemblies, the preconditioned CG, the quantised sweeps,
                                the form and monomial matrices copied, and solve_level1_dense_host copied
src/cpu/cpu_solve_level0.cpp    block (c): the four-colour exact search
src/cpu/cpu_refine_bc.cpp       the analysis-by-synthesis refinement, and the BC palette helpers copied
src/cpu/cpu_objective.cpp       the objective, its two-stage reduction, decode_plane, normalise and twin_sample
                                copied, and objective_check_host copied
src/backend_check.cpp      the per-kernel comparison harness (section 4), about 700 lines. Compiled only when
                           both backends are in the build
```

and one tracked text file in step B:

```
tests/kernel_hashes.txt    nine lines, `sha256  path`, for the six .cu files, the two .cuh headers and model.h
```

`src/cpu/` needs no change to `tests/run_checks.py`'s tree scan: `SEARCHED_DIRS` already names `src` and the scan
walks it recursively, so the forbidden-pattern check polices the new code from its first commit.

### 1.7 The dispatch: one namespace, 46 forwarders, and a linker that proves it complete

Both backends have to be in one executable, because the gate's most valuable new arm compares them, because the
harness of section 4 needs both live at once, and because `--backend` is a run-time flag.

The collision is that `model.h` declares 46 free functions at global scope and the six `.cu` files **define** them
there. Under decision 1 those definitions cannot be renamed, wrapped or moved. So the CPU backend's functions take
different names and the host stops calling the global ones.

**The three pieces.**

```cpp
// src/backend.h -- plain C++17, no CUDA header, included by main.cpp and export.cpp
#include "model.h"                 // DeviceModel is already an opaque forward declaration there
namespace nntc_cpu { struct CpuModel; }

namespace be
{
enum class Backend { Cuda, Cpu, Check };

// The handle the host carries. TWO typed pointers, not a void*: each forwarder unwraps the one its branch
// needs and the compiler checks it, so a wrong-branch unwrap is a type error rather than a cast that compiles.
struct Device
{
    Backend backend = Backend::Cuda;
    DeviceModel*          cuda = nullptr;   // non-null under Cuda and under Check
    nntc_cpu::CpuModel*   cpu  = nullptr;   // non-null under Cpu  and under Check
};

// Set once, from the parsed command line, before device_select is called.
void backend_select(Backend b, int threads);
Backend backend_selected();

// The 46, with DeviceModel* replaced by Device* and every other parameter and return type unchanged.
bool   device_select(int index, DeviceInfo& info);
Device* device_create(const Model& m, const std::vector<Image>& source_chain);
void   device_destroy(Device* d);
size_t device_memory(const Device* d);
int    stencil_monomials(int c0);
void   decoder_ridge_tally(long long out[4]);
double objective_total_ms();
long long objective_passes();
// ... and the other 38, mechanically ...

// Moved here rather than left in model.h, so that model.h itself needs no edit at all.
void reconstruct_levels(Device* d, const Model& m, std::vector<std::vector<uint8_t>>& recon);
}
```

```cpp
// src/cuda_backend.cpp -- the CUDA arm, and the ONLY host file that names the .cu files' functions.
// Built only when nvcc was found. Every body is one line: model.h's global declarations ARE the CUDA
// declarations, so this file forwards to them and adds nothing. The kernels are not touched, not renamed
// and not wrapped; this file exists so that nothing else in the host code has to know they are there.
#include "cuda_backend.h"
#include "model.h"

namespace nntc_cuda
{
void init_level0(DeviceModel* d, Model& m, const float* lw, bool luma_c0) { ::init_level0(d, m, lw, luma_c0); }
// ... 45 more of exactly that shape ...
}
```

```cpp
// src/backend.cpp -- pure routing. It knows the two arms exist and nothing about either.
#include "backend.h"
#ifdef NNTC_CUDA
#include "cuda_backend.h"          // namespace nntc_cuda, the 46 wrappers
#endif
#include "cpu/cpu_backend.h"       // namespace nntc_cpu, the 46 twins

namespace be
{
static Backend g_backend = Backend::Cuda;   // the file static decision 2 asks for
static int     g_threads = 0;

void init_level0(Device* d, Model& m, const float* luma_weights, bool luma_channel0)
{
#ifdef NNTC_CUDA
    if (d->backend == Backend::Cuda) { nntc_cuda::init_level0(d->cuda, m, luma_weights, luma_channel0); return; }
#endif
    nntc_cpu::init_level0(d->cpu, m, luma_weights, luma_channel0);
}
// ... 45 more of exactly that shape ...
}
```

**Why the CUDA arm gets a file of its own** (the owner's refinement, and the reason this is worth two files rather
than one): with it, `src/cuda_backend.cpp` is the only host file in the tree that names a CUDA-side function, and it
is simply not compiled when nvcc is absent. `backend.cpp` becomes routing and nothing else, the two arms are
symmetric -- one file each, both new -- and the encoder gets the mild de-CUDA-ing the owner asked for without a
single edit inside a kernel file. It also means the `#ifdef NNTC_CUDA` pairs live in exactly two places, the
forwarders and the CMake source list, rather than being spread through the host code.

and in the host, `init_level0(d, m, luma, ch0)` becomes `be::init_level0(d, m, luma, ch0)`, 96 times in `main.cpp`
and once in `export.cpp`. That is the whole edit outside the new files, plus the flags, the banner and CMake.

**The six that take no handle.** `device_select`, `device_create`, `stencil_monomials`, `decoder_ridge_tally`,
`objective_total_ms` and `objective_passes` have no `Device*` to read a backend from, which is why `g_backend` exists
and why `backend_select` is called from the argument parser before anything else touches the device. `device_create`
is the one that allocates the `Device` and fills whichever pointer; every function that takes a `Device*` reads
`d->backend` and never `g_backend`, so the file static is consulted exactly six times per run.

**`stencil_monomials` is the special one, and it must be got right.** `device.cuh:plane_work` calls it **unqualified,
at global scope** (`k.nmono = stencil_monomials(m.c0);`), and `device.cuh` is unmodifiable. So:

* the global definition in `solve_level1.cu:104` stays exactly where it is and keeps serving every CUDA translation
  unit, including `plane_work`;
* the CPU backend's copy is `nntc_cpu::stencil_monomials`, and `cpu_model.h`'s own `plane_work` twin calls **that**;
* `be::stencil_monomials` dispatches between them, which for a pure function of one small integer is a formality --
  and the harness turns the formality into a check by asserting the two agree for every `c0` in 1 to 4 and every
  `c1` the layout allows, once, at start-up under `--backend check`.

**The completeness guarantee, which is the reason this design is safe.** In a build configured **without** CUDA the six
`.cu` files are not compiled, `NNTC_CUDA` is undefined, and the global-scope symbols `init_level0`, `solve_decoder`,
`objective_eval` and the rest **do not exist**. `model.h` still declares them, so a call site that was missed still
compiles -- but nothing defines the symbol, and **the link fails and the linker names it.** A CPU-only build that
links clean is therefore a machine-checked proof that every one of the 97 call sites goes through the dispatcher.

That matters more than it looks. Without it, a missed call site in a build that *has* CUDA would silently run the CUDA
function under `--backend cpu`, and the symptom would be an encode that is mysteriously correct. The CPU-only
configure is added to the gate (7.1) precisely so that this proof is re-run on every change, and it is the same build
the ThreadSanitizer arm uses (3.4).

**What `main.cpp` sees.** It includes `backend.h` instead of nothing new, holds a `be::Device*` where it held a
`DeviceModel*`, and calls `be::` names. The round loop's structure, its 96 calls, their order and their arguments are
otherwise untouched.

### 1.8 The duplication, counted, and why it is deliberate

Decision 4 says copy, not extract. That is a real cost and it is stated rather than smoothed over.

| copied from | into | lines |
|---|---|---|
| `sample.cuh`, the whole file: `clamp_index`, `bilinear_tap_pixel`, `sample_bilinear`, `build_phi`, `subtexel_offset`, `site_delta`, `site_row` | `cpu/cpu_sample.h` | 178 |
| `device.cuh:32-119`: the caps, `ChannelBits`, `REDUCE_BLOCKS`, `SITE_BLOCKS`, `STENCIL_BLOCKS`, `PARTIAL_ROWS`, the `ROW_*` and `SC_*` slots, `stencil_offset`, `stencil_slot` | `cpu/cpu_constants.h` | ~90 |
| `device.cuh`: `PlaneWork` and `plane_work` | `cpu/cpu_model.h` | ~75 |
| `solve_decoder.cu`: `tri_index`, `LS_SLOTS`, `gauss_jordan` | `cpu/cpu_solve_decoder.cpp` | ~65 |
| `solve_decoder.cu:242-458`: `RIDGE_LADDER`, `LS_ACCEPT_RISE`, the tally, the trace, the normaliser, `qform`, the four rungs, the float round trip, the acceptance bar and the constant-fit fallback | `cpu/cpu_solve_decoder.cpp` | ~160 |
| `init.cu`: `fit_range` (475-507), `jacobi_eigen` (744-790) | `cpu/cpu_init.cpp` | 80 |
| `solve_level1.cu:104-258`: `stencil_monomials`, `stencil_monomial_values`, `build_form_matrices`, `build_monomial_matrices`, `build_monomial_matrices_level0`, `footprint_range`; and `level1_range` (1147-1159) | `cpu/cpu_solve_level1.cpp` | ~165 |
| `objective.cu`: `normalise` (224-251), `twin_sample` (297-316) | `cpu/cpu_objective.cpp` | 48 |
| `refine_bc.cu`: `bc4_weight8`, `bc4_entry`, `bc0_value` (102-131, with the `__CUDA_ARCH__` branch collapsed to its host half), `refine_grid`, `plane_blocks` | `cpu/cpu_refine_bc.cpp` | ~45 |
| `objective.cu:317-380`: `objective_check_host` | `cpu/cpu_objective.cpp` | 64 |
| `solve_level1.cu:1502-1688`: `solve_level1_dense_host` | `cpu/cpu_solve_level1.cpp` | 187 |
| | **total** | **about 1160** |

Of that, **251 lines are the two brute-force twins** (1.4: copying them does not weaken them), and **about 343 lines
are the sampling rule and the constants**, where a transcription error is a compile error or a wrong array shape
rather than a wrong last bit. The genuinely delicate duplication is the remaining **~570 lines**: the ridge ladder,
`gauss_jordan`, `jacobi_eigen`, `fit_range`, the form and monomial matrices, and `bc0_value`.

**Why this is the right trade, in three sentences.** The encoder is frozen, so the originals cannot drift out from
under the copies: the hash gate of 2.5 makes that a fact and not a hope. The alternative -- extracting the shared
code -- was the first draft's one non-mechanical change and the subject of two review findings, because an extraction
*can* change what the CUDA path computes while a copy provably cannot. And the copies are not guarded by anyone
keeping them in step by hand: the harness of section 4 runs both implementations on identical inputs and diffs the
results, so a copy that disagrees with its original is caught the first time it runs, by a number rather than by a
reading.

**What the duplication costs when the CUDA side legitimately changes.** If a future commit deliberately edits a kernel
file, the hash arm fails and names the file, the owner approves the new hash, and the corresponding CPU copy is
updated in the same commit with the harness re-run. That is risk 14 and it is paid, not retired.

`src/cpu/cpu_sample.h` carries one deliberate spelling change, and it is not cosmetic. `sample.cuh` calls **`floorf`**
unqualified, twice (lines 48 and 49), and nothing else from the math library -- not `fminf`, not `fmaxf`. `<cmath>`
guarantees `std::floor` and `std::floorf`; it does **not** guarantee `::floorf` at global scope, and that is exactly
the kind of thing that compiles on x86 glibc and MSVC and then does not on a different Arm toolchain -- the platform
this backend exists for. The copy therefore includes `<cmath>` and calls `std::floor` on the float argument, which is
`floorf` exactly and by definition, with no rounding consequence at all.

### 1.8a A third arm, later: the seam is built for N backends, not for two

The owner's intent for the dispatcher is a **platform layer**, not a CUDA-or-not switch: "dispatcher routes to either
a cuda platform layer or a CPU platform layer. Later: we can add HIP for AMD, etc." Nothing in this plan builds a
third arm, and nothing in it should make one awkward. Four consequences for the code written in step A:

* **`Backend` is an enum, never a bool**, and `--backend` takes a name, never a flag. Adding `hip` is a new
  enumerator and a new string, not a change of shape.
* **One file per arm, named for the arm.** `cuda_backend.cpp` beside a future `hip_backend.cpp` beside `cpu/`.
  Each is compiled only when its toolchain is present, and `backend.cpp` never names any arm's internals.
* **The `Device` handle carries one typed pointer per arm**, which is the right trade at two and stays readable at
  three. If a fourth ever appears, the forwarders should become a table of function pointers filled in by each arm at
  registration, which removes the `#ifdef` chain entirely; that is a mechanical change from this shape and it is
  deliberately not made now, because a vtable for two arms is indirection without a reader.
* **CMake keeps a per-arm source list.** `NNTC_CUDA_SOURCES` is already that pattern; a third arm adds a list and a
  probe beside `check_language(CUDA)`, and the always-built host and CPU lists do not move.

What a HIP arm would actually cost, so the idea is not free-floating: HIP's kernel language is close enough to CUDA
that the six `.cu` files would port largely by renaming the runtime calls, but they are not to be edited (decision 1),
so a HIP arm means a parallel set of kernel files. That is a much bigger undertaking than this plan, and its value
depends on whether AMD users want to encode on the GPU rather than merely fast enough on the CPU. The CPU backend
lands first precisely because it serves every vendor and every architecture at once.

### 1.9 The round loop can stop at different rounds, and the plan must say so

`src/main.cpp:2065-2072`:

```cpp
const bool level1_still = frozen ? moved1 == 0
                                 : delta_norm <= 1e-7 * (plane_norm > 0.0 ? plane_norm : 1.0);
if (moved0 == 0 && level1_still) { stop_reason = "no-move"; break; }
tol_hits = drop < o.tol ? tol_hits + 1 : 0;
if (tol_hits >= 2)                { stop_reason = "tol";     break; }
```

Three facts about it that the first draft's cross-backend arm did not account for.

* the **no-move** stop is decided on `moved0` and `moved1` -- atomic counters out of `k_level0_search` and
  `k_quant_sweep`, both of which count discrete decisions that a last-bit difference can flip -- and, while level 1 is
  continuous, on `delta_norm` against `plane_norm`;
* it is evaluated **before** the `--tol` stop;
* it is **unaffected by `--tol 0`**. With `o.tol` zero, `drop < 0.0` is false at every round, `tol_hits` never
  reaches 2, and the tol stop never fires -- but the no-move stop still can, and if it fires one round apart on the
  two backends every number downstream differs by far more than any honest tolerance.

So `--tol 0 --rounds R` does **not** pin the round count, and the cross-backend arm must (7.4) assert the printed
`stop_reason` and the round count are equal, assert that neither run stopped early, and move `moved0` / `moved1` out
of the draft's "printed, not asserted" list into quantities with a measured bar. They are the earliest visible
symptom of a discrete divergence and they are the one thing worth failing on.

### 1.10 The report surface that has no CPU counterpart

Six report fields and two printed paragraphs come from CUDA events or from CUDA facts. Each one's decision, stated:

| what | where | becomes |
|---|---|---|
| `Level1Report::ms_assemble`, `ms_precond`, `ms_cg`, `ms_sweeps` | `model.h:381, 396` | a `std::chrono::steady_clock` span around the same four phases, taken on the calling thread. Same field, same units, same meaning; the CPU's are wall clock rather than device time, which for a serial walk is the same thing |
| `Level0Report::ms` | `model.h:443` | the same |
| `BcRefineReport::ms` | `model.h:517` | the same |
| `print_banner`'s `device %d: %s, compute capability %d.%d` | `main.cpp:1500` | **under `cuda` it is byte-identical to today's**, so the gate's banner pins do not move. Under `cpu` it is a different line: the backend, the worker count, `std::thread::hardware_concurrency()`, a platform-read CPU name, and (1.11) the bytes the backend is about to allocate. The CPU line never invents a compute capability |
| `CUDA events around the` | `main.cpp:2941` | backend-conditional: `the wall clock around the` under `cpu` |
| `The planes are issued together, so these overlap and their sum exceeds the block's own time` | `main.cpp:2965-2967` | **this is false on the CPU**, which walks planes serially (3.1 rule 6). Backend-conditional: under `cpu` it reads `each plane's own phases on the wall clock. The planes are walked one at a time, so these SUM to the block's own time`. The per-plane percentages below it stay as they are and mean more, not less |
| `device %.1f MB` | `main.cpp:2958` | `device_memory` on the CPU sums the same buffers and reports the same number under the same name, so the line reads the same way on both backends. That is the point: 605.6 MB for a 1024x1024 four-texture material is a number a user needs whichever backend is holding it |

These are report lines, which decision 3 names as in scope. Nothing else in `main.cpp` below the banner changes.

### 1.11 Host memory, and what happens when an allocation fails

Two facts from the tree. `device_memory(d)` is printed in the **REPORT** (`main.cpp:2637` reads it, 2958 prints it),
after the encode, so as things stand the banner cannot warn a user before a run that will not fit. And `init.cu:103`
routes every device allocation through `CUDA_CHECK`, which prints a named error and exits with the process's one
failure status, while `new[]` and `std::vector` throw `std::bad_alloc` that nothing in this tree catches. (Since
`v1.3.4` `main()` does catch it, and any other exception, including one from a CPU worker thread: one ERROR line naming
the stage and exit 1. GPU out of memory has its own ERROR line with the sizes and free memory.)

**Two decisions.**

1. **The CPU backend's allocation failure exits 1 with an ERROR line, like everything else in this tree.** Every CPU
   allocation goes through one `cpu_alloc<T>(n, what)` which adds to the running byte total and, on failure, prints
   `ERROR: cannot allocate N bytes for <what>: no encode` and exits with `EXIT_FAILURE`. It catches `std::bad_alloc`
   itself rather than leaving it to an uncaught exception, so the message is the tree's and not the runtime's. No
   top-level `try` is added to `main`: that would be a "while we are here" change and decision 3 forbids it. (Added
   later at the owner's request, `v1.3.4`: see the note above.)
2. **Under `--backend cpu` the banner names the total before anything is allocated.** The byte total is a pure
   function of `Model` and the source chain, so `nntc_cpu::bytes_for(m, chain)` can be called from `print_banner`,
   which runs before `device_create`. This adds a line to the **CPU** banner only, so the CUDA banner stays
   byte-identical and the gate's pins do not move. It is the answer to "the banner cannot warn first".

### 1.12 One difference that will not be the port's fault

`src/stb_image_resize2.h` selects a SIMD path from the instruction set it is compiled for -- `STBIR_SSE2`,
`STBIR_AVX`, `STBIR_AVX2` on x86, `STBIR_NEON` on Arm, chosen in the `#if` ladder at its lines 1174-1190. The encoder
runs it on the **host**, in the source chain, on both backends. `docs/DESIGN.md` section 6 already says this out loud
for the cross-toolchain case ("that resizer chooses a code path from what the CPU supports").

The consequence for this plan: **an x86 CPU-backend encode and an Arm one may differ in the source chain rather than
in the port**, whenever `--mip-filter` is anything but `default`. Same-machine CUDA-against-CPU is unaffected, because
both read the same chain built by the same host code in the same process. Noted here so that nobody later attributes
an Arm difference to a transcription error in `src/cpu/`; the first thing to check when an Arm number moves is whether
the command line named a filter.

---

## 2. Step A in detail

Step A adds **no CPU encoder at all**. Its product is a tree in which the CPU backend is a matter of writing
`src/cpu/`, and its proof is that the CUDA encoder's output did not move -- which is now nearly free to believe,
because nothing inside the encode changed: not a kernel, not a header a kernel reads, not a line of the round loop.
The only things that could move a byte are a compile flag the CMake restructure disturbed and a call site the `be::`
prefix mistranscribed, and the linker catches the second.

### 2.1 File by file, and the class of change each one takes

Four classes. **H** is *hashed and untouched*. **C** is a call-site rename with no other edit. **A** is additive: new
lines beside existing ones, nothing removed or reordered. **B** is the build. There is no class that can change an
arithmetic result, which is the difference between this table and the first draft's.

| file | class | what, exactly |
|---|---|---|
| `src/device.cuh` | **H** | nothing. Byte-identical, hash recorded |
| `src/sample.cuh` | **H** | nothing. Byte-identical, hash recorded |
| `src/init.cu` | **H** | nothing |
| `src/solve_decoder.cu` | **H** | nothing |
| `src/solve_level1.cu` | **H** | nothing |
| `src/solve_level0.cu` | **H** | nothing |
| `src/objective.cu` | **H** | nothing |
| `src/refine_bc.cu` | **H** | nothing |
| `src/model.h` | -- | **unchanged.** `reconstruct_levels`' declaration is not edited in place; `backend.h` declares the `be::Device*` form and `export.cpp` includes `backend.h`. The `model.h` declaration stays and simply has no caller, which the compiler is content with and which keeps the seam's own documentation whole |
| `src/main.cpp` | **C, A** | 96 call sites gain a `be::` prefix and one local changes type from `DeviceModel*` to `be::Device*`; `--backend` and `-j` are parsed beside `--device` (2.3); the fallback logic replaces the refusal at 1699-1703; `print_banner` gains a backend line (1.10); two report paragraphs become backend-conditional; two lines of usage text. **The round loop, lines 1812 to 2399, keeps every statement it has** |
| `src/export.cpp` | **C** | one call site: `decode_plane` inside `reconstruct_levels` (437-442), and that function's own signature and include |
| `src/mips.cpp` | -- | untouched. It has no seam call at all |
| `src/backend.h`, `src/backend.cpp` | new | section 1.7: pure routing, no CUDA header |
| `src/cuda_backend.h`, `src/cuda_backend.cpp` | new | section 1.7: the CUDA arm, 46 one-line wrappers, compiled only with nvcc |
| `CMakeLists.txt` | **B** | 2.2 |
| `tests/run_checks.py` | -- | **untouched in step A.** It is the instrument, and an instrument that changes in the same commit as the thing it measures measures nothing. The hash arm and the CPU-only configure arm land in step B |

### 2.2 CMake, with and without nvcc

Today `add_executable(nntc_encode ...)` sits inside `if(CMAKE_CUDA_COMPILER)` (`CMakeLists.txt:69-141`), so a machine
without the toolkit gets a WARNING and no encoder. The encoder target moves out of that branch and the `.cu` files
become conditional:

```cmake
set(NNTC_HOST_SOURCES src/main.cpp src/mips.cpp src/export.cpp src/backend.cpp
                      src/model.h src/image.h src/options.h src/backend.h
                      shared/bc_pack.h shared/nntc_json.h shared/json.h
                      src/stb_image.h src/stb_image_write.h src/stb_image_resize2.h)
set(NNTC_CUDA_SOURCES src/init.cu src/solve_decoder.cu src/solve_level1.cu src/solve_level0.cu
                      src/refine_bc.cu src/objective.cu src/device.cuh src/sample.cuh
                      src/cuda_backend.cpp src/cuda_backend.h)   # the arm goes with the kernels
set(NNTC_CPU_SOURCES  "")     # empty until stage C1; then the eleven files of section 1.6

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    ... the 12.8 version check and find_package(CUDAToolkit 12.8 REQUIRED), unchanged ...
else()
    ... the FATAL_ERROR when -T cuda= was asked for by name, unchanged ...
    message(WARNING "No CUDA compiler was found: the encoder nntc_encode is built CPU-ONLY and runs under "
                    "--backend cpu (--backend cuda will refuse by name). ${NNTC_WITHOUT_CUDA}. "
                    "For the CUDA backend install the CUDA toolkit, 12.8 or newer: ... unchanged ... ")
endif()

add_executable(nntc_encode ${NNTC_HOST_SOURCES} ${NNTC_CPU_SOURCES})
target_include_directories(nntc_encode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/shared)
find_package(Threads REQUIRED)
target_link_libraries(nntc_encode PRIVATE Threads::Threads)
if(CMAKE_CUDA_COMPILER)
    target_sources(nntc_encode PRIVATE ${NNTC_CUDA_SOURCES} src/backend_check.cpp)
    target_compile_definitions(nntc_encode PRIVATE NNTC_CUDA)
    target_link_libraries(nntc_encode PRIVATE CUDA::cudart_static)
    set_target_properties(nntc_encode PROPERTIES CUDA_ARCHITECTURES "80;86;89;90;120" ...)   # unchanged
endif()
... the MSVC runtime, the warning levels and -ffp-contract=off, all unchanged ...
message(STATUS "nntc_encode backends: cpu${NNTC_HAS_CUDA_TEXT}")
```

Five things worth saying out loud about it.

* **In step A the CUDA build's every compile option is the option it has today.** That is the only way this file can
  move a byte, and it is checked by 2.4's invariant rather than by reading.
* **The WARNING's meaning changes and its tone should not.** It stops saying "the encoder is skipped" and starts
  saying the encoder is built CPU-only, names `--backend cpu` as what runs, and keeps the existing paragraph naming
  the CUDA toolkit and how to install it on each platform. It stays a WARNING, because `CMakeLists.txt:15-18`'s own
  rule is that a build that came out smaller than the tree says so in colour.
* **The `-T cuda=` FATAL_ERROR stays exactly as it is** (`CMakeLists.txt:52-57`), and it is worth noticing that it is
  the configure-time twin of 2.3's run-time rule: asking for CUDA by name and not getting it is an error at both
  ends, and a command line that never mentioned CUDA gets the quiet path at both ends.
* **`-ffp-contract=off` on the `CXX` branch already covers `src/cpu/`**, because the new files are `CXX` sources on
  the same target and the option is a `$<$<COMPILE_LANGUAGE:CXX>:...>` generator expression -- on gcc and clang. On
  MSVC there is no such option today and 1.5 is why that is a step-A task.
* **`MSVC_RUNTIME_LIBRARY "MultiThreaded"`** is set today because `CUDA::cudart_static` is built against the release
  static runtime in both configurations. In a CPU-only build nothing requires it. Keep it anyway: one runtime in both
  builds is simpler than two rules, and `std::thread` is fine on the static runtime.

### 2.3 `--backend`, `-j`, and the fallback rules

Both flags are parsed in the existing chain with the existing helpers, in the shape `--device` already has
(`main.cpp:692-696`):

```
  --backend B       auto|cuda|cpu|check: which encoder backend runs                    (default auto)
  -j N              CPU backend worker threads, 0 = as many as the machine has         (default 0)
```

* **`--backend auto` is the default** and tries CUDA first. It falls back to the CPU backend, with a WARNING line
  naming why, in four cases: a binary built without CUDA; no NVIDIA device present; a driver that will not
  initialise; and a `--device N` that does not exist. Today `main.cpp:1699-1703` prints
  `ERROR: no CUDA device: no encode` and returns 1 for the middle two; **that becomes the fallback path**, and the
  WARNING carries the same detail the ERROR carried.
* **`--backend cpu` always works and never warns.** It is available in every build by construction.
* **`--backend cuda` is an ERROR when CUDA is unavailable.** The principle of least surprise: a caller who demanded
  CUDA is told it cannot be used rather than quietly given something else. The message names what was demanded, why
  it cannot be used and the remedy, in the shape of the tree's other refusals:

  ```
  ERROR: --backend cuda was asked for, but this build has no CUDA backend (it was configured without a CUDA
         compiler); re-configure with a CUDA 12.8+ toolset, or run --backend cpu
  ERROR: --backend cuda was asked for, but no NVIDIA device was found; run --backend cpu, or --backend auto
         which falls back to it
  ERROR: --backend cuda --device 3 was asked for, but this machine has 1 CUDA device (0); run --device 0
  ```

* **`--backend check`** is the harness of section 4 and exists only in a build carrying both backends. Asking for it
  in a CPU-only build is the same shape of error as asking for `cuda`.
* **`-j`** takes `0` to mean `std::thread::hardware_concurrency()`, clamped into `[1, 1024]`; a
  `hardware_concurrency()` of 0 becomes 1. Parsed with `value_int` so that `-j 8x` is refused rather than silently
  read as 8. Under `--backend cuda` it is a WARNING and otherwise idle.
* **`--device N`** keeps its meaning under `cuda` and `auto`, and is a WARNING and otherwise idle under `cpu`,
  because a caller who names a GPU and a CPU backend in one command line has made a mistake worth naming.
* **The report says which backend ran, every time**, in the banner (1.10) and again in the report's first line. Not
  only under `cpu`: a CUDA run says `cuda` too, so that a log answers the question without the reader having to
  recognise the absence of a line.
* `be::device_select(o.device, info)` keeps the old function's signature. Under `cuda` it is the `init.cu` function
  (since `v1.3.3` it no longer prints: a device below the compute-capability floor is refused with its name and
  capability left in `DeviceInfo`, and `select_backend` prints the WARNING or the ERROR); under `cpu` it fills
  `DeviceInfo` with the thread count and a name read from the platform and returns true.

### 2.4 The invariant that proves step A safe, and how it is checked

> **Every asset the encoder writes at the end of step A is byte-identical to the one `v1.1.28-woa-tested` writes from
> the same command line, and both gates are green.**

**1. The release gate, `tests/run_checks.py`, in full, twice.** Windows (VS 2026, CUDA 13.4) and WSL (gcc 13.3,
CUDA 13.3) -- the two toolchains `docs/DESIGN.md` section 6 measured and `README.md` names -- and, on Windows, a
Release and a Debug build. The arms that prove step A specifically: `determinism_check`, the fifteen byte-for-byte
arms, `ridge_tally` all-standard on `tiny.png` with the tag's own call count, `ill_conditioned_checks`,
`objective_checks` to 1e-9, `level_checks` and `quantisation_checks`, the three block (b) arms, and `check_tree`.

**2. A direct comparison against the tag**, which the gate does not do and which is the actual invariant. Build
`v1.1.28-woa-tested` into a second directory, run the two binaries over a fixed list of command lines into two output
directories, and `cmp` every `.dds` and `_nntc.json`. A throwaway script under a gitignored `out_` directory; not a
tracked file.

**The list, widened.** The draft's fifteen lines missed several branches of the six kernel files, and the review named
them. The corpus is now **twenty-six lines**:

| # | command line | the branch it reaches |
|---|---|---|
| 1 | `tests/tiny.png` bare | the default path, `--l0 bc8` |
| 2 | a 1024x1024 four-texture material, default layout | the chain, the streams, the staging |
| 3 | `--l0 palette --c0 2 --bits0 3` | block (c), `total_bits` 6 |
| 4 | **`--l0 palette --c0 3 --bits0 3`** | **`total_bits` 9, so `joint == false`: the coordinate-sweep branch with its early break** (`solve_level0.cu:288`) |
| 5 | `--c0 1` | one level-0 channel |
| 6 | `--c0 3` | three, the `t2` two-file layout |
| 7 | **`--c0 4 --bits0 2`** | four channels at the smallest depth |
| 8 | **`--bits1 4`** | level 1 off 8 bits: `replicate_index`, `level1_value`'s grid |
| 9 | `--init0 luma` | |
| 10 | `--init0 texture` | |
| 11 | `--init0 residual --init0-scope chain` | the 4096-bin histogram over the whole chain |
| 12 | `--init pca` | the covariance reduction and `jacobi_eigen` |
| 13 | `--q1-start 0` | the frozen-from-the-start control path |
| 14 | `--bc-refine 0` | |
| 15 | `--bc-outer 0` | |
| 16 | `--k 0` | centre sites only |
| 17 | `--mip-weight sqrt --mip-mix 0.25` | the per-plane weights |
| 18 | **`--l0 palette --bc0 0`** | the uncompressed level-0 twin (the default `--l0 bc8` refuses `--bc0 0` by name, so the bare flag reaches nothing; step A's corpus found this) |
| 19 | **`--bc0 both`** | both files written |
| 20 | **a six-texture material** | `nout` 18, `MAX_NOUT`, the 44 KB staging |
| 21 | **a 16-bit input** | `image.h`'s widest load path |
| 22 | **a grey input** | one source channel |
| 23 | **an alpha input** | `alpha_uses` and the four-channel target |
| 24 | **a `--mip-filter` line** (`mitchell` + `srgb` + `normal_map`) | the vendored resizer, and 1.12's ISA note |
| 25 | **a material JSON** | the merged per-texture settings |
| 26 | **`--check`** | `objective_check_host`, `solve_level1_dense_host` and the finite-difference probe |

**3. The log's numeric columns.** The printed text should differ only in timings and in the new backend line. Diff the
two logs with the millisecond columns stripped; anything else that moved is a finding.

**4. The contraction check** (1.5), which is new work in step A and is two lines plus an assertion: a host-only sweep
over `level1_value` / `level1_index` for every `bits` in 1 to 8 and a spread of `lo` / `hi`, asserting the round trip
and dumping the values as hex floats; run on MSVC x64, MSVC ARM64, gcc and clang, and the four dumps diffed. A
difference on ARM64 is retired with `/fp:contract-` and re-run.

**Correction, found in step A: that design cannot see what it is looking for.** The fused and the separate forms of
`lo + frac * span` differ in the last bit of a DOUBLE, and the value is rounded to a FLOAT, which discards nearly all
of it: a build with both expressions deliberately fused by `std::fma` dumped byte-identically to the shipped one over
7,660 swept values. The difference survives to the float about once in 5,000 values, where `lo` and `frac * span`
nearly cancel. So `tests/fp_contract_check.py` carries the readable sweep (which proves nothing on its own and says
so), **forty witnesses found by search** at which a fused multiply-add provably gives a different float -- each
carrying both answers, so one machine decides alone and exits 1 naming `/fp:contract-` -- and a two-million-value
sweep folded into one hash for machines to compare. A calibration build with contraction forced on fails all forty,
so the check is shown able to fail.

If all four hold, step A ships as a short series -- the dispatcher, then the call sites, then the flags and the
fallback, then CMake, then the contraction check -- tagged in the tree's own style, `v1.2.0-cpu-dispatch`.

### 2.5 How "the kernels are not modified" is checked by a machine

Decision 1 is not a rule anyone is asked to remember. `tests/kernel_hashes.txt` carries nine lines,
`sha256  path`, for `init.cu`, `solve_decoder.cu`, `solve_level1.cu`, `solve_level0.cu`, `refine_bc.cu`,
`objective.cu`, `device.cuh`, `sample.cuh` and `model.h` (the ninth because its declarations are the CUDA arm's),
recorded at `v1.2.0-cpu-dispatch`, whose nine files are byte-identical to `v1.1.28-woa-tested`'s. The gate's first
arm, `kernel_hash_check`, re-hashes the nine and fails **naming the file whose hash moved**.

The file is added in **step B**, not step A, for the same reason `run_checks.py` is untouched in step A: the
instrument and the thing it measures do not change in one commit. From step B onward, the only way a kernel file
changes is a commit that also changes `tests/kernel_hashes.txt`, which is a line in a diff that the owner sees.

---

## 3. The CPU backend's structure

### 3.1 The thread pool's contract

```cpp
struct Pool
{
    explicit Pool(int threads);                             // built once, in device_create
    int  threads() const;
    void run(int chunks, const std::function<void(int)>& body);   // body(c) for every c in [0, chunks)
};
```

Six rules. They are the whole of decision 9's second half, and every one is checkable by reading a call site.

1. **`chunks` is a property of the kernel, never of the thread count.** A kernel mirroring a fixed CUDA grid passes
   `SITE_BLOCKS` (512) or `REDUCE_BLOCKS` (256). A one-thread-per-element kernel passes `ceil(elements / CPU_CHUNK)`
   with `CPU_CHUNK` a compile-time constant. `Pool::run` must never see `threads()` in its first argument, and that
   is the single rule a reviewer checks.
2. **The chunk-to-element map is fixed.** Chunk `c` covers the same elements at every thread count, on every run, on
   every machine.
3. **Static partition, no work stealing.** Worker `w` of `T` takes a contiguous half-open range of chunks, computed
   once. Stealing would not in fact break rule 2 -- the chunk bodies do not interact -- but it is forbidden anyway,
   because a partition a reader can compute is worth more here than a few percent of load balance, and because the
   owner said so.
4. **Every reduction is folded over chunks in increasing `c`, after the join.** No atomic, no lock, no partial sum
   written by one worker and read by another. Counters that only reach the report are folded the same way, which
   makes them exact as well as deterministic.
5. **`-j1` runs the chunk loop on the calling thread with no pool at all**, and by rules 1 to 4 produces the same
   bytes as `-j32`.
6. **No nesting.** One `Pool::run` is live at a time. In particular the per-plane parallelism the CUDA path gets from
   streams is *not* reproduced as an outer parallel loop over planes: the CPU backend walks planes serially and
   parallelises inside each kernel. Planes share no variable, so an outer loop would be correct -- but it would make
   the chunk count depend on the plane count and the thread count together, which is rule 1 gone, and the base plane
   is 86 to 96 % of the work anyway, so there is nothing to win. This is also why `main.cpp:2965-2967`'s prose is
   false on this backend (1.10).

**`-j1` is the adversarial thread count, not `-j7`.** The draft had that backwards and the review is right. Every
kernel writes to its own output slot and a contiguous static partition is thread-count-independent by construction, so
`-j7` exercises nothing `-j2` does not: the *arithmetic* cannot notice where the range boundaries fell. The genuinely
different code path is rule 5's, which skips the pool entirely and runs the chunk loop inline -- different control
flow, and the one place a divergence can hide. Two more counts matter for the same reason: **more threads than
chunks** (a deep mip plane has one or two chunks; a worker with an empty range must do nothing and must not deadlock)
and **`-j0`**, which is the default and goes through `hardware_concurrency()`. Section 7.3 is the arm.

The pool is built in `device_create` and destroyed in `device_destroy`; workers park on a condition variable between
`run` calls, so the per-kernel cost is two notifications and a join, not a thread creation. At the measured kernel
counts -- 13 launches per CG iteration, up to 200 iterations, times the plane count -- that matters: a 2048x2048
five-texture encode issues on the order of a hundred thousand `run` calls, so `run` has to cost microseconds. If it
does not, the fallback is a persistent spin-then-park barrier, which is a change inside `pool.h` and nowhere else.

### 3.2 Reproducing CUDA's fixed-grid two-stage reductions

Ten kernels reduce. Nine use the same idiom -- `__shared__ double sh[N]`, a power-of-two halving tree, one partial per
block into a fixed slot, then a second pass that adds the partials by index -- and one (`k_ls_accumulate`) uses a
staged tile instead. `docs/DESIGN.md` section 6 is explicit that this shape is the reason two encodes agree, and that
a reduction whose blocks add into one accumulator with atomics "returns whatever the scheduler made of it that time".

**The rule for the CPU backend: replay the shape, not just the result.**

For a kernel launched `<<<B, 256>>>` with a grid-stride loop and a shared tree, chunk `b` of `B`:

1. holds a `double slot[256]`, one per CUDA thread of that block;
2. `slot[t]` is the serial sum, in increasing index order, of exactly the elements CUDA thread `(b, t)` would have
   visited: indices `b * 256 + t`, then `+ B * 256`, and so on;
3. then replays the halving loop verbatim --
   `for (half = 128; half > 0; half >>= 1) for (t < half) slot[t] += slot[t + half];` -- and writes `slot[0]` into
   `partial[row * B + b]`;
4. and the second stage adds the `B` partials by index, on one thread, exactly as `k_obj_reduce`, `k_reduce_sums`,
   `k_reduce_diag` and `k_reduce_range` do.

What it buys is worth far more than its cost on the base plane: **the summation contributes no difference at all
between the backends.** A flat left-to-right sum over 256 doubles is not bitwise equal to the tree, so a "simpler" CPU
reduction would introduce a divergence that is entirely avoidable and entirely indistinguishable, at comparison time,
from a real porting mistake. With the tree replayed, every remaining difference is attributable to 1.5's fp32
contraction, which is the one cause that cannot be removed. That property is also what lets section 4's harness hold
half its comparisons **to the bit** rather than to a tolerance, which is worth more than the cycles.

**And on a deep plane it is disproportionate, which the draft did not say.** 256 chunks times a 256-slot tree is
65,280 tree additions plus half a megabyte of scratch touched, to reduce the few thousand values a deep chain plane
holds -- roughly sixteen additions of overhead per useful one. Times several `Pool::run` calls per CG iteration, times
up to 200 iterations, times six or seven planes.

**The one permitted remedy, and why it is permitted.** Not a different reduction: decision 6 forbids that, and the
whole value of the replay is that it is exact. Instead, a **size threshold on the scheduling**: when a kernel's
`chunks * elements_per_chunk` falls below a fixed compile-time bound, the chunk loop and the tree run inline on the
calling thread with no `Pool::run` at all. That is rule 5's path applied per kernel. The arithmetic is identical --
the same chunks, the same slots, the same tree, the same fold order -- and the threshold is a function of the
**plane's size**, not of the thread count, so rule 2 holds: one plane takes one path at every `-j`. Measuring whether
it is needed is stage C1's first task, before a single CG kernel is written.

`k_ls_accumulate` is the exception and is easier than it looks, because **the CUDA kernel is already panelled**: a
block stages 256 sites' `phi` and `target` into shared memory, then walks the up to 775 normal-equation entries, each
one a 256-term double dot product over the staged tile. The CPU form is the same two phases with the barriers
deleted: chunk `b` of 512 walks chunks `b`, `b + 512`, ... of 256 sites, fills a `(mm + nout) x 256` staging tile, and
sums each entry over it in index order into a private `double acc[entries]`. `LS_STRIDE`'s 257 is a shared-memory
bank-conflict fix and becomes 256 on the CPU, which changes addressing and not one addition.

Two reductions are exempt and should be said out loud. `k_proj_hist`'s `atomicAdd` on `unsigned long long` is integer
and therefore already order-independent; the CPU form is a private 4096-bin histogram per chunk, summed in chunk
order -- and it must be **added into a histogram that lives across the whole plane loop**, because `init.cu:1277-1290`
memsets the device histogram once and accumulates every plane of the chain into it before reading it. And
`k_bc_refine`'s two `atomicAdd` calls on `double` are the one place `docs/DESIGN.md` section 6 admits an order
dependence; they are a log number that no decision reads, and the CPU backend's chunk-ordered fold makes them
*stricter* than CUDA's, not looser. Nobody should later "fix" the small difference that shows up in that printed
decrease.

### 3.3 How each kernel family maps onto the pool

| family | kernels | CPU parallel axis | chunk count |
|---|---|---|---|
| per-texel, no reduction | the inits, the snaps, the preconditioner, the mat-vec, the axpys, `k_bc_decode`, `k_decode` | the texel or the value index | `ceil(n / CPU_CHUNK)` |
| fixed-grid reductions | the objective, the dots, the diagonal, the range, the movement probe, the covariance, the peak | the CUDA block index | `SITE_BLOCKS` or `REDUCE_BLOCKS`, replayed per 3.2 |
| the staged accumulation | `k_ls_accumulate` | the CUDA block index | `SITE_BLOCKS` = 512 |
| the gather assemblies | `k_stencil_assemble`, `k_stencil_assemble0` | the unknown plane's texel | `ceil(texels / CPU_CHUNK)` |
| four-colour sweeps | `k_level0_search`, `k_quant_sweep` | the texels **of one colour** | `ceil(colour_texels / CPU_CHUNK)`, four sequential `run` calls |
| per-block refinement | `k_bc_refine` | the 4x4 block **of one colour** | `ceil(colour_blocks / CPU_CHUNK)`, four sequential `run` calls |
| single-thread finishers | `k_reduce_*`, `k_cg_alpha`, `k_cg_beta` | none | a plain serial function, no `run` at all |

The four-colour rows carry the whole of the port's correctness argument for blocks (b), (c) and the refinement.
`solve_level0.cu`'s header proves that two texels of one colour class are two apart while a bilinear footprint spans
only adjacent texels, so **within a colour the texels are provably independent and may be written in place**, which is
exactly what the CUDA kernels do (`p.v0` is passed as both input and output). **Across colours the dependency is
hard**: pass 1 reads pass 0's writes. So each colour is one `Pool::run` and the four `run` calls are a sequential
barrier. Get that wrong and the backend is neither correct nor deterministic; get it right and no synchronisation is
needed inside a colour at all.

### 3.4 Zero data races: the per-kernel audit, the fixes, and the gate

The owner's rule is absolute: there cannot be any data race in the plain C++ port. Not "benign races documented". The
argument is therefore made **per kernel**, not asserted once for the four-colour schemes.

**The audit.** For every kernel: what a worker writes, what it reads, and why no worker can read what another writes.
"This launch" means the one `Pool::run` under discussion; a value written by a previous launch is read-only here,
because rule 6 says one `run` is live at a time and the join is a happens-before edge.

| kernel(s) | a worker writes | a worker reads | why no worker reads another's write |
|---|---|---|---|
| `k_init_level0`, `k_init_level0_cont`, `k_seed_channel` | `v0[t*c0..]`, `k0[t*c0..]` of its own texel | `src`, the palette, the grid | one texel per worker, disjoint; the inputs are not written by this launch. **Ported (C3), `cpu_init.cpp`.** Chunk `c` covers texels `[c CPU_CHUNK, (c + 1) CPU_CHUNK)` and writes only their `v0` / `k0` entries; `src`, the palette and (for the seed) `resid` are written by no chunk of the run. `k_seed_channel` writes one channel of its own texels only. The `bc8` `k0` zeroing and the host `m.k0` copies happen on the calling thread or through `pool_copy`'s disjoint slices |
| `k_init_level1_box`, `k_block_means` | its own level-1 texel's output | `src` over the texel's footprint | outputs disjoint; `src` is read-only all run. **Ported (C3), `cpu_init.cpp`.** Chunk `c` writes the outputs of its own level-1 texels into `scratch` / `means` and nothing else; the float accumulator is a local of the texel |
| `k_quantise_level1` | element `i` of `out` | element `i` of `in` | **in and out ALIAS at the level-0 call sites.** Safe only because one worker reads and writes exactly one element. The CPU chunking must stay per element; a chunk that read a neighbour would break it. Named as a rule, not as a fix. **Ported (C3), `cpu_init.cpp`.** `quantise_run`: chunk `c` reads element `i` and then writes element `i` of `v` and `k`, for the elements of its own range only, so the level-0 snaps' aliasing (`in == v`) is read-then-write by one thread of one element; the grid (`lo`, `hi`) is written by no chunk |
| `k_residual` | its own base pixel's `nout` residual values | both latents, the decoder, `src` | outputs disjoint; latents read-only here. **Ported (C3), `cpu_init.cpp`.** Chunk `c` writes `resid` for its own pixels only; both latents, the decoder and `src` are written by no chunk |
| `k_cov_partial`, `k_proj_peak` | `partial[entry * B + b]` / `cov[b]`, its own slot | `means` / `resid` | one slot per chunk; the fold is serial after the join. **Ported (C3), `cpu_init.cpp`.** `cov_partials`: chunk `c` writes row `c` of a partial array (`entries` doubles) and nothing else, and the calling thread adds the rows in chunk order after the join, applying the plane weight per partial. `proj_peak`: each chunk returns its maximum into `fold_chunks`' own slot |
| `k_proj_hist` | its own private 4096-bin histogram | `resid` | private per chunk, folded in chunk order into a run-long histogram on the calling thread. **Ported (C3), `cpu_init.cpp`.** Exactly that: chunk `c` of `ceil(texels / CPU_HIST_CHUNK)` counts into its own `PROJ_HIST_BINS` row, and the calling thread adds the rows into the histogram `init0_residual_channel` keeps across every plane of the chain. `CPU_HIST_CHUNK` is 65,536 texels, a constant, so a 4096x4096 plane holds 256 private rows (8 MB) rather than the 4096 that `CPU_CHUNK` would give |
| `k_ls_accumulate` | its own `acc[entries]` and its own staging tile | both latents, `src`, the decoder | tile and accumulator are chunk-local stack/scratch; the per-plane fold is serial. **Ported (C4), `cpu_solve_decoder.cpp`.** Chunk `b` of `SITE_BLOCKS` walks tiles `b`, `b + SITE_BLOCKS`, ... of 256 sites; its tile and its accumulator are locals of the chunk body, and the only shared memory it writes is column `b` of its plane's block of `ls_partial` (one slot per entry, `e * SITE_BLOCKS + b`), which no other chunk writes. Both latents and the source are read-only for the whole call; the decoder is not read at all (the features are the latents' samples) |
| `k_ls_reduce` | `reduction[]` | the partials | single-threaded finisher, no `run`. **Ported (C4), `cpu_solve_decoder.cpp`.** A serial loop on the calling thread after every plane's run has joined; the ridge ladder and the solve that follow are serial host code, as they are on the CUDA side |
| `k_stencil_assemble`, `k_stencil_assemble0` | its own texel's 5 stencil blocks and gradient | `v0`, `v1`, `src`, `mono` | **gather, not scatter**: a texel writes only its own slots; nothing in the launch writes the latents. **Ported (C5), `cpu_solve_level1.cpp`.** Chunk `c` of `ceil(texels / CPU_CHUNK)` writes the five blocks and the gradient of its own texels and nothing else; the latents, the source, the decoder, `cw` and the monomial matrices (rebuilt on the calling thread before the run) are written by no chunk; the per-site scratch is local to the texel |
| `k_dot_partial`, `k_diag_partial`, `k_range_partial`, `k_movement_partial` | `partial[row * B + b]` | the CG vectors, the stencil | one slot per chunk; inputs read-only this launch. **Ported (C5), `cpu_solve_level1.cpp`.** Every one is a `fold_chunks` over texel chunks: chunk `c` returns its partials by value into its own slot, read on the calling thread after the join. The two inner products of the CG are computed inside the passes below, each over the chunk's own values after the chunk has written them (so a chunk reads only what it wrote itself); the range's chunk corrects its own values first and then takes their extremes; the movement probe reads the range from the scalar row, written on the calling thread before the run |
| `k_reduce_sums`, `k_reduce_diag`, `k_reduce_range`, `k_obj_reduce`, `k_cg_alpha`, `k_cg_beta` | the scalar row | the partials | single-threaded finishers, run on the calling thread after the join. **Ported (C5).** Exactly that: the plane's scalar row is written only by the calling thread between runs, and the join and the next run's publication order it against every chunk that reads it |
| `k_precondition`, `k_apply_precond`, `k_negate`, `k_axpy`, `k_xpby`, `k_add_correction` | element `i` of one output vector | element `i` of one or two input vectors | element-wise, disjoint, in-place only on its own element. **Ported (C5).** Fused into three passes per iteration (the start: the preconditioner, `x = 0`, `r = -grad`, `z = Minv r`, `p = z`; the update: both axpys, `z = Minv r`; the direction: `p = z + beta p`), each chunk writing only the entries of its own texels and reading, of what the pass writes, only its own entries |
| `k_matvec` | `ap[t*c..]` of its own texel | its own stencil row, and the **forward neighbour's block from its own row** and the **backward neighbour's transposed from that neighbour's row**, plus `p` at both | reads only `stencil` and `p`, neither written by this launch; writes only `ap`, which nothing here reads. **Ported (C5).** The same pass also takes `p . Ap` over the chunk's own texels, reading the `ap` the chunk has just written; `p` and the stencil are written by no chunk of the pass |
| `k_objective`, `k_decode` | its own partial slot / its own pixel's bytes | both latents, the decoder, `src` | disjoint outputs, read-only inputs. **Ported (C2), `cpu_objective.cpp`.** `k_objective`: chunk `c` of `ceil(sites / CPU_CHUNK)` returns its three sums by value into `fold_chunks`' slot `c`, which only the calling thread reads, after the join; the planes, the source, the decoder and `cw` are written by no chunk. `k_decode`: chunk `c` writes the `nout` bytes of pixels `[c CPU_CHUNK, (c + 1) CPU_CHUNK)` of `rgb8` and nothing else; the copy out is `pool_copy`'s disjoint slices after the join. `objective_check_host` is serial on the calling thread |
| `k_level0_search` | `v0[t]`, `k0[t]` of its own texel, and a private moved counter | its own texel, and neighbour `v0` within the site footprint | **the one exception. See below.** **Ported (C6), `cpu_solve_level0.cpp`, with the fix.** One `Pool::run` per colour, the four in order; chunk `c` covers colour texels `[c CPU_CHUNK, ...)` of the `(colour_x + 2i, colour_y + 2j)` sub-grid and writes `v0` and `k0` of those texels only, and only when the texel moved. It reads `v1`, the source, the decoder, `cw` and the palette (written by no chunk), its own `k0` (written only by itself), and - after the hoist - `v0` only at the taps of a site that has its own texel among its four taps: the unclamped pair of each axis is adjacent and the clamp moves an index by at most the step that took it out of range, so every other tap is within one texel of `(tx, ty)` in each axis and not equal to it, a texel of another colour, written by an earlier run or not at all. The moved count is a chunk partial folded in chunk order: an integer, so exactly CUDA's `atomicAdd` |
| `k_quant_sweep` | `v1[t]`, `k1[t]`, `delta[t]` of its own texel | the stencil's 9 blocks (distance 1) and the neighbours' `delta` (distance 1) | same-colour texels are 2 apart, the footprint is 1, so every neighbour read is of a different colour and is not written by this pass. The intra-texel channel loop is Gauss-Seidel over the **worker's own** texel and touches nothing else. **Ported (C5), `cpu_solve_level1.cpp`.** One `Pool::run` per colour, the four in order (the join is the barrier pass `q + 1` needs); chunk `c` covers colour texels `[c CPU_CHUNK, ...)` of the `(colour_x + 2i, colour_y + 2j)` sub-grid and writes `v`, `k` and `delta` of those texels only. The texel reads its own stencil row, the rows of the neighbours it reads backwards, `grad` and `prev` (none written by the sweeps) and `delta` of its eight neighbours, each at Chebyshev distance 1 and so of another colour, written by an earlier run or not at all. The moved count is a chunk partial folded in chunk order: an integer, so exactly CUDA's `atomicAdd` |
| `k_bc_decode` | its own texel's value, index and `delta` | its block's endpoints and selectors | disjoint. `delta` **aliases the level-0 CG solution vector**; the CPU backend names that alias explicitly rather than inheriting it silently. **Ported (C7), `cpu_refine_bc.cpp`.** Chunk `c` of `ceil(texels / CPU_CHUNK)` writes `v0`, `k0` and `cg0_x` of its own texels and nothing else; the endpoints, the selectors, the grid and `q0_prev` are written by no chunk of the run. The driver passes the buffers as `p.cg0_x` and `p.q0_prev` by name |
| `k_bc_refine` | its own 4x4 block's endpoints, selectors, `v0`, `k0`, `delta`; and a private (decrease, count) pair | its own block's 16 texels, the ring of neighbour `delta` at block distance 1, and its own quadratic | same-colour blocks are 2 apart and the ring is 1, so no worker reads a block another is writing. The two `atomicAdd` on `double` become the private pair, folded in chunk order (3.2). **Ported (C7), `cpu_refine_bc.cpp`.** One `Pool::run` per colour, the four in order; chunk `c` covers `CPU_BLOCK_CHUNK` (64) consecutive blocks of the `(colour_x + 2i, colour_y + 2j)` block sub-grid and each block is one serial routine with its quadratic in chunk-local scratch. A block writes its own endpoints and selectors and `v0`, `k0`, `cg0_x` of its own sixteen texels. It reads the stencil, the gradient and `q0_prev` (written by no chunk), its own texels' `v0`, and `cg0_x` of the texels one step outside its edge - a stencil slot's offset is at most one texel in each axis, forward or read backwards as a transpose - which lie in the blocks one step away on the block grid: another colour, because two blocks of one colour are two blocks (eight texels) apart in some axis. The pass figures are returned by value into `fold_chunks`' slots and folded in chunk order, the colours in order |

**The rule for a fix: a reorder or a snapshot that provably changes nothing computed.** Not a lock, not an atomic, not
a copy of a whole plane, and never a change to what is computed. If a race cannot be removed that way, it is a finding
to bring to the owner before any code is written.

**The one known case, and its fix.** `src/solve_level0.cu:126-141`, inside `k_level0_search`:

```cpp
for (int q = 0; q < 4; q++)
{
    if (cx[q] == tx && cy[q] == ty) alpha += cwt[q];
    else { const float* nv = v0 + ((size_t)cy[q] * w0 + cx[q]) * c0;      // <- the read
           for (int i = 0; i < c0; i++) beta[i] += cwt[q] * nv[i]; }
}
if (!(alpha > 0.0f)) continue;    // <- and only then is the site discarded
```

A site at pixel `(tx + dx, ty + dy)` with `|dx|, |dy| <= 1` has a bilinear tap pair spanning `tx - 1` to `tx + 1` in
each axis, so a tap can land on `tx +/- 2` -- **a texel of the same colour**, which another worker of this very pass is
writing. The value is then thrown away, because such a site has no tap on `(tx, ty)` and `alpha` is zero. On a GPU
that is a stale read of an irrelevant number; under the C++ memory model it is a data race, and `-fsanitize=thread`
will say so.

**The fix: hoist the `alpha` test above the accumulation.** Compute `alpha` over the four taps first, `continue` if it
is not positive, and only then walk the four taps again for `beta`. It changes nothing computed, and the argument is
two lines: `alpha`'s additions still happen in `q` order and are identical; `beta`'s additions still happen in `q`
order and are identical for every site that survives, and for a site that does not survive `beta` was never read. And
after the hoist, **every surviving site has `(tx, ty)` among its four taps**, so its other three taps are at distance 1
and are of a different colour -- the race is gone, not merely narrowed.

**That fix lives in `src/cpu/cpu_solve_level0.cpp` and nowhere else.** `solve_level0.cu` is not edited: decision 1.
The CPU file carries a comment naming the kernel, the line range and the reason, so a reader who diffs the two sees
why they differ and that they do not disagree. **Done in stage C6.** With the hoist a ThreadSanitizer build runs
threaded CPU encodes of `tiny.png --l0 palette` and the m1_m4 material in palette mode (whose 512x512 base puts 16
chunks in every colour pass, so chunk boundaries are thread boundaries) with zero reports.

**And what the sanitizer could NOT show, measured rather than assumed.** With the kernel's own order restored in a
scratch build (the `beta` loop ahead of the `alpha` test), ThreadSanitizer did **not** report the race - on the
material at `-O1` and at `-O0`, at `-j4` and at `-j16` - and it did not report a race planted on purpose in the same
function either (every texel writing its own `v0`, or a static array, and reading the texel two rows down, so the last
row of each chunk reads what the next chunk writes). The same build reported a racy counter incremented at the start or
at the end of every chunk at once, and a standalone program running the planted pattern through the same pool is
reported every time. The difference is distance: at a chunk boundary the two accesses are a whole chunk of work apart
(the next chunk writes its first row first, this chunk reads it last), where the counter's are adjacent. So the
sanitizer is a backstop that finds races whose accesses are close in time, and **the audit table above is the proof**
for the four-colour kernels - which is why it argues every kernel on its own rather than pointing at the sanitizer.

**The permanent gate arm.** A **ThreadSanitizer** build on Linux (clang or gcc `-fsanitize=thread -g -O1`), a threaded
CPU encode of `tests/tiny.png` and one small material at `-j4`, with `TSAN_OPTIONS=halt_on_error=1`. **Zero reports
required**; any report fails the gate. Three notes:

* TSan needs the whole program instrumented, and nvcc's device objects are not. So the sanitizer build is the
  **CPU-only configure** -- exactly the build decision 8 makes possible and the same one that proves the dispatch
  complete (1.7). One configure serves two gates;
* TSan costs roughly 8x in time and a great deal of address space, which is why the arm uses `tiny.png` and `-j4`
  rather than a material and `-j32`. Its job is to find a race, and a race in a static partition shows up on the
  smallest input that reaches the kernel;
* MSVC has no ThreadSanitizer, so the arm is Linux-only and is recorded as skipped elsewhere, never as run -- the
  rule the Vulkan arms already follow.

**Built in stage D** as `tsan_check`, the gate's last arm: the CPU-only configure with the C1 recipe (clang++ where it
is on the PATH, else g++) in `out/tsan_build_posix`, persisting between gates; it requires `__tsan_init` in both
binaries, so that zero reports cannot come from an uninstrumented build, and then runs `nntc_cpu_check` (every kernel
at -j4, -j16 and -j0, the pool at up to 64 threads) and `--backend cpu -j 4` encodes of `tiny.png` at the default
layout and at `--l0 palette`, with `halt_on_error=1`: zero reports, 124 s, 2.1 s and 1.6 s under WSL. **Not the
material the paragraph above names**, deliberately: the finding just above - the reinstated race inside a long
material encode, which the sanitizer did not report - says a material buys TSan's eightfold cost and not the race,
so the arm keeps to the inputs where accesses are close in time, and its docstring says it is the backstop and this
section's audit table is the proof.

**And the finding this gate could produce.** If the audit or TSan ever turns up a case where the CUDA code genuinely
reads a value another thread is writing **and keeps it**, that is a finding about the CUDA code and is taken to the
owner. It cannot be dismissed as a GPU idiom: `docs/DESIGN.md` section 6 and the `determinism_check` arm say the CUDA
encoder is deterministic, and a kept read of a racing write is not compatible with that. The port is, among other
things, a second opinion on that claim.

---

## 4. The per-kernel comparison harness

This is the section the first draft did not have, and the review's first finding is why. The draft staged the port so
that stages 2 to 5 each ended with a test -- and each of those tests needed a full encode, which needs every kernel,
which is stage 7. Stage 2's objective test needs `init_level0` and `init_level1` from stage 3; stage 3's "the banner's
init report matches" needs a stop-after-init mode that does not exist; stage 4's ridge tally needs blocks (b) and (c),
and even the post-loop path calls `sweep_level1_all` and `solve_level0_all`; stage 5's arms come from `--check`, which
is a full encode. So the draft's real shape was: write 3200 lines, then debug them all at once.

**The fix is an instrument, not a re-ordering.** Build the comparison first, and verify every kernel the day it is
written.

### 4.1 What it is and how it is invoked

**`--backend check`**, a fourth value of the flag, available only in a build carrying both backends.

It creates **both** device models from the same `Model` and the same source chain, and then, for every one of the 97
dispatched calls:

1. **runs the CUDA side first.** The run's trajectory is CUDA's, always, so one divergence does not cascade into
   fifty and the report stays readable;
2. **mirrors the CUDA model's pre-call state into the CPU model**, through the fourteen transfer functions, so the
   two implementations receive **identical inputs by construction** rather than by having stayed in step;
3. **runs the CPU side** on that state;
4. **reads both outputs back out** and diffs them, and prints one line: the seam function, the plane, the largest
   absolute and relative difference, the count of differing entries, the bar, and `ok` or `FAIL`.

Two modifiers: `--kcheck-only NAME` restricts the printed diff to one seam function while still running the whole
encode, which is what an implementer working on one kernel uses; `--kcheck-stop` stops at the first failure with the
call named. The mode is for small assets and is never timed: the mirroring is a download and an upload of the model's
live state per call, which on `tests/tiny.png` is nothing and on a 2048x2048 material would be the run.

### 4.2 What it compares, and to what bar

Two groups, and the line between them is 1.5's: **fp32 FMA contraction in the feature arithmetic is the one difference
that cannot be removed.** Everything on the other side of that line is held to the bit.

**Exact, bit for bit. Any difference at all is a FAIL.**

* every **integer** output: `k0`, `k1`, the BC endpoints and selectors, the moved counters, the ridge tally's four
  rungs, `Level1Report::iterations`, `Level0Report::joint` and `states`, `stencil_monomials` for every `c0`;
* every **fp64 quantity whose inputs are fp64 and whose reduction shape is replayed** (3.2): given an identical
  uploaded stencil and gradient, the whole conjugate-gradient chain -- `k_precondition`, `k_apply_precond`, `k_matvec`,
  `k_negate`, `k_axpy`, `k_xpby`, `k_add_correction`, `k_dot_partial` / `k_reduce_sums`, `k_diag_partial` /
  `k_reduce_diag` including `SC_LAMBDA`, `k_range_partial`, `k_movement_partial`, `k_cg_alpha`, `k_cg_beta` -- and
  `k_quant_sweep`, `gauss_jordan`, `jacobi_eigen`, `fit_range`, the ridge ladder given identical normal equations,
  `k_ls_reduce`, and `bc0_value`;
* every **transfer**, trivially;
* `k_quantise_level1`, `k_init_level0`, `k_init_level0_cont`, `k_seed_channel`, `k_init_level1_box`, `k_block_means`,
  `k_proj_hist`, `k_proj_peak` and `k_cov_partial`, none of which evaluate the decoder.

**A named tolerance, because fp32 contraction is live.** The kernels that build features or sample bilinearly in
fp32: `k_ls_accumulate` (the normal equations), `k_objective` (`E` and its two companions), `k_decode`, `k_residual`,
`k_stencil_assemble`, `k_stencil_assemble0`, `k_level0_search`'s quadratic, `k_bc_refine`'s quadratic. The proposed
starting bars, **to be replaced by a measurement in stage C1 and only then written into the harness**: `1e-5` relative
per entry on an assembled quantity, `1e-6` relative on a summed scalar. They are a **ceiling, not a target**: what the
harness is actually looking for is a difference of order one, which is a transcription bug, and the bar exists only so
that the contraction noise does not print as a failure every line.

**The one thing the bar cannot catch, and what covers it.** A kernel that takes a *different discrete decision* --
a level-0 argmin, a BC accept, a rung of the ladder -- produces a difference far larger than any tolerance, so it
fails loudly. That is the design working. A kernel that takes the same decision for the wrong reason is caught by the
within-backend structural arms instead (7.2): `E` monotone, the finite-difference gradient, the dense direct solve.

### 4.3 What it can and cannot see, stated plainly

Its granularity is the **seam function plus the probes the seam already carries**. That is one kernel for most of the
port -- `init_level0`, `quantise_level1`, `snap_level0_on_grid`, `assemble_level0_for_refine`, `decode_plane`,
`objective_eval` are each one launch -- and it is four colour passes for `solve_level0_all`.

It is **coarser for block (b)**, where `solve_level1_all` is an assembly, a preconditioner and up to 200 CG
iterations behind one call. Nothing on the host can download the assembled `stencil` or `grad` today, and adding a
CUDA-side download would be a new definition in `solve_level1.cu`, which decision 1 forbids. So block (b) is checked
through what the seam already exposes, which turns out to be a great deal:

* `Level1Report::min_diag` and `max_diag` are reductions over the assembled stencil's diagonal, and `lambda` is
  `--ridge` times its mean. Three independent numbers out of the assembly, per plane, per round, already printed;
* `Level1Report::iterations` and `residual` are the CG's own trajectory, and `iterations` is an integer held exact;
* `level1_delta` downloads the correction vector `cg_x` itself, which is the solve's entire output;
* `level1_range` and the movement probe's `moved`, `delta2`, `plane2` close over the plane after the correction;
* and `solve_level1_dense_host` re-derives the whole system independently **within each backend**, so a wrong
  assembly fails inside the CPU backend on its own terms before any cross-backend comparison happens.

That is stated here rather than discovered later: block (b) is the one place the harness is a seam check and not a
kernel check, and it is compensated for by five probes and an independent twin.

### 4.4 Why it is also the proof that section 1.8's copies agree

Decision 4 duplicates about 1160 lines, and section 1.8 promises that the duplication is guarded by a number rather
than by anyone keeping the copies in step. This is that number. `gauss_jordan`, `jacobi_eigen`, `fit_range`, the ridge
ladder, the form and monomial matrices, `bc0_value` and the sampling rule are all on the **exact** side of 4.2's line,
so a copy that differs from its original by one character in one coefficient fails the first time the harness runs,
naming the seam function it is under. `stencil_monomials` gets its own start-up assertion over every `c0` and `c1`
(1.7). The two brute-force twins are checked by each backend's own `--check` arms, as they are today.

### 4.5 Where it sits in the plan

It is written in **stage C1**, immediately after `CpuModel` and the fourteen transfers, and before any kernel. From
stage C2 onward, **every stage's "done when" is a harness run**, and the per-stage acceptance in section 6 is stated
that way. It costs 3 to 4 days that the first draft did not budget, and it is expected to return them with interest in
stages C4 and C6, where a transcription error found the day it is made is minutes and the same error found at the end
of stage C7 is days.

---

## 5. The kernel-by-kernel port table

Sizes are the CPU implementation, including its comments, in the tree's commenting style. "Traps" lists what a
transcription gets wrong.

### 5.1 `init.cu` -- ten kernels, about 600 lines

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_init_level0` | per texel, palette snap by linear scan over at most 16 entries | texel | the scan's tie rule is strict `<`, so the **lower index wins**; keep it | 40 |
| `k_init_level0_cont` | per texel, no snap | texel | none; `k0` is separately zeroed | 20 |
| `k_init_level1_box` | per level-1 texel, mean of its footprint | level-1 texel | the footprint is computed in `long long` with a degenerate guard; the accumulator is **float, not double** -- do not "improve" it | 45 |
| `k_quantise_level1` | per value | value | in and out **alias** the same buffer at the level-0 call sites; safe only because one worker reads and writes one element (3.4). Preserve that or copy | 25 |
| `k_block_means` | per level-1 texel, all `nout` source channels | level-1 texel | float accumulator again | 40 |
| `k_cov_partial` | the 2-D grid `(SITE_BLOCKS, entries)` flattened; each `(block, entry)` is one chunk's inner sum | block index, per entry | up to 342 entries x 512 blocks. **The CPU should iterate entries in the outer loop and reuse one 256-slot scratch**, replaying the tree per 3.2. Second stage is already on the host | 90 |
| `k_residual` | per base pixel, the decoder's own **fp32** arithmetic | pixel | must use `cpu_sample.h`, not the objective's double twin: this is deliberately the decoder's rounding | 45 |
| `k_proj_peak` | fixed grid, tree of `max`; the peak is max'd on the host **over every plane of the chain** | block index | max is order-independent, so the tree is cosmetic -- replay it anyway for uniformity | 40 |
| `k_proj_hist` | fixed grid, 4096-bin histogram | block index | per-chunk private histogram, folded in chunk order **into one histogram that spans the whole plane loop** (`init.cu:1277-1290` memsets once and accumulates every plane before reading). Exactly equals CUDA's integer atomic | 45 |
| `k_seed_channel` | per texel, clamp then optional palette snap | texel | `lroundf` is round-half-away-from-zero, **not** `std::round`-to-even; and not `std::lrint`, which follows the rounding mode | 35 |
| the drivers | `init_level0`, `init_level1` (box and pca), `init0_residual_channel` | -- | the pca body and the residual seed's host parts -- the covariance fold, `jacobi_eigen`, the `std::sort` of the eigen order, the eigenvector sign fix, the per-texture 3x3 argmax, the percentile scan over the chain-wide histogram -- are copies (1.8). `INIT0_PERCENTILE` and the "divisor is not the peak in the palette mode" rule are part of the copy | 190 |

### 5.2 `solve_decoder.cu` -- block (a), about 360 lines

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_ls_accumulate` | 512 chunks, each staging 256 sites then summing up to 775 entries over the tile | block index | the **256-site chunk boundary is the summation order** and is load-bearing; `LS_STRIDE` becomes 256; a site past the end contributes a row of zeros rather than being skipped | 130 |
| `k_ls_reduce` | serial, entry by entry, planes in index order then blocks in index order, each plane scaled by its `omega` | none | keep the plane-then-block nesting | 30 |
| the ladder | **copied** (1.8): `RIDGE_LADDER`, `LS_ACCEPT_RISE`, the trace, the normaliser, `qform`, the four rungs, the float round trip, the acceptance bar `(q_cand - q_prev) * inv_norm + err_prev + err_cand <= 1e-12`, the constant-fit fallback and the tally | -- | it is a copy and it is on the **exact** side of 4.2's line given identical normal equations, so the harness proves it agrees. `g_ridge_tally` is the CPU backend's own static, reported through `be::decoder_ridge_tally` | 160 |
| `gauss_jordan`, `tri_index` | copied | -- | the pivot rule and the singularity return are the copy's whole content | 40 |

### 5.3 `solve_level0.cu` -- block (c), about 260 lines

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_level0_search` | per texel of one colour: build the exact quadratic over the texel's at most `9K + 1` sites, then argmin | colour texels, four sequential passes | five traps. (1) the colours must stay sequential and the writes stay in place. (2) `joint = (total_bits <= 8)` is computed **once per call of the driver** (`solve_level0.cu:288`), from the layout, and is passed to every plane and every colour: it is **not** per plane and **not** data-dependent. Both branches are ported -- exhaustive enumeration over at most 256 states, or coordinate sweeps with an early break. (3) ties keep the current state (`e < best_e`, strict). (4) the **tap-merge rule**: a texel that appears twice among the four taps gets the **sum** of both weights into `alpha`, and a site with `alpha == 0` is skipped. (5) **the kernel shadows its own loop variables**: `for (int dy...) for (int dx...)` at lines 96-97, then `float dx, dy;` at 103 for the subtexel offset. The CPU copy must not silently use the wrong pair; rename them and say why in a comment | 200 |
| the race fix | the `alpha` test hoisted above the `beta` accumulation, in the CPU file only | -- | 3.4. Provably identical; comment it with the kernel's line range | 10 |
| the driver | four `run` calls per plane, per-chunk moved counters folded in chunk order | -- | CUDA's `atomicAdd` on `unsigned` is exact; the fold matches it | 50 |

### 5.4 `solve_level1.cu` -- blocks (b) and (c'), about 900 lines

This is the largest file and the one to schedule the most time for.

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_stencil_assemble` | per level-1 texel, gathering every site whose footprint can reach it; 75 doubles of monomial accumulator, expanded once at the end | texel | **gather, not scatter** (section 9). `stencil_slot` returns -1 for the four directions the *other* texel of a pair owns. The four-tap dedup-merge loop is the same rule as block (c)'s and must give the same weights. Zero monomials are skipped by `if (s == 0.0) continue`, which is an optimisation and not a rule -- but keeping it keeps the arithmetic identical | 200 |
| `k_stencil_assemble0` | the same over level-0 texels, monomials in the level-1 sample | texel | one structural difference: the centre site is a **single tap of weight 1.0**, because level 0 is at the decoded extent | 190 |
| `k_dot_partial` + `k_reduce_sums` | 256 chunks, tree, then a serial fold | block index | three of these per CG iteration; they are the hot reduction, and 3.2's size threshold is aimed at them | 60 |
| `k_diag_partial` + `k_reduce_diag` | 256 chunks, three trees at once (sum, min, max), then the fold that also writes `SC_LAMBDA = ridge * mean(diag H)` | block index | lambda never leaves the backend, exactly as on the device | 60 |
| `invert_block` + `k_precondition` | per texel, 4x4 Gauss-Jordan in double, **stored as float** | texel | the float store is deliberate: the preconditioner affects only the convergence rate. A singular block falls back to the identity. `invert_block` is `__device__` and the CPU writes its own four-line twin | 60 |
| `k_apply_precond`, `k_matvec`, `k_negate`, `k_axpy`, `k_xpby`, `k_add_correction` | one per element | element | `k_matvec` reads the forward neighbour's block from its own row and the backward neighbour's **transposed** from that neighbour's row, with plain in-bounds tests and **no clamp** -- a different rule from the sampler's clamp-to-edge. Do not unify them | 120 |
| `k_cg_alpha`, `k_cg_beta` | plain serial scalars | none | `alpha = 0` on non-positive curvature; `beta = 0` on a zero `rz`; `k_cg_beta` also writes `SC_RESID` | 30 |
| `k_range_partial` + `k_reduce_range` | 256 chunks, two trees per channel inside a channel loop, so **`2 * c1` partial rows** | block index | the shared arrays are `slo[CG_THREADS]` and `shi[CG_THREADS]`, reused per channel, not one array per channel | 55 |
| `k_movement_partial` | 256 chunks, three trees | block index | the count is accumulated **as a double**; it runs *after* the correction is added, so it sees the new plane | 50 |
| `k_quant_sweep` | per texel of one colour: the neighbour term once, then the `C1` channels **sequentially** | colour texels, four sequential passes per sweep | the intra-texel channel loop is Gauss-Seidel: channel `q` reads the `delta` that channel `j < q` just wrote, on the worker's own texel. The grid is not uniform -- `level1_value` and `level1_index` are `model.h`'s, shared not copied, and tie to the lower index. `k1`, `v1` and `delta` are written in place | 120 |
| the CG driver | setup, then the iteration, then the finish | -- | **keep `CG_PROBE = 4`** (`solve_level1.cu:99`). On a CPU there is no reason to test the residual only every fourth iteration -- it is computed every iteration -- and testing it every iteration would be faster and no less correct. Decision 6 forbids it: a plane that overshoots by up to three iterations on the device must overshoot by the same three here, because the correction it lands on is what the asset carries. Keep the 200-iteration cap and the 1e-10 relative-residual bar too | 110 |
| the sweep driver | `sweeps` x 4 `run` calls per plane | -- | -- | 40 |
| `solve_level1_dense_host` | **copied** (1.8), then two downloads deleted | -- | it re-derives its own tap indexing on purpose (it is the independent twin) -- **do not** make it call `cpu_sample.h` while copying it | 20 |

### 5.5 `objective.cu` -- about 250 lines

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_objective` | 512 chunks, three sums at once, tree replayed. Its device scratch is `__shared__ double sh[3 * OBJ_THREADS]` = **6 KB, three rows**, not one | block index | **everything is double**, including a third independent copy of the bilinear rule (`obj_sample` / `obj_clamp`) distinct from `cpu_sample.h`'s float one and from `twin_sample`. All three must agree and none may be merged. `E` is the loop's stopping criterion, so this kernel's summation order decides the asset and not only the log | 140 |
| `k_obj_reduce` | serial, three rows of 512 | none | trivial | 15 |
| `k_decode` | per pixel, fp32, `saturate` then `lroundf(acc * 255)` | pixel | `lroundf` again | 40 |
| the driver | per plane, then `normalise`, copied | -- | `objective_total_ms` and `objective_passes` are the report's own numbers and must keep counting; they are the CPU backend's own statics, reached through `be::` | 30 |
| `objective_check_host` | **copied** (1.8), already a serial host brute force | -- | ports by deleting three downloads. **Its accumulation order is scan order and is deliberately different** from the kernel's partitioned order -- that difference is the check, and it is the reason the copy is still a witness | 25 |

### 5.6 `refine_bc.cu` -- about 420 lines, and the hardest single kernel

| kernel | CPU shape | axis | traps | lines |
|---|---|---|---|---|
| `k_bc_decode` | per texel, endpoints and selectors to value, index and `delta` | texel | `bc0_value`'s anti-contraction spelling (1.5); `(int)(byte + 0.5f)`; the `delta` buffer **aliases the level-0 CG solution vector**, which the CPU backend names explicitly | 60 |
| `k_bc_refine` | **one 4x4 block per chunk element, serial within the block**, four sequential colours per pass | blocks of one colour | see below | 300 |
| the drivers | `bc_pack_prepare`, `bc_refine`, `bc_blocks_from_device`, `bc_state_save`, `bc_state_restore` | -- | the last three are already almost pure host code | 60 |

`k_bc_refine` deserves its own paragraph, and the first draft got one thing wrong about it.

On the device it is 64 threads per 4x4 block (`REFINE_THREADS = 16 * MAX_CHANNELS`), with a dynamic shared-memory
quadratic of up to 35 KB, and **every actual decision taken by thread 0 alone** with two `__syncthreads()` per
Gauss-Seidel step while the other 63 threads wait at the barrier. That structure exists to get the assembly and the
reductions parallel; the decisions were always serial. So the right CPU shape is a single-threaded routine per 4x4
block, parallelised across blocks, which deletes every barrier and every shared-memory allocation.

**What it does not delete is `refine_reduce`.** The draft said the CPU form "deletes ... `refine_reduce` entirely",
and that is wrong. `refine_bc.cu:136-147` is a 64-slot halving tree, called **six times per channel** in the endpoint
stage (lines 313-317 for `bp`, `bq`, `pap`, `paq`, `qaq`, and 388 for the selector stage's `exact`) and once per
selector step. Its sums are not a log number: they gate the relative determinant guard `det > 1e-12 * pap * qaq`
(line 335), the nine-candidate argmin (352) and the strict-decrease acceptance `if (exact < 0.0)` (388). **A serial
left fold over 64 doubles is not bitwise that tree**, and the three quantities it feeds are all comparisons against
near-zero thresholds where the last bit decides. **The CPU routine replays the 64-slot tree**, per 3.2's rule, inside
the single-threaded block routine: fill a `double red[64]` in thread order (`tid < 16` contributes, the rest zero) and
run the halving loop verbatim. It costs 63 additions per call and it is the difference between this kernel being on
the exact side of 4.2's line and being a permanent source of unexplained decisions.

Four more things it must keep. The **four-colour partition** is not an optimisation: a block reads its neighbours'
`delta` while other blocks of the same launch write theirs, and that is safe only because same-colour blocks are two
apart (3.4). The **endpoint stage's nine-candidate search** (`e0, e1` in `{-1, 0, 1}`) with its `continue` on
`a0 <= a1i` and the determinant guard falling back to the current endpoints. The **strict-decrease acceptance** at
both the endpoint and the selector stage. And the **edge rule**: a texel outside the plane keeps a zero row and its
selector never moves.

### 5.7 `cpu_model` and the transfers -- about 350 lines

`CpuModel` is `DeviceModel` with `float*` and `double*` where the device has the same, allocated through 1.11's
`cpu_alloc`, minus the streams, the events, the pinned host buffers and the fan-out and fan-in machinery. `PlaneWork`
and `plane_work` are copied from `device.cuh` (1.8) with `plane_work`'s `stencil_monomials` call resolving to the CPU
copy. The 14 transfer functions become `std::copy`. `device_memory` adds the same buffers up and reports the same
number under the same name (1.10), and `bytes_for` computes it before anything is allocated so the banner can say it.

---

## 6. The stage plan

Four steps in the owner's fixed order: (A) the dispatcher, (B) regress the CUDA encoder, (C) the backend, (D) compare.
Each stage ends at something testable, with the test named. Day estimates are working days for one implementer who
already knows this tree.

### Step A

**Stage A -- the dispatcher, the flags, the build.** Section 2, in five commits: `backend.h` / `backend.cpp` and
`cuda_backend.h` / `cuda_backend.cpp`; the 97
call sites; the flags, the fallback and the banner; CMake; the contraction check. *Done when* section 2.4's four
instruments are all green: both gates on both toolchains, every asset byte-identical to `v1.1.28-woa-tested` over the
twenty-six command lines, the logs' numeric columns unchanged, and the fp-contraction sweep identical on MSVC x64,
MSVC ARM64, gcc and clang. Tag `v1.2.0-cpu-dispatch`. **3 to 4 days**, most of it the widened corpus rather than the
code: the code is 400 new lines and a prefix, and the linker checks the prefix.

**Done: `v1.2.0-cpu-dispatch`.** Thirty command lines, every written file byte-identical to the reference. Two
corrections to this plan came out of it, both recorded where they apply: the contraction check's sweep-and-diff
design could not detect contraction and was replaced by witnesses and a hash (2.4, item 4), and corpus line 18 needs
`--l0 palette` for `--bc0 0` to reach anything (2.4, item 2). MSVC ARM64 has not run the contraction check yet: there
is no Arm machine at the desk that built it.

### Step B

**Stage B -- regress the CUDA encoder, and arm the hash gate.** Not a refactor but a deliberate pause: run the full
gate Release and Debug on Windows and under WSL, re-encode the measured material sets of `docs/RESULTS.md` and diff
every report number against the tracked logs, and re-run the `--check` arms (**`--cuda-check` does not exist in this
tree** -- it belongs to another project; this tree has `--check` and nothing else). Then add `tests/kernel_hashes.txt`
and the `kernel_hashes` arm (2.5), and the CPU-only-configure arm that proves the dispatch complete (1.7). The point
is that step A is *signed off* as a separate event before anyone writes a line of CPU code, so that a later
disagreement between the backends can never be blamed on the dispatcher. **1 day.**

**Done: `v1.2.0b-cuda-signoff`.** Both gates green, Release and Debug. The tracked logs under `out/` re-run with the
current encoder: every one written by this tree's own numerics reproduces every non-timing number (`out/disclosure/`
both Windows logs, and the WSL ones on the WSL build; all of `out/s9b/` and `out/s9g/`); the rest predate a documented
change (`out/timing/` was written before the last free refit existed and matches up to it; `out/s9a/` predates the
refinement, the last refit and the v0.9b-v0.9g solver changes; `out/s7/` predates the deterministic reductions and
the fitted palette grid). The kernel-hash arm runs first and the CPU-only link runs last, on every gate.

### Step C

**Stage C1 -- the pool, `CpuModel`, the transfers, and the harness.** `pool.h` with its contract and a unit exercise
of rules 1 to 5; `cpu_constants.h` and `cpu_sample.h` (the copies); `cpu_model`; the 14 transfers; then
`src/backend_check.cpp` in full. Plus two measurements before any kernel is written: `Pool::run`'s cost (risk 11), and
**one full fixed-grid reduction at a deep plane's size** (3.2), which decides whether the size threshold is needed.
*Done when* `--backend check` runs a whole CUDA encode of `tests/tiny.png` with the CPU side mirrored at every call,
reports `ok` on all 14 transfers and on `stencil_monomials` for every layout, and the two measurements are recorded.
**3 to 4 days**, of which the harness is most.

**Done: `v1.2.1-c1-harness`.** `src/cpu/pool.h`, `cpu_constants.h`, `cpu_sample.h`, `cpu_model.{h,cpp}`,
`src/backend_check.{h,cpp}` and `tests/cpu_check.cpp` (the `nntc_cpu_check` target: the pool's contract and the
transfers at -j1 to -j64 and -j0, and `--bench`). `--backend check` on `tests/tiny.png` - bare, `--c0 1`, `--c0 3`,
`--c0 4 --c1 4`, `--l0 palette`, the rejected-outer-pass line, and `--check` with both level-0 modes - reports 0
mismatches and 0 flips, every transfer the encode calls `ok`, `stencil_monomials` `ok` for c0 1 to 4, `device_memory`
equal to the CUDA figure to the byte, and every kernel `not implemented`; the same on the WSL build. A check run's
assets are byte-identical to a plain CUDA run's, and both to `v1.2.0c-psnr-gate`'s (bare, `--l0 palette`, a
six-texture material, `--c0 4 --c1 4`, the rejection line). A deliberate one-ulp error in one CPU download is a
MISMATCH at the first call that reads it.

*The two measurements*, on the 32-thread development machine (Windows, MSVC, Release):

| `Pool::run`, per call, every worker woken | -j1 | -j4 | -j16 | -j32 |
|---|---|---|---|---|
| condition variable only (the first version) | 0.02 us | 1.02 us | 21.1 us | 67.8 us |
| spin-then-park, empty body | 0.00 us | 0.16 us | 1.10 us | 2.65 us |
| spin-then-park, tiny body (64 adds a chunk) | 0.02 us | 0.22 us | 1.10 us | 2.52 us |

The first version missed section 3.1's "microseconds" by an order of magnitude at -j32, so the fallback the section
names went in, inside `pool.h` and nowhere else: each worker watches its own generation counter (`std::atomic`,
acquire / release) for up to a millisecond of wall clock before it parks, and the caller watches one countdown the
same way. A budget counted in iterations (tens of microseconds) did not help at all - one late worker parks, its
wake-up takes tens of microseconds and the others' budgets run out meanwhile - so the budget is time. A pool idle for
more than a millisecond pays the wake-ups once on its next run.

| the CPU backend's reduction (a double sum, 4096-element chunks folded in chunk order) | chunks | serial loop | -j1 | -j4 | -j16 | -j32 |
|---|---|---|---|---|---|---|
| base: level 0 of 2048x2048, C0 2 (8.4 M values) | 2048 | 3003 us | 2935 us | 825 us | 256 us | 268 us |
| base: level 1 of 2048x2048, C1 4 (1 M values) | 256 | 388 us | 366 us | 94 us | 28 us | 16 us |
| deep: level 0 of the 16x16 mip, C0 2 | 1 | 0.16 us | 0.20 us | 0.20 us | 0.20 us | 0.20 us |
| deep: level 1 of the 8x8 mip, C1 4 | 1 | 0.00 us | 0.02 us | 0.02 us | 0.02 us | 0.02 us |

So the replacement for the CUDA-tree replay (section 0a) costs nothing measurable: at one thread it is the serial
loop's time, and a deep plane is one chunk, which wakes no worker at all because a run's participants are
min(threads, chunks). **Section 3.2's size threshold is not needed** and is not built.

*The tolerances* (section 0a point 3), in `src/backend_check.cpp` with their measurements beside them: transfers and
integers exact; an index one step off a FLIP, counted and reported and not failed unless more than 1 % of an array
(and more than 8 indices) flips; a per-entry value 1e-5 of its array's largest magnitude, fifty times the measured
known-good pair (the CUDA CG correction against the host dense direct solve, 1.8e-7 of the range on tiny.png, 1.0e-7
under `--l0 palette`); a summed scalar 1e-9 relative, four orders above the CUDA objective against its fp64 host twin
(at worst 7.1e-14 over every pass of the eight lines above, MSVC and gcc alike). The harness re-measures that last pair
on every run and prints it, so the scalar bar's margin is visible every time the harness is used.

*ThreadSanitizer*, zero reports, under WSL (Ubuntu 24.04, clang 18), a CPU-only configure - the command stage D makes
a gate arm of:

```
cmake -B build_tsan -S . -DCMAKE_CUDA_COMPILER=NOTFOUND -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=clang++ \
      "-DCMAKE_CXX_FLAGS=-fsanitize=thread -g -O1" -DCMAKE_EXE_LINKER_FLAGS=-fsanitize=thread
cmake --build build_tsan --target nntc_cpu_check nntc_encode -j
TSAN_OPTIONS=halt_on_error=1 ./build_tsan/nntc_cpu_check
TSAN_OPTIONS=halt_on_error=1 ./build_tsan/nntc_encode tests/tiny.png -o out/tsan --backend cpu -j 4
```

`nntc_cpu_check` runs the pool at eight thread counts, 20000 back-to-back runs, 60 runs after the pool has parked (half
of them with the caller parked too) and every transfer at -j4 and -j13; it was run six times, all clean. The encode
builds the CPU model and uploads the chain threaded before its first kernel refuses. A deliberately racy body through
the same pool is reported by the same build, so the sanitizer is live.

*Where the plan was wrong, and what was done instead.*

* **"All fourteen transfers" counts `reconstruct_levels`**, which is not a transfer and not dispatched: it is
  `export.cpp`'s host loop over `decode_plane`, a kernel (stage C2), and the harness sees it as its `decode_plane`
  calls. The dispatched transfers are the thirteen of section 1.3's row without it, plus `bc_state_save` and
  `bc_state_restore`, which section 1.3 files under the BC refinement but which are pure copies. All fifteen are
  implemented and checked; `upload_decoder_to_device` and `upload_grids_to_device` are reached only by the
  rejected-outer-pass line.
* **Two uploads cannot be compared on the CUDA side**: the seam has no download of the device's decoder or grids, and
  one would be an edit to a kernel file. Their CPU twins are held to the Model they were handed; the CUDA side is
  checked the first time the CPU objective (C2) is compared against the CUDA one.
* **The host Model is compared only for calls that take it mutably.** `main.cpp` downloads straight into `m.k0[i]`,
  so for a function taking `const Model&` the CUDA call changes a Model field under the harness's pre-call copy.
* **Rule 5 changed at the owner's direction**: -j1 does not run "with no pool at all" but through the same chunk walk
  as every other count, with no workers; there is no inline path to diverge.
* **Section 4.1's one printed line per call** would be thousands of lines an encode; the harness prints a line for
  every call that is not a plain `ok` (and for every call of one function under `--kcheck-only`), and one summary line
  per function at the end.
* **The CPU model allocates CUDA's buffer list as it stands**, the SITE_BLOCKS and REDUCE_BLOCKS partials included, so
  that `device_memory` is the same number on both backends (1.10). A later stage that sizes its own chunk partials
  instead changes the CPU figure deliberately, and the harness's `device_memory` comparison with it.
* **Step A committed four sources with LF endings** (`src/backend.h`, `src/cuda_backend.h`,
  `tests/fp_contract_check.{cpp,py}`); stage C1 converted them to CRLF in a commit of their own.

**Stage C2 -- the objective.** `objective_eval`, `k_obj_reduce`, `decode_plane`, the copied `normalise` and
`twin_sample`, and the copied `objective_check_host`. *Done when* the harness reports `E` and its two companions
within 4.2's scalar bar on every plane of `tiny.png` and of one material, `k_decode` exact, and the CPU
`objective_eval` agrees with the CPU `objective_check_host` to the same 1e-9 the gate holds the CUDA one to -- at
`-j1` and at `-j16` with identical bits. **2 to 3 days.**

**Done: `v1.2.2-c2-objective`.** `src/cpu/cpu_objective.cpp`: `k_objective` as chunks of `CPU_CHUNK` consecutive
sites summed in site order and folded in chunk order (no CUDA tree, section 0a), `k_obj_reduce` as that fold,
`k_decode` per pixel with `std::lround` on the float, the copied `obj_sample`, `normalise` and `twin_sample`, and
the copied `objective_check_host`. The harness also measures the CPU objective against the CPU twin on every pass
and fails the run past 1e-9. `nntc_cpu_check` gained a kernel arm: every ported kernel's every output on three
synthetic layouts (both level-0 modes, K 0, 2 and 4, a three-plane chain whose base spans several chunks and whose
deepest plane is one) logged as raw bytes at -j1, -j4, -j16 and -j0, and the logs held identical.

*Measured* (Windows, MSVC, Release, the CUDA kernels built with `-fmad=false`; `--backend check`, every pass of
every run):

| run | objective passes | `E` and its companions, every plane, CPU against CUDA | `k_decode` | CPU objective against the CPU twin | CUDA objective against the CUDA twin |
|---|---|---|---|---|---|
| `tiny.png` bare | 74 | 1.21e-14 | exact, 4 calls | 2.55e-14 | 2.49e-14 |
| `--c0 1` | 72 | 1.21e-14 | exact | 2.63e-14 | 2.88e-14 |
| `--c0 3` | 76 | 1.21e-14 | exact | 6.78e-14 | 6.40e-14 |
| `--c0 4 --c1 4` | 78 | 1.21e-14 | exact | 5.11e-14 | 5.17e-14 |
| `--l0 palette` | 69 | 1.21e-14 | exact | 6.72e-14 | 6.93e-14 |
| the rejected-outer-pass line | 73 | 1.21e-14 | exact | 2.55e-14 | 2.49e-14 |
| `--check` / `--l0 palette --check` | 235 / 230 | 1.21e-14; `objective_check_host` exact | exact | 2.55e-14 / 6.72e-14 | 2.49e-14 / 6.93e-14 |
| `m1_m4_source_material.json --c0 3 --c1 4` | 76 | 1.60e-14 | exact, 7 calls | 3.45e-13 | 3.44e-13 |
| the same two on the WSL build (gcc) | 74 / 76 | 1.21e-14 / 1.84e-14 | exact | 2.55e-14 / 4.95e-13 | 2.49e-14 / 4.96e-13 |

Every figure is a worst relative difference over every pass and every plane. So on identical inputs the CPU and
CUDA objectives differ by the summation order of the doubles and nothing else (about 1e-14, five orders under the
1e-9 summed-scalar bar), the CPU pair sits exactly where the CUDA pair does, and with `-fmad=false` the fp32
decode is **bit-identical**: not one byte one step apart in any run, where C1's bar still allows a flip.
`objective_check_host` is bit-identical too, being the same serial double arithmetic. `-j1` and `-j16` harness
runs of `tiny.png` and the material print identical summaries, and `nntc_cpu_check`'s kernel arm (39 outputs,
191,004 bytes) is identical at every thread count, with the objective 3.4e-14 from its twin. ThreadSanitizer (the
C1 recipe) runs `nntc_cpu_check` three times clean. A CUDA run's assets, and a check run's, are byte-identical to
`v1.2.1b-fmad-option`'s over the bare, `--l0 palette`, six-texture, `--c0 4 --c1 4` and rejected-outer-pass lines.

*Where the plan was wrong.* Section 5.1 and the stage brief say `lroundf` is "not `std::round`-to-even":
`std::round` rounds a half away from zero too, exactly as `lroundf` does. The real trap is `std::lrint` /
`std::nearbyint`, which follow the rounding mode (to-even by default); the port calls `std::lround` on the float,
which is `lroundf` by definition. And section 5.5's "tree replayed" for `k_objective` is superseded by section 0a.

**Stage C3 -- the initialisations.** `init_level0` both modes, `init_level1` box and pca, `init0_residual_channel`
with both scopes and all three `--init0` policies, and the five grid entry points; the copied `fit_range` and
`jacobi_eigen`. *Done when* the harness reports every one of the ten kernels within its bar, the pca principal values
and the residual seed's divisor, eigenvector row and per-channel variance share exact against CUDA's, and the
chain-wide histogram's percentile identical. **3 to 4 days.**

**Done: `v1.2.3-c3-init`.** `src/cpu/cpu_init.cpp`: the ten kernels (`k_init_level0`, `k_init_level0_cont`,
`k_init_level1_box`, `k_quantise_level1` through `model.h`'s `level1_index` / `level1_value`, `k_block_means`,
`k_cov_partial`, `k_residual`, `k_proj_peak`, `k_proj_hist`, `k_seed_channel`), their drivers - `init_level0` both
modes, `init_level1` box and pca, `init0_residual_channel` with both scopes and all three policies, and the five
grid entry points - and the copied `fit_range` and `jacobi_eigen`. The float accumulators stay float, the
palette walk keeps its strict `<`, `k_seed_channel` rounds with `std::lround` on the float, the histogram spans
the chain. The covariance is summed in `CPU_CHUNK` texel chunks folded in chunk order, with the plane weight
applied per partial as the CUDA host applies it per block. The harness gained two things: the percentile's
histogram bin, recovered from the report's `peak` and `divisor` and held exact, and, under `--kcheck-only`, one
line per toleranced quantity of each call. `nntc_cpu_check`'s kernel arm now runs every init kernel too, on five
synthetic layouts (both level-0 modes, both scopes, `--init0 residual` and `texture`, one to three textures): 1297
outputs, 67.6 MB, identical at -j1, -j4, -j16 and -j0.

*Measured*, `--backend check`, 23 command lines on Windows and the same 23 on the WSL build: `tests/tiny.png` bare,
`--c0 1`, `--c0 3`, `--c0 4 --c1 4`, `--l0 palette`, the rejected-outer-pass line, `--check` in both level-0
modes, `--init box`, `--q1-start 0` (which reaches `quantise_level1`, `quantise_level0` and `snap_level0_on_grid`),
`--init0 luma` / `texture` / `residual` at both scopes in both level-0 modes, and `m1_m4_source_material.json`
five ways (`--c0 3 --c1 4` bc8 and palette, `--init0 texture`, `--init0-scope chain`, and palette `--init0 texture
--init0-scope chain`). **0 mismatches and 0 flips on all 46 runs.** Per quantity, worst relative difference
CPU against CUDA over every call, on Windows:

| quantity | worst | note |
|---|---|---|
| `init_level0`, `init_level1` (box and pca), `quantise_level1`, `freeze_level1_grid`, `quantise_level0`, `freeze_level0_grid`, `snap_level0_on_grid`: every value, index, grid end and host index | **0** | identical bits |
| the pca principal values, their peaks and the trace | **0** | exact: the block means of 8-bit sources are short floats whose products and sums are exact in double, so the summation order cannot show |
| the residual seed's eigenvalue | 4.5e-14 | fp64 summation order of the covariance (the residual floats have full mantissas, so the order shows); the summed-scalar bar is 1e-9 |
| its trace, and the share | 3.5e-14, 1.7e-14 | the same cause |
| its direction (the eigenvector row), per entry | 3.7e-14 | the same cause, through Jacobi |
| its peak and divisor | 7.7e-15 | the maximum and the histogram are exact given the projections; the projections carry the direction's difference |
| the percentile's histogram bin | **identical** | on every palette call |
| the seeded `v0`, per entry | 2.3e-13 | a value near zero one float rounding apart; every seeded palette index identical. On WSL the seed's worst over all its quantities is 4.9e-13 |
| the objective, `E` and companions | 2.2e-14 | as stage C2; the CPU objective against its twin 6.1e-13 at worst (5.9e-13 on WSL) |

So the principal values are exact, and the residual seed's divisor, eigenvector row and variance share are not,
and cannot be without replaying CUDA's reduction tree, which section 0a retired: they sit at 1e-14, the fp64
summation-order floor, under the stated bars (1e-9 for a summed scalar, the per-entry bar for the direction).
`-j1` and `-j16` harness runs of five of the lines print identical summaries.
A deliberate tie-rule error (`<=` for `<` in the palette walk) is a MISMATCH on the first `init_level0` call
(`M0 k0: 8192 of 8192 indices one step apart`), so the flip ceiling catches a tie-break slip.

*The per-entry bar, re-measured and tightened.* With the CUDA kernels at `-fmad=false`, every per-entry output
of stages C2 and C3 is within 2.3e-13 of its array's scale (4.9e-13 at worst on gcc), where C1's bar is 1e-5. The bar
cannot go to 1e-12: a double one rounding apart may round to the neighbouring float, up to 1.2e-7 of the scale
at its top binade, and a grid end one float apart moves every value snapped onto it by as much. So the harness
now has two per-entry bars: `BAR_ENTRY_DIRECT` = **1e-6** (eight such roundings) for the DIRECT kernels - the
objective, the decode, the inits and the snaps, whose outputs follow from their inputs with no solve between -
and `BAR_ENTRY` = 1e-5, unchanged, for everything downstream of a solve, because two correct conjugate-gradient
runs stopping at the same 1e-10 residual land as far apart as C1's measured CG-against-dense pair (1.8e-7), not a
rounding apart; stage C5 should re-measure it. The summed-scalar bar stays 1e-9: it is the bar the gate holds the
CUDA objective to its twin at, the CPU pair is held to the same number by stage C2's acceptance, and the stage
C5 scalars (`|delta|^2` after a CG) are not yet measured.

ThreadSanitizer (the C1 recipe): `nntc_cpu_check` three times, and `--backend cpu -j 4` encodes of `tiny.png`,
`tiny.png --l0 palette` and the m1_m4 material, each of which now runs `init_level0`, `init_level1` (pca),
the grid fit and an objective pass threaded before it refuses at `solve_decoder`: zero reports.

*Where the plan was wrong, or did not say.* Section 5.1's `k_cov_partial` shape ("reuse one 256-slot scratch,
replaying the tree") and `k_proj_peak`'s "replay it anyway" are superseded by section 0a; the CPU forms are a
row of partials per `CPU_CHUNK` and a chunk maximum. `k_proj_hist`'s private-histogram-per-chunk form needed a
larger chunk than `CPU_CHUNK` to keep its scratch bounded (`CPU_HIST_CHUNK`, above). The "residual seed's divisor,
eigenvector row and per-channel variance share exact" of this stage's acceptance holds only for the pca init;
for the seed it is the stated 1e-14-level bar, for the reason above.

**Stage C4 -- block (a).** `k_ls_accumulate`'s staged form, `k_ls_reduce`, and the copied ladder and `gauss_jordan`.
*Done when* the harness reports the normal equations within the assembled bar and **the ladder's rung choice and
tally exact** on `tiny.png` and on the one-white-pixel dot, and the decoder weights agree to the last few bits of
float. **2 to 3 days**, the extra day being the ladder copy that the draft's extraction used to give free.

**Done: `v1.2.4-c4-decoder`.** `src/cpu/cpu_solve_decoder.cpp`: `k_ls_accumulate` in its own panelled shape - chunk
`b` of `SITE_BLOCKS` walks the tiles `b`, `b + 512`, ... of 256 sites exactly as CUDA block `b` does, stages each
tile's features, its constant one and its targets, sums every normal-equation entry over the tile as one serial
double dot product and adds the tile sum into its own accumulator, which lands in column `b` of `ls_partial`; a site
past the plane's end is still a row of zeros - then `k_ls_reduce` as the serial second stage, and the host side
copied as it stands: `tri_index`, `gauss_jordan`, the trace and the normaliser, `qform` and its rounding bound, the
three ridges and the refusal, the float round trip before judging, the acceptance bar, the tally (the CPU backend's
own static) and the constant-fit fallback. This is not the CUDA tree section 0a retired: the tile IS the kernel's
summation order, it is the cheapest CPU form anyway, and it makes the accumulation a plain serial double sum in the
same order on both sides. The harness gained three things for this block: a probe
(`src/backend_check_probe.cu`, compiled by nvcc because it reads `DeviceModel`, launching nothing and writing
nothing) that downloads the CUDA arm's normal equations after the call, held against the CPU arm's to a new
`BAR_ASSEMBLED` (the direct kernels' 1e-6 of the array's scale); the rung each call took, read from both arms'
tallies before and after it and held exact - a disagreement after normal equations that agreed is counted as a FLIP,
not failed, and the two decoders are then not compared, being the far sides of the ladder's decision; and the tally
compared as a count with that many flips of slack. `nntc_cpu_check`'s kernel arm runs block (a) twice per layout,
once with no decoder to fall back on and once from the one it left, and logs the normal equations, the decoder and
each call's rung: 1377 outputs, identical at -j1, -j4, -j16 and -j0.

*Measured*, `--backend check`, Windows, MSVC, the CUDA kernels at `-fmad=false`:

| run | block (a) calls | normal equations | rung per call and tally | decoder weights and bias |
|---|---|---|---|---|
| `tests/tiny.png` bare / `--c0 1` / `--c0 3` / `--c0 4 --c1 4` | 28 / 27 / 29 / 30 | **identical bits** | identical; all standard | **identical bits** |
| `--l0 palette` / the rejected-outer-pass line / `--q1-start 0` | 26 / 27 / 28 | identical bits | identical; all standard | identical bits |
| `--check` in both level-0 modes, and the gate's two `--check` lines (`--k 4`, `--k 0 --mips 0`) | 28 / 26 / 26 / 26 | identical bits | identical; all standard | identical bits |
| the one-white-pixel dot, 100x36 / 64x64 / 128x64 (the gate's three extents) | 28 each | identical bits | identical on every call; the tallies 10 / 14 / 12 standard and **18 / 14 / 16 reduced**, no rung flip | identical bits |
| `m1_m4_source_material.json --c0 3 --c1 4`, bc8 and palette | 29 / 28 | identical bits | identical; all standard | identical bits |
| `game2.png` of the development corpus, 1024x1024 | 28 | identical bits | identical; all standard | identical bits |

0 mismatches and 0 flips on all 17 runs. With the CUDA kernels built without contraction and the CPU walking the
same tiles, every feature is the same float, every product the same double and every sum the same sequence of
additions, so the normal equations - and after them the ladder's four-way decision, which is copied host code - come
out bit for bit; the dot images reach the reduced rung on about half their calls and the two arms took the same rung
on every one of them. So the rung-flip allowance is untested by any image here; it exists for the fmad-on build and
for other platforms' compilers, where the features can differ in the last bit. ThreadSanitizer (the C1 recipe):
`nntc_cpu_check` three times, and `--backend cpu -j 4` encodes of `tiny.png` in both level-0 modes and of the m1_m4
material, which now run block (a) threaded before they refuse at block (b) or (c): zero reports. A CUDA run's assets,
and a check run's, are byte-identical to `v1.2.3-c3-init`'s (bare, `--l0 palette`, a six-texture material,
`--c0 4 --c1 4`, the rejected-outer-pass line).

*Where the plan was wrong.* Section 4.3 says nothing on the host can download the CUDA arm's workspaces without a new
definition in a kernel file. A separate translation unit that includes `device.cuh` can read any buffer of
`DeviceModel` once the seam call has finished with it, and that is what the probe does; the kernel files are
untouched. Stage C5 uses the same probe for block (b)'s stencil, gradient and preconditioner, which the section
said the harness could only see through the seam's five probes.

**Stage C5 -- blocks (b) and (c').** Both assemblies, the diagonal and lambda, the preconditioner, the CG loop with
`CG_PROBE` kept at 4, the range and movement probes, the quantised sweeps, and the copied `solve_level1_dense_host`.
*Done when* the harness reports `min_diag`, `max_diag`, `lambda`, `iterations` (exact), `residual`, the downloaded
`level1_delta` and the range and movement probes within their bars on every plane; **and** the gate's three block (b)
arms pass within the CPU backend on their own terms: the CG residual under 1e-10, the dense direct solve on a 64x64
crop agreeing with the iterative one, and the finite-difference gradient at the solution near zero. **5 to 7 days** --
the largest stage, and the one most likely to overrun.

**Done: `v1.2.5-c5-level1`.** `src/cpu/cpu_solve_level1.cpp`: both assemblies copied as gathers (the four-tap
dedup-merge that sums a clamped duplicate, `stencil_slot`'s -1, `if (s == 0.0) continue`, block (c')'s centre site a
single tap of weight 1), the form and monomial matrices, `footprint_range`, the diagonal's sum, minimum and maximum
and `lambda`, `invert_block` and the preconditioner stored as float, the mat-vec with its in-bounds tests and no
clamp, the conjugate gradients with `CG_PROBE` 4, the cap of 200 and the bar of 1e-10 (a plane stops at exactly the
probe where the device drops it, with the device's own `>=` test), the correction, the range and the movement probe,
the four-colour sweeps with their Gauss-Seidel over a texel's channels through `model.h`'s `level1_index` /
`level1_value`, `assemble_level0_for_refine`, and `solve_level1_dense_host` with its two downloads deleted. The
reductions are fixed texel chunks folded in chunk order (section 0a); one CG iteration is three pool runs, not
thirteen launches, with every value the same arithmetic on the same operands. The harness gained: the probe's
`workspace` and `monomials` reads, so that every block (b) and (c') call now compares the monomial matrices and every
plane's stencil and gradient to `BAR_ASSEMBLED`, the preconditioner and level 0's correction to the entry bar, and
the sweeps' origin plane exactly - block (b) is a kernel check after all (below); the CG residual judged as described
below; and **the three block (b) arms on the CPU arm's own terms**: under `--check`, main asks for the dense twin right
after round 1's block (b), and the harness then runs, on the CPU arm's OWN solve of that block (its own starting
planes, answer, correction and report, kept from the `solve_level1_all` call), the CPU dense twin against the CPU CG,
the CPU residual against its bar, and main's finite-difference probe on the CPU objective at the CPU answer, printing
one `own terms` line per plane and failing the run past the gate's bars. `nntc_cpu_check`'s kernel arm now runs
blocks (b) and (c') both ways and the refinement's assembly on its five layouts, logging every workspace, report and
plane - 2291 outputs, 311 MB, identical at -j1, -j4, -j16 and -j0 - and runs the three arms on each layout's deepest
plane.

*Measured*, `--backend check`, Windows, MSVC, the CUDA kernels at `-fmad=false`, over 18 command lines (the 17 of
stage C4 and `frymire.png` of the development corpus, 1024x1024, with `--check`): **0 mismatches and 0 flips.** Per
quantity, the worst CPU-against-CUDA difference over every call of the 17 lines (52 continuous block (b) calls, 44
block (c'), 351 level-1 sweeps, 230 level-0 sweeps; `frymire.png`'s summary is within the same figures):

| quantity | worst | note |
|---|---|---|
| the monomial matrices, every plane's stencil and gradient (both latents), the preconditioner | **identical bits** | the assembly is the kernel's float and double arithmetic in the kernel's order; with `-fmad=false` nothing can differ |
| `min_diag`, `max_diag`, the range lo and hi, every sweep's values, indices and moved count, level 0's correction after a sweep | **identical** | the sweeps start from identical stencils and a `lambda` a rounding apart; no index flipped in any call |
| `lambda` | 5.6e-14 | the mean of the diagonal, summed in each arm's order; the summed-scalar bar is 1e-9 |
| `iterations` | **identical on every call** | the harness keeps its slack of one probe for a residual that crosses the bar within a rounding of it; no call used it |
| the correction (`level1_delta`), per entry | 2.0e-12 of the plane's scale (level 1), 6.3e-13 (level 0) | the inner products' summation order, through the same number of iterations |
| `\|delta\|^2`, `\|plane\|^2` | 1.3e-12, 2.6e-12 | the summed-scalar bar is 1e-9 |
| the plane after the correction, per entry | 6.1e-10 of its scale | a float one rounding apart; under `--q1-start 0`, twenty continuous rounds |
| the dense twin, CPU against CUDA, identical inputs | **identical bits** | the same serial double loop |
| the CG residual | at most 0.24 of the stop bar apart, both under it | **not a measure of the port** (below) |

*The three block (b) arms on the CPU backend's own terms*, the gate's bars:

| run | planes | CG residual (bar 1e-10) | dense against CG, of the range (bar 1e-6) | fd (bar 1e-5) |
|---|---|---|---|---|
| `tiny.png --check` | 4 | 1.6e-11, at most 36 iterations | 1.4e-7 | 8.3e-7 |
| `tiny.png --l0 palette --check`, and the gate's own `--k 4` line | 4 | 4.5e-11 | 1.5e-7 | 7.7e-7 |
| the gate's `--k 0 --mips 0` line | 1 | 5.0e-11 | 1.1e-7 | 5.7e-8 |
| `frymire.png --check` (the dense twin on the four chain planes small enough for it) | 8 | 9.6e-11, at most 44 iterations | 2.8e-7 | 7.5e-7 |

*The downstream bar, re-measured* (section 6's C3 note asked for it): on identical inputs the CPU and CUDA conjugate
gradients stop on the same iteration and land 2.0e-12 of the scale apart, and the CPU CG sits 1.1e-7 to 2.8e-7 of the
range from its own dense direct solve, where C1 measured the CUDA pair at 1.8e-7. `BAR_ENTRY` stays 1e-5: it has to
admit a stop one probe apart, whose two corrections are the second distance apart and not the first.

`-j1` and `-j16` harness runs print identical summaries. ThreadSanitizer (the C1 recipe): `nntc_cpu_check` three
times, and `--backend cpu -j 4` encodes of `tiny.png` (bc8, which now runs the whole round loop on the CPU - blocks
(a), (b) and (c'), continuous and quantised - and refuses at the BC pack), `tiny.png --l0 palette --init0 luma
--check` (round 1's blocks (a) and (b) and all three arms, refusing at block (c)) and the m1_m4 material for three
rounds: zero reports. A CUDA run's assets, and a check run's, are byte-identical to `v1.2.3-c3-init`'s (bare, `--l0
palette`, a six-texture material, `--c0 4 --c1 4`, the rejected-outer-pass line).

*Time, and the first whole CPU rounds.* `game2.png` (1024x1024, default layout, `--rounds 4 --tol 0`) under
`--backend cpu` on the 32-thread development machine prints round lines IDENTICAL to the CUDA run's - every E, every
PSNR, `moved0` and `moved1` - for all four rounds, before it refuses at the BC pack. A round (blocks (a), (b), (c')
and the three objective passes) takes 545 to 696 ms on the CPU against 38 to 97 ms on the RTX 5090, 6x to 14x.

*Where the plan was wrong, or did not say.*

* **Section 4.3's "block (b) is a seam check and not a kernel check"**: the probe of stage C4 reads the stencil,
  the gradient, the preconditioner and the monomial matrices, so every block (b) and (c') call is compared at the
  assembly, where a transcription error would show first; the five seam probes are compared as well.
* **"`residual` within its bar"** cannot mean a relative bar on the residual itself. Below the stop bar the CG's
  residual is the rounding of its own inner products: on identical stencils, identical preconditioners and the same
  iteration count the two arms put it 3.71279e-11 against 3.71289e-11 on one plane and a third apart on a plane at
  1e-12. The harness holds it to being under the stop bar on both arms when both stopped on the same probe (and to the
  entry bar otherwise), prints the difference in units of the bar, and judges the solve by its correction.
* **`assemble_level0_for_refine` is not reached through the seam.** `refine_bc.cu` calls the CUDA global directly, and
  main never calls it, so the harness has no call to compare until stage C7's CPU refinement calls the CPU one. Its
  body is `assemble_plane` at a zero ridge, which every `sweep_level0_cont_all` call compares, and
  `nntc_cpu_check` runs it at every thread count.
* **The gate's 1e-6 dense-solve bar is a property of the image, not only of the solver.** On one of
  `nntc_cpu_check`'s synthetic layouts (a random decoder, a six-texel plane) the dense twin and the CG are 3.2e-6 of the
  range apart; driving the CG to a residual of 4e-16 leaves the gap, and forming the site residual in double in a
  scratch copy of the assembly brings it to 3.2e-7. So the gap is the assembly's fp32 residual - CUDA's as much as the
  CPU's, the two being bit-identical - and `nntc_cpu_check` holds that arm to 1e-5 with the reason beside it.
* **Section 3.1's "13 launches per CG iteration"** is three pool runs here; the arithmetic is unchanged.

**Stage C6 -- block (c).** The four-colour exact search, both the enumeration and the sweep branch, with the race fix
and the shadowed-variable trap. *Done when* the harness reports `k0` and `v0` **exact** after each of the four colour
passes, and `moved0` exact, on a `--l0 palette --c0 2 --bits0 3` run and on a `--c0 3 --bits0 3` run (the
`joint == false` branch). **2 to 3 days.**

**Done: `v1.2.6-c6-level0`.** `src/cpu/cpu_solve_level0.{h,cpp}`: `k_level0_search` per texel of one colour - the
site set (the centre site read nearest with alpha 1, then the K fractional sites of the nine pixels around the texel),
the tap-merge rule, p and Q's columns in float, A0 / A1 / A2 in double, `search_energy`, the joint enumeration in its
mixed-radix order with its skip of the current state, the coordinate sweeps with their early break, ties keeping the
current state, and `v0` / `k0` written only for a texel that moved - with the race fix of section 3.4 (the `alpha`
test hoisted above the `beta` accumulation) and the shadowed pair renamed (`ox, oy` for the neighbour pixel, `sdx,
sdy` for the subtexel offset); the driver decides `joint` once per call from `m.bits0` and runs four pool runs per
plane, the moved counts folded in chunk order. With it, `--l0 palette` encodes end to end on the CPU backend. The
harness gained a colour-by-colour check: a colour pass writes only its own texels, so CUDA's state after pass `q` is
the state before the call with the texels of colours up to `q` taken from the state after it; each CPU pass is run
from exactly CUDA's input to that pass (`level0_search_pass`) and compared with exactly its output - `k0`, `v0` and the
pass's moved count against CUDA's count for that colour - under a tally of its own. `nntc_cpu_check`'s kernel arm runs
block (c) twice on both palette layouts (C0 2 at 4 bits enumerates jointly, C0 3 at 4 bits sweeps): 2391 outputs, the
same bits at -j1, -j4, -j16 and -j0.

*Measured*, `--backend check`, Windows, MSVC, the CUDA kernels at `-fmad=false`:

| run | block (c) calls | colour passes | `k0`, `v0`, moved per colour |
|---|---|---|---|
| `tiny.png --l0 palette --c0 2 --bits0 3` (6 bits, joint) | 23 | 368 | **identical** |
| `--l0 palette --c0 3 --bits0 3` (9 bits, the `joint == false` sweeps) | 24 | 384 | **identical** |
| `--l0 palette --bits0 4` (8 bits, joint) / bare `--l0 palette` | 23 / 23 | 368 / 368 | **identical** |
| `--l0 palette --c0 4 --c1 4` (16 bits, sweeps) | 25 | 400 | **identical** |
| `--l0 palette --check`, and the gate's `--k 4 --check` line | 23 / 23 | 368 / 368 | **identical** |
| `m1_m4_source_material.json --l0 palette --c0 3 --c1 4` | 24 | 672 | **identical** |
| `game2.png` of the development corpus, 1024x1024, `--l0 palette` | 23 | 736 | **identical** |

0 mismatches and 0 flips on all nine runs, and every value bit-identical (worst relative difference 0). That is what
the kernel's structure predicts: the search has no reduction across threads - a texel's quadratic is one serial double
sum over its own sites in the kernel's order, from float features that `-fmad=false` makes the same floats - so on
identical inputs nothing is left to differ. Mutations in a scratch build say what the check can see: the tap-merge rule
broken (`alpha =` for `alpha +=`) is a MISMATCH on the first call of both the joint and the sweep line; the sweeps'
alphabet one short (`kk < levels`) is a MISMATCH on the `--c0 3 --bits0 3` line and invisible on the joint one, as it
should be; the tie rule turned (`<=` for `<`, in either branch) is NOT detected, because two distinct states whose
double energies are exactly equal did not occur on any of these images - the rule is held by reading the code, not by
a measurement. ThreadSanitizer (the C1 recipe): `nntc_cpu_check`, and `--backend cpu -j 4` full encodes of `tiny.png
--l0 palette` and the m1_m4 material in palette mode: zero reports (section 3.4 records what the sanitizer could not
see, and why the audit is the proof). A full CPU encode of `tiny.png --l0 palette` is
byte-identical at -j1, at -j32 and on a repeat. A CUDA run's assets, and a check run's, are byte-identical to
`v1.2.5-c5-level1`'s (bare, `--l0 palette`, a six-texture material, `--c0 4 --c1 4`, the rejected-outer-pass line).

**Stage C7 -- the BC refinement.** `k_bc_decode`, the per-block serial routine with the 64-slot tree replayed, the
four colours, the pass loop, the state save and restore. *Done when* the harness reports the endpoints and selectors
**exact** after every pass on `tiny.png` and one material, and the gate's `--l0 bc8` arms pass under `--backend cpu`:
every refinement pass lowers the block objective or is refused, the outer repack's accept and reject both exercised
with the reject restoring `E` exactly, and the packing PSNR rising over the passes. **4 to 5 days** -- a day more than
the draft, for the reduction tree it wrongly deleted.

At the end of stage C7 the whole gate should run under `--backend cpu`. Tag `v1.2.1-cpu-backend`.

**Done: `v1.2.7-c7-bc`** (the tag named above was never going to fit the numbering: `v1.2.1` is stage C1's harness).
`src/cpu/cpu_refine_bc.{h,cpp}`: `k_bc_decode` per texel; `k_bc_refine` as one serial routine per 4x4 block - the
quadratic gathered from the five owned stencil blocks and the transposes, `a1` with the neighbours' share, `uv = a2 dv`,
the endpoint stage (the five sums, the relative determinant guard, the vertex rounded with `std::floor`, the nine
candidates with the eight-value mode's `continue`, the float-palette re-weighing and the strict `exact < 0`
acceptance), the two Gauss-Seidel selector passes with their strict `de < best`, the edge rule, and the write-back -
spread over the blocks of one colour, four colour runs per plane per pass; the copied `bc4_weight8`, `bc4_entry` and
`bc0_value` (its host half); and the drivers `bc_pack_prepare` (the CPU's own `assemble_level0_for_refine`, then
`export.cpp`'s `level0_pack_seed` exactly where the CUDA driver calls it, then the decode), `bc_refine` and
`bc_blocks_from_device` (through `export.cpp`'s `level0_pack_blocks`). `bc_state_save` and `bc_state_restore` were
stage C1's copies. Every seam function now has a body; `cpu_backend.cpp` refuses nothing. **The CPU backend encodes end
to end.**

The harness gained three things for the refinement. The pack's call compares the level-0 workspace its own assembly
built (the stencil and gradient to `BAR_ASSEMBLED`, the plane the correction is measured from exactly) and the
correction `k_bc_decode` wrote. A refinement call of N passes is taken **one pass at a time on both arms**: the CUDA
arm's `bc_refine` runs the same launches in the same order whether it is asked for N passes once or one pass N times, so
the harness asks N times (and hands main the report it would have had, pass numbers included - a check run's assets are
still byte-identical to a plain CUDA run's), and every pass is a harnessed call whose CPU side is given CUDA's planes and
BC state through the mirror and CUDA's level-0 workspace through the probe (`mirror_level0_workspace`), since the
refinement reads a stencil another call assembled. And each pass is taken **one colour at a time** as well, exactly as
block (c) is: a block of one colour writes only its own endpoints, selectors and texels, so CUDA's input to colour `q`
and its output from it are recovered from the states before and after the pass. `nntc_cpu_check`'s kernel arm runs the
seed pack, two passes, the continued pack and one more pass on its three bc8 layouts (C0 1, 2 and 3): 2910 outputs, 413
MB, identical at -j1, -j4, -j16 and -j0, with no pass accepting a rise.

*Measured*, `--backend check`, Windows, MSVC, the CUDA kernels at `-fmad=false`:

| run | pack calls | refinement passes | colour passes | endpoints, selectors, `v0`, `k0`, correction | accepted decrease |
|---|---|---|---|---|---|
| `tests/tiny.png` bare / `--c0 1` / `--c0 3` / `--c0 4 --c1 4` | 3 each | 12 each | 192 each | **identical** | 8.5e-16 at worst |
| `--bc-outer 3` / `--bc-refine-after 2` / `--check` | 4 / 4 / 3 | 16 / 14 / 12 | 256 / 224 / 192 | **identical** | 9.7e-16 |
| the rejected-outer-pass line (`--bc-refine 0 --bc-outer 2`) | 2 | 0 | 0 | **identical** | - |
| `m1_m4_source_material.json --c0 3 --c1 4` | 3 | 12 | 336 | **identical** | 5.6e-15 |
| `game2.png` / `frymire.png` of the development corpus, 1024x1024 | 3 / 3 | 12 / 12 | 384 / 384 | **identical** | 9.9e-15 / 7.9e-15 |

0 mismatches and **0 flips** on all eleven runs: every endpoint and selector of every pass and every colour identical,
every value and correction bit-identical. The one figure that differs is the pass's summed decrease, by the order of
its additions (the device's is an atomic), a few units in the 16th digit. So the serial fold that section 0a put in
place of the 64-slot tree moved no decision here: on these images no endpoint sum sits within a rounding of the
determinant guard, of a candidate's tie or of zero. Mutations in a scratch build: the eight-value test loosened
(`a0 < a1` for `a0 <= a1`) and one index swapped in the acceptance's quadratic are MISMATCHes on the first line they
touch; the determinant guard loosened a billionfold (`1e-3` for `1e-12`) and the selector argmin's tie turned (`<=`) are
NOT detected on `tiny.png` - no block there sits near either - and, like block (c)'s tie rule, are held by reading.

*The whole encode on the CPU backend* (32 threads, against the RTX 5090, both from the same build; wall clock of the
process). Every CPU encode's every written file is **byte-identical to the CUDA encode's** on all eleven lines - the CUDA
kernels built `-fmad=false`, every reduction on either arm summing the same doubles in an order that happens to give
the same bits on these inputs - so the PSNRs, `E shipped`, the round counts and the outer passes' verdicts are equal to
the last digit; the column says so rather than repeating them.

| input | CUDA | CPU | ratio | texture PSNR (both backends) |
|---|---|---|---|---|
| `tiny.png` default (bc8) / `--l0 palette` / `--c0 4 --c1 4` | 0.32 / 0.25 / 0.39 s | 0.62 / 0.50 / 0.79 s | 2.0x | 33.89 / 28.89 / 39.66 dB |
| `tiny.png --l0 palette --bc0 0` / `--mips 0` / the rejected-outer-pass line | 0.26 / 0.26 / 0.28 s | 0.49 / 0.39 / 0.57 s | 1.5 to 2.1x | 28.50 / 35.00 / 33.39 dB |
| six 32x32 crops of `tiny.png` (the gate's six-texture material) | 0.45 s | 0.41 s | 0.9x | 19.35 16.39 20.52 22.01 17.51 18.13 dB |
| `m1_m4_source_material.json` (4 textures, 512x512) | 1.51 s | 19.07 s | 12.6x | 24.27 33.05 34.45 35.82 dB |
| `pavingstones141_1k_source_material.json` (5 textures, 1024x1024) | 5.69 s | 47.44 s | 8.3x | 28.40 27.60 32.83 33.25 36.64 dB |
| `game2.png`, 1024x1024, default / `--l0 palette` | 2.81 / 1.33 s | 23.72 / 18.57 s | 8.4x / 14.0x | 46.82 / 46.87 dB |

So the whole encode is 8 to 14 times the 5090 on real inputs, the per-round figure stage C5 measured, and not the
study's 2 to 3 times (section 1.2); on the tiny inputs the encode is the host's own work and the two are close. No
optimisation was attempted (decision 6 and this stage's brief).

*Determinism.* The CPU encode is byte-identical at -j1, at -j32 and on a repeat on the eight smaller lines of the table
(every tiny line, the six crops and the m1_m4 material, whose -j1 run takes 72 s). ThreadSanitizer (the C1 recipe):
`nntc_cpu_check`, and `--backend cpu -j 4` full encodes of `tiny.png` and of the m1_m4 material in the default bc8
layout, whose 512x512 base puts 64 block chunks in every refinement colour: zero reports (section 3.4 says what the
sanitizer can and cannot see). A CUDA run's assets, and a check run's, are byte-identical to `v1.2.5-c5-level1`'s
(bare, `--l0 palette`, a six-texture material, `--c0 4 --c1 4`, the rejected-outer-pass line).

*The release gate under the CPU backend* (section 7.2): `tests/run_checks.py --backend cpu` appends `--backend cpu` to
every `nntc_encode` command line and to nothing else, and writes under `out_cpu/`. **It ends "All checks passed."**:
every arm, including the fifteen byte-for-byte ones (`determinism_check` among them), the round loop's monotonicity,
the quantisation round trips, `dds_decode.py`'s agreement, the objective against its twin, block (b)'s three arms, the
ridge tally, the ill-conditioned dot images, every bc8 arm - each refinement pass lowering the block objective, the
outer repack accepting on the default line and rejecting on the `--bc-refine 0 --bc-outer 2` line with the restore
putting `E` back exactly and the restored asset round-tripping, the packing PSNR rising over the passes (37.95 to
38.26 dB on the default line) - the
materials, the six textures, the mip filters and the viewers on the CPU's assets, with zero VUID lines. No arm failed,
so none had to be classified; the arms that compare against a number measured on CUDA pass because, as above, the CPU
backend reproduces CUDA's numbers exactly on these inputs.

*Where the plan was wrong, or did not say.*

* **Section 5.6's 64-slot tree replay** was superseded by section 0a before this stage; the serial fold flipped nothing
  measured (above).
* **"The endpoints and selectors after every pass"** needs the CUDA arm to stop between passes, which main's one call
  of N passes does not; the harness makes N calls of one pass, which the CUDA driver's structure makes identical.
* **The refinement reads a workspace another call wrote** (section 4.3's case): the probe of stage C4 reads the CUDA
  arm's level-0 workspace and the harness writes it into the CPU model before each pass.
* **The block chunk.** `CPU_CHUNK` blocks to a chunk would put a 1024x1024 plane's refinement colour in four chunks, so
  the refinement has its own constant, 64 blocks.
* **`nntc_cpu_check` now links `export.cpp` and the dispatcher**, because the CPU refinement packs through the same host
  code the CUDA driver does; it compiles them without `NNTC_CUDA`, as the CPU-only configure does, and compiles
  stb_image_write's implementation itself.
* **Section 7.1's "one line in `out_asset`"** was not enough for its own purpose: the gate names `out/` directly in 121
  other places, so every one now reads the output root the run chose.

### Step D

**Stage D -- comparison, tolerances, and the portable gate.** Section 7: the gate's `--backend cpu` mode and its
`out_asset` fix, the thread-count arm, the cross-backend arm, the measured corpus behind the tolerances, the
ThreadSanitizer arm on the Linux box, and the write-up. Also the first build and gate run on the Arm laptop, the Intel
laptop and the AMD Linux box, which is the deliverable the whole plan exists for. **4 to 5 days.**

**Done: `v1.3.0-cpu-backend`**, on the development machine (Windows and WSL); the three other machines are not yet
run (below). `tests/run_checks.py` gained three arms, each written up where its section is:

* **the thread-count arm** (7.3), `thread_count_checks`, in every gate: `nntc_cpu_check` (every kernel's every output
  the same bits at -j1, -j4, -j16 and -j0, and the pool's contract at eight counts), then CPU encodes of `tiny.png` at
  the default layout and at `--l0 palette` and of a 128x128 crop of `examples/m3.png` at -j1, -j64 and the default
  count, every written file byte-identical. **28.1 s** of a Release gate (`nntc_cpu_check` 19.2 s, nine encodes
  8.9 s); `nntc_cpu_check` alone is 79 s in a Debug build;
* **the cross-backend arm** (7.4), `cross_backend_checks`, in the default run where both backends can run: `tiny.png`
  and the one-white-pixel dot always (**1.9 s**), and with `--cross-examples` the two `examples/` materials as well
  (**98.7 s**, against a whole Release gate of 101 s before this stage - hence opt-in). All four cases are
  byte-identical between the backends on this machine, so every difference is 0.00 dB;
* **the ThreadSanitizer arm** (3.4), `tsan_check`, last, Linux only: the CPU-only configure with `-fsanitize=thread`
  under `out/tsan_build_posix`, `nntc_cpu_check` and two `-j 4` encodes of `tiny.png`, zero reports.
  **124 s** for `nntc_cpu_check`, 2.1 s and 1.6 s for the encodes, 26 s for a build from nothing with 32 cores.

*The gates, on the final binaries.* Release (**128 s**, 101 s before this stage), Debug (**369 s**, of which the
thread-count arm is 123 s) and `--backend cpu` (**144 s** - section 7.2 asked for this measurement: far under the
twenty minutes past which it proposed a subset, so the whole gate runs) each end "All checks passed." with zero VUID
lines; the WSL build's gate, TSan arm
included, ends the same (**255 s**; the TSan arm 133 s of it, its build persisted from an earlier run). A CPU-only WSL configure (`-DCMAKE_CUDA_COMPILER=NOTFOUND`) prints the CMake
Warning, builds, and encodes `tests/tiny.png` under `--backend auto` with the fallback WARNING first and `backend cpu`
in the report, 33.89 dB, its three files byte-identical to the Windows builds' assets of the same line (both
backends): the deliverable for a machine with no NVIDIA card. The kernel hashes did not move.

*Where the plan was wrong, or did not say.*

* **Section 7.3's six counts** became three in the encode half (-j1, -j64, the default, which is -j32 on this
  machine) plus `nntc_cpu_check`'s four on the kernels: -j2 and -j7 differ from those only in how many participants
  share the chunks, which `nntc_cpu_check` exercises at 1, 2, 3, 7, 16, 32, 64 and 0. A material at six counts was
  not affordable (the m1_m4 material's -j1 encode alone is 72 s); `tiny.png` alone would not have been enough either,
  being one texel chunk, hence the crop, whose base is four chunks and whose refinement is four block chunks a colour.
* **Section 7.4's per-quantity table** (0.05 dB on a texture, `moved0` / `moved1` equal per round, `--tol 0` with
  fixed rounds) was written before the rounding experiment below it, and that experiment retired it: the arm runs
  the default command lines, judges the distribution with the bars of 7.4's "what was built", and reports the round
  counts and stop reasons instead of asserting them.
* **The bias bar needs a distribution.** Without `--cross-examples` the arm judges one texture, and one texture's
  difference is one draw of the path dependence, not a mean; the bias bars apply from five judged textures and the
  gross bars always.
* **The Arm laptop, the Intel laptop and the AMD Linux box** have not built or gated this tag. Nothing here depends
  on them, but it is the deliverable section 7.1 names, and it is still owed.

### Totals, and an honest comparison with the first draft

| step | first draft | now | why |
|---|---|---|---|
| A | 3 to 4 | 3 to 4 | much less code (no moves, no wrap, no extraction) but a corpus of 26 lines instead of 15 and the contraction check |
| B | 1 | 1 | plus the hash file, which is minutes |
| C | 17 to 24 | 21 to 29 | **+3 to 4** the harness (new), **+1** the ladder copy that the extraction used to give free, **+1** the BC reduction tree, **+1** the race audit and the CPU-only TSan configure |
| D | 3 to 4 | 4 to 5 | the TSan arm and the three new machines |
| **total** | **24 to 33 days** | **29 to 39 days** | |

**So it is 7 to 9 calendar weeks, not 6 to 8.** Removing the refactor did not pay for the harness, and it was never
going to: the refactor was three days of mechanical work and the harness is a small program. The trade is a good one
anyway -- the refactor bought nothing but a smaller duplication count, at the price of being the only thing in the
plan that could change what the CUDA encoder computes, while the harness is the difference between verifying 3200
lines as they are written and debugging them all in one sitting. But the honest number went up and it is written here
rather than averaged away.

---

## 7. The test plan

### 7.1 The release gate becomes portable, which is the bigger deliverable

This is the part the first draft did not say out loud, and it is worth more than the encoding.

`tests/run_checks.py` is 4021 lines and about fifty check functions. **Today it cannot run at all on a machine without
an NVIDIA card**, because nearly every arm begins with an `nntc_encode` invocation, and `README.md` says so. That is
why the Windows-on-Arm Snapdragon, the Intel laptop and the Kubuntu AMD box are viewing machines and nothing more: the
two viewers can be exercised there, and the encoder, the format's writer, the quantisation invariant, the determinism
property and every structural assertion about the solver cannot.

**Under decision 8 that changes with no work at all beyond the backend existing.** `--backend auto` on a machine with
no CUDA device falls back to the CPU backend, so **the whole gate runs there, unchanged, on the command line it
already uses.** No new arm, no flag, no branch in the script. A build on the Arm laptop runs `determinism_check`, the
fifteen byte-for-byte arms, `level_checks`, `quantisation_checks`, `objective_checks`, `ill_conditioned_checks`,
`ridge_tally`, `bc_checks`, `bc8_checks`, `material_checks`, `mip_filter_checks`, `six_texture_checks`,
`argument_and_status_checks`, `never_delete_check` and `check_tree`, on a machine that has never compiled the encoder.

Three consequences worth naming:

* **the gate stops being a property of one desktop.** Every structural claim this tree makes about the encoder --
  `E` monotone, no block raising it, every stored index re-quantising to itself, the pack lossless at 2 and 3 bits,
  the repack's reject restoring `E` exactly -- becomes checkable on four machines with four compilers and two
  instruction sets, which is a far stronger statement than checking it on one;
* **a compiler difference that changes a result shows up as a third comparison**, between two CPU runs on different
  machines, which the plan gets for free and should record;
* **1.12's caveat applies**: a `--mip-filter` arm may legitimately differ between an x86 and an Arm CPU run because
  the vendored resizer picks a different SIMD path, and that is the source chain and not the port. The arms that
  compare two files byte for byte *within one run on one machine* are unaffected, which is most of them.

**The one script change, and when it is needed.** `run_checks.py:62`'s `out_asset(name, source)` returns
`os.path.join('out', name)`, a fixed directory per check. Running the gate twice on one machine, once per backend,
would have the second run overwrite the first's assets. The one-line fix is to put the backend in the path --
`os.path.join('out', BACKEND_TAG, name)` with `BACKEND_TAG` empty by default -- and it is needed only if somebody
wants both runs' output kept side by side. **Done in stage C7** as a root rather than a subdirectory: `--backend cpu`
writes under `out_cpu/`, and the gate's 122 references to `out/` all read that root. On a machine with no GPU there is only one backend and the question does
not arise, which is the more important case.

### 7.2 Within the CPU backend: the existing gate, with a flag

The cheapest and by far the most thorough way to test the CPU backend on a machine that *does* have a GPU is to run
the gate that already exists against it. `tests/run_checks.py` gains one argument, `--backend cpu`, which prefixes
`--backend cpu` (and `-j`) onto every `nntc_encode` invocation and leaves every assertion exactly as written. That
buys, with no new assertion authored at all:

* **the fifteen byte-for-byte arms** listed in 2.4. Every one must hold within the CPU backend. In particular
  `determinism_check` -- two CPU encodes of one image, byte-identical over three files -- is decision 9's second half
  asserted directly;
* **the structural checks**, which are the ones that say the port is *right* rather than merely repeatable: `E`
  monotone across every block of every round; no block raising `E` beyond `E_NOISE` / `E_REL`; every stored level-1
  index re-quantising to itself; `tools/dds_decode.py` -- a reader with no GPU and no CPU backend, written in Python --
  agreeing with the report within one count at every mip level and to 0.01 dB; the CG residual, the dense solve and
  the finite-difference gradient; the outer repack's restore leaving `E` at the pre-pass value; the BC pack lossless
  at 2 and 3 bits; the ridge tally all-standard on `tiny.png`; the near-singular dot image monotone at three extents;
* **the refusals, the argument parsing, the never-deleting arm and the tree scan**, unchanged.

Two things to decide with a measurement rather than in advance. **How long it takes**: most of the gate's encodes are
`tests/tiny.png` and are trivial on any backend, but the material and six-texture arms are not, and the whole CPU pass
could be tens of minutes. Measure it in stage D; if it is over about twenty minutes, `--backend cpu` runs a named
subset -- the fifteen byte arms, the structural arms and the two `--l0` modes -- and the rest stays CUDA-only, with
the skipped arms *recorded as skipped* and never as run, which is the rule the Vulkan arms already follow.
**Whether the viewer arms run**: they do not; they are about the asset and not about the backend, and they are skipped
by name.

### 7.3 The new arm: thread-count independence

One encode, six thread counts, one assertion:

```
nntc_encode tests/tiny.png -o out/j1  --backend cpu -j 1
nntc_encode tests/tiny.png -o out/j2  --backend cpu -j 2
nntc_encode tests/tiny.png -o out/j7  --backend cpu -j 7
nntc_encode tests/tiny.png -o out/j32 --backend cpu -j 32
nntc_encode tests/tiny.png -o out/j64 --backend cpu -j 64
nntc_encode tests/tiny.png -o out/j0  --backend cpu -j 0
```

All six `.dds` and `_nntc.json` byte-identical. The draft called `-j7` "the one that matters"; the review corrected
that and it is corrected here. **`-j1` is the adversarial count**: it is the only one that takes a different code
path, rule 5's inline chunk loop with no pool at all. `-j64` on a machine with fewer cores exercises
oversubscription *and* **more workers than chunks**, which a deep mip plane reaches at one or two chunks and which
must neither deadlock nor change a byte. `-j0` is the default and goes through `hardware_concurrency()`. `-j2`, `-j7`
and `-j32` remain, cheaply, as the ordinary cases. Run the whole set on one material as well as on `tiny.png`, because
`tiny.png` may not reach every kernel's reduction.

And, separately and permanently, **the ThreadSanitizer arm of 3.4**: a CPU-only Linux configure with
`-fsanitize=thread`, a `-j4` encode of `tiny.png` and one small material, zero reports required.

**What was built (stage D).** `thread_count_checks` in `tests/run_checks.py`, run by every gate (it names `--backend
cpu` itself, so a gate with no GPU and a `--backend cpu` gate run it alike). It runs `nntc_cpu_check` and requires its
kernels line - every ported kernel's every output identical at -j1, -j4, -j16 and -j0 over its five synthetic layouts
(2910 outputs, 413 MB) - and its pool and transfer lines; then encodes `tests/tiny.png` (default layout and `--l0
palette`) and a 128x128 crop of `examples/m3.png` (default layout) at -j1, -j64 and the default count, and requires
every written file identical across the three. Measured: 28.1 s of a Release gate, all identical. The material at six
counts was not adopted: at 72 s for one -j1 encode it would triple the gate, and the crop reaches multi-chunk planes
(four texel chunks at the base, four block chunks a refinement colour) for a few seconds. The ThreadSanitizer arm is
`tsan_check`; section 3.4 has what it runs and what it cannot see.

### 7.4 The cross-backend arm, and what it may assert

**The corpus.** Tracked, so the arm runs anywhere: `tests/tiny.png`; one single image and one four-texture material
from `examples/`; and, for setting the numbers rather than for the gate, the measured material sets of
`docs/RESULTS.md`.

**Three conditions the arm fixes before it compares anything.** All three exist to stop a last-bit difference being
amplified into a discrete one:

* **`--tol 0` with a fixed `--rounds`**, which removes the tol stop entirely (1.9: with `o.tol` zero, `drop < 0.0` is
  never true and `tol_hits` never reaches 2);
* **`--png 0`**, so the arm compares the report and the asset and not a PNG encoder;
* **and the arm asserts the printed `stop_reason` and the round count are EQUAL, and that neither run stopped early.**
  `--tol 0` does not by itself pin the round count, because the **no-move** stop at `main.cpp:2065-2072` is evaluated
  first and is unaffected by `--tol`. If either backend stops on `no-move` the arm says so by name, because every
  number below is then a comparison between runs of different lengths.

**The proposed tolerances.** Every one of these is a *proposal to be replaced by a measurement*: they are set from the
measured corpus in stage D and only then written into the gate. The starting points are anchored on
`docs/DESIGN.md` section 6's Windows-against-WSL measurement, which is the same tier of agreement and the only
cross-implementation number this tree actually has.

| quantity | measured cross-toolchain | proposed cross-backend bar |
|---|---|---|
| `psnr texture t` | 0.00 dB on all four textures | 0.05 dB |
| `mip psnr M1..Mn` | 0.03 dB at the deepest level, 0.00 to 0.01 elsewhere | 0.10 dB |
| `centre psnr`, `sampled psnr` | not separately recorded | 0.05 dB |
| `E shipped` | 1.9e-5 relative (7.069498521e-04 against 7.069363374e-04) | 1e-4 relative |
| level-0 packing PSNR | not separately recorded | 0.10 dB |
| `rounds` and `stop_reason` | -- | **equal, asserted**, and neither may be `no-move` |
| **`moved0` / `moved1`** | -- | **asserted, not merely printed.** Starting proposal: equal per round. They are integer counts of discrete decisions, they are the earliest visible symptom of a divergence, and the round count depends on them. If the measured corpus shows they differ, the bar becomes a measured spread **with the round at which they first diverged recorded**, never a shrug |
| the refinement's accepted decrease | -- | printed, not asserted (3.2: the device's is an order-dependent atomic) |

**The one case that needs its own handling.** Block (a)'s ridge ladder takes a four-way decision on a float
comparison: the candidate is accepted when `(q_cand - q_prev) * inv_norm + err_prev + err_cand <= 1e-12`, and an
inconclusive comparison is a refusal. On a near-singular normal matrix the rungs are genuinely far apart --
`docs/MATHEMATICS.md` section 3 records the dot image ending at 4.627044e-07 and **60.46 dB** with the ladder against
1.268264e-06 and **53.69 dB** without it, which is about 7 dB between two rungs. A blanket PSNR tolerance over a
corpus that includes such an image would either fail spuriously the first time the two backends took different rungs,
or -- if loosened to 7 dB to accommodate it -- prove nothing anywhere else.

So the arm splits, exactly as the existing gate already splits:

* **on the well-conditioned corpus**, assert the **ridge tally is all-standard on both backends** (`[N, 0, 0, 0]`, and
  the same `N`, since `--rounds` is fixed), and then apply the table above. If the tally is all-standard on both, the
  ladder took the same rung every time and the PSNR comparison is meaningful. If it is not, the image does not belong
  in this corpus and the arm says so by name rather than failing a decibel bar;
* **degenerate images are their own case.** The one-white-pixel dot at its three extents gets the treatment
  `ill_conditioned_checks` already gives it: on each backend independently, `E` monotone over every round, no block
  raising `E`, and the tally **printed and not asserted**. **No cross-backend PSNR bar is applied to it at all**, and
  the reason is written in the check's docstring so that nobody later "tightens" it.

**What the arm reports.** A table of the quantities above, both backends side by side, with the difference and the bar
-- the shape `docs/DESIGN.md` section 6's Windows-against-WSL table already uses, so the two can be read together.

**Measured: the encoder's own sensitivity to rounding, and why the bars above cannot hold.** Before any CPU kernel
existed, the `NNTC_CUDA_FMAD` option gave a clean experiment: the SAME CUDA code, compiled once with nvcc's fused
multiply-add and once with `-fmad=false`, over a 20-case corpus of materials and single images (seven
material sets from the development corpus, both shipped examples, eleven images, layouts from `--c0 1 --c1 2` to `--c0 4 --c1 4`, both level-0
modes, `--bc0 0`, `--mips 0`, a mitchell chain, a short round budget). Nothing but the last bit of each multiply-add
changed. Result:

* **ON versus the pinned pre-option baseline: byte-identical on all 20.** The option restores the old build exactly.
* **OFF versus ON: 15 of 20 within about 0.1 dB on every base texture, and 5 well outside it.** The worst:
  `md` at `--c0 4 --c1 4` moved one texture by -1.03 dB and another by +0.86 dB, with a mip level at -2.32 dB;
  `me` moved -0.53 dB on one texture and +1.18 dB on a mip level; `kodim01` with a mitchell chain moved +0.39 dB.
* **The changes go both ways, and round counts and ridge tallies were identical in every case.** So this is not a
  quality loss: it is path dependence. A last-bit change flips some quantisation decisions, and the solver settles in
  a different optimum of about the same quality. The four-channel and five-texture materials are the most sensitive.

**What that means for this arm.** It measures the encoder's noise floor with zero algorithm change, and a correct CPU
port differs from CUDA in more than one rounding, so it will scatter at least this much. The per-case bars of
0.05 dB and 0.10 dB above are therefore unachievable on about a quarter of any wide corpus, and the arm must be
defined against the noise floor instead:

1. **No systematic bias.** Over the corpus, the mean signed per-texture difference (CPU minus CUDA) must be near zero
   - a porting bug pulls one way; path dependence scatters both ways. Proposed bar: mean within 0.05 dB, and the
   CPU no worse than CUDA on more than two thirds of cases (a coin flip would sit near half).
2. **Scatter no wider than the noise floor.** Each case's worst per-texture difference must lie inside the envelope
   the fmad experiment measured for that same case, with a margin: the arm runs CUDA with `NNTC_CUDA_FMAD` both ways
   to get that case's own envelope, and the CPU result must fall within it (or within 0.1 dB where the envelope is
   smaller). A case outside its own envelope is a lead to follow, not an automatic failure.
3. **The per-kernel harness, not end-to-end PSNR, is what catches bugs.** It feeds each kernel identical inputs on
   both sides, so there is no path to diverge along, and a porting mistake shows as a disagreement orders of
   magnitude outside rounding. End-to-end PSNR is too noisy an instrument for a tight bar; the harness is not.

The numbers above are set from 20 cases and should be re-measured on the wider corpus in stage D.

**Measured with the CPU backend (after stage C7), on the same 20-case corpus** (the development corpus's material
sets and images, both `examples/` materials and `tests/`), default settings, the CUDA kernels at the default
`NNTC_CUDA_FMAD=OFF`: **17 of 20 CPU encodes byte-identical to CUDA's**; over the 51 textures the mean per-texture
difference, CPU minus CUDA, **-0.06 dB, median 0**, no gross difference. CUDA rebuilt with `NNTC_CUDA_FMAD=ON` from the
current tree was byte-identical to a pinned pre-port build on all 20, so the CUDA path did not regress through stages
C1 to C7. The one case outside the scatter is the subject of 7.4a.

**What was built (stage D), and the bars it holds.** Points 1 and 2 above assumed a scatter on every case and an arm
that rebuilds CUDA both ways per case; what the corpus showed is agreement to the bit on most cases and one outlier
whose cause is known (7.4a), so the arm is simpler and its bars are the ones the owner set against that measurement:

| judged | bar | on failure |
|---|---|---|
| any one texture, CPU minus CUDA | worse than -4 dB | FAIL: gross, beyond what path dependence produced anywhere |
| any one mip level (the report's worst texture at that level) | worse than -6 dB | FAIL: gross |
| the mean signed per-texture difference over the judged textures (at least five) | below -0.5 dB | FAIL: systematic bias |
| the same mean | between -0.2 and -0.5 dB | WARNING |
| everything else: byte identity per case, every texture's difference, the worst mip level, both round counts and stop reasons, both ridge tallies, both times | -- | reported, never asserted |

`cross_backend_checks` in `tests/run_checks.py` runs only tracked inputs, with `--png 0` and otherwise the default
command lines: `tests/tiny.png` always, and with `--cross-examples` `examples/m1_m4_source_material.json --c0 3 --c1 4`
and `examples/pavingstones141_1k_source_material.json --c0 4`, whose CPU encodes take 24 s and 64 s on 32 threads -
together 98.7 s, the length of the whole Release gate before this stage, so they are opt-in. It runs in the default
run of the gate and only where `--backend cuda` works; a build without CUDA, or a machine where the device does not
answer, prints the encoder's own ERROR line and records the arm as skipped. The one-white-pixel dot (64x64) is encoded
by both backends and reported with no bar, for the reason above, written into the arm's docstring. A case whose
ridge tally is not all-standard on both backends would be taken out of the judged set by name. Byte identity is
printed per case and never asserted (7.5).

Measured on this machine, both backends from one Release build:

| case | files | textures, CUDA | CPU minus CUDA | worst mip level | rounds | CUDA / CPU time |
|---|---|---|---|---|---|---|
| `tests/tiny.png` | byte-identical (3) | 33.89 dB | +0.00 | +0.00 | 20 / 20, budget | 0.30 / 0.60 s |
| `m1_m4 --c0 3 --c1 4` | byte-identical (4) | 28.74 34.36 35.03 36.68 | +0.00 on all | +0.00 | 20 / 20, budget | 1.55 / 23.63 s |
| `pavingstones141_1k --c0 4` | byte-identical (4) | 35.85 30.82 35.92 39.00 42.27 | +0.00 on all | +0.00 | 20 / 20, budget | 7.28 / 63.76 s |
| the dot, 64x64 (no bar) | byte-identical (3) | 63.88 | +0.00 | +0.00 | 20 / 20; ridge 14 standard, 14 reduced on both | 0.30 / 0.60 s |

### 7.4a The one outlier, investigated

The investigation was made by the session that directed stage D, before stage D began; what follows are its
measurements, recorded as facts.

* **The case.** A five-texture 2048x1024 material of the development corpus at `--c0 4 --c1 4`, with the default
  `--init0 residual`. The CPU backend landed **0.4 to 0.6 dB lower** on average over its textures.
* **The per-kernel harness on that exact material:** 0 mismatches, 0 flips, every kernel, and all 384 BC colour
  passes identical. So on identical inputs every kernel agrees.
* **The residual seed** (`init0_residual_channel`) agreed only to **8.35e-9 relative** there, against about 4e-14 on
  the small inputs stage C3 measured. The eigenproblem is ill-conditioned: four level-0 channels, nearly equal
  eigenvalues, so the fp64 covariance's summation-order difference (section 0a lets each backend sum in its own
  order) is amplified in the eigenvector.
* **End to end, the runs first diverge in round 1**, where the level-1 grid snap moves 552130 texels on CUDA and
  552097 on the CPU.
* **A round-budget sweep** (16 to 24 rounds) had the CPU lower every time. Those runs share the same round-1 fork, so
  they are not independent samples.
* **Six ridge values, 0.00009 to 0.00012**, each an independent fork of the SOLVE: CUDA's mean wandered from 39.08 to
  39.65 dB between basins, the CPU's stayed at 38.97 to 39.01 dB, one basin, and CUDA landed in that same basin at one
  of the ridge values. The CPU was lower in 6 of 6 - because the ridge does not change the seed, which is computed
  before any solve, so every one of those forks started from the same CPU seed.
* **The same seven-ridge test with `--init0 luma`**, whose seed is bit-identical between the backends: the CPU better
  in 3 and worse in 4, **mean -0.094 dB**, the differences from -0.79 to +0.80 dB.

**So the solver has no bias.** The default-settings gap on this material comes from the residual seed's summation
order meeting an ill-conditioned eigenvector: a different seed puts the CPU run in a different basin, and on this
material it happens to be a slightly lower one. With a seed that agrees to the bit, the two backends scatter both ways
about zero, which is path dependence and nothing else.

**An open decision for the owner, not taken here.** Replaying CUDA's reduction order for the residual seed's
covariance only - a narrow exception to section 0a, confined to that one seed's fp64 covariance sums - would make that seed
bit-identical between the backends and remove this case's fork at its source. Stage D does not implement it. (Taken in
`v1.3.1-cpu-parity`, commit `3788a31`: `cov_partials` replays `k_cov_partial`'s order; this material went from -0.606 dB
to +0.092 dB mean, and the corpus from 17 of 20 byte-identical to 19 of 21 - docs/CPU_PORT_LESSONS.md section 4.)

### 7.5 What is deliberately not asserted

Byte equality between the backends, at any granularity. Section 1.5 says why: the device contracts fp32 multiply-adds
and the host does not, so the features differ in the last bit and everything downstream inherits it. Writing a byte
comparison between backends into a gate would be writing a bug report with a scheduled delivery date. Note the
asymmetry with section 4, which *does* hold half the harness's comparisons to the bit: that is possible only because
the harness feeds both sides identical inputs, which an encode does not do after its first round.

With the CUDA kernels built without fused multiply-add (`NNTC_CUDA_FMAD=OFF`, the default since stage C1) the first
half of that reason is gone, and most encodes do come out byte-identical (7.4). The arm therefore REPORTS byte
identity per case, because a change that loses it on a case where it held is worth seeing; it still does not assert
it, because the reduction orders differ by design (0a) and a case like 7.4a's is a correct port.

---

## 8. Risks and unknowns, and how each one is retired

| # | risk or unknown | how it is retired | fallback if it cannot be |
|---|---|---|---|
| 1 | the dispatcher silently changes what the CUDA encoder computes | it cannot reach inside the encode: no kernel file, no header a kernel reads, and no line of the round loop is edited (2.1). Proved by 2.5's hash arm, and measured by 2.4's invariant -- both gates on both toolchains, a `cmp` against `v1.1.28-woa-tested` over twenty-six command lines, and a log diff with timings stripped | revert the one commit; the five-commit split exists so that a failure names its own cause |
| 2 | **a call site is missed and silently runs CUDA under `--backend cpu`** | the completeness guarantee of 1.7: in a build with no CUDA those symbols do not exist, so a missed call site fails to LINK and the linker names it. The CPU-only configure is a gate arm from step B, so the proof is re-run on every change | -- |
| 3 | `cpu_sample.h` loses a function it was getting from `<cuda_runtime.h>` | `sample.cuh` uses exactly one library function, `floorf`, twice. `<cmath>` does not guarantee `::floorf` at global scope, which is why the copy calls `std::floor` on the float argument -- exact, and portable to the Arm toolchains this backend exists for (1.8) | -- |
| 4 | **MSVC on ARM64 contracts the double multiply-add in `level1_value`**, so `verify_level1_on_grid` refuses to write a correct asset on the one machine that most needs the backend | 1.5's two-line sweep, moved into step A: compile `level1_value` / `level1_index` over a grid of arguments, assert the round trip, and diff the hex floats across MSVC x64, MSVC ARM64, gcc and clang. The gate's quantisation round-trip arm then fails loudly if it ever regresses | `/fp:contract-` on the MSVC branch of `CMakeLists.txt`, the exact counterpart of the `-ffp-contract=off` already on the gcc branch. One line |
| 5 | **the backends' fp32 features differ in the last bit and always will**, because the device contracts multiply-adds and the host does not | **not retired.** It is the reason decision 9 says decibels. What 3.2 retires is the *avoidable* part: with the reduction trees replayed, the summation contributes nothing, so any observed difference has exactly one cause -- which is also what lets section 4 hold half its comparisons to the bit | -- |
| 6 | a last-bit difference flips a discrete decision -- a level-0 argmin, a CG stop, a round count, a BC accept -- and the two backends' assets diverge by far more than the tolerance | partly retired by 7.4's three pinned conditions and by asserting `stop_reason`, `rounds`, `moved0` and `moved1` rather than printing them (1.9); and by keeping `CG_PROBE` at 4 so the two backends overshoot identically. Section 4's harness catches the same class earlier and per kernel | the tolerance is set from the measured corpus, and if a quantity's spread is wider than the table proposes, the bar is loosened **with the measurement recorded beside it** rather than the arm dropped |
| 7 | `k_bc_refine` has no CPU analogue: 64 threads, 35 KB of dynamic shared memory, six 64-slot tree reductions per channel, two barriers per Gauss-Seidel step | it is ported as a **serial per-block routine** parallelised across blocks, which is correct because every decision in the kernel is already thread 0's. **The reductions are NOT deleted**: `refine_reduce`'s 64-slot tree is replayed, because its sums gate the determinant guard, the nine-candidate argmin and the strict-decrease acceptance (5.6). Tested by the harness reporting the endpoints and selectors exact after every pass, and by the `--l0 bc8` gate arms | if the serial block routine is too slow, parallelise the assembly *within* a block over its 16 texels -- but only with the same fold order, and only with a measurement saying it was needed |
| 8 | the four-colour independence argument is wrong somewhere, and a colour pass races | it is not an argument this plan makes: `solve_level0.cu`'s own header proves it (class stride 2 against a bilinear footprint of 1) and the CUDA kernels already rely on it. **Section 3.4 shows it per kernel rather than asserting it once**, and it found the one exception -- `k_level0_search`'s `beta` accumulation from a same-colour neighbour two texels away -- and its fix. The ThreadSanitizer arm is the machine check, and `determinism_check` under `--backend cpu` is the second | -- |
| 9 | determinism breaks because a chunk count was derived from the thread count | rule 1 of 3.1 is a one-line review rule (`Pool::run`'s first argument may not mention `threads()`), and 7.3's six-count arm is the measurement. **`-j1` is the adversarial one**, because it is the only different code path; `-j64` covers more workers than chunks | -- |
| 10 | the file-static mutable globals -- `g_ridge_tally`, `g_objective_ms`, `g_objective_calls` -- become inconsistent or are touched from a worker | under decision 4 each backend has its own, reached through `be::`, and the harness compares them; under `--backend check` both are live and both are read, which is exactly what the tally comparison of stage C4 wants. **No solver state is ever written from a worker**: workers execute kernel bodies and nothing else (3.4) | -- |
| 11 | **the fixed-grid replay is disproportionate on deep planes** -- 256 chunks and a 256-slot tree to reduce a few thousand values, several `Pool::run` calls per CG iteration, up to 200 iterations, six or seven planes. The 2 to 3x estimate came from base-plane microbenchmarks and does not carry it | measured in **stage C1, before any CG kernel is written** (1.2, 3.2), alongside `Pool::run`'s own cost | a compile-time **size threshold** below which the chunk loop and the tree run inline on the calling thread: identical arithmetic, and a function of the plane's size rather than of the thread count, so determinism holds. If `Pool::run` itself is the cost, a persistent spin-then-park barrier inside `pool.h`; if even that is not enough, batch the CG's element-wise kernels into one `run`, which is scheduling and not arithmetic |
| 12 | memory: the CPU backend holds the same buffers the device does -- 605.6 MB for a 1024x1024 four-texture material -- plus per-worker scratch, on machines with far less than a 5090 | `device_memory` reports the same number under the same name, **and under `--backend cpu` the banner names the total before anything is allocated** (1.11), which the CUDA path cannot do. An allocation that fails exits 1 with an ERROR line naming the bytes and what they were for | if a material does not fit, that is a property of the layout and not of the backend, and the answer is the same one a small GPU gets |
| 13 | the Arm laptop, the Intel laptop and the AMD Linux box have never built the encoder, because until now they could not | stage D builds and runs it there, and 7.1 is why that is the plan's largest deliverable. The cheap arms run in minutes on any of them | a compiler difference that changes a result shows up as a tolerance failure against the desktop's CPU run, which is a *third* comparison this plan gets for free and should record. **1.12 first**: check whether the command line named a `--mip-filter`, because the vendored resizer's SIMD path differs by ISA and that is the source chain, not the port |
| 14 | **the copies drift from their originals** -- about 1160 lines (1.8), and the drivers, which are transcribed rather than copied and are maintained twice | for the copies: the hash arm means the originals cannot change without a visible commit, and section 4's harness holds most of the copied code to the BIT, so a drifted copy fails the first time it runs. For the drivers: the cross-backend arm, run on every change. Each CPU file carries a comment naming its CUDA twin by file and line | this is the plan's standing cost and it is not retired, only paid. If the drift ever bites twice, the answer is to revisit decision 1 deliberately rather than to loosen a tolerance |
| 15 | the gate under `--backend cpu` takes too long to be run | measured in stage D. If over about twenty minutes, a named subset runs and the rest is **recorded as skipped**, never as run | -- |
| 16 | scope creep: somebody adds SIMD, or turns the gather into a scatter, or threads the mip build | section 9 records each one with the number that was measured for it, so the temptation is answered with an arithmetic fact rather than with a rule | -- |
| 17 | `std::thread::hardware_concurrency()` returns 0, or the machine has one core | `-j` clamps into `[1, 1024]`; `-j1` runs the chunk loop on the calling thread and builds no pool at all, which is also the adversarial arm of 7.3 | -- |
| 18 | the harness's mirroring is itself wrong, so it reports `ok` on two implementations that disagree | it uses the fourteen transfer functions, which are the first thing written and the first thing it checks: stage C1's gate is that every transfer round-trips exactly on both backends before a single kernel is compared. And the harness is not the only instrument -- the within-backend structural arms (7.2) and the two brute-force twins do not use it at all | if the mirror is ever suspected, the cross-backend arm of 7.4 is the independent second opinion: it compares whole encodes and shares no code with the harness |

---

## 9. What is explicitly out of scope

Each of these is a real option, and each is recorded with its number so that the decision not to take it is a decision
and not an oversight.

* **SIMD of any kind.** No SSE, AVX, NEON, no `#ifdef` per instruction set, no compiler-specific vector types.
  Measured: hand-written SSE 4.1 was worth about **1.00x** against scalar on the dominant kernel, while threading was
  worth **14 to 16x** on the same loop. The kernels are bound by double multiply-adds and by stores, not by the width
  of a lane. And the fleet includes a Windows-on-Arm Snapdragon where SSE does not exist, so an intrinsics path would
  be a second implementation to maintain for the platform that needs this backend most. The owner: "nothing fancy,
  it's not worth it."
* **Restructuring the gather assemblies into a scatter.** The CUDA kernels recompute each site once per touching texel
  -- up to four times -- to avoid atomics. A CPU thread owning a band of rows could compute each site **once** and
  scatter it to the at most four texels it touches, and the study measured **4.4x** on that loop. It is not done here,
  for two reasons and one note. It is an algorithm-shaped change: it changes which additions happen in which order,
  which is exactly what decision 6 forbids. And `docs/RESULTS.md` 1.3 already rejected it on the GPU, for determinism
  (a shared-memory scatter needs atomics) and for memory (a two-pass gather needs 1.3 to 3.5 GB of per-site buffer on
  a large material). The note is that **neither of those two objections applies to a CPU with a static row-band
  partition** -- a band owner needs no atomic and no global site buffer -- so this is a genuine future optimisation
  with a measured 4.4x on 23 % of the encode, and it should be revisited **after** the faithful port is measured and
  gated, never instead of it.
* **Threading the host-side mip build and the file I/O.** Measured at **18 %** of the encode and single-threaded today
  on both backends. Threading it would help the CUDA path as much as the CPU one, which is precisely why it is a
  separate piece of work with its own risk to the byte-identical invariant. Noted, not planned.
* **Making the CPU path the default.** `--backend auto` prefers CUDA wherever a device is found. A build or a machine
  without one gets `cpu` and says so at configure time, in the banner and in the report.
* **Changing anything the CUDA path computes.** Not one kernel, not the round loop's structure, not a default. The
  only edits are the three classes of 2.1, and the eight kernel-side files are hashed.
* **Panelling block (a) as a rank-k update** (the study's other restructuring experiment). Unnecessary:
  `k_ls_accumulate` is *already* panelled -- it stages 256 sites and then walks the entries -- so the CPU port
  inherits the register-friendly shape without doing anything.
* **Accumulating anything in float that the device accumulates in double.** `docs/RESULTS.md` 1.3 already measured and
  rejected this for block (a): about eightfold on that kernel, at the price of a relative error in the normal matrix
  ten times larger than the ridge that is the only thing keeping a constant feature from making the system singular.
* **A third backend.** No OpenCL, no SYCL, no HIP, no Metal. The dispatcher would support one; this plan does not plan
  one.
* **Extracting the shared code instead of copying it.** Deliberately reversed since the first draft, and recorded
  here rather than silently dropped: the extraction was the one change in the draft that could alter what the CUDA
  encoder computes, and it bought about 1160 lines of duplication back at that price. Section 1.8 is the trade and
  section 4 is what pays for it.

---

## 9a. Deferred decisions, recorded so they are not lost

Two proposals raised during step B and parked by the owner so the work stays on the port. Neither blocks it.

* **`-fmad=false` on the CUDA build. DONE after stage C1:** `NNTC_CUDA_FMAD`, default OFF at the owner's
  request, ON reproduces every pre-option CUDA asset byte for byte (verified on 20 cases). Measured cost of OFF: 4 to
  23 per cent more wall time per encode, typically 10 to 15. Section 7.4 has what it revealed about the encoder's
  sensitivity to rounding, which is the more important result. The original proposal follows.
  **`-fmad=false` on the CUDA build.** nvcc contracts fp32 multiply-adds by default and the C++ side does not, which is
  the one difference the reduction-tree replay cannot remove. A CMake option that builds the kernels with
  `-fmad=false` would remove it too, and could make the backends agree to the bit on the default chain - which
  would let the cross-backend arm assert byte identity instead of a decibel tolerance. It is a compiler option only:
  no source changes, the kernel hashes still pass. What it costs: the CUDA assets change in their last bits (so the
  byte identity to `v1.1.28-woa-tested` and the tracked logs' exact reproduction are given up, deliberately, once),
  (the README makes no byte-identity claim about the shipped `examples/`, so they need not be regenerated),
  and some GPU time, which is **not yet measured**: a `-fmad=false` build exists at `out_fmad/build` for that one
  measurement. The two places that must agree host-versus-device (`model.h:level1_value`, `refine_bc.cu:bc0_value`)
  already use `__dadd_rn`/`__dmul_rn` explicitly and are unaffected either way. If adopted, the byte-identical tests
  run with the option set consistently on both sides of every comparison. Decide after stage C7, when the harness
  can say how close the backends are without it.
* **Hardening `verify_level1_on_grid` so it cannot cry wolf.** It refuses on exact float inequality between a stored
  level-1 value and the grid value of its index. Within one backend on a non-contracting compiler that is a value
  compared with itself and can only fire on a real solver bug; on a compiler that contracts one copy and not the
  other it would refuse a correct asset. The proposed fix uses the grid as the tolerance: refuse only when the value
  re-quantises to a different index, and demote a last-bit mismatch to a WARNING that names the likely cause. The
  `assert(on_grid)` just before the graceful error aborts a Debug build before the message prints and should go. The
  owner's principle: refuse only on an explicit, obvious solver bug. A sweep of the encoder's other refusals for the
  same pattern goes with it. This is host code (`export.cpp`, `main.cpp`), not a kernel file.

## 10. References

**In this tree**

* `src/model.h` -- the seam: the representation, the grid (`level1_value`, `level1_index`), and the 46 host-callable
  entry points enumerated in section 1.3. Plain C++17, shared by both backends unchanged
* `src/device.cuh` -- `DeviceModel`, the fixed-grid constants, the stencil geometry, `plane_work` (whose unqualified
  `stencil_monomials` call is the reason for 1.7's paragraph), the fan-out and fan-in protocol. **Hashed, unedited**
* `src/sample.cuh` -- the sampling rule, the feature order and the site set. **Hashed, unedited**, and copied
* `src/solve_level0.cu:75-141` -- `k_level0_search`, the tap-merge rule, the shadowed `dx`/`dy`, and the one race fix
* `src/refine_bc.cu:136-147, 305-395` -- `refine_reduce`'s 64-slot tree and the three decisions it gates
* `src/main.cpp:1699-1703, 2065-2072, 2941, 2958, 2965-2967` -- the refusal that becomes the fallback, the round
  loop's two stops, and the report lines that name the device
* `CMakeLists.txt:52-68, 118-137` -- the CUDA probe and its messages, and the two floating-point branches
* `docs/DESIGN.md` section 3 -- the three block solves and the round loop, which the port must reproduce exactly
* `docs/DESIGN.md` section 6 -- determinism: why every reduction is order-independent by construction, the one
  double atomic that is not, the resizer's ISA-dependent path, and the Windows-against-WSL measurement that
  section 7.4's tolerances are anchored on
* `docs/MATHEMATICS.md` section 3 -- block (a)'s ridge ladder, `Q`'s own rounding bound, and the near-singular image
  where the rungs are 7 dB apart
* `docs/MATHEMATICS.md` sections 9 and 10 -- the invariants a change must not break, and what the gate asserts
* `docs/RESULTS.md` section 1.3 -- where a round goes, and the two optimisations already considered and declined
* `tests/run_checks.py` -- the gate; `determinism_check`, `ridge_tally`, `ill_conditioned_checks`, `objective_checks`
  and the fifteen byte-for-byte arms are the instruments of sections 2.4 and 7, and `out_asset` (line 62) is 7.1's
  one-line note
* `tools/dds_decode.py` -- the third, independent reader, and the arbiter when two encoders disagree
* `docs/VULKAN_VIEWER_PLAN.md` -- the plan this one is shaped after, and its section 6's lessons, of which numbers 1
  (test against what the target's users have), 4 (keep the byte-for-byte comparisons) and 7 (a verification has to
  vary what the change could depend on) are the ones this plan tries hardest to obey

**Not in this tree**

* `out_cpu_study/` -- the feasibility study's microbenchmarks (`bench.cpp`, `bench2.cpp`, `bw.cpp`) and its two encode
  logs. Gitignored, and deliberately so: it is a measurement that was made once, section 1.2 carries every number
  from it that this plan rests on, and section 1.2 also says which of those numbers do not carry to the deep planes.
