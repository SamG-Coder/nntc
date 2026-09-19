# The CPU port of the encoder: what we learned

This is the record of porting `nntc_encode` from CUDA-only to CUDA plus a portable CPU backend (tags
`v1.2.1-c1-harness` through `v1.3.4-errors`, September 2026), written as notes for whoever works on it next,
including a future session of the assistant that did most of the work. `docs/CPU_BACKEND_PLAN.md` is the design and
its history; this file is what that plan did not know before the work was done. Every number below was read from an
encoder's own log.

## 1. What the port is, in one paragraph

The CUDA kernels (`src/*.cu`, `device.cuh`, `sample.cuh`, `model.h`) are the reference and their computation was never
edited: `tests/kernel_hashes.txt` pins the files and the gate checks the hashes. (Host-side messages and error handling
in `init.cu`, `device.cuh` and `model.h` did change in `v1.3.3` and `v1.3.4`, with the owner's OK and the pins updated;
every output stayed byte-identical.) A dispatcher (`src/backend.{h,cpp}`) routes each
of the encoder's device calls either to the CUDA arm (`src/cuda_backend.cpp`) or to `src/cpu/`, a scalar C++17 +
`std::thread` transcription of each kernel and its driver: no intrinsics, no SIMD, same math, same decisions, same
edge cases. `--backend auto` uses CUDA when it can and otherwise falls back to the CPU with a WARNING; `--backend cuda`
asked for by name is an ERROR when CUDA is unusable; `-j N` sets the CPU threads (default: every core). The owner's
priorities, in order: faithful, robust, correct, zero data races; speed last.

## 2. The central lesson: the encoder is path-dependent, so judge distributions, not cases

The encoder makes thousands of discrete decisions per round (quantisation snaps, palette argmins, BC endpoint and
selector choices, stop tests). A last-bit difference anywhere upstream flips a few of them, and from then on the two
runs are different optimisation paths that settle in different optima of about the same quality.

The clean measurement came before any CPU code existed: the SAME CUDA code compiled with and without fused
multiply-add (`NNTC_CUDA_FMAD`). Nothing changed but the last bit of each multiply-add, and on a 20-case corpus 5
cases moved by more than 0.1 dB, the worst by -1.03 and +0.86 dB on two textures of one material and -2.32 dB on one
of its mip levels. Round counts and ridge tallies were identical; the changes went both ways.

What that means in practice:

* **Never judge the port on one case.** A single material 1 dB apart proves nothing either way. Judge the corpus:
  the mean signed per-texture difference (a bug pulls one way; path dependence scatters both ways), the worst case,
  and gross outliers (several dB, which path dependence never produced).
* **The owner's bar:** about 0.35 dB per texture and 0.5 dB per mip level is "close enough"; 0.6 dB on a material's
  average is worth chasing, but only if the fix is small.
* **Bit identity between CUDA and CPU is not a goal and must never be stated as one**, least of all to a helper agent,
  which will then try to rebuild the whole CPU backend around it. It happens to hold on most cases (below), which is
  a pleasant consequence, not a requirement. Determinism WITHIN a backend (same build, any `-j`, run twice) is a
  requirement, and it is a different thing.
* **A path-dependence difference is not a bug, but it can hide one.** That is the reason to understand every outlier
  even when the numbers are inside the bar.

## 3. Fused multiply-add

nvcc contracts `a*b + c` into one fused multiply-add by default; MSVC, gcc (with `-ffp-contract=off`, which the tree
sets on the host) and clang do not. So with nvcc's default every fp32 feature and decoder value differs from the CPU's
in the last bit, and every encode is a different path from round 1.

* `NNTC_CUDA_FMAD` (CMake, default **OFF**) compiles the kernels with `-fmad=false`. With it OFF most encodes come out
  byte-identical between the backends.
* `-DNNTC_CUDA_FMAD=ON` reproduces assets made before the option existed **byte for byte**. That build is the CUDA
  regression reference: it must stay byte-identical to a pinned pre-port build.
* `model.h`'s `level1_value` rounds its multiply and add separately on purpose (the writer's round trip and the
  device's snap must land on the same bits); a host compiler that fuses them breaks `verify_level1_on_grid`. Hence
  `-ffp-contract=off` on gcc/clang and `tests/fp_contract_check.py`. MSVC on ARM64 is the one host still to be checked
  on real hardware.

## 4. Reductions: the one mistake that mattered, and its fix

**The mistake.** After the owner said "gate on PSNR, not bits" (plan section 0a), the plan also retired replaying
CUDA's reduction trees on the CPU: each CPU reduction sums in its own chunk order. The owner never asked for that and
said so plainly when it surfaced: "why would we do the sum differently? This is a port." The plan read a statement
about how to JUDGE the result as permission to CHANGE the computation. A port does the same computation; only the
compiler's floating-point behaviour should differ.

**What it cost.** 3 of 20 corpus cases stopped matching, and one material (five textures, 2048x1024, `--c0 4 --c1 4`,
default `--init0 residual`) landed 0.4 to 0.6 dB lower on the CPU on average (-0.606 dB at the default ridge).

**How it was found** (plan 7.4a has the full measurements):

1. The per-kernel harness (`--backend check`) on that exact material: 0 mismatches, 0 flips. Every kernel agreed on
   identical inputs, so it was not a transcription bug.
2. The residual seed (`init0_residual_channel`) agreed only to 8.35e-9 relative on that material, against about
   4e-14 on small inputs. Its fp64 covariance feeds a Jacobi eigendecomposition, and with four level-0 channels the
   eigenvalues are nearly equal, so the eigenvector is ill-conditioned and amplifies a summation-order difference by
   five orders of magnitude.
3. The runs first differed in round 1: the level-1 grid snap moved 552130 texels on CUDA and 552097 on the CPU.
4. The test that separated "biased solver" from "different seed": `--init0 luma`, whose seed is bit-identical on
   both backends, over seven ridge values: CPU better in 3, worse in 4, mean -0.094 dB. No bias.
5. The test that showed the seed was the cause: six ridge values with the residual seed. CUDA's mean wandered across
   basins (39.08 to 39.65 dB); the CPU's stayed in one (38.97 to 39.01) - because the ridge acts after the seed, so
   every CPU run started from the same slightly different seed.

**A measuring mistake worth remembering:** a sweep of the round budget (16 to 24 rounds) had the CPU lower every time
and looked like evidence of bias. It was not: those runs share their first rounds, so they share the fork, and are one
sample, not nine. To get independent samples, change something that acts from the first round (the ridge, the seed
mode), not something that acts at the end.

**The fix** (`v1.3.1-cpu-parity`, commit `3788a31`): `cov_partials` in `src/cpu/cpu_init.cpp` now sums in exactly
`k_cov_partial`'s order - 512 blocks of 256 lanes, lane `x` of block `b` taking texels `b*256 + x`, then every
`512*256` further; the 256 lanes halved into one as the shared-memory tree does; the host adding the block partials in
index order, plane by plane with the plane's weight. One function, +31/-25 lines, the kernel untouched. The products of
two floats in fp64 are exact, so nvcc's double-precision contraction cannot differ from the CPU's here.

**The result:** at the default ridge -0.606 dB became +0.092 (worst texture -0.25); over the six ridges the mean went
from -0.405 to +0.200 dB, the CPU now landing in CUDA's basins some of the time and better at others. The 4096x4096
five-texture material went from mean -0.19 to +0.65 (worst texture -0.27). Byte-identical corpus cases rose from 17 of
20 to 19 of 21.

**An earlier, broad attempt was abandoned:** a helper agent replaying every reduction tree in the CPU backend touched 7
files (+316/-191). The owner rejected it on size: "just do the MINIMAL fix". It is kept out of the tree (a stash and a
patch file in the working copy, not used). The lesson generalises: find the ONE amplifier and fix it there.

**Rule for next time:** replay CUDA's exact summation order wherever the sum feeds something ill-conditioned (an
eigendecomposition, a near-singular solve) or a hard decision on a near-tie that the whole run then depends on. Where
the sum only feeds a smooth quantity (the objective's reported `E`, the normal equations of a well-conditioned solve)
a chunked order is fine and the corpus shows it: those reductions still use chunk order and the backends agree to the
bit on most cases anyway.

## 5. Determinism across thread counts

Zero data races was an absolute requirement, and thread-count independence follows from three rules the CPU backend
keeps everywhere:

* the number of chunks a kernel is cut into is a function of the problem size only, never of the thread count;
* every chunk writes only its own outputs (texels, blocks, partial sums);
* partial results are folded in chunk-index order after the join, never as threads finish.

The pool (`src/cpu/pool.h`) partitions chunks statically, with no work stealing, and `-j1` goes through the same code
path as `-j32` (no inline fast path, so there is one code path to trust). ThreadSanitizer runs in the gate (WSL,
`nntc_cpu_check` at several thread counts plus two encodes) with zero reports. A failed `std::thread` creation (a
process thread limit, an extreme `-j`) is reported and exits 1 rather than aborting silently (`v1.3.1`).

## 6. How the port was tested, and what each layer is for

| layer | what it catches | notes |
|---|---|---|
| per-kernel harness, `--backend check` | transcription bugs: wrong index, bound, tie-break, colour | feeds both sides identical inputs, so no path drift; tolerances per output type, exact on integers |
| kernel hash check | an accidental edit of a CUDA kernel | 9 files pinned |
| cross-backend corpus (scratch driver) | bias, gross errors, outliers | 21 cases incl. a 4096x4096 five-texture material; judged on the distribution |
| wide set | inputs the corpus lacks | grayscale, palette with alpha, RGBA, non-square, 64 px to 3000x1480 |
| random torture materials | layout and content extremes | 24 materials from `tools/crop_set.py`: 1-6 unrelated textures, flat, edge-smeared and non-multiple-of-4 sizes |
| determinism arms | scheduler-dependent results | CPU repeat, `-j1` vs default; must be byte-identical |
| CUDA regression | the port changing CUDA | `NNTC_CUDA_FMAD=ON` build byte-identical to a pinned pre-port build |
| ASan + UBSan, WSL, Debug | out-of-bounds, use-after-free, leaks, UB | clang and gcc; see section 7 |
| Windows Debug | asserts, `/RTC1` | see section 7 for its limit |
| release gate | everything the tree tracks | CUDA and `--backend cpu`, Release and Debug |

**Random torture materials.** `tools/crop_set.py` crops (centred) or edge-extends 1 to 6 PNGs to one size and writes a
source-material JSON, which turns any pile of unrelated images into a maximally decorrelated material. Include
degenerate textures on purpose: a 1x1 source extended to a whole flat texture (encodes at 100 dB), 64x64 sources
smeared across 1280x720, grayscale and RGBA mixed with RGB. `--gray` and `--gray-first` make grayscale materials.

**What the layers measured (CPU against CUDA, default build, `v1.3.1` onward):**

| set | cases | byte-identical | worst texture, CPU minus CUDA |
|---|---|---|---|
| corpus (incl. the 4096x4096 material) | 21 | 19 | -0.27 dB (mean +0.07) |
| wide set (materials at defaults, both examples, 18 single images) | 23 | 18 | -0.01 dB (mean +0.01) |
| 24 random materials, `--c0 4 --c1 4` | 24 | 23 | -0.01 dB |
| 24 random materials, `--c0 3 --c1 4` | 24 | 24 | 0.00 |
| 24 random materials, `--c0 2 --c1 4` | 24 | 24 | 0.00 |
| synthetic sweep: 90 tiny/flat/dot/noise images x 6 layouts | 540 | 535 | -0.13 dB; 0 failures on either backend |
| `-j1`, `-j4`, `-j8` against the default thread count | 24 runs | 24 | - (and 23 of 24 also identical to CUDA) |

The CUDA regression check: the `NNTC_CUDA_FMAD=ON` build of the current tree was byte-identical to the pinned pre-port
build on all 44 corpus and wide-set cases. The release gate passes on six configurations: Windows Release on CUDA and
on `--backend cpu`, Windows Debug, WSL Release on CUDA and on `--backend cpu`, WSL Debug (the WSL runs add the
ThreadSanitizer arm).

## 7. Sanitizers and Debug builds

* **Run sanitizers in WSL (Linux) only.** Build Debug, CPU-only, with
  `-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -g -O1` in both `CMAKE_CXX_FLAGS` and
  the linker flags, in a build directory of its own.
* **Prove the sanitizer is live before trusting a clean run.** `main()` has a self-test compiled only with
  `-DNNTC_SANITIZER_SELFTEST=N`: 1 writes one past the end of a heap array, 2 one before the start, 3 overflows a
  signed int. Build each, confirm the run dies with the sanitizer's report and exit 1, then rebuild without it. Both
  compilers caught all three; gcc reports case 1 through UBSan's object-size check before ASan sees it, clang through
  ASan. Either is fine.
* **A Debug build announces itself:** the encoder prints `DEBUG build` as its first line when `NDEBUG` is not defined,
  as the Vulkan viewer already did - even under `--quiet`, which the gate's quiet check had to be taught (the first
  Debug gate after the banner failed on exactly that).
* **The Windows Debug encoder is weaker than it looks.** It links the RELEASE C runtime (`CUDA::cudart_static` is
  built against it, and mixing runtimes fails to link), so asserts and `/RTC1` are live but MSVC's checked iterators
  are not. The WSL sanitizer builds are the real memory check.
* Measured: 13 CPU encodes at 512x512 (1 to 6 textures, five latent layouts, palette mode, no mips, `-j1`) ran clean
  under clang and gcc ASan+UBSan and under Windows Debug, and all three gave identical PSNRs on every texture.

## 8. Speed and memory

The CPU backend is about 5 to 16 times slower than an RTX 5090 on the corpus with 32 threads (for example 356 s against
67 s for the 4096x4096 five-texture material; the 64x64 test image is only 1.7 times slower). A 4096x4096 five-texture encode needs about 10 GB of RAM. The owner accepted both: the
CPU backend exists so that the encoder works without an NVIDIA GPU, not to compete with one.

## 9. Grayscale inputs: an encoder weakness the port's testing surfaced

**Found by the CUDA regression check, not by the port.** Against the pre-port build, the default CUDA build
(`NNTC_CUDA_FMAD=OFF`) lost 5.03 dB on one image: `bik78`, a 575x445 photo stored as RGB but exactly grayscale. The
FMA-ON build gave 48.43 dB, the default 43.40, the CPU 43.91. Not a porting bug: a last-bit difference picked a
different optimum.

**Root cause** (an agent's diagnosis, every number from logs):

1. Nothing detects R = G = B; a gray PNG loads as three identical channels and `nout` is always 3 per texture.
2. `--init box` sets level-1 channel j to the block mean of source channel j, so a gray FIRST texture starts level 1
   with two or three identical channels (the fourth is a constant). Block (a) then has exactly collinear columns and
   block (b) a null direction; only the ridges and rounding settle them (the final decoder carried weights +0.45 and
   -0.20 on two channels correlated at 0.992).
3. The arbitrary choice sets how strongly the decoder amplifies level 0, and the BC5 pack's error is scaled by that
   gain: 1.1 dB apart before the pack became 5 dB after it. The effect is chaotic, not two basins: every ridge, round
   budget or range policy tried gave a different gap, up to 8.6 dB.

**What did not work** (measured before choosing): replacing the duplicate channels with a constant offset (still a
linear copy), with zero (snaps to a constant, collinear with the bias), with fixed-seed noise (stable across builds but
+-5 dB image to image), `--init pca` for gray inputs (helped 3 of 8, hurt 2), a larger ridge (new swings). Best-of-N
over those would have been safe but needs several encodes per input.

**The fix** (`v1.3.3-grayfirst`, host-side in `main.cpp`, no kernel change): the box init takes the source channels in
order with every exact copy of an earlier channel moved behind every distinct one, and recomputes level 1 on the host
only when that changes the SET of channels it starts from. Everything else is byte-identical by construction (the host
path does not run). A lone gray texture has nothing to promote and is unchanged, accepted as an outlier. Measured over
68 materials x 4 layouts x both CUDA builds: 0 untriggered case changed; gray-first materials +1.3 to +1.55 dB on
average, about 330 of 400 triggered runs better, the result nearly build-independent (177 of 200 same sign on both
builds); the losses sit mostly at `--c1 2`, where level 1's two channels now go to two different textures. In
`--l0 palette` mode (3 bits at 2+4 and 3+4, 4 bits at 4+4, the same 68 materials, default build) the gain is larger:
140 of 147 triggered runs better, 4 worse by more than 0.1 dB, about +2.1 dB on average; all 57 untriggered runs
byte-identical.

**Lesson:** exact duplicates in the inputs of a least-squares fit are a rounding lottery. Where the fix can be exact
and triggered only by exact equality, it can be proven not to touch anything else.

## 10. Never cry wolf: errors, warnings and defaults

An audit of every exit, refusal and numerical guard for valid PNGs with usual options found that no numerical check
refuses an encode (every guard falls back or skips), no NaN or Inf reaches a file, and the CPU backend's guards match
CUDA's constant for constant. What it did find, all fixed in `v1.3.3`:

* **An ERROR on a successful run:** a pre-8.0 GPU under `--backend auto` printed `ERROR: device 0 ...` and then
  succeeded on the CPU. Now a WARNING when falling back (naming the device and its compute capability), an ERROR only
  when `--backend cuda` or `check` must exit. The rule the owner set: a fallback is a WARNING, a failure that exits is an
  ERROR, and both say what happened and why.
* **The optional PNGs gated the asset:** they were written first, and a PNG failure exited 1 with no asset. The asset
  is now written first and a PNG failure is a WARNING. And `--png` now defaults to 0 (the owner found the PNG flood
  annoying); the gate asks for PNGs where it reads them.
* **Cosmetic:** `packing psnr inf dB` became 100 dB, and the report no longer claims `--q1-start 0` when the loop
  simply ended before the freeze.

`v1.3.4` then made every failure say what happened: every CUDA call checked (teardown failures are WARNINGs); GPU out
of memory is one ERROR with the allocation, what is already allocated and the free memory, and no CPU fallback (the
owner's choice: rare, and a clear message is enough); `main()` catches `bad_alloc`, `length_error` and any other
exception, and a CPU worker's exception is rethrown on the calling thread after the join; backend refusals are said
in plain words ("no NVIDIA GPU was found"); inputs over 16384 on either axis are refused from the header before
decoding. Tested with real out-of-memory on both hosts (a WSL `ulimit`, a Windows job object, a 16384x16384 image on
the GPU - the Windows driver spills into system memory first, about 64 GB here). The gate gained 200 degenerate
pictures (ten kinds, 1x1 to 64x4, both level-0 modes) that must exit 0, print no ERROR, nan or inf, and clear a sanity
floor of 8 dB (40 dB for a flat picture): the check is "does it work at all", not quality.

## 11. Working notes (process, for the next session)

* **Minimal changes.** The owner rejects broad changes to fix narrow problems. Scope a fix to the one function that
  is the cause, show the diff size and before/after numbers, and let him decide before committing.
* **Do not overstate goals to helper agents.** "Bit-identical" when the goal is "close enough in PSNR" sends an agent
  off to rebuild a subsystem.
* **Headless only.** Never open a window on the owner's desktop without being asked; the viewers' `--shot` mode
  renders to a hidden window. When he asks to see a viewer, run it from a copy of the executable so a rebuild is not
  blocked by the running file.
* **Shell quoting.** In this environment bash heredocs mangle backslashes and `\n` escapes; write edit scripts to
  files and run them.
* **Out directories** (`out_*/`) are ignored by git; results the owner inspects go outside the tree or in `out_*`.
* **Numbers come from logs, never estimates.** When a claim about a file is wrong (the size of a material, which
  texture index is which), correct it immediately.
* **Tracked files:** CRLF, no absolute paths, no work-tracking markers.

* **Keep `--l0 palette` working.** It is not the default, but it is the planned basis of a custom "supercompression"
  format; test every change on it too, not only on bc8.
* **Frozen files are frozen for the math.** The owner allowed host-side message and error-handling edits in `init.cu`
  (with `tests/kernel_hashes.txt` updated) when that was the simple fix; never a kernel's computation.
* **Pin binaries for long runs.** Copy the encoder into an `out_*` directory before a long measurement so that a
  rebuild by anyone mid-run cannot change what is being measured.
* **`tests/run_checks.py` has no `--help`**: any other argument just runs the whole gate.
* **Never stop a gate mid-run.** A viewer comparison interrupted by a task stop reads as a gate failure; it cost a
  false alarm once. Let it finish, or rerun it whole.
* **Call `wsl` from PowerShell, not Git Bash**: Git Bash rewrites `/mnt/c/...` arguments into Windows paths.
* **The two gates share `out/`.** Run the Windows and WSL gates one after the other, never at once.

## 12. Open items

* `docs/CPU_BACKEND_PLAN.md` still has sections written before section 0a (3.1 rule 5, 3.2, 5.5, 5.6, the C7
  heading, one row of table 3.3) that describe the replay or the `-j1` fast path; 0a supersedes them, and 7.4a's "open
  decision" was taken in `v1.3.1` (section 4 above). Two comments say the dispatcher has 45 functions; it has 46.
* A lone grayscale texture remains rounding-sensitive (section 9); so do two gray textures first, whose channel set
  the rule leaves unchanged. The deeper cause - nothing in the rounds prices the decoder's gain on level 0 before the
  BC pack - is untouched.
* One mip level of one material (`--c0 4 --c1 4`, 2048x1024) remains 0.68 dB lower on the CPU while all its textures
  are within 0.25 dB; it is the most rounding-sensitive case in the corpus.
* MSVC ARM64: run `tests/fp_contract_check.py` on real hardware.
* A HIP backend for AMD would slot into the dispatcher the same way the CPU backend did.
