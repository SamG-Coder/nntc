# RESULTS.md - what the NNTC encoder costs and what it produces

Every number on this page comes from a log under `out/`, named in the row that quotes it. Nothing is estimated and
nothing is rounded by hand. One machine throughout: an RTX 5090, Windows, Release build, CUDA 13.1.

The encoder's own report is the source for the quality figures; `tools/dds_decode.py`, which shares no code with it,
reproduces them from the `.dds` and the `_nntc.json` descriptor alone and is what the checks below assert against.

---

## 1. Performance

### 1.1 What was changed, and why

The starting point was measured, not guessed. At the previous stage a 752x1124 image with a seven-level chain spent
**2.011 s**, of which block (b) alone was **1393 ms** - four times the base plane's own cost for a third of its texels.
That ratio is the tell: the chain was not paying for work. It was paying for *fixed cost per plane per round*, and the
per-plane report said where - a plane whose level 1 is 2x4 texels was costing as much as one of 188x281.

Three things were done, in this order, each measured on its own.

**(a) The fixed cost per plane per round.** The round created and destroyed CUDA events per plane per block, reduced the
stencil diagonal to the host with three synchronous device-to-host copies, and ran the conjugate gradients with three
synchronous copies *per iteration* - about a hundred stalls per plane per round, each of them a command-buffer flush.
Now: the events are created once for the run; every reduction ends in a single-threaded kernel that writes the answer
into the plane's own scalar row on the device, where the next kernel reads it; the ridge, the iteration's inner
products, the movement probe and the sweeps' moved count never leave the GPU; and the residual - the one genuinely
host-side decision - is looked at once every four iterations instead of three times per iteration, which can postpone a
stop by at most three iterations and never miss one. Each plane then got its own stream and its own block (b)
workspace, and a block is issued for every plane at once and waited for exactly once.

**(b) The stencil assembly.** It is a gather: each site is visited by each of the four texels it touches, and what was
recomputed each time was `G^T diag(cw) G`, a `C1 x C1` matrix in double. But `G` is affine in the site's level-0
sample, so that product is a **quadratic form** in it and is carried by six fixed `C1 x C1` matrices that depend on the
decoder alone (`P_0 = A^T C A`, `P_i = M_i^T C A + A^T C M_i`, `P_ii = M_i^T C M_i`, `P_ij = M_i^T C M_j + M_j^T C M_i`).
They are built once a round on the host; what varies per site is six *scalars*. A texel now accumulates six weighted
sums per stencil slot and expands them against the six matrices once, at the end, for itself. The accumulator fell from
five `C1 x C1` blocks (80 doubles, which spilled) to five rows of six. The footprint band was also tightened from ten
pixels per axis to the nine that can actually reach the texel. `docs/DESIGN.md` carries the algebra.

**(c) Block (a).** The first thing tried was the obvious one: the block stages its sites entry-major in shared memory
with a stride of 256 floats, so a warp's reads of different entries at the same site all fall in one bank and are
serialised thirty-two ways. Padding the stride to 257 fixes that and **changed nothing measurable** (138.7 ms against
138.7), which is the answer to where the time goes: the walk is bound by its double multiply-adds, at a sixty-fourth of
the device's float rate, and not by its memory. The padding was kept anyway, with the measurement written beside it.

What did work is arithmetic. The normal matrix is symmetric and only its upper triangle is accumulated now, the host
mirroring it before the solve. That is a third fewer multiply-adds, and - the larger half of the gain - it brings the
entry count under one per thread (165 against 270 for a 256-thread block), so the walk is one pass in which every thread
carries an entry rather than two passes in which fourteen threads carry the second one and the rest wait. 143.9 ms to
85.2 ms.

### 1.2 Before and after, per block

`--png 0`, `--rounds 20`, otherwise the defaults. The three splits are each block's own span, measured with CUDA events
around the whole block (every plane of it, from the first launch to the last); the total is the encode's wall clock,
which also carries the objective passes, the CUDA context, the file I/O and the host's own work.

| image | layout | stage | total s | (a) ms | (b) ms | (c) ms | log |
|---|---|---|---|---|---|---|---|
| model10 752x1124, 7 levels | `c0 2 bits0 3 c1 4 bits1 8` | before | 2.011 | 146.4 | 1393.0 | 188.5 | `out/s7/log_before_model10.txt` |
| | | after (a) streams | 0.927 | 143.9 | 412.3 | 107.8 | `out/s7/log_stepa_model10.txt` |
| | | after (b) and (c) | **0.624** | **85.2** | **179.2** | **103.0** | `out/s7/log_after_model10.txt` |
| game2 1024x1024, 8 levels | `c0 1 bits0 4 c1 2 bits1 8` | before | 0.914 | 57.5 | 518.4 | 89.1 | `out/s7/log_before_game2.txt` |
| | | after (a) streams | 0.504 | 55.1 | 195.6 | 42.1 | `out/s7/log_stepa_game2.txt` |
| | | after (b) and (c) | **0.409** | **36.1** | **118.6** | **41.1** | `out/s7/log_after_game2.txt` |
| mb12 1024x1024, 2 textures | `c0 2 bits0 3 c1 4 bits1 8` | before | 2.889 | 198.2 | 1974.4 | 327.0 | `out/s7/log_before_mb12.txt` |
| | | after (a) streams | 1.463 | 195.3 | 706.4 | 197.1 | `out/s7/log_stepa_mb12.txt` |
| | | after (b) and (c) | **0.957** | **124.1** | **276.5** | **196.2** | `out/s7/log_after_mb12.txt` |

End to end: **model10 3.2x, game2 2.2x, mb12 3.0x**, at the same quality - model10 42.55 dB before and 42.54 after
(a hundredth of a decibel of summation order), game2 37.86 both, mb12 34.11 / 37.95 both.

The three steps separate cleanly, because each touched a different block:

| step | what it is measured on | before | after |
|---|---|---|---|
| (a) streams and no synchronisation inside a block | block (b), model10 | 1393.0 ms | 412.3 ms |
| (b) the assembly as a quadratic form | the base plane's assembly kernel, model10 | 14.199 ms | 5.451 ms |
| | block (b), model10 | 412.3 ms | 179.2 ms |
| (c) the symmetric triangle | block (a), model10 | 143.9 ms | 85.2 ms |

(The base plane's assembly reads 11.271 ms in the before log and 14.199 ms after step (a): the same kernel, now sharing
the device with six other planes' assemblies instead of having it to itself. 5.451 ms is against the 14.199.)

### 1.3 Where a round goes now

model10, `out/s7/log_after_model10.txt`, 20 rounds, 624 ms in all:

| part | ms | share |
|---|---|---|
| block (b), level 1 | 179.2 | 29 % |
| the objective, 63 passes | 112.6 | 18 % |
| block (c), level 0 | 103.0 | 17 % |
| block (a), the decoder | 85.2 | 14 % |
| everything else | 143.7 | 23 % |

Two of these were looked at and deliberately left alone.

**The objective, 18 %.** Three passes a round, so that the log carries the evidence that none of the three blocks
raised `E`, and run_checks asserts on exactly that. It is evaluated in double throughout - the sampling, the decode and
the accumulation - because it is the measurement the solver is held to and because the brute-force host twin behind
`--check` agrees with it to 5e-15, a bar that a float decode would miss by eight decades. Making it fast means making it
stop being that. It stays.

**Everything else, 23 %.** Mostly fixed: an encode of `tests/tiny.png`, whose blocks are trivial, still takes 0.15 s,
and model10 at `--rounds 1` takes 0.199 s against 61 ms of blocks. That is the CUDA context, the PNG decode and the
asset write. It does not grow with the rounds and it is 2.6 % of model34's encode rather than 23 %.

Two further steps were considered and **not** taken:

- *The assembly as a scatter or a two-pass gather*, removing the fourfold visit of each site outright. After step (b)
  the assembly is about 110 ms of model10's 624, and neither form gets all of it: a shared-memory scatter needs atomics,
  which would make the result depend on how blocks were scheduled - this encoder has no seed and no hash anywhere and
  every other reduction in it is deliberately order-independent - and a two-pass gather needs a per-site buffer that is
  1.3 GB on model34 at `c1 2` and 3.5 GB at `c1 4`. The gain is under 20 % of the encode for a real loss of clarity and,
  in one case, of determinism. Named here as the next thing to try if the encode ever has to be faster still.
- *Block (a) accumulating a block's partial in float.* It would be roughly another eightfold on that kernel, which is
  bound by its double multiply-adds and not by its memory. It was rejected: the normal matrix would carry a relative
  error of about 1e-8, which is ten times the ridge that is the only thing keeping a constant feature - level 0's second
  channel starts at zero everywhere - from making the system singular. A wrong answer in a near-null direction is not
  worth 6 % of an encode.

---

## 2. The ablations

The full table, with a log per run, is `out/s7/ablations.txt`; every arm is `--c0 2 --bits0 3 --c1 4 --bits1 8` with
nothing else changed, and every arm was run against what were THEN the defaults, `--k 4 --init pca --q1-start 3
--rounds 20 --mip-weight sqrt`. **The commands in those logs no longer run as they stand: the encoder is `nntc_encode`
now, where every log under `out/` still names it by the one-n spelling it had before the rename (the logs are records
and were not edited), and `--l0 palette` must be added to every one of them.** The logs also predate the descriptor's
`_nntc` suffix, so a log naming `model10.json` is the asset a run today writes as `model10_nntc.json`. `--bits0` is a palette-mode option and the default is now
`--l0 bc8`, which refuses it by name rather than ignoring it, so re-running an s7 or s7/cmp command means writing
`--l0 palette --c0 2 --bits0 3 ...`. The defaults have moved under these logs in other ways too (`box`, `pixels`, the
residual level-0 seed, the refined pack), so a re-run reproduces the arm's SHAPE and not its decibels. Two of those
moved because of what this table says: `--init box` and `--mip-weight pixels` are the defaults now (`src/options.h`),
and the arms below are the measurement that chose them. What it says:

**`--k 0 / 1 / 2 / 4`.** The centre PSNR falls as `K` rises - model10 43.40 / 43.11 / 42.92 / 42.56, game2 46.74 /
46.36 / 46.17 / 45.75 - and the time rises with it, 0.354 s to 0.674 s on model10. **This is the one arm where the
table must not be read as a ranking.** `K` is how many positions *between* texel centres the fit is held to, and the
centre PSNR is measured at the texel centres alone: a run at `--k 0` is being scored on exactly the sites it was fitted
to and on no others, so of course it wins there. Each arm's `sampled psnr` column is measured at its own site set and
cannot be compared across arms either. What `K` buys is behaviour under the hardware's sampler, which is the whole point
of the representation, and the only honest test is the viewer. **The four model10 arms were written out as complete assets
with their recon PNGs and a `--shot` each, to be judged by eye; the images themselves are not carried in the
repository, and re-running the four arms regenerates them.** The default stays `--k 4`.

**`--init box / pca`.** Nothing in it: model10 42.55 against 42.58, game2 45.76 against 45.83, and the same time. (This
is not the same comparison as the one in `docs/DESIGN.md` 3.4, which reads model10 box 41.98 against pca 41.94: that
one is at `--q1-range pct` with `--rgb-weights 9,11,1` and this one is a bare ablation arm, and neither number in the
DESIGN pair has a log in the tree. Both say the same thing, which is that the two inits split and neither is ahead.)
Twenty
rounds of exact block minimisation wash the initialisation out. `pca` is a shade ahead on both, and `box` is
nevertheless the default: three hundredths of a decibel do not pay for an eigenbasis in the explanation, and a run that
wants the other path loses nothing by asking for it.

**`--q1-start 0 / 3`.** The same quality - model10 42.55 both, game2 45.73 against 45.76 - and **twice the time** for
`--q1-start 0` (model10 1.252 s against 0.688 s, game2 1.068 s against 0.738 s), because a continuous plane has to be
solved by conjugate gradients every round while a frozen one is four sweeps. At 8 bits the quantised sweeps are
therefore free quality and half the cost; the margin they were built for is at 4 bits, which `docs/DESIGN.md` section
5.2 measures, and this arm does not re-measure it. `--q1-start 3` stays the default.

**`--rounds 5 / 10 / 20`.** model10 42.28 / 42.44 / 42.56 for 0.297 / 0.414 / 0.647 s; game2 45.51 / 45.66 / 45.75 for
0.319 / 0.443 / 0.728 s. Five rounds are within 0.3 dB of twenty for less than half the time, and ten within 0.12 dB.
The curve is flat enough by round 10 that `--rounds 10` is the setting to reach for when an encode has to be quick; 20
stays the default because it is still under a second.

**`--mip-weight sqrt / pixels / uniform`.** The clearest result on the page. `pixels` is **better than the default at
every level**: model10 43.17 against 42.56 at the base and 41.07 / 39.06 / 36.66 against 40.68 / 38.77 / 36.50 at M1-M3;
game2 46.07 against 45.75 and better at M1-M3 too. `uniform` is worse everywhere (model10 41.00, and 4.0 % of pixels
above 8 against 2.3 %), which is the failure the first default was chosen to avoid. `sqrt` was that default because it
was the recipe the existing good runs used; since `pixels` beats it at every level on both images, **`pixels` is the
default now**, and the banner says so on every run. Everything measured on this page, including section 3's comparison
table, was run under `sqrt` and is left as it was measured.

---

## 3. The comparison table

NNTC against the **evolution-strategy control**. That is the owner's earlier trainer, a separate program in a tree of
its own, which optimises the same representation by an evolution strategy instead of by exact block minimisation and is
what these layouts were originally measured on. Its side of the table is quoted from ITS runs, at 8000
iterations with `--sampler hw-mip --mip-mix 0.5 --mip-weight sqrt --sampler-k 4 --mip-min 8 --rgb-weights 9,11,1`, on
the same GPU. Those runs are not in this repository and the logs quoted here are only NNTC's own, which is the one
place on this page a number's source is outside the tree; it is named rather than hidden. Every
NNTC run here is `--rgb-weights 9,11,1` with the chain on and otherwise the defaults AS THEY STOOD when the table was
measured (`--k 4 --rounds 20 --q1-start 3 --init pca --mip-weight sqrt`; `box` and `pixels` are the defaults now,
section 2). The commands in `out/s7/cmp/` need `--l0 palette` added to run today, and the binary they name is
`nntc_encode` now, for the reasons section 2 gives. PSNR is the same definition on both sides: 8-bit, at the texel
centres, over the original extent, unweighted.

| image | layout | control dB / s | NNTC dB | sampled | mip M1 M2 M3 | worst p50/p90/p99/max | rounds | s | speedup | log |
|---|---|---|---|---|---|---|---|---|---|---|
| model10 750x1122 | `c0 1 bits0 4 c1 2 bits1 8` | 34.05 / 75.0 | 32.30 | 33.29 | 31.86 31.30 30.19 | 4/14/38/121 | 20 | 0.621 | 121x | `out/s7/cmp/log_model10_c1b4_c2b8.txt` |
| model10 | `c0 1 bits0 4 c1 3 bits1 8` | 37.96 / 78.0 | 37.40 | 39.61 | 35.33 33.65 31.32 | 1/8/19/66 | 20 | 0.596 | 131x | `out/s7/cmp/log_model10_c1b4_c3b8.txt` |
| model10 | `c0 1 bits0 3 c1 4 bits1 8` | 36.62 / 81.0 | **37.94** | 41.07 | 36.40 34.58 32.26 | 0/8/18/70 | 20 | 0.637 | 127x | `out/s7/cmp/log_model10_c1b3_c4b8.txt` |
| model10 | `c0 2 bits0 3 c1 3 bits1 8` | 39.17 / 123.0 | **40.89** | 43.66 | 38.48 36.49 34.22 | 0/5/14/65 | 20 | 0.738 | 167x | `out/s7/cmp/log_model10_c2b3_c3b8.txt` |
| model10 | `c0 2 bits0 3 c1 4 bits1 8` | 41.22 / 131.0 | **41.90** | 44.41 | 39.12 36.95 34.34 | 0/4/13/66 | 20 | 0.809 | 162x | `out/s7/cmp/log_model10_c2b3_c4b8.txt` |
| model15 1080x1080 | `c0 1 bits0 4 c1 2 bits1 8` | 33.27 / 100.1 | 31.88 | 34.84 | 30.56 28.00 25.17 | 7/14/25/73 | 20 | 0.868 | 115x | `out/s7/cmp/log_model15_c1b4_c2b8.txt` |
| model15 | `c0 1 bits0 4 c1 2 bits1 4` | 31.55 / 102.8 | 30.68 | 32.72 | 29.61 27.42 24.81 | 9/17/28/75 | 20 | 0.863 | 119x | `out/s7/cmp/log_model15_c1b4_c2b4.txt` |
| model34 2888x4320 | `c0 1 bits0 4 c1 2 bits1 8` | 38.53 / 906.6 | **38.59** | 39.55 | 37.80 36.38 33.76 | 2/7/16/76 | 20 | 5.470 | 166x | `out/s7/cmp/log_model34_c1b4_c2b8.txt` |
| game2 1024x1024 | `c0 1 bits0 4 c1 2 bits1 4` | 36.89 / 88.0 | 35.85 | 37.25 | 35.25 34.58 33.78 | 4/9/18/54 | 20 | 0.777 | 113x | `out/s7/cmp/log_game2_c1b4_c2b4.txt` |
| game2 | `c0 1 bits0 4 c1 3 bits1 4` | 38.59 / 93.0 | 37.71 | 40.11 | 36.61 35.55 34.39 | 3/7/12/52 | 20 | 0.774 | 120x | `out/s7/cmp/log_game2_c1b4_c3b4.txt` |

**The speed.** Between 113x and 167x, on every image and every layout, and the largest image is the largest margin:
model34 is 5.47 s against 906.6 s. The goal of section 0 of the plan - beat the control by a wide margin - is met by two
orders of magnitude.

**The quality, honestly.** It splits by capacity, and the split is consistent:

- With **two level-0 channels or four level-1 channels**, NNTC is ahead: +1.32, +1.72 and +0.68 dB on model10's three
  larger layouts, and +0.06 on model34.
- At the **smallest layout, one level-0 channel and two level-1 channels**, it is behind: -1.75 dB on model10, -1.39 and
  -0.87 on model15, -1.04 and -0.88 on game2.

The reading: block (b)'s step is exact for the level-1 plane and block (c)'s is exact for a level-0 texel, but the
alternation is a coordinate descent and it can stop at a point no single block can leave. The tighter the layout, the
more the answer depends on the two levels moving *together* - with one 4-bit selector channel and two colour channels
there is very little slack anywhere - and a stochastic search over all of it at once, which is what the control is, has
the advantage exactly there. With room in the layout the exact steps win and win quickly.

None of this is a verdict: the logs of every run are in `out/s7/cmp/`, and the assets and recon PNGs they name are
regenerated by re-running the command at the top of each log, with the binary's current name (section 2's note). The
last column of such a table is a verdict by eye.

**Device memory.** 41-90 MB for the 1K images, and **607.9 MB for model34** (2888x4320, nine levels) - the source chain,
both latents at every level, and block (b)'s per-plane workspace, which is the largest part and is the price of solving
every plane at the same time. Nothing near a 32 GB card's limit.

---

## 4. The checks

| check | result |
|---|---|
| `dds_decode.py --ref` on model10, game2, model15, model34 | worst max abs diff **1** at every level of all four -- the encoder decodes in fp32 and this reader in fp64, so a value within half a step of an 8-bit boundary may round the other way; nothing systematic survives it |
| `dds_decode.py --grid` (the fit grid against the sampled grid, from the descriptor alone) | worst \|published - sampled\| **1.7e-08** over every index of every channel, on the block-compressed asset and on the uncompressed twin of the same solve |
| `dds_decode.py --psnr-levels` against the encoder's report, model10 | M0 41.90, M1-M6 39.12 36.95 34.34 32.68 32.42 31.27, **equal to the report at every level** |
| `bc_check` on model10 | level 0 packs **losslessly** at 3 bits (max err 6.5e-06 / 255, packing PSNR 154.80 dB) |
| `nntc_view --shot` on model34, near (`--z -1.2`) | the full-resolution decode fills the quad: smooth skin tones, no blocking, no colour shift against the source |
| `nntc_view --shot` on model34, far (`--z -12`) | the whole 2888x4320 image on about 30x45 screen pixels, reading the deep levels: clean, no splotching and no aliasing sparkle |
| `nntc_view --shot`, the block-compressed asset against the uncompressed twin of the same solve (model10, 2 x 3 bits) | the raw level-0 view differs by at most **4 / 255** and the decoded view by **3 / 255**: this GPU evaluates the BC palette's interior with six-bit weights, and the two formats' grids themselves differ by 0.43 / 255 at 3 bits (`docs/FORMAT.md` section 7) |
| `python tests/run_checks.py` | green, including the timing line it now prints (not a gate) |

## 5. The Linux build

**Exercised on 2026-09-12** under WSL 2 on the same machine: Ubuntu 24.04.3, gcc 13.3.0, CMake 3.28.3, CUDA 13.3.73 from
NVIDIA's `wsl-ubuntu` apt repository (`cuda-toolkit-13-3`), the Windows driver reporting CUDA 13.0. The README's two
Linux lines configure and build `nntc_encode` unchanged; the only warnings are gcc's `-Wmissing-field-initializers` inside
the vendored `stb_image_write.h`, none in the tree's own sources.

| check | result |
|---|---|
| `tests/run_checks.py --build-dir build_wsl` under WSL | **All checks passed** (`bc_check` and the viewer skipped, Windows only) |
| `m1-m4` at the default layout, the Linux binary | `out/disclosure/log_m1234_default_wsl.txt`: the same 25.66 / 33.36 / 31.28 / 36.14 dB, `E shipped` 6.666616484e-04, 1.222 s |
| its `.dds` and `.json` against the Windows build's (`out/disclosure/log_m1234_default.txt`) | **byte-identical**, all three files |

So the determinism of section 6 of `docs/DESIGN.md` holds across compilers and operating systems **on the default
source chain**, not only across reruns: MSVC + nvcc on Windows and gcc + nvcc on Linux write the same bytes. It does
not hold once a texture asks for a filter: the vendored resizer chooses a code path from what the CPU supports, so a
`mitchell` chain agrees to the decibel and not to the byte. DESIGN 6 has the measurement and names the two logs.

Before that run the build had only been read for portability, and the reading stands: **nothing MSVC-only sits outside
a guard**: `/W4`, `_CRT_SECURE_NO_WARNINGS`, the static runtime property and `-Xcompiler=/W4` are all inside
`if(MSVC)`, whose `else()` branch is `-Wall -Wextra` for the host and the same through `-Xcompiler` for the device;
the viewer and `bc_check`, which are Direct3D 11, are the whole contents of an `if(WIN32)`. Everything else - the CUDA
language, `find_package(CUDAToolkit 12.8)`, `CUDA_ARCHITECTURES "80;86;89;90;120"`, `CUDA::cudart_static` and the
C++17 standard - is portable. The encoder's own sources use nothing platform-specific: the only headers outside the
standard library and CUDA are the two vendored stb files.

---

## 6. `--l0 bc8`: level 0 as a continuous 8-bit plane, at equal memory

Level 0 ships as BC4 (one channel) or BC5 (two), and a BC4 plane costs **4 bits per texel whatever the palette depth**:
a 1-bit level 0 and a 4-bit one occupy the same file. So at 4 bits the block format is being paid for and not used -- a
4x4 block holds two endpoints and eight values along the line between them, and the palette mode hands it sixteen fixed
levels to approximate. `--l0 bc8` hands it a continuous plane instead: level 0 is solved by the same sparse least
squares level 1 is solved by (`docs/DESIGN.md` section 3.5), quantised to 8 bits on a per-channel `lo/hi` grid frozen at
`--q1-start`, and BC4 / BC5-encoded **once after the last round**, after which level 1 and the decoder are refitted once
against the packed plane.

The comparison below is therefore at **equal memory on both sides**: `--l0 palette --bits0 4` and `--l0 bc8` produce
byte-for-byte identically sized `_lat0.dds` files (the `lat0 bytes` column), 4 bpp at `C0 1` and 8 bpp at `C0 2`.

**The tables of 6 and 6.1 are the state before the refinement existed: the pack made once and left alone.** Section 6.2
replaces that pack with one chosen by analysis by synthesis and is what the encoder does now; the numbers below are
kept as they were measured, because they are the baseline 6.2's seed-pack column is the successor of. They were also
measured before the residual level-0 seed (section 7.2) and the last free refit, so they are not comparable to 6.2's
figures run for run -- 6.2 says which of its own columns came from which binary.

Every number comes from a log under `out/s9a/`. The command each run made is the first line of its own log, under the
binary's old name (section 2's note); all of them are `--c1 4 --bits1 8 --rgb-weights 9,11,1 --mips 1` with the rest
of the defaults, and the palette arm adds `--bits0 4`. `psnr` is the report's own definition (8-bit, texel centres,
original extent, unweighted); `dds_decode M0` is the same quantity computed by the independent Python reader from the
shipped `.dds` + `_nntc.json` alone, which is the number that counts.

| image | C0 | level 0 | psnr | sampled | M1 | M2 | M3 | worst case | dds_decode M0 | lat0 bytes | time s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model10 750x1122 | 1 | 4-bit palette | 38.62 | 41.51 | 36.82 | 34.91 | 32.40 | p50 1 p90 7 p99 16 max 62 | 38.62 | 564500 | 0.65 |
| model10 | 1 | **bc8** | **39.49** | 40.95 | 37.75 | 35.92 | 33.55 | p50 2 p90 6 p99 15 max 59 | **39.49** | 564500 | 1.01 |
| model10 | 2 | 4-bit palette | 41.51 | 43.76 | 38.87 | 36.63 | 34.20 | p50 1 p90 4 p99 14 max 59 | 41.51 | 1128852 | 0.98 |
| model10 | 2 | **bc8** | **42.84** | 45.11 | 40.41 | 38.42 | 36.26 | p50 1 p90 4 p99 12 max 48 | **42.84** | 1128852 | 1.26 |
| model35 960x960 | 1 | 4-bit palette | 37.16 | 40.82 | 36.84 | 36.57 | 35.63 | p50 2 p90 7 p99 14 max 68 | 37.16 | 614588 | 0.73 |
| model35 | 1 | **bc8** | **40.62** | 43.03 | 40.46 | 40.28 | 38.74 | p50 1 p90 5 p99 12 max 78 | **40.62** | 614588 | 1.13 |
| model35 | 2 | 4-bit palette | 41.06 | 43.83 | 40.76 | 40.50 | 38.95 | p50 1 p90 5 p99 11 max 56 | 41.06 | 1229028 | 1.13 |
| model35 | 2 | **bc8** | **41.95** | 44.87 | 42.08 | 42.07 | 40.57 | p50 1 p90 4 p99 10 max 65 | **41.95** | 1229028 | 1.38 |
| model23 948x1185 | 1 | 4-bit palette | 37.10 | 40.37 | 36.34 | 34.57 | 32.55 | p50 2 p90 7 p99 17 max 131 | 37.10 | 753068 | 0.90 |
| model23 | 1 | **bc8** | **38.51** | 39.75 | 36.74 | 34.89 | 33.10 | p50 2 p90 6 p99 17 max 100 | **38.51** | 753068 | 1.39 |
| model23 | 2 | **4-bit palette** | **40.90** | 43.52 | 40.16 | 38.41 | 36.35 | p50 1 p90 4 p99 12 max 81 | **40.90** | 1505988 | 1.37 |
| model23 | 2 | bc8 | 40.67 | 42.21 | 39.56 | 38.22 | 36.36 | p50 2 p90 5 p99 13 max 78 | 40.67 | 1505988 | 1.68 |
| dmario2 1619x911 | 1 | 4-bit palette | 32.31 | 34.50 | 29.96 | 27.87 | 25.84 | p50 5 p90 14 p99 31 max 139 | 32.31 | 986420 | 1.17 |
| dmario2 | 1 | **bc8** | **32.91** | 34.22 | 30.31 | 28.10 | 26.07 | p50 5 p90 13 p99 31 max 138 | **32.91** | 986420 | 1.83 |
| dmario2 | 2 | 4-bit palette | 34.74 | 37.15 | 32.56 | 30.63 | 28.71 | p50 4 p90 11 p99 24 max 117 | 34.74 | 1972692 | 1.82 |
| dmario2 | 2 | **bc8** | **35.26** | 36.76 | 32.76 | 30.65 | 28.58 | p50 3 p90 11 p99 25 max 125 | **35.26** | 1972692 | 2.18 |

**bc8 wins seven of the eight pairs**, by +0.52 to +3.46 dB, and loses one -- model23 at `C0 2` -- by 0.23 dB. The mip
levels move with the base almost everywhere; the worst-case percentiles move with it too, and the `max` column improves
on five of the eight (model23 `C0 1` goes from 131 to 100, model10 `C0 2` from 59 to 48).

Two columns move the other way and are worth naming rather than burying. **`sampled psnr` is lower for bc8 at `C0 1` on
three of the four images** (model10 -0.56, model23 -0.62, dmario2 -0.28) while the centre psnr is higher: the mode buys
its win at the texel grid and gives a little of it back between the grid points, which is where the level-0 plane is
read bilinearly and where a block's eight values along a line are least like a smooth surface. And on model23 at
`C0 2` and dmario2 at `C0 2` some of the deep levels (M3) are a hundredth or two behind. Neither is a verdict: the out
directories are `out/s9a/*/` and the recon PNGs are beside the assets.

**Time.** bc8 costs 1.2x to 1.5x the palette mode. Block (c') is the whole of it: the per-texel palette search touches
nine sites and enumerates at most 256 states, while block (c') assembles a 9-point stencil over the full-resolution
plane and runs conjugate gradients or four-colour sweeps on it. The absolute numbers stay between 1.0 and 2.2 seconds.

**Device memory.** Block (c')'s workspace is the same five buffers block (b) has over a plane with sixteen times the
texels, so it is the largest allocation an encode makes: 395.7 MB against 94.8 MB for model10 at `C0 2`, and 190.4 MB
at `C0 1`. It is not allocated at all in the palette mode.

### 6.1 What the pack costs, and what the refit wins back

The whole point of measuring this separately is that the post-fit pack happens **after** the solve, so the solve never
sees the block format. This table is the gap that leaves. `continuous` is the plane the loop finished with, `packed` is the same
plane BC4 / BC5-encoded, `refitted` is after one block (b) on level 1's frozen grid and one block (a) -- all three are
the report's `centre psnr` (fp, every output, padded extent), which is the quantity the pack is applied to. `shipped`
is the 8-bit texture PSNR the table above quotes, on a different definition, and is here only so the two can be lined
up. `pack psnr` is the BC step's own error against the continuous plane, in the latent's byte units.

| image | C0 | continuous | packed | refitted | pack psnr | shipped | palette shipped | bc8 - palette |
|---|---|---|---|---|---|---|---|---|
| model10 | 1 | 40.48 | 39.49 | 39.50 | 45.03 | 39.49 | 38.62 | +0.87 |
| model10 | 2 | 45.67 | 42.89 | 42.90 | 40.59 | 42.84 | 41.51 | +1.33 |
| model35 | 1 | 43.59 | 40.74 | 40.64 | 43.69 | 40.62 | 37.16 | +3.46 |
| model35 | 2 | 46.67 | 42.06 | 41.99 | 40.46 | 41.95 | 41.06 | +0.89 |
| model23 | 1 | 39.36 | 38.45 | 38.44 | 44.41 | 38.51 | 37.10 | +1.41 |
| model23 | 2 | 42.33 | 40.61 | 40.62 | 41.72 | 40.67 | 40.90 | -0.23 |
| dmario2 | 1 | 33.28 | 32.91 | 32.92 | 41.59 | 32.91 | 32.31 | +0.60 |
| dmario2 | 2 | 35.99 | 35.26 | 35.27 | 36.92 | 35.26 | 34.74 | +0.52 |

**The pack costs 0.37 to 4.61 dB**, and it costs most exactly where bc8 wins most: model35 at `C0 2` solves to 46.67 dB
continuous and ships 41.95. The refit wins back **essentially nothing** -- between -0.10 and +0.01 dB on this measure.
That is not a bug and it is worth being precise about why: the refit minimises `E`, which carries `--rgb-weights 9,11,1`
and every plane of the chain, and `E` does fall at the refit on every one of the eight runs, by 0.5 to 1.5 per cent of
what the pack added (the gate in `tests/run_checks.py` asserts the direction of each step). The centre psnr is an unweighted average over the padded
base plane, so it can move a hundredth the other way while the thing actually being minimised falls. What the numbers
say is that once level 0 has been packed, level 1 and the decoder have almost no freedom left to absorb the damage: the
error the pack introduced is in the full-resolution plane and level 1 is at a quarter of the resolution per axis.

**So the whole of the remaining headroom is in the pack itself, not in the refit.** That is the case for choosing the
pack by analysis by synthesis rather than against the latent's own values -- each block's endpoints and selectors
chosen for what the decoder makes of them, so that the plane the model is fitted to is the plane the hardware will
decode. On these eight runs that was between 0.4 and 4.6 dB of latent-plane error the solve was not allowed to see.
Section 6.2 is the answer; this section's job was to make the gap visible, and it did.

### 6.2 The pack chosen by analysis by synthesis

Section 6.1 is the case for this one. The post-fit pack is the stb_dxt-style choice fitted against the *latent's* own
values, and nobody samples the latent -- so each 4x4 block's endpoints and selectors are now re-chosen to minimise the
**decoder's output error** over the sites that block's texels touch, a candidate being taken only when that error
measurably falls (`docs/DESIGN.md` section 3.6). It starts from exactly the seed pack of 6.1, so it cannot be worse
than it, and the outer repack loop that follows takes a pass only when the shipped `E` strictly falls, so it cannot be
worse than the pass before it.

**These numbers were re-measured with the CURRENT binary and they are not the ones this section carried when the
refinement first landed.** Two things have arrived since: the residual level-0 seed (`--init0 residual`, now the
default, section 7.2) and the last free refit of level 1 and the decoder after the outer loop. Both lower `E` on every
one of the eight runs, so every bc8 figure below is better than its first-measured counterpart -- `model10` at `C0 2`
ships 44.97 dB where it shipped 43.34 -- and the tables of 6 and 6.1 above, which were measured before either, are
left exactly as they were measured. Where a column below compares the two, it says which binary each side came from.

Every number comes from a log under `out/s9b/`, one directory per run. The runs are the eight of 6.1, at the same
command lines -- `nntc_encode IMAGE.png -o out/s9b/IMAGE_cN/ --c0 N --c1 4 --bits1 8 --rgb-weights 9,11,1 --mips 1`,
everything else the defaults, which includes `--l0 bc8`, `--bc-refine 4` and `--bc-outer 2`. **The `.dds` files are
byte-for-byte the same size as in 6.1** -- the refinement changes which blocks are written, never how many (section 2's
note applies to these commands too).

`E` is the objective the encoder minimises (weighted, every plane of the chain, centre and fractional sites); the psnr
columns are the report's `centre psnr` (fp, every output, padded extent), which is the quantity the pack is applied to.
`recovered` is the share of the pack's own cost in `E` that the refinement, the outer loop and the last refit take
back: `(E_seed - E_shipped) / (E_seed - E_continuous)`.

| image | C0 | E continuous | E seed pack | E refined | E shipped | recovered | psnr continuous | seed pack | refined | shipped | gain over the seed pack |
|---|---|---|---|---|---|---|---|---|---|---|---|
| model10 | 1 | 3.6527e-05 | 4.9042e-05 | 4.5333e-05 | 4.4450e-05 | **37 %** | 42.08 | 40.87 | 41.06 | **41.11** | +0.24 |
| model10 | 2 | 6.0867e-06 | 1.8728e-05 | 1.3972e-05 | 1.3150e-05 | **44 %** | 47.62 | 44.17 | 44.77 | **45.00** | +0.83 |
| model35 | 1 | 1.5908e-05 | 3.3641e-05 | 2.8525e-05 | 2.7951e-05 | **32 %** | 43.88 | 41.05 | 41.36 | **41.42** | +0.37 |
| model35 | 2 | 3.4688e-06 | 2.1215e-05 | 1.2197e-05 | 8.5776e-06 | **71 %** | 47.92 | 42.65 | 44.00 | **45.62** | +2.97 |
| model23 | 1 | 3.9267e-05 | 5.5200e-05 | 5.0428e-05 | 4.9070e-05 | **38 %** | 41.80 | 40.53 | 40.70 | **40.80** | +0.27 |
| model23 | 2 | 4.5102e-06 | 2.0608e-05 | 1.5668e-05 | 1.4607e-05 | **37 %** | 46.82 | 43.51 | 43.98 | **44.15** | +0.64 |
| dmario2 | 1 | 1.8922e-04 | 2.2387e-04 | 2.1345e-04 | 2.0989e-04 | **40 %** | 34.78 | 34.33 | 34.41 | **34.44** | +0.11 |
| dmario2 | 2 | 2.1844e-05 | 5.7899e-05 | 4.4977e-05 | 4.3453e-05 | **40 %** | 40.81 | 39.08 | 39.31 | **39.38** | +0.30 |

**Every one of the eight improves, as it must** -- the refinement starts from the seed pack and only accepts decreases
-- and the recovered share of the pack's cost is **32 % to 71 %**, worth +0.11 to +2.97 dB of centre psnr over packing
once. The two ends are worth naming. `model35` at `C0 2` is the run where the pack costs most (5.27 dB from the
continuous plane to the seed pack) and it is also where most comes back: 71 % of the `E` cost and **+2.97 dB**, from
42.65 to 45.62. `dmario2` at `C0 1` is the run where the pack costs least (0.45 dB) and there is correspondingly little
to win: +0.11 dB on a 40 % recovery.

**The refinement, the outer loop and the last refit split the win differently per image.** The refinement alone
accounts for nearly all of it on the `C0 1` runs (the outer loop's two passes are accepted but move `E` in the third or
fourth digit), while on `model35 C0 2` what follows the refinement is worth more than the refinement itself: 44.00 dB
after it and 45.62 shipped. **Both outer passes were accepted on all eight runs** -- none was rejected and restored, so
the accept-only-if-lower rule is not what is holding the numbers back; a third pass was not measured.

**`--bc-refine-after 1`, the ablation** (`out/s9b/ablate_*`: the same eight commands with `--bc-refine-after 1` added).
One more refinement pass *after* level 1 and the decoder have been refitted against the packed plane is worth, in the
table's order, +0.01 +0.03 +0.02 **+0.06** +0.02 +0.03 +0.00 +0.01 dB of shipped psnr. It is a hundredth or two
everywhere and never more than six, so the default is 0; it stays as the flag to reach for on a run where the pack
costs a great deal, since that is where it has anything left to find.

**The shipped asset, read from the files alone.** `dds_decode.py --psnr-levels`, which shares no code with the encoder,
reproduces the report's `psnr` at the base of **every one of the eight, to the last digit**, and the `.dds` sizes are
unchanged from 6.1. The `4-bit palette` column was re-run with the same binary at the same command lines plus
`--l0 palette --bits0 4` (`out/s9b/*_palette/`), so the comparison is one binary on both sides at byte-for-byte equal
`lat0` size:

**The `4-bit palette` column was re-measured at the predecessor tree's `v0.9g` and the eight figures below are NOT the
ones this section carried at the predecessor tree's `v0.9f`.** That column was falling over: at `C0 2` it measured WORSE than at `C0 1` on `model10` (36.38
against 39.00) and on `model23` (31.72 against 36.69), which is not something a second full-resolution channel can
honestly do, and this section carried the inversion as a caveat. The cause and the fix are in `docs/DESIGN.md` 3.4 --
the residual seed divided its projection by the chain's PEAK, which under `--l0 bc8` is a gauge the `--q1-start` freeze
removes and under `--l0 palette`, whose palette is fixed on `[-1,1]`, is not; and it snapped the channel onto that
palette blindly, so every further channel re-picked the first one's direction. The seed now divides by the 99 %
percentile in the palette mode and follows each seeded channel with one level-0 search and one refit. **The bc8 column
is byte-for-byte unchanged** -- the encoder writes an identical `_lat0.dds`, `_lat1.dds` and `_nntc.json` on `model10 C0 2`
before and after -- so this table is one binary on both sides at byte-for-byte equal `lat0` size, as it was.

| image | C0 | shipped psnr (8-bit) | dds_decode M0 | 4-bit palette | the predecessor tree's v0.9f palette | bc8 - palette | lat0 bytes | time s | refine ms |
|---|---|---|---|---|---|---|---|---|---|
| model10 | 1 | 41.12 | **41.12** | 39.68 | 39.00 | **+1.44** | 564500 | 1.423 | 61.8 |
| model10 | 2 | 44.97 | **44.97** | 41.74 | 36.38 | **+3.23** | 1128852 | 1.951 | 123.8 |
| model35 | 1 | 41.40 | **41.40** | 38.84 | 37.43 | **+2.56** | 614588 | 1.557 | 67.7 |
| model35 | 2 | 45.55 | **45.55** | 41.45 | 35.92 | **+4.10** | 1229028 | 2.197 | 135.2 |
| model23 | 1 | 40.85 | **40.85** | 37.34 | 36.69 | **+3.51** | 753068 | 1.906 | 82.8 |
| model23 | 2 | 44.16 | **44.16** | 41.29 | 31.72 | **+2.87** | 1505988 | 2.731 | 165.0 |
| dmario2 | 1 | 34.43 | **34.43** | 33.49 | 32.84 | **+0.94** | 986420 | 2.463 | 108.2 |
| dmario2 | 2 | 39.34 | **39.34** | 35.87 | 34.53 | **+3.47** | 1972692 | 3.531 | 216.9 |

**bc8 wins all eight pairs against the 4-bit palette at equal memory.** The one pair the post-fit pack of 6/6.1 lost --
`model23` at `C0 2` -- is won. That is why `--l0 bc8` is the default. The margins are now **+0.94 to +4.10 dB** rather
than the +1.59 to +12.44 the broken palette arm made them look, and the two pairs the caveat named are the two that
move most: `model23 C0 2` from +12.44 to +2.87, `model10 C0 2` from +8.59 to +3.23 (the caveat's own estimate of that
one, taken from the `--init0 luma` control, was +3.39, and the fixed seed beats that control).

**The palette arm is internally consistent again.** Two channels beat one on all four images -- 39.68 to 41.74, 38.84
to 41.45, 37.34 to 41.29, 33.49 to 35.87 -- and the residual seed is at or above the `--init0 luma` control at every
layout measured, on shipped `E` as well as on psnr (`out/s9g/gate_*/`, the eight runs of the gate):

| image | C0 | residual psnr | luma psnr | residual sampled | luma sampled | residual E shipped | luma E shipped | time s |
|---|---|---|---|---|---|---|---|---|
| model10 | 1 | **39.68** | 38.62 | **42.20** | 41.51 | **6.584e-05** | 8.466e-05 | 0.677 / 0.645 |
| model10 | 2 | **41.74** | 41.58 | **44.20** | 43.87 | **3.070e-05** | 3.441e-05 | 1.030 / 0.971 |
| model23 | 1 | **37.34** | 37.10 | 39.55 | **40.37** | **9.157e-05** | 1.146e-04 | 0.942 / 0.907 |
| model23 | 2 | **41.29** | 41.08 | **44.05** | 43.81 | **3.694e-05** | 4.133e-05 | 1.458 / 1.397 |

The one column that goes the other way is `model23 C0 1`'s sampled psnr, 39.55 against 40.37, while its centre psnr and
its shipped `E` -- the quantity actually minimised, which carries the fractional sites and every plane of the chain --
are both better. It is named rather than buried; the recon PNGs are in the out directories.

**What the seed costs in time.** The per-channel search is one block (c) per seeded channel and is counted as block (c)
in the report, so the palette arm's encode is 4 to 6 % longer than the `--init0 luma` control's at the same layout
(1.030 s against 0.971 on `model10 C0 2`, 1.458 against 1.397 on `model23 C0 2`) and the palette arm is still
roughly half the bc8 arm's time.

**Time.** `refine ms` is the first refinement's four passes over every plane of the chain (CUDA events); the total
encode is 1.4 s on `model10 C0 1` to 3.5 s on `dmario2 C0 2`, against 0.7-1.8 s for the palette arm of the same
layouts. The refinement itself is double-precision work on one CUDA block per 4x4 block; it is not where a further
order of magnitude would be looked for first.

**The packing psnr is not the measure, and it moves both ways.** The refined pack's distance to the continuous plane is
worse than the seed pack's exactly where much was won -- `model35 C0 2` goes from 40.55 dB to 34.02 and `model10 C0 2`
from 42.49 to 40.76 -- because the refined pack is not trying to be near that plane at all. It is trying to make the
decoder's output right, and the continuous plane is only where the search started. Where little was won it drifts the
other way instead (`model10 C0 1` 41.86 to 42.47, `model23 C0 1` 42.37 to 42.95), which says the same thing: the
quantity is a by-product and not an objective.

**Where the rest of the gap is.** 29 % to 68 % of the pack's cost is still there, and most of it is the block format's
own: eight values on a line per 4x4 block per channel, with the endpoints on the 8-bit grid. The refinement reaches a
**local minimum of that mixed-integer problem by alternating descent** -- the endpoints by least squares plus a `+-1`
neighbourhood with the selectors held, the selectors by exact argmin with the endpoints held, two Gauss-Seidel passes,
four rounds, no joint endpoint/selector move and no six-value mode -- and the fourth round still improves 24 021 of
76 805 blocks on `model35 C0 2`, which says the descent is still finding coordinates to move rather than that it has
run out. So the remainder is the constraint plus whatever a joint move would have found, and not a failure to search
the coordinates it does search. The next
place to look is the one the owner named: the pack **inside** the loop, so that the continuous solve and level 1 and
the decoder are all fitted to a plane that is already on the block format's manifold, rather than being packed once at
the end and repaired.

### 6.3 The checks on the mode

| check | result |
|---|---|
| `E` through the loop, `tests/run_checks.py` | monotone across every block of every round, block (c') included |
| the pack and the refit | `E` rises at the pack (a constraint being applied) and falls at the refit (two minimisations), on every run; both asserted by direction, since the pack's error is in the latent's byte units and `E` is in the output's |
| the refinement | `E` after it is at or below `E` after the stb_dxt-style seed pack, and the seed it reports is the pack the line above measured; every pass reports a decrease, never an increase (`tests/run_checks.py`) |
| the outer repack loop | every accepted pass lowered the shipped `E`, the loop stops at its first rejected pass, and `E` shipped is at or below `E` after the refinement |
| the restore after a rejected pass | the `level 0 final` line's own starting `E` is exactly the last ACCEPTED pass's `E`, or the pre-loop `E` when none was accepted -- which is the statement that the restore put every plane, index, block, grid and the decoder back. `tests/run_checks.py` runs a case that rejects (`--bc-refine 0 --bc-outer 2`) and asserts it |
| the last free refit | one block (b) and one block (a) over a level 0 that does not move, after the loop: `E` falls and `E shipped` in the report is exactly the number that line ends at |
| `nntc_view --shot` launched five times on one asset | five byte-identical frames. `--shot` takes no keyboard input at all, so the frame is a function of the command line and the asset; it used to read the live keyboard while its own window was foreground, which made one launch in ten differ from the others (`tests/run_checks.py`) |
| `dds_decode.py --ref` on the bc8 asset | worst max abs diff **0** at every level |
| `dds_decode.py --psnr-levels` against the report | equal at every level, on `tiny.png` and on all eight comparison runs (the `dds_decode M0` column above is that number) |
| `dds_decode.py --grid` | the bc8 asset publishes no palette; the byte rule and the index rule agree to **0.000e+00** on both levels, as they must at 8 bits |
| `bc_check` on a `--bc0 both` bc8 asset | no lossless claim at 8 bits ("not exact by construction above 3 bits"); on the refined pack, packing PSNR **36.70 dB** against the uncompressed pre-pack twin where the encoder reported 36.72, the 0.02 being bcdec's float palette against `bc_pack.h`'s (max diff 2.8e-05 of a byte) |
| `nntc_view --shot` on `model35 C0 2` bc8, refined | opens the BC5 asset, reads `lo/hi` off the level-0 `range` entry and draws the quad: the skin gradients are smooth with no blocking, the sand grain over them survives, the printed swimwear keeps its small blue and pink detail, and there is no colour shift against the source |
| `nntc_view --shot` on `model10 C0 2` bc8 | opens the BC5 asset, reads `lo/hi` off the level-0 `range` entry and draws the quad: skin tones and the printed fabric are clean, no blocking, no colour shift against the source |
| `nntc_view --shot` on `dmario2 C0 2` bc8 | the crowded scene decodes with its small coloured objects and the shop signage intact; no splotching anywhere in the frame |


## 7. The deep-plane trade, and the init0 arms

### 7.1 A chain that collapses while the base improves (`mg1-4`)

The four-texture material `mg1-4` (2048x2048; `mg2` is a 16-bit greyscale read as its top byte, `mg3` 8-bit greyscale)
at the defaults, with `--c0 2`:

| | per-texture psnr at the base | texture 1's chain M1..M8 | E shipped |
|---|---|---|---|
| `--c1 3` | 38.19 / 51.09 / 41.81 / 40.36 | 47.65 46.16 44.85 43.52 42.99 44.37 46.57 49.12 | 6.281e-05 |
| `--c1 4` | **49.26** / 50.75 / 42.48 / 40.54 | **47.16 41.71 34.38 28.82 26.74 27.77 31.01 36.18** | **3.733e-05** |

Texture 0's base gains 11 dB, texture 1's chain loses up to 16 dB, and the objective prefers the second run by a third.

**What it is not.** Measured, one at a time: the level-1 grid (`--q1-range minmax` moves M4 by 0.01 dB, so nothing is
clipping); the level-1 initialisation (`--init pca`, which starts the four channels independent instead of the box
init's three near-collinear ones, gives M3..M5 of 34.39 / 28.85 / 26.75 - the same collapse); the level-0 seed (all four
`--init0` x `--init0-scope` arms give M3..M5 within 0.1 dB of each other); the freeze, which snaps and re-sweeps every
plane, and the outer repack loop, which re-solves and refits every plane - the collapse is already fully present at
**round 1**, before either runs (`mip psnr M4 34.59 M5 32.59` at round 1 against the `--c1 3` run's `M4 40.10 M5 39.66`).

**What it is.** The shared decoder, starved of deep-plane weight. `--diag`'s decoder table gives texture 1's level-0
figure as **1.90** at `--c1 4` against **8.90** at `--c1 3`: with a fourth level-1 channel available, the base's least
squares serves texture 1 from level 1 and builds it almost no full-resolution path, and under `--mip-weight pixels
--mip-mix 0.5` planes M4 and below hold 0.59 % of block (a)'s mass between them and cannot argue. Every deep plane is
then held to what a quarter-resolution latent can carry alone, and it sits there: box-filtering texture 1's own source
mips down by four and bilinearly back up gives 39.95 35.68 30.79 27.12 25.92 27.18 30.58 35.85 for M1..M8, which is the
collapsed chain to within the little level 0 still contributes.

**What moves it.** Only the weights, and they trade:

| `--c0 2 --c1 4` | tex0 base | tex1 base | tex1 M4 | tex1 M5 | E shipped |
|---|---|---|---|---|---|
| `--mip-weight pixels --mip-mix 0.5` (default) | 49.26 | 50.75 | 28.82 | 26.74 | 3.733e-05 |
| `--mip-weight pixels --mip-mix 0.7` | 49.20 | 50.07 | 29.01 | 26.91 | - |
| `--mip-weight sqrt` | 47.68 | 46.48 | 31.15 | 29.00 | - |
| `--mip-weight uniform` | 47.20 | 47.77 | 37.42 | 36.34 | - |

`uniform` buys most of the chain back and costs about 2 dB of base on every texture, which is the same trade stated
twice. Which is wanted is a decision about the asset, so the default is unchanged and `--diag` is the thing that makes
the trade visible. `docs/DESIGN.md` section 4.3.1 has the mechanism.

### 7.2 The init0 arms

`mg1-4`, `--c0 2`, the four arms of `--init0` x `--init0-scope`:

| arm | per-texture psnr | E shipped |
|---|---|---|
| `--c1 4` residual, base (the default) | 49.26 / 50.75 / 42.48 / 40.54 | 3.733e-05 |
| `--c1 4` texture, base | 48.77 / 50.72 / 43.34 / 40.55 | **3.540e-05** |
| `--c1 4` residual, chain | 48.92 / 50.76 / 42.41 / 40.53 | 3.773e-05 |
| `--c1 4` texture, chain | 48.77 / 50.72 / 43.34 / 40.55 | 3.540e-05 |
| `--c1 3` residual, base (the default) | 38.19 / 51.09 / 41.81 / 40.36 | 6.281e-05 |
| `--c1 3` texture, base | 38.42 / **58.48** / 41.73 / 40.45 | **5.969e-05** |
| `--c1 3` residual, chain | 38.26 / 52.20 / 41.74 / 40.36 | 6.214e-05 |
| `--c1 3` texture, chain | 38.42 / 58.47 / 41.73 / 40.45 | 5.969e-05 |

Three deliberately unrelated pictures, `model` + `modelb` + `modelc` at 512x512, `--rgb-weights 9,11,1`:

| arm | per-texture psnr | E shipped |
|---|---|---|
| `--c0 3 --c1 3` residual, base | 24.92 / 30.42 / 26.27 | **1.0412e-03** |
| `--c0 3 --c1 3` texture, base | 24.74 / 30.52 / 26.56 | 1.0580e-03 |
| `--c0 3 --c1 3` residual, chain | 24.93 / 30.42 / 26.24 | 1.0385e-03 |
| `--c0 3 --c1 4` residual, base | 31.31 / 29.29 / 29.72 | 5.241e-04 |
| `--c0 3 --c1 4` texture, base | 29.79 / 29.87 / 28.36 | **5.186e-04** |
| `--c0 4 --c1 4` residual, base | 34.49 / 34.86 / 31.95 | **1.684e-04** |
| `--c0 4 --c1 4` texture, base | 33.73 / 34.75 / 31.73 | 1.763e-04 |
| `--c0 4 --c1 4` residual, chain | 31.94 / 33.07 / 31.59 | 2.037e-04 |

Neither arm is one-sided, so both stay controls and the defaults (`residual`, `base`) are unchanged - and are
byte-identical to the predecessor tree's `v0.9d.1` on `model10` bare, on `model10 --rgb-weights 9,11,1` and on `chroma --rgb-weights 9,11,1`.

The three-picture case is worth its own sentence, because it looks like a basin failure and is not. At `--c0 3 --c1 3`
three full-resolution channels for three independent pictures ought to give each picture one, and the seed does not:
the directions it picks are blends of textures 1 and 2 with texture 0 at **zero**, because the box level-1 init has
already given level 1's three channels to texture 0's own RGB and texture 0 therefore has the smallest residual at the
moment the channels are handed out. `--init0 texture` makes each channel dedicated and the run still lands in the same
place (24.74 / 30.52 / 26.56 against 24.92 / 30.42 / 26.27) - the assignment is not what is binding. What IS binding is
that same level-1 init: `--init pca`, which spreads level 1 over the material's own principal directions instead of
handing it to texture 0, takes E from 1.0412e-03 to **7.465e-04** (-28 %) and the psnrs to 27.24 / 28.86 / 26.79, and at
`--c0 3 --c1 4` from 5.241e-04 to 4.874e-04. On a material of several unrelated pictures, `--init pca` is the flag to
reach for; it is not made the default because on single images the two inits split (section 2).
