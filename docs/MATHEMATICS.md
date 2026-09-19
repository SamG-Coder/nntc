# MATHEMATICS.md: NNTC's optimiser, for someone about to change it

This is the working knowledge an engineer (or an agent) needs before touching `src/`. `DESIGN.md` is the method as
written for a reader; this file is the same material as a set of facts, invariants and traps, in the order they bite.
Every statement here is about the code as it stands. Numbers come from `docs/RESULTS.md` (logged under `out/`) unless
marked "session measurement", which means a run made while writing this and not carried in the tree. A Fable 5.1 review
checked every claim against the source as it stood at the predecessor tree's commit e65c262, the state later imported
here as v1.0.0-nntc; its corrections are folded in.

## 1. The model, in one screen

Two latent textures and one affine map.

- **Level 0** (`s`): full resolution, `C0` = 1..4 channels. Either a palette index (`--l0 palette --bits0 N`, 1-4 bits,
  a uniform palette on [-1, 1]) or, by default (`--l0 bc8`), a continuous plane on a per-channel 8-bit grid
  `lo + k/255 (hi - lo)` that is packed to BC4 / BC5 and shipped as those blocks.
- **Level 1** (`c`): a quarter of the resolution per axis (the block is a fixed 4), `C1` = 1..4 channels, each on a
  per-channel grid `lo + rep(k)/255 (hi - lo)` where `rep(k)` is the index replicated over the byte (section 4); that
  equals `lo + k/levels (hi - lo)` only at 4 and 8 bits. 4..8 bits, stored uncompressed (R8, R8G8, RGBA8). (An
  uncompressed level 0 in palette mode, `--bc0 0`, sits on the same replicated-byte rule.)
- **The decoder**: `out = W phi + b`, `phi = [c_j] [s_i] [s_i c_j]` in exactly that order (`j` inner for the products),
  `nin = C1 + C0 + C0 C1`, `nout = 3 T` for `T` = 1..6 RGB textures. Identity output; the consumer saturates to [0, 1].

Everything follows from two algebraic facts:

1. **`out` is affine in `s` when `c` is held, affine in `c` when `s` is held, and linear in `(W, b)` when both are
   held.** Each block of unknowns therefore has an exact minimiser given the others. That is the whole optimiser.
2. **Hardware filtering is a convex combination, and the dequantisation is affine and applied after the blend**, so the
   blend of latent samples decodes to the blend of decodes for every affine part of the decoder. The only term that does
   not commute with filtering is the product `s_i c_j`, whose error under a filter footprint is a weighted local
   covariance of `s` and `c`; `c` is smooth at level 0's scale, so it is small, and the encoder fits it anyway (section 2).

## 2. The objective `E`

`E` is a weighted sum of squared output errors over a fixed, deterministic **site set**, summed over every stored mip
plane `m`:

- **centre sites**: one per pixel of plane `m`; level 0 read at the texel itself (a bilinear sample at a texel centre IS
  the texel), level 1 read bilinearly at the pixel centre with the hardware clamp; the target is the plane's pixel of
  the run's own source chain (`docs/DESIGN.md` 4.2: the iterated 2x2 box by default, a per-texture filter when one is
  asked for);
- **fractional sites**: `K = --k` (default 4) per pixel at fixed offsets, the rotated-grid pattern
  `P = {(-3/8,-1/8), (-1/8,+3/8), (+1/8,-3/8), (+3/8,+1/8)}` in plane pixels, rotated by 90 degrees per texel phase
  (`(tx & 1, ty & 1)`; at K = 4 the rotation is a no-op because `P` is symmetric under it, it matters at K = 1); BOTH
  levels read bilinearly, the target is the source sampled the same way with the same clamp.

`E = sum_m w_m sum_sites sum_c cw[c] (out_c - t_c)^2`, `cw[c] = weights[c/3] * rgb_weights[c%3] * 3 / sum(rgb_weights)`.
Centre and fractional sites carry equal weight (the implicit fractional share is `K/(K+1)`). The mip weights `w_m`
(`--mip-weight pixels` by default: every CHAIN plane the same per-site weight; the base's share is `1 - mix` under every
rule, `--mip-mix 0.5`, and its per-site weight equals the chain's only at `mix = chain / (base + chain)`)
enter ONLY the decoder's least squares (section 3a). Blocks (b) and (c) are per plane and do not see them.

`E` is evaluated in double on the device by a fixed-order two-stage reduction and has a brute-force host twin
(`--check`) that agrees to ~1e-14. **The solver kernels work in float with double accumulation; the acceptance tests in
the round loop compare `E` values with a 1e-9 relative tolerance for that reason.** A "strict decrease" inside a kernel
is measured on its own double quadratic; a rounding-level rise of the float-measured `E` is not a bug.

The sampled-PSNR the report prints is on the same K sites (it is what E fits, reported as fitted). The texel-centre PSNR
and the per-level mip PSNR are what the independent decoder (`tools/dds_decode.py`) reproduces from the files alone.

## 3. The three block solves, and what each guarantees

**(a) The decoder, `W` and `b`.** Normal equations `A = sum omega v v^T`, `rhs_c = sum omega v t_c`, `v = [phi; 1]`, over
EVERY site of EVERY plane with `omega = w_m`. One `A` shared by all outputs (`cw` scales each output's equation uniformly
and cannot move its argmin, so it is left out). Gauss-Jordan with partial pivoting in double on the host, ridge
`1e-9 mean(diag A)`.

**What it guarantees is that `E` does not rise across it, and that is NOT the same statement as "global minimiser".**
What the solve returns is the minimiser of `E + ridge ||[W b]||^2`. The two coincide only while the ridge is negligible
against the solution, and on a picture that drives `A` nearly singular it is not: one white pixel on a 100x36 black
field takes level 1's channels to a span of 0.155, the exact minimiser's norm past 100, and `E` from 1.875599e-06 to
2.112583e-06 ACROSS BLOCK (a) -- a 13 % rise, and again every round from the ninth. Raising the ridge to 1e-6 ends that
run six times worse (8.800630e-06 against 1.268264e-06 at round 20) and still rises; dropping it to zero makes `A`
singular at the init, where level 0 is a constant plane, and the run collapses at 8.951838e-05 in one round. So the
block measures its own step, for free: `A` and `rhs` ARE the quadratic `E` is in `(W, b)` up to the constant
`sum omega cw t^2`, so `Q(x) = sum_c cw[c] (x_c^T A x_c - 2 x_c . rhs_c)` differenced between the incoming decoder and a
candidate is the change in `E`'s weighted sum -- up to the fp32 rounding of the features `A` and `rhs` were accumulated
from, which `E` is measured in double from the same planes, about 1e-12 in the printed units at a decoder norm of 300 --
and dividing by `sum(cw)` times the weighted site count is the change in the printed `E`. The shipped ridge is tried
first -- a well-conditioned `A` therefore gives the decoder it always gave, byte for byte -- then `1e-12 mean(diag A)`,
then zero; a candidate is taken unless it raises `E` by more than 1e-12 (`Q`'s own rounding is 1e-16 there, and a
re-solve of a converged system returns the same `Q` to the last bit and must still be taken, or the run changes basin
for nothing), and if none is taken the previous decoder is kept and the block does not step.

**`Q`'s own rounding is computed, not assumed.** `Q` is a difference of two large nearly-cancelling sums, so its
absolute rounding is about `eps mm sum_ij |x_i| |a_ij| |x_j| + 2 eps mm sum_i |x_i| |rhs_i|`, growing with `|x|^2`. The
lower rungs are reached exactly when `A` is near-singular, which is when `|x|` is large, and at `|x|` of 1e6 to 1e8 the
computed difference of two `Q`s is noise of random sign. So the block computes that bound beside each `Q`, for the
incoming decoder and the candidate both, and takes the candidate only when `dQ + err_prev + err_cand <= 1e-12`: an
inconclusive comparison is a refusal. A solve that comes back with an infinity or a NaN anywhere is refused before `Q`
is evaluated at all. The report line `block (a) ridge:` counts how many calls of the run ended on each rung. On the dot image the fallbacks are not merely a safety net: the run ends at 4.627044e-07 and 60.46 dB
where it ended at 1.268264e-06 and 53.69 dB. No tuning parameter: the ladder and the bar are fixed.

**(b) Level 1 (and, under bc8, level 0), a sparse least squares.** Hold `s` and `W`. Each site's output is
`m(s) + G(s) sum_t w_t c_t` with `G(s) = A + sum_i s_i M_i` (the `a` block plus the `sc` rows weighted by the level-0
sample). Assembled by GATHER per texel (no atomics): a 9-point block stencil of `C1 x C1` blocks, five owned per texel,
double throughout (160 B per texel at C1 = 2, 640 at C1 = 4). Solved by block-Jacobi preconditioned conjugate gradients
to a relative residual of 1e-10 (the residual is probed every fourth iteration, so a stop can overshoot by three; a
cap of 200 iterations), on a PROXIMAL form: the unknown is the correction from the current plane, with a ridge
`lambda ||delta||^2`, `lambda = --ridge (1e-4) * mean(diag H)`. The ridge is not cosmetic: with `T = 1` and `C1 = 4`
every site's `G^T G` has rank at most 3, so a texel whose footprint is flat has a genuine null direction and only the
ridge keeps its range bounded. The same machinery solves level 0 under bc8 with the roles exchanged (`Q(c) = B +
sum_j c_j N_j`; on every plane each pixel is its own level-0 texel, so the centre site has weight 1 on one tap).

**(c) Level 0 in palette mode, the exact per-texel search.** For each texel, accumulate over its sites (its own centre
site plus the K fractional sites of the 3x3 pixel neighbourhood whose bilinear taps touch it; a clamped duplicate tap ADDS
its weights) the quadratic `E(x) = A0 + 2 x.A1 + x.A2.x` in the texel's `C0` values, from the affine structure
(`out = p + alpha Q x`, `Q` the `s` columns of `W` at the site's level-1 sample). Then the exact argmin over the palette:
joint enumeration when the bit total is at most 8 (at most 256 states), coordinate sweeps above. Ties keep the current
state; only a strict decrease moves.

**Independence.** Both (b)'s quantised sweeps and (c) run in four colour passes `(tx & 1, ty & 1)` (level 1) or over the
same parity classes of texels (level 0). A bilinear site's two taps per axis are ADJACENT texels, so same-colour texels
(two apart) share no site and each pass is an exact joint block step for any offset; the offsets being at most 3/8 (and
the clamp only pulling inward) is what makes the 3x3 pixel enumeration of a texel's sites complete. `E` is
non-increasing across every block of every round; the round loop warns, naming the round and the block, when one rises
beyond the loose tolerance - a warning in both configurations and never an assert, because it is a floating-point
judgement and an assert that can cry wolf on rounding is an assert nobody trusts.

## 4. Quantisation, and the one invariant that must never break

- Level 1's grid (and level 0's under bc8) is fitted at round `--q1-start` (3) over the base AND every chain plane
  together (the format carries one `lo/hi` per channel per texture), from the 0.1 / 99.9 percentiles (`--q1-range pct`),
  then frozen. From then on block (b) is four-colour Gauss-Seidel sweeps with an EXACT per-channel grid argmin: the
  objective in one value is a parabola `a x^2 + 2 b x`, `x* = -b/a` (rounded to float), and the grid point is the nearest
  VALUE to `x*` (a round as the starting guess, then the two neighbours compared by value, tie to the lower index). At 4 bits this is
  worth 6-7 dB over rounding at the end; at 8 bits it is nearly free.
- **The fit grid must be the grid the GPU samples.** A UNORM8 sampler returns `rep(k)/255` where `rep` is the index with
  its bits replicated into the byte; that equals `k/levels` only at 1, 2, 4 and 8 bits. An uncompressed plane is therefore
  fitted on `rep(k)/255`; a BC plane at 1-3 bits on `k/(2^bits - 1)` (the standard palette decodes it exactly). Get this
  wrong and every 3-bit or 5-7-bit asset carries a systematic bias of up to half a BYTE (1/510 of the span, 0.0034 of a
  [-1, 1] channel at 3 bits) that no PSNR at the texel centres will show you (it was found by a code review, not a
  picture).
- **Zero-change rule.** The device and the host compute a grid value with the SAME arithmetic (`__dadd_rn` /
  `__dmul_rn` on the device, no FMA contraction; `-ffp-contract=off` for GCC/Clang hosts). Before writing, every stored
  LEVEL-1 index is re-quantised and must map to itself; the writer refuses otherwise. Level 0 is not re-quantised: under
  bc8 the file holds the refined blocks (`m.bc0_blocks`) and `m.k0` is only the pre-pack plane's index; under the palette
  mode the index is exact by construction. nvcc WILL fuse `lo + frac * span` into an
  FMA if you write it plainly, and two thirds of the values then disagree in the last bit.

## 5. The bc8 path: an 8-bit continuous level 0 shipped as BC4 / BC5

This is the default and it is where the quality comes from: +0.94 to +4.10 dB over a 4-bit palette at equal memory on
the eight runs of RESULTS 6.2, and +7.4 dB on model36 at one channel (session measurement, 32.96 -> 40.33 dB).

1. Level 0 is solved by block (b)'s machinery: continuous conjugate gradients until `--q1-start`, then the quantised
   four-colour sweeps on its own 8-bit grid (frozen with level 1's, in the same place).
2. After the last round: pack every 4x4 block to BC4 / BC5 (BC4 for one channel, BC5 for two, BC5 + BC4 for three, two
   BC5s for four) with the stb_dxt-style seed (block low/high endpoints, nearest selector), DECODE the blocks back into
   the device plane (the plane the loop carries becomes the BC-decoded plane), and measure `E` there.
3. **Refine the pack by analysis by synthesis** (`--bc-refine 4`): each block's error is the level-0 stencil quadratic
   restricted to that block (`A2` = the block's `H` entries, `A1` = `g` plus the neighbours' current contribution), so
   endpoints are a 2x2 least squares in `(r0, r1)` through the eight-value palette weights, rounded to the 8-bit grid with
   its +-1 neighbourhood tried, and selectors are an exact 8-way argmin per value, two Gauss-Seidel passes; four colours
   over the BLOCK grid; every candidate accepted only on a strict decrease of the block's error, re-weighed in the float
   arithmetic the plane is written in. The refinement searches only the eight-value mode (`r0 > r1`); the seed pack emits
   `r0 == r1` on flat blocks, which is formally the six-value mode and which the refinement reads with the eight-value
   formula (harmless while every selector is 0); the palette mode at 1-2 bits uses the six-value mode on purpose.
   The acceptance is measured on the block's unnormalised, cw-weighted, per-plane quadratic (no mip weight, ridge 0 in
   this assembly, deliberately), in double; the loop's `E` is normalised and measured in float.
   The pack's PSNR against the continuous plane FALLS during refinement: that is the point, the target is the decoder's
   output, not the latent.
4. After the refinement one refit of level 1 and the decoder against the packed plane; then the **outer repack
   passes** (`--bc-outer 2`): refit level 1 and the decoder, re-solve level 0 (the sweeps on the frozen grid; continuous
   plus a snap under `--q1-start 0`), pack and refine again, and KEEP the result only if the shipped `E` strictly fell;
   otherwise restore everything (planes, indices, blocks, decoder, both grids, the report fields) and stop. Then one free
   refit. `--bc-refine-after` (default 0) runs the refinement once more after that; measured worth hundredths.
5. Measured: the seed pack costs 0.45-5.27 dB against the continuous plane; steps 3-4 recover 32-71 % of that. What remains
   is the block format's constraint (eight values on a line per block); the planned, unmeasured lever for it is the pack
   INSIDE every round.

The report, the recon PNGs and `dds_decode.py` all describe the PACKED plane. If you add a step after the outer loop,
note that the device's block buffers (`bc_ep`, `bc_sel`) are not restored on a rejected pass, and that NO function
uploads `m.bc0_blocks` back to the device: `bc_pack_prepare(seed = true)` re-packs from `m.k0`, the PRE-pack plane (the
refinement would be silently discarded), and `seed = false` uses whatever the rejected pass left. Write that upload
before refining again. Also: under bc8 `m.k0` stays the pre-pack plane while the device's `k0` is the nearest index to
the BC-decoded byte; `--bc0 both` writes the former as the "pre-pack twin".

## 6. Initialisation (it decides which basin you land in)

The joint problem is non-convex (the `s c` products), and exact block descent cannot leave a basin, so the start matters
more than in a gradient method.

- Level 1: block means of the source over each 4x4 footprint (`--init box`, the default; channels beyond 3T start at
  zero, so a single RGB texture with `C1 = 4` starts channel 3 dead until the solve gives it a direction; a source
  channel that is an exact copy of an earlier one - a grayscale texture's G and B - is taken only after every distinct
  channel, so a gray first texture does not give level 1 identical, collinear channels whose split only rounding
  would decide; a lone gray texture has nothing else to take and is unchanged) or their PCA
  over the plane (`--init pca`, each component scaled by its max |value|, NOT its standard deviation, which clipped the
  dominant component and cost half a dB). The two tie on single images; on a multi-texture set of unrelated pictures
  `pca` is the one to use (section 6, last bullet).
- Level 0 (`--init0 residual`, the default): level 0 starts at zero, the decoder is fitted with level 1 alone, then for
  each channel in turn: the full-resolution residual `r = out - target` at the base's texel centres, its `nout x nout`
  covariance (centred), the top eigenvector `e` (Jacobi on the host, sign fixed so the largest component is positive),
  channel `k = (r - mu).e / peak` with `peak` the largest |projection| over every plane of the chain, then a decoder
  refit, which is the deflation (what the channel explains leaves the residual through the fit, not by subtraction). `e`
  is computed on the base and reused for every mip plane so a channel means the same direction at every level. Channel 0
  is not assumed to be luma; on a photo it comes out luma-like, on a normal map it is an axis. Under the palette mode the
  seed is snapped to the palette; under bc8 it stays continuous and `m.k0` is zeros until the freeze.
- Why it matters: with channels beyond the first started at zero, 4 + 4 channels on md1-md4 was WORSE than 3 + 4 (a
  zero column gets no weight, the alternation must invent the channel from nothing). With the residual seed 4 + 4 beats
  3 + 4 by 0.9-4.6 dB per texture (DESIGN 3.4; the md logs are not in the tree).
- Known limit: the seed is greedy in residual VARIANCE. When level 1 is starved (a 3-channel normal map with `C1 = 2`),
  the largest residual direction is a near-constant plane (Z) and the first level-0 channel is spent on it (-1.2 dB);
  with `C1 = 4` the same image gains 2.3 dB. On deliberately UNCORRELATED sets (three unrelated pictures) 3 + 3
  channels reach only 25-30 dB where the pair reached 39 (session measurements, RESULTS 7.2) - and the
  per-texture seed that looked like the fix is NOT one, measured: `--init0 texture` dedicates each channel to the
  picture with the most remaining residual and the run lands in the same place (24.74 / 30.52 / 26.56 against
  24.92 / 30.42 / 26.27, E 1.058e-03 against 1.041e-03). What binds there is the LEVEL-1 init: `box` gives level-1
  channel `j` to source channel `j`, so on a multi-texture material texture 0 gets the whole of level 1 and every other
  picture gets none, and texture 0 therefore has the smallest residual at the moment level 0's channels are handed out
  and is seeded nothing. `--init pca`, which spreads level 1 over the material's own principal directions, takes E from
  1.0412e-03 to 7.465e-04 (-28 %) at 3 + 3. See `docs/RESULTS.md` section 7.2.

## 7. The mip chain

Every mip plane of BOTH latents is a parameter (nothing is filtered into existence); plane `m` is solved by the same
three blocks against mip `m` of the run's OWN source chain (`docs/DESIGN.md` 4.2: the iterated 2x2 box by default --
floor halving, a 1-wide plane repeats its row -- or the per-texture filter a material asked for). The 1:1
rule: output mip `m` reads mip `m` of both latents. Level 1's natural LOD sits two below level 0's because it has a
quarter of the texels per axis; without a correction the decoder is fed level 1's mip 0 plane against level 0's mip 1
and the result goes splotchy from mip 1 down. The viewers read level 1 with `SampleGrad` / `textureGrad`, both UV
derivatives multiplied by `2^lod_bias_level1` = 4, which puts its LOD exactly on level 0's; a sampler `MipLODBias = +2`
does the same on hardware that honours it under magnification, but blurs on Intel Xe (README, "the two ways to apply
level 1's mip shift"). The JSON carries `lod_bias_level1: 2`. Planes are independent given `W`, so they run on separate streams; the mip weights govern only how
`W`'s capacity is split between levels. Trilinear and anisotropic filtering need no training change: they are blends of
samples and the decoder is affine in the samples.

## 8. Determinism and the reductions

Two identical runs produce byte-identical files (a gate). Every reduction that feeds a result is a fixed-grid two-stage
reduction (per-block partials to a buffer, summed in index order): the decoder's normal equations, `E`, the covariances,
the CG dot products. The only atomics are integer counters and one double sum that feeds the LOG only. There is no seed
and no random draw anywhere; the only LCG is `--check`'s finite-difference probe, fixed-seed, and it restores every value
it pokes. The atomics, precisely: two unsigned moved-counters and two doubles in the BC refinement (the accepted decrease
and the improved-block count), all log-only. Streams: every host-to-device upload a plane stream reads goes through the
master stream plus a fan-out event (the one exception is the source upload in `device_create`, before any stream exists);
a pageable synchronous `cudaMemcpy` is NOT ordered against `cudaStreamNonBlocking` streams (this was a real race).
Two addressing rules coexist and agree to rounding: the solver kernels use the pixel-scaled float rule
(`bilinear_tap_pixel`); the objective kernel, its host twin and the dense twin go through `u` in double. Change one and
the `--check` agreement is what tells you.

**What the six-texture cap cost.** Every kernel's per-output array is sized by `MAX_NOUT = 3 * MAX_TEXTURES` at compile
time, so raising the cap from four textures to six widened those arrays for every run, one-texture runs included, and
nothing is templated on the actual count. The cost was measured rather than argued. `model10` (752x1124, one texture,
the default layout), the two runs recorded in the predecessor tree's commit `a56d651`, which is the source of these numbers:

    before   time 1.793 s total, (a) 128.924 ms / (b) 208.056 ms / (c) 417.736 ms / E 131.639 ms over 74 passes
    after    time 1.784 s total, (a) 121.935 ms / (b) 217.839 ms / (c) 416.082 ms / E 133.083 ms over 74 passes

`psnr 45.65 dB` and `E shipped 1.935153956e-05` on both sides, so the wider cap changed no number in the asset and cost
nothing measurable in time: the nine milliseconds between the two totals are smaller than the spread between two runs
of either.

## 9. Gotchas, in the order they have actually bitten

1. `E` must not rise across a block; if it does after your change, the block is not exact any more (or you broke the
   four-colour independence). The freeze round's block (b) is the one exempt step (a constraint being applied); level 0's
   snap under bc8 happens in the same place, before block (b), so that round's block (c') is NOT exempt and must not be.
   Block (a) is the one that holds the line by MEASURING rather than by being exact: it is exact in `E` plus its ridge,
   and a near-singular `A` can make the ridge the binding term (section 3). If you touch its ridge, its solve or the
   `Q` test, the image that catches it is the one the gate writes: a black field with one white pixel, where the ridge
   moved `E` by 13 % of itself every round. A tolerance is not a fix there; the objective's own quadratic is free.
2. The site set and the texel addressing: `x = ((px + 0.5 + dx) * w1) / w0 - 0.5` without a round trip through `u`;
   rounding through `u` in fp32 is relative to `u`, so the error in `x` grows with the plane, about 1e-4 of a texel at
   2888 wide, while the pixel-scaled form is exact at `w == w0`.
3. Level 0's stencil accumulators are sized by `MAX_MONOMIALS` (a `static_assert` ties it to `MAX_CHANNELS`); it was
   sized for 2 channels until the predecessor tree's v0.7b, and 3-4 channel layouts overran it.
4. The report's numbers must be of what ships: after any packing step, decode the packed blocks into the recon before
   printing or writing PNGs.
5. The level-1 range is fitted over the whole chain; a chain plane whose distribution differs from the base's can be
   clipped by the percentile fit. `--q1-range minmax` is the diagnostic.
6. `--mip-weight`, `--mip-mix` affect ONLY block (a) - but that is the one shared `W`, so they decide what a deep plane
   is ALLOWED to decode, not merely how well. Under the default `pixels` / `0.5`, planes M4 and below of a nine-plane
   chain hold 0.59 % of block (a)'s mass between them: if the base does not need a full-resolution path to some output,
   `W` will not build one and every deep plane is held to what level 1 carries alone for that output, however well its
   own solve runs. When a chain is wrong and its own solve, init and range are all clean, `--diag`'s decoder table is
   the next place to look - a texture with no level-0 figure has no full-resolution path at all (`DESIGN.md` 4.3.1).
7. More channels can lose to fewer at the same memory, and there are two different reasons. At the BASE it is a
   basin/init problem (4 + 4 vs 3 + 4 on md1-md4 before the residual seed). In the CHAIN it can instead be the decoder
   being re-allocated: mg1-4 at `--c0 2 --c1 4` buys 11 dB of texture 0's base and loses up to 16 dB of texture 1's
   deep planes, because the fourth level-1 channel lets `W` serve that texture without level 0 and the deep planes
   cannot afford that. E falls either way; the difference is which levels the asset is for.
8. The viewer's shader decodes at most `nin <= 24`, `nout <= 18`; the encoder's caps are `C0, C1 <= 4`, `T <= 6`.
   `MAX_TEXTURES` in `model.h` is the one constant and `MAX_NOUT` is three times it; the shader's `W`, `bias` and
   `outv` are sized by the same number and `viewer/main.cpp` refuses a shape past it. The layout does NOT grow with
   `T`: six textures share the same latent planes, at most four channels each, that four textures share.
9. The block refinement's endpoint least squares has a determinant guard relative to `pap * qaq`; a plain `> 0` admits
   rounding noise on rank-1 (edge) blocks.
10. The bitrate report counts bytes per texel; BC levels are whole 4x4 blocks, so odd chain planes are understated by
    the partial blocks (to be fixed).
11. `--mip-mix 1` with mips on is refused: it would drop the base from block (a) entirely.

## 10. How to verify a change (the gate, `tests/run_checks.py`)

`run_checks.py` builds nothing (a warning-free Release and Debug build is a manual step before it). `E` monotone over
every round that ran on the test image (the loop may stop on `--tol`) and over every round of the near-singular case
the gate writes itself, a black field with one white pixel at three extents; device `E` equals the host brute force to 1e-9; block (b)'s CG residual, a dense direct solve on a 64x64 crop and a finite-difference gradient at the solution;
every stored level-1 index re-quantises to itself; the independent Python decoder within 1 count at every mip level and
its per-level PSNR equal to the report to 0.01 dB; the BC pack lossless at 2 and 3 bits (1 bit is not exercised) and the
two-file layout's sizes; the
viewer's frame from a BC file byte-identical to its own load-time pack of the uncompressed twin; two encodes
byte-identical; the outer loop's restore leaving `E` equal to the pre-pass value on a rejecting run; no forbidden pattern
in the tree. Run it after every change. Then look at the image in the viewer: PSNR at the texel centres is not the whole
story, and the owner judges by eye.
