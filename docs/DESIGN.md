# DESIGN.md -- the NNTC method

What the encoder minimises, how, and why each step is exact. The asset it produces is specified in `docs/FORMAT.md` and
the measurements are in `docs/RESULTS.md`. This file describes the code as it stands.

## 0. The representation

One material of `T` textures (`nout = 3T` output channels) is stored as two quantised latent textures and one affine
decoder:

* **level 0**, the selector plane, at the full decoded extent: `C0` channels of a continuous 8-bit value on a
  per-channel affine grid, BC4 / BC5-encoded (the default, `--l0 bc8`, sections 3.5 and 3.6), or `C0` channels of
  1-4 bits each on a uniform palette (`--l0 palette`, section 3.3). **Under the default the shipped `.dds` holds
  OPTIMISED BC4 / BC5 blocks**: the pack is not a post-process applied to a finished plane but a representation the
  encoder goes on optimising - every block's endpoints and selectors re-chosen against the decoder's own output error
  (3.6), and the whole model refitted and repacked in outer passes taken only when the shipped objective falls. A
  consumer binds those blocks and samples them; nothing is decompressed and nothing is packed at load;
* **level 1**, the colour plane, at a quarter of the extent per axis, `C1` channels of 4-8 bits on a per-channel affine
  grid;
* **the decoder**, one affine map `out = W phi(z) + b` over `phi = [c_j ; s_i ; s_i c_j]`.

Both textures are read by the hardware's own sampler, one `Sample()` each, with whatever filter the consumer selects;
the dequantisation of both is affine and is applied after the blend, so the hardware's blend of quantised texels is the
blend of the values (`docs/FORMAT.md` section 6). That is the property the whole design rests on: it is what makes a
quantised latent samplable at all, and it is why the encoder's model of the GPU is exact rather than approximate.

## 1. The objective

`E` is the weighted squared error of the decoded material over a set of **sites**, and a site is a position, not a
texel. Every pixel `p` of every stored plane `m` carries:

* one **centre** site at `u = (p.x + 0.5) / W_m`, `v = (p.y + 0.5) / H_m`, whose target is `src_m(p)`, the run's own
  source chain at that level (4.2). At that position level 0's filtered sample collapses onto texel `p` itself -- level 0 is at the
  plane's own extent, so the blend has weight 1 on one tap -- and level 1 blends four of its texels. The code reads
  level 0 directly there for that reason and calls it the nearest read; that is a fact about the position, not about
  the sampler state;
* `K` **fractional** sites at `u = (p.x + 0.5 + dx_j) / W_m`, `v = (p.y + 0.5 + dy_j) / H_m`, where **both** levels are
  filtered by the hardware's bilinear rule with the clamp, and the target is the **source** read by that same rule at
  that same position.

**There are two addressings in the tree and they are not the same one.** The SOLVER's fp32 path carries a site's
position as the pixel of the plane being fitted, scaled straight into the plane being read,
`x = (p.x + 0.5 + dx) * W / W_m - 0.5`, rather than through a normalised `u`: the round trip through `u` rounds twice
in fp32 and the first rounding is relative to `u`, so the error in `x` grows with the plane - about 1e-4 of a texel at
`W_m = 2888`. At `W == W_m`, which is how level 0 and the source are read, the scaling is exactly 1 and the addressing
is exact. The objective's double pass and the host twins that check it go through `u` instead, in double, where the
round trip costs nothing that a float comparison can see; keeping the two spellings is deliberate, since a check that
computed its position the same way would not be checking the position.

The fractional sites are the point of the whole exercise. A texel-by-texel codec is right at the grid and free to be
wrong between grid points; a texture is sampled between grid points almost everywhere. Fitting fixed fractional
positions is what makes the representation valid under the sampling operator the GPU applies rather than only at the
texel centres -- and, since a fractional site's level-0 sample is a blend of four selector texels, it is also what gives
the selector plane a reason to be smooth where the image is smooth.

```
E = sum over planes m of  w_m * sum over the sites of plane m of  sum over outputs c of  cw[c] (out_c - t_c)^2
out = W phi(z) + b
```

`cw[c] = weights[c / 3] * rgb_weights[c % 3] * 3 / sum(rgb_weights)` is the output's own weight -- the texture's weight
inside the material times the channel's weight inside the texture, the three rgb weights normalised to mean 1 so that
changing their balance does not change the scale of `E`. `w_m` is the per-site mip weight of section 4.3, the same for
every site of a plane. Centre and fractional sites carry equal weight, so at `K = 4` the fractional share is 4/5.

`E` is reported normalised by `sum(cw)` times the total weighted site count, so it reads like a mean squared error in
[0,1] units: at `K = 0` with one stored plane and equal weights it IS the plain mean squared error of the decode at the
texel centres. The encoder PRINTS that comparison - `E` beside the centre mean squared error of the same pass - and
`tests/run_checks.py` is what asserts the two agree.

**The subtexel offsets are fixed, not drawn.** The base pattern is the standard rotated grid,

    P = { (-3/8, -1/8), (-1/8, +3/8), (+1/8, -3/8), (+3/8, +1/8) }

in units of one pixel of the plane, turned a quarter turn per texel by `(dx, dy) -> (-dy, dx)` applied
`n = (ty & 1) * 2 + (tx & 1)` times, so the four texels of a 2x2 cell carry the four rotations. `P` is invariant under
that quarter turn, so at `K = 4` the rotation only permutes a texel's own four sites; it bites at `K = 1`, where the
single offset `(+1/4, +1/4)` then visits a different quadrant in each texel of a 2x2 cell. `K = 2` is the diagonal pair
`(-1/4, +1/4)`, `(+1/4, -1/4)`, unrotated. `K = 0` is centres only. Eight numbers, no seed, nothing to reproduce.

## 2. The feature vector

    phi = [ c_j ; s_i ; s_i c_j ]        nin = C1 + C0 + C0 * C1

`s` is level 0's dequantised sample, `c` is level 1's, and the output is `W phi + b` with the identity on the end: no
activation, no hidden layer, nothing to invert. Three reasons it is exactly this:

* **the cross terms are the model.** With only `[c_j ; s_i]` the decoder would be one affine map and level 0 could do
  nothing but add a fixed direction per channel to every texel. The products let level 0 *modulate* level 1:
  `out = (b + B s) + (A + sum_i s_i M_i) c`, so the selector plane chooses, per texel, which linear combination of the
  colour channels is applied. That is what a 1-4 bit plane at full resolution can usefully carry;
* **it stays affine inside each block.** `out` is affine in `c` with `s` held and affine in `s` with `c` held, which is
  what makes both plane solves exact quadratics (section 3) instead of gradient descent;
* **the identity output keeps everything from the texel to the pixel affine.** A sigmoid or a leaky ReLU on the end
  would not break the sampling argument -- it is applied after the blend -- but it would make every block solve
  inexact, and it buys nothing measurable at these rates.

`nin` is 14 at the default layout (`C0 = 2`, `C1 = 4`), so a single-texture decoder is 45 numbers. The feature order is
a contract with the consumer and is written out in `docs/FORMAT.md` section 4.

## 3. The three block solves

The unknowns are the decoder `(W, b)`, level 1's plane and level 0's plane. Hold two of the three and `E` in the third
is something that can be minimised exactly:

1. **(a) the decoder** -- a global linear least squares over every site of every plane;
2. **(b) level 1** -- a sparse weighted least squares on a 9-point block stencil, solved by preconditioned conjugate
   gradients while it is continuous and by exact quantised sweeps once its grid is frozen;
3. **(c) level 0** -- a per-texel exact search over the palette, in four colour passes; or, under `--l0 bc8`, block
   **(c')**, which is (b)'s own least squares over level 0's plane instead (3.5).

Each is the exact minimiser of `E` in its own variables with the other two held -- (a) up to the ridge its normal
equations carry, which is why it also MEASURES its own step and keeps the previous decoder when the ridge and not the
data decided it (3.1) -- so **`E` cannot increase across any of them** and the round loop is monotone by construction
rather than by tuning. A rise beyond a loose tolerance (a relative 1e-6 plus an absolute 1e-8) prints a warning
naming the round and the block; it is deliberately not an assert, since it is a floating-point judgement. The one exception is the round that freezes the grids: snapping a plane onto
a grid applies a constraint rather than taking a step, and a constraint can only cost `E`. (The post-fit BC pack of
`--l0 bc8` is the same kind of step and is outside the loop entirely -- 3.5.)

### 3.1 Block (a): the decoder

With `v = [phi ; 1]` of length `mm = nin + 1`, the minimiser of `E` in `(W, b)` is the solution of the `mm x mm` normal
equations

    A     = sum over sites of  omega_site * v v^T          one matrix, shared by every output
    rhs_c = sum over sites of  omega_site * v t_c          one right-hand side per output
    [W_c b_c] = A^-1 rhs_c

`cw[c]` scales output `c`'s own equation uniformly and therefore cannot move its argmin, which is why it does not
appear; the per-site mip weight `omega` does not cancel, and is what makes a deep plane count for less (or more) than
the base. `A` is symmetric, so only its upper triangle is accumulated -- a third fewer multiply-adds, and few enough
entries that the walk is one pass with one entry per thread. The accumulation is a two-stage reduction on a fixed grid
(section 6), and the small dense system is solved on the host by Gauss-Jordan with partial pivoting in double, with a
ridge of `1e-9 * mean(diag A)` so that a constant feature -- level 0's second channel starts at zero everywhere --
cannot make `A` singular.

**What the ridge costs, and what the block therefore guarantees.** The solve returns the minimiser of `E` plus
`ridge ||[W b]||^2`, not of `E`, and those are the same point only while the ridge is negligible against the solution.
On an image that drives `A` nearly singular they are not. A 100x36 black image with one white pixel at (50, 18) pushes
level 1's channels to a span of 0.155 -- nearly constant, so the columns of `A` are nearly dependent -- and the exact
minimiser's norm grows past a hundred; the ridge's own penalty then moves the solution far enough that `E` RISES across
the block. Measured at the default layout: `E` 1.875599e-06 after round 8's block (c'), 2.112583e-06 after round 9's
block (a), a 13 % rise, and again every round to the end. It is not a tuning problem: at `1e-6` the same run ends six
times worse (`E` 8.800630e-06 against 1.268264e-06 at round 20) and still rises, and at zero `A` is singular at the
init, where level 0 is a constant plane, and the run collapses at `E` 8.951838e-05 in one round.

So the block measures instead of asserting, and it costs no objective pass to do it. `A` and `rhs` ARE the quadratic
`E` is in `(W, b)`, up to the constant `sum omega cw t^2`, so

    Q(x) = sum_c cw[c] ( x_c^T A x_c - 2 x_c . rhs_c )

differenced between two decoders is the change in `E`'s weighted sum -- up to the fp32 rounding of the features `A` and
`rhs` were accumulated from, `E` itself being measured in double from the same planes, about 1e-12 in the printed units
at a decoder norm of 300 -- and dividing by `sum(cw)` times the weighted site count gives the change in the `E` the
round loop prints, a few thousand double multiply-adds on numbers already on the host. The shipped ridge is tried
first, so every run whose `A` is well conditioned gets exactly the decoder it always got, byte for byte; if that
candidate raises `Q` by more than `Q`'s own rounding (`1e-12` in `E`'s units, five orders of magnitude under the effect
measured above) the ridge and not the data is deciding the step, and `1e-12 * mean(diag A)` and then a zero ridge are
tried in turn; if none of them clears the bar the decoder the block came in with is kept and the block takes no step.

`Q`'s rounding is bounded rather than assumed, because the lower rungs are reached precisely when `A` is near-singular
and the solutions are large: beside each `Q` the block computes `eps mm (sum_ij |x_i| |a_ij| |x_j| + 2 sum_i |x_i|
|rhs_i|)` in the same units, for the incoming decoder and the candidate both, and takes the candidate only when
`dQ + err_prev + err_cand <= 1e-12`. At `|x|` of 1e6 to 1e8 the computed `dQ` is noise of random sign, and a comparison
that cannot resolve the sign is a refusal, not a pass. A non-finite solve is refused before `Q` is evaluated. The
report's `block (a) ridge:` line counts how many of the run's calls ended on each rung.

**Block (a) therefore guarantees that `E` does not rise across it beyond the tolerance -- by construction, not by
argument** -- and on the dot image above the fallbacks are what the loop needs rather than a
mere safety net: it ends at `E` 4.627044e-07 and 60.46 dB where it ended at 1.268264e-06 and 53.69 dB before.
`tests/run_checks.py` writes that image and encodes it as a gate.

### 3.2 Block (b): level 1

Hold level 0 and the decoder fixed. The decoder is affine in its features and the features are affine in the level-1
sample, so `E` is an exact quadratic in the plane's values and the plane's exact minimiser is one linear system.

Split `W`'s columns by the feature order `[a: c_j][b: s_i][sc: s_i c_j]`: `A` is the a-block (`nout x C1`), `B` the
b-block (`nout x C0`), and `M_i` the `sc` rows of selector channel `i` (`nout x C1`). At a site whose level-0 sample is
`s` and whose level-1 taps are the texels `t` with weights `w_t`,

    out  = m(s) + G(s) . sum_t w_t c_t ,   m(s) = b + B s ,   G(s) = A + sum_i s_i M_i     (nout x C1)

The tap weights are accumulated BY TEXEL INDEX, so a site on the edge whose two clamped taps are the same texel gives
that texel the sum of both weights - the hardware's clamp-to-edge rule, and the rule `E` itself is measured under.

Writing the whole plane as one unknown vector, the weighted normal equations over its sites are

    H[t][t'] += w_t w_t'  G^T diag(cw) G           (a C1 x C1 block per texel pair)
    grad[t]  += w_t       G^T diag(cw) r ,         r = out(c_prev) - target

with every site of the plane carrying weight 1: the mip weights govern only how much of the shared decoder each plane
gets, and a plane's own level-1 solve is independent of every other plane's.

The step is **proximal**. The unknown is the correction `d` from the current plane, minimising `E(c_prev + d) +
lambda |d|^2`, so `grad` is evaluated at `c_prev` and the system is

    (H + lambda I) d = -grad ,   c = c_prev + d ,   lambda = --ridge * mean(diag H)

The ridge is not cosmetic. With one texture (`nout = 3`) and `C1 = 4`, every site's `G^T diag(cw) G` has rank at most 3
in a four-dimensional space, so a texel whose footprint is flat in `s` has a genuine null direction. Without the ridge
the solve wanders along it and blows up that channel's `lo/hi` range, which costs every other texel a quantisation step.
The smallest and largest diagonal entry of `H` are reported per plane for exactly that reason: a degenerate layout
shows up there as a minimum several decades below the maximum.

**The stencil.** A site reads four level-1 texels, so two texels couple only if they are within one texel of each other
in both axes: a 9-point block stencil. `H` is symmetric with `H[t][t'] = H[t'][t]^T`, so each texel stores five blocks -
its own diagonal and the blocks to the right, down-left, down and down-right - and the other four directions are read as
the transpose of the neighbour's block. That is `5 C1^2` **doubles** per texel -- 160 bytes at `C1 = 2`, 640 at `C1 = 4` -- because the stencil is
assembled and iterated in double for the reason the solver paragraph below gives.

The assembly is a **gather**. One thread per level-1 texel walks every site whose bilinear footprint can reach it - a
band of nine pixels per axis, and for each pixel the centre site and the `K` fractional sites - recomputes `s`, `G` and
the tap weights, and accumulates into its own blocks in double. No atomics, no contention and no dependence on how the
blocks were scheduled; the price is that each site is visited by each of the four texels it touches.

That price is paid in the cheap currency, because the expensive part of a site's contribution does not have to be
recomputed at all. `G` is affine in the site's level-0 sample, so `G^T diag(cw) G` is a **quadratic form** in it:

    G(s)    = A + sum_i s_i M_i
    G^T C G = P_0 + sum_i s_i P_i + sum_{i <= j} s_i s_j P_ij

with `P_0 = A^T C A`, `P_i = M_i^T C A + A^T C M_i`, `P_ii = M_i^T C M_i` and `P_ij = M_i^T C M_j + M_j^T C M_i`. At
`C0 = 2` that is six fixed `C1 x C1` matrices and at the channel cap of 4 it is fifteen; they depend on the decoder
alone and are built once a round on the host. What varies from site to site is that many **scalars**. So a texel
accumulates one weighted sum per monomial per stencil slot and expands them against the matrices once, at the end, for
itself: the double arithmetic per site per touching texel falls from a `C1 x C1` matrix product and its scatter to a
handful of multiply-adds per slot, and the thread's accumulator falls from five `C1 x C1` blocks to five rows of
monomials - which is the difference between spilling to memory and staying in registers. The
gradient has no such structure (its residual is not affine in anything the decoder fixes) and is still formed per site;
it is `C1` rows against `nout`, not `C1 x C1`, so it was never the cost.

**The solver.** Conjugate gradients preconditioned by the block Jacobi of the stencil: each texel's diagonal block plus
`lambda I` is inverted once in double and kept as floats, and the iteration needs only the stencil mat-vec, two vector
updates and three dot products, each a deterministic two-stage reduction in double. The stencil is a hat-function Gram
matrix rather than a Laplacian, so its condition number does not grow with the resolution and the iteration count stays
in the tens. The preconditioner is float; the stencil and the iteration's own vectors are double, because in the
degenerate layout above the condition number is the largest diagonal of `H` over `lambda` and a small residual then
permits a much larger error.

### 3.3 Block (c): level 0 as a palette index (`--l0 palette`)

Hold level 1 and the decoder fixed and ask, for one level-0 texel, which of its palette states minimises `E`. Only the
sites whose level-0 footprint touches that texel can move at all, and over those sites `E` is an exact quadratic in the
texel's own `C0` dequantised values.

The decoder is affine in `s` for a fixed level-1 sample `c`, so at a site

    out = b + A c + (B + sum_j c_j M_j) s

Split the site's level-0 sample into this texel's part and the rest, `s = beta + alpha x`, where `x` is the texel's own
value vector, `alpha` the weight this texel carries in the site's blend and `beta` what the other taps contribute. Then

    out(x) = p + alpha Q x ,   p = the decode with this texel's channels at zero,   Q[:,i] = B[:,i] + sum_j c_j M_i,j

and with `r = p - target` and `C = diag(cw)`,

    E(x) = A0 + 2 x.A1 + x.A2.x ,   A0 += r^T C r ,   A1 += alpha Q^T C r ,   A2 += alpha^2 Q^T C Q

which is the quadratic `C0 + 1` decodes per site would produce - one at `x = 0` and one per channel moved by `alpha` -
with the decodes replaced by the affine structure they were sampling.

**The sites of one texel.** Level 0 is at full resolution, so a texel is a pixel:

- its own **centre** site, where the filtered level-0 sample collapses onto this one texel (level 0 is at the decoded
  extent, so the blend has weight 1 on it): `alpha = 1`,
  `beta = 0`, and the site depends on no other texel;
- the `K` **fractional** sites of each pixel of the 3x3 neighbourhood, where level 0 is read bilinearly. A position
  inside a pixel addresses the two texels either side of it in each axis, so the pixels whose four-tap footprint can
  reach this texel are exactly this pixel and its eight neighbours. A texel that appears twice among the four taps
  because the clamp pulled both indices onto it carries the **sum** of the two weights, the same rule `E` is measured
  under.

A site whose `alpha` is zero adds the same constant to every state and is skipped, so `A0` is the movable part of the
texel's share of `E` plus a constant - which is all an argmin needs.

**The argmin is exact.** With `sum(bits0) <= 8` the joint enumeration walks all `prod_c 2^bits_c` states (at most 256)
and takes the best; above that it is `--sweeps` coordinate sweeps, each channel in turn over its whole alphabet with the
others held, which is exact in one channel but not jointly. Ties keep the current state: only a strict decrease moves a
texel, which is what stops two equal-valued states from cycling.

**The four colour passes.** A pass moves the texels of one `(tx & 1, ty & 1)` class, one kernel launch each. Two texels
of one class are two apart in an axis and a site's four taps are adjacent, so no site can touch two of them: within a
pass the texels' quadratics are independent, and the pass is one exact joint minimisation over all of them rather than a
sequence of single-texel steps. That is also why the kernel may write its texels in place while reading its neighbours -
every neighbour a site of this pass reads belongs to another class.

The accumulators are double. A texel sees at most `9K + 1` sites -- its own centre site, plus the `K` fractional sites
of each of the nine pixels of its neighbourhood -- so the wider accumulation costs nothing beside
the sampling around it, and the argmin compares states that can differ in the last bits of a float - a comparison the tie
rule then turns into a decision about whether a texel moves at all.

### 3.4 The round loop

    init:   level 1 = --init box (or pca)
            level 0 = --init0 residual: every channel at zero
            block (a)
            for k = 0 .. C0-1:  channel k = the leading principal direction of the residual,
                                projected onto every plane, then block (a) again - and, under
                                --l0 palette only, one block (c) and one more block (a), so the
                                snap is chosen by the objective
            (--init0 luma instead: channel 0 = the snapped source luminance, the rest at zero,
             and no seeding loop)
    round r = 1 .. --rounds:
            (a) the decoder's least squares, over every site of every plane      E_a
            (b) level 1, EVERY plane m = 0..M, each on its own:
                          r <  --q1-start   continuous, ridged                   E_b <= E_a
                          r == --q1-start   fit the grid over every plane together, freeze it,
                                            snap every plane, then the sweeps
                          r >  --q1-start   --q1-sweeps quantised four-colour sweeps per plane
            (c) level 0's exact search, EVERY plane m = 0..M                     E_c <= E_b
            stop when (E_prev - E_c) / E_c < --tol twice running, or when (c) moved no texel of any
            plane and (b) moved no stored value of any plane
    the end: one more block (b) per plane - the sweeps on the frozen grid, or, at --q1-start 0, the one
             fit and snap of the continuous planes
             one more level-0 search per plane against the values level 1 now holds
             ONE last block (a) over every plane, so W is optimal for what the asset holds

Each block minimises `E` in its own variables with the other two held - blocks (b) and (c) exactly, block (a) up to the
ridge and the acceptance test of section 3.1, which is what makes "`E` does not rise across block (a) beyond the
tolerance" the statement rather than "block (a) is the exact global minimiser" - so `E` is non-increasing across every
one of them and the whole sequence is monotone by construction - a warning line naming the round and the block
when one rises beyond the loose tolerance, never an assert and never a tuning knob. The one exception is block (b) of the round that freezes level 1's
grid (section 5): snapping the plane onto the grid is a constraint being applied, not a step being taken, and a
constraint can only cost `E`. The progress line carries all three values, the base's centre and sampled PSNR, the
chain's per-level centre PSNR (from the same device pass, so a round costs no second decode to report it), the texels
block (c) moved and the level-1 values block (b) moved - both summed over every plane, and the level-1 count meaning a
correction above half a grid step while the plane is continuous and an actual change of stored index once it is not -
the smallest diagonal of `H` over every plane, and the round's kernel time. The freeze prints a `grid frozen:` line
naming the round, the policy and every channel's `lo/hi` and step.

**The residual init (`--init0 residual`, the default).** Level 0's channels are seeded one at a time from what the
decode still gets wrong, in the owner's words: *PCA, project, deflate, PCA again, project*. Level 1 is initialised
first and level 0 starts at zero everywhere, so the first block (a) fits the decoder to what level 1 alone can say.
Then, for `k = 0 .. C0-1`:

1. the residual `r(p) = out(p) - target(p)` at the base plane's own texel centres, `3T` numbers per pixel, in the
   decoder's own arithmetic and without the saturate the recon PNGs go through - the objective is measured on the
   unclamped output, so this is the error the next block (a) will see;
2. its mean `mu` and its `3T x 3T` covariance over the plane - the covariance of the CENTRED residual,
   `E[(r - mu)(r - mu)^T]`, a device reduction into doubles exactly as the pca init's is;
3. its leading eigenvector `e` (Jacobi on the host, the same routine), with the SIGN of `e` fixed so that its largest
   component is positive. An eigenvector is defined up to its sign and Jacobi's choice of one carries no meaning;
   fixing it makes the reported direction readable - a luma-like direction comes out positive rather than negative half
   the time - and changes nothing else, because the decoder's own column absorbs it;
4. channel `k` = `(r(p) - mu) . e / D`, the CENTRED projection. A constant offset is the decoder's bias's business and
   not a channel's, which is why the mean comes out before the projection and not after it. The channel is then left
   continuous under `--l0 bc8`, where level 0's grid does not exist until `--q1-start` fits it, and taken to the
   nearest value of the channel's palette under `--l0 palette`;
5. block (a) again;
6. **under `--l0 palette` only**, one block (c) - level 0's exact per-texel search - and one more block (a).

**The divisor `D` is not the same number in the two modes, and that is the whole of the palette mode's scale.** Under
`--l0 bc8` it is the `peak`, the largest `|(r(p) - mu) . e|` over **every plane of the chain**, so the channel fills
`[-1,1]` and nothing is clipped - the scaling rule the pca init settled on, for the same reason. Under `--l0 palette`
it is the **99 % percentile** of that same magnitude instead, read off a 4096-bin histogram on `[0, peak]` and taken at
the bin's upper edge, with the clamp that follows saturating the hundredth of the texels above it.

The reason is the grid, and it is the reason the same seed was worth +2 dB in one mode and -5 dB in the other. A
residual's projection is heavy-tailed: on `model10` the peak is 1.06 while the standard deviation along `e` is 0.064,
so dividing by the peak leaves a channel whose mass sits inside a fiftieth of `[-1,1]`. Under `--l0 bc8` that costs
nothing - `--q1-start` fits `lo/hi` to whatever plane the solve produced and the stored 8-bit index spans it, so the
seed's scale is a gauge the freeze removes. Under `--l0 palette` there is no such freeze: the palette is fixed at
`pal(k) = -1 + 2k / (2^b - 1)` on `[-1,1]`, a seed using a fiftieth of that range lands on the two or three entries
around zero with its detail already gone, the refit fits a large decoder weight to the small channel that is left, and
the run settles where level 0's alphabet is barely used and the decoder's gain on it is correspondingly high - which is
also why the BC4 / BC5 pack of a 4-bit plane, whose error is in the plane's own units, came out multiplied in the
output. The percentile was chosen by measurement: on `model10`, `--l0 palette --bits0 4`, shipped `E` at
99.9 / 99 / 98 / 95 / 90 % was 6.86 / 6.65 / 6.70 / 6.85 / 7.22 e-05 at `C0 1` and
3.55 / 3.14 / 3.24 / 3.53 / 3.33 e-05 at `C0 2` - the percentile alone, taken before step 6 below existed. 99 % wins
both, and is what `INIT0_PERCENTILE` in `src/model.h` holds.

**Step 5 is the deflation.** Nothing is subtracted from the residual. What the new channel can explain leaves the
residual because the refit is free to use the channel and does, so the next channel's covariance is taken on a residual
that has already been given the chance to spend this one. A subtraction would deflate along `e` whether or not the
decoder could use that direction at full resolution, and the decoder, not the covariance, is what decides.

**Step 6 exists because a snap is not a minimisation.** The seed writes the nearest palette entry to a number, chosen
without the decoder, level 1 or a single site in sight, and the residual that channel `k + 1` is then read off carries
the whole of that snap's own error - which lies along channel `k`'s own direction. So the leading eigenvector came back
pointing where it already pointed: on `model10` at `C0 2` the first two directions were `+0.4356 +0.6500 +0.6227` and
`+0.4413 +0.6594 +0.6087`, a cosine of 0.9995, where `--l0 bc8` on the same image picks `+0.8924 -0.4134 -0.1808` for
the second. Two collinear channels are a near-singular pair of columns in block (a)'s normal equations; the decoder
fits a large, nearly cancelling combination of them, and a level-0 plane whose stored values are then perturbed by the
pack comes out of the decoder multiplied. Running one block (c) and one block (a) after each seeded channel makes the
palette assignment the thing the OBJECTIVE chose, and the direction that comes back next is one the representation is
actually still missing: with the percentile alone `model10 C0 2` ships 41.57 dB, with the search as well 41.74. Both
steps are minimisations, so `E` is monotone across them exactly as it is across the rest of the seed, and
`tests/run_checks.py` asserts both. Under `--l0 bc8` there is nothing to do here - the seed is continuous, no snap
happens, and block (c') is not this search - so the bc8 path is byte-for-byte what it was.

`e` is computed on the **base** plane and reused for every plane of the chain, exactly as the pca init's basis is and
for the same reason: one decoder is shared by every stored level, so channel `k` has to mean the same direction of the
material at every level. The residual and the projection are each plane's own, taken against that plane's own source
mip; the direction is shared, and so is the divisor - the peak and the percentile are both taken over the whole chain,
so channel `k` is on one scale at every level as well as pointing one way.

**Channel 0 is not luma.** It is seeded like every other channel, from the residual of a decode with level 0 still at
zero, so the first direction is whatever the level-1 reconstruction misses most at full resolution. On a photograph that
lands close to luminance anyway (`model10`: `e = +0.436 +0.650 +0.623`, 92.5 % of the residual's variance). On a tangent
-space normal map at `C1 = 4` it does not (`e = +0.113 +0.992 -0.059`, the Y axis). Assuming luminance there is assuming
the answer.

**What it is worth.** The control, `--init0 luma`, leaves every channel beyond the first at zero, and a channel that is
zero everywhere is a dead column in block (a)'s normal equations: block (a) hands it whatever the ridge leaves and the
alternation has to invent the channel out of nothing. The more of them there are, the poorer the basin the loop settles
in - to the point where **more capacity measured worse**. On the four-texture material `md1-md4` at 2048x1024, the same
memory either way at the time the table was measured (`--c0 3` and `--c0 4` both shipped two BC5 files then; `--c0 3`
now ships a BC5 and a BC4 and is a quarter smaller, which does not change what the table compares):

| | per-texture psnr | sampled |
|---|---|---|
| `--c0 3 --c1 4`, `--init0 luma` | 37.55 / 39.79 / 33.05 / 31.39 | 36.18 |
| `--c0 4 --c1 4`, `--init0 luma` | 38.73 / 38.30 / 33.23 / 29.62 | 35.95 |
| `--c0 3 --c1 4`, `--init0 residual` | 41.18 / 43.45 / 35.84 / 31.78 | 38.10 |
| `--c0 4 --c1 4`, `--init0 residual` | 42.09 / 46.39 / 38.13 / 36.36 | 41.78 |

Those four rows, and the single-image figures in the paragraph below them, were measured while the seed was being
chosen and **have no log in the tree**; `docs/RESULTS.md` section 7.2 is the logged form of the same question on the
same material. The inversion is gone - the fourth channel now buys 0.9 to 4.6 dB per texture instead of costing - and
both arms gain.
On single images: `model10` at `--c0 2` 47.80 dB sampled against 45.77, at `--c0 1` 43.30 against 41.15 (`C0 = 1` is
**not** unchanged, because channel 0 itself is now chosen rather than assumed); `normal_map` at `--c0 2 --c1 4`
40.82 dB sampled against 38.54.

**What the palette mode's two steps were worth.** At `--l0 palette --bits0 4 --c1 4 --rgb-weights 9,11,1`, shipped
psnr, against the `--init0 luma` control at the same command line (`out/s9b/*_palette/`, `out/s9g/gate_*/`):

| | `--c0 1` | `--c0 2` |
|---|---|---|
| `model10`, the seed as it was at the predecessor tree's v0.9f | 39.00 | **36.38** |
| `model10`, `--init0 luma` | 38.62 | 41.58 |
| `model10`, the seed as it is now | **39.68** | **41.74** |
| `model23`, the seed as it was at the predecessor tree's v0.9f | 36.69 | **31.72** |
| `model23`, `--init0 luma` | 37.10 | 41.08 |
| `model23`, the seed as it is now | **37.34** | **41.29** |

Two channels beat one again - they did not before, which is what made this a defect and not a tuning question - and
the residual seed is at or above the luma control at every one of the four layouts, on shipped `E` as well as on psnr.
`docs/RESULTS.md` section 6.2 carries the whole eight-run table.

**Where it loses.** `normal_map` at `--c0 2 --c1 2`: 34.51 dB sampled against the control's 35.70. The cause is visible
in the direction it picks - `e = +0.002 -0.026 +1.000`, the Z axis. With only two level-1 channels for a three-channel
normal map, level 1 cannot carry Z at all, so Z dominates the residual's variance and the first level-0 channel is spent
reproducing a nearly constant plane instead of the X/Y detail. The seed is greedy in variance, not in what the rest of
the representation can still be made to do; when level 1 is given the channels to carry the material (`--c1 4`) the same
image gains 2.3 dB. That is a known limit of a greedy leading direction and is left as it measures.

**Two arms on the same seed, and where each lands.** The direction above is the leading eigenvector of the residual's
covariance **over every output of the material**, taken **on the base plane**. Both of those are choices, and both have
a control:

* `--init0 texture` takes the direction from **one texture's own 3x3 block** - the texture whose block carries the most
  residual variance - and leaves every other output at zero, so a full-resolution channel is DEDICATED to a picture
  rather than being a blend of all of them. On one texture the block IS the whole covariance and the two arms are
  byte-identical, which `tests/run_checks.py` asserts.
* `--init0-scope chain` accumulates the covariance over **every stored plane** with the objective's own per-site weight
  instead of over the base plane alone, so a texture whose full-resolution need only appears down the chain is visible
  at the moment the channels are handed out.

Both are controls and neither is the default, because the measurement is not one-sided (`docs/RESULTS.md`, the init0
arms). `--init0 texture` is worth 5 % of E on `mg1-4` at either `--c1` and takes texture 1's base from 51.09 to
58.48 dB at `--c1 3`; on three unrelated pictures it is worth 1 % at `--c0 3 --c1 4` and costs 2 to 5 % at `--c0 3
--c1 3` and `--c0 4 --c1 4`. `--init0-scope chain` is worth about 1 % on a multi-texture material and costs about the
same on every single image, whose deep planes want the direction its base wants anyway.

**What was tried and rejected: weighting the covariance by `cw`.** The channel is a scalar and what the decoder can
take off the error with it is a rank-one approximation of the residual weighed by the objective's own per-output
weights, so the direction "should" be `diag(sqrt(cw)) u` for `u` the leading eigenvector of
`diag(sqrt(cw)) C diag(sqrt(cw))`. It was implemented and measured and it **loses**: `model10` with
`--rgb-weights 9,11,1` went from E 1.314974e-05 to 1.346578e-05 and 44.97 to 44.23 dB, and the three-picture material
split two arms each way. A seed is the point the alternation starts from, not the answer, and a direction that is
better by the objective's lights at step one is not therefore in a better basin. The plain covariance stays.

**The pca init.** Level 1 starts from the block mean of the 3T source channels over each level-1 texel, projected onto
the first `C1` principal directions of those means' own covariance over the base plane (the covariance a device
reduction into doubles, the eigenproblem a Jacobi rotation sweep on the host), each component divided by **the largest
magnitude it actually reaches** so that it fills `[-1,1]` and nothing is clipped. The directions are computed on the
base plane and used for every plane of the chain, because one decoder is shared by every stored level and channel `j`
therefore has to mean the same direction of the material at every level. Against the box init it is what gives the
fourth channel of a single RGB texture a real direction - the image's third principal direction - instead of a plane of
zeros that only the level-0 cross terms could ever move.

Scaling by the standard deviation instead - the obvious choice - clips. A principal
component's values are not bounded by two standard deviations: on a single texture the dominant direction carries most
of the variance (92 % of it on `model10`) and its tails run several deviations out, so dividing by sigma and clamping
flattens every texel in those tails onto the ends and throws away exactly the contrast the first component was chosen to
carry. It cost 0.46 dB on `model10`: 41.48 dB by sigma against 41.94 dB by the peak, same layout, same loop.

**Which init is the default.** box, and only because the two split. At `--c0 2 --bits0 3 --c1 4 --bits1 8` (these four
figures were measured while the mode was being chosen and **have no log in the tree**; `docs/RESULTS.md` section 2 is
the same question asked as a logged ablation arm, at different settings, and lands in the same place): `model10`
box 41.98 dB against pca 41.94; `game2` pca 45.12 dB against box 44.99; the two-texture material `mb1 + mb2`, where the
covariance is 6x6 and all four level-1 channels get a real direction, 35.29 / 36.89 dB from both to the second decimal.
No image is hurt by either, so the default is the one that needs no eigenbasis to explain, and `--init pca` is one flag
away for the image that likes it.

### 3.5 Block (c'): level 0 as a continuous 8-bit plane (`--l0 bc8`)

**This is the default mode.** Level 0 ships as BC4 or BC5, and **a BC4 plane costs four bits a texel whatever the
palette depth**. At `--bits0 4` the block format is therefore being paid for and not used: a 4x4 block carries two
endpoints and eight values along the line between them, and 3.3 hands it sixteen fixed levels on a palette chosen
before anything was measured. `bc8` hands it a continuous plane instead. It is the same arrangement level 1 has always
had, moved to level 0; `--l0 palette` is the older mode and is kept because its pack is lossless at 1-3 bits.

**The solve.** Under `bc8` level 0 is not an index on a palette. It is a plane of `C0` continuous channels at the
decoded extent, and the objective is exactly quadratic in it once level 1 and the decoder are held, so it is solved by
3.2's own machinery. Writing `A` for the decoder's a-block, `B` for its b-block and `N_q` for the sc-columns of level-1
channel `q` (column `i` of `N_q` is `W[:, C1 + C0 + i C1 + q]`), at a site whose level-1 sample is `c` and whose
level-0 taps are the texels `t` with weights `w_t`:

```
out      = n(c) + Q(c) . sum_t w_t x_t ,   n(c) = b + A c ,   Q(c) = B + sum_q c_q N_q      (nout x C0)
H[t][t'] += w_t w_t' Q^T diag(cw) Q
grad[t]  += w_t      Q^T diag(cw) r ,      r = out(x_prev) - target
```

This is 3.2's `G` with the two latents exchanged: there the output was affine in the level-1 sample with a coefficient
matrix quadratic in the level-0 one, here it is affine in the level-0 sample with a coefficient matrix quadratic in the
level-1 one. Every consequence carries across. `Q^T diag(cw) Q` is a quadratic form -- in the level-1 sample this time
-- so the assembly again accumulates `1 + C1 + C1 (C1 + 1) / 2` scalars per stencil slot and expands them against fixed
matrices once per texel. The step is the same proximal one, `(H + lambda I) d = -grad` with `lambda = --ridge` times
`mean(diag H)`, solved by the same block-Jacobi preconditioned conjugate gradients.

**The site set of one level-0 texel** is the one 3.3's search already enumerates: its own centre site, where level 0 is
read *nearest* so the single tap is the texel itself at weight 1, plus the `K` fractional sites of each pixel of its 3x3
neighbourhood, where level 0 is read bilinearly at its own resolution and a tap the clamp pulled onto this texel adds
its weight to it. That is the only place the two assemblies genuinely differ -- block (b)'s unknown is read bilinearly
at the centre site too -- and it is one branch. A texel therefore couples only to texels within one in both axes: the
same 9-point stencil, the same five owned blocks, the same four exactly independent colours.

**The grid.** Level 0's per-channel `lo/hi` are fitted at `--q1-start`, under `--q1-range`, over the base and every
chain plane together, and frozen -- for the same reason level 1's are (4.4): the JSON carries one `lo/hi` pair per
channel per latent texture. Both grids are frozen at the same point in the same round, *before* block (b) assembles, so
the one step of the loop that is a constraint rather than a minimisation is one step and shows up in one place in the
round line. It prints a `level 0 grid frozen:` line of its own beside level 1's `grid frozen:` line, naming the round,
the range policy and every channel's `lo/hi` and step, and the report repeats both at the end. From then on block (c') is the quantised four-colour sweeps with the exact per-channel grid argmin at 8
bits, where `rep(k) = k` and the sampled-grid rule is `value = lo + k / 255 * (hi - lo)`.

**The round** is then `(a)` the decoder, `(b)` level 1, `(c')` level 0 -- three minimisations of one objective, as
before, and `E` is monotone across all of them. The palette search of 3.3 is not run at all.

**The post-fit pack, and the refit.** After the last round the plane is 8-bit and storable, but it is not what the
`.dds` will hold: BC4 gives a block eight values along a line, not 256 free ones. So the plane is packed once, exactly
as the writer will pack it, and **decoded back into the device plane**; that pack is then REFINED block by block
(3.6), and level 1 and the decoder are refitted against the result -- one block (b) on level 1's frozen grid and one
block (a). Everything the report describes is that packed plane, which is the same rule the 4-bit palette mode follows
(5.2). The encoder prints `E` and the psnr of the continuous plane, of the seed pack, of the refined pack and of the
refitted model, plus each pack's own psnr against the continuous plane, so the cost of the block format and what is
taken back from it are numbers in the log and not inferences.

**What the mode fixes.** `--bits0` does not apply (level 0 is 8 bits, which is what a BC4 block carries) and `--bc0 0`
does not apply (the pack is the point; an uncompressed 8-bit level 0 costs twice what the palette mode does). Both are
refused by name rather than ignored, because a run whose flag was silently dropped still looks like a result.
`--bc0 both` is allowed and writes the pre-pack plane beside the asset, which is what gives `bc_check` a reference to
measure the pack against.

### 3.6 The pack refined by analysis by synthesis (`--bc-refine`, `--bc-outer`)

The post-fit pack of 3.5 is the one every BC4 encoder makes: the block's lowest and highest value become the endpoints, each
texel takes the nearest of the eight palette entries, and the whole choice is fitted **against the latent's own
values**. Nobody samples the latent. What is sampled is the decoder's output, and measured on eight runs
(`docs/RESULTS.md` section 6.1) that pack costs 0.37 to 4.61 dB of it while the refit of level 1 and the decoder wins
back essentially nothing -- once the error is in the full-resolution plane, level 1 at a quarter of the resolution per
axis has nowhere to put a correction for it.

So the pack is chosen the way a BC1-7 or ASTC encoder chooses one: **by analysis by synthesis**. Each 4x4 block's
endpoints and selectors are re-chosen to minimise the decoder's output error over the sites that block's texels touch,
and a candidate is taken only when that error measurably falls. The starting point is the pack above, so the result is
never worse than it; where nothing improves, the bytes are the same bytes.

**The block objective.** With level 1 and the decoder held, `E` is exactly quadratic in level 0's values -- that is
3.5's premise, and block (c')'s assembly already builds the quadratic:

```
E(x_prev + d) = E(x_prev) + 2 g . d + d^T H d
```

with `g` the gradient at the plane the stencil was assembled at and `H` the 9-point stencil. Restricted to one block
`B`, every texel outside it held at the value it currently has, that is a quadratic in the block's `16 C0` values
alone:

```
E_blk(d_B) = const + 2 A1 . d_B + d_B^T A2 d_B
A2[(t,i)][(t',j)] = H[t][t'][i][j]                            for t, t' in B
A1[(t,i)]         = g[t][i] + sum_{t' not in B} (H[t][t'] d_t')[i]
```

`A2` is dense inside the block, because a fractional site's bilinear taps couple neighbouring texels; it is read
straight out of the five stencil blocks each texel owns plus the transposes of its neighbours', in double, once per
block. Nothing is re-enumerated: the sites the brief describes -- the sixteen centre sites plus the `K` fractional
sites of every pixel whose level-0 taps reach a block texel -- are exactly the sites that assembly summed over.

**The constraint.** Each channel of the block is one BC4 block in the eight-value mode: two 8-bit endpoints `r0 > r1`
and a 3-bit selector per texel, so the stored byte is

```
byte_t = a(sel_t) r0 + (1 - a(sel_t)) r1 ,   a = 1, 0, 6/7, 5/7, 4/7, 3/7, 2/7, 1/7 for sel = 0..7
```

-- the standard palette of `src/bc_pack.h`, written as the affine weight on the high endpoint -- and the value a
sampler returns is the plane's own `lo_c + byte / 255 * (hi_c - lo_c)`. The six-value mode is not searched: it trades
two interpolants for a hard 0 and 255, which a latent on a fitted range has no use for.

**The search**, per block, per round (`--bc-refine N`, default 4):

* **the endpoints, per channel, by least squares.** `byte_t` is affine in `(r0, r1)` at fixed selectors, so with
  `p_t = s a(sel_t)` and `q_t = s (1 - a(sel_t))`, `s = (hi - lo) / 255`, a move `(dr0, dr1)` changes the block's
  values by `d_t = p_t dr0 + q_t dr1` and
  ```
  dE = 2 (b.p) dr0 + 2 (b.q) dr1 + (p A2 p) dr0^2 + 2 (p A2 q) dr0 dr1 + (q A2 q) dr1^2 ,   b = A1 + A2 d
  ```
  -- five scalars, from which the exact minimiser is one 2x2 solve and every candidate costs three multiplies. The
  vertex is rounded onto the 8-bit grid and its +-1 neighbours in both endpoints are tried with it (nine candidates,
  which is what catches the rounding), each required to keep `r0 > r1`;
* **the selectors, one texel-channel at a time, by exact argmin.** With every other value held, `E_blk` in one value is
  a parabola, so the eight palette entries are simply evaluated and the smallest is taken. Two Gauss-Seidel passes over
  the block's `16 C0` selectors, each step seeing the ones before it.

**The acceptance rule** is the whole of it: a step is taken **only if it strictly lowers `E_blk`**, and the deciding
number is always evaluated in the arithmetic the plane is written in -- the least squares above is the *search*, and
the candidate it picks is weighed again against the float-rounded palette values before it is accepted. Since every
accepted step lowers the same quadratic and the rest of the plane is held, `E` over the plane cannot rise.

The acceptance is decided on the **double** quadratic `E_blk` while the `E` the gates read is measured in **float**
over the whole plane, so the two are not the same arithmetic and a step whose true gain is smaller than float rounding
can leave the measured `E` flat or a last-bit above. That is what the `1e-9` relative tolerance in
`tests/run_checks.py` covers; the decision itself is exact in the arithmetic it is made in.

What the search reaches is a **local minimum of a mixed-integer problem by alternating descent**, not a global one:
the endpoints are optimised with the selectors held and the selectors with the endpoints held, never jointly, the
endpoint step searches only a `+-1` neighbourhood of the rounded vertex, and the six-value mode is excluded. A joint
move that raised `E_blk` on one coordinate to lower it more on another is therefore never found.

**The four colours.** Two blocks couple when a fractional site's taps span their border, which reaches one texel past
a block edge, so blocks two apart in either axis share no site. Colouring the *block* grid by `(gx & 1, gy & 1)` and
running the four colours as four kernel launches makes every block of a pass see a frozen neighbourhood -- exactly what
3.2's four-colour sweeps do over texels. One CUDA block per 4x4 block, 64 threads, `A2` in shared memory.

**The outer repack loop (`--bc-outer N`, default 2).** Level 1 and the decoder are now fitted to the *decoded* plane,
so the continuous plane that best serves them is a different plane, and packing that one may land better than packing
the first. Each pass is: refit level 1 and the decoder against the packed level 0; solve level 0 as a continuous plane
again (block (c')) against them; pack it and refine the pack. The pass is taken **only if the shipped `E` -- the
objective measured on the packed plane, which is what a consumer gets -- strictly falls**; otherwise the previous
pass's planes, indices, blocks, decoder, both grids and the report's own figures are restored and the loop stops. So
the loop can only improve the asset, and its cost when it cannot is one pass.

Both steps of a pass are minimisations under `--q1-start 0` as much as under the frozen grid. The refit is level 1's
**quantised sweeps on the grid it already carries**, and step (2) is the continuous block (c') followed by a snap onto
level 0's existing grid. Re-fitting either range inside a pass would move the grid under the plane the pass is being
weighed against, and the accept-if-lower rule would then be comparing two numbers that are not on one scale.

**`--bc-refine-after N`.** The refinement above is weighed against the level 1 and the decoder that were fitted
*before* the pack. `--bc-refine-after` runs `N` more refinement passes after the refit instead, so the blocks are
re-weighed against the model as it now stands; the blocks are kept and only re-judged, so this too can only lower `E`.
It is 0 by default because it is worth +0.00 to +0.06 dB on the eight comparison runs, the most on the one where the pack
costs most (`docs/RESULTS.md` 6.2), so it is a flag for the case that looks like that one and not a default.

**The last refit.** Whatever pack the asset ends with -- the post-fit step's, the last accepted pass's, or the one a
rejected pass's restore put back -- the level 1 and the decoder beside it were fitted before that pack. One more block
(b) and block (a) over a level 0 that does not move are two exact minimisations, so the shipped `E` can only fall; it
is worth about a hundredth of a decibel and it is what the report's `E shipped` is measured after.

**What it is worth** is `docs/RESULTS.md` section 6.2: on the same eight runs the refinement and the outer loop take
back 32 % to 71 % of the pack's `E` cost, +0.11 to +2.97 dB of centre psnr over what the post-fit pack of 3.5 shipped
on its own, at 0.06 to 0.22 s of refinement per image, inside a whole encode of 1.4 to 3.5 s. The remaining gap is the block format's own: eight values on a line per 4x4 block, endpoints on an 8-bit grid.

## 4. The mip chains

`--mips 1` (the default) stores a full chain for both latents; `--mips 0` writes one level per texture, and everything
below then applies to a chain of one.

### 4.1 The 1:1 rule

**Output mip `m` is decoded from mip `m` of BOTH latents.** Level 0's plane `m` is `mip_dim` applied `m` times to the
padded extent and level 1's is the same halving applied to a quarter of it, so the two chains are halved together and
the pair that decodes a level is always the pair that was fitted for it. The chain stops while level 0 stays at or
above `--mip-min` per axis.

Floor halving (`max(1, n/2)`) means level 1's plane `m` is not exactly a quarter of level 0's once an odd size appears
in either chain. The drift is sub-texel and is exactly what the hardware's own chains do, so it is left alone.

**On the GPU the rule is one sampler-state field.** The hardware picks each texture's mip from that texture's own texel
density, and level 1 has `1/block` of the texels per axis, so its LOD sits `log2(block)` below level 0's: when level 0
is at mip 1, level 1 is still at mip 0 - a plane fitted beside level 0's mip 0, not beside its mip 1. Level 1's sampler
therefore carries `MipLODBias = log2(block) = 2`, the `lod_bias_level1` the JSON publishes. Its LOD then equals level
0's at every distance, and at 1:1 on screen it goes from -2 to 0, still mip 0, so nothing changes at the base. Without
the bias the decoder is fed the wrong colour plane from mip 1 down and the image goes splotchy at level-1 texel scale
while the base stays right. Any consumer of the asset must do the same; the viewer does it by default, and key `L` (or
`--nobias` with `--shot`) turns it off for comparison.

### 4.2 Every plane is a parameter

No latent is ever filtered into existence. The **source** chain is built once at start-up and plane `m` of both latents
is then **solved** against `src_m` by the same three blocks the base gets, every round.

**The chain is per texture, and it is the only place the encoder chooses its own ground truth.** The default is the
iterated 2x2 box (`src_m = box2x2(src_{m-1})` over all `3T` channels, a one-wide or one-high plane repeating its only
row or column), and when no texture asks for anything else that is the whole of it, byte for byte. A texture with a
`--mip-filter` other than `default` is derived differently: every level `m >= 1` is resized **directly from the padded
base** to that level's own size through the vendored `stb_image_resize2`, with the texture's own filter (`box`,
`mitchell`, `catmullrom`) and edge mode (`clamp` or `wrap`, the FILTER's edge mode -- the fit's own sampler stays
clamped, as it always was). An iterated filter is a different filter at every depth, which is why the deep levels come
from the base and not from the level above.

**`srgb`, `edge` and `normal_map` imply a filter when nothing else named one.** All three describe how a chain is
DERIVED, and the built-in iterated box derives nothing they can apply to. What asks for a derivation is the VALUE and
not the key: `"srgb": true`, `"edge": "wrap"`, `"normal_map": true`. A key carrying its inert value (`"srgb": false`,
`"edge": "clamp"`, `"normal_map": false`) asks for nothing at all -- it implies no filter and contradicts none,
whatever the filter is -- and the settings row still prints the value with its `(json)` source. When a texture does
ask and its filter is still the program's own default -- nobody typed a filter, in the JSON or on the command line --
the merge switches that texture to `box`, stb's trapezoid, so that the key has a derivation to act on. That switch is
not free: the implied `box` RE-DERIVES THE WHOLE CHAIN FROM THE BASE, which is the picture the built-in iterated box
gives only where every level is an exact halving. Wherever an odd plane is halved the two differ in the deeper planes
-- see the footprint paragraph below -- so an implied box is a different ground truth, not a no-op. The settings row
prints `filter box (implied by srgb)`, naming the first of `srgb`, `edge`, `normal_map` the texture set in that fixed
precedence, so a log never shows a filter nobody asked for without saying where it came from, and the asset's
`source.inputs` records the effective `box`. When the filter WAS named as `default` -- a JSON `"filter": "default"`, or
`--mip-filter default` -- the two statements contradict each other and the run is refused instead, because guessing
which half of a contradiction was meant is not the encoder's business; the message says which of the two asked for it.

Three things happen around that resize, in this order:

* **the clamp to [0,1]**, always and immediately. Mitchell and Catmull-Rom have negative lobes and `stb_image_resize2`'s
  float path does not clamp, while the objective reads the chain raw and the PNGs and the PSNR are written from it: a
  level is a target and has to stay inside the byte range the source lives in. The source is in [0,1], so the clamp
  takes nothing away that was ever in the picture. (`box` is stb's trapezoid, a box exactly at integer ratios and with
  one-texel ramps otherwise.)
* **`srgb`**: the base is converted to linear light before the resize and back to sRGB after it, in float, never
  through 8 bits. `M0` is never converted -- it is the source as loaded, and the objective fits the source's own byte
  encoding at every level. Only the derivation of the deeper targets is in linear light;
* **`normal_map`**: each filtered level is renormalised by the viewer's own rule (`viewer/bin/nntc_view.hlsl`, key `V`)
  -- `n = rgb * 2 - 1`, and the texel is rewritten only where `|n| > 0.5` and `n.z > 0`, so a grey texel with no
  direction to normalise towards and a texel pointing into the surface are left exactly as the filter left them.
  Averaging two unit vectors that disagree does not give a unit vector, and re-lengthening it is what a consumer's
  shader does. Object-space normals are out of scope: they may point anywhere, so this rule would refuse to touch half
  of one.

**The footprint differs between the two paths.** The iterated box drops the last column or row of an odd plane, so it
covers a sub-rectangle of the base; a direct resize covers all of it. It is reachable whenever an odd plane is halved
-- 68 -> 34 -> 17 -> 8 at the default `--mip-min 8` -- and it is a real difference in what the deep planes are fitted
against, not a rounding one.

A filter, `srgb`, `edge` and `normal_map` all describe how the levels BELOW the base are derived. When there is no
level below the base - under `--mips 0`, or on a source whose `--mip-min` leaves the base as the only stored level -
they are simply idle: the filter is a setting, the chain is the encoder's decision from the image's size, and a texture
that gets no mipmaps gets no mipmaps. The LATENT TEXTURES are then the ones the run without the setting writes, which
the gate asserts byte for byte; the descriptor is not, because it records the setting that was asked for whether or
not there was a level for it to act on. A padded input's chain is derived from the PADDED image -- `wrap` in
particular treats the replicated edge as part of the picture -- so a padded input is a testing convenience and not the
shape a material should ship in; the padding warning says so.

**The planes are independent, given `W`.** Hold the decoder and the `M + 1` problems have no variable in common: plane
`m`'s level-1 solve reads plane `m`'s level 0 and plane `m`'s source and writes plane `m`'s values, and the level-0
search is the same. So blocks (b) and (c) are issued for **every plane at once**, each on its own stream, and waited for
exactly once: a plane owns its own block (b) workspace, the only ordering the streams need is at the two ends of a
block, and in between the base keeps the device busy while the deep planes - a few thousand texels each - cost little
more than their launches. Measured on one image, issuing them one after another instead cost the chain about four times
the base's time for a third of its texels; almost all of that was launch latency and the host waiting for a plane's
scalars, not work.

That is also why nothing inside a block travels to the host. The proximal ridge, the conjugate gradients' inner
products, the movement probe and the sweeps' moved count are all reduced on the device and read by the next kernel where
they lie; the host copies one small scalar row per plane after the block's single synchronisation. The one host decision
left in the iteration is when to stop, and the residual is looked at once every four iterations rather than three times
per iteration - a stop is never missed, only postponed by at most three iterations.

The one thing the levels share is the decoder, and that is the whole reason the chain has to be solved rather than
initialised: `W` is fitted over every site of every plane, so a chain left at its initialisation drags `W` towards
fitting an initialisation, and the levels it then decodes are that initialisation seen through a decoder pulled away
from the base to accommodate it.

**The chain the run generates is the golden reference.** The objective, every level's PSNR, the `--diag` table and the
`--png` source images are all against it, and nothing anywhere regenerates a second one to compare with. So two runs
with different filters are two runs against two different ground truths, and their mip PSNRs are not on one scale; the
report says `the run's own source chain` on every line that quotes one, for that reason.

Level 0's chain starts from the residual seed above - each plane's own residual against its own source mip,
projected onto the direction the base chose - or, under `--init0 luma`, from the source mip's own luminance, snapped; level 1's from the box or pca init of that mip's
block means, with the **principal directions taken once on the base plane and used for every plane**, because channel
`j` has to mean the same direction of the material at every level for one `W` to fit them all.

### 4.3 The mip weights - block (a) only

Let `N_m` be plane `m`'s pixel count and `r_m` its raw weight: `1` for `uniform`, `sqrt(N_m)` for `sqrt`,
`N_m` for `pixels`. The base is held out of the split - it takes `s_0 = 1 - mix` - and the chain shares `--mip-mix`
between its levels in proportion to `r_m`; dividing a plane's share by its site count `N_m (1 + K)` gives the weight
`w_m` one of its sites carries, which is the `omega` that multiplies `v v^T` in the decoder's normal equations.

`pixels` is the default: it gives every plane of the chain the same weight per sample, so the chain reads as one big
image, and it measured better at every level than `sqrt` on both ablation images (`docs/RESULTS.md`). `uniform` gives a
deep 8x8 plane `N_0 / 64` times the base's per-sample weight, which is why it is not the default.

The weights enter **block (a) and nothing else**. Blocks (b) and (c) are per plane and independent, so a mip weight
could not change what they do: scaling one plane's objective by a constant does not move that plane's argmin. What the
weights govern is how much of the one shared `W`'s capacity each level gets - the only place the levels compete at all.

### 4.3.1 What the weights cost a deep plane, and how to see it (`--diag`)

The weights are a policy and they have a consequence worth stating plainly, because it looks like a bug when it is
met for the first time. Under the default `pixels` / `--mip-mix 0.5`, a nine-plane chain over a 2048x2048 base splits
block (a)'s mass as

    M0 0.500   M1 0.375   M2 0.0938   M3 0.0234   M4 0.00586   M5 0.00146   M6 0.000366   M7 0.0000916   M8 0.0000229

so the shared `W` is chosen by the base and `M1` and essentially nothing else. Every plane still gets an exact block
(b) and block (c) of its own, so a deep plane's LATENTS are optimal; what a deep plane cannot do is change the decoder
it is decoded by. If the base does not need a full-resolution path to some output, `W` will not build one, and every
deep plane is then held to what the quarter-resolution level 1 alone can carry for that output - at every level.

That is exactly what happens on a four-texture material whose second texture is an easy, smooth one. At `--c0 2
--c1 4` on `mg1-4` the base reaches 49.26 / 50.75 / 42.48 / 40.54 dB and texture 1's mip chain runs
47.16 41.71 34.38 28.82 26.74 27.77 31.01 36.18 - and a box-filter-and-bilinear-upsample of texture 1's own source
mips, which is the ceiling of a quarter-resolution latent with no help at all, runs 39.95 35.68 30.79 27.12 25.92
27.18 30.58 35.85. The run sits AT that ceiling from M4 down: level 0 is contributing nothing to that texture there.
The decoder says why - the `--diag` table's level-0 figure for texture 1 is 1.90 at `--c1 4` against 8.90 at
`--c1 3`, where level 1 has no channel to spare for that texture and `W` must route it through level 0 instead.

Nothing in the loop can notice: the same run's E is 3.73e-05 against the `--c1 3` run's 6.28e-05, so by the objective
the collapsed chain is the better asset, and it is - by that objective. Moving the weights moves the trade rather than
removing it (measured on the same material in `docs/RESULTS.md`): `uniform` lifts texture 1's M5 from 26.74 to 36.34 dB
and costs the base about 2 dB, `sqrt` lands between the two. Which is wanted is a decision about the asset and not
about the solver, which is why `--mip-weight` is a flag and why this section is a statement of the consequence rather
than a fix.

**`--diag` is the instrument for it.** It prints, of the state the asset is in, every texture's psnr at every stored
level (the report's `psnr texture t` line is its base row and `mip psnr` is the worst of each later row), every latent
channel's mean, standard deviation, range and share of values on an end of the shared grid, plane by plane, and how
hard each texture leans on each latent channel - the rms of that texture's decoder rows over the columns the channel
reaches. A texture with no level-0 figure of any size has no full-resolution path and its chain is the ceiling above;
a channel whose standard deviation is a grid step or two carries nothing at that level whatever is asked of it.

### 4.4 The shared level-1 range

The asset format carries **one `lo/hi` pair per channel per latent texture**, for the whole chain, because the GPU
dequantises the sampled value with one scale and one bias whatever mip its sampler chose. The range is therefore fitted
over the base **and every chain plane together**: at the freeze round every value of every plane enters the percentile
(or the min/max) with the same weight, the grid is frozen, and every plane is snapped onto it. The quantised sweeps
then run per plane on that one grid.

Weighting every value equally gives the base the most say, since it has the most values - which is right, because it
also carries the most error - while a chain plane that genuinely reaches past the base still moves the range. A range
fitted on the base alone would clamp such a plane onto the ends of the grid for the rest of the run.

## 5. The quantisation

**The grid is a function of the shipped format.** The encoder has to fit the values a consumer's sampler RETURNS, not
the values the index nominally stands for, and those are not always the same number. A UNORM8 texel reaches the shader
as `byte / 255`, and the byte of an uncompressed plane is the index with its bits replicated -- `rep(k)`, which is
`k / levels` exactly at 1, 2, 4 and 8 bits and not at 3, 5, 6 and 7, where the two stand up to half a byte apart. A
block-compressed level 0 is the other case: BC4's palette at 1-3 bits is `k / (2^bits - 1)` of full scale exactly
(section 6), which IS the index rule. So there are FOUR rules and each belongs to a format:

    level 0, --l0 bc8 (the default)  g_c(k) = lo_c + k / 255 * (hi_c - lo_c) ,      levels = 255
    level 0, palette, uncompressed   pal_c(k) = -1 + 2 rep(k) / 255
    level 0, palette, BC             pal_c(k) = -1 + 2 k / (2^b - 1)
    level 1, always uncompressed     g_c(k) = lo_c + rep(k) / 255 * (hi_c - lo_c) ,  levels = 2^bits1 - 1

**Under the default `--l0 bc8` level 0 is on the first of those**, a fitted per-channel range at 8 bits exactly as
level 1 is (3.5), and **block (c) is not run at all**: there is no palette to search, the plane is solved continuously
by block (c') and the bytes the file holds are the BC blocks the pack and its refinement chose (3.6), whose value a
sampler returns is `lo + byte / 255 * (hi - lo)`. At 8 bits `rep(k) = k`, so the index rule and the byte rule are one
number and the distinction below does not arise.

Under `--l0 palette` level 0 lives on the second or the third, per channel, and block (c) searches that palette
exactly, so level 0 is never anything but storable. Level 1 lives on the fourth, one `lo/hi` pair per channel for the
whole chain (the asset format carries exactly one), and block (b)'s quantised step takes the nearest grid VALUE rather
than the nearest index -- one comparison more than a round, because the grid is monotone but no longer uniform.

Fitting `k / levels` while shipping `rep(k) / 255` is a systematic per-entry bias in every decode, worth 0.0034 of a
level-0 channel's [-1,1] range at 3 bits and 0.0015 of level 1's span at 5-7 bits; `tools/dds_decode.py --grid` asserts
index by index that the value the asset publishes is the value a sampler returns.

The dequantisation is affine and is applied **after** sampling, which is what lets a quantised plane be handed to a
hardware sampler at all: the blend of the indices is the blend of the values.

**Which filter the consumer uses is not the encoder's business.** Trilinear filtering blends two mips of both latents,
and anisotropic filtering is several trilinear samples along the footprint's long axis; in both cases what reaches the
decoder is a weighted average of latent samples, the dequantisation is affine, and the decoder is affine in the
samples, so a blend of samples decodes to the blend of the decodes. Nothing about the fit has to change for either -
the encoder fits the site set of section 2 and the hardware may blend those decodes however it likes. `nntc_view` turns
anisotropy on by default in its trilinear mode for that reason: it is free, and it is what a game would do.

### 5.1 The range: `--q1-range`

`minmax` takes the channel's own smallest and largest value. `pct` (the default) takes its 0.1 % and 99.9 %
percentiles, by a sort of the channel's values, and lets everything outside clamp onto the end of the grid. The
difference is what a handful of outliers cost everybody else. Measured on `model10` at `--c0 2 --bits0 3 --c1 4`, the
grid frozen at round 3:

| channel | `minmax`, 8 bits | `pct`, 8 bits | `minmax`, 4 bits | `pct`, 4 bits |
|---|---|---|---|---|
| ch0 | [-1.0078, 1.3420] | [-0.7920, 1.1524] | [-1.8880, 3.7530] | [-1.1037, 2.9808] |
| ch1 | [-1.2334, 1.1797] | [-0.9841, 1.0671] | [-2.9945, 4.1700] | [-2.0598, 3.2269] |
| ch2 | [-1.1675, 1.2096] | [-0.9952, 1.0681] | [-2.4345, 3.5872] | [-1.6847, 2.5668] |
| ch3 | [-6.3631, 33.1999] | [-4.8322, 21.4276] | [-13.9034, 13.8515] | [-11.5114, 9.6422] |
| psnr | 41.94 dB | **41.98 dB** | 34.74 dB | **36.02 dB** |

The fourth channel is the interesting one. With one texture (`nout = 3`) and `C1 = 4` the level-1 solve has a genuine
null direction at any texel whose footprint is flat in `s` (section 3.2), and the ridge bounds that direction without
removing it: ch3's range comes out an order of magnitude wider than ch0-ch2's, so its grid step is an order of magnitude
coarser. `pct` takes a third of that width back (39.6 wide down to 26.3 at 8 bits), and because the sweeps then
re-optimise every texel **on** the frozen grid, the width it recovers turns into quality rather than into a smaller
residual on paper.

### 5.2 Rounding at the end, and the quantised sweeps

Two ways to land on the grid:

1. `--q1-start 0`: level 1 stays continuous for the whole loop and is snapped once at the end. The snapped plane is the
   minimiser of the **unconstrained** problem pushed to the nearest storable point, which is not the minimiser of
   anything the asset can hold.
2. `--q1-start R0` (the default, `R0 = 3`): at round `R0` the range is fitted, the grid is **frozen**, every plane is
   snapped onto it, and from then on block (b) is `--q1-sweeps` four-colour Gauss-Seidel sweeps whose per-value step is
   the exact argmin over that grid. The plane is storable from the freeze onwards and nothing is rounded at the end.

**The step.** With level 0 and the decoder held, `E` is exactly quadratic in the plane, so with the correction `d`
measured from the plane the stencil was assembled at and the same proximal ridge,

    f(d) = E(c_prev + d) - E(c_prev) + lambda |d|^2 = 2 g.d + d^T H d + lambda |d|^2

and in one value `x = d[t][q]`, with every other value held,

    f = a x^2 + 2 b x + const ,   a = H[t][t][q][q] + lambda ,   b = g[t][q] + (H d)[t][q] - H[t][t][q][q] x

- `b` is the gradient component at the current `d` with `x`'s own term taken back out. Since `a > 0` the parabola falls
towards its vertex `x* = -b / a` from both sides, so its minimiser over the grid is the grid point whose VALUE is
nearest the vertex. The round is the guess and the comparison is the answer (`level1_index` in `model.h`):

    k = clamp(round((c_prev + x* - lo) / step), 0, levels) ,   step = (hi - lo) / levels
    index = whichever of k-1, k, k+1 has the nearest grid VALUE, ties to the lower

The two agree at 4 and 8 bits, where `rep(k) / 255` IS `k / levels` and the grid is uniform; at 5, 6 and 7 bits it is
not, and the round alone would land a step away near the ends. Three comparisons more than a divide - exact, and still
not a search over the alphabet. The four colours `(tx & 1, ty & 1)` make
a pass an exact joint step: the stencil couples only texels within one texel in both axes and two texels of one colour
are two apart, so nothing in a pass sees anything else move. The `C1` channels of one texel do couple, through the
off-diagonal of its own block, so a thread walks them in turn and each uses the values the earlier channels just took.

`f` falls at every step and `f(0) = 0`, so `f(d) <= 0` and `E(c_prev + d) <= E(c_prev) - lambda |d|^2`: **the sweeps
cannot raise `E`**, which is why the ridge can stay in the quantised phase. The one step of the whole loop that may
raise `E` is the freeze itself, and it must: applying a constraint is not taking a step.

The measurement, `--c0 2 --bits0 3 --c1 4 --rgb-weights 9,11,1`, `--q1-range pct`, `--init box`:

| image | bits1 | option 1 (`--q1-start 0`) | option 2 (`--q1-start 3`) | gain |
|---|---|---|---|---|
| model10 | 4 | 28.91 dB, 15 rounds, 0.753 s | **36.02 dB**, 20 rounds, 0.756 s | **+7.11 dB** |
| model10 | 8 | 41.94 dB, 14 rounds, 0.690 s | **41.98 dB**, 14 rounds, 0.572 s | +0.04 dB |
| game2 | 4 | 30.70 dB, 17 rounds, 0.855 s | **36.78 dB**, 20 rounds, 0.850 s | **+6.08 dB** |
| game2 | 8 | 44.25 dB, 14 rounds, 0.798 s | **44.99 dB**, 13 rounds, 0.621 s | +0.74 dB |

At 4 bits a half step is a thirtieth of the channel's range and rounding at the end throws six to seven decibels away.
At 8 bits it is worth a fraction of a decibel - and it is *free*, because a sweep is far cheaper than the conjugate
gradients it replaces: on `model10` at 8 bits block (b) cost 306 ms over the run as pure CG against 198 ms as three
continuous rounds plus eleven quantised ones (44 ms continuous, 154 ms quantised, nearly all of it the assembly the two
paths share - the four sweeps themselves are about 0.3 ms a round against 7 ms of CG).

### 5.3 The zero-change assertion

One function in `model.h` defines `value <-> index` for the whole encoder, and the device spells its multiply and add
out as separately rounded operations so that nvcc cannot contract them into an fma the host has no counterpart for.
Every path onto the grid - the snap, the sweeps, the writer - goes through it, and two checks hold the promise:

- before the asset is written, every value the device holds must be exactly the grid value of the index stored beside
  it, or the encoder refuses to write the file;
- in the writer, every stored index dequantised and quantised again by the `lo/hi` that same JSON publishes must come
  back as itself.

Together they say that the numbers the encoder measured `E` on and the numbers a reader recovers from the file are the
same numbers, not merely close ones. The first of the two is not theoretical: a device fused multiply-add against a
host multiply and add disagrees in the last bit on most of a plane, which is why one function defines the conversion
and why it spells its arithmetic out.

## 6. Determinism

Two runs of the encoder on one image, with one command line, produce byte-identical `.dds` and `_nntc.json` files. That is a
property of the code, not an accident of it, and it is a release gate.

There is no seed in anything that reaches the asset -- the subtexel offsets are the eight fixed numbers of section 1
and every initialisation is computed from the source. (The one pseudo-random generator in the tree is the linear
congruential probe `--check` uses to pick finite-difference coordinates. It is fixed-seeded, it restores every value it
perturbs, and it runs only when `--check` is named: nothing it touches is in the file.) So the only way a run could
differ from itself is the order in which floating-point additions land. Addition is not associative, so a reduction whose blocks add into one accumulator with atomics returns
whatever the scheduler made of it that time, and a decoder that differs in the last bit moves a level-0 argmin
somewhere, which moves a texel, which changes the file.

So every reduction whose result is used is **order-independent by construction**: a fixed grid of blocks, each block
walking its own stride of the work in increasing order and writing its partial to its own slot, and a second pass that
adds the partials by index. That covers the decoder's normal equations, the objective, the conjugate gradients' inner
products, the stencil's diagonal, the level-1 range and movement probes, and the pca covariance. The remaining
atomics count texels that moved, and a count does not depend on the order it was counted in.

One atomic adds doubles: the BC refinement (3.6) sums the decrease its blocks accepted, for the log. Its last digits do
depend on the order the blocks finished in, and it is a **report number only** -- no decision reads it. Each block's
own decisions are taken from its own quadratic in its own shared memory, and the four-colour order fixes which
neighbours it sees, so the blocks the file holds are the same blocks every time. The determinism gate runs the default
command line, which is `--l0 bc8` with the refinement on, and asserts exactly that.

The block solves themselves are per texel and write only their own texel, so they carry no order at all; the stencil
assembly is a gather for the same reason.

**What determinism does NOT promise.** It is one machine, one build, one command line. Nothing here is promised across
platforms or compilers, and the source chain's filters are the newest reason to say so out loud: a run with a
`--mip-filter` other than `default` puts the vendored resizer between the source and the objective, and that resizer
chooses a code path from what the CPU supports. The gate asserts byte-identical reruns on one machine -- including one
pair under `mitchell` + `srgb` and one under `mitchell` + `normal_map`, which are the two paths the plain determinism
check does not reach.

Measured across the two toolchains (VS 2026 + CUDA 13.4 against WSL 2 / gcc 13.3 + CUDA 13.3, one machine, the four
`m1`-`m4` textures at the default layout). With no filter the two builds' `.dds` files are still **byte-identical**
(`out/disclosure/log_m1234_default.txt` against `out/disclosure/log_m1234_default_wsl.txt`). With `mitchell` on all
four textures, `srgb` on the albedo and `normal_map` on the normal map they are **not**: all three files differ, and
the numbers do not move where it matters. The two runs of that one material, from
`out/disclosure/log_m_material_win.txt` and `out/disclosure/log_m_material_wsl.txt`:

| | Windows | WSL |
|---|---|---|
| `psnr texture 0..3` | 25.49 / 33.50 / 30.41 / 36.25 dB | 25.49 / 33.50 / 30.41 / 36.25 dB |
| `mip psnr M1..M6` | 26.47 / 27.89 / 28.28 / 28.71 / 30.92 / 33.09 dB | 26.47 / 27.89 / 28.28 / 28.70 / 30.92 / 33.12 dB |
| `E shipped` | 7.069498521e-04 | 7.069363374e-04 |

So a filtered chain is promised to the decibel across platforms -- the base agrees to the second decimal and the chain
to within 0.03 dB at its deepest level, where a plane is a handful of texels and one of them is worth a great deal --
and to the byte only within one.

## 7. What is deliberately absent

No entropy coder, no evolutionary search, no hidden layers, no per-block decoder bank, no container format and no
runtime library. The asset's size is fixed by the layout -- so many texels at so many bits -- which means a noisier
plane at equal quality costs nothing here, and rate is not a column in any table.

## 8. Future work

Named so that the shape of the thing is on record, not as a plan with dates. Nothing below is built.

* **The BC encode inside EVERY round.** What 3.6 does is the per-block endpoint and selector search this bullet used
  to propose, and it is built: it runs after the last round and again in each outer repack pass, and it takes back
  32 % to 71 % of the pack's cost in `E` (`docs/RESULTS.md` section 6.2). What is still not built is running it
  *inside* the round loop, so that the continuous solve of block (c'), level 1 and the decoder are all fitted to a
  plane that is already on the block format's manifold rather than to one that is packed afterwards and repaired. `E`
  would stay monotone for the same reason it does now - a block step accepts only a strict decrease of that block's own
  quadratic - and the cost is that every round pays for a pack.
* **RGBA textures.** The format reserves the possibility and the code does not implement it: a four-channel input is
  reduced to RGB, with a warning, rather than encoded as four channels. `nout` would become `4T` and nothing else in the method changes.
* **More than six textures per material.** `nout <= 18` is the shader's cap, not the method's. What has not been
  measured is where one shared `W` stops being able to serve them all: the layout does not grow with the texture
  count, so six textures share the same latent planes (at most four channels each) as four textures do, and `--diag`'s decoder table
  is what says which one is starving.
* **An opt-in NVIDIA BC4 palette (`--bc-palette nvidia`).** Some hardware evaluates the interior of the eight-value BC4
  palette with six-bit weights (`n / 64`) rather than `k / 7`. Since the decoder is affine in the sampled value, the
  encoder could fit against the palette that hardware actually returns and publish those values in the JSON, which
  would remove a sub-byte shift on it. It would be wrong everywhere else, so it can only ever be an explicit mode; the
  default stays the standard palette that every decoder is required to produce (`docs/FORMAT.md` section 7).
