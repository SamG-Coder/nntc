# Prior art disclosure: NNTC (Not Neural Texture Compression), its encoder, asset format and viewer

Richard Geldreich, 2026-09-12. NNTC, Not Neural Texture Compression, is a texture and material compressor whose
encoder is `nntc_encode` and whose reference decoder is the Direct3D 11 viewer `nntc_view`. This file is a public,
dated description of every algorithm, technique and method in this tree, written so that a graphics programmer could
reproduce the system from it. It covers the encoder (`src/`), the decoder as it runs on a GPU (`viewer/`), the asset
format (`docs/FORMAT.md`), the choice of BC4 / BC5, and the techniques that follow naturally from the design and have
not yet been built (section 10). The development history of this code, from `v0.1-skeleton` (2026-09-11) to
`v0.10g.2-weight-clamp` (2026-09-12), is a sequence of local annotated tags in the predecessor tree this repository
was copied from; here it begins at `v1.0.0-nntc`, the imported tree, and `v1.0.1-rename`, the rename to NNTC. The
earlier evolution-strategies encoder this one replaces lives in a tree of its own and had its own disclosures.

Everything stated as measured is in `docs/RESULTS.md` with the log it came from. Everything in section 10 is
explicitly **not implemented** unless it says otherwise.

---

## 1. The idea in one paragraph

A texture, or a material of up to six textures, is stored as **two quantised latent planes and one small affine
decoder** (two `.dds` textures, three when level 0 has three or four channels), and is reconstructed in a pixel
shader from **two ordinary hardware `Sample()` calls** -- three when level 0 has three or four channels -- and a
handful of multiply-adds. Level 0 is a full-resolution plane of one to four channels shipped as ordinary BC4 / BC5 blocks; level
1 is a plane at a quarter of the resolution per axis, one to four channels of 4-8 bits, uncompressed. The decoder is
`out = W phi + b` over `phi = [c ; s ; s (x) c]`: the level-1 sample, the level-0 sample and their pairwise products.
The representation is fitted to be **valid under the GPU's sampling operator**: not texel by texel but at fractional
positions where both latents are blended by the hardware's own bilinear rule, so the same asset is correct under
bilinear, trilinear and anisotropic filtering with no runtime code beyond the samples. The encoder is not a neural
network trainer: it is an **alternating sequence of exact block solves** (linear least squares, a sparse quadratic on a
9-point stencil, and per-texel or per-block exact searches over the shipped format's own values), running on CUDA in about one second to about a minute depending on the pixel count, the texture count and the layout, 113-167x faster than the evolutionary encoder it replaces, within about 2 dB of it either way and ahead where the layout has room. The BC4 / BC5 blocks the file holds are **chosen by analysis by synthesis against the decoder's output**,
the way a BC7 or ASTC encoder chooses blocks, not by packing a finished plane.

The one-sentence framing: *a learned GPU block compression format in which compact per-block and per-texel discrete
data are reconstructed by a small affine decoder acting as the inverse transform, with the hardware texture filter
inside the fitting objective.*

---

## 2. The representation

### 2.1 The two latents

| | extent | channels | bits per channel | storage |
|---|---|---|---|---|
| level 0, "the selector plane" | the decoded extent (full resolution) | `C0` = 1-4 | 8 (default, `--l0 bc8`) or 1-4 (`--l0 palette`) | BC4 (1 ch), BC5 (2), BC5 + BC4 (3), two BC5 (4); optionally uncompressed `R8`/`R8G8`/`R8G8B8A8` under the palette mode |
| level 1, "the colour plane" | a quarter of the extent per axis (`block` = 4) | `C1` = 1-4 | 4-8 | always uncompressed `R8`, `R8G8` or `R8G8B8A8` |

Both carry a full mip chain by default (`--mips 1`). The default layout is `C0 = 2`, `C1 = 4`, `bits1 = 8`:
8 + 2 = 10 bits per pixel at the base, 13.3 with the chain, **for the whole material** however many textures it holds.

Inputs are 1-6 RGB images of one size (an alpha channel is ignored with a warning), each dimension divisible by 4 or padded by edge
replication with a warning and both sizes recorded in the JSON. The layout does not grow with the texture count:
six textures share the same two latent planes (two or three `.dds` textures) as four do, so `nout = 3T` is at most 18 and `--diag` is what says
which texture is starving. They are named on the command line or in a small material JSON, which also carries
per-texture settings the command line has no spelling for (5.4).

### 2.2 Dequantisation after the blend

Each channel of each latent carries one affine range `lo/hi` for the whole mip chain, and the value a shader uses is
`lo + sample * (hi - lo)` where `sample` is the UNORM float the sampler returns **after** its blend. Since the map is
affine it commutes with the weighted average the hardware computes, so **the hardware's blend of quantised texels is
the blend of the dequantised values**. This is the invariant that lets a quantised latent be handed to a bilinear,
trilinear or anisotropic sampler at all, and it is why the encoder's model of the GPU is exact rather than
approximate. The decoder's product terms are formed after both samples are dequantised, so they do not break it.

### 2.3 The decoder

```
phi = [ c_0 .. c_(C1-1) ,  s_0 .. s_(C0-1) ,  s_i * c_j (i outer, j inner) ]      nin = C1 + C0 + C0*C1
out = W phi + b                                                                     nout = 3T,  T = textures
```

No hidden layer, no activation: the output is the affine map itself, clamped to [0,1] only on 8-bit write-out. The
cross terms are the model: `out = (b + B s) + (A + sum_i s_i M_i) c`, so the full-resolution selector plane chooses,
per texel, which linear combination of the low-resolution colour channels is applied. At the default layout the
decoder is 45 numbers for one texture. A member of the same family with fixed coefficients is BC1's or PVRTC's
interpolation; here the coefficients are fitted per material and the low-resolution plane is a learned coordinate
field the selector re-reads.

The order of `phi` is a contract published in the JSON (`terms: "a b sc"`) and asserted by every reader.

### 2.4 What "valid under the sampling operator" means

At the centre of a base-level pixel, level 0's bilinear sample collapses onto that one texel (level 0 is at the
decoded extent) and level 1 blends four texels. Anywhere else both are blended. The encoder fits **both**: every pixel
centre, plus `K = 4` fixed fractional positions per pixel where both latents are read by the hardware rule with
clamp-to-edge and the target is the source read by the same rule. A texel codec is right on the grid and free to be
wrong between grid points; a texture is sampled between grid points almost everywhere.

---

## 3. The decoder at runtime (the viewer, `viewer/`)

`nntc_view` is a Direct3D 11 program that binds the asset's `.dds` files as ordinary shader resources, samples each
once per pixel in the pixel shader at the same texture coordinate with the same filter, dequantises, forms `phi` and
applies the one layer. Nothing is decompressed, repacked or repaired at load. Specifics that a consumer must copy:

* **Level 1's sampler carries `MipLODBias = log2(block) = 2`** (the JSON's `lod_bias_level1`). Output mip `m` is
  decoded from mip `m` of both latents (the 1:1 rule, section 5.4), but the hardware picks each texture's mip from its
  own texel density and level 1 has a quarter of the texels per axis, so its LOD sits two mips finer than level 0's.
  The bias lines them up with no shader arithmetic. Without it the base is right and everything from mip 1 down is
  decoded from the wrong colour plane and looks splotchy at level-1 texel scale. Found by eye; key `L` toggles it.
* **Trilinear and anisotropic filtering need nothing extra.** Both are blends of more latent samples, the
  dequantisation is affine and the decoder is affine in each sample, so a blend of samples decodes to the blend of the
  decodes. The viewer uses 8x anisotropy by default in trilinear mode.
* A two-file level 0 (3-4 channels) binds its second file to a second slot; the shader reads `.rg` from both and uses
  only `.r` of the second at three channels, which is what a BC4 in that slot returns.
* Mips off is the sampler's `MaxLOD` (key `M`), nothing reloaded.
* Key `V` renormalises the shown output as a tangent-space normal (unpack to [-1,1], unit length, repack), leaving
  grey-ish pixels (no direction) and black-ish or z < 0 pixels (into the surface) alone. A display aid, not part of the
  format.
* The overlay prints the GPU memory the bound textures cost in bits per pixel of the decode extent divided by the material's texture count (2.50 bpp at the base for the four-texture example of 8.1): BC levels counted as whole 4x4 blocks, base and whole chain (x4/3) separately.
* `--shot` renders one frame headless with no keyboard input at all, so its bytes are a function of the command line and
  the asset; the release gate runs it five times and asserts they agree.

The viewer also contains a **load-time BC4 / BC5 packer** (`src/bc_pack.h`, key `4`) for assets whose level 0 arrives
uncompressed: 1-3 bits per channel pack losslessly (endpoints 0 and 255 through the mode that provides them, the index
becomes the selector), 4 bits pack lossily by the stb_dxt-style min/max endpoint rule. It is a validation of the pack
and the seed of the encoder's own refinement, not what a default asset contains. `bc_check` validates `bc_pack.h`
against iOrange's `bcdec` (precise float decoder, integer decoder and the tree's own decoder), and asserts per channel
that at 1-3 bits the stored index survived the block format.

---

## 4. The asset format (`docs/FORMAT.md` is the byte-level specification)

Three files, four when level 0 takes two:

| file | holds |
|---|---|
| `PREFIX_lat0.dds` (or `_lat0a` + `_lat0b`) | level 0 with its whole chain, DX10 `.dds`, `BC4_UNORM` (80) / `BC5_UNORM` (83), or `R8`/`R8G8`/`R8G8B8A8` when uncompressed |
| `PREFIX_lat1.dds` | level 1 with its chain, uncompressed |
| `PREFIX_nntc.json` | `format: "nntc-dds-1"` (the readers here also accept `"ntc-dds-1"`, the same format under the name it had before the rename), source and decoded sizes, `block`, `lod_bias_level1`, per-texture entries, the decoder |

Format decisions worth recording:

* **The descriptor's name is `PREFIX_nntc.json`**, beside the `.dds` files, and `-o name.json` names it outright
  instead. The suffix exists so that a prefix taken from an input's own base name can never name the input -- a
  material JSON encoded in its own directory would otherwise have been overwritten by the asset describing it.
  A consumer takes the descriptor's path as given and derives nothing from it.
* **Plain DX10 `.dds` and plain JSON**, no container, no entropy coder, no runtime library. An asset's size is fixed by
  its layout. The JSON names the `.dds` files by base name and is read by a single-file public-domain parser in the
  viewer (sheredom's `json.h`).
* **The stored byte of an uncompressed plane is the index replicated across the byte** (`k k k..`), so
  `k = byte >> (8 - bits)` recovers the index exactly and the byte goes straight to a UNORM sampler with index 0 at 0
  and the top index at 255. The value of an index is therefore `rep(k)/255` of the range, which equals `k/levels` at
  1, 2, 4, 8 bits and not at 3, 5, 6, 7. **The encoder fits the grid the sampler returns, per format**: `rep(k)/255`
  uncompressed, `k/(2^b-1)` for a BC-packed 1-3-bit plane (the BC palette), `k/255` for the 8-bit bc8 plane. Fitting
  the nominal `k/levels` while shipping the replicated byte is a systematic per-entry bias (0.0034 of a channel's [-1,1] span at 3 bits: half a byte, 0.43/255, over a span of 2), which is why this is done and asserted (`tools/dds_decode.py --grid`).
* `dequantise.kind` is `"range"` (`lo`, `hi`, `levels`) or `"palette"` (explicit per-channel value lists, `lo`/`hi`
  the first and last entries), and the reader applies `lo + sample*(hi-lo)` in both cases. `kind`, not the level
  index, says how a plane is read; both levels may carry either.
* A two-file level 0 is a `files` array of objects each carrying its own `file`, `dxgi_format`, `dxgi_format_id` and
  `channels_stored`, because at three channels the two files differ in format (BC5 then BC4). A reader that knows only
  `file` finds nothing and stops rather than decoding half a plane.
* `bc_palette: "standard"` records that the plane was fitted against the Direct3D specification's `k/7` BC4 palette
  (section 6).
* One `lo/hi` pair per channel covers the whole mip chain, because the GPU applies one scale and bias whatever mip its
  sampler chose; the encoder therefore fits the range over base and chain together.
* `decoder`: `type: "bilinear"`, `terms: "a b sc"`, `C0`, `C1`, `nin`, `nout`, `output: "identity"`, `hidden: []`,
  one layer of `nout x nin` row-major weights and a bias. A reader refuses anything else rather than guessing.
* An independent reader, `tools/dds_decode.py` (about four hundred lines of Python, no shared code), parses the
  headers, decodes the BC blocks, dequantises, applies the layer and reproduces the encoder's PSNR at every level; the
  release gate runs it on every asset written.

---

## 5. The encoder (`src/`, CUDA only)

### 5.1 The objective

`E` is the weighted squared error of the decoded material over a **set of sites**, summed over every stored plane `m`
of the chain:

```
E = sum_m  w_m  sum_{sites of plane m}  sum_c  cw[c] (out_c - t_c)^2
```

A site is a position, not a texel. Every pixel `p` of every plane carries one **centre** site (target: the run's own
source chain at that level, which is the iterated 2x2 box unless a texture asked for a filter) and `K = 4` **fractional** sites at fixed subtexel offsets, where both latents are
bilinearly filtered with clamp-to-edge exactly as the hardware does and the target is the source read by the same rule.
The offsets are the standard rotated grid `P = {(-3/8,-1/8), (-1/8,3/8), (1/8,-3/8), (3/8,1/8)}` in pixels, turned a
quarter turn per texel of a 2x2 cell (which matters at `K = 1` only; `K = 2` is the unrotated diagonal pair `(-1/4, 1/4)`, `(1/4, -1/4)`). **No random numbers, no seed**:
eight fixed constants. `cw[c]` is the texture's weight (`--weights`, default 1 each) times the channel's weight
(`--rgb-weights`, normalised to mean 1). `w_m` is the per-plane mip weight (5.4). `E` is reported divided by `sum(cw)` times the total weighted site count, so it reads as a mean squared error in [0,1] units; at `K = 0` with one stored plane and equal weights it is exactly the centre MSE.

The site's position is carried as a pixel of the plane being fitted scaled into the plane being read
(`x = (p + 0.5 + d) * W / W_m - 0.5`), not through a normalised `u`, because the round trip through `u` rounds twice in
fp32. The double-precision checker goes through `u` on purpose so it is checking the position and not the spelling.

### 5.2 Three exact block solves, alternated

The unknowns are the decoder `(W, b)`, level 1's plane and level 0's plane. Holding any two makes `E` an exact
quadratic in the third, so each block is an **exact minimiser** and `E` cannot increase across any of them; the loop is
monotone by construction (an assert in Debug, a warning naming the round and the block in Release). The BC refinement's acceptance is decided on a double quadratic while `E` is measured in float, so the release gate compares `E` with a `1e-9` relative tolerance. The only steps
that may raise `E` are the grid freezes, which apply a constraint.

**(a) The decoder: global linear least squares.** With `v = [phi ; 1]`, every site of every plane accumulates
`omega v v^T` into one `(nin+1)^2` normal matrix shared by all outputs and `omega v t_c` into one right-hand side per
output; only the upper triangle is accumulated; the small system is solved on the host by Gauss-Jordan with partial
pivoting in double with a ridge of `1e-9 * mean(diag)` so a constant feature cannot make it singular. The mip weights
enter here and nowhere else.

**(b) Level 1: a sparse quadratic on a 9-point block stencil.** With `s` held, `out = m(s) + G(s) sum_t w_t c_t`
where `G(s) = A + sum_i s_i M_i` splits `W` by feature group. A site reads four level-1 texels, so two texels couple
only if within one texel in both axes: a 9-point stencil of `C1 x C1` blocks, five stored per texel by symmetry, in
double. Techniques:

* **Gather assembly, no atomics.** One thread per level-1 texel walks every site whose footprint reaches it and
  accumulates into its own blocks, so the result is independent of scheduling.
* **The quadratic-form trick.** `G^T diag(cw) G` is a quadratic form in the site's level-0 sample `s`:
  `P_0 + sum_i s_i P_i + sum_{i<=j} s_i s_j P_ij` with fixed matrices built once a round from the decoder. So a thread
  accumulates one **scalar** per monomial per stencil slot (6 at `C0 = 2`, 15 at the cap) and expands against the
  matrices once per texel: a `C1 x C1` product per site per touching texel becomes a few multiply-adds and the
  accumulator fits in registers. Measured 14.2 ms to 5.5 ms on the base assembly kernel.
* **Proximal step.** The unknown is the correction `d` from the current plane: `(H + lambda I) d = -grad` with
  `lambda = --ridge (1e-4) * mean(diag H)`. The ridge is structural: with one texture and `C1 = 4` each site's
  contribution has rank at most 3 in a 4-dimensional space, so flat-in-`s` texels have a genuine null direction that
  would otherwise blow up that channel's range and cost every other texel a quantisation step.
* **Block-Jacobi preconditioned conjugate gradients** in double (the preconditioner in float), residual checked every
  four iterations so nothing stalls the device. The stencil is a hat-function Gram matrix, not a Laplacian, so the
  condition number does not grow with resolution and the iteration count stays in the tens.
* **Frozen grid and quantised sweeps.** At round `--q1-start` (default 3) each channel's `lo/hi` is fitted over the
  base and every chain plane together by the 0.1 % / 99.9 % percentiles (`--q1-range pct`; `minmax` is the control and cost 1.3 dB at 4 bits on `model10`, a development measurement with no log in the tree), frozen, and every plane snapped. From then on block (b) is `--q1-sweeps`
  **four-colour Gauss-Seidel sweeps** on the same stencil whose per-value step is the exact argmin over the grid: `E`
  in one value is a parabola, so the minimiser is the grid point whose **value** is nearest the vertex, found by one round to `k` and then whichever of `k-1, k, k+1` has the nearest grid value, ties to the lower (the grid is monotone but not uniform at 5-7 bits). The four colours
  `(tx&1, ty&1)` make a pass an exact joint step, since two texels of one colour are two apart and no site touches
  both. The sweeps cannot raise `E` (`f(0) = 0`, every step lowers `f`), so the proximal ridge stays. Measured during development (`docs/DESIGN.md` 5.2, no log in the tree): at 4 bits this is worth +6 to +7 dB over solving continuously and rounding at the end; the logged 8-bit ablation in `docs/RESULTS.md` 2 shows it at equal quality in half the time.
* **One `lo/hi` per channel for the whole chain**, fitted over every plane with equal weight per value, because the
  format carries one.

**(c) Level 0 under `--l0 palette`: an exact per-texel search.** With `c` held, `out = b + A c + (B + sum_j c_j M_j) s`
is affine in `s`, and a texel's share of `E` over the sites that touch it (its centre site at weight 1, plus the `K`
fractional sites of each of its 3x3 neighbours with the clamp's duplicated taps summed) is an exact quadratic
`A0 + 2 x.A1 + x.A2.x` in its own `C0` values, built from the decoder's structure rather than from repeated decodes.
The joint argmin over all `prod 2^bits_c` states is enumerated when `sum(bits0) <= 8` (at most 256 states), else by
coordinate sweeps; ties keep the current state so equal states cannot cycle. Four colour passes make each pass one
exact joint minimisation over all its texels and let the kernel write in place. Accumulators in double.

**(c') Level 0 under `--l0 bc8` (the default): a continuous 8-bit plane solved like level 1.** With `c` held, `out` is
affine in `s` with a coefficient matrix `Q(c) = B + sum_q c_q N_q` quadratic in `c`: exactly block (b)'s structure with
the two latents exchanged, so the same gather, the same quadratic-form accumulation (`1 + C1 + C1(C1+1)/2` scalars per
slot), the same proximal PCG, the same freeze at `--q1-start` (range fitted over the chain, `level 0 grid frozen:`
printed), and the same quantised four-colour sweeps at 8 bits. Level 0's centre site reads the texel itself at weight
1, which is the one branch that differs.

### 5.3 The BC4 / BC5 pack chosen by analysis by synthesis (`--bc-refine`, `--bc-outer`)

Under `bc8` the plane after the last round is 8-bit but BC4 gives a block only eight values on a line. The pack is
therefore treated as part of the optimisation, not as a post-process:

1. **Seed pack.** The plane is packed once by the ordinary rule (block min/max as endpoints, nearest of the eight
   palette values per texel) and **decoded back into the device plane**. Measured cost on eight runs: 0.37 to 4.61 dB
   of output PSNR, and a refit of level 1 and the decoder alone wins back essentially nothing, because a
   quarter-resolution plane cannot correct full-resolution error.
2. **Per-block refinement, `--bc-refine 4` rounds.** With level 1 and the decoder held, `E` is exactly quadratic in
   level 0 and block (c')'s stencil already holds it; restricted to one 4x4 block with its neighbours held, it is a
   dense quadratic in the block's `16 C0` values, read out of the stencil once per block into shared memory
   (`A2 = H` within the block, `A1 = g + H d` from outside it). Then, per channel:
   * **endpoints by least squares**: the stored byte is affine in `(r0, r1)` at fixed selectors, so `dE` is a 2x2
     quadratic in `(dr0, dr1)` from five scalars; the vertex is rounded to 8 bits and its +-1 neighbours (nine
     candidates) are tried, each required to keep `r0 > r1`;
   * **selectors by exact argmin**: two Gauss-Seidel passes over the block's `16 C0` selectors, each evaluating the
     eight palette entries on its parabola.
   * **Acceptance only on a strict decrease of the block's quadratic**, evaluated in the arithmetic the plane is
     written in (float-rounded palette values), so `E` over the plane cannot rise and where nothing improves the bytes
     are the same bytes. Only the eight-value mode is searched.
   * **Four block colours** `(gx&1, gy&1)` as four launches, one CUDA block of 64 threads per 4x4 block, since two
     blocks couple only through sites within one texel of their border.
3. **Outer repack passes, `--bc-outer 2`.** Refit level 1 (quantised sweeps on its frozen grid) and the decoder against
   the packed level 0; re-solve level 0 continuously (block (c')) against them; snap onto level 0's existing grid; pack
   and refine again. **A pass is kept only if the shipped `E` strictly falls**; otherwise every plane, index, block,
   decoder and grid is restored and the loop stops. Neither range is refitted inside a pass so the comparison is on one
   scale.
4. **`--bc-refine-after N`** re-judges the blocks against the refitted model (default 0; worth +0.00 to +0.06 dB on the eight runs, the most on the run where the pack costs most).
5. **The last refit**: one block (b) and one block (a) against the final blocks, then `E shipped` is measured.

Measured (`docs/RESULTS.md` 6.2): the refinement and outer passes recover **32 % to 71 % of the pack's cost in `E`**,
+0.11 to +2.97 dB of centre PSNR over the seed pack, at 0.06 to 0.22 s per image. Against a 4-bit palette at equal
memory, `bc8` wins all eight image-and-layout pairs (`model36` at one channel 32.96 to 40.33 dB, "night and day" by eye; a development measurement with no log in the tree). The remaining gap to the continuous plane is the block format's own constraint.

### 5.4 The mip chains

* **Every plane is a parameter.** No latent mip is ever filtered into existence. The **source** chain is built
  once; plane `m` of both latents is then solved against `src_m` by the same three blocks, every round. Given `W` the planes are independent, so blocks (b) and (c) for every
  plane are issued at once on their own CUDA streams and waited for once; nothing inside a block travels to the host.
* **The 1:1 rule.** Output mip `m` is decoded from mip `m` of both latents, halved together (floor halving, chain stops while level 0 is at or above `--mip-min`, default 8, per axis). On the GPU this is the `MipLODBias = 2` of section 3.
* **Mip weights enter block (a) only.** The base takes `1 - mix` (`--mip-mix 0.5`) and the chain shares `mix` in
  proportion to `1`, `sqrt(N_m)` or `N_m` (`--mip-weight uniform|sqrt|pixels`, default `pixels`: equal weight per sample, so the chain reads as one big image); dividing a plane's share by its site count `N_m (1 + K)` gives the per-site weight `w_m` that multiplies `v v^T`. Consequence, measured and documented rather than fixed: under
  `pixels` the deep planes hold under 1 % of block (a), so a texture the base does not route through level 0 has no
  full-resolution path at any level and its chain sits at level 1's ceiling; `uniform` lifts a deep level by 8.6-9.6 dB and costs the base 2-3 dB. `--diag` prints per-texture PSNR per level, per-channel statistics per plane and how hard
  each texture leans on each latent channel, so the trade is visible.
* **The source chain is chosen PER TEXTURE**, because it is the only place the encoder picks its own ground truth
  and a material carries textures that want different treatment. The default is the iterated 2x2 box (a one-wide
  plane repeating its row), and a bare command line uses it for every texture, byte for byte. A texture given
  `box`, `mitchell` or `catmullrom` instead has every level `m >= 1` resized DIRECTLY FROM THE BASE to that
  level's size, with its own edge mode (`clamp` or `wrap`, the filter's, not the fit's sampler); every resized
  level is clamped to [0,1] at once, because the cubics have negative lobes and the objective reads the chain
  raw. Two further per-texture rules follow from what a texture IS rather than from how it is filtered: `srgb`
  derives the deeper levels in linear light (in float, never through 8 bits, with the base itself never
  converted), and `normal_map` renormalises each filtered level by the viewer's own rule -- a filtered normal is
  shorter than the normals it came from, and re-lengthening it is what a consumer does. The chain a run builds is
  then the golden reference for everything that run reports, so two filters are two ground truths and their mip
  PSNRs are not on one scale. **Where those per-texture settings are written** is the MATERIAL JSON that 2.1 points
  here for: `nntc_encode material.json` takes one JSON array in texture order, each entry a `file` and any of `type`,
  `filter`, `srgb`, `edge`, `normal_map`, `weight` and `rgb_weights`. An unknown key is refused, the command line
  overrides a key and names the value it replaced, and the asset's own JSON echoes the merged settings back as
  `source.inputs`.
* The level-1 range, the residual-PCA direction and the pca basis are all taken once on the base and used for every
  plane, because one decoder serves every level and channel `j` has to mean the same direction everywhere.

### 5.5 Initialisation

* **Level 1**: `--init box` (default): the block mean of the source over each level-1 texel; `--init pca`: those means
  projected onto the first `C1` principal directions of their covariance (Jacobi on the host), each scaled by its
  **peak** magnitude rather than its standard deviation (sigma clips the tails and cost 0.46 dB, a development measurement with no log in the tree). `pca` helps materials of unrelated pictures, where the box init hands level 1 to the first picture's own RGB: on a three-picture set at `--c0 3 --c1 3` it takes `E` from 1.0412e-03 to 7.465e-04 (-28 %, `docs/RESULTS.md` 7.2). The two split on single images, so `box` is the default.
* **Level 0, `--init0 residual` (default): residual PCA, project, refit, repeat.** Level 0 starts at zero, block (a)
  fits the decoder to level 1 alone; then for each channel `k`: take the residual at the base's texel centres
  (unclamped, `3T` values per pixel), its mean and covariance, the leading eigenvector `e` with its sign fixed
  (largest component positive), set channel `k` to the **centred** projection `(r - mu).e / D`, and run block (a)
  again. **Deflation is by the refit**, not by subtraction: what the new channel can explain leaves the residual
  because the decoder is free to use it, and the decoder rather than the covariance decides what a full-resolution
  channel can carry. `D` is the peak over the whole chain under `bc8` (the freeze rescales it anyway) and the **99 % percentile** of that magnitude under the palette mode, read off a 4096-bin histogram on `[0, peak]` and taken at the bin's upper edge, the clamp saturating the tail (fixed palette on [-1,1]; a peak-scaled seed used a fiftieth of the range and
  lost 5 dB), and under the palette mode each seeded channel is followed by one block (c) and one block (a) so the snap
  is chosen by the objective (else the next eigenvector re-picks the previous direction, cosine 0.9995 on `model10`).
  **Channel 0 is not assumed to be luma**: on a photograph the first direction lands near luminance
  (`+0.436 +0.650 +0.623`), on a tangent-space normal map it is the Y axis. Measured during development, with no log in the tree (`docs/RESULTS.md` 7.2 is the logged form of the same question): the fourth level-0 channel went from costing to buying 0.9-4.6 dB per texture on a four-texture material; `model10` 45.77 to 47.80 dB sampled.
  Weighting the covariance by `cw` was tried and loses.
* Controls: `--init0 texture` takes the direction from the one texture whose own 3x3 residual block carries the most
  variance, dedicating a channel to a picture; `--init0-scope chain` accumulates the covariance over every stored plane
  with the objective's weights; `--init0 luma` is the old control.

### 5.6 The round loop

```
init (5.5); block (a)
round r = 1 .. --rounds (20):
    (a) decoder LS over every site of every plane
    (b) level 1, every plane on its own stream: continuous PCG before --q1-start, freeze at it, sweeps after
    (c)/(c') level 0, every plane
    stop when (E_prev - E) / E < --tol twice running, or nothing moved
end: one more (b) per plane, one more level-0 solve per plane, one last (a)
bc8: seed pack -> refine -> outer passes -> last refit (5.3)
write; assert the zero-change round trip
```

`--rounds 5 / 10 / 20` on `model10`: 42.28 / 42.44 / 42.56 dB in 0.297 / 0.414 / 0.647 s (`docs/RESULTS.md` 2), so ten
rounds are the quick setting and twenty the default.

### 5.7 Determinism and exactness

* **Byte-identical reruns** are a release gate. There is no seed in anything that reaches the asset (the one pseudo-random generator in the tree is the fixed-seeded linear congruential probe of the `--check` finite-difference test, which restores every value it perturbs); every reduction that feeds a result is a
  fixed-grid two-stage reduction (each block writes its partial to its own slot, a second pass adds by index): the
  decoder's normal equations, the objective, CG's inner products, the stencil diagonal, the range and movement probes,
  the pca covariance. The only atomics count moved texels (order-free) and sum the BC refinement's accepted decrease
  for the log (a report number no decision reads).
* **One function defines value <-> index**, and on the device it spells its multiply and add as separately rounded
  operations (`__dmul_rn` and `__dadd_rn` on the device; `-ffp-contract=off` on the gcc / clang host build) so nvcc cannot fuse them into an FMA the host has no
  counterpart for. Before writing, every level-1 value (and level 0's under the palette mode) must equal the grid value of its stored index or the encoder refuses to write; in the writer, every stored index dequantised and requantised by the published `lo/hi` must come back as itself. Under `bc8` level 0 holds the refined blocks, not an index, and is not part of that check.
* Level-1 indices are re-quantised with a zero-change assertion; the objective has a double-precision host twin
  (`--check`) with finite differences for the block solves.
* Speed (RTX 5090): `model10` 750x1122 with seven levels 0.62 s for the whole encode in the palette mode (`c0 2 bits0 3 c1 4 bits1 8`) and 1.42 to 1.95 s under the default `bc8`; `model34` 2888x4320 5.47 s
  against 906.6 s for the evolutionary encoder at equal PSNR (38.59 vs 38.53). Techniques that got there: streams per
  plane and no host synchronisation inside a block, the quadratic-form assembly, the symmetric triangle in block (a),
  residual checks every four CG iterations.

---

## 6. Why BC4 / BC5, and which palette

* **BC4 / BC5 are the natural formats for a latent plane**: independent UNORM channels, 4 bits per channel per texel,
  universally supported, and their 8-value-per-block structure is close enough to a smooth latent that the encoder can
  optimise inside it (5.3). Level 1 stays uncompressed because 4-8 bits of a fitted range would not survive a block
  palette at a quarter of the resolution, and it costs only `C1/2` bits per pixel anyway.
* **Channel-count layout**: 1 channel one BC4 (4 bpp), 2 one BC5 (8), **3 a BC5 plus a BC4 (12)** rather than two BC5s
  with a padding channel nobody reads, 4 two BC5s (16). Both files carry the whole chain; the JSON records each
  file's own format.
* **1-3 bits per channel pack losslessly**: with endpoints 255 and 0 the eight-value palette *is* the 3-bit grid
  `k*255/7`; 2 bits use the six-value mode with endpoints 85 and 170 (palette 0, 85, 170, 255); 1 bit the six-value
  mode's fixed 0 and 255. The index survives (`byte >> (8-bits)`), the exact value shifts by up to half a byte at 3
  bits and the JSON publishes the palette of the format actually shipped.
* **The standard palette is the default and the only one written today.** `bc_palette: "standard"` means the
  Direct3D functional specification's `k/7` interpolation, which every decoder on every vendor is required to return
  within tolerance. Measured on one NVIDIA GPU, the interior of the eight-value palette is evaluated with six-bit
  weights (9, 18, 28, 36, 46, 55 over 64), so a raw 3-bit level-0 view differs by at most 4/255 from the uncompressed
  twin and the decoded image by 3-5/255. Fitting against that would be a vendor-specific mode (section 10) and is
  opt-in only, never the default; AMD, Intel, Qualcomm and software decoders return the standard values.
* `--bc0 both` writes the uncompressed twin of the same solve (`PREFIX_u`) beside the BC asset, so the two decode paths can be compared without a second encode; it is what gives `bc_check` its reference.
* Only the eight-value mode is searched by the refinement; the six-value mode trades two interpolants for a hard 0 and
  255, which a latent on a fitted range has no use for.

---

## 7. What is deliberately absent

No entropy coder, no DCT plane, no evolutionary search, no hidden layers or activations, no per-block decoder bank,
no container, no CPU encoder, no runtime library. The asset is fixed-rate by layout, so a noisier plane at equal
quality costs nothing and rate is not a column in any table.

---

## 8. Measurements on record (`docs/RESULTS.md`)

| what | figure |
|---|---|
| speed vs the evolutionary encoder, ten image-and-layout pairs | 113x to 167x, at -1.75 to +1.72 dB |
| `model34` 2888x4320, `c0 1 bits0 4 c1 2 bits1 8` | 38.59 dB in 5.47 s vs 38.53 dB in 906.6 s |
| `bc8` vs 4-bit palette at equal memory, eight pairs, one binary on both sides | wins all eight, +0.94 to +4.10 dB |
| BC refinement + outer passes, eight runs | 32-71 % of the pack's `E` cost recovered, +0.11 to +2.97 dB |
| frozen grid + quantised sweeps vs round-at-end, 4-bit level 1 | +6.08 and +7.11 dB (development measurement, `docs/DESIGN.md` 5.2, no log in the tree) |
| independent Python reader vs encoder report | equal at every level; worst |published - sampled| grid value 1.7e-08 |
| BC pack at 3 bits | lossless (packing PSNR 154.80 dB, max err 6.5e-06/255) |
| determinism | byte-identical `.dds` and `_nntc.json` across reruns (gate) |

### 8.1 A worked material example: `m1-m4`, the default layout

Four 512x512 textures of one material (`m1.png` .. `m4.png`), encoded with no layout flags, so at the default
`--c0 2 --c1 4 --bits1 8 --l0 bc8 --mips 1` layout (`out/disclosure/log_m1234_default.txt`, the JSON beside it as `m1234_default.json`; 2026-09-12, RTX 5090; the log records the header `4 textures, 512x512 source, nout 12`, not the command line; the log predates the `_nntc` descriptor suffix and the rename, so it names the descriptor `m1.json` where a run today writes `m1_nntc.json`):

```
nntc_encode m1.png m2.png m3.png m4.png -o out_m1234_disclosure/
```

| | |
|---|---|
| decoder | `nout` 12, `nin` 14, 180 weights and 12 biases, `terms a b sc` |
| level 0 | 2 channels x 8 bits, one `BC5_UNORM` file, 7 mip levels (512x512 .. 8x8) |
| level 1 | 4 channels x 8 bits, `R8G8B8A8_UNORM`, 7 levels (128x128 .. 2x2) |
| files | `m1_lat0.dds` 349,652 + `m1_lat1.dds` 87,524 + `m1.json` 6,647 = 443,823 bytes |
| in-memory | 10.00 bpp for the set at the base (level 0 8.00 + level 1 2.00), 13.33 with the chain; 2.50 bpp per texture at the base, 3.33 with mips |
| per-texture PSNR, base | 25.66 / 33.36 / 31.28 / 36.14 dB (8-bit, texel centres) |
| centre / sampled PSNR | 29.79 / 31.72 dB (fp, every output; sampled = the K = 4 fractional sites) |
| mip PSNR, worst texture per level | M1 26.23 M2 27.60 M3 27.77 M4 27.49 M5 26.90 M6 28.94 dB |
| the pack | continuous 29.88 -> seed pack 29.64 -> refined 29.71 -> refitted 29.79 dB; `E` 6.620e-04 -> 6.913e-04 -> 6.793e-04 -> 6.667e-04; both outer passes accepted |
| residual-PCA seed | ch0 45.5 % of the residual's variance, ch1 48.5 % of what was left |
| time | 1.321 s total: (a) 64 ms, (b) 292 ms, (c') 205 ms, 74 objective passes 113 ms; 20 rounds |
| device memory | 157.9 MB |

The first texture is the hard one of the four (a busy albedo) and the fourth the easy one; the per-texture figures are
what `--weights` is for. The decoded PNGs of every texture at every level are written beside the asset by the run itself, and
`build\Release\nntc_view.exe out_m1234_disclosure\m1_nntc.json` (the log's own run wrote it as `m1.json`) shows the
asset on the hardware sampler.

---

## 9. Relation to other work

* **Bart Wronski, "Dimensionality reduction for image and texture set compression" (2020,
  https://bartwronski.com/2020/05/21/dimensionality-reduction-for-image-and-texture-set-compression/)** stores a
  texture set as a few shared low-rank planes plus per-texture coefficients, the observation this design starts from.
  NNTC adds the full-resolution selector plane and the products, so the combination varies per texel, the exact
  alternating solves, the fit under the hardware filter, and the shipped BC blocks chosen against the decoder.
* **The author's "Experiments in Luma-Optimized and Mipmapped DXT1 Compression"
  (https://web.archive.org/web/20201024153426/https://sites.google.com/site/richgel99/luma_chroma_texture_compression)**
  split a texture into a full-resolution luma plane and a lower-resolution chroma plane in a block format with mipmaps;
  the two-resolution layout here is that split with both planes learned and the decoder fitted per material.
* **BC1 / PVRTC / ASTC** are fixed-coefficient members of the "small discrete data reconstructed by an affine map"
  family; here the coefficients are fitted per material, the low-resolution plane is learned, and one decoder yields a
  whole material.
* **Analysis by synthesis is how BC7 / ASTC encoders choose blocks**; the new element is that the synthesis is the
  decoder's output under the sampling operator, not the block's own texels, and that the outer loop refits the
  rest of the model against the packed blocks and keeps a pass only if the shipped objective falls.
* The **evolutionary / finite-difference encoder** in the predecessor tree established the site set, the bilinear
  decoder family, the joint selector search and the filter-in-the-loop framing; this tree keeps only what the shader
  decodes and replaces the search with exact solves.

---

## 10. Techniques that follow from the design and are not built

Named here so their shape is on record. **None of these is implemented** unless a line says so.

1. **The BC encode inside every round.** Run the per-block refinement of 5.3 after block (c') in every round, so
   level 1 and the decoder are always fitted to a plane already on the block format's manifold. `E` stays monotone
   for the same reason (a block step accepts only a strict decrease); the cost is a pack per round. This is the
   obvious next lever, since what remains after 5.3 is the block constraint itself.
2. **Joint endpoint-and-selector moves, and the six-value mode.** The current refinement alternates endpoints and
   selectors and searches a +-1 neighbourhood; a joint move (re-choose selectors for each of the nine endpoint
   candidates before judging), a wider endpoint neighbourhood, or admitting the six-value mode for blocks that want a
   hard 0 or 255 would reach lower local minima of the same quadratic.
3. **A post-pack refit for the palette mode**, and a BC-aware palette search: the 4-bit palette mode packs lossily and
   is not refined; the same block quadratic applies to it directly.
4. **The opt-in NVIDIA palette (`--bc-palette nvidia`)**: fit against the `n/64` interior weights that hardware
   returns and publish those values; correct there, wrong everywhere else, so never the default.
5. **Level 1 or level 0 in richer block formats**: BC7 (three or four correlated channels per block with partitions),
   BC6H for a wider-range latent, ASTC on mobile, or two BC4s vs one BC5 with a shared selector plane. Any format whose
   decode is affine in its stored values after the blend keeps the sampling argument; the block quadratic of 5.3
   applies to any of them with its own endpoint and selector structure.
6. **A mip-weight schedule per texture.** The sharper and sRGB-aware source filter this item used to propose IS
   BUILT (5.4): the chain is chosen per texture, with `box` / `mitchell` / `catmullrom` resized from the base,
   an `srgb` derivation in linear light and a `normal_map` renormalisation. What is still not built is the other
   half of the item -- a mip weight that favours the chain when a material has an easy texture, `uniform` or
   `sqrt` per texture rather than per set.
7. **RGBA textures** (`nout = 4T`, nothing else changes). The five-or-six-texture half of this item IS BUILT:
   the cap is six (`nout <= 18`), the shader carries the wider constant buffer and the viewer cycles all six.
   What has NOT been measured is where one shared `W` stops serving them, since the layout does not grow with
   the count.
8. **Perceptual weighting with no format change**: a per-pixel `cw` from local variance of the source (texture
   masking) multiplied into `E`, so the exact solves reallocate toward low-masking regions with zero bits and no decoder
   change. The objective already carries per-output weights; a per-site weight is the same accumulator.
9. **A learned or fixed post-filter in the objective**: a small neighbourhood filter on the decoded output, fitted
   jointly (the predecessor tree measured a learned 5x5 filter as a Pareto win under evolution strategies); here it
   would be a second least-squares block if kept linear.
10. **More sites, other sites**: `K = 8` or a per-texel phase pattern; mip-level blends (trilinear sites, two levels'
    inputs blended before one decode) and anisotropic tap patterns as sites, so the objective matches the consumer's
    filter exactly rather than by the affine argument alone.
11. **Load-time transcoding of the decoded material to BC7 / BC5 / ASTC** for engines that will not run the sampling
    shader, with the transcoder inside the outer loop's acceptance test (decode, transcode, measure, keep only on
    improvement), as in the predecessor's disclosures.
12. **A shared decoder across materials** (one `W` for a family of assets, fitted over their union of sites) and
    **per-mip decoders** (a `W_m` per level, which removes the block (a) competition of 5.4 at the cost of `45` numbers
    per level).
13. **A CPU or Vulkan / D3D12 compute encoder** for machines without CUDA: every block is a kernel and the algorithms
    carry over unchanged; only NVIDIA is supported today, and that is the toolchain, not the method.
14. **Rate-aware layouts**: an outer search over `C0`, `C1` and `bits1` per material against a bpp budget, since the
    encoder is fast enough to try several layouts and every asset's size is known in advance.
