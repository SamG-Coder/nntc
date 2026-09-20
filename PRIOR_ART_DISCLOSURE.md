# Prior art disclosure: NNTC (Non-Neural Texture Compression), its encoder, asset format and viewer

Richard Geldreich, 2026-09-12; revised 2026-09-18. NNTC, "Non-Neural Texture Compression", is a texture and material compressor whose
encoder is `nntc_encode` and whose reference decoders are the Direct3D 11 viewer `nntc_view` and its Vulkan twin `nntc_view_vk`. NNTC is a bilinear latent material codec with analysis-by-synthesis block encoding.

This file is a public,
dated description of every algorithm, technique and method in this tree, written so that a graphics programmer could
reproduce the system from it. It covers the encoder (`src/`, with its CPU backend in `src/cpu/`), the decoder as it runs on a GPU (`viewer/`,
`viewer_vk/`), the asset
format (`docs/FORMAT.md`), the choice of BC4 / BC5, and the techniques that follow naturally from the design and have
not yet been built (section 10). The development history of this code, from `v0.1-skeleton` (2026-09-11) to
`v0.10g.2-weight-clamp` (2026-09-12), is a sequence of local annotated tags in the predecessor tree this repository
was copied from; here it begins at `v1.0.0-nntc`, the imported tree, and `v1.0.1-rename`, the rename to NNTC. The
earlier evolution-strategies encoder this one replaces lives in a tree of its own and had its own disclosures.

Everything the 2026-09-12 text states as measured is in `docs/RESULTS.md` with the log it came from; every
measurement added on 2026-09-18 names the document that carries it. Everything in section 10 is explicitly **not
implemented** unless it says otherwise.

**What the 2026-09-18 revision changes.** It revises the 2026-09-12 text, which described the tree at `v0.10g.2` /
`v1.0.1-rename`, to the tree at `v1.3.4`. Level 1's mip shift is now applied in the texture-coordinate gradients and
not as a sampler LOD bias, and why (3); what filtering preserves is stated exactly, including the product terms under
trilinear and anisotropic filtering (2.4); the representation's gauge freedom (2.5); the Vulkan viewer and its
cooperative-vector decode (3.1, 3.2); the ridge ladder of block (a) (5.2); corrections and additions to the BC pack
chosen by analysis by synthesis (5.3); the exact-duplicate-channel rule of the level-1 init (5.5); the portable CPU
backend (5.8) and the encoder's measured rounding noise floor (5.9); what the palette mode's lossless pack guarantees,
exactly (6); an extended relation to other work (9); and section 10 items 15-18, of which 15 says which decoder inputs
keep the sampling property and which break it. Stale statements of the earlier text are corrected in place.

---

## 1. The idea in one paragraph

A texture, or a material of up to six textures, is stored as **two quantised latent planes and one small affine
decoder** (two `.dds` textures, three when level 0 has three or four channels), and is reconstructed in a pixel
shader from **two ordinary hardware `Sample()` calls** -- three when level 0 has three or four channels -- and a
handful of multiply-adds. Level 0 is a full-resolution plane of one to four channels shipped as ordinary BC4 / BC5 blocks; level
1 is a plane at a quarter of the resolution per axis, one to four channels of 4-8 bits, uncompressed. The decoder is
`out = W phi + b` over `phi = [c ; s ; s (x) c]`: the level-1 sample, the level-0 sample and their pairwise products.
The representation is fitted to be **valid under the GPU's sampling operator**: not texel by texel but at fractional
positions where both latents are blended by the hardware's own bilinear rule, so the same asset serves
bilinear, trilinear and anisotropic filtering with no runtime code beyond the samples and one scale on level 1's
texture-coordinate gradients: exactly at the fitted bilinear positions, and under trilinear and anisotropic
filtering exactly in every affine term, with a stated, bounded error in the product terms (2.4). The encoder is not a neural
network trainer: it is an **alternating sequence of exact block solves** (linear least squares, a sparse quadratic on a
9-point stencil, and per-texel or per-block exact searches over the shipped format's own values), running on CUDA in about one second to about a minute depending on the pixel count, the texture count and the layout (or, with no NVIDIA GPU, on a portable CPU backend 8 to 14 times slower, 5.8), 113-167x faster than the evolutionary encoder it replaces, within about 2 dB of it either way and ahead where the layout has room. The BC4 / BC5 blocks the file holds are **chosen by analysis by synthesis against the decoder's output**,
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
approximate. The decoder's product terms are formed after both samples are dequantised, so they do not break it (the
invariant of this paragraph; they do not commute with the blend, 2.4).
The commutation needs only that the tap weights sum to one, which every standard filter's do (a border address mode
fits the same argument, the border colour counting as one more tap in stored units). A per-texel decode the texture
unit performs BEFORE its blend -- a BC block, an sRGB or a float format -- is on the other side of the argument and
needs nothing: what the sampler blends is the decoded texel.

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

**What filtering preserves, exactly.** Write `S(u) = sum_t omega_t y[t]` for the sampler's normalised blend of stored,
decoded texels and `f(s, c) = b + A c + B s + sum_ij m_ij s_i c_j` for the decoder, `m_ij` the column of `W` that
multiplies `s_i c_j`. At every site the encoder fits, the shader evaluates `f` of exactly the two hardware blends the
encoder modelled, so the fit is exact there, not approximate. Away from the fitted sites every affine term commutes
with any blend, because an affine map commutes with a normalised convex combination; the product terms do not. They
are products of FILTERED values, not filtered products: over a footprint whose taps pair positions `q` with weights
`omega`,

```
sum_q omega_q f(s_q, c_q)  -  f( sum_q omega_q s_q , sum_q omega_q c_q )  =  sum_ij m_ij Cov_omega(s_i, c_j)
```

the weighted covariance of a selector channel and a colour channel inside the footprint, bounded by `|m_ij| sigma_s
sigma_c` and small where `c`, a quarter-resolution plane, is smooth at level 0's scale. Under trilinear filtering the
error is exact and closed-form. Level 1 is read with its gradients scaled so that its LOD equals level 0's (3), so
both textures blend their two mips `a` and `b` with the SAME fraction `t`, and with `s_a, c_a` the bilinear samples of
mip `a`

```
f(s, c) = (1 - t) f(s_a, c_a) + t f(s_b, c_b) - t (1 - t) sum_ij m_ij (s_a - s_b)_i (c_a - c_b)_j
```

(expand the bilinear form in `t`: the affine terms cancel exactly). The decode of a trilinear blend is the blend of
the two fitted mip decodes except for one bilinear term that vanishes at the mips, peaks at `t = 1/2` with the factor
1/4, and is the product of how much the selector plane and the colour plane each change between the two levels. Under
anisotropic filtering with scaled gradients level 1's taps lie along a line four times longer in texture space than
level 0's, so the two footprints are not paired tap by tap and the covariance form above describes the error rather
than equalling it. None of these terms has been measured; section 10 item 10 names the sites that would put them
inside the objective, and item 15 says which other inputs keep or break each of these properties.

### 2.5 The gauge freedom of the representation

For invertible `P` (`C0 x C0`) and `Q` (`C1 x C1`) and offsets `a` and `beta`, the maps `s -> P s + a` and `c -> Q c +
beta` send `[1 ; c ; s ; s (x) c]` to an invertible linear transform of itself: `(P s + a) (x) (Q c + beta) = (P (x)
Q)(s (x) c) + (P s) (x) beta + a (x) (Q c) + a (x) beta`, and every term on the right is a linear combination of
entries of `phi` and the constant. So there is a decoder under which the transformed latents decode to exactly the
same output, and block (a) finds it. The representation is defined only up to an affine reparametrisation of each
latent, and that choice is free to be made for the benefit of quantisation and packing. The per-channel `lo/hi` freeze
of 5.2 is one such choice (diagonal `P` and `Q` with offsets) and is built. A rotation of level 1's or level 0's
channels before the freeze -- to equalise their ranges, to decorrelate the two channels of a BC5 file, or to align a
channel with what a block format represents best -- costs nothing at runtime and is **not built**. The same algebra is
why the sign convention of the residual seed (5.5) changes nothing but readability.

---

## 3. The decoder at runtime (the viewers, `viewer/` and `viewer_vk/`)

`nntc_view` is a Direct3D 11 program, and `nntc_view_vk` its Vulkan transliteration (3.1), that binds the asset's `.dds` files as ordinary shader resources, samples each
once per pixel in the pixel shader at the same texture coordinate with the same filter, dequantises, forms `phi` and
applies the one layer. Nothing is decompressed, repacked or repaired at load. Specifics that a consumer must copy:

* **Level 1 is sampled with both texture-coordinate gradients multiplied by `2^lod_bias_level1` = 4** (the JSON's
  `lod_bias_level1` = `log2(block)` = 2), through `SampleGrad` / `textureGrad` (`viewer/bin/nntc_view.hlsl:81`,
  `viewer_vk/bin/view.frag:73`); level 0 goes through the same explicit-gradient path with the unscaled derivatives
  (`nntc_view.hlsl:73`, `view.frag:65`), so that no implementation can filter the two latents differently (an
  implicit sample beside an explicit-gradient one may be: Mesa's llvmpipe gave only the implicit one anisotropy). Output mip `m` is decoded from mip `m` of both latents (the 1:1 rule, section 5.4),
  but the hardware picks each texture's mip from its own texel density and level 1 has a quarter of the texels per
  axis, so left alone its LOD sits two mips finer than level 0's. Since `W1 = W0 / 4`, scaling both UV derivatives by
  4 makes level 1's texel-space derivatives equal to level 0's, so its computed LOD is level 0's exactly, in the LOD formula (hardware LOD precision is
  implementation-defined), before any clamp and at every distance; at 1:1 on screen it goes from -2 to 0, still mip 0, so the base is untouched. Two
  consequences are part of the method: under trilinear filtering both textures blend their two mips with the same
  fraction `t` (2.4 relies on it), and the rule extends to any latent at a power-of-two ratio `2^k` of level 0, read
  with its gradients scaled by `2^k` (a non-power-of-two ratio would give each texture its own `t`). Without the shift
  the base is right and everything from mip 1 down is decoded from the wrong colour plane and looks splotchy at
  level-1 texel scale. Found by eye; key `L` toggles it.
* **Why gradients and not a sampler LOD bias.** Until `v1.1.21` both viewers applied the shift as `MipLODBias = +2` on
  level 1's sampler. That selects the same mips where the hardware keeps a negative base LOD under magnification
  (NVIDIA, AMD), but on Intel Xe integrated graphics (a 13th-generation Core i7, under Direct3D 11 and Vulkan) it made
  a magnified picture very blurry, consistent with a base LOD floored near 0 before the bias is added. The gradient
  form raises the LOD inside the hardware's own computation and is right on all three, so it is what the format asks a
  consumer for. It is not the same filter under anisotropy: scaling both gradients also makes the line the anisotropic
  taps are spread along four times longer, and against a 16x-supersampled reference on close, steeply angled views the
  bias was ahead by 3 to 7 dB on an RTX 5090 and on an integrated Radeon, with no difference square-on or with
  anisotropy off (`README.md`, "The two ways to apply the level-1 mip shift"). A refinement, documented there and not
  shipped, scales only the SHORT axis of the footprint ellipse: the Jacobian's singular vectors (the closed-form eigenvectors of `J^T J` for the 2x2 UV Jacobian `J`) give the
  axes, the short one is scaled by 4 and the two axis vectors go to `SampleGrad`. It must know the
  sampler's maximum anisotropy `A`, because the hardware's LOD is `log2(max(short, long / A))` and it is that whole
  quantity which has to rise by 2; measured, it matches the bias or edges it by up to 2 dB, and with `A` wrong it is
  worse than the plain scale. A consumer that knows its vendor may use the bias there; applying both is a shift of
  four mips. Mesa's lavapipe gives explicit-gradient samples no anisotropy at all, which the specification allows.
* **Trilinear and anisotropic filtering need no code beyond the samples.** Both are blends of more latent samples and
  the dequantisation is affine, so every affine term of the decode is exactly the blend of its decodes; the product
  terms differ from that blend by the footprint covariance of 2.4, exactly `t (1 - t)` times one bilinear term under
  trilinear filtering. The viewers use 8x anisotropy by default in trilinear mode.
* A two-file level 0 (3-4 channels) binds its second file to a second slot; the shader reads `.rg` from both and uses
  only `.r` of the second at three channels, which is what a BC4 in that slot returns.
* Mips off is the sampler's `MaxLOD` (key `M`), nothing reloaded.
* Key `V` renormalises the shown output as a tangent-space normal (unpack to [-1,1], unit length, repack), leaving
  grey-ish pixels (no direction) and black-ish or z < 0 pixels (into the surface) alone. A display aid, not part of the
  format.
* The overlay prints the GPU memory the bound textures cost in bits per pixel of the decode extent divided by the material's texture count (2.50 bpp at the base for the four-texture example of 8.1): BC levels counted as whole 4x4 blocks, base and whole chain (x4/3) separately.
* `--shot` renders one frame headless with no keyboard input at all, so its bytes are a function of the command line and
  the asset; the release gate runs it five times and asserts they agree.

The viewer also contains a **load-time BC4 / BC5 packer** (`shared/bc_pack.h`, key `4`) for assets whose level 0 arrives
uncompressed: 1-3 bits per channel pack losslessly (endpoints 0 and 255 through the mode that provides them, the index
becomes the selector), 4 bits pack lossily by the stb_dxt-style min/max endpoint rule. It is a validation of the pack
and the seed of the encoder's own refinement, not what a default asset contains. `bc_check` validates `bc_pack.h`
against iOrange's `bcdec` (precise float decoder, integer decoder and the tree's own decoder), and asserts per channel
that at 1-3 bits the stored index survived the block format.

### 3.1 The same asset through a second graphics API (`viewer_vk/`)

`nntc_view_vk` draws the same asset on Vulkan 1.3, on Windows and Linux, with a fragment shader that transliterates
the Direct3D shader line for line -- the feature order and the dequantise-after-sampling rule are the format's
contract, not something either viewer restates -- and the same load-time pack from the same `shared/bc_pack.h`. It is
the cheapest evidence that the fit is to the hardware sampling operator and not to one API. Measured on one RTX 5090
(`viewer_vk/README.md`, "Is it the same picture?"): with anisotropy off the two viewers' frames are byte-identical in
every state a headless `--shot` can be put in (a single image, each texture of a four-texture material, a two-file
level 0, mips and level 1's shift live, the raw latents, the cube, the load-time pack); with anisotropy on they differ
only where the taps are placed: 46 to 49 of 255 on one texture of the material at 49.85 dB, over a small fraction
of the pixels, and 149 of 255 on a few silhouette pixels of a 64x64 image at `--z -50` (64.34 dB);
on an integrated Radeon the difference is that vendor's bilinear weight rounding, 4 to 9 of 255 at 65.5 to 66.5 dB.

### 3.2 The decode as one cooperative-vector instruction (`viewer_vk/bin/view_coopvec.frag`)

Because the decoder is one affine layer, everything after the samples and the products is one matrix-vector
multiply-add, `out = W phi + b`, which is exactly the operation `VK_NV_cooperative_vector` puts in a shader. The
shading language requires the matrix shape and its interpretation arguments to be compile-time constants while `nin`
and `nout` vary per asset, so `W` is zero-padded to the format's maximum, 18 by 24, and `phi` is zero past `nin`: a
zero row is an output nobody reads and a zero column multiplies a zero feature, so the padded product is the unpadded
one in exact arithmetic and one shader serves every asset. At load `W` is written into that padded row-major fp32
matrix and converted once by `vkConvertCooperativeVectorMatrixNV` -- called first with no destination to learn the
size, then to fill it -- into the device's implementation-defined inferencing-optimal fp16 layout (1152 bytes for
18x24 on the machine measured); the bias follows at the next 64-byte boundary, and its offset reaches the shader as a
specialization constant, because the size of the driver's layout is the driver's business
(`viewer_vk/main.cpp:2089-2101`, the specialization constant at `:2443-2464`). The path is chosen by a runtime query -- the extension, its feature, the FRAGMENT
stage in `cooperativeVectorSupportedStages` (a driver may offer the extension for compute only), and an enumerated
type tuple -- preferring an fp32 result and taking the fp16 x fp16 -> fp16 tuple the extension mandates where that is
all the device offers; an asset whose `W` or bias exceeds fp16's 65504 stays on the plain path, and every other device
draws the plain shader and never hears of the extension. Measured (`viewer_vk/README.md`, "Cooperative vectors"; RTX
5090, driver 581.80, 2560x1440, 200 frames, GPU timestamps around the scene draw): 12.0 against 63.7 us (5.3x) at
`nin` 24, `nout` 15, and 10.7 against 42.9 us (4.0x) at `nin` 19, `nout` 12. The plain side is the viewers' own loop,
which cannot unroll because its bounds are uniforms, so this compares the two shaders a viewer of this format runs,
not the instruction against a hand-optimised multiply (section 10 item 18). The frames differ by at most 1 of 255
(74.62 and 72.52 dB): the fp16 rounding of the weights and of `phi`, some weights being fp16 subnormals (the smallest
nonzero `|W|` of `pavingstones141_1k_c0_4` is 3.24e-5).

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

## 5. The encoder (`src/`: CUDA, and a portable CPU backend in `src/cpu/`)

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

### 5.2 Three block solves, alternated

The unknowns are the decoder `(W, b)`, level 1's plane and level 0's plane. Holding any two makes `E` an exact
quadratic in the third. Blocks (b) and (c) take that quadratic's **exact minimiser**; block (a) solves a ridged form of
it and takes the step only when its own quadratic says `E` does not rise beyond a fixed tolerance, trying the shipped
ridge first so a well-conditioned system gets the decoder it always got, bit for bit. `E` therefore does not increase
across any block beyond that tolerance and the loop is monotone by construction, with a warning naming the round and
the block when one rises beyond the loose comparison - a warning in both configurations, never an assert, because it is
a floating-point judgement. The BC refinement's acceptance is decided on a double quadratic while `E` is measured in float, so the release gate compares `E` with a `1e-9` relative tolerance. The only steps
that may raise `E` are the grid freezes, which apply a constraint.

**(a) The decoder: global linear least squares.** With `v = [phi ; 1]`, every site of every plane accumulates
`omega v v^T` into one `(nin+1)^2` normal matrix shared by all outputs and `omega v t_c` into one right-hand side per
output; only the upper triangle is accumulated; the small system is solved on the host by Gauss-Jordan with partial
pivoting in double with a ridge of `1e-9 * mean(diag)` so a constant feature cannot make it singular. The mip weights
enter here and nowhere else.

**The ridge ladder.** The ridged solve minimises `E + ridge ||[W b]||^2`, which is `E`'s minimiser only while the
ridge is negligible against the solution; on a picture that drives the normal matrix nearly singular it is not (one
white pixel on a black field raised `E` by 13 % across block (a), `docs/DESIGN.md` 3.1). So the block measures its own
step at no cost: the accumulated `A` and `rhs` ARE `E`'s quadratic in `(W, b)` up to a constant, `Q(x) = sum_c cw[c]
(x_c^T A x_c - 2 x_c . rhs_c)`, and the difference of `Q` between the incoming decoder and a candidate is the change
in `E`. The candidates are the solves at `1e-9`, `1e-12` and `0` times `mean(diag A)`, in that order
(`src/solve_decoder.cu:242`); beside each `Q` a rounding bound
`eps mm sum_c cw[c] (sum_ij |x_i| |a_ij| |x_j| + 2 sum_i |x_i| |rhs_i|)`, divided by the same normaliser as `dQ`
(`src/solve_decoder.cu:351-372`), is computed for both decoders, and a candidate is taken only when `dQ + err_prev + err_cand <= 1e-12` in
`E`'s units, so a comparison that cannot resolve its own sign is a refusal; a non-finite solve is refused outright; if
no rung is taken the incoming decoder is kept. On the one-white-pixel image the run ends at 60.46 dB where it ended at
53.69 dB without the ladder (`docs/MATHEMATICS.md` 3).

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
   palette values per texel) and **decoded back into the device plane**. Measured cost on the eight runs of `docs/RESULTS.md` 6.2,
   current binary: 0.45 to 5.27 dB of centre PSNR (6.1 recorded 0.37 to 4.61 dB with the binary before the
   residual seed), and a refit of level 1 and the decoder alone wins back essentially nothing, because a
   quarter-resolution plane cannot correct full-resolution error.
2. **Per-block refinement, `--bc-refine 4` rounds.** With level 1 and the decoder held, `E` is exactly quadratic in
   level 0 and block (c')'s stencil already holds it; restricted to one 4x4 block with its neighbours held, it is a
   dense quadratic in the block's `16 C0` values, read out of the stencil once per block into shared memory
   (`A2 = H` within the block, `A1 = g + H d` from outside it). The quadratic is expanded about the continuous plane
   the loop left: the stencil is assembled once there and the pack enters as the correction `d = packed - continuous`
   (`src/refine_bc.cu:155-175`), so one assembly serves the seed and every refinement round, and `A1` carries the
   neighbours' already packed corrections -- a block is judged against packed neighbours. The quadratic is per plane,
   with no mip weight and no ridge. Then, per channel:
   * **endpoints by least squares**: the stored byte is affine in `(r0, r1)` at fixed selectors, so `dE` is a 2x2
     quadratic in `(dr0, dr1)` from five scalars; the vertex is rounded to 8 bits and its +-1 neighbours (nine
     candidates) are tried, each required to keep `r0 > r1`. The vertex is used only when the 2x2 determinant exceeds
     `1e-12` of the product of its diagonal (`src/refine_bc.cu:335`): on a rank-1 block, every selector equal, `p` and
     `q` are parallel and what survives is cancellation noise, and the nine candidates are then taken around the
     current endpoints;
   * **selectors by exact argmin**: two Gauss-Seidel passes over the block's `16 C0` selectors, each evaluating the
     eight palette entries on its parabola.
   * **Acceptance only on a strict decrease of the block's quadratic**, evaluated in the arithmetic the plane is
     written in (float-rounded palette values), so `E` over the plane cannot rise and where nothing improves the bytes
     are the same bytes. Only the eight-value mode is searched.
   * **Four block colours** `(gx&1, gy&1)` as four launches, one CUDA block of 64 threads per 4x4 block, since two
     blocks couple only through sites within one texel of their border. Texels of a partial edge block that lie
     outside the plane (odd-sized deep mips) carry zero rows, so their selectors never move.
3. **Outer repack passes, `--bc-outer 2`.** Refit level 1 (quantised sweeps on its frozen grid) and the decoder against
   the packed level 0; re-solve level 0 against them -- under the default frozen grid by block (c')'s quantised four-colour sweeps on level
   0's frozen 8-bit grid, and, when no grid was frozen (`--q1-start 0`, or a loop that stopped before the freeze round), continuously
   followed by a snap onto level 0's existing grid --
   then pack it FROM SCRATCH with the seed rule and refine again. The fresh seed matters: the refinement is an
   alternating descent that stops in a local minimum, and re-seeding from a re-solved plane is what lets a pass leave
   it, where `--bc-refine-after` below continues from the blocks it holds. **A pass is kept only if the shipped `E` strictly falls**; otherwise every plane, index, block,
   decoder and grid is restored and the loop stops. Neither range is refitted inside a pass so the comparison is on one
   scale.
4. **`--bc-refine-after N`** re-judges the blocks against the refitted model (default 0; worth +0.00 to +0.06 dB on the eight runs, the most on the run where the pack costs most).
5. **The last refit**: one block (b) and one block (a) against the final blocks, then `E shipped` is measured.

Three further facts fix the pack's place in the loop. **Every stored plane of level 0's chain is packed and refined**,
each plane's blocks against that plane's own sites, all planes concurrently on their own streams
(`src/refine_bc.cu:547-560`), and the `.dds` chain holds exactly those blocks (`src/export.cpp:404-412`): where a
conventional BC mip chain downsamples and then compresses each level, here every level's blocks are chosen against the
decoder's output at that level. **Directly after the first refinement** level 1 (quantised sweeps on its frozen grid)
and the decoder are refitted once against the packed plane, before any outer pass, and no range of either latent is
refitted anywhere after the first pack, so the grid a block was chosen on is the grid that ships. **The two acceptance
tests are on different objectives by design**: a block step is judged on its own plane's quadratic (weighted by `cw`, not by mip), a pass
on the shipped, mip-weighted `E` of the whole chain. The report prints the packing PSNR against the continuous plane beside the output PSNR, and the first moves both
ways: it falls where the refinement wins most (`model35` at `C0 2` 40.55 to 34.02 dB, `model10` at `C0 2` 42.49 to
40.76) and rises a little where it wins least (`model10` at `C0 1` 41.86 to 42.47, `model23` at `C0 1` 42.37 to
42.95), while the output PSNR rises on every run (`docs/RESULTS.md` 6.2): the distance to the latent is a
by-product, not the target.

Measured (`docs/RESULTS.md` 6.2): the refinement and outer passes recover **32 % to 71 % of the pack's cost in `E`**,
+0.11 to +2.97 dB of centre PSNR over the seed pack, at 0.06 to 0.22 s per image. Both outer passes were accepted on
all eight runs; the refinement carries nearly all of the gain at `C0 1`, while on `model35` at `C0 2` what follows it
is worth more than the refinement itself (44.00 dB after it, 45.62 shipped). Against a 4-bit palette at equal
memory, `bc8` wins all eight image-and-layout pairs (`model36` at one channel 32.96 to 40.33 dB, "night and day" by eye; a development measurement with no log in the tree). The remaining gap to the continuous plane is the block format's own constraint.

### 5.4 The mip chains

* **Every plane is a parameter.** No latent mip is ever filtered into existence. The **source** chain is built
  once; plane `m` of both latents is then solved against `src_m` by the same three blocks, every round. Given `W` the planes are independent, so blocks (b) and (c) for every
  plane are issued at once on their own CUDA streams and waited for once; nothing inside a block travels to the host.
* **The 1:1 rule.** Output mip `m` is decoded from mip `m` of both latents, halved together (floor halving, chain stops while level 0 is at or above `--mip-min`, default 8, per axis). On the GPU this is level 1's gradient scale of 4 of section 3.
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
* **Exactly duplicated source channels** (`src/main.cpp:1942-2031`). `box` gives level-1 channel `j` the block mean of
  source channel `j`, so a grayscale FIRST texture, whose G and B are exact copies of its R, would start level 1 with
  two or three identical channels: exactly collinear columns in block (a) and a null direction in block (b) that only
  the ridges and last-bit rounding settle. The split is a rounding lottery, and the decoder amplifies it: the
  arbitrary split sets the decoder's gain on level 0, and the BC5 pack's error is scaled by that gain (1.1 dB apart
  before the pack became 5 dB after it; every ridge, round budget or range policy tried gave a different gap, up to
  8.6 dB), and two builds differing only in fused multiply-add landed 5.03 dB apart on one grayscale photograph
  (`docs/CPU_PORT_LESSONS.md` 9). The rule: take the source channels in order with every exact copy of an earlier
  channel -- equal at every texel of every plane of the chain -- moved behind every distinct one, and recompute level
  1 on the host only when that changes WHICH channels it starts from, so every other input is untouched and
  byte-identical by construction; a lone gray texture has nothing to promote and is unchanged. Measured under `bc8`
  over 68 materials at four layouts on both CUDA builds: gray-first materials gain about 1.3 to 1.5 dB on average and
  the result becomes nearly build-independent, the losses concentrated at `--c1 2`, where two level-1 channels now go
  to two different textures (`docs/DESIGN.md` 3.4); under `--l0 palette` over the same 68 materials, 140 of 147
  triggered runs better, 4 worse by more than 0.1 dB, about +2.1 dB on average, and all 57 untriggered runs
  byte-identical (`docs/CPU_PORT_LESSONS.md` 9). Measured and rejected: replacing the duplicates by a constant, by
  zero, or by fixed-seed noise (stable across builds, +-5 dB image to image), `--init pca` for gray inputs, a larger
  ridge. The general point, for any least-squares encoder: exact duplicates among the inputs of a fit are decided by
  rounding, and a fix triggered only by exact equality can be proven to touch nothing else.
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

### 5.8 The portable CPU backend (`src/cpu/`, `src/backend.h`)

Every device function of the encoder -- 46 of them, the kernels and their drivers -- has a scalar C++17 twin using
`std::thread` and nothing else (no intrinsics, no SIMD), behind a dispatcher (`--backend auto|cuda|cpu|check`, `-j
N`); `auto` uses CUDA where it can and otherwise falls back to the CPU with a warning. The algorithms are unchanged,
which is itself the point: a second implementation is evidence that the method is a method and not one vendor's
arithmetic. Techniques that carry over to any port of an encoder of this kind:

* **The dispatcher proves itself complete.** In a build without CUDA the kernel files are not compiled and their
  symbols do not exist, so a call site that bypasses the dispatcher fails to LINK, naming the symbol; a CPU-only build
  that links is a machine-checked proof that every call is dispatched.
* **Determinism at any thread count.** The number of chunks a kernel is cut into is a function of the problem size
  only, never of the thread count; the chunk-to-element map is fixed; chunks are partitioned statically with no work
  stealing; every chunk writes only its own outputs; partial results are folded in chunk-index order after the join;
  one thread runs the same code path as thirty-two. The four-colour classes of blocks (b), (c) and the BC refinement
  map onto one parallel run per colour, since texels or blocks of one colour share no site. Same build, any `-j`, any
  repeat: byte-identical files, with ThreadSanitizer in the gate reporting nothing (`src/cpu/pool.h`,
  `docs/CPU_PORT_LESSONS.md` 5).
* **No fused multiply-add on either side.** The CUDA kernels are compiled with `-fmad=false` by default
  (`NNTC_CUDA_FMAD`, `CMakeLists.txt`) and the host with `-ffp-contract=off` on gcc and clang (MSVC does not contract
  under `/fp:precise`), so both backends round every multiply and add of one expression the same way;
  `-DNNTC_CUDA_FMAD=ON` reproduces the assets made before the option byte for byte.
* **Replay the device's summation order where, and only where, the sum feeds something ill-conditioned.** The CPU
  folds most reductions in its own chunk order. The exception is the residual seed's covariance (5.5): it feeds an
  eigenproblem whose leading eigenvalues can be nearly equal, and a summation-order difference of 8.35e-9 relative
  forked a five-texture 2048x1024 material at `--c0 4 --c1 4` in round 1 and landed the CPU encode about 0.6 dB below
  CUDA's. `cov_partials` (`src/cpu/cpu_init.cpp:252`) therefore sums in exactly the kernel's order -- 512 blocks of
  256 grid-stride lanes, the shared-memory halving tree, block partials added in index order -- and the difference
  became +0.092 dB (`docs/CPU_PORT_LESSONS.md` 4). The rule generalises: replay the order where a sum feeds an
  eigendecomposition, a near-singular solve or a near-tie decision the rest of the run depends on; elsewhere a chunked
  order is fine.
* **A per-kernel comparison harness** (`--backend check`, `src/backend_check.cpp`). It runs a CUDA encode and, at
  every dispatched call, gives the CPU twin the CUDA arm's exact pre-call state, runs both and compares every output
  by kind: transfers and integers exactly; an index one grid step away as a counted FLIP -- a near-tie a last-bit
  difference may legitimately move -- rather than a failure, unless flips exceed a share of the array; floats against
  bars set by measuring a known-good kernel pair. Device workspaces with no download are read by a probe of the
  harness's own after the CUDA call. Since every call starts from identical inputs, a transcription error shows at the
  call that made it, orders of magnitude outside rounding, where an end-to-end PSNR would only say that something
  somewhere is wrong.

Measured: before the covariance replay, 17 of 20 corpus encodes byte-identical to CUDA's, mean per-texture difference
-0.06 dB, median 0 (`docs/CPU_BACKEND_PLAN.md` 7.4); after it, 19 of 21 byte-identical with the worst texture -0.27 dB
and the mean +0.07 dB, and on further sets 18 of 23 wide-set cases, 23, 24 and 24 of 24 random materials at three
layouts, and 535 of 540 synthetic cases byte-identical (`docs/CPU_PORT_LESSONS.md` 6). The CPU encode takes 8 to 14
times the RTX 5090's time on real inputs with 32 threads (`README.md`).

### 5.9 The encoder's rounding noise floor

The encoder takes thousands of discrete decisions per round -- grid snaps, palette argmins, BC endpoints and
selectors, stop tests -- so it is path-dependent: a last-bit difference upstream flips a few of them and the run
settles in a different optimum of about the same quality. Measured cleanly before any CPU code existed, by compiling
the SAME CUDA source with and without fused multiply-add over a 20-case corpus: 15 cases within about 0.1 dB on every
base texture and 5 well outside it, the worst moving one texture by -1.03 dB and another by +0.86 dB with a mip level
at -2.32 dB, the changes going both ways with identical round counts and ridge tallies; the four-channel and
five-texture materials are the most sensitive (`docs/CPU_BACKEND_PLAN.md` 7.4, `docs/CPU_PORT_LESSONS.md` 2). Two
consequences are part of the method. Two implementations of this encoder are compared on the distribution of
per-texture differences over a corpus -- a bug pulls one way, path dependence scatters both ways -- and never on one
case or on bits. And every discrete choice fed by an ill-conditioned quantity is a place where that noise is
amplified, which is what the duplicate-channel rule of 5.5 and the summation replay of 5.8 remove at their source.

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
* **What "lossless" means, exactly.** At 1-3 bits the BC encode is a fixed relabelling table and no search at all
  (`shared/bc_pack.h:96-131`): 3 bits take `(r0, r1) = (255, 0)` and `sel` = 0 for `k = 7`, 1 for `k = 0`, `8 - k`
  otherwise; 2 bits take the six-value mode's `(85, 170)` and `sel` = 6, 0, 1, 7 for `k = 0..3`; 1 bit takes `(0,
  255)` and `sel` = 6, 7. The per-texel search of block (c) runs over exactly the values those blocks decode to --
  `level0_value` is `-1 + 2 k / (2^bits - 1)` for a BC-shipped plane and `-1 + 2 rep(k) / 255` for an uncompressed one
  (`src/model.h:201-210`) -- so the index the search picks is the selector written and the value it fitted is the
  value a decoder returning the standard palette returns: exact in real arithmetic, and to the float rounding of `k/7`
  in practice (`bc_check`'s round trip on `model10`, largest error 6.5e-06/255, `docs/RESULTS.md` 4). At 1 and 2 bits only endpoint entries and the
  six-value mode's constant 0 and 255 are ever selected, so no interpolation weight enters and the decode is exact on
  EVERY decoder, including one with non-standard interpolation weights; at 3 bits six of the eight entries are
  interpolants and inherit the decoder's precision -- on the NVIDIA part measured with `n/64` weights (below) the
  stored index survives and the values move by the few 255ths recorded there. The grids of the BC and uncompressed
  twins of one solve are identical at 1, 2 and 4 bits and differ by up to half a byte at 3.
* **4 bits, and a mixed `--bits0` list.** Sixteen values do not fit a block's eight entries, so after the loop each
  block takes `lround(k_max * 255/15)` and `lround(k_min * 255/15)` as the eight-value mode's endpoints and every
  texel the nearest entry (a flat block gets `r0 = r1` and selector 0); the plane is then decoded back so that the
  report, the PNGs and `tools/dds_decode.py` describe the file (`src/main.cpp:2719-2741`). Level 1 and the decoder are
  not refitted against it and the blocks are not refined (section 10 item 3). `--bits0` is per channel, so one BC5
  file may hold an exact 1-3-bit channel beside a lossy 4-bit one, each packed by its own rule.
* **The price of hardware sampling.** A BC4 channel costs 4 bits a texel whatever its depth, so the lossless pack
  spends 4 bits on at most 3 bits of index; `--bc0 0` writes the same indices uncompressed (`R8` / `R8G8` /
  `R8G8B8A8`, the index replicated over the byte) at 8 bits a channel. That a 4-bit budget buys a whole BC4 block is
  what `--l0 bc8` exploits (5.3).
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
no container, no runtime library. The asset is fixed-rate by layout, so a noisier plane at equal
quality costs nothing and rate is not a column in any table.

---

## 8. Measurements on record (`docs/RESULTS.md` unless a row names another document)

| what | figure |
|---|---|
| speed vs the evolutionary encoder, ten image-and-layout pairs | 113x to 167x, at -1.75 to +1.72 dB |
| `model34` 2888x4320, `c0 1 bits0 4 c1 2 bits1 8` | 38.59 dB in 5.47 s vs 38.53 dB in 906.6 s |
| `bc8` vs 4-bit palette at equal memory, eight pairs, one binary on both sides | wins all eight, +0.94 to +4.10 dB |
| BC refinement + outer passes, eight runs | 32-71 % of the pack's `E` cost recovered, +0.11 to +2.97 dB |
| frozen grid + quantised sweeps vs round-at-end, 4-bit level 1 | +6.08 and +7.11 dB (development measurement, `docs/DESIGN.md` 5.2, no log in the tree) |
| independent Python reader vs encoder report | equal at every level; worst \|published - sampled\| grid value 1.7e-08 |
| BC pack at 3 bits | lossless (packing PSNR 154.80 dB, max err 6.5e-06/255) |
| determinism | byte-identical `.dds` and `_nntc.json` across reruns (gate) |
| level-1 mip shift, sampler bias against scaled gradients, anisotropic, close steep views against 16x supersampling | bias ahead by 3 to 7 dB (RTX 5090, integrated Radeon); equal square-on or with anisotropy off (`README.md`) |
| Direct3D 11 against Vulkan viewer, anisotropy off, one GPU | byte-identical frames in every `--shot` state (`viewer_vk/README.md`) |
| cooperative-vector decode against the plain shader, 2560x1440, RTX 5090 | 4.0x and 5.3x on the scene draw; max 1/255, 72.52 and 74.62 dB (`viewer_vk/README.md`) |
| CPU backend against CUDA, 21-case corpus | 19 of 21 byte-identical, worst texture -0.27 dB, mean +0.07 dB (`docs/CPU_PORT_LESSONS.md` 6); 8 to 14x the RTX 5090's time on 32 threads (`README.md`) |
| one CUDA source with and without fused multiply-add, 20 cases | 15 within about 0.1 dB, 5 outside, worst -1.03 / +0.86 dB on a texture (`docs/CPU_BACKEND_PLAN.md` 7.4) |
| exact-duplicate channel rule, gray-first materials | about +1.3 to +1.5 dB mean under `bc8` (`docs/DESIGN.md` 3.4); palette mode 140 of 147 better, about +2.1 dB (`docs/CPU_PORT_LESSONS.md` 9) |
| block (a) ridge ladder, one-white-pixel image | 60.46 dB against 53.69 dB without it (`docs/MATHEMATICS.md` 3) |

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
what `--weights` is for. The decoded PNGs of every texture at every level are written beside the asset by the run itself under `--png 1`, and
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
* **Simon Fenney, "Texture compression using low-frequency signal modulation" (Graphics Hardware 2003, the basis of
  PVRTC)** stores two low-resolution colour images A and B -- one texel of each per 4x4 block in the 4 bpp mode -- and
  a full-resolution 2-bit modulation value per texel; a texel is A and B upscaled (bilinearly in the format the paper specifies; its initial test system used bicubic)
  and blended by its modulation value, and the compressor alternates between choosing the quantised modulation values
  and a least-squares solve of A and B over overlapping windows. It is the closest classical relative of NNTC's
  structure: a full-resolution, low-precision plane modulating a quarter-resolution one, fitted by alternating
  discrete choices and least squares. NNTC differs in that the combination is a decoder fitted per material rather
  than a fixed blend of two colours, one representation yields several textures, both planes are ordinary textures
  that the hardware filters BEFORE the decode (PVRTC decodes texels, which are then filtered), the fit is made at
  fractional positions and on every mip level, and the planes ship as standard BC4 / BC5 and 8-bit textures rather
  than a format of their own.
* **Hollemeersch, Pieters, Lambert and Van de Walle, "A new approach to combine texture compression and filtering"
  (The Visual Computer 28, 2012)** describe, in their abstract, a system based on linear transforms that merges the
  decompression and filtering phases, formalised for any linear transform, adapted to the DCT and implemented in
  shaders. NNTC shares the principle that a linear stage lets filtering move ahead of decoding, but its stored
  quantities are fitted latents rather than transform coefficients, the filter is the texture unit's own bilinear,
  trilinear or anisotropic rule applied to ordinary textures, and its decoder is not linear in the two samples: the product terms are
  exactly where filtering before decoding stops being exact (2.4), which NNTC handles by fitting under the filter.
* **Mavridis and Papaioannou, "Texture compression using wavelet decomposition" (Computer Graphics Forum 31(7),
  Pacific Graphics 2012)** store the coefficients of a modified Haar transform in standard DXT5 / BC7 blocks, with
  chroma subsampled in YCoCg-R, and invert the transform in a shader. Because each stored texel carries a 2x2 group of
  coefficients, the hardware cannot filter the stored textures directly and the paper filters in the shader after
  decoding. NNTC uses standard block formats as a store in the same spirit, but its latents are fitted so that the
  hardware's own filter of the stored texels, followed by the decode, is the intended result.
* **Vaidyanathan, Salvi, Wronski, Akenine-Moller, Ebelin and Lefohn, "Random-access neural compression of material
  textures" (ACM Transactions on Graphics 42(4), 2023)** compress a material's textures and mip chains together into a
  pyramid of quantised feature levels, each with a high- and a low-resolution grid, decoded per texel by an MLP of two
  hidden layers of 64 fed by learned interpolation of four neighbouring features of the high-resolution grid, a
  bilinear sample of the low-resolution one, a positional encoding tiled every 8x8 texels and a LOD value; filtering
  follows the decode (the paper implements software bilinear and trilinear filtering by decoding four or eight texels, and proposes
  stochastic filtering), the network being fitted only at discrete texel positions and mip levels. NNTC shares the per-material,
  multi-texture decoder and the joint fit of the mip chain, and differs in decoding hardware-filtered latents with no
  position input (2.4; section 10 item 15 says why), in a one-layer affine decoder fitted by exact block solves, and
  in shipping standard BC4 / BC5 and 8-bit textures that the texture unit samples.
* **Weinreich, de Oliveira, Houdard and Nader, "Real-time neural materials using block-compressed features" (Computer
  Graphics Forum 43, Eurographics 2024)** store a material's learned features as four mipmapped BC6 textures,
  emulating BC6 decompression during training so the features export as ordinary BC6 textures; at runtime each feature
  texture is sampled with the hardware's trilinear filter at a scale computed from the UV derivatives plus a bias from
  its resolution, and a small MLP (12 inputs, one hidden layer of 16) decodes the filtered features in the shader,
  trained at random positions and continuous scales against a filtered reference so that nothing is filtered after the
  decode. It is the closest prior art for decoding hardware-filtered, block-compressed latents, including per-texture
  LOD offsets for latents of different resolutions. NNTC differs in the decoder (one affine layer over two latents and
  their products, no hidden layer, no activation), in the fit (alternating exact block solves at a fixed,
  deterministic site set, no gradient descent and no random sampling), in the block format (BC4 / BC5 blocks chosen
  after a continuous solve by an explicit endpoint least squares and exact selector argmin against the decoder's
  error, 5.3, rather than BC6 parameters trained through an emulated decoder), in the layout (a full-resolution latent
  of 1-4 channels beside a quarter-resolution one), in 8-bit channels where BC6H is a half-float format, and in the
  exact trilinear statement of 2.4.
* **Belcour and Benyoub, "Hardware accelerated neural block texture compression with cooperative vectors"
  (High-Performance Graphics 2025)** extend Weinreich et al. with four BC1 latent textures at two to four resolutions
  (two of them shifted by half a texel), trained through a replicated BC1 texture unit with quantisation-aware
  optimisation, a one-hidden-layer MLP of 16 to 64, and hardware anisotropic filtering of the latents -- observed to
  work although training used isotropic filtering -- with the MLP evaluated through cooperative vectors in a classification pass that sorts 8x4 tiles by MLP, feeding tile-based G-buffer and lighting passes; they report 0.55 ms at 1080p on an Intel B580. NNTC
  differs in the decoder and the fit (above), in BC4 / BC5, whose channels are independent 8-bit ramps, rather than
  BC1's 5:6:5 endpoints and 2-bit selectors, in fitting a fixed, deterministic site set that covers every mip in every solve, rather than one randomly
  sampled LOD per gradient step, and in stating the trilinear error in closed form (2.4), and in its use of cooperative vectors: one zero-padded 18x24 matrix-vector instruction per
  pixel in a plain fragment shader, with no tiling or classification pass (3.2).
* **Fujieda and Harada, "Neural texture block compression" (Eurographics Workshop on Material Appearance Modeling 2024,
  arXiv:2407.09543)** learn a mapping from
  uncompressed textures to block-compressed ones to cut storage, the blocks being produced at the texture loading
  phase. That is the load-time-transcoding direction of section 10 item 11; NNTC's shipped planes are already BC
  blocks that a sampler reads directly.

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
   BC6H for a wider-range latent, ASTC on mobile, or two BC4s vs one BC5 with a shared selector plane. Any format the
   texture unit decodes per texel before its blend keeps the sampling argument, whatever that decode is (2.2); what
   the endpoint step of 5.3 needs in addition is a decoded value affine in the endpoints at fixed selectors, which
   makes it a small least squares, and the block quadratic of 5.3 then applies with the format's own endpoint and
   selector structure.
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
13. **A Vulkan / D3D12 compute encoder** for machines without CUDA: every block is a kernel and the algorithms
    carry over unchanged. The CPU half of this item IS BUILT (5.8); an encoder on any vendor's GPU is not.
14. **Rate-aware layouts**: an outer search over `C0`, `C1` and `bits1` per material against a bpp budget, since the
    encoder is fast enough to try several layouts and every asset's size is known in advance.
15. **Which decoder inputs keep the representation valid under the sampling operator.** The inputs today are two
    hardware samples and their products. This item states which other inputs keep that property and which break it;
    the parts marked BUILT are the current design. Write `y_k[t]` for latent `k`'s stored texels after the texture
    unit's per-texel decode and `S_k(u) = sum_t omega_t y_k[t]` for the sampler's blend (`omega >= 0`, `sum omega =
    1`, over the bilinear taps, two mips or an anisotropic line of taps), and keep three properties apart:

    * **(V) valid**: the shader evaluates exactly the function the encoder fitted at every site. It holds when every
      decoder input is a hardware sample of stored texels, a fixed linear combination of such samples, or a per-pixel
      constant, and the encoder models the same `omega` at its sites;
    * **(C) commuting**: `f(sum omega y) = sum omega f(y)` for every blend, so the decode of a filtered sample is the
      filter of the decodes. It holds iff the decoder is affine in the filtered inputs;
    * **(X) exactly solvable**: the objective is an exact quadratic in each latent with the others held, which the
      block solves of 5.2 need. It holds when the decoder is multilinear across latents -- no monomial contains one
      latent twice -- and its output is affine in the features.

    The inputs, one by one:

    a. **Dequantisation after the blend** (BUILT, 2.2): V, C, X. A per-channel map applied by the shader after the
       blend that is not affine (a non-uniform palette lookup, a gamma) keeps V only if the encoder models it, and
       breaks C and X.
    b. **A per-texel non-linearity applied by the texture unit before the blend** -- an `_SRGB` format
       (`R8G8B8A8_UNORM_SRGB` for level 1's colour channels; BC4 / BC5 have no sRGB form), a 16-bit float format, any block format: V, C and X all hold, because the decoded texel is the unknown and the
       blend is linear in it. A shared-exponent format decodes per texel too, so V and C hold, but it couples a
       texel's channels through the exponent, so its grid is not a per-channel monotone one and the per-channel
       sweeps would need a joint step. BUILT for BC4 / BC5, NOT BUILT otherwise. What it buys is a non-uniform quantisation
       grid at no shader cost (an sRGB byte gives finer steps near 0); the quantised sweeps already take the grid
       value nearest the parabola's vertex on any monotone grid (5.2). The same transform applied offline by the
       encoder buys nothing -- it only renames the unknown -- and an inverse applied by the shader AFTER the blend
       turns the blend into a generalised mean (a stored logarithm gives a geometric-mean interpolation), keeping V only if the encoder models it and
       breaking C and X.
    c. **More latent textures at other resolutions** -- a third plane at a sixteenth of the extent, a second
       full-resolution plane, one at half (NOT BUILT). V holds when every resolution is a power-of-two ratio `2^k` of
       level 0 read with its gradients scaled by `2^k` (or, on hardware that keeps a negative base LOD under magnification, an integer sampler LOD bias), so that every texture reads the same
       mip index with the same trilinear fraction; a non-power-of-two ratio gives each texture its own fraction and
       breaks the 1:1 pairing. X holds when the new plane enters through its own affine terms and products with the
       other planes, triple products included. C holds for the affine terms and each new product adds its own
       footprint covariance (2.4). It buys capacity past `C1 <= 4` and another scale, for one more sample per pixel
       and a larger `nin`.
    d. **The same latent read again at a coarser LOD** (NOT BUILT). Level 0 sampled a second time with its gradients
       scaled by `2^k` returns a sample of its own stored mip `m + k`: a second, blurred scale at no memory cost. V
       and C hold. X holds per plane, but plane `m + k` becomes both the latent of output level `m + k` and an input
       of level `m`, so the planes stop being independent given `W` (5.4) and their block solves couple; the last `k`
       planes of the chain read the clamped deepest mip.
    e. **Offset taps and fixed linear combinations of samples** -- `sum_j a_j S(u + delta_j)`, a difference of taps as
       the gradient of the filtered latent at the footprint's scale (NOT BUILT). C and X hold, every such input being
       linear in the stored texels. V needs `delta` in texels of the mip actually read (`delta_uv = d 2^lambda / W`,
       `lambda` from `CalculateLevelOfDetail` / `textureQueryLod`); a fixed offset in texture coordinates is a
       different feature at every mip, served by one shared `W`. The stencils of blocks (b) and (c') widen with the
       reach of the taps: offsets of one texel make a site's taps span four texels per axis, so an independent
       colouring needs a period of four per axis, sixteen colours instead of four. A non-linear function of the taps
       (a gradient's magnitude) breaks C and X. `Gather` returns unfiltered texels with no mip selection and no
       anisotropy, so an input built from it is a texel codec and not this representation.
    f. **Screen-space derivatives of samples** (`ddx` / `ddy` of a filtered latent; NOT BUILT, not recommended):
       linear in the stored texels but a function of the view, so the decoded material would change with the camera
       beyond what filtering does; dividing by the UV Jacobian recovers a texture-space gradient only at the
       resolution of a 2x2 pixel quad.
    g. **The level of detail itself as an input** (NOT BUILT). `lambda` is one number per pixel, constant over its
       footprint, so it raises no filtering question. A decoder `out = (1 - t) W_m phi + t W_(m+1) phi` using the
       hardware's own `t`, or `out = W_0 phi + lambda W_1 phi`, is linear in `phi` at every `lambda`: V, C and X hold,
       block (a) grows to the features `phi (x) [1 ; lambda]` or to one `W` per plane, and trilinear sites fit the
       fractional levels. It is the seamless runtime form of the per-mip decoders of item 12, and it can absorb the
       systematic part of the product terms' covariance, which grows with the footprint.
    h. **Position or phase encodings** evaluated in the shader, such as a sawtooth or triangle wave of `u W_m` (NOT
       BUILT, deliberately). They keep V at the fitted sites only. Under minification and anisotropy the latents are
       averages over the footprint while the phase is evaluated at one point, the sawtooth's jump at a texel edge is
       filtered by nothing, and the two mips of a trilinear blend disagree on the phase; C fails, and a decoder that
       uses them has to be evaluated per texel and filtered afterwards -- by hand or stochastically -- which gives up
       the hardware filter. The full-resolution level 0 already carries the per-texel information such a code would
       have to synthesise. A filter-compatible form of the same idea (NOT BUILT, not measured): a fixed,
       asset-independent periodic pattern texture -- a hat or a checker of period two to four texels at level 1's
       resolution, with its own box-filtered chain -- sampled with the same sampler and gradients as the latents. It
       is a latent frozen to a pattern, so V, C and X hold, and under minification it averages toward its mean, so the
       positional term fades instead of aliasing.
    i. **Products of samples** (BUILT, the `s_i c_j` terms): V and X hold; C holds for the affine part only, the
       products being products of filtered values and not filtered products, with the covariance and the trilinear
       term of 2.4. Products within ONE latent (`s_i s_j`, `c_j^2`; NOT BUILT) would keep V where fitted, but break X
       (the objective becomes quartic in that latent) and break C with a one-signed error: the square of a blend falls
       short of the blend of squares by the footprint's variance, so the decode drifts with distance in one direction,
       as a filtered normal shortens. An activation on the output likewise keeps V and breaks C and X
       (`docs/DESIGN.md` 2).
    j. **Minimum / maximum reduction samplers** (the Direct3D minimum / maximum filters,
       `VK_EXT_sampler_filter_minmax`; NOT BUILT) return an order statistic of the footprint: V only if the encoder
       models it, C and X broken.
    k. **A point-sampled latent** (NOT BUILT) is a linear functional at each position but is not averaged under
       minification, so it aliases, and it breaks the contract that the consumer chooses the filter: both latents are
       read with the consumer's filter.
16. **Supercompressed palette-mode index planes.** At 1-3 bits the BC4 / BC5 encode of a palette-mode level 0 is a
    fixed relabelling table with constant endpoints (section 6), so the plane can travel as its raw index stream under
    any lossless entropy coder and become valid BC4 / BC5 blocks at load, bit for bit the blocks the encoder would
    have written, for one table lookup per texel and with nothing decoded or re-encoded. Neither the coder nor the
    load-time transcoder is built.
17. **An 8-bit decoder for integer matrix engines.** The cooperative-vector extension also enumerates 8-bit integer
    input tuples with 32-bit integer results (`viewer_vk/README.md`); `W` quantised per output row to 8 bits with a
    row scale, and `phi` to 8 bits on a fixed range, would decode through those or through `DP4a`. It is a format
    change -- the published `W` is fp32 -- so the grid would be chosen against the objective, as the latent grids are,
    and not rounded afterwards. Not built, not measured.
18. **A hand-optimised plain decode path**: the fixed 18x24 product unrolled, `W` in a storage buffer, `phi` built
    without a running index. The 4 to 5x of 3.2 is against the plain loop the viewers ship, which cannot unroll; how
    much of it an optimised loop takes back is not measured.
