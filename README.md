# NNTC: Not Neural Texture Compression

The `nntc_encode` tool (for Windows or Linux) compresses a 24-bpp texture, or a PBR material of up to six 24-bpp textures, into **two mipmapped latent planes and a few dozen
fitted matrix coefficients**. It outputs two `.dds` textures, or three when the full-resolution latent plane has three or four channels. A pixel shader decodes
the whole material back from **two ordinary `Sample()` calls**, or three when the full-resolution plane has three or
four channels, and one matrix multiply of a few dozen to a few hundred multiply-accumulates per pixel for the whole
material (the count is in [How it works](#how-it-works-briefly), below). No runtime library, no decompression at load, no neural inference: the files are standard `.dds`
(BC4 / BC5 and 8-bit) and a `.json` containing the fitted bilinear model's decoder coefficients.

The `nntc_view` is a small Windows D3D11 viewer that loads the .json and .dds files and displays the sampled and decoded results on a quad or a cubemap.

Importantly, this method is compatible with normal GPU texture hardware bilinear, trilinear, and anisotropic filtering. The encoder ensures the encoded latents and the fitted coefficients are compatible with hardware filtering. The end result: large runtime memory savings due to packing 2-6 correlated material textures into a single set of latent textures, which are sampled normally and then decoded by a small pixel shader function.

A Prior Art Disclosure, dated September 13, 2026 is [here](https://github.com/richgel999/nntc/blob/main/PRIOR_ART_DISCLOSURE.md).

## What it does

* **Input**: one to six 24-bit RGB images of the same size (PNG, JPEG, TGA, BMP and the other formats `stb_image`
  reads), named on the command line or in a material JSON. An alpha channel is ignored, with a warning: the tool
  encodes RGB only. Sizes should be divisible by 4; others are padded with a warning.
* **Default layout**: `--c0 2 --c1 4`: latent 0 has two channels at full resolution (one BC5 texture), latent 1 has
  four channels at a quarter resolution (an 8-bit RGBA texture); 10 bits per pixel at the base for the whole material,
  13.3 with mipmaps. Latent 1 is always stored at 8 bits per channel in the `.dds`.
* **Output**: `NAME_lat0.dds` (a full-resolution BC4 / BC5 texture; two files, `NAME_lat0a.dds` and
  `NAME_lat0b.dds`, at 3 or 4 channels), `NAME_lat1.dds` (a quarter-resolution 8-bit texture) and `NAME_nntc.json`
  (the sizes, the per-channel ranges of both levels and the decoder's weights). Every texture carries its mipmaps;
  `--mips 0` writes one level each.
* **The output descriptor** is `NAME_nntc.json`, beside `NAME_lat0.dds` and `NAME_lat1.dds`. The `_nntc` suffix is there
  so that a name taken from an input can never be the input; `-o` decides where they land (see **Encoding**).
* **Output size in bits**: the default layout (two channels in latent 0, four in latent 1) costs 10 bits per pixel at the base mipmap level, or 13.3 bpp
  with mipmaps, **for the whole material, whatever its texture count**. Every texture shares the same files, so the
  per-texture figure with mipmaps is that 13.3 divided by however many there are: 3.3 bits per pixel per texture at
  four, 2.2 at the cap of six.
* **Encode Speed**: about one second to about a minute on an RTX 5090, growing with the pixel count, the texture count and
  the layout, with the top of that range as described below. Measured:
  1.3 s for four 512x512 textures at the default layout (`out/disclosure/log_m1234_default.txt`), 5.47 s for one
  2888x4320 image at `--l0 palette --c0 1 --bits0 4 --c1 2` (`out/s7/cmp/log_model34_c1b4_c2b8.txt`), 15.2 s for four
  2048x1024 textures at `--c0 4 --c1 4` (`out/timing/log_md1234_c4c4.txt`). Larger inputs take longer: four 4096x4096
  textures run on the order of a minute (seen during development). The encoder tool currently only supports CUDA; without
  an NVIDIA GPU it does not encode, however the assets it writes decodes on any GPU (using plain pixel shaders).
* **Encode Quality**: The encoder fits the output for the hardware filter, not for texel by texel decoding, so the method is compatible with standard bilinear, trilinear and anisotropic sampling. `docs/RESULTS.md` has the measurements.

## Building

You need CMake, a C++17 compiler and the CUDA toolkit. CUDA 12.8 or newer is the minimum the build accepts; 13.4
(VS 2026) and 13.3 (WSL) are what this tree is built and gated with.

Windows, the toolchain this tree is developed on -- **Visual Studio 2026 with CUDA 13.4** (CMake 4.2 or
newer; the CMake bundled with VS 2026 is fine):

```
cmake -B build -S . -G "Visual Studio 18 2026" -T cuda=13.4
cmake --build build --config Release
```

That produces `build\Release\nntc_encode.exe`, `nntc_view.exe` and `bc_check.exe`.

Visual Studio 2022 with CUDA 13.1 is also supported and produces byte-identical assets:

```
cmake -B build -S . -G "Visual Studio 17 2022" -T cuda=13.1
cmake --build build --config Release
```

Linux:

```
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

That produces `nntc_encode` only; the viewer is Direct3D 11 and Windows only. Verified under WSL 2 (Ubuntu 24.04,
gcc 13.3, CUDA 13.3 from NVIDIA's `wsl-ubuntu` apt repository): the gate passes and the asset it writes is
byte-identical to the Windows build's -- on the default source chain. 

Notes: A run whose mipmaps come from `stb_image_resize2` (a named filter, or the `box` implied by `srgb`, `edge` or
`normal_map`) can differ
in the last bit between platforms, so those runs agree across platforms in PSNR rather than byte for byte
(`docs/DESIGN.md` section 6 has the measurement). Inside WSL, prefer the `cuda-toolkit-13-x` package: the `cuda` and
`cuda-drivers` metapackages try to install a Linux driver, which WSL does not use (the driver is Windows'). Put
`/usr/local/cuda-13.x/bin` on your PATH yourself.

## Encoding

```
nntc_encode albedo.png                          -> albedo_lat0.dds, albedo_lat1.dds, albedo_nntc.json in the current directory
nntc_encode albedo.png normal.png rough.png     -> a 3-texture material, named after the first input
nntc_encode a.png b.png -o out/                 -> the same files under out/
nntc_encode a.png b.png --weights 2,1           -> the first texture matters twice as much
nntc_encode a.png --c0 1 --c1 2                 -> a smaller layout: 4 + 1 = 5 bits per pixel at the base
nntc_encode rock_material.json -o out/          -> the same, with the inputs and their settings named in a file
nntc_encode a.png -o out/name.json              -> the descriptor is exactly out/name.json, the .dds out/name_lat0.dds
nntc_encode me1.png me2.png me3.png m4.png --l0 3        -> use 3 channels on latent 0 (harder material)
nntc_encode me1.png me2.png me3.png m4.png m5.png --l0 4 -> use 4 channels on latent 0 (even harder material)
```

The tree also carries two small source-material examples under `examples/`, with their source PNGs and precompressed
assets. To view the included assets directly:

```bat
build\Release\nntc_view.exe examples\m1_m4_c0_3_c1_4_nntc.json
build\Release\nntc_view.exe examples\pavingstones141_1k_c0_4_nntc.json
```

To recompress those examples from the source JSONs:

```bat
build\Release\nntc_encode.exe examples\m1_m4_source_material.json -o out\examples\m1_m4_c0_3_c1_4.json --c0 3 --c1 4 --png 0
build\Release\nntc_encode.exe examples\pavingstones141_1k_source_material.json -o out\examples\pavingstones141_1k_c0_4.json --c0 4 --png 0
```

and then view those freshly written assets:

```bat
build\Release\nntc_view.exe out\examples\m1_m4_c0_3_c1_4.json
build\Release\nntc_view.exe out\examples\pavingstones141_1k_c0_4.json
```

`-o` names where the asset lands, and what it means depends on the spelling:

| `-o` | what it names |
|---|---|
| `out/` (a trailing separator) | a **directory**, created if it does not exist; the asset takes the input's own base name |
| `out/name` (no extension) | also a **directory**, unless a file of that name already exists |
| `out/name.ext` (any other extension) | the **prefix** itself: `out/name.ext_lat0.dds`, `out/name.ext_nntc.json` |
| `out/name.json` | **exactly that descriptor**, with `out/name_lat0.dds` beside it |

`out/` is created if it does not exist; `--no-mkdir` refuses instead, naming the directory, before anything is read
or written.

The defaults are the recommended settings, and a bare call needs no flags at all. The
two knobs worth knowing:

| flag | default | what |
|---|---|---|
| `--c0 N` | 2 | channels in the full-resolution texture, 1-4. Each costs 4 bits per pixel (BC4 / BC5) |
| `--c1 N` | 4 | channels in the quarter-resolution texture, 1-4. Each costs 0.5 bits per pixel |

`--weights` and `--mip-filter` are **per-texture lists**: give exactly one entry per input texture, or none at all and
every texture takes the default. There is no single-value broadcast (`--bits0` has one, because it is per level-0
*channel* and how many there are is `--c0`'s business).

`--help` lists the everyday options, `--help-advanced` the solver's. `--diag` prints a per-texture, per-mip
diagnosis table before the report, and with it the range of every source level, which is where the chain's clamp is
visible. `--quiet` prints nothing but the warnings and the errors: no banner, no progress, no `wrote` lines and no
report; the asset and its descriptor are the account of what the run produced. `--png 1` (the default) writes both the decoded and the
source PNGs per texture per mip level (`_recon_` and `_src_`; with two or more textures the names carry `_t<n>_`);
`--png 0` skips them.

`nntc_encode` returns 0 on success and 1 on any error.

### The source mip chain, and the material JSON

The encoder makes the source mipmaps itself: it takes each input image and builds its mipmap chain with either the
built-in 2x2 box filter (the default) or `stb_image_resize2`, and those mipmaps are the ground truth the deeper levels
are fitted against. The choice is **per texture**, because an albedo wants filtering in linear light, a tangent-space
normal map wants renormalising afterwards, and a packed mask wants neither. `--mip-filter default,mitchell,..` picks
the filter (`default` is the built-in 2x2 box; any other name, `box`, `mitchell` or `catmullrom`, uses
`stb_image_resize2` directly from the base level); everything else needs a material JSON, which also spares you a
command line with four paths on it:

```json
[
  { "file": "albedo.png", "type": "albedo", "filter": "mitchell",   "srgb": true,       "edge": "wrap",  "weight": 2,   "rgb_weights": [1, 1, 0.5] },
  { "file": "normal.png", "type": "normal", "filter": "catmullrom", "normal_map": true, "edge": "clamp" },
  { "file": "rough.png",  "type": "mask",   "filter": "box",        "weight": 0.5 }
]
```

`nntc_encode rock_material.json -o out/` then writes `out/rock_material_lat0.dds`, `out/rock_material_lat1.dds` and
`out/rock_material_nntc.json` -- the asset takes the JSON's own name, and the descriptor's `_nntc` suffix is what keeps
it from being written over the material itself.

This is the input format, so here it is in full. Every key is optional but `file`, and an unknown one is refused:

| key | default | what |
|---|---|---|
| `file` | *required* | the image, relative to the JSON |
| `type` | `""` | a free label, echoed into the asset's JSON and never interpreted |
| `filter` | `default` | the source chain's filter: `default` (the iterated 2x2 box), `box`, `mitchell`, `catmullrom` |
| `srgb` | `false` | filter in linear light, converting in and back out |
| `edge` | `clamp` | the filter's edge mode: `clamp` or `wrap` |
| `normal_map` | `false` | renormalise each filtered texel as a tangent-space normal |
| `weight` | `1` | this texture's weight in the objective |
| `rgb_weights` | `[1, 1, 1]` | the weights of its three channels, normalised to mean 1 within the texture |

The built-in 2x2 box filter does plain averaging and nothing else: no sRGB-correct filtering, no wrap edge, no normal
map renormalisation. Those need `stb_image_resize2`, which any filter name other than `default` turns on. So if a
texture asks for `srgb`, `edge: wrap` or `normal_map` and names no filter, the encoder sets its filter to `box` for it
(the settings row says `filter box (implied by srgb)`), and if it names `mitchell` or `catmullrom` that filter is used.
Naming `default` explicitly together with one of those keys is a contradiction and is refused. `normal_map` with
`srgb` is refused too, since a normal is not a colour. The command line overrides a JSON key and prints a warning
naming the value it replaced; `--weights` and `--mip-filter` take exactly one entry per texture, or none at all. A bare
command line stays the vanilla path: the built-in box chain, weights 1, no JSON needed.

## Viewing

```
build\Release\nntc_view.exe out\albedo_nntc.json
```

The viewer takes the asset's descriptor (`PREFIX_nntc.json`) and reads the `.dds` files it names beside it. It binds
the asset's `.dds` files (two, or three when level 0 takes two), samples each once per pixel and runs the decoder in
the pixel shader. Keys:

| key | what |
|---|---|
| arrows, `W`/`S`, `A`/`D`, `Q`/`E` | move, zoom, yaw, pitch; `Space` resets, `Esc` quits |
| `N` | next texture of the material |
| `P` / `B` / `T` | point, bilinear or trilinear filtering; `X` anisotropic on/off (trilinear only) |
| `M` | mips on/off |
| `L` | level 1's LOD bias on/off (*off is visibly wrong from mip 1 down*; that is what it is for) |
| `V` | show the texture renormalised as a tangent-space normal |
| `1` / `2` | the raw level-0 or level-1 texture instead of the decode |
| `C` | quad or cube |

`--shot FILE.bmp` renders one frame and exits. `viewer/README.md` has the full list and the shader.

**If you write your own consumer**, the one thing to get right: **the quarter-resolution texture's sampler needs
`MipLODBias = 2`** (the JSON's `lod_bias_level1`), so that both textures are read from the same mip. Everything else is
`lo + sample * (hi - lo)` on each channel, then `out = W * [c, s, s*c] + b`. `docs/FORMAT.md` specifies the files byte
by byte, and `tools/dds_decode.py` is a reference decoder of about 500 lines in Python.

## Checking an asset

```
python tools\dds_decode.py out\albedo --ref out\albedo_recon    independent decode, max |diff| against the encoder's PNGs
python tools\dds_decode.py out\albedo --psnr-levels             the encoder's psnr and mip psnr lines, reproduced (needs the run's _src_ PNGs, so --png 1)
python tools\dds_decode.py out\albedo --grid                    the published ranges match what a sampler returns
build\Release\bc_check.exe out\albedo_nntc.json                 the BC blocks against iOrange's bcdec
python tests\run_checks.py                                      the whole release gate
```

`--ref` reports the max |diff| per level and no PSNR; `--psnr-levels` is the command that reproduces the encoder's
`psnr` and `mip psnr` lines. The checkers take different argument shapes: `dds_decode` takes the asset's PREFIX (or,
since it is the name you have in front of you, the descriptor path, which it strips back to the prefix), while
`bc_check` and `nntc_view` take the descriptor itself.

`run_checks.py` writes about 200 MB of assets under `out/`, which is gitignored except the tracked logs.

Two runs of one command on one image produce byte-identical files.

## How it works, briefly

Level 0 (full resolution, 1-4 channels) and level 1 (quarter resolution, 1-4 channels) are both fed through the
hardware filter and dequantised by an affine range, so the hardware's blend of stored texels is the blend of the
values. The decoder is one affine map over the level-1 channels, the level-0 channels and their products. The encoder
alternates three exact solves -- the decoder by least squares, level 1 by a sparse quadratic on a 9-point stencil,
level 0 by the same solve and then a per-block BC4 / BC5 search against the decoder's own output -- and the objective
is measured at fractional positions where both textures are blended, not only at texel centres.

The runtime deliberately samples the latent textures first and decodes afterwards. That keeps the asset on the normal
GPU texture path: the two latent `.dds` files still get the texture units' address modes, bilinear, trilinear,
anisotropic filtering, mip selection and BC decode. Decoding texels first and then filtering the decoded material would
need a custom gather/filter path or a decompressed cache; NNTC instead binds ordinary textures, calls `Sample()` on them
and runs the small decoder on the filtered latent values.

This works well enough visually because the parts of the decode that dominate are affine in the sampled latents. The
dequantisation is affine and happens after `Sample()`, so dequantising a hardware-filtered latent sample is the same as
filtering the dequantised latent values. The decoder's linear terms commute with the filter for the same reason. The one
non-affine piece is the product term: filtering first gives `avg(s) * avg(c)`, while filtering a decoded image would
contain `avg(s * c)`. The difference is the local covariance of the full-resolution selector plane and the
quarter-resolution colour plane inside the filter footprint. On normal material data that covariance is usually small:
the colour plane is smooth at level 0's scale, the selector plane carries the high-frequency choice/detail, and the
textures of one surface tend to share the same edges and regions.

The encoder does not leave that to luck. It fits fixed fractional subtexel sites as well as texel centres, with both
latents and the source sampled by the same bilinear clamp rule the GPU will use. It also fits the stored mip chain, and
the format requires level 1's sampler to carry `MipLODBias = log2(block)` (`lod_bias_level1`, 2 for the fixed block of
4), so output mip `m` reads mip `m` of both latent textures. The optimisation target is therefore the real runtime
operation -- filter the latents first, then decode -- rather than an ideal texel-grid reconstruction that the shader
would never actually evaluate.

**Decode cost.** Per pixel, for the whole material: the samples, the `C0 x C1` products that build the input vector,
and the matrix multiply, `nin x nout` multiply-accumulates. `nin = C1 + C0 + C0 C1` is 14 at the default layout and at
most 24 (`--c0 4 --c1 4`); `nout` is 3 per texture, at most 18. So the multiply is 42 multiply-accumulates for one
texture at the default layout, 15 at the smallest layout (`--c0 1 --c1 2`), and 432 for six textures at the largest.

**Why a material fits in so few channels.** The textures of a PBR material describe one surface, each a different
property of the same geometry and the same material, so their channels are highly correlated: the albedo, the normal map, the roughness, the occlusion and the masks typically follow
the same edges, the same grain and the same regions (not always, but usually), and many of them are greyscale to begin with. A material of `T`
textures has `3T` output channels but far fewer independent ones. The representation is a low-rank model of exactly
that: the quarter-resolution plane holds a few shared channels `c` that say which kind of surface a region is, the
full-resolution plane holds one to four channels `s` that say where within it a texel sits, and the decoder is
`out = W [c ; s ; s c] + b`, so the products let `s` choose, per texel, which linear combination of the shared channels
each output takes. Every output channel is one row of `W` applied to the same per-texel feature vector
`phi = [c ; s ; s c]`, `C1 + C0 + C0 C1` numbers (14 at the default layout): the whole material is a linear function of
one short vector per texel, and the products are what let that function differ from texel to texel.
That is why the layout does not grow with the texture count and why the m1-m4 material (twelve output channels)
ships from two channels at full resolution plus four at a quarter, 10 bits per pixel for the set. What it costs is
exactly what a low-rank model costs: a texture whose detail is unlike every other's has to share the same few
channels, and `--diag` shows which one is starving. A material whose textures are **decorrelated** -- unrelated
pictures, or several independent colour textures -- needs more level-0 or level-1 channels (`--c0`, `--c1`); with too
few there can be visible crosstalk between the material's textures, one texture's detail showing faintly in another,
and `--diag` is how to see which texture is short.

**Related work.** The idea that a texture set is low-rank and can be stored as a few shared planes plus per-texture
coefficients is Bart Wronski's [Dimensionality reduction for image and texture set
compression](https://bartwronski.com/2020/05/21/dimensionality-reduction-for-image-and-texture-set-compression/)
(2020); NNTC adds the full-resolution selector plane, the products, the exact solves and the fit under the hardware
filter. The luma-plus-chroma split of a texture into a full-resolution plane and a low-resolution one, packed in a
block format with mipmaps, goes back to the author's [Experiments in Luma-Optimized and Mipmapped DXT1
Compression](https://web.archive.org/web/20201024153426/https://sites.google.com/site/richgel99/luma_chroma_texture_compression) (2012).

`docs/DESIGN.md` is
the method, `docs/MATHEMATICS.md` the notes for whoever changes it, `PRIOR_ART_DISCLOSURE.md` the public description.

## Limits

* RGB only; an image's alpha channel is ignored, with a warning naming the file.
* At most six textures per material. The layout does not grow with the count: six textures share the same two latent
  planes (two or three `.dds` textures) as four do, so more textures means fewer bits per texture.
* Encoding needs an NVIDIA GPU (compute capability 8.0 or newer). Decoding does not.
* No entropy coding: an asset's size is fixed by its layout.
* The encoder never deletes a file or a directory. A run leaves any stale sibling under its prefix alone; the
  descriptor names the files that belong to the asset.
* The less correlated the inputs, the more channels needed to avoid crosstalk between textures. Conversely, the lower the dimensionality (i.e. the more correlated) the input textures are, the better this technique works, and the less channels you'll need to avoid crosstalk.
* Only up to 4 channels are supported on the 2nd (1/4 resolution) latent plane. This may be too limiting on complex PBR materials. (I'm unsure - the materials I've tried so far worked fine with 4.) This limit could be expanded by just adding another sampled latent texture and changing the encoder.

## Licences

| path | licence |
|---|---|
| everything that is not third-party code: `src/` (except the three stb headers and `src/json.h`), `tools/`, `tests/`, `docs/`, `CMakeLists.txt`, the READMEs, `PRIOR_ART_DISCLOSURE.md`, `viewer/main.cpp`, `viewer/bin/nntc_view.hlsl`, `viewer/bc_check.cpp` | **Apache License 2.0**, Copyright (C) 2026 Richard Geldreich Jr. (`LICENSE`) |
| `src/json.h` | public domain (Unlicense), sheredom |
| `viewer/bcdec.h` | MIT / Unlicense dual, iOrange |
| `src/stb_image.h`, `src/stb_image_write.h`, `src/stb_image_resize2.h` | public domain / MIT dual, Sean Barrett |
| `examples/PavingStones141_1K-PNG_*.png` | CC0, the PavingStones141 material from [ambientCG](https://ambientcg.com); the other example textures are the author's |
