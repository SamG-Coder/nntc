<!-- Copyright 2026 Richard Geldreich, Jr. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NNTC: Non-Neural Texture Compression

The `nntc_encode` tool (for Windows or Linux) compresses a 24-bpp texture, or a PBR material of up to six 24-bpp textures, into **two mipmapped latent planes and (typically) a few dozen
fitted matrix coefficients**. It outputs two `.dds` textures, or three when the full-resolution latent plane has three or four channels. A pixel shader decodes
the whole material back from **two ordinary `Sample()` calls**, or three when the full-resolution plane has three or
four channels, and one matrix multiply of a few dozen to a few hundred multiply-accumulates per pixel for the whole
material (the count is in [How it works](#how-it-works-briefly), below). No runtime library, no decompression at load, no neural inference: the files are standard mipmapped `.dds`
(BC4 / BC5 and 8-bit) and a `.json` containing the fitted bilinear model's decoder coefficients. The [VK_NV_cooperative_vector](https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_vector.html) extension is supported in the Vulkan viewer for potentially faster decoding.

NNTC is a generalized latent/modulation codec, in the same broad family of ideas as PVRTC1, but with more latent dimensions and a fitted bilinear decoder. It uses a bilinear/degree-2 polynomial decoder, i.e. a degree-2 polynomial whose only quadratic terms are the products between the two latents stored as standard BC4/BC5/uncompressed mipmapped textures. The full-resolution latent texture(s) represent edge/detail information, while the quarter-resolution latent texture represents slowly varying material/color state.

The `nntc_view` tool is a Windows D3D11 viewer, which runs on any GPU, that loads the .json and .dds files and displays the sampled and decoded results on a textured quad or a cube, with keyboard camera controls. `nntc_view_vk` is the same viewer on Vulkan, which also runs on any GPU, and with the same window and keys, for Windows and Linux (`viewer_vk/README.md`).

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

You need CMake, a C++17 compiler and, for the encoder, the CUDA toolkit. CUDA 12.8 or newer is the minimum the
build accepts; 13.4 (VS 2026) and 13.3 (WSL) are what this tree is built and gated with.

**CUDA is optional for the viewers.** CMake probes for a CUDA compiler: with one, the encoder is built; without one it
prints `-- No CUDA compiler was found: the encoder nntc_encode is skipped` and builds the rest alone -- on Windows
both viewers and `bc_check`, on Linux the Vulkan viewer (the Direct3D viewer and `bc_check` are Windows programs) --
so an Arm laptop or an AMD or Intel box can build and run the viewer on assets encoded elsewhere (the `examples/`
directory ships several; `build/nntc_view_vk examples/pavingstones141_1k_c0_4_nntc.json` opens one). The release gate needs the encoder and does not run on such a machine.

### Windows

The toolchain this tree is developed on is **Visual Studio 2026 with CUDA 13.4** (CMake 4.2 or newer; the CMake
bundled with VS 2026 is fine):

```
cmake -B build -S . -G "Visual Studio 18 2026" -T cuda=13.4
cmake --build build --config Release
```

That produces `build\Release\nntc_encode.exe`, `nntc_view.exe` and `bc_check.exe`, and -- when the Vulkan SDK is
installed -- `nntc_view_vk.exe`, the second viewer (`viewer_vk/README.md`).

**The Vulkan SDK is optional.** It is what `find_package(Vulkan)` needs to find a loader and a `shaderc_combined`,
which is how that viewer compiles its GLSL at runtime; this tree was built and measured against SDK 1.4.357.0. Without
it -- or with an SDK that ships no `shaderc_combined` -- CMake prints one `-- ...nntc_view_vk is skipped` line at
configure time, that one target is not built, everything else configures and builds exactly as before, and the release
gate reports the Vulkan cases as skipped rather than failing them.

Visual Studio 2022 with CUDA 13.1 is also supported and produces byte-identical assets:

```
cmake -B build -S . -G "Visual Studio 17 2022" -T cuda=13.1
cmake --build build --config Release
```

### Linux

Install the packages first (the Debian / Ubuntu line; the table below has the other families), then build. Release
is the default when no build type is named; `-DCMAKE_BUILD_TYPE=Debug` asks for Debug:

```
sudo apt install -y build-essential cmake pkg-config libvulkan-dev libshaderc-dev libglfw3-dev mesa-vulkan-drivers vulkan-tools
cmake -B build -S .
cmake --build build -j
```

A full configure prints no warning. Anything the configure has to leave out is a **CMake Warning** that names the
package to install; the ones a Linux box can meet:

| the warning says | what is missing | install (Debian / Ubuntu) |
| --- | --- | --- |
| `No CUDA compiler was found: the encoder nntc_encode is skipped` | the CUDA toolkit; expected on any box without an NVIDIA card | nothing, unless you want the encoder: NVIDIA's `cuda-toolkit-13-x`, then `cmake -UCMAKE_CUDA_COMPILER` |
| `Vulkan was not found: the target nntc_view_vk is skipped` | the Vulkan loader's headers and library | `libvulkan-dev` |
| `No shaderc was found ...: nntc_view_vk is skipped` | the run-time GLSL compiler's library and header | `libshaderc-dev` |
| `GLFW was not found: nntc_view_vk builds HEADLESS` | the window library; only `--shot` runs without it | `libglfw3-dev` |
| `cooperative vectors are compiled OUT` | Vulkan headers older than 1.4.307; expected on every current distribution, and nothing is missing on AMD or Intel | nothing; for NVIDIA under Linux, the LunarG SDK's headers |

After installing, run `cmake -B build -S .` again: the probes re-run on their own (only the CUDA miss is cached, hence
its `-U`).

That produces `nntc_encode`, and `nntc_view_vk` wherever `find_package(Vulkan)` finds a loader and a shaderc --
with a GLFW window when GLFW is found (native X11 or Wayland with a GLFW 3.4, X11 through Xwayland with the 3.3 of Ubuntu and Debian 12) and headless when it is not, since its `--shot` needs no window, no surface
and no desktop. `nntc_view`, the Direct3D 11 viewer, is Windows only. Verified under WSL 2 (Ubuntu 24.04,
gcc 13.3, CUDA 13.3 from NVIDIA's `wsl-ubuntu` apt repository): the gate passes and the asset it writes is
byte-identical to the Windows build's -- on the default source chain.

The packages, by distribution family. The encoder needs the CUDA toolkit on top of these; without `nvcc` the
configure skips it and builds the viewer alone.

| | packages |
| --- | --- |
| Debian, Ubuntu, Mint, Pop!_OS | `build-essential cmake pkg-config libvulkan-dev libshaderc-dev libglfw3-dev vulkan-tools`, plus the driver: `mesa-vulkan-drivers` for AMD and Intel, or the NVIDIA driver package (`nvidia-driver-NNN`, which carries its own Vulkan driver) (and `vulkan-validationlayers` for a Debug build) |
| Fedora | `gcc-c++ cmake pkgconf vulkan-loader-devel vulkan-headers libshaderc-devel glfw-devel vulkan-tools`, plus the driver: `mesa-vulkan-drivers` for AMD and Intel, or the NVIDIA driver from RPM Fusion (and `vulkan-validation-layers`) |
| Arch | `base-devel cmake vulkan-icd-loader vulkan-headers shaderc glfw vulkan-tools`, plus the driver for the part -- `vulkan-radeon`, `vulkan-intel` or `nvidia-utils` (and `vulkan-validation-layers`) |
| openSUSE | `gcc-c++ cmake pkgconf-pkg-config vulkan-devel vulkan-headers shaderc-devel libglfw-devel vulkan-tools`, plus the driver: `libvulkan_radeon` or `libvulkan_intel`, or the NVIDIA driver (and `vulkan-validationlayers`) |

The Debian and Ubuntu rows were used on real machines; the Fedora, Arch and openSUSE names are from those
distributions' package indexes and have not been typed on a box here, so if one is wrong the package search of
that distribution (`dnf search shaderc`, `pacman -Ss glfw`, `zypper se glfw`) is the fix.

Two things about those lists are worth stating rather than discovering.

**shaderc.** The viewer compiles its GLSL at run time, so it needs shaderc's library and header. On Linux the build
prefers the distribution's SHARED `libshaderc`, because `libshaderc_combined.a` is not self-contained on every
release -- Ubuntu 24.04's `libshaderc-dev` ships a 242 KB archive that defines neither `spvValidatorOptionsDestroy`
nor `spvtools::Optimizer`, and a link against it alone fails on both -- while the shared library carries its
dependencies as its own `NEEDED` entries. The combined archive stays the fallback, which is what a [LunarG SDK install](https://www.lunarg.com/products/vulkan-sdk/)
with no shared library on the path uses, and is what the Windows build links.

**Cooperative vectors.** `VK_NV_cooperative_vector`, the accelerated decode path, first appears in Vulkan-Headers
1.4.307. Older headers are the norm -- Ubuntu 22.04 ships 204, Debian 12 ships 239, Ubuntu 24.04 ships 275 -- and
against them that path is COMPILED OUT: the configure says so in one line, the viewer says so in one line at
start-up, `--coopvec 1` is refused with an error, and the plain GLSL decode draws. That is not a reduced build. The
plain path is the product, it is what every measurement in this tree is of, and on AMD and Intel parts the extension
does not exist at all, so it is what runs there in any case. To compile it in, install the LunarG SDK (or a
distribution new enough to ship 1.4.307 headers) and point the configure at it with `-DVulkan_INCLUDE_DIR=...`, or
let `find_package(Vulkan)` pick it up from `VULKAN_SDK`. Whether it then RUNS is still the device's own query.

The floor for the viewer is Vulkan headers 204 -- 1.3 core, which is what every frame is recorded with. Anything
older stops at a `#error` naming the package to install.

Notes: A run whose mipmaps come from `stb_image_resize2` (a named filter, or the `box` implied by `srgb`, `edge` or
`normal_map`) can differ
in the last bit between platforms, so those runs agree across platforms in PSNR rather than byte for byte
(`docs/DESIGN.md` section 6 has the measurement). Inside WSL, prefer the `cuda-toolkit-13-x` package: the `cuda` and
`cuda-drivers` metapackages try to install a Linux driver, which WSL does not use (the driver is Windows'). Put
`/usr/local/cuda-13.x/bin` on your PATH yourself, and do it BEFORE the first configure: with `nvcc` off the PATH the
build does not refuse, it skips the encoder, and the probe's answer is cached, so a tree configured too early keeps
skipping it until you re-configure with `-UCMAKE_CUDA_COMPILER` (or delete the build directory).

## Encoding

```
nntc_encode albedo.png                                      -> albedo_lat0.dds, albedo_lat1.dds, albedo_nntc.json in the current directory
nntc_encode albedo.png normal.png rough.png                 -> a 3-texture material, named after the first input
nntc_encode a.png b.png -o out/                             -> the same files under out/
nntc_encode a.png b.png --weights 2,1                       -> the first texture matters twice as much
nntc_encode a.png --c0 1 --c1 2                             -> a smaller layout: 4 + 1 = 5 bits per pixel at the base
nntc_encode rock_material.json -o out/                      -> the same, with the inputs and their settings named in a file
nntc_encode a.png -o out/name.json                          -> the descriptor is exactly out/name.json, the .dds out/name_lat0.dds
nntc_encode me1.png me2.png me3.png me4.png --c0 3          -> 3 channels on latent 0, a BC5 and a BC4 (a harder material)
nntc_encode me1.png me2.png me3.png me4.png me5.png --c0 4  -> 4 channels on latent 0, two BC5s (a harder material still)
nntc_encode albedo.png --rounds 60                          -> Use more encoding rounds for higher quality (slower)
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

There is a **second viewer on Vulkan**, `nntc_view_vk`, which draws the same asset through another graphics api with
the same keys, the same fourteen flags plus `--size` and `--device`, and the same overlay -- the cheapest evidence there
is that the format's claim is about *the* hardware sampling operator and not about one vendor's. On one GPU the two
viewers' frames are byte-identical with anisotropy off, in every state a --shot can be put in. Its
`--shot` needs no window, no surface and no desktop at all. On a device that offers
`VK_NV_cooperative_vector` it also decodes through that extension's one matrix-vector instruction instead of the
shader's own loop -- a second pipeline behind a runtime query, the same picture to 1 part in 255, and 4 to 5 times
faster on the scene draw at 2560x1440; every other device never hears of it.
`viewer_vk/README.md` has its key table, its flags, the parity numbers, the cooperative-vector measurements and what
differs between the two.

### The two ways to apply the level-1 mip shift

The quarter-resolution latent has to be read from the mip the encoder paired with level 0's: `lod_bias_level1` = 2
levels coarser than the GPU would pick on its own. There are two ways to ask the hardware for that, and they are not
interchangeable on every GPU.

| | **A. sampler LOD bias** (NVIDIA / AMD) | **B. scaled gradients** (Intel / generic; what the viewers ship) |
| --- | --- | --- |
| how | level 1's sampler carries `MipLODBias` / `mipLodBias` = `lod_bias_level1` (+2); the shader uses a plain `Sample` / `texture` | the shader samples level 1 with `SampleGrad` / `textureGrad`, both UV derivatives multiplied by `2^lod_bias_level1` (4); no sampler bias |
| cost | one sampler-state field, nothing in the shader | explicit-gradient sampling, which some hardware runs a little slower than a plain sample |
| mip selection | correct on NVIDIA and AMD. **Wrong on Intel integrated graphics**: a magnified texture comes out very blurry (seen on a 13th-generation Core i7 under Direct3D 11 and Vulkan; the base LOD of a magnified texture seems not to go negative there, so +2 lands on mip 2 instead of clamping back to mip 0) | correct on NVIDIA, AMD and Intel: the +2 is inside the LOD the hardware computes, before any clamp |
| trilinear / bilinear / point | the reference | the same mips and the same picture as A (measured byte-identical on an integrated Radeon and within rounding on an RTX 5090) |
| anisotropic | the better picture: the bias moves the LOD and leaves the line the anisotropic taps are spread along at its true length | somewhat blurrier on oblique surfaces: scaling both gradients also makes that tap line 4 times longer. Measured against a 16x-supersampled reference on close, steeply angled views, A was ahead by 3 to 7 dB on both an RTX 5090 and an integrated Radeon; square-on, or with anisotropy off, there is no difference |

**How much anisotropy level 1 gets under B is up to the driver.** Level 0 is a plain sample and always gets the
hardware's full anisotropic filter. Level 1, read through explicit gradients, has been seen to behave two ways. On
NVIDIA and AMD it is filtered anisotropically with the tap line 4 times too long (the row above), and on Intel
integrated graphics (a 13th-generation Core i7) toggling anisotropy visibly changes level 1 as well, so it is
filtered anisotropically there too (seen by eye, not measured against a reference). On Mesa's lavapipe, the CPU
Vulkan driver, explicit-gradient samples get no anisotropy at all, only trilinear filtering, so toggling anisotropy
changes level 0 and leaves level 1 byte-identical (the specification allows this, and some mobile GPUs may do the
same). None of these is a wrong picture. Level 1 is the
smooth, slowly varying latent, so what it loses at grazing angles costs little, and method A is the one that gives
it the full anisotropic filter.

**A refinement of B, documented and not shipped.** B's extra blur under anisotropic filtering comes from scaling
both gradients. Scaling only the footprint's SHORT axis by 4 raises the LOD by the same 2 levels and leaves the line
the anisotropic taps are spread along at its true length. `ddx(uv)` and `ddy(uv)` are not that ellipse's axes, so
the shader has to find them first (the closed-form 2x2 eigen-decomposition of the UV Jacobian, about ten
instructions), scale the short one, and pass the two axis vectors to `SampleGrad`. It also has to know the
sampler's maximum anisotropy `A`, because the hardware's LOD is `log2(max(short, long / A))` and it is that whole
quantity which must rise by 2: with `A` wrong the result is worse than the plain scale (measured), so it is more
fragile than it looks. Measured here it matches the bias or edges it by up to 2 dB. The viewers keep the two-line form
on purpose: it is simple, it is right on every vendor, and grazing angles only need to look good enough. An engine
that wants the last few dB on oblique surfaces can add this itself.

**Which to use.** B if one code path has to be right on every GPU, which is why both viewers use it. A if you know
the hardware is NVIDIA or AMD, or you select per vendor at run time (PCI vendor `0x8086` is Intel): it is cheaper and
it is the sharper picture under anisotropic filtering. Do not apply both at once: that is a shift of 4 levels, and
it is blurry everywhere. In the viewers, key `L` (or `--nobias`) turns the shift off entirely, which shows what it
is for.

**A change of method (2026-09-17, v1.1.21).** Earlier versions of both viewers, and of this guidance, applied the
shift as a sampler LOD bias of +2 on the second (quarter-resolution) latent. That was switched to scaled gradients
because of mipmap LOD calculation differences between Intel parts and AMD / NVIDIA ones: on Intel integrated graphics
(observed on a 13th-generation Core i7 under both Direct3D 11 and Vulkan) the +2 bias made the second latent, and so
the decoded picture, very blurry whenever the texture was magnified, while on AMD and NVIDIA it was correct. Scaled
gradients pick the right mip on all three, so they are what the viewers ship; the next section sets the two methods
side by side, including what the gradient form costs under anisotropic filtering. Nothing in the files changed:
`lod_bias_level1` in the descriptor is the same number, applied a different way.

The D3D 11 viewer has also been successfully tested on Windows ARM, Snapdragon X Elite (Lenovo ThinkPad T14s Gen 6-2024).

**If you write your own consumer**, the one thing to get right: **the quarter-resolution texture must be sampled with
its UV gradients scaled by `2^lod_bias_level1` = 4** (`SampleGrad` / `textureGrad`), so that both textures are read
from the same mip; a sampler LOD bias of the same value selects the same mips on hardware that keeps a negative base LOD under magnification (NVIDIA, AMD) and is the sharper picture there under anisotropic filtering, but was observed to blur a magnified picture badly on Intel Xe integrated graphics, so the gradient form is the one to ship. Everything else is
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
the format requires level 1 to be sampled with its UV gradients scaled by `2^lod_bias_level1` (`log2(block)` = 2 for
the fixed block of 4), so output mip `m` reads mip `m` of both latent textures. The optimisation target is therefore the real runtime
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

Also see: 
- [Texture Compression using Wavelet Decomposition](https://onlinelibrary.wiley.com/doi/10.1111/j.1467-8659.2012.03203.x) by Mavridis and Papaioannou (2012)
- [A new approach to combine texture compression and filtering](https://www.researchgate.net/publication/289735007_A_new_approach_to_combine_texture_compression_and_filtering) by Hollemeersch, Pieters, Lambert and Van de Walle (2012)
- [Texture compression using low-frequency signal modulation](https://www.researchgate.net/publication/221249059_Texture_compression_using_low-frequency_signal_modulation) by Fenney (2003)
- [Real-Time Neural Materials using Block-Compressed Features](https://onlinelibrary.wiley.com/doi/10.1111/cgf.15013) by Weinreich et al. (2024)
- [Hardware Accelerated Neural Block Texture Compression with Cooperative Vectors](https://arxiv.org/abs/2506.06040) by Belcour and Benyoub (2025)

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
| everything that is not third-party code: `src/` (except the three stb headers), `shared/` (except `shared/json.h`), `tools/`, `tests/`, `docs/`, `CMakeLists.txt`, the READMEs, `PRIOR_ART_DISCLOSURE.md`, `viewer/main.cpp`, `viewer/bin/nntc_view.hlsl`, `viewer/bc_check.cpp`, `viewer_vk/` | **Apache License 2.0**, Copyright (C) 2026 Richard Geldreich Jr. ([`LICENSE`](https://github.com/richgel999/nntc/blob/main/LICENSE)) |
| `shared/json.h` | public domain (Unlicense), [sheredom](https://github.com/sheredom/json.h) |
| `viewer/bcdec.h` | MIT / Unlicense dual, [iOrange](https://github.com/iOrange/bcdec) |
| `src/stb_image.h`, `src/stb_image_write.h`, `src/stb_image_resize2.h` | public domain / MIT dual, [Sean Barrett](https://github.com/nothings/stb) |
| `examples/PavingStones141_1K-PNG_*.png` | CC0, the PavingStones141 and another material from [ambientCG](https://ambientcg.com) |
