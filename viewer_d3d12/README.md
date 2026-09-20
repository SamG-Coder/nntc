# nntc_view_d3d12: the Direct3D 12 viewer

`nntc_view_d3d12` takes an asset's descriptor (`PREFIX_nntc.json`) and the `.dds` files it names beside it
(`../docs/FORMAT.md`), samples each latent plane once per pixel, runs the decoder in the pixel shader and draws the
result on a quad or a cube. It is one of the tree's three viewers, beside the Direct3D 11 one in `../viewer/` and the
Vulkan one in `../viewer_vk/`, and it has their window, their keys and their overlay.

What is its own: it compiles **the Direct3D 11 viewer's own shader file to the same bytes**, so the only thing that can
differ between the two Direct3D frames is the API path -- descriptors, samplers, upload, LOD, rasterisation. It is
Direct3D 12 at **feature level 11_0** and nothing beyond it, against the Windows SDK's `d3d12`, `dxgi` and
`d3dcompiler`: no Agility SDK, no redistributable runtime, nothing vendored, so a default build runs on any Direct3D 12
GPU with no package to install. Performance is not a goal -- the frame loop waits for the GPU at the end of every frame,
because the simplest correct loop is the one a reader can check.

An optional Shader Model 6.10 linear-algebra decode path sits behind the CMake option `NNTC_D3D12_LINALG`, which is
`OFF`. The last section of this file says what it is and what it measures; `../docs/D3D12_LINALG_BUILD.md` is how to
build it, and `PHASE_B_NOTES.md` beside this file is how to diagnose it on a machine where it does not work.

## Run it

Headless first, because it is what the release gate and every comparison use. The viewer does not create the directory
it writes into, so the `mkdir` is part of the recipe: without it the run exits 1 with
`ERROR: cannot write 'out_d3d12\frame.bmp': 'out_d3d12' is not a directory this program can see`. These are PowerShell
lines, from the root of the tree:

```powershell
mkdir out_d3d12 -Force | Out-Null
build\Release\nntc_view_d3d12.exe examples\m1_m4_c0_3_c1_4_nntc.json --shot out_d3d12\frame.bmp --tex 2
```

Without `--shot` the same command opens a **window**, on the adapter it picks by itself or on the one `--device` names:

```powershell
build\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json
build\Release\nntc_view_d3d12.exe examples\m1_m4_c0_3_c1_4_nntc.json --device 2
```

Every adapter is listed at start-up with its memory and whether it is hardware or WARP, and the chosen one is named, so
there is never any doubt which device drew the frame:

```
device 0: NVIDIA GeForce RTX 5090 (32187 MB, hardware)
device 1: AMD Radeon(TM) Graphics (2021 MB, hardware)
device 2: Microsoft Basic Render Driver (0 MB, WARP, software)
drawing on device 0: NVIDIA GeForce RTX 5090
```

**WARP is always the last entry** and `--device` selects it like any other, so a machine's hardware adapters keep the
indices DXGI gave them and the software rasteriser is one number past the end of them. An adapter with no Direct3D 12
device at feature level 11_0 is listed with ` - no Direct3D 12 device at feature level 11_0` after its name, and asking
for it by number is a refusal rather than a second measurement.

`--shot` is headless by design: no window class, no window and no swap chain at all, so there is no keystroke to
receive and no compositor to interact with, and five launches write five identical files.

An unknown flag, a malformed number, a `--shot` with no name, a `--shot` whose directory is not there (refused before
any Direct3D object exists), a `--size` with one value or outside 16..16384, a `--device` that is not a number or past
the end, `--bench` together with `--shot`, a descriptor that is missing, is a directory or is empty, and a
`lod_bias_level1` outside `[0, 8]` all print a line beginning `ERROR` to stderr and exit 1. A machine this viewer
cannot run on at all says **`no Direct3D 12 device`**, which is the phrase the release gate reads to skip its
Direct3D 12 arms rather than fail them.

## Keys

The Direct3D 11 viewer's keys, letter for letter, and its `F1` help page. Each toggle prints that viewer's line, in
that viewer's words.

| key | what it does |
|---|---|
| arrows | move in x and y |
| `W` / `S` | zoom in and out |
| `A` / `D` | yaw |
| `Q` / `E` | pitch |
| `Shift` | a third of the speed, while it is held |
| `Space` | reset the camera, the raw views, the renormalise flag and the shown texture |
| `Esc` | quit (with the help page open: close the page only) |
| `F1` | a help page of every key, below the overlay. Any key closes it and does nothing else, and a movement key that closed it moves nothing until it is released |
| `C` | the cube rather than the quad |
| `P` / `B` / `T` | point / bilinear / trilinear filtering |
| `X` | anisotropic filtering on / off (on by default, `MaxAnisotropy` 8; it applies in TRILINEAR mode only, and in the other two the overlay says the flag is remembered and idle) |
| `M` | mips on / off (the second sampler set with `MaxLOD` 0, so the hardware reads mip 0 of both textures; nothing is reloaded) |
| `L` | level 1's LOD shift on / off (on by default: its UV gradients scaled by `2^lod_bias_level1`, which is the encoder's 1:1 mip rule) |
| `N` | the next output texture of the material, cycling through all of them -- up to six, which is what the shader's `W[108]`, `bias[5]` and `outv[18]` carry |
| `V` | renormalise the shown triple as a tangent-space normal (unpack, unit length, repack; grey-ish and black-ish / z < 0 pixels are left alone; off by default) |
| `1` / `2` | show level 0's texture as stored, or level 1's |
| `4` | level 0 from the BC4 / BC5 pack made at load -- which exists only when the file's level 0 is UNCOMPRESSED, and **does nothing** on the block-compressed default |
| `R` | recompile `nntc_view.hlsl` from beside the executable. A shader that does not compile prints `SHADER ERROR` with the compiler's own line numbers and **both previous pipeline state objects are kept** (they are built into locals and committed only when both succeed) |
| `5`-`8` | the shader's spare debug constants, which do nothing in the shader as it ships |
| `K` | switch the decode path. **The key exists only in a build made with `NNTC_D3D12_LINALG`**; in a default build nothing is bound to it, and pressing it does nothing and prints nothing. In a build that has it, a device the query refused, or a run given `--linalg 0`, makes it print the reason and change nothing |

## Flags

`--help` or `-h` prints this list and exits 0. Every build parses every flag, including the last three.

| flag | what it does |
|---|---|
| `--shot FILE.bmp` | render ONE frame to a 24-bit bottom-up `.bmp` and exit, with **no window and no swap chain at all** |
| `--tex N` | show output texture `N` of a material (the key `N` cycles them) |
| `--cube` | the cube rather than the quad (the key `C`) |
| `--z F`, `--yaw F`, `--pitch F` | the camera, in the other viewers' units |
| `--nomips` | the sampler's `MaxLOD` is 0, so every fetch reads mip 0 (the key `M` off) |
| `--nobias` | level 1 without its LOD shift of `log2(block)`, i.e. its UV gradients unscaled (the key `L` off): the control, not a preference |
| `--noaniso` | anisotropy off, which applies in the trilinear mode only (the key `X` off) |
| `--renorm` | renormalise the decoded triple as a tangent-space normal (the key `V`) |
| `--raw0` | show level 0's own channels instead of the decode (the key `1`) |
| `--raw1` | show level 1's own channels instead of the decode (the key `2`) |
| `--bc` | start with level 0 on its BC4 / BC5 pack, made at load when the file itself is uncompressed (the key `4`) |
| `--nooverlay` | draw the scene without the debug strip, so two `--shot` frames can be compared byte for byte |
| `--size W H` | the `--shot` frame's size, 2560x1440 by default, and the window's, 1280x720 by default; the window is resizable and the picture follows |
| `--device N` | draw on DXGI adapter `N` rather than on the first one that has a Direct3D 12 device |
| `--linalg 0\|1` | the decode through the Shader Model 6.10 linear-algebra instruction (1) or through the plain matrix multiply (0); the key `K` switches at run time. The path is taken by default wherever the build and the device offer it, and `--linalg 1` where either does not is a **refusal** -- exit 1, one `ERROR` line -- rather than a quiet fallback. `--linalg 0` is always honoured, on every build |
| `--linalg-layout row\|optimal` | which layout the fp16 matrix is read in: row-major as the host writes it, or the device's own multiply-optimal layout through `ConvertLinearAlgebraMatrix`. The default is the optimal one where the runtime offers it, and the two are meant to draw one picture |
| `--bench N` | render N frames headless, with no window and no file written, and print the mean and the minimum GPU time of the scene draw in microseconds. It times **whichever decode path is running** and names it. The scene draw **ends with the overlay draw**, so a timing without `--nooverlay` includes that draw. **Informational only.** Refused together with `--shot` |
| `--help`, `-h` | the usage, exit 0 |

In a **default** build `--linalg 0` is a no-op, `--linalg-layout` changes nothing, `--bench` times the one decode path
there is, and `--linalg 1` exits 1 with
`ERROR: --linalg 1 was asked for and this device cannot: this build was made without the CMake option
NNTC_D3D12_LINALG ...`.

## The shader

`../viewer/bin/nntc_view.hlsl` is **not copied and not adapted**. Both Direct3D viewers read that one file and
compile it with `D3DCompileFromFile` to `vs_5_0` and `ps_5_0`, and Direct3D 12 accepts DXBC, so the two hand their
drivers the same shader bytes. The decode, the sampling, the feature order and the `SampleGrad` gradients are
therefore one reading of the format's contract rather than two, and the frame comparison below is of the API path.

This viewer reads the file from **beside the executable and from nowhere else**, which is the Vulkan viewer's rule and
the one difference from the Direct3D 11 viewer's handling of it. A stray `nntc_view.hlsl` in the directory a run was
started from is not the one that gets compiled, and the path is printed on every compile (`shader compiled: ...`). The
build copies `../viewer/bin/nntc_view.hlsl` beside the executable every time: edit it there, build, and press `R`.

## What is inside

* the asset load through `../shared/nntc_json.h` and `../shared/dds.h` -- the same headers, the same header walk, the
  same bounds and the same cross-checks against the `.dds` headers, including the two-file level 0 and the older
  `ntc-dds-1` format string read with a note. It refuses exactly what the other two viewers refuse, in the same words
  and with the same exit code;
* the upload: one committed default-heap texture per file at the Direct3D 11 viewer's own format choice (`R8_UNORM`,
  `R8G8_UNORM`, `R8G8B8A8_UNORM`, `BC4_UNORM`, `BC5_UNORM`), every level through one staging buffer whose row pitches
  come from `GetCopyableFootprints` -- Direct3D 12 wants 256-byte-aligned rows, a `.dds` is packed tight, and for a
  block format a "row" is a row of 4x4 blocks;
* eight sampler states -- `[mips on / off][point, bilinear, trilinear, anisotropic 8x]`, one set, since level 1's LOD
  shift is in the gradients and not in a sampler -- written **twice each** into a sampler heap of sixteen, so that one
  descriptor table of two gives `s0` and `s1` the same state;
* a root signature of four parameters: `b0` and `b1` as root CBVs (the shader's two `cbuffer`s unchanged, `W[108]` and
  `bias[5]` included), one table of SRVs -- three, or four in a build with the linear-algebra option, whose fourth is
  that path's weight buffer at `t3` -- and one table of two samplers, version 1.0 so that the `5_0` DXBC binds to it
  with no `RootSignature` attribute in the HLSL;
* the load-time BC4 / BC5 pack of an uncompressed level 0 (key `4`), through the same `../shared/bc_pack.h`, printing
  the same round-trip numbers as the other two viewers because it is the same header doing the packing;
* a Debug configuration that enables `ID3D12Debug::EnableDebugLayer`, `ID3D12Debug1::SetEnableGPUBasedValidation` and
  DRED's auto-breadcrumbs and page faults, and sets the info queue to break on a `CORRUPTION` or `ERROR` message. A
  machine without the Graphics Tools optional feature has no `D3D12SDKLayers.dll`: that is one `WARNING` line and the
  viewer runs without them.

## The overlay

The other two viewers' four-line debug strip, drawn with the same 8x8 font at the same scale in the same layout: the
asset, the state, the camera, and the key reminder. Below those four **every build** draws a fifth line naming the
decoder that drew the frame, so this viewer's strip is **104 rows** where the other two are 84. The fifth line's
glyphs begin at row 84, which leaves the four shared lines the whole of their 84-row band.

The strip is rasterised on the CPU into one RGBA buffer -- that part is the Direct3D 11 viewer's code, because nothing
about a bitmap font is API-specific -- copied into a default-heap texture at the top of the frame's command list when
the text has changed, and drawn as one alpha-blended quad by a second pipeline state whose two shaders are string
constants.

The fifth line has four wordings, one per state:

| the line reads | when |
|---|---|
| `Decode: linear algebra, through Shader Model 6.10` | the linear-algebra path drew the frame |
| `Decode: plain, though this device offers linear algebra` | the plain path drew it by choice (`--linalg 0`, or the key `K`) |
| `Decode: plain, no linear algebra - <reason>` | this device has no other path, and the reason says why |
| `Decode: plain, no linear algebra in this build` | a build made without `NNTC_D3D12_LINALG` |

## Is it the same picture as the Direct3D 11 viewer's?

`../tools/frame_diff.py` answers that as a number. `--noaniso` is the other viewers' convention, since anisotropic tap
placement is free to differ between implementations -- though here, uniquely among the three, it does not:

```
build\Release\nntc_view.exe       ASSET_nntc.json --nooverlay --noaniso --shot a.bmp --tex 0
build\Release\nntc_view_d3d12.exe ASSET_nntc.json --nooverlay --noaniso --shot b.bmp --tex 0 --size 2560 1421
python tools\frame_diff.py a.bmp b.bmp --max-diff 2 --min-psnr 40
```

**On one adapter the two viewers' frames are byte-identical, in every state a `--shot` can be put in, with anisotropy
ON as well as off.** On `examples/pavingstones141_1k_c0_4` (five textures, a two-file BC5 + BC5 level 0) and
`examples/m1_m4_c0_3_c1_4` (four textures, a BC5 + BC4 level 0), each at `--tex 0` through `--tex 3`, at the default
camera and at `--z -50`, with `--nomips`, `--nobias`, `--raw0`, `--raw1`, `--renorm`, with the cube at
`--yaw 30 --pitch 20`, with `--bc`, and with anisotropy left on at both cameras: 28 pairs, every one of them
`max |diff| 0`, `PSNR inf`. Twelve more on another tree's measured materials (a single-file two-channel level 0, four
output textures) at each of their four textures: the same. So did the load-time BC pack of an uncompressed level 0,
which both viewers make from the same `../shared/bc_pack.h`, and which is the gate's one arm pinned at `max 0`.

The Vulkan pair is byte-identical only with anisotropy off (`../viewer_vk/README.md`); here the DXBC is literally the
same, so the sampler, the fill rule and the decode arithmetic are one implementation's and not two.

**Across adapters the difference is a different vendor's bilinear weight rounding, not this program.** Each row is this
viewer on the named adapter against the RTX 5090's Direct3D 11 frame, `--noaniso`:

| this viewer on | asset and camera | max \|diff\| | PSNR |
|---|---|---|---|
| the integrated Radeon (`--device 1`) | `pavingstones141_1k_c0_4`, `--tex 0` | 2 of 255 | 67.89 dB |
| WARP (`--device 2`) | `pavingstones141_1k_c0_4`, `--tex 0` | 2 of 255 | 68.30 dB |
| the integrated Radeon | `m1_m4_c0_3_c1_4`, `--z -50` | 2 of 255 | 92.19 dB |
| WARP | `m1_m4_c0_3_c1_4`, `--z -50` | 1 of 255 | 93.06 dB |
| the integrated Radeon | the gate's 64x64 asset | 6 of 255 | 65.41 dB |
| WARP | the gate's 64x64 asset | 5 of 255 | 64.96 dB |

The 64x64 rows are larger because a magnified texel there covers many pixels. The gate's bar for a cross-adapter pair
is a loose 16.

Two things in this viewer could differ from the Direct3D 11 one and never have: the depth attachment is `D32_FLOAT`
here and `D24_UNORM_S8_UINT` there (no geometry in this viewer is coplanar, so no depth comparison is decided by the
format's precision, and the cube pairs above confirm it), and `--shot` renders into an offscreen target rather than
into a swap-chain back buffer.

## What differs from the Direct3D 11 viewer

Four things, all deliberate.

* **`--size W H`**, which that viewer has no equivalent of, and **`--device N`**, which it has no need of: it lets
  Direct3D 11 pick the adapter. Both are the Vulkan viewer's flags with the Vulkan viewer's spellings -- this `--shot`
  creates no window, so it renders at whatever size it is given, while the Direct3D 11 `--shot` renders into its
  **window**, which Windows clamps to the desktop's work area;
* **the headless `--shot`**: no window class, no window and no swap chain at all, where the Direct3D 11 viewer creates
  a hidden window because its swap chain needs an `HWND`;
* **the shader is read from beside the executable and from nowhere else** (above), where the Direct3D 11 viewer looks
  in the working directory first;
* **the window opens at 1280x720** rather than at 2560x1440, because a 2560-wide window does not fit every desktop;
  `--size` sets both it and the frame.

## Building

The root CMakeLists builds `nntc_view_d3d12` inside its Windows block, beside `nntc_view`, with `/W4` and no warnings
and with no dependency the Direct3D 11 viewer does not already have:

```
cmake -B build -S . -G "Visual Studio 18 2026" -T cuda=13.4
cmake --build build --config Release
cmake --build build --config Debug
```

That is the default build and the product. For the optional linear-algebra path, which is a second build directory and
three preview packages, see `../docs/D3D12_LINALG_BUILD.md`.

## What the release gate checks

`../tests/run_checks.py` runs the Direct3D 12 arms after the Vulkan ones and **skips them with the machine's own
reason** when `nntc_view_d3d12` was not built (it is a Windows target) or when the machine reports no Direct3D 12
device. The arms:

* the pairs against the Direct3D 11 viewer -- a single image, a two-texture material at `--tex 0` and `--tex 1`, a
  six-texture one, a two-file level 0, `--z -50 --noaniso` with and without level 1's LOD shift, `--raw0`, `--raw1`,
  `--renorm` and `--cube`, all at `--max-diff 2 --min-psnr 40`, and the load-time BC pack at `--max-diff 0`. The two
  Direct3D 12 frames of a pair are asserted to differ **from each other**, so a pair cannot be one state compared
  twice;
* every other adapter of the machine, each at the looser `--max-diff 16` and each with its own five identical
  launches;
* the six `--tex` frames of a six-texture material pairwise different, and a `--tex` past the end warning in the same
  words as the other viewers and falling back to texture 0;
* `--nomips`, `--nobias` and `--noaniso` each changing the frame at `--z -50`, and `--raw0` / `--raw1` / `--renorm`
  each changing it at the default camera; `--bc` binding the load-time pack of an uncompressed level 0 and doing
  nothing on a block-compressed one;
* the overlay: the strip's first 84 rows byte-identical to the Direct3D 11 viewer's **over the whole width** in every
  build, the fifth line's rows present with the overlay on and absent with `--nooverlay`, that line wholly on screen at
  1280x720, 1920x1080 and 2560x1440, and the rows below the 104-row strip equal to each viewer's own `--nooverlay`
  frame. The gate cannot read the line as text, so it measures how far it runs: the four wordings are four different
  lengths, and the rightmost white pixel says which one was drawn;
* `--shot` with no `--size` being a 2560x1440 24-bit frame, and five identical launches;
* `--help` exiting 0 and ten refusals exiting 1 with their own lines;
* a broken `nntc_view.hlsl` beside the executable printing `SHADER ERROR`, exiting 1 and writing no frame -- and a
  decoy `nntc_view.hlsl` in the working directory **not** being the one compiled.

## The optional Shader Model 6.10 linear-algebra decode path

Off by default, and **a build with it off is the viewer described above and nothing else** -- no preview package, no
`D3D12SDKVersion` export, no second shader, and frames byte for byte what they were. Everything specific to it is
inside `#if NNTC_D3D12_LINALG` in `main.cpp`. How to build and run it is `../docs/D3D12_LINALG_BUILD.md`; what to do
when it does not work is `PHASE_B_NOTES.md` beside this file; the design record is
`../docs/D3D12_LINEAR_ALGEBRA_PLAN.md` part II.

**What it is.** One instruction in place of two loops. Where `nntc_view.hlsl` writes the 18 by 24 product as nested
loops over the constant buffer's `W`, `bin/nntc_view_linalg.hlsl` beside this file hands it to the Shader Model 6.10
linear-algebra API as one thread-scope `MultiplyAdd`, with the fp16 matrix read from a `ByteAddressBuffer` at `t3` and
the bias beside it.
The rest of that file is a transliteration of `nntc_view.hlsl` -- the same two `SampleGrad` calls, the same
dequantise-after-sampling, the same feature order, the same clamp, the same renormalisation. It is the same shape of
thing the Vulkan viewer does with `VK_NV_cooperative_vector`, through a vendor-neutral API instead of an NVIDIA one.

`--linalg`, `--linalg-layout`, `--bench` and the key `K` steer it (above). Every run prints the device query step by
step and ends it with `linear algebra: available - ...` or `linear algebra: not available - <why>`, then prints
`decode: linear algebra` or `decode: plain` once the pipeline exists; `PHASE_B_NOTES.md` reads those lines one at a
time.

**What it measures**, at 2560x1440 with `--nooverlay --noaniso`, with the **picture** as the only acceptance bar:

| pair | asset | max \|diff\| | mean per channel | PSNR |
|---|---|---|---|---|
| RTX 5090, linear algebra vs plain (fp16 bias and result, multiply-optimal) | `pavingstones141_1k_c0_4` | 1 of 255 | 0.0024 / 0.0017 / 0.0026 | 74.62 dB |
| RTX 5090, linear algebra vs plain | `m1_m4_c0_3_c1_4` | 1 of 255 | 0.0029 / 0.0054 / 0.0025 | 72.52 dB |
| RTX 5090, `--linalg-layout row` vs `optimal` | `pavingstones141_1k_c0_4` | 0 | -- | identical |
| preview WARP, linear algebra vs plain (fp32 bias and result) | `pavingstones141_1k_c0_4` | 1 of 255 | 0.0005 / 0.0006 / 0.0008 | 79.94 dB |
| preview WARP, linear algebra vs plain | `m1_m4_c0_3_c1_4` | 1 of 255 | 0.0008 / 0.0010 / 0.0014 | 77.79 dB |
| preview WARP linear algebra vs RTX 5090 linear algebra | `pavingstones141_1k_c0_4` | 2 of 255 | 0.0111 / 0.0093 / 0.0079 | 68.39 dB |

The first two rows are the Vulkan cooperative-vector path's numbers to every printed digit, per-channel means
included, for the same two assets (`../viewer_vk/README.md`): two APIs, two instructions, one driver, the same fp16
weights, the same answer. WARP is tighter than either because it grants the fp32-result row, so the accumulation is not
forced through fp16. The RTX 5090 grants only the mandatory all-fp16 row (`SUPPORTED | TRANSPOSE`); the preview WARP
grants both (`SUPPORTED | EMULATED_OUTPUTS` and `SUPPORTED`). The integrated Radeon answers Shader Model 6.8 and is
refused by name. `--bench` at 640x480 over 20 frames on the RTX 5090, informational: mean 6.8 us on the
linear-algebra path against 22.2 us on the plain one.

**The two pipelines differ by more than that one instruction.** The linear-algebra pipeline's **vertex** stage is
compiled by DXC for `vs_6_10` while the plain pipeline's is `fxc`'s DXBC, because a pipeline state that mixes the two
is refused, and a last-bit difference in a clip-space position can move a pixel of an edge, which is not fp16 rounding
in the decode. Separating the compiler from the instruction needs a third arm, the plain shader compiled by DXC, which
is the plan's B.6 and is **not implemented**. The differences in the table are of the size fp16 rounding accounts for
and they agree with the Vulkan path digit for digit, but they have not been shown to be fp16 rounding alone.

**Four traps.**

* An option-ON executable **without** its `D3D12\` folder beside it gets **no Direct3D 12 device at all**, on every
  adapter including WARP. The runtime does not fall back to the system one. The viewer says so in as many words, and
  the gate's copy-the-executable arms copy those files with it.
* The preview `d3d10warp.dll` has to sit **beside the executable**, not in the `D3D12\` folder. A copy under `D3D12\`
  is silently not picked up and the run gets the system WARP, which answers Shader Model 6.8 and tier 0.
* A pipeline state whose **vertex** stage is `fxc` DXBC and whose pixel stage is DXIL is refused:
  `CreateGraphicsPipelineState` returns `E_INVALIDARG`, on the RTX 5090 and on WARP alike.
* The Agility SDK's header spells the granular query with the word repeated,
  `D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT`, where the runtime specification writes it
  once. The header decides.

**What the gate adds** when the option is on: the two paths' picture at `--max-diff 4 --min-psnr 50`, the two layouts'
picture at the same bar, `--linalg 1` refused with an `ERROR` and exit 1 on a device that cannot, `--bench` running and
printing a number on both paths (and refusing `--shot` alongside it), and the two paths' overlay strips byte-identical
above the decode line and differing in it at 1280x720, 1920x1080 and 2560x1440. Every baseline arm names `--linalg 0`
for itself, so what they compare is still the plain path.

**A skip and a failure are not the same thing** (`d3d12_viewer_checks` in `../tests/run_checks.py`). The gate decides
whether the machine is capable from the diagnostic lines -- the tier line reading `TIER_1_0`, the shader-model line
answering 6.10 or better -- and reads the **last** `linear algebra: available` / `not available` line, not the first,
because the query can pass and a later step can still take the path away. A build made without the option, and a device
that genuinely cannot, are **skips** with their own reasons. A capable machine that then lost the path is a
**failure**: a broken shader, a missing or unloadable `dxcompiler.dll`, or a resource that would not allocate. Two
exceptions are pinned by wording: the asset's own fp16 range guard, whose message names 65504, stays a skip, and an
option-ON build reporting no Direct3D 12 device at all is a failure, because that is its preview runtime missing from
`D3D12\` rather than a machine without Direct3D 12.
