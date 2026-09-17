# viewer_vk: the same asset, on Vulkan

A second viewer for what `nntc_encode` writes (`../docs/FORMAT.md`), beside the Direct3D 11 one in `../viewer/`. The
design, the stages and the reasoning behind every choice in it are `../docs/VULKAN_VIEWER_PLAN.md`; this file says what
exists today and how to build it.

Why a second viewer at all: the format's claim is that the representation is valid under **the** hardware sampling
operator, not under one vendor's, and a second graphics api on the same asset is the cheapest evidence there is that
the claim is about hardware and not about Direct3D. Vulkan is also the portable path to Linux, where the encoder
already builds and where nothing can currently be looked at.

## What stage this is at

**Stage 4 of four, the last one: cooperative vectors.** The section at the end of this file is that stage; what
follows here is stage 3, which it is built on and which is untouched by it.

**Stage 3: parity.** The viewer reads `PREFIX_nntc.json` and the `.dds` files it names, uploads every level of
every one of them, builds the sixteen samplers, compiles its two GLSL files at runtime and draws the decoded material
on the quad or the cube -- and it now has **every key the Direct3D viewer has, with the same letters and the same
meaning**, its **four-line overlay**, its **load-time BC4 / BC5 pack** of an uncompressed level 0 (key `4`, through the
same `../shared/bc_pack.h`), and **every one of its flags, in the same spellings**. It refuses exactly what that viewer
refuses, in the same words and with the same exit code, because a descriptor one viewer opens and the other does not
would make the whole comparison meaningless.

What is here, level by level:

* the asset load through `../shared/nntc_json.h` and `../shared/dds.h` -- the same headers, the same header walk, the
  same bounds and the same cross-checks against the `.dds` headers, including the two-file level 0 and the older
  `ntc-dds-1` format string read with a note;
* the upload: one `VkImage` per file at the mapped `VkFormat` (`R8_UNORM`, `R8G8_UNORM`, `R8G8B8A8_UNORM`,
  `BC4_UNORM_BLOCK`, `BC5_UNORM_BLOCK`), every level through one staging buffer at 16-byte-aligned offsets, and a
  per-format check for linear filtering so that a device which cannot sample a block format is refused **by name**
  rather than by a wrong picture. The unused third slot gets a 1x1 image, because Vulkan has no null descriptor here;
* the sixteen samplers of the plan's section 1.10 -- `[mips on / off][point, bilinear, trilinear, anisotropic 8x]`,
  (one set: level 1's LOD shift is in the gradients, not in a sampler) -- with clamp
  addressing throughout, and a line at load if the device's `maxSamplerLodBias` is below what the descriptor asks for;
* one uniform buffer of 2032 bytes holding the other viewer's two constant blocks end to end, and a descriptor set of
  separate sampled images and samplers, so that a key rewrites one sampler descriptor and nothing else;
* `bin/view.vert` and `bin/view.frag`, copied beside the executable by the build, compiled at runtime by `shaderc`
  **from beside the executable and from nowhere else**, and re-read from that same place by key `R` -- the path is
  printed on every compile. The fragment
  shader is a transliteration of `../viewer/bin/nntc_view.hlsl`'s `DecodeNNTC` and `PSMain`, line for line: the feature
  order `[c ; s ; s x c]` and the dequantise-after-sampling rule are the format's contract and not something either
  viewer restates;
* the load-time pack, for an uncompressed level 0 only, printing the same round-trip numbers as the other viewer
  because it is the same header doing the packing;
* the overlay: the same 8x8 font at the same scale, the same four lines in the same layout, blended over the scene by
  a second pipeline;
* the quad and the cube, the camera, and `--shot` rendering the asset headless.

Stage 4 is the cooperative-vector experiment, and nothing of it is here.

Vulkan **1.3 core** and nothing beyond it: dynamic rendering (no `VkRenderPass`, no `VkFramebuffer`) and
synchronisation2 (one `VkImageMemoryBarrier2` per layout change). The extensions are `VK_KHR_surface` plus the
platform's surface extension, `VK_KHR_swapchain`, and in a Debug build `VK_LAYER_KHRONOS_validation` with
`VK_EXT_debug_utils` when the machine has them -- a missing layer is a warning and not a failure. The device features
asked for are `textureCompressionBC` and `samplerAnisotropy`, each of them reported at load when the device lacks it.
The loader is linked as `vulkan-1` through CMake's `FindVulkan`; there is no volk, no memory allocator, no windowing
library and nothing vendored.

**What is asked of the device rather than assumed of it.** Four things that this machine would never have caught, on
the principle that a viewer whose whole argument is portability should not be portable by luck. The **depth format**
is chosen once, after the physical device is picked, from `optimalTilingFeatures` in the order `D32_SFLOAT`,
`X8_D24_UNORM_PACK32`, `D16_UNORM`, and printed -- the last two are what the spec makes mandatory and the first is
merely what this machine has. The swapchain's **composite alpha** is the first of `OPAQUE`, `INHERIT`,
`PRE_MULTIPLIED`, `POST_MULTIPLIED` that the surface reports, since only "at least one" is guaranteed and an
unsupported bit is a failed swapchain rather than a fallback. **`VK_KHR_swapchain`** is required of a device before
it is offered as one to draw on, so a part that can present and does not expose the extension is refused with `no
VK_KHR_swapchain` in the device list instead of failing inside `vkCreateDevice`. And the two **1.3 features** this
viewer draws every frame with, `dynamicRendering` and `synchronization2`, are queried by name through
`vkGetPhysicalDeviceFeatures2` and refused by name -- a 1.3 device implies them, and querying is the defensive form
the Khronos guidance recommends. The window is also created hidden and shown only once the device, the asset and both
shaders are past refusing, so a run that exits 1 never flashes a window up.

One guard belongs to the asset rather than the device: every cooperative-vector type tuple this viewer can use reads
the matrix as **fp16**, whose largest finite value is 65504, so at load the padded `W` and bias are checked against
that range. An asset past it says so in one line, draws on the plain path (which reads `W` as the fp32 it is), and
refuses `--coopvec 1` with that reason -- an infinity in the converted matrix would be a NaN over the whole quad.

## Controls

The Direct3D viewer's keys, letter for letter:

| key | what it does |
|---|---|
| arrows | move in x and y |
| `W` / `S` | zoom in and out |
| `A` / `D` | yaw |
| `Q` / `E` | pitch |
| `Shift` | a third of the speed, while it is held |
| `Space` | reset the camera, the raw views, the renormalise flag and the shown texture |
| `Esc` | quit |
| `C` | the cube rather than the quad |
| `P` / `B` / `T` | point / bilinear / trilinear filtering |
| `X` | anisotropic filtering on / off (on by default, `maxAnisotropy` 8; it applies in TRILINEAR mode only, and in the other two the overlay says the flag is remembered and idle). On a device that does not report `samplerAnisotropy` the key says `anisotropy unsupported on this device` and changes nothing, and the overlay reads `Aniso:OFF (unsupported)` |
| `M` | mips on / off (a second sampler set with `maxLod` 0, so the hardware reads mip 0 of both textures; nothing is reloaded) |
| `L` | level 1's LOD shift on / off (on by default: its UV gradients scaled by `2^lod_bias_level1`, which is the encoder's 1:1 mip rule) |
| `N` | the next output texture of the material, cycling through all of them -- up to six, which is what the shader's `W[108]`, `bias_[5]` and `outv[18]` carry |
| `V` | renormalise the shown triple as a tangent-space normal (unpack, unit length, repack; grey-ish and black-ish / z < 0 pixels are left alone; off by default) |
| `1` / `2` | show level 0's texture as stored, or level 1's |
| `4` | level 0 from the BC4 / BC5 pack made at load -- which exists only when the file's level 0 is UNCOMPRESSED, and **does nothing** on the block-compressed default |
| `K` | the decode through `VK_NV_cooperative_vector` or through the plain GLSL matrix multiply, printing `decode: cooperative vectors` or `decode: plain`. On a device without the extension it says why and changes nothing; the overlay reads `Coop:ON`, `Coop:OFF` or `Coop:n-a` |
| `R` | recompile `view.vert`, `view.frag` and, where the extension is there, `view_coopvec.frag` from beside the executable. A shader that does not compile prints `SHADER ERROR` with `shaderc`'s own line numbers and **both previous pipelines are kept** (they are built into locals and committed only when both succeed) |
| `3`, `5`-`8` | the shader's spare debug constants, which do nothing in the shaders as they ship |

Each toggle prints the other viewer's line, in the other viewer's words.

## Running

```
build\Release\nntc_view_vk.exe examples\pavingstones141_1k_c0_4_nntc.json
build\Release\nntc_view_vk.exe examples\m1_m4_c0_3_c1_4_nntc.json --shot out_vk\frame.bmp --tex 2
build\Release\nntc_view_vk.exe examples\m1_m4_c0_3_c1_4_nntc.json --device 1
```

| flag | what it does |
|---|---|
| `--shot FILE.bmp` | render ONE frame to a 24-bit bottom-up `.bmp` and exit, with **no window and no surface at all** |
| `--tex N` | show output texture `N` of a material (the key `N` cycles them) |
| `--cube` | the cube rather than the quad (the key `C`) |
| `--z F`, `--yaw F`, `--pitch F` | the camera, in the other viewer's units |
| `--nomips` | the sampler's `maxLod` is 0, so every fetch reads mip 0 (the key `M` off) |
| `--nobias` | level 1 without its LOD shift of `log2(block)`, i.e. its UV gradients unscaled (the key `L` off): the control, not a preference |
| `--noaniso` | anisotropy off, which applies in the trilinear mode only (the key `X` off) |
| `--renorm` | renormalise the decoded triple as a tangent-space normal (the key `V`) |
| `--raw0` | show level 0's own channels instead of the decode (the key `1`) |
| `--raw1` | show level 1's own channels instead of the decode (the key `2`) |
| `--bc` | start with level 0 on its BC4 / BC5 pack, made at load when the file itself is uncompressed (the key `4`) |
| `--nooverlay` | draw the scene without the debug strip, so two `--shot` frames can be compared byte for byte |
| `--size W H` | the `--shot` frame's size, 2560x1440 by default, and the window's, 1280x720 by default; the window is resizable and the picture follows (see below) |
| `--device N` | draw on physical device `N` rather than on the first discrete GPU that can present |
| `--coopvec 0|1` | the decode through `VK_NV_cooperative_vector` or through the plain GLSL multiply (the key `K`). On by default where the device offers it; `--coopvec 1` where it does not is refused. See the last section |
| `--bench N` | render `N` frames headless and print the mean and the minimum GPU time of the scene draw, in microseconds, for the current decode path. Writes no file |
| `--help`, `-h` | the usage, exit 0 |

The name of the device that was chosen is printed at start either way, so there is never any doubt which GPU drew the
frame, and so is the surface format the swapchain took. An unknown flag, a malformed number, a `--shot` with no name, a
`--shot` whose directory is not there (refused before any Vulkan object exists), a `--size` with one value or below 16,
a `--device` that is not a number or past the end, a frame larger than the device's `maxImageDimension2D`, a
`lod_bias_level1` outside `[0, 8]`, the range the mip shift can mean, and a surface that offers neither
`B8G8R8A8_UNORM` nor `R8G8B8A8_UNORM` all print a line beginning `ERROR` to stderr and exit 1. A machine this viewer
cannot run on at all -- no loader, a loader below 1.3, or no device that is 1.3 -- says **`no Vulkan device`**, which is
the phrase the release gate reads to skip its Vulkan arms rather than fail them. **Running on other GPUs** below is the
whole list of what a device has to offer and every refusal it can meet.

`--shot` is the release gate's mode, and it is headless by design rather than by accident: with no window there is no
keystroke to receive and no compositor to interact with, so five launches write five identical files. It **takes no
keyboard input at all**, which is the rule the other viewer's `--shot` also follows and the reason the gate's five
identical shots are identical.

## The overlay

The same four-line debug strip, drawn with the same 8x8 font at the same scale in the same layout:

1. **the asset**: the source size, then each level's `WxHxchannels`, its stored format, its index bit depth and its mip
   count, then the block factor, then the GPU memory of what is bound. Level 0's format reads `BC5 (file)` or
   `BC5+BC4 (file)` when the file itself is block-compressed and nothing was packed here, `U8` when it is uncompressed
   and the load-time pack is not bound, and `BC5 lossless` or `BC5 41.2dB` when it is (key `4`);
2. **the state**: quad or cube, the filter, the anisotropy (and whether it is idle), mips on or off, level 1's LOD shift
   on or off, which texture of the material is shown, whether the decode or a raw latent is shown, and the renormalise
   flag;
3. **the camera**: x, y, z, yaw, pitch;
4. **the keys**, as a reminder line.

It is rasterised on the cpu into one `OVL_W x OVL_H` RGBA buffer -- that part is the other viewer's code, because
nothing about a bitmap font is api-specific -- copied into an image at the top of the frame's command buffer when the
text has changed, and drawn as one alpha-blended quad by a second pipeline whose two shaders are string constants. At
the Direct3D frame's own size the strip's 84 rows are **byte-identical** to that viewer's, and the rows below them are
the same as each viewer's own `--nooverlay` frame -- which is as close to "the same overlay" as two programs get, and
is what the release gate asserts rather than only that the strip is drawn.

## What differs from the Direct3D viewer

Six things, and all six are deliberate (the plan's risk 16: no capability that is not a strict superset justified there
or behind a flag whose default leaves the other viewer's behaviour intact).

* **`--size W H`**, which the other viewer has no equivalent of. It exists for the comparison rather than for its own
  sake: this `--shot` creates no window, so it renders at whatever size it is given, while the Direct3D `--shot`
  renders into its **window**, which Windows clamps to the desktop's work area. Two frames of different shapes are two
  different projections, so the comparison asks for the Direct3D frame's size;
* **the headless `--shot`**: no surface extension, no swapchain and no presentation engine at all, which is what lets
  the gate check a frame over a remote session, in a service, and on a Linux box with no desktop;
* **the runtime shader compile is `shaderc` on GLSL** rather than `D3DCompileFromFile` on HLSL, so key `R` reads
  `view.vert` and `view.frag` rather than one `.hlsl`, and a compile error arrives with `shaderc`'s line numbers. The
  two shader files are separate because `shaderc` compiles each stage on its own, which is also why the uniform block
  is declared verbatim in both;
* **the device choice is explicit**: the physical devices are enumerated and printed, the first discrete GPU with a
  queue family that can draw and present is taken, and **`--device N`** overrides that outright -- which is how the
  baseline path is exercised on a non-NVIDIA part on the same machine. Direct3D 11 picks the adapter for us;
* **the cooperative-vector decode** (`--coopvec`, the key `K`, the last section of this file), which has no
  Direct3D 11 equivalent and cannot have one -- the extension is a Vulkan extension and Microsoft withdrew its
  preview of the same idea. It is a second pipeline behind a runtime query, and the baseline is untouched by it;
* **`--bench N`**, GPU timestamps around the scene draw, which exists to measure that second pipeline against the
  first. It writes no file and it is off unless it is asked for.

## Building

The root CMakeLists builds `nntc_view_vk` wherever `find_package(Vulkan)` finds a loader **and** a shaderc -- the
SDK's `shaderc_combined` on Windows, the distribution's shared `libshaderc` on Linux, with the combined archive as
the fallback there; without one there is no way to compile a shader at all -- and skips it with a message saying
which is missing, so a tree with no SDK still configures and still builds the encoder. The reverse holds too:
a machine with no CUDA compiler skips the encoder and still builds this viewer, which is how it is tested on GPUs from
other vendors (see the root README's Building section). With the toolkit, and without it, in turn:

```
cmake -B build -S . -G "Visual Studio 18 2026" -T cuda=13.4
cmake -B build -S . -G "Visual Studio 18 2026"
cmake --build build --config Release
```

The Vulkan SDK is the one build dependency this target adds (the loader, the headers, the validation layers and
`shaderc_combined` all come from it); the **shipped executable's runtime dependency is `vulkan-1.dll` alone**. It was
built and measured against SDK 1.4.357.0. A Debug build enables `VK_LAYER_KHRONOS_validation` when the machine has it
and is silent under it, windowed and headless both.

On Linux the same target builds with a GLFW window when GLFW 3 is found, and that is the build a user wants: the
window, the keys, the resize, the minimise and the close all behave as the Direct3D viewer's do, on Wayland and on
X11 desktops alike. Which way it reaches the desktop depends on the distribution's GLFW: a 3.4 (Fedora, Arch,
Debian 13) speaks Wayland and X11 both and chooses at run time; a 3.3 (Ubuntu 22.04 and 24.04, Debian 12, Mint,
Pop!_OS) is built for X11 and runs on a Wayland desktop through Xwayland, which every desktop ships -- installing the
distribution's `libglfw3-wayland` in its place gives a native Wayland window there. Without GLFW CMake says `GLFW was not found
(libglfw3-dev, glfw-devel or glfw): nntc_view_vk builds headless, so --shot runs and the window does not`, and only
the headless `--shot` runs. The Windows build stays on its native Win32 window and needs no GLFW.

### The Linux packages

| | packages |
| --- | --- |
| Debian, Ubuntu, Mint, Pop!_OS | `build-essential cmake pkg-config libvulkan-dev libshaderc-dev libglfw3-dev vulkan-tools`, plus the driver: `mesa-vulkan-drivers` for AMD and Intel, or the NVIDIA driver package (`nvidia-driver-NNN`, which carries its own Vulkan driver) (and `vulkan-validationlayers` for a Debug build) |
| Fedora | `gcc-c++ cmake pkgconf vulkan-loader-devel vulkan-headers libshaderc-devel glfw-devel vulkan-tools`, plus the driver: `mesa-vulkan-drivers` for AMD and Intel, or the NVIDIA driver from RPM Fusion (and `vulkan-validation-layers`) |
| Arch | `base-devel cmake vulkan-icd-loader vulkan-headers shaderc glfw vulkan-tools`, plus the driver for the part -- `vulkan-radeon`, `vulkan-intel` or `nvidia-utils` (and `vulkan-validation-layers`) |
| openSUSE | `gcc-c++ cmake pkgconf-pkg-config vulkan-devel vulkan-headers shaderc-devel libglfw-devel vulkan-tools`, plus the driver: `libvulkan_radeon` or `libvulkan_intel`, or the NVIDIA driver (and `vulkan-validationlayers`) |

The Debian and Ubuntu rows were used on real machines; the Fedora, Arch and openSUSE names are from those
distributions' package indexes and have not been typed on a box here, so if one is wrong the package search of
that distribution (`dnf search shaderc`, `pacman -Ss glfw`, `zypper se glfw`) is the fix.

**Which shaderc.** On Linux the build prefers the distribution's SHARED `libshaderc` over `libshaderc_combined.a`,
because the combined archive is not self-contained on every release: Ubuntu 24.04's `libshaderc-dev` ships a 242 KB
one that defines neither `spvValidatorOptionsDestroy` nor `spvtools::Optimizer`, and a link against it alone fails on
both. The shared library names its dependencies itself and links everywhere. The combined archive remains the
fallback -- a LunarG SDK install with no shared library on the path -- and is what the Windows build uses. The
configure prints which of the two it chose. Without either, the target is skipped and the message says so.

### Vulkan headers, and the compile-time state of the cooperative-vector path

The floor is **header 204**: Vulkan 1.3 core, which is what `dynamicRendering` and `synchronization2` need and what
every frame here is recorded with. Anything older stops at a `#error` naming the package to install.

`VK_NV_cooperative_vector` first appears in **Vulkan-Headers 1.4.307**, and a distribution's stock headers are
routinely older -- Ubuntu 22.04 ships 204, Debian 12 ships 239, Ubuntu 24.04 ships 275. Against those, the whole
cooperative-vector path is compiled out behind the header's own `VK_NV_cooperative_vector` macro: its shader is never
compiled, its descriptor binding is never declared, and the start-up line reads

```
cooperative vectors: not available - this build's Vulkan headers (version 275, from VK_HEADER_VERSION) predate
VK_NV_cooperative_vector, which arrived in 1.4.307; the plain path is used
```

which is the same shape of refusal, in the same one line, as a device that does not offer the extension. `--coopvec 1`
is refused with an `ERROR` and exit 1 exactly as it is there, `--bench` of that path with it, and the key K says so
and stays plain. The configure says the same thing earlier, as a CMake Warning, so the absence is readable before the
build runs:

```
CMake Warning at CMakeLists.txt (message):
  nntc_view_vk: Vulkan headers 1.x.275 in /usr/include - cooperative vectors are compiled OUT
  (VK_NV_cooperative_vector needs 1.4.307 or newer). The viewer still builds and runs: it decodes with its
  standard shader, which is the default on every GPU; the extension only makes the decode faster on recent
  NVIDIA cards, so on AMD and Intel nothing is missing. For that faster path on an NVIDIA card under Linux,
  build against newer headers: the LunarG SDK (source its setup-env.sh before cmake), or a distribution
  whose vulkan-headers package is 1.4.307 or newer (Fedora 42, Arch, Debian 13)
```

This is not a reduced build. The plain GLSL decode is the product; every claim this tree makes about the format is a
claim about that path, and on AMD and Intel parts the extension does not exist at all, so it is what runs there
however new the headers are. To compile the path in, install the LunarG SDK or a distribution that ships 1.4.307 or
newer and point the configure at those headers (`find_package(Vulkan)` honours `VULKAN_SDK`; `-DVulkan_INCLUDE_DIR=`
names one that is not on the path). Whether it then runs is still the device's own four-part query, unchanged.

`../tests/run_checks.py` now runs there too: every arm that needs no Direct3D frame -- the five identical launches, the
six `--tex` frames, the older format string, the refusals, the flag toggles, the overlay against `--nooverlay` and the
broken shader -- runs wherever this viewer was built, and the cross-viewer pairs report skipped with that reason.

## Running on other GPUs

Everything above was written and measured on one machine. This section says what the viewer actually asks of a device,
so that a reader with different hardware knows in advance whether it will run and, if it refuses, which refusal they
are looking at.

**What it needs**

* **Vulkan 1.3**, from the loader and from the device. `dynamicRendering` and `synchronization2` are what the frame is
  recorded with; both are core since January 2022 and are asked for by name rather than assumed;
* **the BC formats**, *for a block-compressed asset only*. The encoder's default level 0 is a BC4 / BC5 pair, and a
  device without `textureCompressionBC` cannot sample it. That is not the end of the road: the same material encoded
  with `--l0 palette --bits0 N --bc0 0` (or `--bc0 both`, which writes both twins) has an uncompressed level 0 that any
  device samples, and the viewer's refusal says exactly that;
* **a queue family that both draws and presents**, in one family. The viewer takes a single queue; a device that
  separated the two would need a second one and is refused rather than half-supported;
* **anisotropy is optional**. A missing `samplerAnisotropy` is one `note:` line and the key `X` is then permanently
  off;
* **no `maxSamplerLodBias` requirement at all.** Level 1's LOD shift is applied to the UV gradients handed to
  `textureGrad`, not to a sampler, so no sampler here carries a bias and the limit does not apply. What is checked
  instead is that the descriptor's `lod_bias_level1` lies in `[0, 8]` -- the range a mip shift can mean -- refused by
  name at load. Until v1.1.21 this viewer used a sampler LOD bias of +2 on the second latent instead; it was switched because of mipmap LOD calculation differences on Intel parts against AMD and NVIDIA ones. A sampler LOD bias of the same value is equivalent on hardware that keeps a negative base LOD under magnification (NVIDIA, AMD), but was observed to blur a magnified picture badly on Intel Xe integrated graphics, so the gradient form is the one to ship.

Nothing here is an extension: `VK_KHR_swapchain` is the only one the windowed path enables, and the
cooperative-vector path is a separate, optional, one-vendor addition that the last section of this file describes.

**Which parts these rules pass and fail**

* **AMD and Intel**, on current drivers, meet all of it: Vulkan 1.3 with the BC formats and anisotropy;
* **older Intel integrated parts -- Gen7.5 and Gen8, Haswell and Broadwell -- stop at Vulkan 1.1 on the Windows
  driver** and are refused by name, with the version they offered printed beside the device's name;
* **Adreno on Android, on Qualcomm's proprietary driver, likewise stops at 1.1** and is refused the same way;
* **Adreno on Windows-on-Arm works** -- the native Qualcomm ICD reports Vulkan 1.3 with the BC formats -- **but only
  through that ICD**. Microsoft's Vulkan compatibility pack is Mesa's Dozen layer over Direct3D 12 and reports Vulkan
  **1.0**, so the viewer refuses under it while the same machine's native driver runs it. The refusal names the
  version, which is how the two are told apart;
* the runtime shader compile runs `shaderc` **with no optimisation pass**, which is also what the Qualcomm Windows
  compiler currently needs; the GLSL is compiled as written.

**The refusals a user can meet**, all of them a line beginning `ERROR` on stderr and exit code 1:

| what happened | what the line says |
|---|---|
| the loader is below 1.3 | `no Vulkan device`, the version the **loader** offered, and to update it (the graphics driver installs it) |
| nothing enumerated, or no device is 1.3 | `no Vulkan device`, and what was wanted of them |
| devices draw but none presents to this window | that, in those words -- **not** `no Vulkan device`, because `--shot` renders headless on the same device |
| a device reports 1.3 without `dynamicRendering` or `synchronization2` | which of the two is missing, by name |
| a BC asset, no `textureCompressionBC` | the descriptor's format by name (`BC4_UNORM_BLOCK` / `BC5_UNORM_BLOCK`) and the `--l0 palette --bits0 N --bc0 0` encode that fixes it -- **before any `.dds` is opened** |
| a format the device will not filter with linear taps | the format by name, and whether `textureCompressionBC` was the reason |
| no depth attachment format at all | the three that were tried |
| the surface offers no non-sRGB format | that, and every format it did offer |

An uncompressed level 0 on a device with no BC is not a refusal at all: the load-time pack behind the key `4` is the
only thing that needs BC there, so it is skipped with one `note:` line and the key has nothing to bind.

`no Vulkan device` is the one phrase the release gate reads to **skip** its Vulkan arms rather than fail them, which is
why it belongs to the "this machine cannot run it at all" cases and to nothing else.

**Linux**: `--shot` runs anywhere the target builds, and the shaders are found through `/proc/self/exe`, so a copy on
the `PATH` still reads the two files beside itself. The window needs a desktop session; a build without GLFW, or
a session with neither `DISPLAY` nor `WAYLAND_DISPLAY`, refuses the window by name and still runs `--shot`.

## Is it the same picture?

`../tools/frame_diff.py` answers that as a number rather than as an opinion: it reads the two `.bmp` files and reports
the largest and the mean absolute difference per channel and the PSNR between them.

The `--noaniso` in it is not decoration: anisotropic tap placement is one of the things the plan's section 1.14 says is
free to differ between two APIs, so a comparison that leaves anisotropy on is measuring that rather than the decode.
The command as printed passes.

```
build\Release\nntc_view.exe    ASSET_nntc.json --nooverlay --noaniso --shot a.bmp --tex 0
build\Release\nntc_view_vk.exe ASSET_nntc.json --nooverlay --noaniso --shot b.bmp --tex 0 --size 2560 1421
python tools\frame_diff.py a.bmp b.bmp --max-diff 2 --min-psnr 40
```

The numbers below are this machine's, both viewers on the same RTX 5090 unless the integrated part is named, and each
one is a run of the two commands above with the camera or the flag it names rather than a recollection. They are all
of the trilinear default, because that is the filter a `--shot` can be asked for; point and bilinear are keys, so
what is claimed of them is only what can be seen by eye.

* **With anisotropy off the two viewers' frames are byte-identical, in every state a `--shot` can be put in**: a single
  image, a four-texture material at each of its four textures, and a two-file level 0; the default camera and `--z -50`,
  where mips and level 1's LOD shift are live; `--nomips` and `--nobias`; `--raw0`, `--raw1`, `--cube`, and the
  load-time BC pack of an uncompressed level 0, which both viewers make from the same `../shared/bc_pack.h`.
* **With anisotropy on (the default) they part company only where the taps are placed.** On the four-texture material
  that is a largest difference of 46 to 49 out of 255 on texture 0 at a PSNR of 49.85 dB (6, 21 to 23 and 24 on the
  other three, at 61.9, 51.4 and 55.0 dB), over a small fraction of the quad's pixels; on the 64x64 single image at
  `--z -50` it reaches 149 on a few silhouette pixels at 64.34 dB. Both disappear entirely under `--noaniso`.
* **`--renorm` under anisotropy magnifies that**, as it must: the renormalise divides by a length, so a tap difference
  in one channel moves all three. Up to 75 at 48.56 dB on the material's texture 0, where the same camera without
  `--renorm` gives 49 at 49.85 dB; on the 64x64 image at the default camera it is 1, at 118.5 dB.
* **On the integrated Radeon (`--device 1`)** the difference is a different vendor's bilinear weight rounding rather
  than this program: 4 to 9 out of 255 at 65.5 to 66.5 dB in the default view (1 at 72.6 dB on the material's texture
  1), and up to 63 at 59.3 dB under `--renorm`.

`../tests/run_checks.py` runs that comparison on the assets it builds itself, and drives this viewer through the same
cases it drives the other one through -- the six `--tex` frames of a six-texture material, the older format string, the
five refusals and its own four on `--size` and `--device`, `--bc` both ways round, `--raw0` / `--raw1` / `--renorm` /
`--cube` compared against the Direct3D frame as well as asserted to change this one, the overlay (whose 84 rows must be
byte-identical between the viewers, with the rows below them equal to each viewer's own `--nooverlay` frame), a broken
`view.frag` beside the executable, and five identical headless launches -- skipping whatever needs a device, or a
Direct3D frame, rather than failing there.

## Cooperative vectors

**This is stage 4, and it is the only part of this viewer that is optional at run time.** Everything above runs on any
reasonable Vulkan 1.3 device from any vendor and is the product; this is a **second pipeline**, built only where a
device query passes, and on every other device the viewer says one line about why not and never mentions the extension
again. Nothing above changes when it is on: the same textures, the same samplers, the same uniform block, the same
feature vector.

### What it is

`VK_NV_cooperative_vector` puts a matrix-vector multiply in the shader: the vector lives in the invocation, the matrix
lives in a buffer the shader addresses, and one instruction multiplies them and adds a bias, with the implementation
free to gather a subgroup's worth of them behind the scenes and run the whole thing on the tensor cores. This format's
decode is *exactly* that and nothing else --

```
phi = [ z1_0 .. z1_(C1-1) , z0_0 .. z0_(C0-1) , z0_i * z1_j ]     nin  = C1 + C0 + C0*C1, up to 24
out = W phi + b                                                   nout = 3T,              up to 18
```

-- one layer, no activation, no hidden layer, so it maps onto one `coopVecMatMulAddNV` call with M = `nout` and
K = `nin`. The shading language needs M and K to be compile-time constants and this format's are per-asset, so the
matrix is **always the padded 18 by 24**: zero rows past `nout`, zero columns past `nin`, and `phi` zero past `nin`. A
zero row contributes an output that is already ignored and a zero column multiplies a zero feature, so the padded
product is the unpadded one in exact arithmetic -- and every measurement below is of one shape.

It is NVIDIA-only today, it has no Khronos successor, and the Direct3D 12 preview of the same idea was withdrawn before
Shader Model 6.9 shipped. That is why the path is permanently optional rather than the decode.

### The query, and what this machine answered

Four things are checked before a line of the second pipeline is built, and none of them is a formality:

1. `VK_NV_cooperative_vector` is in the device's extension list;
2. `VkPhysicalDeviceCooperativeVectorFeaturesNV::cooperativeVector` is on (the training feature is never asked for);
3. `VkPhysicalDeviceCooperativeVectorPropertiesNV::cooperativeVectorSupportedStages` contains the **fragment** stage.
   The language permits every stage "subject to api-specific limitations" with no compile-time check, so a driver that
   offered the extension for compute alone would compile the shader and fail at pipeline creation;
4. `vkGetPhysicalDeviceCooperativeVectorPropertiesNV` enumerates a type tuple this viewer can use. There are no size
   fields in that struct -- `maxCooperativeVectorComponents` covers the sizes, and 24 is checked against it.

A fifth is this program's own: glslang compiles a cooperative-vector module with `OpCapability VulkanMemoryModel`, so
the device must offer `vulkanMemoryModel`, and the all-fp16 form below also needs `shaderFloat16`.

**The tuple is the one finding of this stage worth reading twice.** The plan asked for fp16 weights with an **fp32**
accumulation, on the argument that a 32-bit result keeps the accumulation exact and confines the whole difference from
the plain path to the 11-bit mantissa of the weights. That tuple is asked for first and **this driver does not have
it**: on the RTX 5090, driver 581.80, `VK_NV_cooperative_vector` revision 4, the sixteen enumerated tuples contain no
fp32 result type at all. The only floating-point one is

```
input fp16 as fp16 x matrix fp16 + bias fp16 -> fp16
```

(the others are the 8-bit integer and `E4M3` / `E5M2` paths, whose matrices this format does not have: `W` is fp32 in
the asset and quantising it would be a format change made on the viewer's side of the fence). So the viewer takes that
one, says so at start-up, and the numbers below are of the all-fp16 form. The fp32 tuple is still tried first and still
preferred, in one preference list in `main.cpp`, so a driver that grows it gets it with no change here.

**The driver is conforming, and the plan's expectation was the thing at fault.** The fp32-result tuple was never in
the extension's mandated set: the six rows an implementation must enumerate are fp16 x fp16 -> fp16, three sint8
input / sint32 result rows, and two `E4M3` / `E5M2` rows. A driver that offers no fp32 result at all therefore breaks
nothing, and the tuple this viewer runs on is **the first of those mandated rows** rather than a fallback in any
meaningful sense. The consequence for reading this file: the `NNTC_CV_FP32` branch of `bin/view_coopvec.frag` is
**compiled-only**. No device here has executed it, so what is claimed of it is that it compiles and that the viewer
would select it -- never anything about a frame it drew.

The other half of why the two paths differ at all is in the asset and not in the tuple: some of these weights are
**subnormal in fp16**. On `pavingstones141_1k_c0_4` the smallest nonzero |W| is 3.24e-5, under fp16's smallest normal
value of 6.10e-5, so it is carried with fewer than eleven mantissa bits rather than with all of them. That and the
accumulation order are between them the unit of 255 in the table further down.

On device 0:

```
cooperative vectors: available - VK_NV_cooperative_vector revision 4, the fragment stage, 512 components,
  input fp16 x matrix fp16 + bias fp16 -> fp16 (this device enumerates no fp32 result, so the accumulation is the
  implementation's own and not fp32) (16 tuples enumerated)
decode: cooperative vectors
```

and on device 1, the integrated Radeon:

```
cooperative vectors: not available - the device does not offer VK_NV_cooperative_vector
decode: plain
```

### The weights

At load, once: `W` is written into a padded 18-by-24 row-major fp32 matrix, and
`vkConvertCooperativeVectorMatrixNV` is called **twice** -- first with a null destination to learn the size the driver
wants, then to fill it. The one call does the layout change and the fp32 -> fp16 conversion together, and it has to:
the inferencing-optimal layouts are implementation-defined and nothing else may produce one. On this device the 18x24
matrix converts to **1152 bytes**.

The converted bytes go at offset 0 of a storage buffer and the bias at the next 64-byte boundary after them, which is
the extension's rule and not a preference (the matrix offset must be 64-byte aligned, the bias offset 16-byte, and both
requirements apply to the base of the buffer too). The shader gets the bias offset as a **specialization constant**,
because how large the driver's own layout turned out to be is the driver's business. Nothing is transposed anywhere:
the format publishes `W` row-major and the extension's row-major semantics are the same order.

### The shader

`bin/view_coopvec.frag` is `bin/view.frag` with one thing changed. The two texture fetches, the dequantisation after
sampling, the feature order, the clamp, the texture selection, `--raw0` / `--raw1` and the renormalisation are that
file's line for line, because the feature order and the dequantisation rule are the format's contract and the two
decode paths must not be two readings of it. What differs is the middle: where `view.frag` writes the product as two
nested loops over the uniform block's `W`, this file hands it to one `coopVecMatMulAddNV`.

It is one file and not two variants: the input, bias and result types come in as the `NNTC_CV_FP32` macro that the
viewer defines from the tuple its query found, so a device with the fp32 tuple and a device with the fp16 one run the
same source. Key `R` reloads it along with the other two.

### Using it

| | |
|---|---|
| `--coopvec 1` | the cooperative-vector path. On a device without the extension this is **refused** with an `ERROR` and exit 1 -- a run that quietly fell back would be a measurement of the other path under this one's name |
| `--coopvec 0` | the plain path. Always honoured, everywhere |
| neither | cooperative vectors **on** where the device has them, plain everywhere else |
| key `K` | switches at run time, printing `decode: cooperative vectors` or `decode: plain` |
| the overlay | `Coop:ON`, `Coop:OFF` or `Coop:n-a` at the right-hand end of the state line. The strip is 2560 pixels wide at 1:1; in a narrower window it is drawn scaled down to the window's width, so the token and the four lines stay whole and smaller |
| `--bench N` | N frames rendered headless, with no window and no file written, timed on the **GPU** with `vkCmdWriteTimestamp2` either side of the scene draw. It times whichever path is current, so a comparison is two runs that differ in `--coopvec` alone |

`--bench` exists because wall-clock frame time would measure the wrong thing: this viewer's frame loop waits on the
device at the end of every frame by design (the plan's section 1.12), so a host clock times the wait. `--shot` and
`--bench` together are refused: each owns what its run prints.

### What it measures

**The timings.** RTX 5090, driver 581.80, `--size 2560 1440`, 200 frames, the default camera, `--nooverlay`. Every
number is a run of the command beside it and none is estimated:

```
build\Release\nntc_view_vk.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --bench 200 --size 2560 1440 --coopvec 1
build\Release\nntc_view_vk.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --bench 200 --size 2560 1440 --coopvec 0
build\Release\nntc_view_vk.exe examples\m1_m4_c0_3_c1_4_nntc.json        --nooverlay --bench 200 --size 2560 1440 --coopvec 1
build\Release\nntc_view_vk.exe examples\m1_m4_c0_3_c1_4_nntc.json        --nooverlay --bench 200 --size 2560 1440 --coopvec 0
```

| asset | shape | cooperative vectors | plain | |
|---|---|---|---|---|
| `pavingstones141_1k_c0_4` | nin 24, nout 15, five textures | mean **12.0 us**, min 11.6 us | mean **63.7 us**, min 62.3 us | 5.3x |
| `m1_m4_c0_3_c1_4` | nin 19, nout 12, four textures | mean **10.7 us**, min 10.2 us | mean **42.9 us**, min 41.7 us | 4.0x |

(The paving-stones figures repeat to 12.0 / 63.9 on a second run of the same command, so the third digit is the machine
and not the method.)

**That is far more than the plan expected, and the reason matters more than the number.** The plan's section 3.7
predicted "at best 2x on the decode arithmetic alone, and quite possibly no measurable change in frame time at all",
reasoning from 432 multiply-accumulates being a tiny load for a tensor core. What the measurement is actually of is
**these two shaders**, and the plain one is not a fast path: its inner loop indexes a uniform array with a running
index whose bounds (`nin`, `nout`) are themselves uniforms, so it cannot be unrolled and every term is a dynamically
indexed constant-buffer load. So read the table as "one instruction beats this loop by 4 to 5x at 3.7 megapixels",
which is what a viewer that decodes this format actually pays, and not as a claim about the instruction in isolation.
A hand-optimised plain path -- a fixed 18x24 unrolled with the weights in a `std430` buffer, say -- would narrow it,
and would also no longer be the transliteration of `nntc_view.hlsl` that stage 3's parity rests on.

**The precision.** Both example assets, the default camera, `--noaniso` so that the sampler is identical and the fp16
weights are the only difference, `--nooverlay`, `--size 2560 1440`:

```
build\Release\nntc_view_vk.exe ASSET_nntc.json --nooverlay --noaniso --shot on.bmp  --size 2560 1440 --coopvec 1
build\Release\nntc_view_vk.exe ASSET_nntc.json --nooverlay --noaniso --shot off.bmp --size 2560 1440 --coopvec 0
python tools\frame_diff.py on.bmp off.bmp
```

| asset | max abs diff | mean abs diff (r / g / b) | PSNR |
|---|---|---|---|
| `pavingstones141_1k_c0_4` | **1** of 255 | 0.0024 / 0.0017 / 0.0026 | **74.62 dB** |
| `m1_m4_c0_3_c1_4` | **1** of 255 | 0.0029 / 0.0054 / 0.0025 | **72.52 dB** |

The plan's section 3.8 stated the expectation in advance -- "a max difference of 1, occasionally 2, out of 255" -- and
that is what happened, on the all-fp16 tuple rather than on the fp32-accumulate one it was written for. A larger
difference would have been a bug to find and not a tolerance to widen: an fp16 conversion error, a layout mismatch, a
wrong M or K and a padding bug all look like "a bit off".

### What the gate checks

`../tests/run_checks.py` runs four cooperative-vector arms on Windows, and **skips them with the device's own reason**
when the query did not pass, exactly as it skips the whole Vulkan case on a machine with no driver:

* the two paths' `--shot` frames of `examples/m1_m4_c0_3_c1_4_nntc.json` with `--noaniso` differ by at most **4** out
  of 255 at a PSNR of at least **50** (measured: 1 and 71.36 dB at the gate's 640x480), with each run's own start-up
  line asserted to name the path it took, so the comparison cannot be one path against itself;
* `--coopvec 1` on a device without the extension exits 1 with an `ERROR` line -- on this machine, device 1;
* `--bench 20` runs on both paths and prints a mean and a minimum for each;
* the overlay strip at 2560x1440 is byte-identical between the two paths **left of the token's column** and differs
  over the whole width, which is the token and nothing else.

Every arm that was written for stages 1 to 3 now names `--coopvec 0` explicitly, because those arms are about the
baseline path and the default became the other one on this machine. That is also why the cross-viewer comparison of the
overlay is now made over the columns the two strips share: the Direct3D viewer has no decode-path state and cannot grow
one, so its strip has no token and this one's is drawn past the end of everything the four shared lines print.
