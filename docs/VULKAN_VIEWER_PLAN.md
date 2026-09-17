# VULKAN_VIEWER_PLAN.md -- a second viewer, on Vulkan

A design for `viewer_vk/`, a sibling of the Direct3D 11 viewer that loads the **same** asset (`docs/FORMAT.md`) and
draws the **same** picture through a Vulkan sampler. It is a plan, not code.

**The hard constraint, first.** Nothing in the encoder, the asset format or `viewer/main.cpp` changes. The new viewer
is an additional CMake target that reads `PREFIX_nntc.json` and the `.dds` files it names with the same two headers
(`src/nntc_json.h`, `src/json.h`) and the same `src/bc_pack.h`. If a line of this plan ever requires a change on the
other side of that line, the plan is wrong and the plan gives way.

**Why a second viewer at all.** Three reasons, in order of weight:

1. the format's claim is that the representation is valid under *the* hardware sampling operator, not under one
   vendor's. A second graphics API on the same asset is the cheapest evidence there is that the claim is about
   hardware and not about Direct3D;
2. Vulkan is the portable path to Linux, where the encoder already builds and where nothing can currently be looked
   at;
3. the cooperative-vector extension (section 3) exists on Vulkan first, and our decode -- one matrix-vector multiply of
   at most 18 by 24 per pixel -- is exactly the shape that extension addresses. There is no way to try it on Direct3D
   11 at all.

---

## 1. Recommended architecture

**Where the shared headers live, since v1.1.0.** This section was written before the shared directory of amendment 3
existed, so it names `src/json.h`, `src/nntc_json.h` and `src/bc_pack.h` throughout. Every one of them is `shared/` now
-- `shared/json.h`, `shared/nntc_json.h`, `shared/bc_pack.h`, and `shared/dds.h`, which is the `.dds` parsing lifted out
of the Direct3D viewer -- and both viewers and the encoder include from there. The names are left as they were written
rather than rewritten, so that the plan still reads as the plan it was; this note is the correction.

### 1.1 What the D3D11 viewer does, as a checklist

Everything below is what `viewer/main.cpp` (1112 lines) and `viewer/bin/nntc_view.hlsl` (94 lines) do today. The
Vulkan viewer's stage 3 is done when every row is ticked.

| piece | D3D11 today | Vulkan equivalent |
|---|---|---|
| window | raw Win32, `WS_OVERLAPPEDWINDOW`, client rect read back after create | the same code, verbatim |
| device | `D3D11CreateDeviceAndSwapChain`, feature level 11_0 | instance, physical-device pick, one graphics+present queue, device |
| swap chain | 2 buffers, `R8G8B8A8_UNORM`, flip-discard | `VK_KHR_swapchain`, `B8G8R8A8_UNORM`, `FIFO` present mode |
| depth | `D24_UNORM_S8_UINT` | `D32_SFLOAT` |
| asset load | `load_dds` -> immutable mipped texture + view | staging buffer -> `vkCmdCopyBufferToImage` per mip -> `SHADER_READ_ONLY_OPTIMAL` |
| formats | 61 / 49 / 28 / 80 / 83 | `R8_UNORM`, `R8G8_UNORM`, `R8G8B8A8_UNORM`, `BC4_UNORM_BLOCK`, `BC5_UNORM_BLOCK` |
| load-time pack | `src/bc_pack.h`, key 4 | the same header, unchanged |
| samplers | `[mips on/off][point, bilinear, trilinear, aniso]`, twice (level 1 carries `MipLODBias`) | 16 `VkSampler` objects, created once, the same table |
| constants | `SceneConstants` 128 B (b0) + `DecoderConstants` 1904 B (b1) | one `std140` uniform buffer, 2032 B, one binding |
| textures bound | t0 level 0, t1 level 1, t2 level 0's channels 2-3 | three sampled images + two samplers, separate descriptors |
| geometry | quad and cube vertex/index buffers, pos `float3` + uv `float2` | one host-visible buffer holding both, same 20-byte stride |
| shader | runtime `D3DCompileFromFile`, key R reloads | runtime `shaderc` compile of a `.glsl`/`.vert`/`.frag` pair, key R reloads |
| overlay | 8x8 font rasterised on the cpu into a dynamic texture, alpha-blended quad | the same rasteriser, staged into an image, second pipeline with blending |
| `--shot` | copy back buffer to a staging texture, write 24-bit bottom-up bmp | render offscreen (no surface at all), `vkCmdCopyImageToBuffer`, same bmp writer |
| flags | 14 of them, `--help` prints all | the same fourteen in the same spellings, plus `--size` and `--device` |

Two behaviours are load-bearing and must survive: **`--shot` reads no keyboard state at all** (the reason the release
gate's five identical shots are identical), and **level 1's sampler carries `mipLodBias = lod_bias_level1`**, which is
the format's section 5 and not a preference.

### 1.2 The recommendation in one paragraph

Raw Win32 window; **Vulkan 1.3 core** with **dynamic rendering** and **synchronisation2** and no extension beyond
`VK_KHR_surface` + `VK_KHR_win32_surface` + `VK_KHR_swapchain`; the loader linked normally as `vulkan-1.lib` through
CMake's own `FindVulkan` (**no volk**); shaders written as **GLSL** and compiled **at runtime** by `shaderc_combined`,
which the Vulkan SDK already ships, so key R works exactly as it does today and the executable still depends on
nothing but the loader; a **single uniform buffer** for both constant blocks, because 2032 bytes will not fit in any
guaranteed push-constant budget; **one frame in flight with a device wait at the end of each frame**, so every
descriptor, buffer and image can be updated in place with no per-frame duplication; and **no vendored third-party file
at all**. The only new source files are one `main.cpp`, two shader files and a `README.md`.

### 1.3 Dependencies, and why the list is this short

| dependency | where it comes from | licence | why |
|---|---|---|---|
| Vulkan loader + headers | the Vulkan SDK, found by CMake's built-in `FindVulkan` | Apache 2.0 | unavoidable; it is the api |
| `shaderc_combined` | the same SDK, `FindVulkan` component `shaderc_combined` | Apache 2.0 | runtime glsl -> spir-v so key R works. Static, so the built exe still needs only the loader at runtime |
| `src/json.h` | already in this tree | public domain | the descriptor |
| `src/nntc_json.h`, `src/bc_pack.h` | this tree's own | this tree's | the descriptor wrapper and the load-time pack |
| Win32, the c runtime | the platform | -- | window, files, printing |

**Nothing is vendored.** That is the whole point of the choices below: every candidate that would have added a file to
this repository was rejected in favour of something the Vulkan SDK already installs, or of code we write ourselves in
a few dozen lines. The SDK on Windows ships the loader, the headers, the validation layers, `glslang`, `shaderc`,
`volk` and `dxc` (Vulkan SDK 1.4.357.0 release notes,
https://vulkan.lunarg.com/doc/view/latest/windows/release_notes.html), and CMake's `FindVulkan` module knows how to
find every one of them by component name, including `glslc`, `glslangValidator`, `shaderc_combined`, `volk` and `dxc`
(https://cmake.org/cmake/help/latest/module/FindVulkan.html). So the SDK is a *build* dependency the same way the CUDA
toolkit already is, and the shipped executable's runtime dependency is `vulkan-1.dll` alone.

### 1.4 File list

```
viewer_vk/main.cpp              the whole viewer: window, device, load, samplers, frame loop, overlay, shot
viewer_vk/bin/view.vert         vertex shader (glsl), copied beside the exe by the build, reloadable
viewer_vk/bin/view.frag         fragment shader (glsl): the decode, the transliteration of nntc_view.hlsl
viewer_vk/README.md             what viewer/README.md is for the other one
```

and in the root `CMakeLists.txt`, a new target guarded by `find_package(Vulkan)` so that a tree without an SDK still
configures and still builds the encoder and the Direct3D viewer. **The `if(WIN32)` in the sketch below is not what was
built**, and amendment 4 is why: the target itself is platform-independent, and only what is genuinely Win32 -- the
manifest, `VK_USE_PLATFORM_WIN32_KHR`, `/W4` and the `/IGNORE:4099` for the SDK's own missing PDBs -- sits inside an
`if(WIN32)`, with `-Wall -Wextra` and the optional xcb in the `else()`:

```cmake
find_package(Vulkan COMPONENTS shaderc_combined)     # quiet, optional
if(Vulkan_FOUND AND TARGET Vulkan::shaderc_combined)
    add_executable(nntc_view_vk viewer_vk/main.cpp src/bc_pack.h src/nntc_json.h src/json.h)
    target_link_libraries(nntc_view_vk PRIVATE Vulkan::Vulkan Vulkan::shaderc_combined)
    # ... the manifest, /W4, and the POST_BUILD copy of the two shader files, exactly as nntc_view has
endif()
```

`main.cpp` will be long -- the D3D11 one is 1112 lines and Vulkan is wordier at the bottom, so expect 1600 to 2000 --
and that is accepted deliberately: one file that can be read top to bottom beats a framework that has to be learned.
The wordiness is nearly all in three functions (device creation, swapchain creation, image upload), each written once
and then never looked at again.

### 1.5 Which Vulkan version, and why 1.3

**Target `VK_API_VERSION_1_3` and refuse anything lower with one clear line.**

* **1.3 makes dynamic rendering core.** `vkCmdBeginRendering` / `vkCmdEndRendering` take the colour and depth image
  views directly, so there is no `VkRenderPass`, no `VkFramebuffer`, no subpass dependency and no attachment-reference
  bookkeeping -- between 120 and 200 lines of pure ceremony deleted, and the ceremony that confuses every reader of a
  Vulkan program for the first time. The Khronos sample that shows precisely this is
  https://docs.vulkan.org/samples/latest/samples/api/hello_triangle_1_3/README.html.
* **1.3 makes synchronisation2 core**, which turns each image-layout transition into one `VkImageMemoryBarrier2` with
  stage and access in the same struct. The `.dds` upload does one of these per image; the frame does two.
* **1.3 makes extended dynamic state core**, so viewport and scissor are set on the command buffer and a window resize
  needs no pipeline rebuild.
* **It is old.** Vulkan 1.3 was released in January 2022 and every Windows desktop driver that matters has shipped it
  for years (https://docs.vulkan.org/guide/latest/versions.html).

We deliberately do **not** target 1.4, even though the SDK is at 1.4.x: 1.4 buys us nothing here (its relevant change
is raising `maxPushConstantsSize` to 256, and we need 2032), and it would exclude older drivers for no gain.

**Everything we need is core 1.3.** The extension list is:

* instance: `VK_KHR_surface`, `VK_KHR_win32_surface` (and on Linux later, `VK_KHR_xcb_surface`);
* device: `VK_KHR_swapchain`, and in stage 4 only, `VK_NV_cooperative_vector` when it is present;
* instance layer, debug builds only: `VK_LAYER_KHRONOS_validation`, with `VK_EXT_debug_utils` for the message
  callback. This is the equivalent of the D3D11 viewer's `D3D11_CREATE_DEVICE_DEBUG`, and like it, a failure to enable
  it is a warning and not a failure.

**Features to request** on the device, all of them core 1.0 booleans:

* `textureCompressionBC` -- required, and the reason is not optional: it is what makes `VK_FORMAT_BC4_UNORM_BLOCK` and
  `VK_FORMAT_BC5_UNORM_BLOCK` guaranteed to support `SAMPLED_IMAGE_BIT` and, critically,
  `SAMPLED_IMAGE_FILTER_LINEAR_BIT` in `optimalTilingFeatures` (the features chapter,
  https://docs.vulkan.org/spec/latest/chapters/features.html). Without linear filtering on a block format the asset
  cannot be sampled the way the encoder fitted it;
* `samplerAnisotropy` -- for key X. If a device lacks it the viewer runs with anisotropy permanently off and says so,
  rather than refusing to start (same chapter);
* and from 1.3's own structs, `dynamicRendering` and `synchronization2`.

Even with `textureCompressionBC` the loader still calls `vkGetPhysicalDeviceFormatProperties` for the exact format
each `.dds` names, and refuses with the format id in the message if the bits are not there. That check costs four
lines and turns "the picture is wrong" into "this device cannot sample this format".

### 1.6 The loader: link it, do not vendor a meta-loader

**Link `vulkan-1.lib` directly** (`Vulkan::Vulkan` from `FindVulkan`). Do not use volk.

volk (https://github.com/zeux/volk, MIT) is a meta-loader: `volk.c` and `volk.h` dynamically load every entry point so
that the program does not link the loader import library at all, and `volkLoadDevice` bypasses the loader's dispatch
for a small speed win. Both of its benefits are things we do not want:

* *no link-time dependency* matters to a library that must start on a machine with no Vulkan. Our viewer is useless
  without a Vulkan driver, so failing at load with "vulkan-1.dll not found" is an honest outcome. (If that message is
  ever judged too blunt, the delayed-load linker flag gives a friendlier one in one line, still with no vendored
  file.)
* *dispatch bypass* is a performance argument, and the owner's brief for this viewer is explicitly not performance.

And it has a cost we do want to avoid: two files vendored into `viewer_vk/`, generated from the registry, that the
release gate would have to be told to skip.

The handful of extension entry points we need that the loader does not export statically -- in stage 4,
`vkGetPhysicalDeviceCooperativeVectorPropertiesNV` and `vkConvertCooperativeVectorMatrixNV` -- are fetched with two
`vkGetInstanceProcAddr` / `vkGetDeviceProcAddr` calls stored in two globals. That is the whole reason volk exists,
and at two entry points it is not a reason.

### 1.7 The window, and the Linux port

**Stage 1 to 3: raw Win32, lifted from `viewer/main.cpp`.** The message loop, `wnd_proc`, `process_held_keys`,
`GetAsyncKeyState`, the client-rect read-back after `CreateWindow` and the `--shot`-takes-no-input rule are all
already written, already debugged, and already the subject of a comment in that file explaining why they are the way
they are. Copying them keeps the two viewers' behaviour identical by construction, which is the point of the exercise.
The surface is then four lines: `VkWin32SurfaceCreateInfoKHR` with the `HINSTANCE` and `HWND`,
`vkCreateWin32SurfaceKHR`.

**Plan for Linux now, but only by drawing one line.** Do not add a windowing library. Instead, keep every platform
call behind six functions declared near the top of `main.cpp`:

```
bool  win_open(int w, int h, const char* title);
void  win_pump(void);                 // messages, and the held-key scan
bool  win_should_quit(void);
void  win_client_size(int* w, int* h);
VkSurfaceKHR win_create_surface(VkInstance);
double win_seconds(void);             // the qpc clock
```

The Win32 bodies go in one `#ifdef _WIN32` block, and a later xcb block sits beside them: `xcb_connect`,
`xcb_create_window`, `VK_KHR_xcb_surface`, and a key-state map maintained from `XCB_KEY_PRESS`/`XCB_KEY_RELEASE`
because xcb has no `GetAsyncKeyState`. That is perhaps 200 lines, links `libxcb` alone, and vendors nothing. The
alternative -- adopting a windowing library so that one file serves both -- costs a vendored dependency now for work
we will not do until Linux matters, and none of the candidates is a single header: they are all real builds. Note also
that `--shot`, the mode the release gate uses, will need **no window at all** (section 1.13), so a Linux box with no
desktop can run the gate's viewer checks before any of that xcb code exists.

### 1.8 Shaders: glsl, compiled at runtime

**Write the shaders in GLSL and compile them at runtime with `shaderc`.**

The requirement that decides this is key R: the owner edits the shader beside the executable and presses R, exactly as
today. Three ways to get that:

1. **`shaderc_combined`, linked statically** -- `shaderc_compile_into_spv` takes a source string, a shader kind and a
   file name and returns a spir-v module or an error string with line numbers. It is the direct analogue of
   `D3DCompileFromFile`, it is one function call, and the SDK already installs it as a static library that CMake finds
   by component name (https://github.com/google/shaderc, Apache 2.0). **Recommended.**
2. **shell out to `glslc.exe`** from the SDK on each reload, then read the `.spv` file. Zero link-time dependency, but
   it needs the SDK present on any machine that reloads a shader, it needs a child process and a temporary file, and
   its error reporting is a captured pipe. A reasonable *fallback* if `shaderc_combined` turns out to be awkward to
   link against MSVC's runtime, and worth keeping in mind as three lines of `CreateProcess`.
3. **glslang as a library** -- what `shaderc` itself wraps, with a much larger and less stable C++ surface. No reason
   to prefer it over the wrapper.

Build-time compilation into embedded spir-v bytes is *also* worth doing, but as a belt rather than as the mechanism: a
`POST_BUILD` copy of the two `.glsl` files beside the executable (exactly what `nntc_view.hlsl` gets today) is enough,
and the runtime compiler reads them from beside the executable. It did start out with the same two-step search
`viewer/main.cpp` implements -- the working directory first, then beside the executable -- and that step was removed in
`v1.1.4-vk-fixes`: a stray `view.frag` in whatever directory the viewer happened to be started from replaced the real
one with no word said, which is a blank frame and an exit code of 0.

**Why GLSL and not the existing HLSL through DXC.** The SDK ships `dxc`, and DXC compiles HLSL to spir-v: `register(bN)`
maps to descriptor binding `N` by default, `-fvk-b-shift` / `-fvk-t-shift` / `-fvk-s-shift` separate the register
spaces that would otherwise collide, and `[[vk::binding]]` overrides both
(https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst). It is genuinely tempting: one shader
file, both viewers, and no transliteration to keep in step. It is rejected for three reasons, and the first is
decisive:

* **it couples the two viewers.** `viewer/bin/nntc_view.hlsl` would acquire a second consumer with its own
  constraints, and the constraint of this whole plan is that the Direct3D side does not move. A Vulkan-driven
  `[[vk::binding]]` annotation in that file is a change to the Direct3D viewer;
* **the constant-buffer layout differs.** DXC's default for a Vulkan uniform buffer is a vector-relaxed std140, not
  DirectX packing; `-fvk-use-dx-layout` gets DirectX packing back but pulls in `VK_EXT_scalar_block_layout`, which is
  an extension we would otherwise not need (same document). Our block is all `vec4`/`ivec4` and arrays of them, so
  under std140 it maps one-to-one anyway -- but only because we would have written it that way, which is the GLSL
  path's argument, not the HLSL path's;
* **clip space differs** (`-fvk-invert-y`), and so the two viewers would still not be running the same text.

The fragment shader is about 60 lines and is a mechanical transliteration: `Texture2D` + `SamplerState` become a
`texture2D` and a `sampler` combined in the shader with `texture(sampler2D(lat0, samp), uv)`, `saturate` becomes
`clamp(x, 0.0, 1.0)`, `float4` becomes `vec4`, `[unroll]` is dropped. The decode arithmetic -- `phi` in the order
`a`, `b`, `sc`, then `W phi + b` -- is copied verbatim, because the feature order is the format's contract
(`docs/FORMAT.md` section 4) and not an implementation detail either viewer is free to restate.

**Clip space.** Vulkan's y axis points down and its depth range is already 0 to 1, which is what
`mat_perspective` in `viewer/main.cpp` produces. So the projection matrix is copied as it stands and the y flip is
done with a **negative viewport height** (`VkViewport.y = height, VkViewport.height = -height`), core since Vulkan 1.1.
One line, no matrix edit, and the two viewers therefore share a camera that can be compared parameter for parameter.

### 1.9 Uploading the `.dds` levels

The loader is the D3D11 `load_dds` with its parsing kept exactly as it is -- the 148-byte header check, the
`1..16384` extent bounds, the mip-count bound, the per-level offset walk and the "bytes expected / bytes present"
check are all format validation and have nothing to do with the api. Only the tail changes.

**Format mapping.** `61 -> VK_FORMAT_R8_UNORM`, `49 -> VK_FORMAT_R8G8_UNORM`, `28 -> VK_FORMAT_R8G8B8A8_UNORM`,
`80 -> VK_FORMAT_BC4_UNORM_BLOCK`, `83 -> VK_FORMAT_BC5_UNORM_BLOCK`. Any other id is the same refusal the Direct3D
loader already prints.

**The upload, per texture:**

1. `vkCreateImage`: `VK_IMAGE_TYPE_2D`, the mapped format, `mipLevels = nmip`, `arrayLayers = 1`,
   `VK_IMAGE_TILING_OPTIMAL`, usage `TRANSFER_DST | SAMPLED`, `initialLayout = UNDEFINED`;
2. allocate device-local memory for it and bind. **Write the four-line allocator once**: walk
   `VkPhysicalDeviceMemoryProperties.memoryTypes`, take the first index whose bit is in
   `VkMemoryRequirements.memoryTypeBits` and whose `propertyFlags` contain what was asked for. This is the whole of
   what a memory allocator library would do for us here: a viewer makes perhaps ten allocations in its life, and
   `maxMemoryAllocationCount` is guaranteed to be at least 4096
   (https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html);
3. one `HOST_VISIBLE | HOST_COHERENT` staging buffer per texture, sized to the whole chain, **with each mip placed at
   a 16-byte-aligned offset** rather than at the file's own tightly packed offset. This matters: `bufferOffset` in
   `VkBufferImageCopy` must be a multiple of 4 and of the format's texel-block byte size, so a BC5 chain (16 bytes a
   block) is fine as it lies in the file but an `R8` chain with an odd width is not. Re-laying the mips out while
   copying into the staging buffer costs one `memcpy` per level and removes the whole class of problem;
4. one `vkCmdCopyBufferToImage` with `nmip` regions: `bufferRowLength = 0` and `bufferImageHeight = 0` (meaning
   "tightly packed", which each level is), `imageSubresource.mipLevel = i`, `imageExtent` the level's own size. For a
   block format the extent is in texels, not blocks, and a final level smaller than 4x4 is legal precisely because it
   is the whole mip;
5. two barriers around it: `UNDEFINED -> TRANSFER_DST_OPTIMAL` before, `TRANSFER_DST_OPTIMAL ->
   SHADER_READ_ONLY_OPTIMAL` after, both as `VkImageMemoryBarrier2` covering all mips;
6. `vkCreateImageView`, `VK_IMAGE_VIEW_TYPE_2D`, all mips, identity swizzle. **Not** an identity swizzle if we wanted
   to imitate Direct3D's BC4 "returns (r, 0, 0, 1)" -- but we do not need to: the shader already reads only `.r` of
   the second texture when `C0` is 3, exactly as the Direct3D shader does, so no channel that is not there is ever
   used.

All of this happens on one transient command buffer from a one-off command pool, submitted and waited on with
`vkQueueWaitIdle` during load. No transfer queue, no async anything.

**The dummy image.** The Direct3D viewer binds a null shader-resource view to `t2` when level 0 is one texture.
Vulkan does not allow a null image view in a descriptor without `VK_EXT_robustness2`'s `nullDescriptor`, so the viewer
creates a **1x1 `R8G8B8A8_UNORM` image** once and binds that in the slot instead. It is never sampled -- `sel.z` tells
the shader the texture count, as it does today -- but it must be a valid view.

**The load-time pack (key 4)** is `src/bc_pack.h` unchanged: it produces block bytes per mip, and those bytes go
through the same upload path into a second set of images. The bc round-trip check and its printed PSNR are the same
code and print the same numbers.

### 1.10 Samplers

Sixteen `VkSampler` objects created once at load, indexed exactly as the Direct3D viewer's
`g_samplers[mips][filter]` and `g_samplers1[mips][filter]`:

| state | `VkSamplerCreateInfo` |
|---|---|
| point | `magFilter = minFilter = NEAREST`, `mipmapMode = NEAREST` |
| bilinear | `LINEAR`, `LINEAR`, `mipmapMode = NEAREST` |
| trilinear | `LINEAR`, `LINEAR`, `mipmapMode = LINEAR` |
| anisotropic | trilinear plus `anisotropyEnable = VK_TRUE`, `maxAnisotropy = 8` |
| mips off (key M) | `maxLod = 0.0f` |
| mips on | `maxLod = VK_LOD_CLAMP_NONE` |
| level 1 (key L on) | any of the above plus `mipLodBias = lod_bias_level1` |

with `addressModeU/V/W = CLAMP_TO_EDGE` throughout, because the encoder fits the taps clamped at the edges.

**One limit to check and to say out loud.** `maxSamplerLodBias` has a *minimum required value of 2*
(https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html), and our bias at `block` 4 is exactly
2.0. We are therefore standing precisely on the floor of what Vulkan guarantees. **This paragraph used to say that
`mipLodBias` is clamped to `[-maxSamplerLodBias, +maxSamplerLodBias]` rather than rejected, and that is wrong**: the
valid-usage rule on `VkSamplerCreateInfo` requires the absolute value to be at most the limit, so a sampler past it is
out of spec - eight validation errors and whatever the driver felt like, not a clamp. So the viewer reads
`limits.maxSamplerLodBias` and REFUSES a descriptor whose `lod_bias_level1` exceeds it in either direction, naming the
value and the limit, before a single sampler is created (`v1.1.4-vk-fixes`). A device at the bare minimum still runs
today's assets; a hypothetical future asset at `block` 8 (bias 3) would be refused by name on such a device rather than
drawn wrongly. `maxSamplerAnisotropy` is guaranteed to be at least 16, so our 8 is safe (same table).

### 1.11 The decoder constants: a uniform buffer, not push constants

The two Direct3D constant buffers are 128 bytes (`SceneConstants`: a `float4x4` and four `float4`) and **1904 bytes**
(`DecoderConstants`: four `float4` of dequantisation, two `int4`, `float4 W[108]` = 1728 bytes, `float4 bias[5]` = 80).
2032 bytes together.

Push constants are out: `maxPushConstantsSize` is guaranteed to be only **128 bytes** in Vulkan 1.0 through 1.3, and
256 in 1.4 (same table). The scene block alone would exactly fill the 128 and leave nothing.

So: **one `VkBuffer` of 2032 bytes**, `HOST_VISIBLE | HOST_COHERENT`, persistently mapped, holding both blocks as one
`std140` uniform block at descriptor binding 0. `maxUniformBufferRange` is guaranteed to be at least 16384, so the
block is comfortably inside every implementation (same table). The C struct maps to std140 with no padding thought
required, because every member is a `vec4`, an `ivec4`, a `mat4` or an array of `vec4` -- std140's array stride for a
`vec4` is 16, which is the array's natural stride. `memcpy` the struct over the mapped pointer once a frame, exactly
as `set_uniforms` does today.

The descriptor set layout is five bindings in set 0:

```
0  UNIFORM_BUFFER   fragment + vertex   the scene and the decoder
1  SAMPLED_IMAGE    fragment            level 0
2  SAMPLED_IMAGE    fragment            level 1
3  SAMPLED_IMAGE    fragment            level 0's channels 2-3, or the 1x1 dummy
4  SAMPLER          fragment            level 0's sampler  (key M, key X, key P/B/T)
5  SAMPLER          fragment            level 1's sampler  (the same, plus key L's bias)
```

Sampled images and samplers are kept **separate** rather than combined, which is what makes the key handling trivial:
pressing L or M or X rewrites one or two sampler descriptors and touches nothing else, and pressing 4 rewrites two
image descriptors. In GLSL that is `texture(sampler2D(lat0, samp0), uv)`, which is ordinary Vulkan GLSL and needs no
extension directive.

### 1.12 The frame, and the overlay

**One frame in flight, and `vkDeviceWaitIdle` at the end of each frame.** This is a deliberate simplification and it
is the right one here: it makes every descriptor write, every uniform update and every texture swap safe with no
double-buffering of anything, and it removes the single most error-prone part of a first Vulkan program. The cost is
that cpu and gpu do not overlap, which for a viewer whose brief says "does not need to be high performance" is not a
cost -- and section 3's measurement uses gpu timestamp queries, which are unaffected by it.

The pieces: one command pool, one command buffer re-recorded each frame; `vkAcquireNextImageKHR` with a per-frame
semaphore; `vkCmdBeginRendering` on the acquired image view plus the depth view, `LOAD_OP_CLEAR` to the same
`{0.2, 0.2, 0.2, 1}`; the scene pipeline and draw; the overlay pipeline and draw; `vkCmdEndRendering`; a barrier to
`PRESENT_SRC_KHR`; submit with a **per-swapchain-image** render-finished semaphore (per-image, not per-frame: a
semaphore signalled for presentation cannot be reused until that image comes back, and getting this wrong is the
classic validation error in exactly this kind of program); `vkQueuePresentKHR`. `VK_ERROR_OUT_OF_DATE_KHR` or
`VK_SUBOPTIMAL_KHR` from acquire or present, and `WM_SIZE`, all funnel into one `recreate_swapchain()`.

Present mode `VK_PRESENT_MODE_FIFO_KHR`, the only one Vulkan guarantees, which also matches the Direct3D viewer's
`Present(1, 0)`.

**The overlay** keeps `g_font8x8` and `blit_char` and `update_debug_text` verbatim -- they produce an `OVL_W x OVL_H`
`R8G8B8A8` buffer on the cpu and nothing about that is api-specific. What changes is the upload: when
`debug_dirty`, `memcpy` into a persistent host-visible staging buffer and `vkCmdCopyBufferToImage` into a
`SHADER_READ_ONLY_OPTIMAL` image at the top of the frame's command buffer (with the two barriers around it). The draw
is a second pipeline over four vertices with `blendEnable`, `SRC_ALPHA` / `ONE_MINUS_SRC_ALPHA`, depth test off, and
its own two-line shader pair compiled from a string constant, exactly as `DEBUG_HLSL` is today.

**Geometry.** Keep the vertex and index buffers; do not generate the quad in the vertex shader. The cube exists too
(key C), the buffers are a few hundred bytes, and a vertex buffer is what a real asset viewer has. One
`HOST_VISIBLE | HOST_COHERENT` buffer with usage `VERTEX_BUFFER | INDEX_BUFFER` holds the quad's vertices, the quad's
indices, the cube's vertices and the cube's indices at four offsets, written once at load; `vkCmdBindVertexBuffers`
and `vkCmdBindIndexBuffer` pick the pair. No staging buffer, no device-local copy, because at this size it does not
matter and the honest simple thing reads better.

### 1.13 `--shot`, and the headless win

The Direct3D `--shot` renders a frame into the back buffer, copies it into a staging texture and writes a 24-bit
bottom-up bmp. Vulkan can do better, and should: **`--shot` creates no window and no surface at all.**

* no `VK_KHR_surface`, no `VK_KHR_win32_surface`, no swapchain, no present queue -- just an instance, a device with a
  graphics queue, and an offscreen `R8G8B8A8_UNORM` image with usage `COLOR_ATTACHMENT | TRANSFER_SRC` at the
  requested size (2560x1440 by default, as today);
  **and that default is why `--size` exists.** 2560x1440 is the size the Direct3D viewer's window is ASKED for; what it
  renders is that window's CLIENT AREA, which Windows clamps to the desktop's work area - 2560x1421 on this machine.
  A headless frame at 2560x1440 and a windowed one at 2560x1421 are two different projections and cannot be compared,
  so the gate takes the Direct3D frame first and asks this viewer for that frame's own size;
* render into it with the same command recording, barrier it to `TRANSFER_SRC_OPTIMAL`,
  `vkCmdCopyImageToBuffer` into a host-visible buffer, `vkQueueWaitIdle`, map, and run the **same bmp writer bytes for
  bytes** -- including the b/g/r swap and the bottom-up row order;
* the frame then depends on the command line and the asset and on nothing else, for a stronger reason than the
  Direct3D viewer's: there is no window to receive a keystroke and no compositor to interact with. The release gate's
  "five launches, five identical files" check becomes trivially true;
* and it runs over a remote desktop session, in a service, and on a Linux box with no display -- which is what makes
  the gate's viewer checks portable before any xcb code is written. **What the gate does there today** (since
  `v1.1.5-vk-gate-docs`): every arm that needs no Direct3D frame runs wherever this viewer was built -- the five
  identical launches, the six `--tex` frames, the older format string, the refusals, the flag toggles, the overlay
  against `--nooverlay`, `--help` and the broken shader -- and the cross-viewer pairs are recorded as skipped, naming
  the Direct3D viewer's absence as the reason.

The interactive path keeps the window and the swapchain. The two share everything except the target image, which is
the one function argument that differs.

One caveat to state plainly: a **headless offscreen frame and a swapchain frame are not guaranteed to be the same
bytes** even on one device, because the swapchain image's format may be `B8G8R8A8` where the offscreen one is
`R8G8B8A8`, and because a compositor may sit in between. The shot path is defined as the offscreen one, and that is
what the gate compares.

### 1.14 Parity with the Direct3D viewer: what can and cannot be asserted

**What can.** Both viewers sample the same textures with the same filter and evaluate the same affine decoder in fp32,
so the two `--shot` frames of the same asset at the same camera should agree to within a small number of least
significant bits. The check to write is the one this tree already uses elsewhere: max `|diff|` per channel and a PSNR
between the two bmps, with a stated threshold.

**What was measured, which is stronger than that expectation and weaker in one place.** On the RTX 5090, with
anisotropy off, the two frames are **byte-identical** in every state a `--shot` can be put in (the trilinear
default, since the other two filters are keys): a single image, a four-texture material at each of its four
textures, and a two-file level 0; the default camera and `--z -50`, where mips and level 1's LOD bias are live;
`--nomips` and `--nobias`; `--raw0`, `--raw1`, `--cube`, and the load-time BC pack of an uncompressed level 0. The
expectation of "1 or 2 out of 255" is therefore not what happens on one GPU on this side of the line: 0 is. With anisotropy ON -- the default, and the one state the section below says is free to differ -- the difference is
46 to 49 out of 255 at 49.85 dB on the material's texture 0 (6, 21 to 23 and 24 on the other three), 149 on a few
silhouette pixels of the 64x64 image at `--z -50` at 64.34 dB, and up to 75 at 48.56 dB when `--renorm` divides those
taps by a length. On the integrated Radeon through `--device 1`, where the sampler is a different vendor's, it is 4 to
9 at 65.5 to 66.5 dB in the default view and up to 63 at 59.3 dB under `--renorm`. `viewer_vk/README.md` carries the
same numbers with the commands that produce them.

**What cannot.** Byte-for-byte equality across two apis is not a reasonable requirement and should not be written into
a gate. The rasteriser's fill rule at triangle edges, the bilinear weight precision (Direct3D and Vulkan both specify
a minimum number of fractional bits, not an exact value), the block-compression palette evaluation (`docs/FORMAT.md`
section 7 and `viewer/README.md` already record that one NVIDIA gpu evaluates the BC4 interior with 6-bit weights
rather than `k/7`), and anisotropic tap placement are all free to differ.

The Vulkan viewer's stronger equality claim is **internal**: its plain path and its cooperative-vector path (section
3) are the same picture from the same textures, and that comparison is worth making byte-exact-or-explained.

---

## 2. The stage plan

Four stages, each ending somewhere the work can be left. **All four are built**, at the tags named below.

### Stage 1 -- a window and a clear colour  *(built, tag v1.1.1-vk-stage1)*

Instance (1.3, validation in debug), physical device pick (first discrete gpu with a graphics+present queue that
supports the surface, else the first that does), device and queue, Win32 surface, swapchain, depth image, command
pool and buffer, semaphores and a fence, the frame loop with `vkCmdBeginRendering` clearing to
`{0.2, 0.2, 0.2, 1}`, `WM_SIZE` -> `recreate_swapchain`, Esc quits.

*Done when* the window clears, resizes without a validation error, and closes clean; and `--shot out.bmp` writes a
2560x1440 grey bmp **with no window created**.

Approximately 700 lines. This is the stage where nearly all of Vulkan's ceremony is paid for.

Stage 1 also does the two housekeeping items of section 5: the `shared/` directory is created and both viewers
include from it, and `tests/run_checks.py`'s `SEARCHED_DIRS` gains `viewer_vk` and `shared`, so the gate's tree scan
polices the new code from the first commit.

### Stage 2 -- the quad, the two textures, the decode  *(built, tag v1.1.2-vk-stage2)*

The `.dds` loader (the D3D11 parser, the Vulkan tail), the descriptor set layout and pool, the uniform buffer, the
samplers, the vertex and index buffer, the two pipelines, the two shader files and the `shaderc` compile, the JSON
load through `src/nntc_json.h`, and the fragment shader's transliterated decode.

*Done when* `nntc_view_vk asset_nntc.json` shows the decoded image on the quad and it looks like what
`nntc_view` shows; and when a two-file level 0 (three and four channels) and a one-file level 0 (one and two channels)
all load, `--raw0` / `--raw1` show the latents, and a device without `textureCompressionBC` is refused by name rather
than by a wrong picture.

### Stage 3 -- parity  *(built, tag v1.1.3-vk-stage3)*

Keys P/B/T/X/M/L/N/V/C/1/2/4/R/Space/arrows/WS/ADQE, the four-line overlay, all fourteen flags with the same spellings
and the same `--help` text, the load-time bc pack and its printed round-trip numbers, the cube, and the camera.
`viewer_vk/README.md`. Then the parity measurement of section 1.14 against the Direct3D viewer on a handful of assets
(one single image, one four-texture material, one three-channel level 0, one uncompressed level 0), and a line in the
release gate that runs the Vulkan `--shot` the same way it runs the Direct3D one -- guarded so that a machine with no
Vulkan driver skips it rather than fails.

*Done when* both usages exit 0 and name the same fourteen flags in the same spellings -- the Vulkan one naming those
fourteen plus its own `--size` and `--device`, which is what `tests/run_checks.py` pins, and which is a check on the
spellings and on the list being complete rather than on two texts reading alike -- and the parity numbers are in the
document.

### The review of stage 3  *(applied, tags v1.1.4-vk-fixes and v1.1.5-vk-gate-docs)*

Not a stage: the findings of a review of what stages 1 to 3 built, applied in two commits.

**`v1.1.4-vk-fixes`, the code.** The crash on minimise or a zero-extent resize (the frame loop acquired on a null
swapchain); a `lod_bias_level1` past the device's `maxSamplerLodBias` refused rather than left out of spec (section
1.10's note that Vulkan clamps it was wrong - the valid-usage rule rejects it); the shaders read from beside the
executable and from nowhere else, closing the hole where a stray `view.frag` in the working directory silently replaced
the real one; the load-time pack's printed round-trip error computed in double, so both viewers print the same number;
the validation layer keyed on `NDEBUG` rather than MSVC's `_DEBUG`; key X a no-op that says so on a device without
`samplerAnisotropy`; the surface format printed, and a surface offering no `UNORM` form refused instead of an sRGB one
taken silently; the pack's partial failure cleaned up; key R committing both pipelines or neither; a full teardown on
the normal exit path, with the Debug gate silent under validation; `--size` checked against the device's limits;
`--shot`'s directory checked before any Vulkan object exists; and the phrase `no Vulkan device` in all three refusals
that mean "this machine cannot run this viewer".

**`v1.1.5-vk-gate-docs`, the gate and these documents.** The cross-viewer pairs at `--z -50 --noaniso` with and without
level 1's bias (the default-camera pairs alone never exercised a threshold, and are kept with a comment saying they
test the y flip and the UV orientation); `--raw0`, `--raw1`, `--renorm`, `--cube` and the load-time BC pack compared
against the Direct3D frame rather than only asserted to change this one; the overlay's 84 rows asserted byte-identical
between the viewers with the rows below them equal to each viewer's own `--nooverlay` frame; a skipped Vulkan arm never
recorded as a run one; `--device` parsed from the usable devices only and the device named in the summary; the default
`--size` and the 24-bit depth pinned; the Vulkan-only refusals and a broken shader at start-up gated; the arms that
need no Direct3D frame run off Windows; and the Vulkan arms' own runtime in the summary line.

### Stage 4 -- cooperative vectors  *(built, tag v1.1.6-vk-coopvec)*

Section 3. A second fragment shader (`viewer_vk/bin/view_coopvec.frag`), the four-part runtime device query, the
converted matrix in a storage buffer, two graphics pipelines, the key `K` and `--coopvec 0|1` to switch between
them, `--bench N` with GPU timestamps, and the write-up in `viewer_vk/README.md`.

*Done when* the viewer runs unchanged on a device without the extension, runs the extension path on a device with
it, the two paths' images are compared and the difference explained, and the frame times of both are recorded at
2560x1440. All four hold; the numbers are in sections 3.7 and 3.8 below, restated from the runs.

**Three things did not go as this document predicted, and each is recorded where it was predicted rather than
here**: the driver enumerates no fp32 result type at all, so the all-fp16 tuple is what runs (3.5); the measured
difference is 4 to 5 times rather than "probably nothing", for a reason that is about the plain shader and not
about the tensor cores (3.7); and `--bench` counts FRAMES rather than evaluating the decode N times per pixel as
3.8 sketched (3.8).

---

## 3. Cooperative vectors

### 3.1 What the extension is

`VK_NV_cooperative_vector` adds matrix-vector arithmetic to a shader: a small vector lives in the invocation, a
matrix lives in a buffer the shader addresses, and one instruction multiplies them and adds a bias, with the
implementation free to gather the invocations of a subgroup behind the scenes and run the whole thing on the tensor
cores. It is the small-network inference primitive, and it is deliberately lower level than an inference object: the
matrix is just an address, so different invocations may use different matrices.

It is device extension number 492, revision 4, and its only dependency is `VK_KHR_get_physical_device_properties2`
or Vulkan 1.1 -- **it does not require cooperative matrix and it does not require buffer device address**
(https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_vector.html; the design rationale is at
https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_vector.html). Unlike cooperative
matrix it needs neither uniform control flow nor a fully occupied subgroup, "although these do increase the
likelihood of being on the fast path" (same proposal).

Underneath it is `SPV_NV_cooperative_vector` (complete, revision 2, 2024-07-25, requires SPIR-V 1.6:
https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_vector.html) and, in the shading
language, `GL_NV_cooperative_vector` (revision 2, last modified 2025-08-27:
https://github.com/KhronosGroup/GLSL/blob/main/extensions/nv/GLSL_NV_cooperative_vector.txt).

### 3.2 Where it stands today, stated plainly

* **One vendor.** NVIDIA only. The minimum hardware NVIDIA names is an RTX 20-series part (Turing) and newer, with
  Vulkan SDK 1.3.296.0 or later and a public driver of **572.16 or later**
  (https://github.com/NVIDIA-RTX/RTXNS). AMD has an open request with no response
  (https://github.com/GPUOpen-Drivers/AMD-Gfx-Drivers/issues/98, opened 2026-04-05).
* **No Khronos successor.** Cooperative *matrix* was promoted to `VK_KHR_cooperative_matrix`; cooperative *vector*
  was not, and the research behind this document found no evidence of a KHR or EXT version. Treat "a portable
  version is coming" as unconfirmed.
* **The Direct3D 12 path is a dead end for now.** Microsoft previewed cooperative vectors in Shader Model 6.9 with
  a shader linear-algebra namespace (https://devblogs.microsoft.com/directx/cooperative-vector/, 2025-06-02) and
  then announced that "the original Cooperative Vector spec will be deprecated", to be replaced by a unified
  vector-matrix and matrix-matrix design
  (https://devblogs.microsoft.com/directx/shader-model-6-9-and-the-future-of-cooperative-vector/, 2025-09-11); when
  Shader Model 6.9 shipped retail (https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/,
  2026-02-26) cooperative vector was not in it. So there is no Direct3D arm of this experiment to plan, which is
  itself an argument for doing it on Vulkan.
* **Tool support is there.** glslang has supported `GL_NV_cooperative_vector` since v15.2.0, 2025-02-24
  (https://github.com/KhronosGroup/glslang/blob/main/CHANGES.md), so any recent SDK's `glslangValidator` compiles
  it; `shaderc` wraps glslang, so `glslc` and `shaderc_combined` from the same SDK should, though no shaderc
  release note was found that says so in as many words. Slang has had it since 2025-01-30
  (https://shader-slang.org/blog/2025/01/30/coop-vec-available/).

The conclusion to draw from that list is not "skip it". It is: **the extension path is permanently optional**. The
viewer must be correct and complete without it, must detect it at runtime, and must never require it to build.

### 3.3 Our decode is one matrix-vector multiply

`docs/FORMAT.md` section 4 gives the whole runtime as

```
phi = [ z1_0 .. z1_(C1-1) , z0_0 .. z0_(C0-1) , z0_i * z1_j ]          nin  = C1 + C0 + C0*C1,  up to 24
out = W phi + b                                                        nout = 3T,               up to 18
```

so the decode is exactly `coopVecMatMulAddNV` and nothing else: an M-by-K matrix times a K-vector plus an M-vector,
with M = `nout` and K = `nin`. There is no hidden layer and no activation, so there is no second call, no
`max(0, x)` between calls, and no intermediate to carry. This is about as clean a mapping onto the instruction as
exists -- and, for the same reason, about as small a one.

**M and K must be compile-time constants.** The shading-language specification requires `M`, `K`, `matrixLayout`,
`transpose` and the three interpretation arguments to be constant expressions. Our `nin` and `nout` vary per asset.
The answer is **zero padding, decided at load**: always compile the shader for M = 18, K = 24, and at load write `W`
into a padded 18-by-24 matrix whose unused rows and columns are zero and whose unused `phi` components are zero. The
padding is exact -- a zero row contributes a zero output we already ignore, a zero column multiplies a zero feature
-- so the padded result is the unpadded result in exact arithmetic. It also means the measurement is always of the
same 18x24 shape, which makes it comparable across assets.

### 3.4 The two shader variants

The fragment shader keeps everything up to `phi` identical -- two or three `texture()` calls and the affine
dequantisation -- and then branches, at pipeline-creation time rather than per pixel, into one of two endings.

**The plain ending (today's shader, transliterated):**

```glsl
float outv[18];
for (int r = 0; r < 18; ++r) {
    float acc = bias_[r / 4][r % 4];
    if (r < nout) for (int c = 0; c < nin; ++c) acc += W_[r * 6 + c / 4][c % 4] * phi[c];
    outv[r] = acc;
}
```

**The cooperative-vector ending:**

```glsl
#extension GL_NV_cooperative_vector : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

layout(set = 0, binding = 6, std430) restrict readonly buffer Weights {
    float16_t data[];          // the converted matrix at one offset, the bias at another
} wbuf;                        // buffer base 64-byte aligned; matrix offset a multiple of 64, bias offset of 16

const uint M = 18, K = 24;     // the padded shape; literal constants, as the extension requires

coopvecNV<float16_t, K> pv;
for (uint i = 0; i < K; ++i) pv[i] = float16_t(phi[i]);     // phi already zero-padded past nin

coopvecNV<float32_t, M> res;
coopVecMatMulAddNV(res, pv, gl_ComponentTypeFloat16NV,
                   wbuf.data, MATRIX_OFFSET, gl_ComponentTypeFloat16NV,
                   wbuf.data, BIAS_OFFSET,   gl_ComponentTypeFloat16NV,
                   M, K, gl_CooperativeVectorMatrixLayoutInferencingOptimalNV,
                   false, 0);                               // stride ignored for the optimal layout

float outv[18];
for (uint r = 0; r < M; ++r) outv[r] = float(res[r]);
```

Three things in that sketch are the extension's actual rules and not invention, all from the shading-language
specification linked above: the matrix array **must be in buffer storage** (a plain storage-buffer member array is
exactly what the specification's own example uses -- no buffer reference needed); `matrixOffset` must be **64-byte
aligned** and `biasOffset` **16-byte aligned**, and those requirements apply to the base of the array and of the
buffer too; and the row-major semantics are `result[j] = sum over k of input[k] * matrix[offset + stride*j + k]`,
which is the row-major order `docs/FORMAT.md` already publishes, so no transpose is needed at any point.

Two things in it are honest weak spots, flagged again in section 4: the per-component writes into `pv` are the
access pattern the specification itself warns "may be suboptimal", and the 32-bit float result type has to be one
the device actually enumerates. The second of those is what happened (3.5).

**Two details of the sketch were built differently, and both are simplifications.** The weights array is declared
`uint data[]` rather than `float16_t data[]`: the extension ignores the array's own scalar type and reads raw bytes
per the interpretation arguments, so 32-bit words keep the shader off the 16-bit storage feature it would otherwise
have to require for nothing. And `BIAS_OFFSET` is a specialization constant rather than a literal, because how
large the driver's own inferencing-optimal layout turns out to be is the driver's business -- 1152 bytes for 18x24
fp16 on this one. A third is not a simplification but a requirement found by building it: glslang emits
`OpCapability VulkanMemoryModel` for a cooperative-vector module, so `vulkanMemoryModel` is a device feature this
path needs, and the all-fp16 form needs `shaderFloat16` as well.

### 3.5 Preparing the weights at load

The matrix cannot go into the buffer as the descriptor's row-major 32-bit floats and be used in the optimal layout:
the optimal layouts are implementation-defined and must be produced by the extension's own conversion. So, at load,
once:

1. build the padded 18-by-24 row-major matrix from `layers[0].weights` and the padded 18-entry bias, exactly as
   `load_asset` already builds its constant block;
2. call `vkConvertCooperativeVectorMatrixNV` (the host-side one -- there is a command-buffer variant,
   `vkCmdConvertCooperativeVectorMatrixNV`, and we do not need it) **twice**: first with `dstData` null to learn
   `pDstSize`, then with a destination buffer of that size.
   `VkConvertCooperativeVectorMatrixInfoNV` carries `srcComponentType` = `VK_COMPONENT_TYPE_FLOAT32_KHR`,
   `dstComponentType` = `VK_COMPONENT_TYPE_FLOAT16_KHR`, `numRows` 18, `numColumns` 24,
   `srcLayout` = row major, `srcStride` = 24 * 4, `dstLayout` = inferencing optimal. The one call does the layout
   change and the type conversion together;
3. write the converted bytes at a 64-byte-aligned offset in the storage buffer, and the 16-bit bias at a
   16-byte-aligned offset after it, and hand both offsets to the shader as specialization constants.

**Which type combinations are legal is a runtime question.** `vkGetPhysicalDeviceCooperativeVectorPropertiesNV`
enumerates supported tuples of input type, input interpretation, matrix interpretation, bias interpretation, result
type and transpose -- note there are **no size fields in that struct**, so it says nothing about M and K. The viewer
enumerates, looks for the tuple it wants, prints the table at load under a verbose flag, and falls back to the plain
path if the tuple is absent.

**WHAT WAS MEASURED, and it is this stage's one real surprise.** On the RTX 5090 at driver 581.80, with
`VK_NV_cooperative_vector` at revision 4, the sixteen enumerated tuples contain **no fp32 result type at all**. The
only floating-point one is `input fp16 as fp16 x matrix fp16 + bias fp16 -> fp16`; the rest are the 8-bit integer
and `E4M3` / `E5M2` paths that the paragraph above rules out for this format. So the recommendation of this section
-- fp16 weights with an fp32 accumulation -- is what the viewer asks for FIRST and cannot have here, and the
all-fp16 tuple is what it takes, saying so at start-up. The preference list is one array in `viewer_vk/main.cpp`,
so a driver that grows the fp32 result gets it with no change. The precision this cost is in 3.8: nothing, at the
resolution of an 8-bit output.

**And the driver is within its rights, which is the part this section originally got wrong.** The tuple this plan
asked for was never in the extension's mandated set. What `VK_NV_cooperative_vector` requires an implementation to
enumerate is six rows, and none of them has an fp32 result: fp16 x fp16 -> fp16, three rows of sint8 input and
sint32 result (the sint8, sint8-packed and mixed forms), and two of `E4M3` / `E5M2` input with an fp16 result.
Everything else is optional, so a driver that enumerates no fp32 result type at all is **conforming**, and the
all-fp16 tuple this viewer fell back to is the spec's *first mandated row* rather than a vendor's afterthought.
The preference list stays as it is -- an fp32 result is still better where it exists, and asking for it costs one
loop iteration -- but the fp32 path should be read as an optimisation that may never run, not as the design with a
fallback. The `NNTC_CV_FP32` branch of `viewer_vk/bin/view_coopvec.frag` is on that footing today: it is
**compiled-only**, in the sense that no device this was written on has ever executed it, so what is claimed of it is
that it builds and that the viewer selects it, and nothing about a frame it drew.

There is a second reason the two paths cannot be expected to agree bit for bit, and it is in the asset rather than in
the tuple: some of these weights are **subnormal in fp16**. On `pavingstones141_1k_c0_4` the smallest nonzero |W| is
3.24e-5, below fp16's smallest normal value of 6.10e-5, so it is carried with fewer than the format's eleven mantissa
bits -- and a weight smaller still would flush to zero. That, with the accumulation order, is the whole of why the
two decode paths differ by a unit of 255 rather than by nothing (3.8).

One correction to the table's reading while we are here: the `transpose` field says whether an opaque-layout matrix
of those types ALSO supports transposition. It is a capability, not a requirement, so a tuple reported with
`transpose` true is usable untransposed -- which is what this decode does, since the format publishes `W`
row-major.

**About "8-bit weights".** The extension would let us feed the matrix as 8-bit integers or as 8-bit floats
(`E4M3` / `E5M2`), with the matrix's storage type ignored and reinterpreted per the matrix interpretation argument,
and with a 32-bit integer accumulator that would be exact. But **our `W` is not 8-bit quantised**: the asset stores
it as 32-bit floating-point JSON numbers, and it is the *latents* this format quantises, not the decoder. Quantising
`W` to use the integer path would change the decoded picture and would be a format change, which this plan does not
make. So 16-bit float weights with a 32-bit float result is the right first move, and the 8-bit integer path is a
later experiment that belongs on the encoder's side of the fence, not the viewer's.

### 3.6 Choosing the path at runtime

At device-creation time:

1. enumerate device extensions; if `VK_NV_cooperative_vector` is absent, the plain path is the only path and the
   viewer says one line at startup and carries on;
2. if present, chain `VkPhysicalDeviceCooperativeVectorFeaturesNV` into `vkGetPhysicalDeviceFeatures2` and require
   `cooperativeVector` (we never need `cooperativeVectorTraining`);
3. chain `VkPhysicalDeviceCooperativeVectorPropertiesNV` into `vkGetPhysicalDeviceProperties2` and check that
   `cooperativeVectorSupportedStages` contains `VK_SHADER_STAGE_FRAGMENT_BIT`. **This check is not a formality**:
   the language explicitly says the supported stages are an api query with no compile-time check, so a driver that
   offered the extension for compute only would compile the shader and fail at pipeline creation;
4. enumerate the type tuples as above;
5. only if all four pass, enable the extension on the device, compile the second fragment shader, and build the
   second pipeline.

**Two pipelines, not a specialization constant.** The `#extension ... : require` directive and the cooperative-vector
types must not appear in a module compiled for a device that lacks the capability, and a specialization constant
cannot remove a capability declaration from a module. So the two endings live in two files -- or, tidier, in one file
behind a preprocessor guard, compiled twice with different `shaderc` macro definitions, which keeps the shared two
thirds in one place. Specialization constants are still used *within* each variant, for the matrix and bias offsets.

A key -- say `K` -- switches the bound pipeline between the two paths, and the overlay says which is live. That
makes the comparison a keystroke rather than a rebuild, which is what makes it likely to actually get measured.

### 3.7 Will it be faster? *(Measured: 4 to 5 times, and the reason is not the one below)*

**The measurements first, then the prediction they replace.** RTX 5090, driver 581.80, `--size 2560 1440`, 200
frames, the default camera, `--nooverlay`, GPU timestamps either side of the scene draw, `--bench 200` with and
without `--coopvec`:

| asset | shape | cooperative vectors | plain | |
|---|---|---|---|---|
| `pavingstones141_1k_c0_4` | nin 24, nout 15, five textures | mean **12.0 us**, min 11.6 | mean **63.7 us**, min 62.3 | 5.3x |
| `m1_m4_c0_3_c1_4` | nin 19, nout 12, four textures | mean **10.7 us**, min 10.2 | mean **42.9 us**, min 41.7 | 4.0x |

**And the honest reading of it, which is not "the tensor cores did it".** The comparison is between two shaders,
and the plain one is not a fast path: its inner loop indexes a uniform array with a running index whose bounds are
themselves uniforms, so nothing unrolls and every term is a dynamically indexed constant-buffer load. What the
table says is that one instruction beats THAT loop by 4 to 5 times at 3.7 megapixels -- which is what a viewer of
this format actually pays today, and is worth publishing as such -- and not that the instruction is 4 to 5 times a
well-written 18x24 multiply. A hand-optimised plain path would narrow it, and would stop being the transliteration
of `nntc_view.hlsl` that stage 3's parity rests on.

The reasoning below is what this document expected before the measurement, kept as it was written.

#### The prediction this replaced

An 18-by-24 matrix-vector product is 432 multiply-accumulates. That is a tiny fraction of what one tensor-core tile
consumes, and the hardware only wins when a whole subgroup's matrix-vector products coalesce into a matrix-matrix
operation. The one independent benchmark found for this document -- Kostas Anagnostou, "Adventures in Neural
Rendering part 2: Cooperative Vectors", an RTX 3080 Mobile at 1080p, compute shader against cooperative vectors
(https://interplayoflight.wordpress.com/2026/02/21/adventures-in-neural-rendering-part-2-cooperative-vectors/) --
measures a 6-3-3-1 network at 1.26 ms against 0.64 ms (about 2x) and a 6-64-64-64-1 at 240.5 ms against 1.39 ms
(about 173x), and concludes of the small cases that "this kind of workload does not provide the Tensor cores with
enough data to get a meaningful acceleration."

Our single 18x24 layer is *smaller* than his smallest case. The honest expectation is therefore: **at best the 2x
end of that range on the decode arithmetic alone, and quite possibly no measurable change in frame time at all**,
because a full-screen quad's frame is dominated by the texture sampling and the rasterisation rather than by 432
multiply-accumulates. Two further mechanical costs push the same way: M = 18 and K = 24 are not multiples of 16, so
padding inside the tensor-core tile is likely, and building the input vector component by component is the pattern
the specification warns about.

That is not a reason to skip stage 4. The point of stage 4 is that **the path exists and is measured**: the shape of
this decode is exactly the shape the extension was designed for, this tree's claim is about what a real sampler and
a real shader do, and a measured "no faster at this size, here is the number" is a result worth having and worth
publishing. It also puts the plumbing in place for the day the decoder is not one affine layer.

For contrast, and to be fair to the extension: NVIDIA reports a "2-4x improvement in inference throughput" for its
own neural texture compression work with these extensions (https://github.com/NVIDIA-RTX/RTXNTC), and Slang reports
"up to 4x speedup in decompression compared to using DP4a instructions"
(https://shader-slang.org/blog/2025/01/30/coop-vec-available/). Both are for networks around 48 inputs and 64 wide
with several layers -- materially bigger than ours.

### 3.8 Precision, and what to compare *(measured: max 1 of 255, 72 to 75 dB)*

**What the runs said.** Both example assets, the default camera, `--noaniso` so the sampler is identical and the
fp16 weights are the only difference, `--nooverlay`, `--size 2560 1440`, one `--shot` per path through
`tools/frame_diff.py`:

| asset | max abs diff | mean abs diff (r / g / b) | PSNR |
|---|---|---|---|
| `pavingstones141_1k_c0_4` | **1** of 255 | 0.0024 / 0.0017 / 0.0026 | **74.62 dB** |
| `m1_m4_c0_3_c1_4` | **1** of 255 | 0.0029 / 0.0054 / 0.0025 | **72.52 dB** |

So the expectation stated in advance below -- a max difference of 1, occasionally 2 -- is exactly what happened,
and it happened on the ALL-FP16 tuple rather than on the fp32-accumulate one this section argues for, which that
driver does not offer (3.5). The 11 bits of the weights are the whole of it at the resolution of an 8-bit output.

**One thing here was not built as sketched.** The third bullet below asks for a `--bench N` that evaluates the
decode N times per pixel with the results summed so nothing is optimised away, reported at N = 1, 8 and 64. What
was built instead is a `--bench N` that renders N whole FRAMES headless and reports the mean and the minimum GPU
time of the scene draw. The reason is that the per-pixel repetition measures a loop the shader does not contain and
would need the decode's result folded into the output to survive the optimiser, which is a third shader to keep in
step with the other two; and the frame measurement turned out to separate the two paths by 4 to 5 times on its own
(3.7), so the isolating variant had nothing left to disambiguate. The text below is what was planned.

#### What was planned

`W` is 32-bit float in the asset and 16-bit float in the converted matrix. A 16-bit float carries 11 bits of
mantissa, so each weight picks up a relative error of the order of 5e-4; the sum of 24 such terms, with values of
order one, lands well inside the 1/255 = 0.0039 that the 8-bit output can represent -- but **not** at zero, so the
two paths will not be byte-identical everywhere. Asking for a 32-bit float result type, rather than a 16-bit
accumulator, keeps the accumulation exact and confines the error to the weights themselves, which is the cheapest
half of the fix and should be done if the device enumerates that tuple.

So the comparison to run, and to write down:

* **image**: `--shot` the same asset, same camera, `--nooverlay`, once per path. Report max absolute difference per
  channel and the PSNR between the two frames. The expectation to state in advance is **a max difference of 1,
  occasionally 2**, out of 255. A larger difference means something other than 16-bit rounding, and is a bug to
  find, not a tolerance to widen;
* **time**: gpu timestamps, not wall clock. A two-entry `VK_QUERY_TYPE_TIMESTAMP` pool, `vkCmdWriteTimestamp2`
  either side of the scene draw, `timestampPeriod` from `VkPhysicalDeviceLimits` to turn ticks into nanoseconds, and
  `timestampValidBits` on the queue family checked first. Take 200 frames at 2560x1440 with the quad filling the
  view, drop the first 20, report the median and the interquartile range for each path. Wall-clock frame time is
  useless here because the frame loop waits on the device by design (section 1.12);
* **and, because the decode is not the frame**, add `--bench N`: evaluate the decode `N` times per pixel, summing
  the results into the output so nothing is optimised away, and report the two paths at N = 1, 8 and 64. N = 1 is
  the honest "does this change the frame" answer; the larger values isolate the arithmetic and say what the
  instruction is actually worth per evaluation. Report both. Quoting only the large-N number would be the kind of
  estimated statistic this tree does not publish.

---

## 4. Risks and unknowns, and how each one is retired

| # | risk or unknown | how it is retired |
|---|---|---|
| 1 | Vulkan's verbosity swamps the readability the owner asked for | stage 1 is written and reviewed before anything else is attempted, with the validation layers on from the first line. If stage 1 does not read well as one file, the plan is reconsidered before stage 2 is started, not after |
| 2 | `shaderc_combined` will not link cleanly against this tree's runtime choice (the encoder forces the static release c runtime for the cuda runtime's sake) | try it in stage 2, in an hour. The fallback is already designed: shell out to `glslc.exe` and read the compiled module, three lines of process creation, no vendored file either way |
| 3 | a device does not support linear filtering on `BC4_UNORM_BLOCK` / `BC5_UNORM_BLOCK` | `textureCompressionBC` is requested as a device feature and `vkGetPhysicalDeviceFormatProperties` is checked per format at load; a device that fails is refused by name at load rather than rendering something wrong |
| 4 | `mipLodBias` = 2.0 sits exactly on Vulkan's guaranteed floor for `maxSamplerLodBias`, and a bias past the limit is OUT OF SPEC (this row used to say "clamped rather than rejected", which is wrong: see section 1.10) | read `limits.maxSamplerLodBias` at startup and refuse, naming the value and the limit, if the descriptor's `lod_bias_level1` exceeds it in either direction, before any sampler is created. Today it cannot (the format fixes `block` at 4), so this is a guard against a future asset and a hand-authored descriptor, not a live problem |
| 5 | the classic swapchain synchronisation bug: a render-finished semaphore reused while its image is still presenting | one render-finished semaphore per swapchain image from the start, validation layers on in debug, and a deliberate resize-storm test at the end of stage 1 |
| 6 | the two viewers' frames differ more than expected and it is unclear which is right | `tools/dds_decode.py` is the third, independent reader and is the arbiter: it already decodes an asset with no gpu at all. Any disagreement between the two viewers is resolved against it, not against either one |
| 7 | the release gate's forbidden-pattern check does not cover the new directory | `tests/run_checks.py` has a searched-directory list. Add the new viewer's directory to it in stage 1. That is a change to the tests, which the constraint allows; it is not a change to the encoder, the format or the Direct3D viewer |
| 8 | the gate's machine has no Vulkan driver, or a headless one | the Vulkan viewer's gate check is guarded: if the executable does not exist or exits saying it found no Vulkan device, the check reports skipped, never failed. The Direct3D checks are untouched, so gate coverage cannot regress |
| 9 | **cooperative vectors: fragment-stage support is unverified** on real NVIDIA drivers. The language permits all stages "subject to api-specific limitations", and the supported stages are a runtime query with no compile-time check | query `cooperativeVectorSupportedStages` first and print it. If the fragment bit is absent, the fallback within stage 4 is a compute pass that decodes into an image which the fragment shader then samples -- a different program, worth writing only if the query says so. **Retired by measurement: the fragment bit IS there** on the 5090 at 581.80, and `maxCooperativeVectorComponents` is 512, so the compute variant was never needed |
| 10 | **cooperative vectors: which type tuples the driver reports is unverified.** No published list was found; the query enumerates combinations, not sizes | `vkGetPhysicalDeviceCooperativeVectorPropertiesNV` is enumerated and printed at load. The path is built only for a tuple that is actually present, and the plain path runs otherwise. **This is the risk that fired**: the driver offers sixteen tuples and no fp32 result among them, so the preferred one is unavailable and the all-fp16 one runs (3.5). The design held -- the viewer took a tuple that is present and said which |
| 11 | **cooperative vectors: whether M and K may be specialization constants** rather than literal constants is unverified; the specification says "constant expressions" | irrelevant by construction: the padding decision of section 3.3 makes M and K the literals 18 and 24 for every asset. If specialization constants turn out to work, they are an optimisation to revisit, not a dependency. **Retired as designed**: M and K are the literals and were never tested as anything else; the bias OFFSET, which is not one of them, is a specialization constant and compiles |
| 12 | **cooperative vectors: `shaderc` may not accept the extension** even though glslang 15.2.0 and later does | test it in ten minutes with `glslc` on a stub shader before any of stage 4 is written. If `shaderc` refuses, `glslangValidator` from the same SDK does not, and the stage-4 shader can be compiled at build time while the plain shader keeps its runtime reload. **Retired by measurement and the fallback was not needed**: the SDK's `glslc` compiled the stub (SPIR-V 1.6, `OpCapability CooperativeVectorNV`, `spirv-val` clean) and the linked `shaderc_combined` compiles the real shader at run time, so key `R` reloads all three files |
| 13 | **cooperative vectors: one vendor, no portable successor, and the Direct3D preview deprecated** | the extension is optional at build time and at run time, and the viewer's correctness never depends on it. If the extension were withdrawn tomorrow the viewer would lose one keystroke and one paragraph |
| 14 | the 16-bit path is not byte-identical to the 32-bit one and the difference is argued about later | the expectation is written down in advance (section 3.8: a max difference of 1, occasionally 2, out of 255) together with what a larger difference would mean. **Retired by measurement: max 1 of 255 at 72 to 75 dB** on both example assets, and the release gate holds it to 4 and 50 dB so a regression is caught rather than argued about |
| 15 | the Linux port drifts because nobody exercises it | the six platform functions of section 1.7 are written on day one even though only the Win32 bodies exist, and `--shot` needs no window at all, so the Linux build has a runnable target from the start |
| 16 | scope creep: the Vulkan viewer grows features the Direct3D one does not have, and the two stop being comparable | stages 1 to 3 add nothing. Every new capability -- the headless shot, the gpu timestamps, the cooperative-vector key -- is either a strict superset justified above or lives behind a flag that is off by default |

---

## 5. References

**Specifications and required behaviour**

* Vulkan features chapter (`textureCompressionBC`, `samplerAnisotropy`):
  https://docs.vulkan.org/spec/latest/chapters/features.html
* Vulkan required limits (`maxPushConstantsSize` 128 in core and 256 in 1.4, `maxUniformBufferRange` 16384,
  `maxSamplerLodBias` 2, `maxSamplerAnisotropy` 16, `maxMemoryAllocationCount` 4096):
  https://docs.vulkan.org/refpages/latest/refpages/source/Required_Limits.html
* `VkPhysicalDeviceLimits`:
  https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html
* Vulkan formats and the compressed-format appendix:
  https://docs.vulkan.org/spec/latest/chapters/formats.html and
  https://docs.vulkan.org/spec/latest/appendices/compressedtex.html
* Vulkan versions and what each one made core:
  https://docs.vulkan.org/guide/latest/versions.html
* The Khronos registry mirrors the same pages at https://registry.khronos.org/vulkan/specs/latest/ ; it refuses
  automated fetches, so every citation above is to the `docs.vulkan.org` copy of the same generated text.

**Samples and tutorials worth reading before stage 1**

* Hello Triangle with Vulkan 1.3 features (dynamic rendering, synchronisation2, extended dynamic state), the closest
  published thing to what stage 1 is:
  https://docs.vulkan.org/samples/latest/samples/api/hello_triangle_1_3/README.html
* Sascha Willems' examples, the reference collection for "how is this actually done":
  https://github.com/SaschaWillems/Vulkan
* vulkan-tutorial, still the clearest walkthrough of device, swapchain, pipeline and the image-upload path:
  https://vulkan-tutorial.com/ , Khronos-hosted version https://docs.vulkan.org/tutorial/latest/
* `vk_minimal_latest`, a single-file modern sample -- useful as a shape to imitate, though it targets Vulkan 1.4 and
  uses volk, a memory allocator and a windowing library, all of which this plan declines:
  https://github.com/nvpro-samples/vk_minimal_latest (Apache 2.0)

**Tools**

* CMake's `FindVulkan`, and the component names used in section 1.4:
  https://cmake.org/cmake/help/latest/module/FindVulkan.html
* Vulkan SDK release notes (1.4.357.0, 28 July 2026; the Windows package ships the loader, headers, validation
  layers, glslang, shaderc, volk and dxc): https://vulkan.lunarg.com/doc/view/latest/windows/release_notes.html
* shaderc (Apache 2.0): https://github.com/google/shaderc
* volk (MIT), considered and declined in section 1.6: https://github.com/zeux/volk
* the DirectX shader compiler's SPIR-V back end, considered and declined in section 1.8:
  https://github.com/microsoft/DirectXShaderCompiler/blob/main/docs/SPIR-V.rst

**Cooperative vectors**

* `VK_NV_cooperative_vector` reference page:
  https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_vector.html
* the extension's design rationale:
  https://docs.vulkan.org/features/latest/features/proposals/VK_NV_cooperative_vector.html
* `VkPhysicalDeviceCooperativeVectorFeaturesNV`:
  https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeVectorFeaturesNV.html
* `VkPhysicalDeviceCooperativeVectorPropertiesNV`, the supported-stages mask and
  `maxCooperativeVectorComponents`:
  https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceCooperativeVectorPropertiesNV.html
* `SPV_NV_cooperative_vector`:
  https://github.khronos.org/SPIRV-Registry/extensions/NV/SPV_NV_cooperative_vector.html
* `GL_NV_cooperative_vector`, the shading-language specification with the exact argument order, the alignment rules
  and the row-major semantics quoted in section 3.4:
  https://github.com/KhronosGroup/GLSL/blob/main/extensions/nv/GLSL_NV_cooperative_vector.txt
* glslang's changelog, for the version that added it (15.2.0, 24 February 2025):
  https://github.com/KhronosGroup/glslang/blob/main/CHANGES.md
* NVIDIA's neural shading samples, the Vulkan cooperative-vector sample set and the source of the driver and
  hardware requirements: https://github.com/NVIDIA-RTX/RTXNS
* NVIDIA's neural texture compression release, for the throughput claim and for the load-time against sample-time
  framing: https://github.com/NVIDIA-RTX/RTXNTC
* cooperative vectors in OptiX, 17 April 2025:
  https://developer.nvidia.com/blog/neural-rendering-in-nvidia-optix-using-cooperative-vectors
* Slang's cooperative-vector support, 30 January 2025:
  https://shader-slang.org/blog/2025/01/30/coop-vec-available/
* the Direct3D 12 preview and its deprecation:
  https://devblogs.microsoft.com/directx/cooperative-vector/ ,
  https://devblogs.microsoft.com/directx/shader-model-6-9-and-the-future-of-cooperative-vector/ and
  https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/
* the one independent benchmark of small networks with and without the extension:
  https://interplayoflight.wordpress.com/2026/02/21/adventures-in-neural-rendering-part-2-cooperative-vectors/
* an AMD request for the extension, open and unanswered:
  https://github.com/GPUOpen-Drivers/AMD-Gfx-Drivers/issues/98

**In this tree**

* `docs/FORMAT.md` -- the asset, the feature order, the level-1 bias rule, the two-file level 0
* `viewer/README.md` and `viewer/main.cpp` -- the program this one is a sibling of
* `viewer/bin/nntc_view.hlsl` -- the decode being transliterated
* `tools/dds_decode.py` -- the independent reader, and the arbiter when two viewers disagree

## 5. Amendments: the owner's decisions after reading the plan

Recorded here so the build follows them rather than the earlier text where the two differ.

1. **The fallback is the baseline, not a fallback.** The plain-GLSL decode (stages 1 to 3) is the product and must run
   on any reasonable Vulkan device from AMD, Intel, NVIDIA or Qualcomm. It is brought up first and tested on every
   device within reach: the RTX 5090, the integrated Radeon on the same machine (through `--device`, below), the
   RX 9060 XT when it arrives, and the Linux box. Cooperative vectors (stage 4) are a second pipeline chosen only when
   the device query passes; on every other device the viewer never mentions the extension. A mobile part without BC
   formats reads the uncompressed level-0 twin the encoder's `--bc0 0` palette mode writes; nothing in the encoder or
   the format changes for that. *(Since `v1.1.5-vk-gate-docs` that twin is not only loaded but COMPARED: the gate
   draws an uncompressed level 0 with `--bc` on both viewers with anisotropy off and asserts the two frames are
   identical, which they are - both make the pack at load from the same `shared/bc_pack.h`, so the blocks are the same
   bytes and only the sampler could differ.)*
2. **The device choice is the standard one.** Enumerate the physical devices, take the first discrete GPU with a queue
   family that supports both graphics and presentation to the window's surface, else the first device that does.
   `--device N` overrides the choice for testing, and the chosen device's name is printed at start so there is never
   any doubt which GPU rendered. (On the development machine device 1 is the Ryzen's integrated Radeon; it is ignored
   by default and selected on purpose to exercise the baseline path on a non-NVIDIA device.)
3. **Shared source, no library.** The code both viewers need - `json.h`, `nntc_json.h`, `bc_pack.h`, and the `.dds`
   loader lifted out of the Direct3D viewer's `main.cpp` into a header - lives in one `shared/` directory that both
   include from. Each viewer stays one executable target in the one `CMakeLists.txt`; there is no library target and
   no new dependency. The Direct3D viewer changes only in its include paths, not in what it does.
4. **Linux is designed in from the start, not later.** The platform layer of section 1.7 is written with both bodies
   from the first pass, Win32 and xcb, behind one `#ifdef`; the headless `--shot` path, which needs no window, is the
   Linux build's first runnable target and is what the gate runs there. The Linux box that is coming is the test
   machine; WSL only proves the code compiles under gcc, since its Vulkan goes through a translation layer and not the
   NVIDIA driver.
   **What has actually been done off Windows, stated plainly:** stage 1 was built under WSL and its `--shot` ran there
   on llvmpipe. Stages 2 and 3 build there and have NOT been re-verified by hand since, so no frame from that side is
   quoted anywhere in this tree. What `v1.1.5-vk-gate-docs` adds is the gate's side of it: every arm that needs no
   Direct3D frame now runs wherever the viewer was built rather than being skipped for not being Windows, so the first
   Linux box to run the gate exercises them without a further change here.
5. **Simple.** One `CMakeLists.txt` for the whole tree as today, one target per program, no build-time code
   generation, no package manager, nothing fancier than the Direct3D viewer already is. `shaderc_combined` being a
   large static library is accepted: this is a sample.
6. **The gate.** `tests/run_checks.py`'s `SEARCHED_DIRS` gains `viewer_vk` and `shared` in stage 1 (the tree scan
   only looks where it is told), and the gate drives the Vulkan viewer's `--shot` the way it drives the Direct3D one
   in stage 3.
7. **The Linux window is GLFW, not xcb (v1.1.13).** The xcb block this plan describes was written and worked
   (v1.1.12) and was replaced the same day: an X11-only window kept through Xwayland on desktops that now default
   to Wayland, with the keyboard mapping, the auto-repeat pair and the close protocol all ours to keep right, is a
   maintenance burden a sample should not carry. GLFW 3 is the small C library every Vulkan sample uses for its
   window, the distributions package it, and the Linux block is sixty lines. The Windows block stays native Win32,
   so the Windows build still needs nothing fetched. The platform layer lives in `viewer_vk/platform.h`,
   `platform_win32.cpp` and `platform_linux.cpp`. Wherever this plan says xcb, read GLFW.

## 6. The build log: what was actually done, in order, and what it taught

Written after the fact from the tags. Each entry is a commit or a group of them, what it changed, and where it
departed from the plan above. The plan's sections 1 to 4 are left as they were written so the two can be compared.

| tag | what |
| --- | --- |
| `v1.1.0-shared` | `shared/` created: `json.h`, `nntc_json.h`, `bc_pack.h` moved from `src/`, and the `.dds` header walk lifted out of the Direct3D viewer into `shared/dds.h`. The one change the Direct3D viewer took in the whole effort, 40 lines, no behaviour change. |
| `v1.1.1` to `v1.1.3` | Stages 1 to 3 as planned: window and clear, the quad and the decode, then parity. Byte-identical frames against the Direct3D viewer in every non-anisotropic state on the RTX 5090; anisotropy differs by design of the two drivers' filters (49.9 dB). |
| `v1.1.4`, `v1.1.5` | The stage 3 review: the minimise crash, the unchecked LOD bias, shaders read from the working directory, the fp16 round-trip print, `_DEBUG` where `NDEBUG` was meant, unchecked D3D creates, `atof` parsing, `--shot` exit codes. |
| `v1.1.6` | Stage 4, cooperative vectors, behind a runtime query. The driver offers only the all-fp16 tuple; the fp32-result tuple the plan wanted first is not enumerated by any driver seen. Max 1 of 255 against the plain path, 72 to 75 dB; 12 us against 64 us for the plain loop, with the caveat that the plain loop is not specialised. |
| `v1.1.7`, `v1.1.8` | The portability reviews: depth format, composite alpha, swapchain extension and 1.3 features queried rather than assumed; a BC asset refused by name once, from the descriptor, with the `--bc0 0` remedy; `readlink` of `/proc/self/exe`; SUBOPTIMAL only recreates when the extent changed; IDENTITY pre-transform when offered; the loader-version message; HOST_CACHED readback. |
| `v1.1.9`, `v1.1.10` | CUDA probed, not required, so a machine without the toolkit builds the viewers. A named `-T cuda=` toolset that fails is still an error; the cached NOTFOUND is explained. |
| `v1.1.11` | **The first real Linux finding.** Every WSL check so far had used the Windows SDK's Vulkan headers. Distribution headers are older (Ubuntu 22.04: 204, Debian 12: 239, Ubuntu 24.04: 275) and `VK_NV_cooperative_vector` first appears in 1.4.307, so the viewer had never compiled against a stock Linux package. All cooperative-vector code went behind the header's own macro, with a printed reason when compiled out. Also: Ubuntu's `libshaderc_combined.a` lacks the SPIRV-Tools objects and cannot link alone, so Linux links the distribution's shared `libshaderc`. |
| `v1.1.12` | The xcb window, as the plan described: two hundred lines, tested under WSLg from a second X client that sent keys, resizes and `WM_DELETE_WINDOW`. Replaced the same day (amendment 7). |
| `v1.1.13` | GLFW for the Linux window; the platform layer split into `platform.h`, `platform_win32.cpp`, `platform_linux.cpp`, `vk_check.h`; the instance extensions asked of the window system rather than written down. Windows stays native Win32. |
| `v1.1.14`, `v1.1.15` | Sane defaults from two reviews written as a Linux developer: Release when no build type is named, `install()` rules, `-Wall -Wextra` on the Linux target, no `glfwFocusWindow` (Wayland prints an error for it), the work-area clamp, honest GLFW 3.3 versus 3.4 text, per-vendor driver packages and an openSUSE row, the compiled-out cooperative-vector line naming NVIDIA so an AMD owner does not go looking for the SDK. The one thing declined: a too-old distribution CUDA still fails the configure, because this is a sample. |
| `v1.1.16` | Anything the configure leaves out is a CMake WARNING, after an AMD box built the headless viewer without noticing. The window opens at 1280x720 (the 2560x1440 default was bigger than the desktop it opened on); the `--shot` frame stays 2560x1440 so the gate's comparisons are unchanged; the overlay scales to a narrower window. |
| `v1.1.17`, `v1.1.18` | `DEBUG build` as a Debug build's first line. The decoder's index arithmetic as shifts and masks in both shaders, so fxc's two integer-division warnings are gone; frames byte-identical before and after. |

### What the plan got right

* **The fallback is the product.** Every device we have seen other than the 5090 takes the plain path, and on
  Linux distributions the accelerated path is not even compiled. Building the plain path first and gating it byte
  for byte against the Direct3D viewer is what made every later change safe: a shader edit, a platform swap and a
  file split were each proved by the same frame comparison.
* **`--shot` as a function of the command line alone.** It is what let a gate run headless, what let WSL prove the
  Linux code path on lavapipe without a GPU, and what let a Linux build be compared to a Windows one.
* **Refuse by name.** Each refusal carries the reason and, where there is one, the remedy. The reviews kept
  finding places where the message blamed the wrong thing (the headers instead of the vendor, the device instead
  of the loader), and each was a one-line fix because the structure was already there.
* **One CMakeLists, one target per program, nothing fetched.** Held throughout. GLFW and shaderc come from the
  distribution; the Windows build still needs nothing beyond the SDK.

### What the plan got wrong, or did not see

* **The window system.** Section 1.7's "xcb, two hundred lines, links libxcb alone" was written and worked, and
  was the wrong call: a hand-rolled X11 window on desktops that default to Wayland is a maintenance burden with no
  payoff. GLFW was the answer the plan's own "minimal dependencies" rule had argued against, and the rule was
  wrong here: the right measure is not the number of libraries but who maintains the code you would otherwise
  own.
* **The headers.** The plan treated "the Vulkan SDK" as the source of headers on every platform. On Linux the
  SDK is the exception and the distribution package is the rule, and distribution headers lag by a year or more.
  Every early "compiles under WSL" claim was hollow because WSL was pointed at the Windows SDK's include directory.
  The test that caught it - a syntax check against Khronos header tags 204 through 310 - is now the one to keep.
* **shaderc's packaging.** "shaderc_combined from the same SDK" is a Windows fact. Distribution archives are
  incomplete or absent; the shared library is what links everywhere.
* **Sizes.** 2560x1440 was chosen so the two viewers' frames were the same shape. It is a fine frame size and a
  poor window size; the two were one constant until a desktop smaller than the window pointed out the difference.
* **Messages nobody reads.** A STATUS line saying the viewer was built headless was true and useless; the user
  found out at run time. Anything that leaves a program or a capability out is a WARNING now.
* **Cooperative vectors on other hardware.** Section 3 assumed the extension's absence would be a run-time
  answer from the device. On Linux it is a compile-time answer from the headers first, and the message had to say
  that AMD and Intel are not missing anything.

### Lessons for the next port

1. Test against the headers and libraries the target's users actually have, not the vendor SDK. A matrix of
   header versions is cheap and catches what a single "it compiles" cannot.
2. When a review says "works on my box", ask which box. Two reviews written as an AMD or Intel Linux developer
   found more that mattered than the code reviews before them.
3. Default sizes, build types and message levels are user-facing behaviour. Decide them on purpose.
4. Keep the gate's byte-for-byte comparisons. They turned every later change into a yes-or-no question.
5. Do not open windows on the owner's desktop without asking. The WSLg window tests were run from a second X
   client that sent keys, resizes and the close message, which is the right way to test a window, and the wrong
   thing to do while someone is typing.

### Still open

* The RX 9060 XT, the Kubuntu AMD box, the Windows Intel machine and the Snapdragon X Elite have not run it. Items
  1 to 6 of the v1.1.8 review (the BC refusal, `exe_dir` through `PATH`, persistent SUBOPTIMAL, pre-rotation, the
  separate present family) are unreachable on the development machine and get their first real test there.
* GLFW 3.4 native Wayland has not been run; every WSLg test was GLFW 3.3 on X11 through lavapipe.
* A specialised plain-decode baseline (`nin` and `nout` as specialization constants) would make the cooperative-
  vector benchmark a fair one; today's 4 to 5 times is against an un-unrolled loop.
* Vendoring the Khronos headers (the owner's OpenCL practice) would give Linux NVIDIA users the accelerated path
  without the LunarG SDK. Not done; the guard makes it optional.
