# D3D12_LINEAR_ALGEBRA_PLAN.md -- a Direct3D 12 viewer at parity with the other two, and an optional linear-algebra decode path

**Status, 2026-09-19: both phases are built, measured and committed.** Everything below the line that follows this
block is the plan as it was written that afternoon, before any of it existed. Nothing in it has been deleted for
turning out wrong, because what was expected and what happened are together worth more than a silent edit; instead
every place the tree now disagrees with the plan carries a note beginning **Built**, **Answered**, **Corrected** or
**Wrong**, and those notes are the only part of this document written after the work. The commits are `cf52403`
(phase A), `e5bfe21` (phase B), `ee8b973` (the build document) and `77dcec5` and `f6c32b2` (two rounds of review
fixes), all of 2026-09-19.

Three other documents carry what a reader usually wants, and this one does not repeat them:

* `viewer_d3d12/README.md` -- the viewer as built: its keys, its flags, the parity numbers against the Direct3D 11
  viewer, and the linear-algebra path's measured table;
* `docs/D3D12_LINALG_BUILD.md` -- building and running the optional path: the CMake option, the three preview
  packages and what trusting them means, what lands beside the executable and why, what a machine needs;
* `viewer_d3d12/PHASE_B_NOTES.md` -- the diagnosis guide for a machine where the path does not run or draws the wrong
  picture: the detection chain step by step with the exact line each step prints, and the measured frame comparisons.

The corrections a reader should not have to hunt for, each expanded where it belongs below:

* the conversion query does not answer with a stride. `DestStride` is an input field; the application passes 0 for the
  multiply-optimal layout and 48 for row-major, and only `DestSize` comes back (B.5);
* a pipeline state whose vertex stage is DXBC and whose pixel stage is DXIL is refused with `E_INVALIDARG`, so both of
  the linear-algebra pipeline's stages are DXC's (B.4);
* the preview `d3d10warp.dll` goes beside the executable, not in the `D3D12\` folder, where it is silently not picked
  up (B.2);
* the three-arm shader comparison was never built. Only two arms exist, and that is why the measured differences
  cannot be laid at fp16 rounding's door alone (B.6);
* the RTX 5090, which this plan expected to skip phase B entirely, is the device phase B was written and measured on
  (section 3).

---

A study, not code. Two phases, decided by the owner after the first draft of this document:

* **Phase A** -- a Direct3D 12 viewer, `viewer_d3d12/`, at parity with the Direct3D 11 viewer (`viewer/`) and the
  Vulkan viewer (`viewer_vk/`): the same flags, the same keys including the F1 help page, the load-time BC4 / BC5
  pack, a headless `--shot`, the same plain decode shader with `SampleGrad` on both latents. **Stable toolchain only**:
  the Windows SDK's Direct3D 12, no Agility SDK, no Developer Mode, no experimental features; it runs on any Direct3D 12
  GPU. Its purpose is to support Direct3D 12 the way the tree supports Direct3D 11 and Vulkan: a sample, test and
  verification viewer. Performance is not a goal. Built and verified on the owner's main machine (Windows 11, an RTX
  5090, an integrated Radeon, and WARP).
* **Phase B** -- later and optional: a Shader Model 6.10 linear-algebra ("LinAlg") decode path, the D3D12 counterpart of
  the Vulkan viewer's `VK_NV_cooperative_vector` path, behind a CMake option that is **off by default** and, at run
  time, used only when the preview runtime, Shader Model 6.10 and the LinAlg tier are all present; otherwise one line
  saying why, and the plain path. Verified for **correctness against the plain path**, with WARP as the reference;
  `--bench` is informational only. Finished by the owner on a second machine (Windows 11, an AMD Radeon RX 9060 16 GB,
  RDNA 4, AMD's preview driver 26.10.07.02). NVIDIA and Intel are out of Phase B's scope.
  **Wrong, and it is the plan's largest miss.** Phase B was written, measured and gated on the **first** machine,
  because the RTX 5090's driver answers Shader Model 6.10 and grants the mandatory fp16 row; the preview WARP of the
  same machine was the reference beside it. The owner then ran the finished build on the second machine, whose card is
  a Radeon RX 9060 **XT** and which reported tier 1.0, the mandatory fp16 row, both conversion interfaces and
  `decode: linear algebra`. NVIDIA was out of scope only until the query was run on it.

Every claim about a specification, a toolchain or a driver carries a URL; every claim about this tree carries a file and
line; every number that is an estimate says so; nothing here had been built, compiled or measured when it was written.
Where the independent review of the first draft corrected a point, the correction is folded in and marked "(review)";
where the work that followed corrected the plan, the note says so and names what was measured instead.

**The short answer.** Phase A is ordinary Direct3D 12 work with nothing exotic in it: the shader is the Direct3D 11
viewer's own `nntc_view.hlsl`, compiled by the same `D3DCompileFromFile` to the same DXBC (section A.3), so the decode,
the sampling and the feature order are not re-read, and the D3D11 frame is the reference the new viewer is held to
byte for byte where the API allows (section A.6). What is new is the Direct3D 12 plumbing -- heaps, root signature,
barriers, upload and readback copies, fences, PSOs -- around asset loading, window, keys, overlay and camera code that
already exists (section A.4). Phase B then adds one `Load` and one `MultiplyAdd` in a second pixel shader compiled for
SM 6.10 by the preview DXC, a feature query in the order the Vulkan viewer already uses, and a weights buffer with the
Vulkan path's padding and range guard (sections B.4-B.6); everything about it is optional, gated and diagnosed, and
nothing in a default build depends on a preview package (section B.2). Estimates, human-equivalent and at this tree's
agent pace, are in A.7 and B.7.

---

## 0. What this document rests on

* This tree: `viewer/main.cpp` (1174 lines, Direct3D 11) and `viewer/bin/nntc_view.hlsl` (100 lines);
  `viewer_vk/main.cpp` (3680 lines), `viewer_vk/bin/view_coopvec.frag` (145 lines), `viewer_vk/README.md`;
  `shared/dds.h`, `shared/nntc_json.h`, `shared/bc_pack.h`, `shared/json.h`; `docs/FORMAT.md` section 4 (lines 224-250);
  `PRIOR_ART_DISCLOSURE.md` sections 3 and 3.2; `README.md` "How it works, briefly" (lines 414-466);
  `tests/run_checks.py` (the viewer arms, lines 539-700); `tools/frame_diff.py`; `CMakeLists.txt:245-284`;
  `docs/VULKAN_VIEWER_PLAN.md` for the shape of a viewer plan; the git log of 2026-09-15 for the pace record (A.7).
* The runtime specification, version 0.9 draft, June 2026:
  https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html (source:
  https://raw.githubusercontent.com/microsoft/DirectX-Specs/master/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.md).
* The HLSL proposals: 0035 "Linear Algebra Matrix" (Accepted, SM 6.10),
  https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md; 0029 "Cooperative Vectors"
  (Rejected, superseded by 0035), https://github.com/microsoft/hlsl-specs/blob/main/proposals/0029-cooperative-vector.md;
  0026 "HLSL Long Vector Type" (SM 6.9), https://github.com/microsoft/hlsl-specs/blob/main/proposals/0026-hlsl-long-vector-type.md.
* The DirectX developer blog: https://devblogs.microsoft.com/directx/cooperative-vector/ (June 2, 2025);
  https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/ (February 26, 2026);
  https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/ and
  https://devblogs.microsoft.com/directx/d3d12-linalg-preview/ (April 27, 2026);
  https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/ (May 28,
  2026); the Agility SDK release list, https://devblogs.microsoft.com/directx/directx12agility/.
* Compiler and package listings: https://github.com/microsoft/DirectXShaderCompiler/releases,
  https://www.nuget.org/packages/Microsoft.Direct3D.DXC, https://www.nuget.org/packages/Microsoft.Direct3D.D3D12,
  https://www.nuget.org/packages/Microsoft.Direct3D.WARP.
* Microsoft's LinAlg samples, https://github.com/llvm-beanz/linalg-examples, for the preview-SDK plumbing (B.2).
* AMD: https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/ (August 13, 2026) and the
  release-note URLs in section 3. Microsoft's own documentation pages cited inline.

---

# Part I -- Phase A: the Direct3D 12 viewer on the stable toolchain

## A.1 Scope: the parity checklist

The D3D11 viewer and the Vulkan viewer agree on a command line and a key map, and the new viewer takes both unchanged.

**Flags** (`viewer/main.cpp:892-919`, `viewer_vk/main.cpp:3510-3543`): `--shot FILE.bmp`, `--tex N`, `--cube`,
`--z F`, `--yaw F`, `--pitch F`, `--nomips`, `--nobias`, `--noaniso`, `--renorm`, `--raw0`, `--raw1`, `--bc`,
`--nooverlay`, `--size W H`, `--device N`, `--help`; every numeric flag validated as `viewer/main.cpp:920-930` does,
an unknown flag refused by name with the usage (`viewer/main.cpp:951`). `--device N` is the Vulkan viewer's flag
(`viewer_vk/main.cpp:3534`) and here selects DXGI adapter N in enumeration order, with the WARP adapter listed last and
selectable by the same flag, so that the main machine's three devices (RTX 5090, integrated Radeon, WARP) are all one
number away. `--bench N` is Phase B's; Phase A does not add it.

**Built.** Every flag and key above is in `viewer_d3d12/main.cpp`, and the `--bench` rule held: the phase A commit
adds no `--bench` at all. Phase B then added it **outside** the `#if NNTC_D3D12_LINALG` guards, together with
`--linalg 0|1` and `--linalg-layout row|optimal`, so a build made without the option still accepts and documents all
three -- `--linalg 0` is a no-op, `--linalg-layout` changes nothing, `--bench` times the one path there is, and
`--linalg 1` is refused with an `ERROR` and exit 1. The key `K` is the one thing a build without the option really
lacks: its `case` is inside the guards.

**Keys**, from the F1 help page `viewer/main.cpp:573-600`: arrows, W/S, A/D, Q/E, Shift, Space; C, N, 1, 2, V; P/B/T,
X, M, L, 4; 5-8, R, F1, Esc. `--shot` takes no keyboard input at all (`viewer/main.cpp:780-790`).

**Rendering**: the quad and the cube (`viewer/main.cpp:539-570`); the eight sampler states, four filters by mips on or
off, clamp addressing, no `MipLODBias` on either level (`viewer/main.cpp:1050-1067`); level 1's LOD shift in the
gradients (`nntc_view.hlsl:75-81`); the three-texture binding of a two-file or packed level 0 (`viewer/main.cpp:1107-1118`);
the load-time BC4 / BC5 pack from `shared/bc_pack.h` (`viewer/main.cpp:297-355`); the 2560-pixel overlay strip and the
help page (`viewer/main.cpp:571-752`).

**`--shot`**: one frame to a 24-bit bottom-up `.bmp` (`viewer/main.cpp:1142-1169`), and **never a visible window in a
test**: the frame is rendered to an offscreen target with no swap chain and no window at all, as the Vulkan viewer's
`--shot` does (`viewer_vk/README.md:120`, `148-151`). The D3D11 viewer creates a hidden window because its swap chain
needs one (`viewer/main.cpp:993-996`); the D3D12 viewer has no such need.

## A.2 The toolchain, and why it is the stable one

Phase A uses `d3d12.h`, `dxgi1_6.h` and `d3dcompiler.h` from the Windows SDK, links `d3d12 dxgi d3dcompiler`, and
creates its device at `D3D_FEATURE_LEVEL_11_0` as the D3D11 viewer does (`viewer/main.cpp:1019`). No Agility SDK is
involved: the system Direct3D 12 runtime of every supported Windows has everything a DXBC pixel shader with `SampleGrad`
needs, so the viewer runs on the RTX 5090, the integrated Radeon, WARP, the Intel integrated part and the Snapdragon
laptop alike, with no package to install and no ARM64 redistributable question (the Snapdragon note is Phase B's, B.8).

In the **Debug** configuration, the counterpart of `viewer/main.cpp:1022-1031`'s D3D11 debug layer:

* `ID3D12Debug::EnableDebugLayer` and `ID3D12Debug1::SetEnableGPUBasedValidation` (review), obtained through
  `D3D12GetDebugInterface` before the device exists
  (https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-d3d12-debug-layer-gpu-based-validation);
* DRED (review): `ID3D12DeviceRemovedExtendedDataSettings::SetAutoBreadcrumbsEnablement` and
  `SetPageFaultEnablement`, so that a device removal names the last command and the faulting address
  (https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred);
* an info-queue break on error, and, as in D3D11, a retry without the layer with a `WARNING` when the layer's DLL is not
  installed (`viewer/main.cpp:1027-1031`).

## A.3 The shader compiler: DXBC from fxc, and why

**Decision: the plain path compiles `viewer/bin/nntc_view.hlsl` with `D3DCompileFromFile`, targets `vs_5_0` and
`ps_5_0`, exactly as `viewer/main.cpp:260-273` does.** Direct3D 12 accepts DXBC, and the viewer's two shaders are then
the same bytes the D3D11 viewer hands its driver. That is what makes the D3D11 comparison a test of the **API path** --
descriptors, samplers, upload, LOD, rasterisation -- and not of two compilers: any difference between the two viewers'
frames is then in the plumbing, which is the thing Phase A exists to verify. The alternative, DXIL from DXC for
`ps_6_0`, would put a second compiler's instruction selection into every frame difference, and the Vulkan viewer's
parity story (byte-identical frames with anisotropy off, `viewer_vk/README.md:369-372`) was only readable because its
shader was a line-for-line transliteration with nothing else changed (`view_coopvec.frag:1-12`). DXBC also needs no
DLL beside the executable: `d3dcompiler_47.dll` is a system component, where DXC's `dxcompiler.dll` and `dxil.dll` would
have to be shipped.

A **DXC plain arm** -- the same `nntc_view.hlsl` compiled to DXIL `ps_6_0` by a retail DXC (NuGet
`Microsoft.Direct3D.DXC 1.9.2607.13`, https://www.nuget.org/packages/Microsoft.Direct3D.DXC, or the Windows SDK's
`dxc.exe`) -- is the second of the three shader arms the review asked for (fxc plain, DXC plain, DXC LinAlg). It belongs
to **Phase B's** preparation, because its purpose is to separate "a different compiler" from "a different instruction"
when the LinAlg frames are compared; Phase A may ship it behind `--dxc` if the implementer finds it cheap, but the
parity gate of A.6 is on the fxc arm alone. Whether Phase A carries it is decision A-3.

**Answered, and the answer is no** (decision A-3, and B-5 with it). There is no `--dxc` flag and no DXC plain arm, in
phase A or in phase B: the only two shader arms that exist are the fxc plain one and the DXC linalg one. The decision
above about the plain path held exactly as written -- `compile_hlsl_file` calls `D3DCompileFromFile` for `vs_5_0` and
`ps_5_0` on the unchanged `viewer/bin/nntc_view.hlsl` -- and it paid: the two viewers' frames are byte-identical on
one adapter with anisotropy **on** as well as off (A.6). What the missing arm costs is paid in B.6, where the
linear-algebra pipeline turned out to need a DXC vertex stage of its own.

## A.4 Architecture: what is reused, what is new

| piece | reused from | new in `viewer_d3d12/main.cpp` |
|---|---|---|
| asset: JSON, `.dds` chains, decoder constants | `shared/nntc_json.h`, `shared/dds.h`, `shared/json.h`; `viewer/main.cpp:356-527` (`load_dds`, `load_asset`, the `W` and bias packing at `514-521`) | committed default-heap textures per level (`R8_UNORM` / `R8G8_UNORM` / `R8G8B8A8_UNORM` / `BC4_UNORM` / `BC5_UNORM`, the D3D11 viewer's choices), one upload heap, `CopyTextureRegion` per mip with 256-byte-aligned row pitch, a barrier to `PIXEL_SHADER_RESOURCE` |
| the BC4 / BC5 pack at load | `shared/bc_pack.h`; `viewer/main.cpp:297-355` (`bc_pack_level0`, the two packed textures and their PSNR) | the same packed levels uploaded as `BC4_UNORM` / `BC5_UNORM` resources |
| window, keys, camera, quad, cube, overlay | `viewer/main.cpp:65-99` (font, sizes), `214-239` (`mat_*`), `539-570`, `571-752` (overlay text, `update_debug_text`, `blit_char`), `792-866` (`wnd_proc`, `process_held_keys`, the `--shot` no-input rule), `867-891` (`set_uniforms`), `892-930` (usage, parsing), `931-1013` (main's flag loop, file checks, window class) | the overlay texture's upload and its alpha-blended PSO from `viewer/main.cpp:605-611`'s two tiny shaders (DXBC too) |
| shaders | `viewer/bin/nntc_view.hlsl` unchanged; `viewer/main.cpp:260-291` (`compile_hlsl_file`, `load_shader`, `reload_shader`, key R) | `CreateGraphicsPipelineState` in place of `CreateVertexShader` / `CreatePixelShader` / `CreateInputLayout`; the previous PSO kept when a recompile fails, as `viewer_vk/README.md:105` describes |
| `--shot` | the `.bmp` writer, `viewer/main.cpp:1152-1165` | an offscreen `R8G8B8A8_UNORM` render target and `D32_FLOAT` depth, `CopyTextureRegion` into a `READBACK` heap with the 256-byte row pitch unpacked, no swap chain |
| Direct3D 12 plumbing | -- | `CreateDXGIFactory2`, `EnumAdapters1` (plus `EnumWarpAdapter` as the last entry) for `--device N`, the device, one direct queue, two frames of command allocators, one command list, a fence and event, the swap chain (flip-discard, two buffers, the window path only), RTV and DSV heaps, one shader-visible CBV/SRV heap, one sampler heap of the eight states, the root signature (A.5), two constant buffers in an upload heap mapped once and written per frame as `set_uniforms` does, the two PSOs (scene, overlay) |

The size is the D3D11 viewer's plus the plumbing: **2,000 to 2,600 lines** of C++ (estimate; the Vulkan viewer, with
two window back-ends and its optional path, is 3,680).

**Built.** `viewer_d3d12/main.cpp` was 1,724 lines at the end of phase A and is 2,578 with phase B's optional path in
it, so the estimate is right for the finished file and overshoots phase A on its own. Every row of the table above
held; the one thing the table does not mention is `--bench`'s timestamp queries, which phase B added.

## A.5 Root signature, descriptors, frame

* Root parameters: `b0` (scene) and `b1` (decoder) as root CBVs -- the two `cbuffer`s of `nntc_view.hlsl:11-29`,
  unchanged, `DecCB`'s `float4 W[108]` and `float4 bias[5]` included (`viewer/main.cpp:166-171`); one descriptor table of
  three SRVs `t0-t2`; one descriptor table of two samplers `s0-s1`. Root-signature version 1.0 through
  `D3D12SerializeRootSignature`, so that the `5_0` DXBC binds to it with no `RootSignature` attribute in the HLSL.
  **Built**, with one addition from phase B: the SRV table is `SRV_PER_SET` descriptors wide, which is three in a
  build without the option and **four** with it, `t3` being the weights buffer of B.5. The plain pipeline ignores the
  fourth, and `write_raw_srv` writes a null view into it when there is no buffer, so every descriptor of the table is
  valid either way.
* The eight samplers of `viewer/main.cpp:1055-1067` become the eight entries of one sampler heap; the frame sets the
  table's base to the pair the keys select (`viewer/main.cpp:1119-1120`: both levels through the **same** state, the
  `MaxLOD = 0` row for key M, the anisotropic slot for key X). The three-texture decision of `viewer/main.cpp:1107-1118`
  is unchanged and `sel.z` still tells the shader.
* The frame: the D3D11 frame of `viewer/main.cpp:1099-1141` with barriers, the render targets set on the heap handles,
  and a fence wait per frame -- the simplest correct loop, since speed is not a goal here. `--shot` records the same
  commands into the offscreen target, waits, reads back, writes, exits.

## A.6 Verification

1. **Frames against the D3D11 viewer**, the recipe of `viewer_vk/README.md:359-362` with the new executable in the
   second line: `--nooverlay --noaniso --shot`, the same asset, `--tex`, camera and toggles, then
   `python tools\frame_diff.py a.bmp b.bmp --max-diff 2 --min-psnr 40`. With the same DXBC on the same GPU and
   anisotropy off, byte identity is the expectation, as it was between the two APIs (`viewer_vk/README.md:369-372`); it
   is not written into the gate, for the reasons `tools/frame_diff.py`'s header gives (fill rules, bilinear weight
   precision, BC palette evaluation, anisotropic tap placement), and the number found is what the README records. The
   D3D11 `--shot` renders at its window's client size, which is why the Vulkan recipe says `--size 2560 1421`
   (`viewer_vk/README.md:180-182`, `360`); the D3D12 `--shot` takes `--size` the same way.
2. **Five identical launches** of the same `--shot` (`tests/run_checks.py:562-581`, `shot_stable`).
3. **Gate arms**, mirroring the Vulkan viewer's in `tests/run_checks.py` (its arms are summarised at
   `viewer_vk/README.md:384-390`): the six `--tex` frames of a six-texture material and the refusal of a seventh
   (`run_checks.py:667-700`); the flag toggles `--raw0`, `--raw1`, `--renorm`, `--cube`, `--nomips`, `--nobias`, each
   compared against the D3D11 frame **and** asserted to change the viewer's own frame; `--bc` both ways round; the
   `--size` and `--device` refusals; the overlay: with the overlay on, the 84 strip rows byte-identical to the D3D11
   viewer's and the rows below equal to the viewer's own `--nooverlay` frame; a broken `nntc_view.hlsl` beside the
   executable printing `SHADER ERROR` and exit 1. Each arm reports SKIPPED with the machine's own reason when the
   executable was not built or the adapter is missing, as `vulkan_viewer` and `vk_skip_note` do
   (`run_checks.py:583-598`, `658`).
4. **On every device of the main machine**: `--device 0` (RTX 5090), the integrated Radeon, WARP. WARP's frame against
   the hardware frame is a new pair the tree has not had; it is expected to differ in bilinear weight rounding as the
   integrated Radeon did (`viewer_vk/README.md:380-382`), and the number is recorded, not asserted beyond the bar.
5. **Debug build once through every key** with the debug layer, GPU-based validation and DRED on (A.2), clean.

**Built, and the expectation of item 1 was met more strongly than it was written.** On one adapter the two viewers'
frames are byte-identical in every state a `--shot` can be put in, with anisotropy **on** as well as off: 28 pairs on
the two example materials at four textures, two cameras, `--nomips`, `--nobias`, `--raw0`, `--raw1`, `--renorm`, the
cube and `--bc`, plus twelve more on another tree's measured materials, every one of them `max |diff| 0`, `PSNR inf`.
The gate still asserts the plan's `--max-diff 2 --min-psnr 40` rather than byte identity, for the reasons item 1
gives, with the load-time BC pack the one arm pinned at 0. Item 3's overlay clause is the one place the tree has
since moved: this viewer's strip grew a fifth line of its own (note 9), so the gate compares the **first** 84 rows
against the Direct3D 11 viewer's, over the whole width, and asserts the rows below **104** equal to the
`--nooverlay` frame. Across adapters (item 4) the integrated Radeon is **2 of 255 at 67.89 dB** against the RTX 5090's Direct3D 11 frame and WARP is **2 at 68.30 dB** on the paving-stones asset,
**2 at 92.19 dB** and **1 at 93.06 dB** on the four-texture material, and up to **6 at 65.41 dB** on the gate's own
64x64 asset, where a magnified texel covers many pixels -- vendor bilinear rounding, recorded and not asserted, as
planned. Items 2 and 5 are done as written, and item 3 but for that clause; the arms live in `tests/run_checks.py`,
in `d3d12_viewer_checks`, and `viewer_d3d12/README.md` holds the numbers with the command beside each one.

## A.7 Stages and effort (estimates)

| stage | what | done when |
|---|---|---|
| A1 | device, adapter list and `--device`, offscreen target, clear colour, headless `--shot` to `.bmp` | a frame of the clear colour, five launches identical |
| A2 | textures, samplers, constant buffers, the DXBC PSO, the quad and the decode | `frame_diff` against the D3D11 viewer on the two example assets |
| A3 | parity: every key and flag, cube, camera, overlay and help page, BC pack, key R, the window path | A.6 items 1-3 pass; the Debug run of item 5 is clean |
| A4 | the gate arms, the README (`viewer_d3d12/README.md` in the shape of `viewer_vk/README.md`), the CMake target beside `CMakeLists.txt:245-284` with the shader copied on every build as `278-284` does, a line in `README.md`'s viewer section (`README.md:323-331`) | `python tests/run_checks.py` green on the main machine, on all three devices where a device is needed |

**Human-equivalent**: 1.5 to 2.5 weeks for one implementer who knows the tree (estimate). **At this tree's agent
pace**: the Vulkan viewer's stages 1, 2 and 3 were committed at 16:00, 16:34 and 17:01 on 2026-09-15, its gate at
18:21 and its stage 4 at 19:13, after a plan committed at 15:13 (git log); the D3D12 boilerplate is heavier than the
Vulkan viewer's only in the places Vulkan is not verbose, so **half a day to one day** for A1-A4 (estimate), with the
owner's review passes on top of that as they were for the Vulkan viewer (the 17:58, 19:45 and 21:42 commits of the
same day).

**Built.** The pace record for the next plan: this document was committed at 17:02 on 2026-09-19 and phase A at 18:41
the same day, so A1-A4 took about an hour and forty minutes rather than the half day to a day estimated -- one
commit, not four, with the owner's review passes folded into it. Phase B then landed at 22:09, the build document at
22:21 and two rounds of review fixes at 22:45 and 23:05. The human-equivalent estimates are left as they were; nothing
here measures them.

## A.8 Decisions for the owner, Phase A, in order

* **A-1.** Confirm `--device N` as the adapter index with WARP listed last (versus a separate `--warp` flag).
  **Answered: as proposed.** `--device N` selects the DXGI adapter by enumeration index, WARP is appended as the last
  entry, the list is printed at start-up with each adapter's memory and whether it is hardware, and the chosen one is
  named. There is no `--warp` flag.
* **A-2.** Confirm the default `--shot` size (2560x1440 as the Vulkan viewer, `viewer_vk/README.md:132`) and that the
  D3D11 comparison in the gate uses `--size` to match the D3D11 client area, as the Vulkan gate does.
  **Answered: as proposed.** `--shot` is 2560x1440 by default and the window 1280x720; the comparison against the
  Direct3D 11 viewer passes `--size 2560 1421` to match that viewer's client area, and `--help` says why.
* **A-3.** Whether Phase A carries the DXC plain arm (`--dxc`, A.3) or leaves it to Phase B.
  **Answered: neither carries it.** See A.3 and B.6.
* **A-4.** Whether the fp16 conversion and the padded-matrix build that Phase B will need are lifted from
  `viewer_vk/main.cpp:2015-2021`, `2068` into `shared/` now (Phase A touches nothing else in `shared/`).
  **Answered: not lifted.** `float_to_half` is a copy in `viewer_d3d12/main.cpp`, the Vulkan viewer's function
  unchanged, so the two accelerated paths quantise the weights identically; `shared/` was not touched by either phase.
  The padded matrix is built in `build_linalg_weights` out of the same `DecoderConstants` the plain shader reads.
* **A-5.** The README's placement of the D3D12 viewer beside the other two (`README.md:323-331`) and whether
  `PRIOR_ART_DISCLOSURE.md` 3.1 ("The same asset through a second graphics API") gains a sentence for a third.
  **Half answered.** `README.md` gained its section on the third viewer, and its opening paragraph names all three.
  `PRIOR_ART_DISCLOSURE.md` was **not** touched: section 3.1 still describes a second graphics API only, 3.2 still
  describes the cooperative-vector decode alone, and section 8's table has no Direct3D 12 row beside the Vulkan one.
  That part of A-5, and stage B4's row in the same table, are **still open**.

---

# Part II -- Phase B: the optional linear-algebra decode path

## B.1 What the feature is, and which generation of it

The URL the owner gave is the **second** design of the feature. The first, "cooperative vectors", was a Shader Model
6.9 preview in Agility SDK 1.717.0-preview (May 30, 2025, https://devblogs.microsoft.com/directx/directx12agility/)
with DXC 1.8.2505 (https://github.com/microsoft/DirectXShaderCompiler/releases); its HLSL proposal 0029 is marked
Rejected, "superceded by 0035-linalg-matrix.md", and when SM 6.9 went retail (Agility SDK 1.619, DXC 1.9.2602.16,
February 26, 2026) Microsoft wrote: "Cooperative Vector has been deprecated in favor of a future design unifying
matrix-matrix and vector-matrix operations, coming in Shader Model 6.10"
(https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/). `viewer_vk/README.md:418-419` records the
same fact. Anything written against `D3D12_FEATURE_COOPERATIVE_VECTOR`, `MatrixRef` or `MulAdd` no longer compiles:
DXC 1.10.2605.2 "Removed support for experimental Cooperative Vectors"
(https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.2).

The second design is **Shader Model 6.10 "LinAlg"**: the runtime spec at the owner's URL (version history 0.1 October
2025 to 0.9 June 2026, "Draft"), HLSL proposal 0035, DXC 1.10.2605.x and Agility SDK 1.720/1.721-preview, all preview
since April 27, 2026 (https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/). Only its
**thread-scope vector-matrix** part concerns this format; the wave- and threadgroup-scope matrix-matrix parts are for
products the decode does not have.

### B.1.1 The runtime (https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html)

* **Tier**: `CheckFeatureSupport(D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT, ...)` returns `D3D12_LINEAR_ALGEBRA_TIER`
  `NOT_SUPPORTED` (0x0) or `TIER_1_0` (0x10).
* **Granular query**: `D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT`, a union selected by
  `D3D12_LINEAR_ALGEBRA_OPERATION_TYPE_THREAD_VECTOR_MATRIX_MULTIPLY`, with inputs `VectorInputType`, `MatrixInputType`,
  `BiasInputType`, `VectorResultType` (a `D3D12_LINEAR_ALGEBRA_DATATYPE`: `SINT8, UINT8, SINT16, UINT16, SINT32,
  UINT32, FLOAT16, FLOAT32, FLOAT8_E4M3FN, FLOAT8_E5M2, NONE`) and the output `SupportFlags`: `SUPPORTED` (0x1),
  `EMULATED_INPUTS` (0x2), `EMULATED_OUTPUTS` (0x4), `TRANSPOSE` (0x8). The D3D12 form of the Vulkan viewer's
  type-tuple match (`viewer_vk/main.cpp:783-799`), one tuple per call.
* **Enumeration query** (new in 0.9): `D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION`, a two-call pattern
  returning "a flat list of native driver configurations" -- the counterpart of the tuple list the Vulkan viewer
  prints (`viewer_vk/main.cpp:805-811`). "Unsupported on v1 drivers" (DDI feature version `_0115_1`), so its failure is
  a diagnostic, not a refusal.
  **Answered: there is nothing to call.** `D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION` is not in
  1.721.3-preview's `d3d12.h` at all, so the viewer prints one line saying so and never asks. Its absence is the
  diagnostic the plan wanted; if a later Agility SDK adds it, that line is the one to replace.
* **Tier 1 mandatory vector-matrix rows**: `SInt8 x SInt8 -> SInt32`, `UInt8 x UInt8 -> SInt32`, `Fp32 x SInt8 ->
  SInt32`, **`Fp16 x Fp16 -> Fp16`**; the `Fp8` matrix rows optional. "Required means implementations MUST accept and
  multiply natively (no EMULATED_INPUTS)", and (0.9) "a Required vector-matrix row may still report EMULATED_OUTPUTS".
  **Wording (review)**: an fp16 *result* type does not mean an fp16 *accumulation*; the implementation is free to
  accumulate internally in fp32 and round at the end, and `EMULATED_OUTPUTS` is precisely the flag by which it reports
  a result-type conversion. So the sample prints the flags of the row it runs on, and states the precision it observed
  (B.6) rather than asserting the arithmetic.
  **Measured, and the wording rule was worth having.** The RTX 5090 grants the mandatory all-fp16 row only, as
  `SUPPORTED | TRANSPOSE`; the preview WARP grants both, the fp16-result row as `SUPPORTED | EMULATED_OUTPUTS` and the
  fp32-result row as `SUPPORTED`. The viewer prints both rows on every run and says which one it took, and the
  measured differences in B.6 are quoted with that row named. `TRANSPOSE` is ignored: this decode never transposes.
* **Layouts**: `ROW_MAJOR`, `COLUMN_MAJOR`, `MUL_OPTIMAL`, `OUTER_PRODUCT_OPTIMAL`; `MUL_OPTIMAL` is implementation-defined
  and produced by the conversion API, the counterpart of `VK_COOPERATIVE_VECTOR_MATRIX_LAYOUT_INFERENCING_OPTIMAL_NV`
  (`viewer_vk/main.cpp:2086`).
* **Conversion**: `ID3D12DevicePreview::GetLinearAlgebraMatrixConversionDestinationInfo(D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO*)`
  (`DestSize` out; `DestLayout`, `DestStride`, `NumRows`, `NumColumns`, `DestDataType` in) and
  `ID3D12GraphicsCommandListPreview::ConvertLinearAlgebraMatrix(D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_INFO*, UINT)`
  (the dest info, a `SRC_INFO` of `SrcSize`, `SrcDataType`, `SrcLayout`, `SrcStride`, and `DestVA` / `SrcVA`), "supported
  on compute and graphics command lists (not bundles)". Unlike `vkConvertCooperativeVectorMatrixNV`, which runs on the
  host (`viewer_vk/main.cpp:2089-2099`), this is a GPU command on GPU virtual addresses. **(Review)**: treat it as a
  **layout** conversion and do the **type** conversion on the host: fp32 to fp16 in the viewer, then fp16 row-major to
  fp16 `MUL_OPTIMAL` on the GPU with `SrcDataType == DestDataType == FLOAT16`. That is correct whether or not a runtime
  also converts types, and it keeps the range guard (B.5) on the host where the numbers are.
  **Built exactly so, and this bullet's reading of the struct is the correct one**: `DestSize` is the only field the
  query answers and `DestStride` is an input. Section B.5's stage-B3 paragraph below contradicted this bullet, and
  that contradiction was a real bug in the first implementation of it; both are corrected there.
* **Alignment**: matrix base VA and offsets 128-byte aligned, stride 16-byte aligned, buffer size a multiple of 16;
  `DestSize` and `DestStride` multiples of 16, `DestVA` 128-byte aligned. The bias's alignment is not specified.
* **Validation**: PSV0 metadata is checked at pipeline creation and "unsupported operations/formats/dimensions fail with
  descriptive errors".

### B.1.2 The HLSL (https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md)

* `#include <dx/linalg.h>`, namespace `dx::linalg` (https://devblogs.microsoft.com/directx/d3d12-linalg-preview/);
  `Matrix<ComponentType, M, N, MatrixUse, MatrixScope>`; `ComponentType` in `I8, U8, I16, U16, I32, U32, I64, U64,
  F8_E4M3FN, F8_E5M2, F16, F32, F64, BFloat16`; `MatrixUse::A`; `MatrixScope::Thread`.
* **Stages -- the point that keeps the decode a pixel shader.** Verbatim: "All operations on `Thread` scope matrices
  are available in all shader stages. Operations on `Wave` and `ThreadGroup` scope matrices are available in compute
  shaders." No wave-uniformity requirement applies to thread scope. No public sample exercises the SM 6.10 API from a
  pixel shader that this study could find (Microsoft's and AMD's are compute), so the spike's first shader confirms it
  (B.7, stage B0).
  **Answered: the pixel stage works.** `nntc_view_linalg.hlsl` is a pixel shader with a thread-scope `MultiplyAdd` in
  it and it draws on all three implementations the path has been run on -- the RTX 5090, the preview WARP and the
  owner's RX 9060 XT. The compute fallback of B.8 note 6 was never needed and is not implemented.
* **Loading**: "Thread scope matrices can only be read from `ByteAddressBuffer` objects"; `Load(ByteAddressBuffer,
  uint StartOffset, uint Stride, MatrixLayoutEnum Layout)` with layouts `RowMajor = 0, ColMajor = 1, MulOptimal = 2,
  MulOptimalTranspose = 3, OuterProductOptimal = 4, OuterProductOptimalTranspose = 5`; "permits loads from RowMajor,
  ColumMajor and Optimal layouts for Thread scope matrices"; the matrix start "must be 128-byte aligned for Thread
  scope matrices". **A row-major fp16 load is legal**, so the first working path needs no conversion API; `MulOptimal`
  is a second, measurable step (B.5). The Vulkan path had to convert before it could draw (`viewer_vk/main.cpp:2072-2099`).
  **Built, both of them**, and `--linalg-layout row|optimal` picks between them at run time. Row-major is also the
  fallback wherever the two `Preview` interfaces are missing or the device asks for a zero-byte optimal matrix. On the
  RTX 5090 the two layouts' frames are **byte-identical to each other**, which is the strongest thing that pair of
  runs could have said.
* **The product**: `MultiplyAdd<OutputElTy>(Matrix<MatrixDT, M, K, MatrixUse::A, MatrixScope::Thread>, vector<InputElTy, K>,
  vector<BiasElTy, M>)` returning `vector<OutputElTy, M>`; `Multiply` without the bias; an `InterpretedVector` overload
  via `MakeInterpretedVector<DT>(vec)`. "If the Bias vector's interpretation type differs from the output vector type,
  the Bias vector is converted to the output vector type before the multiply-add operation." The output type is a
  template parameter and may differ from the matrix's (the proposal's example does `Multiply<ComponentType::F32>` on
  F16 inputs).
* **Dimensions**: "The minimum and maximum K dimension for matrices is hardware dependent and varies by scope"; the
  HLSL/DXIL validation bound for Thread scope is `[4, 128]`; no separate bound is published for M. The padded shape
  M 18, K 24 (`view_coopvec.frag:83`) is inside it; M and K are template parameters, hence compile-time constants,
  hence the padding (`view_coopvec.frag:14-17`).
* **Long vectors** (proposal 0026, SM 6.9): `vector<T, N>` up to 1024, all stages, loadable from `ByteAddressBuffer`,
  but **not** in "Cbuffers or Tbuffers" nor in the shader signature. So `W` and the bias, which `nntc_view.hlsl:27-28`
  keeps in a constant buffer, move to a raw buffer for this path, as the Vulkan path moved them to a storage buffer
  (`view_coopvec.frag:70-76`); the plain path keeps its constant buffer.
* **16-bit types**: `-enable-16bit-types`, as Microsoft's sample compiles
  (https://raw.githubusercontent.com/llvm-beanz/linalg-examples/main/sin-network/sin-network.cpp); SM 6.9 retail made
  16-bit shader ops required (https://devblogs.microsoft.com/directx/directx12agility/, entry 1.619.5).

### B.1.3 Versions

| piece | version | date | source |
|---|---|---|---|
| DXC, SM 6.10 preview | 1.10.2605.2 | April 27, 2026 | https://github.com/microsoft/DirectXShaderCompiler/releases |
| DXC, patch 1 ("Added VectorAccumulate to LinAlg Matrix API") | 1.10.2605.24 | May 26, 2026 | same |
| DXC, patch 2; current preview package | 1.10.2605.37 / NuGet 1.10.2605.37-preview | August 12, 2026 | same; https://www.nuget.org/packages/Microsoft.Direct3D.DXC |
| DXC, current retail | 1.9.2607 / NuGet 1.9.2607.13 | July 29, 2026 | same |
| Agility SDK preview, SM 6.10 | 1.720.0-preview in the release list; the LinAlg post asks for "1.720.1-preview or later" (a version slip between the two pages, noted (review)) | April 27, 2026 | https://devblogs.microsoft.com/directx/directx12agility/; https://devblogs.microsoft.com/directx/d3d12-linalg-preview/ |
| Agility SDK preview, current | 1.721.3-preview (NuGet) | July 30, 2026 | https://www.nuget.org/packages/Microsoft.Direct3D.D3D12 |
| Agility SDK retail, current | 1.619.6 (NuGet) | September 14, 2026 | same; SM 6.9, no LinAlg |
| WARP preview | 1.65535.20-preview | May 28, 2026 | https://www.nuget.org/packages/Microsoft.Direct3D.WARP; "supporting all features" per the 721 post |

Microsoft's samples pin `1.721.3-preview`, `1.10.2605.37-preview` and WARP `1.65535.20-preview`
(https://github.com/llvm-beanz/linalg-examples/tree/main/sin-network); AMD's MiniDXNN says "Microsoft AgilitySDK
1.721-preview and DXC v1.10.2605.4 are now required" (https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/).
The spec is a 0.9 draft; when it reaches 1.0 the pins and this document are re-checked against it (B.8, note 1).

**Built: those three pins, unchanged.** `NNTC_LINALG_AGILITY_VERSION`, `NNTC_LINALG_DXC_VERSION` and
`NNTC_LINALG_WARP_VERSION` in `CMakeLists.txt` are `1.721.3-preview`, `1.10.2605.37-preview` and
`1.65535.20-preview`. The *Reference* section of `docs/D3D12_LINALG_BUILD.md` lists the three URLs, the byte counts and the SHA-256 of
what this machine received, and says plainly that the build checks no hash of its own.

### B.1.4 A discrepancy to resolve first

Microsoft's sample queries `D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT` (the repeated word is
in `sin-network.cpp`) and reads `support.LinearAlgebraTier`; the spec writes
`D3D12_FEATURE_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT`. The header of 1.721.3-preview decides; the spike records which,
and every D3D12-specific name of this path lives in one file so that the next rename is one edit.

**Answered: the header's repeated word wins.** The viewer calls
`D3D12_FEATURE_LINEAR_ALGEBRA_LINEAR_ALGEBRA_MATRIX_OPERATION_SUPPORT` and reads `LinearAlgebraTier` out of
`D3D12_FEATURE_DATA_LINEAR_ALGEBRA_SUPPORT`, which is what 1.721.3-preview's `d3d12.h` spells. Every name of this path
is in `viewer_d3d12/main.cpp` behind `#if NNTC_D3D12_LINALG`, so a rename in a later SDK is one file's edit.

## B.2 The build: a CMake option, off by default

`NNTC_D3D12_LINALG`, default `OFF`. Off, `viewer_d3d12` is exactly Phase A's program: the Windows SDK's Direct3D 12,
no NuGet package, no `D3D12SDKVersion` export, nothing preview anywhere in the build. On:

* the three NuGet packages are obtained at pinned versions the way Microsoft's `cmake/LinalgSample.cmake` does
  (`nuget.exe` from `https://dist.nuget.org/win-x86-commandline/latest/nuget.exe`, then `Microsoft.Direct3D.D3D12`,
  `Microsoft.Direct3D.DXC`, `Microsoft.Direct3D.WARP`;
  https://raw.githubusercontent.com/llvm-beanz/linalg-examples/main/cmake/LinalgSample.cmake), and copied beside the
  executable: `D3D12\D3D12Core.dll`, `D3D12\D3D12SDKLayers.dll`, `D3D12\d3d10warp.dll`, `dxcompiler.dll`, `dxil.dll`,
  `dx\linalg.h`, plus the second shader file; the Agility SDK's `build/native/include` goes first on the include path;
  **two corrections here, and the second one cost an afternoon.**
  **Corrected, the fetch**: `nuget.exe` is not used and is not needed. A `.nupkg` is an ordinary zip archive, so the
  build fetches the three flat-container URLs with `file(DOWNLOAD)` and unpacks them with `file(ARCHIVE_EXTRACT)` into
  `<build dir>/nntc_linalg_packages/`, which is one fewer download and nothing on the machine. It then checks seven
  named files exist before it decides the path can be built, and a missing one is printed by name.
  **Wrong, `d3d10warp.dll`**: it belongs **beside the executable**, not in `D3D12\`. A copy under `D3D12\` is
  silently not picked up -- the run gets the **system** WARP, which answers Shader Model 6.8 and tier 0, and the whole
  detection chain then politely says no for the wrong reason. Beside the executable is also where the WARP package's
  own `.targets` file puts it. The other two placements in the list are right, and `d3d12SDKLayers.dll` is spelled
  with a lower-case `d` in the package.
* the program exports the two symbols the redistributable spec defines
  (https://microsoft.github.io/DirectX-Specs/d3d/D3D12Redistributable.html), as Microsoft's sample does:
  `__declspec(dllexport) extern const UINT D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;` and
  `__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";`;
* it compiles with `NNTC_D3D12_LINALG=1`, which is the only thing the source tests.

A configure with the option on and no network fails with a CMake Warning naming the packages and builds the viewer
without the path, in the tree's style (`README.md:113-122`). The vendored-versus-fetched question is decision B-3.
This is also why the option cannot be a run-time switch alone: the `D3D12SDKVersion` export is compile-time, and a
retail build must not carry it.

**Built as described, with three things the plan did not foresee:**

* **an option-ON executable without its `D3D12\` folder gets no device at all.** The runtime refuses to create one
  rather than falling back to the system one, on every adapter including WARP, and the run exits 1. The viewer says so
  in as many words, and the gate's arms that copy the executable into a scratch directory copy those files with it.
  The same trap bit `cmake --install`, whose rules shipped the executable without the preview runtime until
  `77dcec5`; the install rules now mirror the build tree;
* **a 32-bit build is refused by name** at configure time, with a warning, and the viewer is built without the path.
  The packages do publish a `win32` folder; an untested arm of an optional preview path is not worth carrying. An
  arm64 build takes each package's `arm64` folder and everything measured here is x64;
* `dxcompiler.dll` is loaded with `LoadLibraryEx` rather than linked (`load_dxc`), so an executable that has lost that
  one file still runs and still draws the plain path, with a line saying why. That is deliberate, and it is the
  opposite of how the `D3D12\` folder behaves.

`docs/D3D12_LINALG_BUILD.md` is the whole of this ground written for someone who only wants to build it, and
`viewer_d3d12/PHASE_B_NOTES.md` section 3 for someone whose build will not run.

## B.3 Run-time detection, diagnostics, refusal

The order is the Vulkan viewer's (`viewer_vk/main.cpp:646-665`, `706-812`), transliterated; every step prints on
failure exactly one line, `linear algebra: not available - <why>`, and the plain path draws. Nothing below is a
failure of the program.

1. **Built with the option** -- otherwise the line says so, as the Vulkan viewer's headers-too-old line does
   (`viewer_vk/README.md:258-262`).
2. **The runtime actually loaded** (review, the silent-fallback diagnostic): after device creation,
   `GetModuleHandleW(L"D3D12Core.dll")`, `GetModuleFileNameW` and the file's version resource are printed --
   `d3d12core: <path> <version>`. If the module is absent or is the system's, the preview redistributable was not
   picked up (a missing `D3D12\` folder, a wrong `D3D12SDKPath`, an export the linker dropped) and the run says so
   instead of failing three steps later with a puzzling `E_NOINTERFACE`.
3. **`D3D12EnableExperimentalFeatures(1, &D3D12ExperimentalShaderModels, nullptr, nullptr)` before the device**,
   required by the LinAlg preview post ("Must enable D3D12ExperimentalShaderModels via D3D12EnableExperimentalFeatures()
   before device creation", https://devblogs.microsoft.com/directx/d3d12-linalg-preview/). It "returns ... E_NOINTERFACE
   if an unrecognized feature is specified or Developer Mode is not enabled"
   (https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12enableexperimentalfeatures); the line then
   names Developer Mode. The retail runtime refuses it by design ("Now asking for D3D12ExperimentalShaderModels returns
   E_NOINTERFACE", https://devblogs.microsoft.com/directx/directx12agility/, entry 1.616.1). Note the learn page's
   warning that calling it again with a different list puts every device in `DEVICE_REMOVED`: it is called once.
4. **Shader Model 6.10** (review): `D3D12_FEATURE_SHADER_MODEL` with `HighestShaderModel = D3D_SHADER_MODEL_6_10`
   (https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_shader_model); the answer
   is printed, and a driver that stops at 6.9 stops here.
5. **The tier**: `D3D12_FEATURE_LINEAR_ALGEBRA_SUPPORT` at least `TIER_1_0`.
6. **The granular row**: fp16 vector, fp16 matrix, fp16 bias, fp16 result must report `SUPPORTED`; the flags are printed
   in words, `EMULATED_OUTPUTS` included (B.1.1). Then, as the Vulkan viewer prefers an fp32 result and bias when the
   device has one (`viewer_vk/main.cpp:783-789`), the fp16/fp16/fp32/fp32 row is asked for and taken if `SUPPORTED`;
   which one ran is printed, because the precision numbers mean different things under the two.
7. **The enumeration list**, printed in full when available, or `enumeration: not available on this driver`, which on a
   v1 driver is expected (B.1.1).
8. **The asset's range bound** (B.5).
9. **The PSO**: the SM 6.10 pixel shader compiled by `IDxcCompiler3` and `CreateGraphicsPipelineState`; a failure
   prints the compiler's or the runtime's own message and the plain path draws.

**Built, in that order, with three changes.** The chain is `query_linear_algebra`, called from `init_common`, and
`viewer_d3d12/PHASE_B_NOTES.md` section 5 walks it step by step with the exact line each step prints and what each
answer means -- that file, not this one, is where a reader on another machine should go. What differs from the list
above: step 8, the asset's range bound, moved **out** of the query and into `build_linalg_weights`, because it is a
property of the asset and the query runs before the asset is loaded; a step was added at the end of the query for
`dxcompiler.dll`, since a shader compiled at run time is worth failing early for; and a step was added for the matrix
layout, which the plan had only inside B.5. A step that fails after the query has said "available" -- the range bound,
a buffer that would not allocate, a pipeline that would not build -- prints a second `linear algebra: not available`
line, which is why the gate compares the **first** verdict with the **last** rather than reading either alone.

One rule was tightened in review (`77dcec5`): a failure of this path must never end a run that did not ask for it.
Three buffer allocations and a `QueryInterface` used to return false all the way out to `init_common`, which turns
false into exit 1 whether or not `--linalg 1` was given. They now leave the path unavailable and let the plain one
draw, and only `--linalg 1` makes any of them an exit.

Flags and their semantics copy the Vulkan viewer's `--coopvec` exactly (`viewer_vk/README.md:508-512`,
`viewer_vk/main.cpp:269`, `898-899`, `2039-2042`; the gate's reading of it, `tests/run_checks.py:602-613`):

| | |
|---|---|
| `--linalg 1` | the linear-algebra path. Where any step above fails this is **refused**: `ERROR: --linalg 1 was asked for and this device cannot: <why>` (or `this asset cannot`, at step 8), exit 1 -- "a run that quietly fell back would be a measurement of the other path under this one's name" |
| `--linalg 0` | the plain path, always honoured, on every build |
| neither | the linear-algebra path where every step passes, plain everywhere else |
| key `K` | switches at run time, printing `decode: linear algebra` or `decode: plain` |
| the overlay | `LinAlg:ON`, `LinAlg:OFF` or `LinAlg:n-a` at the right-hand end of the state line, past the columns the three viewers share, so the cross-viewer overlay comparison stays over the shared columns as `viewer_vk/README.md:584-586` explains |
| `--bench N` | N headless frames timed with two `D3D12_QUERY_TYPE_TIMESTAMP` queries around the scene draw, `ResolveQueryData` to a readback buffer, ticks over `GetTimestampFrequency` in double precision (https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing); **informational only**, printed in the Vulkan viewer's one-line form (`viewer_vk/main.cpp:3153`), refused together with `--shot` (`viewer_vk/README.md:518-519`) |
| `--linalg-layout row\|optimal` | **added in the building**, and not in this plan: which layout the fp16 matrix is read in, row-major as the host writes it or the device's own multiply-optimal one through `ConvertLinearAlgebraMatrix`. The default is the optimal layout where the runtime offers it. The plan had the two layouts as stage B3 with no way to ask for either, and the comparison of B.6 needs one |

**Built, every row but the overlay's.** There is no token: the strip carries a fifth line instead, `Decode: ...` at
its left margin, drawn by every build (note 9). The refusal's wording is the plan's, narrowed to say what could not:
`ERROR: --linalg 1 was asked for and this device cannot:` from the query, `this run cannot` from a resource or
interface that failed, and `this asset cannot` from the range bound. `--bench` and `--shot` together are refused as
two different runs. The three flags are parsed outside the `#if NNTC_D3D12_LINALG` guards (A.1), so they mean the
same thing in every build.

## B.4 The shader, and the vertex-stage question

`viewer_d3d12/bin/nntc_view_linalg.hlsl`: `nntc_view.hlsl` with the middle changed, following `view_coopvec.frag:89-115`
line for line elsewhere -- the two `SampleGrad`s (`nntc_view.hlsl:73-81`), dequantise after sampling (`84-85`), the
feature order of `docs/FORMAT.md:239-247`, the clamp, the texture selection, key V.

```
#include <dx/linalg.h>
using namespace dx::linalg;
// b0 and b1 as nntc_view.hlsl:11-26, minus W and bias (long vectors cannot live in a cbuffer); b1 gains uint BiasOffset
ByteAddressBuffer Weights : register(t3);        // fp16 matrix at byte 0 (128-byte aligned as the buffer base), bias at BiasOffset
static const uint M = 18, K = 24;                // the padded shape, as view_coopvec.frag:83
using WMatrix = Matrix<ComponentType::F16, M, K, MatrixUse::A, MatrixScope::Thread>;

void DecodeNNTC(float4 z0, float4 z1, out float outv[18]) {
    float phi[24]; ... // nntc_view.hlsl:54-59 verbatim
    vector<half, K> v; [unroll] for (uint i = 0; i < K; i++) v[i] = (half)phi[i];   // phi rounded to fp16 on the way in, as view_coopvec.frag:106-107
    WMatrix W = WMatrix::Load<MatrixLayoutEnum::RowMajor>(Weights, 0, K * 2);      // stride 48 bytes, a multiple of 16; the MulOptimal stage changes the layout and stride
    vector<CV_BIAS, M> b = Weights.Load<vector<CV_BIAS, M> >(BiasOffset);          // CV_BIAS / CV_RESULT = half or float, defined from the query's row, one file for both as view_coopvec.frag:26-46
    vector<CV_RESULT, M> r = MultiplyAdd<CV_RESULT>(W, v, b);
    [unroll] for (uint j = 0; j < M; j++) outv[j] = (float)r[j];
}
```

Compiled at run time by `IDxcCompiler3::Compile` with `-E PSMain -T ps_6_10 -enable-16bit-types -I <dir of dx/linalg.h>
-D CV_FP32=0|1`. `ps_6_10` follows DXC's profile naming (Microsoft's sample compiles `cs_6_10`); the spike confirms the
pixel profile. The GLSL specialization constant for the bias offset (`view_coopvec.frag:78-81`,
`viewer_vk/main.cpp:2452-2464`) becomes a `uint` in the decoder constant buffer, since `Load`'s offset is a run-time value.

**The vertex stage** (review). Phase A's vertex shader is `vs_5_0` DXBC. A PSO whose VS is DXBC and whose PS is DXIL is
something the runtime may or may not accept; the study found no statement either way, so it is a **spike item** (B.7,
B0): try the mixed PSO; if it is refused, compile `VSMain` with DXC to `vs_6_0` for the LinAlg PSO only, and note that
the LinAlg PSO's vertex stage is then a second compiler's -- a difference the DXC plain arm (A.3, B.6) measures away.

**Answered: the mixed pipeline is refused.** `CreateGraphicsPipelineState` returns `E_INVALIDARG` (`0x80070057`) for a
vertex stage of DXBC and a pixel stage of DXIL, measured on the RTX 5090 and on the preview WARP alike. So
`nntc_view_linalg.hlsl` carries its own `VSMain` -- the same three lines as the plain shader's -- and the viewer
compiles it for `vs_6_10` with the same `compile_dxil_file`. The sentence is in the shader's header comment and in
`load_shader`'s comment over the second pipeline as well, because it is the kind of fact that gets re-discovered.

And the escape hatch that sentence ends with **does not exist**: the DXC plain arm was not built (A.3), so the
difference the linalg pipeline's vertex stage makes has never been measured away. B.6 says what follows from that.

**Built, with the sketch's names changed.** The shader is `viewer_d3d12/bin/nntc_view_linalg.hlsl`; the real spellings
are `LA_M`, `LA_K` and `WMatrix`, `LA_SCALAR` (`half` or `float`) from `-D NNTC_LA_FP32=0|1`, and `LA_LAYOUT`
(`MatrixLayoutEnum::RowMajor` or `MulOptimal`) from `-D NNTC_LA_MUL_OPTIMAL=0|1`, which the sketch above did not have
because the plan had only one layout in the shader. The bias offset the sketch calls `BiasOffset` rides in the decoder
constant buffer as `linalg.x`, with the matrix's stride beside it at `linalg.y`; `W` and `bias` stay **declared** in
that constant buffer and unread, so the one struct `set_uniforms` writes has the same offsets in both shaders. The
profile names were right: `ps_6_10` and `vs_6_10`, compiled with `-enable-16bit-types` and `-I <the executable's own
directory>`, which is where the build copies `dx/linalg.h`.

## B.5 The weights buffer, the range bound, the two layouts

At load, once, mirroring `viewer_vk/main.cpp:2015-2021`: the padded 18 by 24 row-major fp32 matrix from the decoder
constants (`viewer/main.cpp:519-520`'s packing), the bias of 18, both converted to fp16 **on the host** (the
`float_to_half` of `viewer_vk/main.cpp:2068`, shared per decision A-4).

**The range bound (review).** The Vulkan viewer refuses an asset whose largest `|W|` or `|bias|` exceeds fp16's 65504
(`viewer_vk/main.cpp:2023-2043`). That guards the stored numbers but not the arithmetic: under an all-fp16 row the
products, the running sum and the result are fp16 too, and each can overflow with every weight in range. The stronger
bound this path checks, per output row `r`:

```
sum over j of |W_rj| * max|phi_j|  +  |b_r|   <=   65504
```

with `max|phi_j|` from the asset itself: a level-0 or level-1 feature is a dequantised sample, `|z| <= max(|lo|, |hi|)`
of its channel (`docs/FORMAT.md` section 6; `nntc_view.hlsl:84-85`), and a product feature is bounded by the product of
its two channels' bounds. The bound is exact arithmetic on numbers the viewer already holds, so it costs nothing and
refuses only what would overflow; the refusal reads like the Vulkan one, and with `--linalg 1` it is an `ERROR`
(B.3, step 8). The small end needs no guard (`viewer_vk/main.cpp:2027-2028`), though the fp16 subnormal weights the
Vulkan README records (`viewer_vk/README.md:459-462`) are part of why the two paths differ by one of 255.

**Layout of the one default-heap buffer, row-major form:**

```
byte    0 .. 863     the fp16 matrix, 18 rows x 48 bytes, row-major (MulOptimal: DestSize bytes from the conversion, then padding to 128)
byte  864 .. 895     zero to the next 128-byte boundary
byte  896 .. 931     the bias: 18 halves, or 18 floats (72 bytes) under the fp32-bias row
        .. 1023     zero: the size a multiple of 16 (the spec), rounded to 64 as viewer_vk/main.cpp:2101-2106 explains -- a long-vector load may read past the bias's own length
```

The matrix alignment is the spec's; the bias's 128 is this plan's conservatism where the spec is silent (a practitioner
found 64 under the 2025 API, https://interplayoflight.wordpress.com/2026/02/21/adventures-in-neural-rendering-part-2-cooperative-vectors/).
Bound as a raw SRV (`D3D12_BUFFER_SRV_FLAG_RAW`, `R32_TYPELESS`) at `t3`; the root signature of A.5 gains that one
descriptor and Phase A's plain PSO ignores it.

**Stage B3, the optimal layout (review sequence).** `GetLinearAlgebraMatrixConversionDestinationInfo` with
`NumRows 18, NumColumns 24, DestDataType FLOAT16, DestLayout MUL_OPTIMAL` gives `DestSize`, and `DestStride` is passed
in to it rather than read out of it (corrected; the note under this paragraph says what it first said). The fp16
row-major matrix goes into a source buffer in the `NON_PIXEL_SHADER_RESOURCE` state; the destination buffer is created
with `D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS` and put in `UNORDERED_ACCESS`; `ConvertLinearAlgebraMatrix` on the
load-time command list with `SrcDataType FLOAT16, SrcLayout ROW_MAJOR, SrcStride 48`; a UAV barrier; a transition to
`PIXEL_SHADER_RESOURCE`; the bias copied in after `DestSize` rounded up to 128; the shader's `Load` becomes
`MatrixLayoutEnum::MulOptimal` with a stride of 0. The two `Preview` interfaces are obtained by `QueryInterface`, and their
absence is one more line and `RowMajor`. Row-major and optimal are then two `--bench` rows and two frame comparisons; the
Vulkan viewer never had a row-major accelerated form (`viewer_vk/main.cpp:2086`), so which of the two the correctness
numbers come from is stated with them.

**Corrected in place: the conversion query does not answer with a stride.** As first written this paragraph said the
query "gives `DestSize` and `DestStride`" and had the shader's `Load` take that `DestStride`. It was wrong both
times, and against section B.1.1 above, which had it right. `DestStride` is an **input** field of
`D3D12_LINEAR_ALGEBRA_MATRIX_CONVERSION_DEST_INFO`: the application supplies 0 for the multiply-optimal layout, whose
addressing is the device's own and whose stride the load ignores, and 48 -- one row of 24 fp16 -- for row-major. Only
`DestSize` comes back. Reading a stride back out of that struct relies on a contract the API does not offer, and it
was a live bug: the first implementation initialised the stride to 48, overwrote it with the struct's `DestStride`,
and only then ran the fallbacks that can take the optimal layout away, so a device answering a zero-byte optimal
matrix would have loaded row-major at stride 0 -- row 0 for every row. The stride is now a function of the layout that
was actually chosen, decided after the fallbacks have settled (`77dcec5`), and the value the shader receives in
`linalg.y` is the application's own number either way. The code's comment says so at the one place it is read, and
`viewer_d3d12/PHASE_B_NOTES.md` section 8 item 3 repeats it for a reader chasing a wrong picture.

**The buffer, both layouts, as the viewer prints it on every run.** Row-major is exactly the layout drawn above: 864
bytes of matrix at stride 48, the bias at 896, the buffer 1024. Multiply-optimal on the RTX 5090: `DestSize` 1152 at
stride 0, the bias at 1152, the buffer 1280 -- the conversion's destination is larger than the row-major source, which
is the device's own business and is why the size is asked for rather than computed.

## B.6 Verification: correctness against the plain path, WARP as the reference

Every comparison is `--nooverlay --noaniso --shot` at the same size and camera, through `tools/frame_diff.py`, in the
form of `viewer_vk/README.md:551-563`. The three shader arms (review):

| arm | shader | compiler | what a difference from the arm above it means |
|---|---|---|---|
| fxc plain | `nntc_view.hlsl` | `D3DCompileFromFile ps_5_0` (Phase A) | the reference; byte-compared to the D3D11 viewer (A.6) |
| DXC plain | `nntc_view.hlsl` | `IDxcCompiler3 ps_6_0`, retail or preview DXC (`--dxc`) | the compiler alone: instruction selection, fp contraction. **Not built** |
| DXC linalg | `nntc_view_linalg.hlsl` | `IDxcCompiler3 ps_6_10` | the instruction: fp16 weights, fp16 `phi`, the driver's accumulation. **Built**, and its vertex stage is DXC's too (B.4) |

**Only two of the three arms exist, and it matters more than it looked like it would.** The middle row was never
built: there is no `--dxc`, and the plan had already made it optional twice over (decisions A-3 and B-5). When it was
written, the missing arm cost only the separation of "a different compiler" from "a different instruction" in the
pixel shader. Then B.4's answer arrived -- a DXBC vertex stage cannot share a pipeline with a DXIL pixel stage -- and
the linear-algebra pipeline's **vertex** stage became a second compiler's as well. So the two pipelines being compared
differ by more than the one decode instruction, and a last-bit difference in a clip-space position can move a pixel of
an edge, which is not fp16 rounding in the decode. **The measured differences below are what the two pipelines
together do**; attributing all of them to fp16 rounding is more than has been shown. The sentence is in
`viewer_d3d12/PHASE_B_NOTES.md` section 7 and in `load_shader`'s comment, in those words.

Expectations and bars, all copied from the Vulkan path and none of them yet measured here: fxc plain against DXC plain
small and not necessarily zero (recorded, bar `--max-diff 2 --min-psnr 40` as A.6); DXC linalg against DXC plain at most
1 of 255, 74.62 and 72.52 dB on the two example assets under Vulkan (`viewer_vk/README.md:560-563`), gate bar
`--max-diff 4 --min-psnr 50` with each run's start-up line asserted to name its path (`viewer_vk/README.md:575-577`).
**WARP is the reference**: it runs the linalg arm on both of the owner's machines (section 3), so the gate's linalg
pair runs everywhere the build has the option on, and the hardware linalg frame is compared to the WARP linalg frame as
well as to its own plain frame; a hardware-versus-WARP gap beyond the bar points at a driver's accumulation or a layout
bug and is worth knowing. `--linalg 1` on a device without the feature exits 1 with the `ERROR` line; `--linalg 0`
draws. `--bench` numbers, if the owner takes them, go in the README as "informational, RX 9060, driver 26.10.07.02"
and nowhere else; speed is not what this phase demonstrates.

**Measured, at 2560x1440 with `--nooverlay --noaniso`, on the owner's first machine.** The bar the plan set was met
everywhere: linear algebra against plain is **1 of 255** at **74.62 dB** on `pavingstones141_1k_c0_4` and **72.52 dB**
on `m1_m4_c0_3_c1_4` on the RTX 5090 under the all-fp16 row, **79.94 dB** and **77.79 dB** on the preview WARP under
the fp32-bias-and-result row it grants, and the two matrix layouts are **byte-identical to each other**. The 5090's
two figures are the Vulkan cooperative-vector path's numbers to every printed digit, per-channel means included: two
APIs, two instructions, the same fp16 weights, the same answer, arrived at independently, and that is the strongest
single piece of evidence the path is right. WARP against the 5090, both on the linear-algebra path, is 2 of 255 at
68.39 dB. `viewer_d3d12/README.md` and `viewer_d3d12/PHASE_B_NOTES.md` section 7 carry the table with the means.

**Three things about the gate came out differently.** The overlay arm is not the one the paragraph below describes:
there is no token to compare left of, so what it asserts is the strip's fifth line differing between the two paths,
and the rows above it identical, at 1280x720, 1920x1080 and 2560x1440 (note 9). The hardware-versus-WARP comparison
is **not** a gate arm: it was run by hand and its number recorded, because it needs two adapters and the gate's
linear-algebra arms run on the default device. And the expectation in the paragraph that closes this section -- that
the RTX 5090 and the integrated Radeon would SKIP and "the pass is WARP" -- was half wrong: the integrated Radeon does skip, by name, at Shader Model
6.8, but the 5090 runs every arm. `--bench` numbers did go into `viewer_d3d12/README.md`, informational, and they
are the RTX 5090's (7.0 us against 22.2 us at 640x480 over 20 frames) rather than the RX 9060 XT's.

Gate arms mirror `viewer_vk/README.md:572-581`: the two paths' frames within the bar, the refusal's exit code, `--bench`
running on both paths, the overlay identical left of the token's column -- each SKIPPED with the device's own reason
when a step of B.3 fails, which on the main machine is the expected outcome for the RTX 5090 (out of scope; its driver
question is not pursued) and for the integrated Radeon, and the pass is WARP.

## B.7 Stages and effort (estimates)

| stage | what | done when |
|---|---|---|
| B0, the spike (main machine, WARP; then the second machine) | Developer Mode on; `NNTC_D3D12_LINALG=ON` configure fetching the pinned packages; Microsoft's `sin-network` built and run; a `--linalg-query` mode (or a 150-line `nntc_linalg_query` tool) printing B.3's steps 2-7 for every adapter; one `ps_6_10` pixel shader with a `MultiplyAdd` through PSO creation, once with the DXBC VS and once with a DXC VS | one page of output per adapter; B.1.4 and B.4's questions answered |
| B1 | the weights buffer, the host fp16 conversion, the range bound, `nntc_view_linalg.hlsl` with `RowMajor`, the second PSO, `--linalg`, key K, the token, the diagnostics lines | B.6's linalg-versus-plain pair on WARP within the bar |
| B2 | `--bench`, the gate arms, the README section in the shape of `viewer_vk/README.md:392-586` | `run_checks.py` green with the option on (WARP arms pass, hardware arms SKIPPED with the reason) |
| B3 | the optimal layout through the conversion API (B.5) | both layouts within the bar on WARP; the README says which layout its numbers are from |
| B4, the owner's machine | the same build on the RX 9060 with driver 26.10.07.02: the query page, the pairs (hardware plain, hardware linalg, WARP linalg), the gate, the numbers into the README, a row in `PRIOR_ART_DISCLOSURE.md` section 8 beside the Vulkan one (`PRIOR_ART_DISCLOSURE.md:690`) | every number in the README is a run of the command beside it |

**Built, and stage B0 was not done as a separate thing.** There is no `nntc_linalg_query` tool, no `--linalg-query`
mode and no build of Microsoft's `sin-network` in this tree: the query page became the viewer's own
`query_linear_algebra`, printed at start-up on every run, and B.1.4's and B.4's questions were answered by building
the path and reading what the header and the runtime said. B1, B2 and B3 landed in one commit (`e5bfe21`) with the
`--linalg-layout` flag B3 needed, and B1's token was later replaced by the strip's fifth decode line (note 9); B4 is
the owner's run on the RX 9060 XT, which reported tier 1.0, the mandatory fp16 row, both conversion interfaces and
`decode: linear algebra`. What B4 did **not** produce is a frame comparison from that machine -- no `frame_diff`
number from the RX 9060 XT is in this tree -- and the row in
`PRIOR_ART_DISCLOSURE.md` section 8 is still not written (A-5).

**Human-equivalent**: 1 to 1.5 weeks for B0-B3 (estimate), plus the owner's day on the second machine. **At this tree's
agent pace**: the Vulkan viewer's cooperative-vector stage, from a working plain viewer to the committed path with
its query, conversion, guard, bench and gate arms, took the afternoon of 2026-09-15 (stage 3 at 17:01, stage 4 at 19:13,
git log); B0-B3 are the same work with more package plumbing, so **one afternoon to one day** (estimate), and B4 is
whatever the owner's runs take. The whole of Phase B waits on Phase A being green and on the owner having the second
machine ready; nothing in Phase A depends on it.

## B.8 Risks and notes

1. **The preview moves.** It already did once (B.1); the spec is a 0.9 draft; the granular query changed shape in 0.9;
   a header and the spec disagree on one name (B.1.4). Retired by pinned versions, one file of D3D12 names, and a
   README that says "on this SDK, this DXC, this driver" as `viewer_vk/README.md:523` does. **Note**: when the spec
   reaches 1.0, re-pin the three packages and re-read B.1.1 against it.
2. **Thin validation on a preview feature.** For the 2025 API a practitioner found the debug layer silent and PSO
   creation failing "with no indication why" (https://interplayoflight.wordpress.com/2026/02/21/adventures-in-neural-rendering-part-2-cooperative-vectors/);
   the 0.9 spec promises "descriptive errors". The query page, WARP as the oracle, DRED and GPU-based validation
   (A.2) and the two-path frame comparison are the defences.
   **Built, and they were not what caught the bugs.** Debug with the debug layer, GPU-based validation and DRED runs
   clean, on both paths and both builds. The defects that mattered -- the stride read back out of the query, a failed
   query printed as the device's answer, an optional path able to end a plain run, an install that shipped without the
   preview runtime -- were found by reading the code against the specification, not by any layer. The query page did
   earn its place: the `d3d10warp.dll` placement was diagnosed from its `D3D12Core.dll` and `d3d10warp.dll` lines.
3. **fp16.** The range bound (B.5); `EMULATED_OUTPUTS` and the wording of what was measured (B.1.1); the expectation
   of one of 255 and the rule that a larger difference is a bug, not a tolerance to widen
   (`viewer_vk/README.md:565-568`).
4. **Per-device rows.** Ask the mandatory fp16 row first, prefer the fp32-result row where granted, print which ran
   (B.3, step 6); never assume an fp32 result.
5. **Developer Mode** is a condition of the demonstration, not of the format: the plain path is the product, as
   `viewer_vk/README.md:394-396` says of its own optional stage.
6. **The compute-fallback note.** The proposal allows thread scope in every stage (B.1.2), so no compute alternative
   is planned. Were a driver to refuse the pixel stage, a full-screen compute decode would be the fallback, and since
   Shader Model 6.6 compute shaders have quad-based derivatives (`ddx`/`ddy` in 2D-tiled thread groups,
   https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html) (review), `SampleGrad`'s gradients
   would still be the hardware's; it would still cost a UAV round trip and the drop-in property the format is built
   on (`README.md:423-427`). A pixel-stage refusal would be a driver bug to report, not a design to adopt.
7. **Linux and translation layers.** The D3D12 viewer is Windows-only; vkd3d-proton does not implement SM 6.10 LinAlg
   (review; not verified here against https://github.com/HansKristian-Work/vkd3d-proton), so no Linux arm of Phase B
   exists or is planned.
8. **ARM64 packages.** Phase A needs no package. Phase B on the Snapdragon laptop would need the preview NuGet packages'
   ARM64 binaries and an Adreno driver with the tier, neither of which this study verified; Snapdragon is out of Phase
   B's scope (note only).
9. **Byte parity across the three viewers' overlays**: the token column rule of `viewer_vk/README.md:584-586` keeps the
   shared columns comparable; the D3D12 token sits at the same column as the Vulkan one so that the gate's constant
   (`tests/run_checks.py:616`, `VK_COOP_COLUMN`) serves both.
   **Built, and then found wrong.** The token was drawn at that column and has been removed. The strip is authored
   2560 pixels wide and drawn one pixel to one pixel from the left edge, so on any window narrower than that its
   right-hand end is off screen: the token was never once seen on a monitor here. `rasterise_overlay` draws a fifth
   line at the left margin instead, in every build, and the two Direct3D viewers' strips are compared over the
   **whole** 2560 columns of the four lines they share. `VK_COOP_COLUMN` remains, and serves the Vulkan viewer alone.

**Where the other notes stand.** Note 1 is unchanged and still the standing instruction: the spec is a 0.9 draft, the
pins are the three of B.1.3, and 1.0 means re-pinning and re-reading B.1.1. Notes 3, 4 and 5 were built as written --
the range bound, both rows asked for in order with the one that ran printed, and Developer Mode as a condition of the
demonstration rather than of the format. Note 6's compute fallback was never needed (B.1.2) and is deliberately not
implemented. Note 7 is unchanged and still unverified. Note 8 is partly retired: the CMake option does pick each
package's `arm64` folder on an arm64 build, but nothing arm64 has been built or run, and Snapdragon remains out of
scope.

## B.9 Decisions for the owner, Phase B, in order

* **B-1.** Go / no-go for Phase B after Phase A is green (nothing in A depends on B).
  **Answered: go**, the same day phase A landed.
* **B-2.** The option's name and default (`NNTC_D3D12_LINALG`, `OFF`), and that a retail build carries no
  `D3D12SDKVersion` export (B.2).
  **Answered: as proposed**, name and default both. A build without the option has no export, no package, no second
  shader and no preview anything, and its frames are byte for byte what they were.
* **B-3.** Packages fetched at configure time versus vendored under the tree (the tree vendors only single headers
  today, `CMakeLists.txt:260-262`).
  **Answered: fetched**, into `<build dir>/nntc_linalg_packages/`, about 100 MB, kept between configures, never
  written into the source tree and never committed. A machine with no network unpacks the three archives there by
  hand, which the *Reference* section of `docs/D3D12_LINALG_BUILD.md` spells out.
* **B-4.** The pinned versions: `1.721.3-preview`, `1.10.2605.37-preview`, WARP `1.65535.20-preview` (B.1.3), or
  whatever is current when B0 runs; and the re-pin when the spec reaches 1.0.
  **Answered: those three exactly**, and the re-pin at 1.0 stands as a standing instruction (B.8 note 1).
* **B-5.** Whether the DXC plain arm ships as `--dxc` (A-3) so that B.6's three-arm table is complete.
  **Answered: it does not ship**, and B.6 says what that costs.
* **B-6.** Whether stage B3 (the optimal layout) is done at all, given that correctness is the goal and `RowMajor`
  already exercises the instruction; it adds the conversion API's coverage and one more comparison.
  **Answered: done**, and it grew a flag, `--linalg-layout row|optimal`, so that either can be asked for. The optimal
  layout is the default where the runtime offers it, and the two draw byte-identical frames on the RTX 5090.
* **B-7.** Whether `--bench` numbers from the RX 9060 go in the README at all (informational) or only the query page
  and the precision table.
  **Answered: `--bench` numbers are in `viewer_d3d12/README.md`, marked informational**, but they are the RTX 5090's
  rather than the RX 9060 XT's, since that is the machine the path was measured on.

---

## 3. Hardware and drivers, as far as they bear on the two phases

**Phase A** needs a Direct3D 12 device at feature level 11_0 and nothing else; every GPU in the owner's fleet has one,
WARP included (the system WARP, no package).

**Phase B**, by device:

* **AMD Radeon RX 9060 16 GB (RDNA 4)** -- the second machine. Microsoft: linalg::Matrix "Supported on AMD Radeon RX 9000
  series graphics products" with "AMD Software: AgilitySDK Developer Preview Edition 25.30.41.02"
  (https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/); the 721 post names
  "AMD Developer Preview Edition 26.10.07.02" for its features on "RX 7000 and 9000 series hardware", with the
  LinAlg addition (VectorAccumulate) on RX 9000 (https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/).
  **(Review, correcting the first draft's reading)**: the RX 7000 mention covers the other 721 features; LinAlg itself
  is RX 9000-only in both posts, and the RX 9060 is an RX 9000. AMD's own word: "For supported AMD Radeon RX 9000
  Series GPUs, the AMD Software: AgilitySDK Developer Preview Edition 26.10.07.02 driver is required"
  (https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/). The driver is a separate public
  download from the regular Adrenalin package; its release notes
  (https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html and the
  25.30.41.02 page beside it) timed out on every attempt here and are unread -- the owner will have them on the machine.
  **Measured, and the card is an RX 9060 XT.** The owner ran the finished build on it: tier 1.0, the mandatory fp16
  row supported, both `Preview` interfaces present, and `decode: linear algebra`. So RDNA 4 runs a thread-scope
  linear-algebra multiply in a **pixel** shader. No frame comparison from that machine is recorded in this tree.
  One thing worth adding from `viewer_d3d12/PHASE_B_NOTES.md`: the preview driver **replaces** Adrenalin, so
  re-installing Adrenalin reverts the machine to a driver that answers 6.8.
* **WARP 1.65535.20-preview** -- "The WARP software device supports all these features"
  (https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/,
  https://www.nuget.org/packages/Microsoft.Direct3D.WARP). It is the reference on both machines and the only device
  on the main machine expected to run the path.
* **The main machine's RTX 5090 and integrated Radeon** are expected to SKIP Phase B's arms with a printed reason (the
  Radeon is not an RX 9000; NVIDIA's preview driver is not distributed publicly per its developer forum,
  https://forums.developer.nvidia.com/t/shader-model-6-10-preview-driver/368330, and NVIDIA is out of scope). **Intel**
  has announced LinAlg for Xe2 "in a future driver" (the 721 post) and is out of scope. Nothing was found for
  **Qualcomm**. Performance comparisons across vendors are not part of this plan; where `--bench` prints a number it is
  informational (B.6).
  **Wrong about the RTX 5090, and that is why Phase B exists in the shape it does.** The driver on the owner's first
  machine answers Shader Model 6.10 and grants the mandatory all-fp16 row (`SUPPORTED | TRANSPOSE`), so the whole path
  -- query, conversion, both layouts, the pipeline, the gate arms -- was written and measured there, and every hardware
  number in B.6 is that card's. The integrated Radeon was predicted correctly: it answers Shader Model 6.8 and is
  refused by name at step 3. Intel and Qualcomm were not tried.

**Not verified here**: the AMD release-note pages (timeouts); the per-device answers to B.3's steps 4-7 (only the query
on the machine tells); the bias alignment the driver requires; the exact enumerator names of the 1.721.3-preview header
(B.1.4); whether a DXBC vertex stage may share a PSO with a DXIL pixel stage (B.4).

**Where that list stands now.** The per-device answers were run on four devices and are in B.6 and in
`viewer_d3d12/PHASE_B_NOTES.md` section 5; the enumerator names are the header's (B.1.4); the mixed pipeline state is
refused (B.4). Still unverified: the AMD release-note pages, which have not been read here; the bias alignment a
driver actually requires, since the implementation uses 128 and a buffer rounded up to 128 past the bias without ever
testing less; arm64 anything (B.8 note 8); and Intel, Qualcomm and the translation layers.

---

## 4. Bottom line

* **Phase A** is straightforward Direct3D 12 work, verified by the same means as the Vulkan viewer against the D3D11
  frame: half a day to a day at this tree's agent pace, 1.5 to 2.5 weeks human-equivalent (estimates). It needs no
  package, no preview, no Developer Mode, and runs on every GPU the owner has. Do it first, and hold it to byte parity
  with the D3D11 viewer where anisotropy is off, recording rather than asserting the number.
* **Phase B** is one shader, one buffer and one query in the Vulkan viewer's order, behind a CMake option that is off
  by default and a run-time detection that never fails the program: one afternoon to a day at agent pace on WARP,
  about a week human-equivalent, then the owner's day on the RX 9060 with driver 26.10.07.02 (estimates). It
  demonstrates the same decode as one instruction through a vendor-neutral API, verified for correctness against the
  plain path with WARP as the reference; speed is not its claim.
* **What to do first in B**: the spike of B0 -- the package pins, Microsoft's `sin-network`, the query page on every
  adapter, one `ps_6_10` pixel shader through PSO creation with each vertex-stage variant. Its output settles B.1.4 and
  B.4 before a line of the path is written.
* **What not to do**: write against anything that mentions `D3D12_FEATURE_COOPERATIVE_VECTOR`, `MatrixRef` or Agility
  SDK 1.717; that API is gone from DXC (B.1).

**What actually happened, 2026-09-19.** Both phases were built and committed the same day the plan was written. Phase
A is the viewer the plan describes, and its frames are byte-identical to the Direct3D 11 viewer's on one adapter in
every state, with anisotropy on as well as off -- a stronger result than the plan asked for, and the reward for
compiling the same file to the same DXBC. Phase B is the optional path, and it runs on the RTX 5090, on the preview
WARP and on the owner's RX 9060 XT. Where the two paths have been compared, which is on the first machine, the
difference is 1 of 255 and the numbers are the Vulkan cooperative-vector path's to every printed digit. The spike of B0 was never a separate program; the questions it was to answer were
answered by building the thing. What the plan got wrong is listed at the top of this document, and the one that should
be read before anything else is built on it is the missing third shader arm (B.6): two pipelines are being compared,
not two instructions.

---

## 5. References

Specification and proposals

* https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html -- the runtime spec, v0.9 draft, June 2026
* https://raw.githubusercontent.com/microsoft/DirectX-Specs/master/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.md -- its source
* https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md -- HLSL Linear Algebra Matrix (Accepted, SM 6.10)
* https://github.com/microsoft/hlsl-specs/blob/main/proposals/0029-cooperative-vector.md -- HLSL Cooperative Vectors (Rejected, superseded)
* https://github.com/microsoft/hlsl-specs/blob/main/proposals/0026-hlsl-long-vector-type.md -- HLSL Long Vector Type (SM 6.9)
* https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_Derivatives.html -- derivatives in compute shaders since SM 6.6
* https://microsoft.github.io/DirectX-Specs/d3d/D3D12Redistributable.html -- D3D12SDKVersion / D3D12SDKPath, D3D12Core.dll placement

Microsoft announcements, downloads, documentation

* https://devblogs.microsoft.com/directx/cooperative-vector/ -- the 2025 preview (Agility SDK 1.717.1-preview, DXC 1.8.2505)
* https://devblogs.microsoft.com/directx/shader-model-6-9-retail-and-more/ -- SM 6.9 retail; cooperative vectors deprecated (February 26, 2026)
* https://devblogs.microsoft.com/directx/shader-model-6-10-agilitysdk-720-preview/ -- SM 6.10 preview and vendor support (April 27, 2026)
* https://devblogs.microsoft.com/directx/d3d12-linalg-preview/ -- the LinAlg preview post: prerequisites, API, drivers
* https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/ -- 1.721-preview, VectorAccumulate, WARP (May 28, 2026)
* https://devblogs.microsoft.com/directx/directx12agility/ -- the Agility SDK release list
* https://github.com/microsoft/DirectXShaderCompiler/releases -- DXC releases
* https://github.com/microsoft/DirectXShaderCompiler/releases/tag/v1.10.2605.2 -- the SM 6.10 preview compiler
* https://www.nuget.org/packages/Microsoft.Direct3D.D3D12 -- 1.619.6 retail, 1.721.3-preview
* https://www.nuget.org/packages/Microsoft.Direct3D.DXC -- 1.9.2607.13 retail, 1.10.2605.37-preview
* https://www.nuget.org/packages/Microsoft.Direct3D.WARP -- 1.0.20 retail, 1.65535.20-preview
* https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-d3d12enableexperimentalfeatures -- Developer Mode, E_NOINTERFACE, before device creation
* https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_shader_model -- the shader-model query
* https://learn.microsoft.com/en-us/windows/win32/direct3d12/using-d3d12-debug-layer-gpu-based-validation -- the debug layer and GPU-based validation
* https://learn.microsoft.com/en-us/windows/win32/direct3d12/use-dred -- Device Removed Extended Data
* https://learn.microsoft.com/en-us/windows/win32/direct3d12/timing -- timestamp queries, GetTimestampFrequency

Samples and accounts

* https://github.com/llvm-beanz/linalg-examples -- Microsoft's LinAlg samples (compute); `sin-network` pins 1.721.3-preview / 1.10.2605.37-preview / WARP 1.65535.20-preview
* https://raw.githubusercontent.com/llvm-beanz/linalg-examples/main/sin-network/sin-network.cpp -- the exports, the experimental-features call, the queries, the DXC arguments
* https://raw.githubusercontent.com/llvm-beanz/linalg-examples/main/cmake/LinalgSample.cmake -- the NuGet plumbing
* https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/ -- AMD MiniDXNN v0.4.0 (SM 6.10 LinAlg, driver 26.10.07.02, RX 9000)
* https://interplayoflight.wordpress.com/2026/02/21/adventures-in-neural-rendering-part-2-cooperative-vectors/ -- a pixel-shader account of the 2025 preview
* https://github.com/HansKristian-Work/vkd3d-proton -- the translation layer named in B.8 note 7

Vendors

* https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html -- AMD preview driver (unread: timeouts)
* https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-25-30-41-02.html -- AMD preview driver (unread: timeouts)
* https://forums.developer.nvidia.com/t/shader-model-6-10-preview-driver/368330 -- "NVIDIA does not distribute preview drivers publicly" (out of scope)

Vulkan counterpart, for the transliteration

* https://docs.vulkan.org/refpages/latest/refpages/source/VK_NV_cooperative_vector.html -- the extension the Vulkan viewer uses (`README.md:11`)
