# PHASE_B_NOTES.md -- diagnosing the Direct3D 12 linear-algebra decode path

**Who this is for.** Another agent, on the owner's second machine (Windows 11, an AMD Radeon RX 9060 XT 16 GB, RDNA 4,
AMD's preview driver 26.10.07.02), helping him find out why the optional Shader Model 6.10 decode path in
`nntc_view_d3d12` does not run, or runs and draws the wrong picture. It assumes nothing about any earlier conversation.
Everything below was measured on the owner's **first** machine (an RTX 5090, an integrated Radeon and WARP) unless it
says otherwise.

**The order to work in, and it is deliberate:**

1. **does it run at all** -- sections 1 to 6. Most of the guide.
2. **is the picture right** -- sections 7 and 8. This is the only acceptance bar the path has.
3. **is it fast** -- section 9, last, optional, and nothing depends on it.

The owner's words: *all we care about: does it work correctly.* A slow linear-algebra path is a success. A fast one
that draws the wrong picture is a failure.

---

## 1. What the feature is

Shader Model 6.10 "LinAlg": a matrix type in HLSL with a vector-matrix multiply-add, and the Direct3D 12 runtime
support for it. Only the **thread-scope vector-matrix** part is used here -- one thread multiplies one 18x24 fp16
matrix by one 24-vector and adds an 18-vector bias. The wave- and threadgroup-scope matrix-matrix parts of the same
feature are for products this format does not have.

* Runtime specification, **version 0.9 draft, June 2026**:
  <https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html>
* HLSL proposal **0035 "Linear Algebra Matrix"** (Accepted, SM 6.10):
  <https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md>
* The preview announcements: <https://devblogs.microsoft.com/directx/d3d12-linalg-preview/> and
  <https://devblogs.microsoft.com/directx/announcing-agilitysdk-721-preview-and-more-shader-model-6-10-features/>

**This is the second design of the feature.** The first, "cooperative vectors" (SM 6.9 preview, Agility SDK
1.717-preview), is deprecated and was **removed from DXC in 1.10.2605.2**. If anything on this machine mentions
`D3D12_FEATURE_COOPERATIVE_VECTOR`, `MatrixRef` or `MulAdd`, it is the old API and it will not compile. The Vulkan
viewer in `../viewer_vk/` uses `VK_NV_cooperative_vector`, which is a *different, NVIDIA-only* extension and is not
related to anything here except by analogy.

The viewer's own design document is `../docs/D3D12_LINEAR_ALGEBRA_PLAN.md`, part II.

---

## 2. What has to be installed, and by whom

`../docs/D3D12_LINALG_BUILD.md` is the same ground written for someone who only wants to build it. This section is
here so that this file stands on its own when the build has already been made and the question is why it does not run.

### 2.1 What the build fetches by itself

Nothing has to be installed by hand. Configuring with `-DNNTC_D3D12_LINALG=ON` downloads three preview NuGet packages
(they are ordinary zip archives) into the **build directory** and unpacks them there. Nothing is written into the
source tree and nothing is installed onto the machine.

| package | pinned version | URL |
|---|---|---|
| `Microsoft.Direct3D.D3D12` (Agility SDK) | `1.721.3-preview` | `https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/1.721.3-preview/microsoft.direct3d.d3d12.1.721.3-preview.nupkg` |
| `Microsoft.Direct3D.DXC` (the compiler) | `1.10.2605.37-preview` | `https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.dxc/1.10.2605.37-preview/microsoft.direct3d.dxc.1.10.2605.37-preview.nupkg` |
| `Microsoft.Direct3D.WARP` (the reference device) | `1.65535.20-preview` | `https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp/1.65535.20-preview/microsoft.direct3d.warp.1.65535.20-preview.nupkg` |

They land in `<build dir>/nntc_linalg_packages/<id>.<version>/`, and are kept: re-running `cmake` does not fetch
them again. About 100 MB is **downloaded** -- 34, 47 and 16 MB of `.nupkg` -- and the three unpack to roughly **363
MB** beside those archives, so the directory wants something over 460 MB of disk rather than 100.

The pins are in `../CMakeLists.txt`, in the `NNTC_D3D12_LINALG` block
(`NNTC_LINALG_AGILITY_VERSION`, `NNTC_LINALG_DXC_VERSION`, `NNTC_LINALG_WARP_VERSION`). **Do not float them.** The
specification is a 0.9 draft; a different Agility SDK may rename an enumerator, and a different DXC may change what
`dx/linalg.h` declares.

On a machine with no network, download the three `.nupkg` files elsewhere, unpack each one into
`<build dir>/nntc_linalg_packages/<id>.<version>/` so that `build/native/` is directly under it, and configure again --
the build skips the download when that directory is already there.

### 2.2 What the owner must do himself

**Nothing should be needed, but verify both:**

1. **Developer Mode must be ON.** `D3D12EnableExperimentalFeatures` returns `E_NOINTERFACE` without it, and that is
   step 2 of the detection chain in section 5. It was already on on both of the owner's machines. To verify without changing
   anything:

   ```
   reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /v AllowDevelopmentWithoutDevLicense
   ```

   `0x1` means on. If it reads `0x0` or the value is missing, **ask the owner to turn it on himself** -- Settings >
   System > For developers > Developer Mode > On. It is a machine-wide Windows setting and an agent must not flip it.

2. **The AMD preview driver.** LinAlg on Radeon needs **AMD Software: AgilitySDK Developer Preview Edition
   26.10.07.02**, which is a *separate public download* from the regular Adrenalin package
   (<https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/> names it; the release notes are
   at <https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html>). The
   owner says it is installed. If section 5 says the Shader Model is 6.8, the running driver is not it -- **ask the
   owner to install or re-install it**; a driver install is machine-wide and an agent must not do it.

Nothing else. No Windows SDK component, no Visual Studio workload, no PATH change, no admin action.

---

## 3. Building with the option on

Build into a **separate directory**, so the ordinary build stays exactly what it is:

```
cmake -S . -B build_linalg -G "Visual Studio 18 2026" -DNNTC_D3D12_LINALG=ON
cmake --build build_linalg --config Release --target nntc_view_d3d12
```

Use the **same generator and the same platform argument** as the ordinary `build/` directory -- which here means no
`-A` at all. The release gate's last arm configures one shared scratch build under `out/` using the generator and
platform of whichever build directory it was pointed at, and two build directories that disagree about the platform
make that one cache unusable (`CMake Error: generator platform: x64 / Does not match the platform used previously`).
If you hit it, the fix is to make the two agree, or to delete `out/cpu_only_build_windows`.

Configure prints one of two lines. Read it before anything else:

```
-- nntc_view_d3d12: the linear-algebra path is compiled IN (Agility 1.721.3-preview, DXC 1.10.2605.37-preview, WARP 1.65535.20-preview, x64); whether it RUNS is still Developer Mode and the device's own query
```

or a `CMake Warning` naming what could not be fetched, after which **the viewer still builds, without the path**. A
missing preview package must never fail a build; the plain decode is the product.

The build then puts these beside `nntc_view_d3d12.exe`, and **where each one sits matters**:

| file | where | why there |
|---|---|---|
| `D3D12Core.dll` | `D3D12\` | the folder the `D3D12SDKPath` export names |
| `d3d12SDKLayers.dll` | `D3D12\` | the preview debug layer, matched to that runtime |
| `d3d10warp.dll` | **beside the .exe, not in `D3D12\`** | measured: a copy under `D3D12\` is **not** picked up and the run silently gets the system WARP, which answers Shader Model 6.8 and tier 0. This is what the WARP package's own `.targets` file does, and the plan's section B.2 was wrong about this one file |
| `dxcompiler.dll`, `dxil.dll` | beside the .exe | the preview compiler and its signing library, loaded at run time |
| `dx\linalg.h` | `dx\` beside the .exe | what that compiler `#include`s |
| `nntc_view_linalg.hlsl` | beside the .exe | the second pixel shader, compiled at run time |
| `nntc_view.hlsl` | beside the .exe | the plain shader, as always |

`dxcompiler.dll` is loaded with `LoadLibraryEx` rather than linked (`load_dxc` in `main.cpp`), so an executable that
has lost *that* file still runs and still draws the plain path -- it says
`linear algebra: not available - dxcompiler.dll is not beside this executable ...` instead of failing in the loader.

**But the `D3D12\` folder is not like that, and this is the trap.** An option-ON executable exports `D3D12SDKVersion`,
and the Direct3D 12 runtime then **refuses to create any device at all** unless it finds that version under `D3D12\`
beside the executable. It does not fall back to the system runtime. A copy of the .exe on its own prints:

```
device 0: ... - no Direct3D 12 device at feature level 11_0
device 1: ... - no Direct3D 12 device at feature level 11_0
ERROR: no Direct3D 12 device: none of the 2 adapter(s) offers one at feature level 11_0
```

on **every** adapter, including WARP, and exits 1. If you see that from a build that worked a minute ago, you are
running a copy of the .exe without its `D3D12\` folder. (This is also why the release gate, when it copies the
executable into a scratch directory to test a broken shader, copies these files with it.)

**Without** `-DNNTC_D3D12_LINALG=ON` none of the above exists: no package, no `D3D12SDKVersion` export, no second
shader, and the frames are byte for byte what they were. That is checked in section 10.

---

## 4. Running it

Every command block in this file is **PowerShell**, which is the shell on both of the owner's machines: `$V` and `$A`
are PowerShell variables and `&` is how it runs a program named by one. The same lines in `cmd.exe` would need
`set V=...` and `%V%`, and pasting one shell's into the other silently does the wrong thing rather than failing.

`--shot` refuses a directory that is not there -- `ERROR: cannot write '...': '...' is not a directory this program can
see` -- and `out_d3d12_linalg\` is not in the repository, so make it once, here, before the first shot:

```powershell
mkdir out_d3d12_linalg -Force | Out-Null
build_linalg\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --noaniso --shot out_d3d12_linalg\hw.bmp
```

`--shot` opens **no window at all**. Never launch the windowed viewer on the owner's desktop without asking.

The flags this path adds:

| flag | meaning |
|---|---|
| `--linalg 1` | the linear-algebra path, and a **refusal** (exit 1, one `ERROR` line) where the build or the device cannot -- a run that quietly fell back would be a measurement of the other path under this one's name |
| `--linalg 0` | the plain path, always honoured, on every build and every device |
| neither | the linear-algebra path where every step of section 5 passes, the plain path everywhere else, and the `decode:` line says which |
| `--linalg-layout row\|optimal` | which matrix layout: row-major as the host writes it, or the device's own multiply-optimal layout through `ConvertLinearAlgebraMatrix`. Default: the optimal one where the runtime offers it |
| `--bench N` | N headless frames timed. **Informational only** (section 9) |
| key `K` | switches path at run time in the windowed viewer; prints `decode: ...` |
| the overlay | the strip's fifth line, at its left margin, in one of the four wordings below |

Every Direct3D 12 build draws that fifth line, and it is the quickest way to see which decoder drew a frame. Its
four wordings, one per state:

```
Decode: linear algebra, through Shader Model 6.10
Decode: plain, though this device offers linear algebra
Decode: plain, no linear algebra - <this device's own reason, cut to fit the line>
Decode: plain, no linear algebra in this build
```

The reason in the third is section 5's, shortened; the whole of it is on stdout at start-up.

---

## 5. The detection chain, in order, with the exact line each step prints

Every run prints this block before it loads the asset (`query_linear_algebra`, called from `init_common`). Read it
top to bottom; the first step that stops prints `linear algebra: not available - <why>` and the plain path draws.

```
linear algebra: the Shader Model 6.10 query, step by step
```

**Step 0 -- the build.** In a build made *without* the CMake option there is no block at all, only:

```
linear algebra: not available - this build was made without the CMake option NNTC_D3D12_LINALG, so it carries no
Shader Model 6.10 shader, no Agility SDK and no preview runtime; the plain path is the product and runs everywhere
```

*Means:* you are running the wrong executable. Go back to section 3. (The stub `query_linear_algebra`, the `#else`
arm of the same `#if NNTC_D3D12_LINALG` the real one is in.)

**Step 1 -- which runtime actually loaded.**

```
  D3D12Core.dll: <path> (1.721.3.0)
  d3d10warp.dll: <path> (1.65535.20.0)
```

Both lines are printed on every run; on a hardware adapter the second one reads `not loaded by this process`, which is
correct and means nothing. **These are the two most useful lines in the output.** What you want to see is a path
**inside your build directory** and the version `1.721.3.0`. What you may see instead:

| what it says | what it means |
|---|---|
| you never get here, because every adapter said `no Direct3D 12 device at feature level 11_0` | the exports are there and the `D3D12\` folder is not, so the runtime refused to make a device rather than falling back. See the end of section 3 |
| a path under the Windows system32 directory, or a version like `10.0.26100.x` | the process is on the in-box runtime, which means the `D3D12SDKVersion` / `D3D12SDKPath` exports were dropped by the linker (check with `dumpbin /exports`). Everything below will then politely answer "no" and the run will look like a device that simply lacks the feature |
| `not loaded by this process` | the same |
| `d3d10warp.dll:` naming the Windows system32 copy, on the WARP adapter | the preview WARP was not picked up -- almost always because `d3d10warp.dll` was put in `D3D12\` instead of beside the .exe (section 3). WARP then reports Shader Model 6.8 and tier 0 |

**Step 2 -- experimental shader models.**

```
  experimental shader models: enabled before device creation
```

or the run stops with

```
linear algebra: not available - D3D12EnableExperimentalFeatures(D3D12ExperimentalShaderModels) returned 0x80004002
(E_NOINTERFACE: either Developer Mode is off, or the runtime that answered is the retail one, which refuses
experimental shader models by design)
```

*Means:* Developer Mode (section 2.2, item 1) or step 1's runtime. Those are the only two causes. The call is made
exactly once, before any device exists (`enable_experimental_shader_models`, called from the top of `create_device`),
because calling it twice with different lists puts every device into `DEVICE_REMOVED`.

**Step 3 -- the Shader Model.**

```
  shader model: 6.10 asked for, highest answered 0x6A (6.10)
```

`0x6A` is 6.10 and is what you want. Measured elsewhere: `0x68` (6.8) on the integrated Radeon and on the **system**
WARP. If this says 6.8 or 6.9 on the RX 9060 XT, the AMD preview driver is not the one running -- section 2.2, item 2.
The refusal reads:

```
linear algebra: not available - this device's highest shader model is 6.8 and the decode needs 6.10
```

**Step 4 -- the tier.**

```
  linear algebra tier: 0x10 (TIER_1_0)
```

`0x10` is `TIER_1_0`; `0x0` is `NOT_SUPPORTED` and stops the run with

```
linear algebra: not available - this device reports linear algebra tier 0x0 and the decode needs TIER_1_0 (0x10)
```

A device that answered 6.10 at step 3 and tier 0 here is a driver that has the Shader Model without the feature.

**Step 5 -- the two granular rows.** Both are printed, always:

```
  row: vector fp16 x matrix fp16 + bias fp16 -> fp16: SUPPORTED | TRANSPOSE
  row: vector fp16 x matrix fp16 + bias fp32 -> fp32: none
```

The **first** row is the one Tier 1 makes mandatory and it must say `SUPPORTED`; if it does not, the run stops with

```
linear algebra: not available - this device does not report the fp16 vector-matrix row that Tier 1 makes mandatory
(vector fp16 x matrix fp16 + bias fp16 -> fp16), so the decode has no row to run on
```

and that is a **driver bug worth reporting**, because Tier 1 requires it. The **second** row is preferred when it is
granted: an fp32 bias and result mean the accumulation is not forced through fp16, which is a different set of numbers
and is why the viewer says which row ran. Measured: the RTX 5090 grants only the first (`SUPPORTED | TRANSPOSE`); the
preview WARP grants both (`SUPPORTED | EMULATED_OUTPUTS` and `SUPPORTED`).

`EMULATED_OUTPUTS` is **not** a failure. An fp16 *result type* is not an fp16 *accumulation*: an implementation may
accumulate in fp32 and round at the end, and this flag is precisely how it says it converted the result. Print it,
quote it beside any number you publish, and assert nothing about the arithmetic from it. `TRANSPOSE` says the
implementation could also transpose an optimal-layout matrix of these types; this decode never transposes anything and
the flag is ignored.

**Step 6 -- the enumeration query.** Always the same line today:

```
  enumeration: not asked for - this Agility SDK's d3d12.h has no D3D12_FEATURE_LINEAR_ALGEBRA_OPERATION_ENUMERATION,
  and a v1 driver would not answer it
```

The 0.9 draft added a query that returns the driver's whole list of native configurations. **It is not in
1.721.3-preview's header at all**, so there is nothing to call; the draft also says a v1 driver (DDI `_0115_1`) would
not answer it. Its absence is a diagnostic, never a refusal. If a later Agility SDK adds it, this is the line to
replace.

**Step 7 -- the matrix layout.**

```
  matrix layout: the device's multiply-optimal, through ConvertLinearAlgebraMatrix (ID3D12DevicePreview and
  ID3D12GraphicsCommandListPreview are both present)
```

or `row-major, no conversion` when those two `QueryInterface` calls do not both succeed, or when `--linalg-layout row`
was given. Row-major is legal for a thread-scope load and needs no conversion API at all, so this is a fallback and not
a failure.

**Step 8 -- the compiler.** The query also checks that `dxcompiler.dll` can be loaded before it says "available":

```
linear algebra: not available - dxcompiler.dll is not beside this executable, and the Shader Model 6.10 shader is
compiled at run time; the build copies it there, so an executable moved on its own loses the path
```

**The verdict.**

```
linear algebra: available - tier 1.0, Shader Model 6.10, vector fp16 x matrix fp16 + bias fp16 -> fp16
```

**Step 9 -- the pipeline state**, printed with the shaders:

```
shader compiled: ...\nntc_view.hlsl
shader compiled: ...\nntc_view_linalg.hlsl (ps_6_10, fp16 bias and result, multiply-optimal layout)
```

A failure here prints DXC's own message (or the runtime's `0x...`) and then

```
linear algebra: not available - the Shader Model 6.10 pipeline state could not be built (the message above is the
compiler's or the runtime's own)
```

**On a capable machine that line FAILS the release gate, and is meant to.** A device that answered 6.10 and tier 1.0
and then could not build the pipeline is a broken shader, a `dxcompiler.dll` that would not load, or a resource that
would not allocate -- something about the build rather than about the machine. Only the range guard of step 10 is
skipped past. Section 7 has the whole rule.

**Note for anyone changing this**: the LinAlg pipeline's **vertex** shader is compiled by DXC (`vs_6_10`) and not by
`fxc`. A pipeline state whose vertex stage is DXBC and whose pixel stage is DXIL is **refused** --
`CreateGraphicsPipelineState` returns `E_INVALIDARG` (`0x80070057`) for that pair, measured on the RTX 5090 and on the
preview WARP. That answers the plan's open question B.4, and it is why `nntc_view_linalg.hlsl` carries its own
`VSMain`.

**Step 10 -- the asset's range bound**, printed while the asset loads:

```
  linear algebra: the 18x24 matrix as fp16 in the device's multiply-optimal layout, 1152 bytes at stride 0, with the
  fp16 bias at offset 1152; the buffer is 1280 bytes
  linear algebra: the largest this decoder's output can reach in fp16 is 4.58015 of 65504 (output row 9), so the range
  bound passes
```

The bound is per output row `r`: `sum over j of |W_rj| * max|phi_j| + |b_r| <= 65504`, with `max|phi_j|` taken from the
asset's own dequantisation range. It guards the **arithmetic**, not just the stored numbers: under the all-fp16 row the
products, the running sum and the result are fp16 too, so a decode can reach infinity with every stored weight
comfortably inside range. An asset that fails it is refused the path with its own line and drawn by the plain one
(`build_linalg_weights`).

**The start-up line that says what actually happened** is the last one:

```
decode: linear algebra
```

or `decode: plain`. The release gate requires this line to be there and accepts **either** word on its first probe
run, and its linear-algebra arms then assert which of the two each of their own runs printed -- two runs that printed
the same word compared one path with itself. What the gate decides skip-or-fail on is **not** this line: it is the
tier and shader-model lines of steps 3 and 4 and the last `linear algebra:` verdict, and section 7 is the rule.

---

## 6. When it says "not available": what to check first

In this order, and each one is a line you already have:

1. **Does it even start?** `ERROR: no Direct3D 12 device: none of the N adapter(s) offers one at feature level 11_0`,
   on a machine that plainly has a GPU, means an option-ON executable without its `D3D12\` folder. End of section 3.
2. **Which executable?** `linear algebra: not available - this build was made without the CMake option` means you ran
   `build\Release\...` instead of `build_linalg\Release\...`.
3. **`D3D12Core.dll:` line** -- is the path inside your build directory and the version `1.721.3.0`? If not, nothing
   below it means anything. Check that `D3D12\D3D12Core.dll` is beside the .exe; check that the .exe exports
   `D3D12SDKVersion` (`dumpbin /exports nntc_view_d3d12.exe`); rebuild.
4. **`E_NOINTERFACE` at step 2** -- Developer Mode (verify, do not change; section 2.2).
5. **Shader model 6.8 at step 3** -- the running driver is not the AMD preview one (section 2.2), or the adapter you
   picked with `--device N` is the wrong one. Run without `--device` first and read the adapter list.
6. **Tier 0x0 at step 4 on a device that answered 6.10** -- the driver has the Shader Model without the feature.
7. **`--device N`** -- on a machine with more than one adapter, the default is the *first adapter with a Direct3D 12
   device*, which may not be the RX 9060 XT. The adapter list is printed at start-up, WARP last:

   ```
   device 0: AMD Radeon RX 9060 XT (16384 MB, hardware)
   device 1: Microsoft Basic Render Driver (0 MB, WARP, software)
   drawing on device 0: AMD Radeon RX 9060 XT
   ```

   **WARP is always the last entry and it always works**, because the preview WARP implements the whole feature in
   software. If the hardware adapter fails and WARP succeeds, the problem is the driver and not this program -- that is
   exactly what WARP is here for.

### 6.1 Vendor caveats, before you chase a bug that is not yours

* **RX 9000 only, on AMD.** Microsoft's two preview posts name `linalg::Matrix` as "supported on AMD Radeon RX 9000
  series graphics products", and AMD's own MiniDXNN page says the 26.10.07.02 preview driver is required "for
  supported AMD Radeon RX 9000 Series GPUs". The 721 post's mention of RX 7000 covers that release's *other* features,
  not LinAlg. The RX 9060 XT is an RX 9000, so it is in scope; **an RX 7000 part is not**, and on one the right
  outcome is step 3 or step 4 of section 5 saying no.
* **Thread scope on RDNA 4 is reported to gather to WMMA across the wave.** That is, the hardware instruction underneath
  a thread-scope `MultiplyAdd` is a wave-wide matrix unit, and the driver collects the lanes' matrices into it. **Not
  verified on this tree's machines** -- it is background, and it matters for exactly two things: performance is not
  something this path can reason about locally (section 9, which says do not try), and a decode where the lanes of a
  quad disagree about which matrix to load is a shape this path never produces, since every lane loads the same matrix
  from the same offset.
* **Pixel-shader behaviour is undocumented in practice.** The HLSL proposal is explicit that "all operations on
  `Thread` scope matrices are available in all shader stages", and no wave-uniformity requirement applies to thread
  scope -- but every public sample Microsoft and AMD have published is a **compute** shader, so this viewer's pixel
  shader is an early user of that sentence. It works on the RTX 5090 and on the preview WARP (measured). If it does
  **not** work on the RX 9060 XT, that is a driver bug to report, not a design to work around: the fallback would be a
  full-screen compute decode, which costs a UAV round trip and the drop-in property the format is built on, and it is
  deliberately not implemented. Say so plainly and hand the owner the two frames.
* **WARP is software and implements the whole thing.** Its fp16-result row reports `SUPPORTED | EMULATED_OUTPUTS` and
  its fp32-result row reports `SUPPORTED`; there is no hardware matrix unit anywhere in it, so every row it grants is
  its own arithmetic. That is exactly why it is the reference: it is not a second vendor's driver, and it runs on both
  machines. Never read a WARP timing as a performance number.
* **NVIDIA and Intel are out of this phase's scope.** The plan expected the RTX 5090 to skip Phase B's arms entirely;
  the driver on the owner's first machine turned out to support the feature, which is why there are hardware numbers in
  section 7 at all. Intel announced LinAlg for Xe2 "in a future driver". Nothing was found for Qualcomm.
* **The specification is a 0.9 draft.** The granular query changed shape in 0.9; the enumeration query it added is not
  in the header this build uses (section 5, step 6); and a header and the spec disagree on one enumerator's name --
  this Agility SDK spells the granular query with the word repeated where the specification writes it once, which is
  the first of the three bullets under "Three things the spike settled" in `README.md` and the comment over the
  `CheckFeatureSupport` call in `query_linear_algebra`. When the spec reaches 1.0, re-pin the three packages and
  re-read the plan's section B.1.1 against it.

---

## 7. What "correct" looks like

**The bar is the picture.** The same asset, the same camera, the same flags, drawn once by each path, compared with
`tools/frame_diff.py`. Anisotropy off and the overlay off, because anisotropic tap placement is free to differ and the
strip's decode line differs by design:

```powershell
mkdir out_d3d12_linalg -Force | Out-Null
$V = "build_linalg\Release\nntc_view_d3d12.exe"
$A = "examples\pavingstones141_1k_c0_4_nntc.json"
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\la.bmp    --linalg 1
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\plain.bmp --linalg 0
python tools\frame_diff.py out_d3d12_linalg\la.bmp out_d3d12_linalg\plain.bmp --max-diff 4 --min-psnr 50
```

`$V` and `$A` stay set for the rest of the shell session, and every block below uses them.

**The bar is `--max-diff 4 --min-psnr 50`, and it is not a tolerance chosen to pass.** It is what 11 bits of mantissa
in the weights and in `phi` can do to an 8-bit output. An fp16 conversion error, a layout mismatch, a wrong `M` or `K`,
a padding bug or a stale barrier would all be **far** larger. If a comparison misses the bar, the answer is never to
widen the bar.

What was actually measured on the owner's first machine, 2560x1440, on `pavingstones141_1k_c0_4` and on
`m1_m4_c0_3_c1_4`:

| pair | asset | max \|diff\| | mean per channel | PSNR |
|---|---|---|---|---|
| RTX 5090, linear algebra vs plain (all-fp16 row, multiply-optimal) | `pavingstones141_1k_c0_4` | 1 of 255 | 0.0024 / 0.0017 / 0.0026 | 74.62 dB |
| RTX 5090, the same with `--linalg-layout row` | `pavingstones141_1k_c0_4` | 1 of 255 | 0.0024 / 0.0017 / 0.0026 | 74.62 dB |
| RTX 5090, row-major vs multiply-optimal | `pavingstones141_1k_c0_4` | **0** | -- | identical |
| RTX 5090, linear algebra vs plain | `m1_m4_c0_3_c1_4` | 1 of 255 | 0.0029 / 0.0054 / 0.0025 | 72.52 dB |
| preview WARP, linear algebra vs plain (fp32 bias and result) | `pavingstones141_1k_c0_4` | 1 of 255 | 0.0005 / 0.0006 / 0.0008 | 79.94 dB |
| preview WARP, linear algebra vs plain | `m1_m4_c0_3_c1_4` | 1 of 255 | 0.0008 / 0.0010 / 0.0014 | 77.79 dB |
| preview WARP linear algebra vs RTX 5090 linear algebra | `pavingstones141_1k_c0_4` | 2 of 255 | 0.0111 / 0.0093 / 0.0079 | 68.39 dB |

**What these differences have NOT been shown to be.** They are of the size fp16 rounding in the weights and in `phi`
accounts for, and the agreement with the Vulkan path below is strong evidence that the arithmetic is what moves them.
But **the two pipelines differ by more than the one decode instruction**: the linear-algebra pipeline's **vertex**
stage is compiled by DXC for `vs_6_10` while the plain pipeline's is `fxc`'s DXBC, because a pipeline state that mixes
the two is refused (section 5, step 9). A last-bit difference in a clip-space position can move a pixel of an edge,
and that is not fp16 rounding in the decode. Separating the compiler from the instruction needs a **third** arm -- the
plain shader compiled by DXC -- which is the plan's B.6 and is **not implemented**. So the numbers in this table are
what the two pipelines *together* do; do not attribute all of them to fp16 rounding. The same sentence is in
`main.cpp`, in `load_shader`, in the comment over the second pipeline.

**The RTX 5090 rows are the Vulkan cooperative-vector path's numbers to every printed digit** -- 74.62 dB with means
0.0024 / 0.0017 / 0.0026 and 72.52 dB with means 0.0029 / 0.0054 / 0.0025 are exactly what `../viewer_vk/README.md`
records for those two assets. Two different APIs, two different instructions, the same fp16 weights, the same answer,
arrived at independently. That is the strongest single piece of evidence that this path is right, and it is the first
thing to check against if you change anything here. WARP is tighter than either because it grants the fp32-result row,
so only the weights and `phi` are fp16 and the accumulation is not.

On the RX 9060 XT the numbers will be its own. What must hold is the **bar**, and the shape: max 1 of 255 between the
two paths is what this decode does at 8 bits out, and anything past 4 is a defect.

**WARP is the reference.** It implements the whole feature in software and it runs on every machine, so:

* if **WARP's** two paths agree and the **hardware's** do not, the driver is wrong;
* if **neither** agrees, this program is wrong;
* a hardware-vs-WARP gap beyond the bar points at the driver's accumulation or at a layout bug and is worth knowing.

Run the WARP arm by naming WARP's index -- and **read the adapter list a run prints and name its last entry** rather
than copying the number below. `--device 1` is WARP on a machine whose adapter list is the two of section 6, and it is
`--device 2` on the owner's first machine, which has three. Naming the wrong one measures a hardware adapter under
WARP's name, or, on an adapter that cannot take the path, refuses the first line with
`ERROR: --linalg 1 was asked for and this device cannot: this device's highest shader model is 6.8 and the decode
needs 6.10`, writes no `.bmp`, and leaves `frame_diff.py` complaining about a file that is not there:

```powershell
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\warp_la.bmp    --device 1 --linalg 1
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\warp_plain.bmp --device 1 --linalg 0
python tools\frame_diff.py out_d3d12_linalg\warp_la.bmp out_d3d12_linalg\warp_plain.bmp --max-diff 4 --min-psnr 50
```

Also compare the two **layouts** against each other -- they are two different buffers read by two different loads and
must draw one picture:

```powershell
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\row.bmp --linalg 1 --linalg-layout row
& $V $A --nooverlay --noaniso --shot out_d3d12_linalg\opt.bmp --linalg 1 --linalg-layout optimal
python tools\frame_diff.py out_d3d12_linalg\row.bmp out_d3d12_linalg\opt.bmp --max-diff 4 --min-psnr 50
```

And the gate, which does all of the above and more:

```powershell
python tests\run_checks.py
```

Its Direct3D 12 case runs the baseline arms with `--linalg 0` and then the linear-algebra arms: the two paths' picture,
the two layouts' picture, the `--linalg 1` refusal on a device that cannot, `--bench` on both paths, and the two
paths' overlay strips identical above the decode line and differing in it, at 1280x720, 1920x1080 and 2560x1440.

**What it skips and what it fails**, because the two are not the same and the difference is the point of the rule
(`d3d12_viewer_checks` in `tests/run_checks.py`):

* it decides whether the machine is **capable** from the **diagnostic** lines and not from the verdict: the
  linear-algebra tier line of step 4 reading `0x10 (TIER_1_0)`, and the shader-model line of step 3 answering 6.10 or
  better. Both are printed before anything else in the run can go wrong;
* it reads the **last** `linear algebra: available` / `not available` line and not the first, because the query can say
  available and a later step -- the pipeline state of step 9, the range guard of step 10 -- can still take the path
  away, and reading the first line would send the arms off to demand a path the run had already given up on;
* a **capable** machine that then lost the path is a **FAILURE**, not a skip. That is a broken shader, a missing or
  unloadable `dxcompiler.dll`, or a resource that would not allocate -- exactly the three things a skip would have
  hidden. The one exception is the **asset's** own fp16 range guard (step 10), which the gate recognises because its
  message names 65504 and nothing else the viewer can say does; it also pins that wording in `main.cpp`, so a reword
  cannot quietly turn that skip into a failure;
* a build compiled **without** the CMake option, and a device that genuinely cannot -- too low a Shader Model, or tier
  0 -- are **skips**, each with its own reason, and the summary line says which it was;
* a build made **with** the option that then reports **no Direct3D 12 device at all** is a **FAILURE** too. For that
  build it means the preview runtime is missing from `D3D12\` beside the executable (the end of section 3), not that
  the machine lacks Direct3D 12, and the viewer prints a hint sentence naming its `D3D12SDKVersion` export which only
  such a build can print. That is how the gate tells the two apart without guessing.

So a green gate on a machine like this one is a gate that **ran** these arms, and a skip here is a claim about the
build or the device that the run printed in its own words.

---

## 8. When it runs but the picture is wrong

"Wrong" means the comparison in section 7 missed `--max-diff 4 --min-psnr 50`. Check these in order; the first three
account for nearly everything.

1. **Is it actually the other path?** Both runs print `decode:` at start-up. If both say the same thing, you compared
   one path with itself. The gate asserts this explicitly.

2. **fp16 range.** Look for the `the largest this decoder's output can reach in fp16 is X of 65504` line. If `X` is
   close to 65504 the decode is at the edge of the type and individual pixels can be far off even though the guard
   passed; if the picture has large flat wrong-coloured regions, that is an infinity or a NaN spreading. Try a
   different asset. The guard is the **first** substantive block of `build_linalg_weights`, right after the padded
   matrix and bias are read out of `DecoderConstants` and before anything is converted, allocated or recorded -- it
   decides whether there is a path at all. Only its `so the range bound passes` message is printed near the end of the
   function, beside the buffer's own line, which is where a reader sees it in the output. What is at the end of the
   function is the barrier block of item 4 below.

3. **Layout conversion.** Run `--linalg-layout row`. Row-major needs no conversion API at all, so if **row-major is
   right and multiply-optimal is wrong**, the fault is in `ConvertLinearAlgebraMatrix` or in what is handed to it --
   and that is a clean, reportable driver finding. Check, in `build_linalg_weights`:
   * `DestSize` comes from `GetLinearAlgebraMatrixConversionDestinationInfo` and is **not** guessed. `DestStride` is
     an INPUT to that query, not an answer: the application passes 0 for the multiply-optimal layout, whose addressing
     is opaque, and 48 (24 fp16 columns) for row-major. Reading it back out of the struct would be relying on a
     contract the API does not offer;
   * `SrcDataType == DestDataType == FLOAT16`. The conversion here changes the **layout only**; fp32 to fp16 is done on
     the host (`float_to_half`) so that it is identical to the Vulkan path's and so that the range
     guard sits where the numbers are;
   * `SrcStride` is 48 -- one row of 24 fp16 -- and a multiple of 16, as the spec requires.

4. **Barriers and states.** In the multiply-optimal path: the source buffer is `NON_PIXEL_SHADER_RESOURCE`, the
   destination is created with `ALLOW_UNORDERED_ACCESS` and is in `UNORDERED_ACCESS` when the conversion runs, then a
   **UAV barrier**, then a transition to `PIXEL_SHADER_RESOURCE`. A missing UAV barrier reads as a matrix of garbage,
   which looks like noise rather than like "slightly off". The whole sequence is in one block at the end of
   `build_linalg_weights`.

5. **Padding.** `M` is 18 and `K` is 24 always -- they are template arguments and must be compile-time constants, and
   this format's `nout` and `nin` vary per asset, so the matrix is zero-padded and `phi` is zero past `nin`. A zero row
   feeds an output that is ignored and a zero column multiplies a zero feature, so the padded product equals the
   unpadded one **in exact arithmetic**. If someone "optimises" the padding away the shapes stop being constants and
   nothing compiles; if someone leaves the padding non-zero, the extra outputs are wrong.

6. **Row versus column major.** The format publishes `W` row-major and `MatrixLayoutEnum::RowMajor` means the same
   order: `result[r] = sum over c of input[c] * matrix[stride * r + c]`. Nothing here transposes anything. A
   transposed matrix draws a recognisable picture in the wrong colours -- if the image looks like a channel swap,
   check this first.

7. **The bias.** Its type follows the row that was granted -- fp32 under the preferred row, fp16 under the mandatory
   one -- and the viewer says which. It sits at the next 128-byte boundary after the matrix, and the buffer is rounded
   up to 128 past its end because a long-vector load of eighteen entries may read in wider units than the bias's own
   length. The offset is handed to the shader in the decoder constant buffer (`linalg.x`), never baked into the shader.

8. **The shader itself.** `bin/nntc_view_linalg.hlsl` is `../viewer/bin/nntc_view.hlsl` with `DecodeNNTC` replaced and
   **nothing else changed** -- the same two `SampleGrad` calls, the same dequantise-after-sampling, the same feature
   order, the same clamp, the same renormalisation. If you have to edit it, edit only `DecodeNNTC`; any other change
   makes the two paths two readings of the format instead of one, and the comparison stops meaning anything.

9. **A last resort that costs nothing**: run the same comparison on WARP (`--device` its index). WARP is software and
   is not wrong. If WARP's two paths agree to 1 of 255 and the RX 9060 XT's do not, you have a driver finding, with two
   frames and one command line to reproduce it.

---

## 9. When it is slower (last, and optional)

Speed is **not** a goal of this viewer and is not an acceptance bar for this path. The frame loop waits for the GPU at
the end of every frame on purpose. `--bench N` exists only so a number can be quoted beside a run rather than
estimated, and it prints `(informational)` for that reason:

```powershell
& $V $A --nooverlay --bench 20 --size 640 480 --linalg 1
& $V $A --nooverlay --bench 20 --size 640 480 --linalg 0
```

`--nooverlay` is part of the measurement and not a tidiness flag: `--bench` times `record_scene`, and `record_scene`
ends with the overlay draw unless the overlay is off.

`--bench` and `--shot` are refused together: one times frames and writes nothing, the other writes one frame and times
nothing.

If the linear-algebra path is slower than the plain one, **that is an acceptable outcome and nothing needs doing**. Do
not tune it, and do not spend time on it. If the owner asks anyway, the two things worth saying are: the input vector
is written one component at a time, which the specification itself warns "may be suboptimal" and which this format's
feature vector forces (`phi` is built by a loop with a running index); and the row-major layout is not the layout the
hardware wants, so `--linalg-layout optimal` is the one to quote.

For the record, on the RTX 5090 at 640x480, 20 frames: linear algebra mean 7.0 us / min 5.4 us, plain mean 22.2 us /
min 20.1 us per scene draw.

---

## 10. The rule that outranks everything above

**A build without `-DNNTC_D3D12_LINALG=ON` keeps the frames, the constant-buffer bytes, the descriptor layout, the
root signature and the binary's dependencies that it had before this path existed, and its frames stay byte-identical
to the committed ones.** That is the claim, and it is what the `#if NNTC_D3D12_LINALG` guards in `main.cpp` are there
to keep.

It is **not** the claim that nothing at all changes, because three things do. An option-off build still:

* prints one extra line on stdout, from the stub `query_linear_algebra`: `linear algebra: not available - this build
  was made without the CMake option NNTC_D3D12_LINALG, so it carries no Shader Model 6.10 shader, no Agility SDK and
  no preview runtime; the plain path is the product and runs everywhere`. Deliberate: every run says which decode path
  it could have taken, and the gate reads that line;
* accepts `--linalg 0|1`, `--linalg-layout row|optimal` and `--bench N`. All three are parsed outside the guards, so
  `--linalg 0` is a no-op, `--linalg-layout` changes nothing, `--bench` times the one path there is, and `--linalg 1`
  is refused with `ERROR: --linalg 1 was asked for and this device cannot: ...` and exit 1;
* lists all three in `--help`.

The key `K` is the one thing that really is absent: its `case` is inside the guards, so in an option-off build the key
is not handled, does nothing and prints nothing.

If you touch `main.cpp`, re-prove the frames. `before.bmp` is a frame of the **option-off** build as it was **before**
the change, and no command elsewhere in this file writes one, so the order matters:

**One -- before you edit anything**, with `build\` at the revision you are about to change:

```powershell
mkdir out_d3d12_linalg -Force | Out-Null
cmake --build build --config Release --target nntc_view_d3d12
build\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --noaniso --shot out_d3d12_linalg\before.bmp
```

**Two** -- make the change.

**Three** -- rebuild the same directory and compare:

```powershell
cmake --build build --config Release --target nntc_view_d3d12
build\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --noaniso --shot out_d3d12_linalg\after.bmp
python tools\frame_diff.py out_d3d12_linalg\after.bmp out_d3d12_linalg\before.bmp --max-diff 0 --min-psnr 100
python tests\run_checks.py
```

A frame from an option-ON build is not a substitute for `before.bmp`: on WARP the two builds' frames differ by up to 1
of 255 for the reason in the last bullet of this section. If step one was skipped there is nothing to compare against
and this check cannot be run; take `before.bmp` from the revision in `git`.

And the CPU arm of the gate, which has nothing to do with this path and must not notice it:

```powershell
build\Release\nntc_encode.exe tests\tiny.png -o out\tiny --backend cpu --quiet
```

Two things that legitimately differ in an option-**ON** build and are not regressions:

* the overlay strip's fifth line reads differently, because an option-OFF build says `no linear algebra in this
  build`, so two builds' strips are not the same bytes below row 84. The four lines above that row are, and the
  comparison against the Direct3D 11 viewer's strip runs over exactly those **84 rows and the whole 2560 columns**
  in either build;
* frames drawn on **WARP** differ from an option-OFF build's by up to 1 of 255 (108 dB measured), because the option-ON
  build uses the preview `d3d10warp.dll` and the option-OFF build uses the system one. Two different rasteriser builds.

---

## 11. Where everything lives

| what | where |
|---|---|
| the whole optional path in the viewer | `viewer_d3d12/main.cpp`, every `#if NNTC_D3D12_LINALG` block |
| the redistributable exports | the `extern "C"` block near the top of `main.cpp` (`D3D12SDKVersion`, `D3D12SDKPath`) |
| the DXC compile | `load_dxc` and `compile_dxil_file` |
| the second pipeline state, and the DXC vertex stage | the `#if NNTC_D3D12_LINALG` block at the end of `load_shader` |
| `float_to_half` | the Vulkan viewer's, so the two accelerated paths quantise identically |
| the weights buffer, the range bound, the conversion | `build_linalg_weights` |
| the raw SRV at `t3` and the descriptor-set layout | `write_raw_srv`, and the `SRV_PER_SET` constant it is counted out of |
| the overlay's decode line | `rasterise_overlay`, the `l4` block at the end of it: the left margin, `y = 4 + 4 * LINE_ADV` |
| the detection chain | `query_linear_algebra`, and its `#else` arm for a build without the option |
| the "no device at all" hint when the `D3D12\` folder is missing | inside `create_device`, under `this build carries the preview redistributable's D3D12SDKVersion export` |
| `D3D12EnableExperimentalFeatures` | `enable_experimental_shader_models`, called once from the top of `create_device` |
| `--linalg`, `--linalg-layout`, `--bench` | the argument loop in `main`, all three outside the guards; the `--bench` + `--shot` refusal is the `two different runs` message just after that loop |
| `--bench` itself | `render_bench`, which times `record_scene` and so includes the overlay draw unless `--nooverlay` |
| the shader | `viewer_d3d12/bin/nntc_view_linalg.hlsl`: the type is the `WMatrix` typedef, the load is `WMatrix::Load<LA_LAYOUT>`, the multiply is `MultiplyAdd<LA_SCALAR>` |
| the plain shader it is a transliteration of | `viewer/bin/nntc_view.hlsl` |
| the CMake option, the pins and the file copies | `CMakeLists.txt`, the `NNTC_D3D12_LINALG` block |
| the gate arms | `tests/run_checks.py`, `d3d12_viewer_checks`, the `D12_PLAIN` constant and the linear-algebra arms |
| the design, and every claim's source | `docs/D3D12_LINEAR_ALGEBRA_PLAN.md` part II |
| the Vulkan counterpart, for the analogy | `viewer_vk/main.cpp`, `viewer_vk/bin/view_coopvec.frag`, `viewer_vk/README.md` |

**House rules for any edit**: CRLF line endings; no absolute paths in a tracked file; match the comment density and
voice of the file you are in; never open a window on the owner's desktop (headless `--shot` only); never commit without
being asked.
