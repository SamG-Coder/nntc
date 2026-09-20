# The Direct3D 12 viewer's Shader Model 6.10 decode path

`nntc_view_d3d12` decodes in its pixel shader, as two nested loops over a small matrix of weights. Shader Model 6.10
adds a matrix type to HLSL and a `MultiplyAdd` over it, and the CMake option `NNTC_D3D12_LINALG` builds a second copy
of that shader with those loops replaced by one such call. It answers one question: does a vendor-neutral matrix
instruction draw the same picture as the loop?

**The option is OFF by default and you need none of this to use the viewer.** A build without it is the viewer
[`viewer_d3d12/README.md`](../viewer_d3d12/README.md) describes -- no preview package, no Agility SDK, no Developer Mode, the same frames byte for
byte. The plain path is the product. Everything here is for someone who wants the extra path as well.

## What you need to build

Most of this the tree already needs on Windows:

| | |
|---|---|
| **Windows** | the option exists on Windows and nowhere else |
| **Visual Studio** with *Desktop development with C++* and a Windows SDK | 2026 or 2022, from <https://visualstudio.microsoft.com/downloads/>. This target needs neither CUDA nor the Vulkan SDK |
| **CMake 3.20 or newer** | <https://cmake.org/download/>, or the copy bundled with Visual Studio |
| **a network connection, the first configure only** | CMake fetches three preview packages, about 100 MB, into the build directory |

Nothing is installed system-wide, nothing needs administrator rights, and nothing is written into the source tree.
There is no preview SDK to install, no `nuget.exe`, no Visual Studio extension and no PATH change. **No driver and no
Developer Mode are needed to build**; a machine with no suitable device still compiles this, and the executable it
produces still runs and draws the plain path. x64 and arm64 are supported; a 32-bit build is refused at configure time,
with a warning, and the viewer is built without the path.

## Build it

```
cmake -S . -B build_linalg -G "Visual Studio 18 2026" -DNNTC_D3D12_LINALG=ON
cmake --build build_linalg --config Release --target nntc_view_d3d12
```

With Visual Studio 2022 the generator is `-G "Visual Studio 17 2022"`; nothing else changes. Build into a separate
directory so the ordinary `build/` stays what it is and the two can be compared -- and give it **the same generator and
the same platform argument as the tree's other build directories**, which in this tree means no `-A` at all. The
release gate configures one shared scratch build under `out/` with the generator and platform of whichever build
directory it was pointed at, so two that disagree about the platform make that cache unusable and the gate stops with
`CMake Error: generator platform: x64 / Does not match the platform used previously`.

Configure prints one line that is the whole answer:

```
-- nntc_view_d3d12: the linear-algebra path is compiled IN (Agility 1.721.3-preview, DXC 1.10.2605.37-preview, WARP
1.65535.20-preview, x64); whether it RUNS is still Developer Mode and the device's own query
```

A `CMake Warning` in its place means the path was **not** compiled in and the viewer was built without it. Either way
configure succeeds and the build succeeds, which is deliberate: a missing preview package must never fail a build.

## What you need to run it

Two more things, neither of which the build can arrange for you.

**1. Developer Mode on.** `D3D12EnableExperimentalFeatures` returns `E_NOINTERFACE` without it, and experimental
shader models are how a preview runtime is allowed to answer 6.10 at all. To read the setting without changing it:

```
reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\AppModelUnlock" /v AllowDevelopmentWithoutDevLicense
```

`0x1` is on. It is turned on in Settings, under System > For developers > Developer Mode. It is machine-wide, so the
owner of the machine turns it on.

**2. Something that answers Shader Model 6.10 at linear algebra tier 1.** Any one of:

* **NVIDIA** -- a GeForce RTX card on its ordinary shipping driver. The figures below were measured on an RTX 5090 on
  driver 616.92, with no preview driver and no special install of any kind;
* **AMD** -- a Radeon RX 9000 series card with **AMD Software: AgilitySDK Developer Preview Edition 26.10.07.02**,
  whose release notes carry the download link:
  <https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-MS-AGILITY-SDK-26-10-07-02.html>. It is a
  separate public download from the regular Adrenalin package and it **replaces** it: installing Adrenalin again
  reverts the machine to a driver that answers Shader Model 6.8;
* **the preview WARP the build already copied**, which implements the whole feature in software and so runs on any
  machine that can build this at all. It is the last adapter in the start-up list and `--device N` picks it. WARP is
  also the right reference when a hardware result looks wrong: if WARP's two paths agree and the hardware's do not, the
  driver is the suspect.

**What this has actually run on.** Three, and only three:

| | |
|---|---|
| the preview WARP above | the software implementation, so it is the one machine-independent result |
| a Radeon RX 9060 XT | AMD Software: AgilitySDK Developer Preview Edition 26.10.07.02 |
| a GeForce RTX 5090 | the ordinary game-ready driver 616.92, with no NVIDIA preview driver |

No other card, driver, architecture or vendor has been tried, and the feature itself is a draft (see the end of this
file). A device that answers the query is expected to work and is not known to; if yours does something else, that is
worth reporting rather than working around, and [`viewer_d3d12/PHASE_B_NOTES.md`](../viewer_d3d12/PHASE_B_NOTES.md) is written for exactly that.

## Check it worked

Every run prints the query step by step, before it loads the asset, and ends it with a verdict:

```
linear algebra: the Shader Model 6.10 query, step by step
  D3D12Core.dll: <your build dir>\Release\D3D12\D3D12Core.dll (1.721.3.0)
  d3d10warp.dll: not loaded by this process
  experimental shader models: enabled before device creation
  shader model: 6.10 asked for, highest answered 0x6A (6.10)
  linear algebra tier: 0x10 (TIER_1_0)
  row: vector fp16 x matrix fp16 + bias fp16 -> fp16: SUPPORTED | TRANSPOSE
  ...
linear algebra: available - tier 1.0, Shader Model 6.10, vector fp16 x matrix fp16 + bias fp16 -> fp16
```

The two most useful lines are the first two: what you want is a path **inside your own build directory** at version
`1.721.3.0`. A step that fails prints `linear algebra: not available - <why>` naming that step, and the plain path
draws.

Then, once the pipeline exists, and again whenever the path changes:

```
decode: linear algebra
```

or `decode: plain`. That is the line to quote, and to check on **both** runs of a comparison: two runs that print the
same thing compared one path with itself. The overlay's last line names the decoder in use as well, in words, on every
build. The key `K` switches path at run time in the windowed viewer and prints the `decode:` line again. `--linalg 1`
asks for this path by name and is a **refusal** -- exit 1, one `ERROR` line saying why -- wherever the build or the
device cannot, because a run that quietly fell back would be a measurement of the other path under this one's name;
`--linalg 0` is the plain path and is always honoured.

**The comparison that settles it** is the same asset and the same camera drawn once by each path. Anisotropy off,
because anisotropic tap placement is free to differ; the overlay off, because it names the path and so differs by
design:

```powershell
mkdir out_linalg -Force | Out-Null
build_linalg\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --noaniso --linalg 1 --shot out_linalg\la.bmp
build_linalg\Release\nntc_view_d3d12.exe examples\pavingstones141_1k_c0_4_nntc.json --nooverlay --noaniso --linalg 0 --shot out_linalg\plain.bmp
python tools\frame_diff.py out_linalg\la.bmp out_linalg\plain.bmp --max-diff 4 --min-psnr 50
```

`--shot` opens no window at all; the viewer does not create the directory it writes into, which is what the `mkdir` is
for. `tools/frame_diff.py` needs numpy. `--max-diff 4 --min-psnr 50` is what 11 bits of mantissa can do to an 8-bit
output, not a tolerance chosen to pass: an fp16 conversion error, a layout mismatch, a wrong matrix dimension or a
stale barrier would all be far larger.

What that gives at 2560x1440:

| device | asset | largest difference | PSNR |
|---|---|---|---|
| RTX 5090 | `pavingstones141_1k_c0_4` | 1 of 255 | 74.62 dB |
| RTX 5090 | `m1_m4_c0_3_c1_4` | 1 of 255 | 72.52 dB |
| preview WARP | `pavingstones141_1k_c0_4` | 1 of 255 | 79.94 dB |

The two hardware figures are the Vulkan viewer's cooperative-vector numbers to every printed digit, per-channel means
included: two APIs, two instructions, the same fp16 weights, the same answer. WARP is tighter because it grants the
fp32-bias-and-result row, so the accumulation is not forced through fp16. On a Radeon RX 9060 XT with the AMD preview
driver the query reports tier 1.0 and the run prints `decode: linear algebra`.

`--linalg-layout row|optimal` chooses between the host's row-major fp16 matrix and the device's own multiply-optimal
layout through `ConvertLinearAlgebraMatrix`; they are different buffers read by two different loads and draw
byte-identical frames. `python tests\run_checks.py` runs this comparison and several more. `--bench N` times the scene
draw and is informational; nothing in this tree depends on it. [`viewer_d3d12/README.md`](../viewer_d3d12/README.md) holds the full table with the
per-channel means.

## When it does not work

[`viewer_d3d12/PHASE_B_NOTES.md`](../viewer_d3d12/PHASE_B_NOTES.md) is the diagnosis guide: the detection chain step by step, the exact line each step
prints, what each answer means, and what to look at when the picture misses the bar. Three cases belong here because
they are the build's and not the device's.

**The packages could not be fetched.** A failed download prints the url and the reason, then one summarising
`CMake Warning` naming the three packages it wanted and where it wanted them. Configure and the build then **succeed**,
the viewer is built without the path, every other target is untouched, and the executable runs and draws the plain
decode. A run of it says so, which is how you tell this case from a device that lacks the feature:

```
linear algebra: not available - this build was made without the CMake option NNTC_D3D12_LINALG, so it carries no
Shader Model 6.10 shader, no Agility SDK and no preview runtime; the plain path is the product and runs everywhere
```

See *Configuring without a network* below. A file missing from a half-unpacked package is named on its own line,
`nntc_view_d3d12: <path> is not there`, before the same warning.

**A copy of the executable on its own gets no Direct3D 12 device at all.** An option-ON executable exports
`D3D12SDKVersion`, and the Direct3D 12 runtime then refuses to create **any** device -- it does not fall back to the
system runtime -- unless it finds that version under `D3D12\` beside the executable. Every adapter, WARP included, then
reports:

```
device 0: ... - no Direct3D 12 device at feature level 11_0
ERROR: no Direct3D 12 device: none of the 3 adapter(s) offers one at feature level 11_0
```

and the run exits 1. If a build that worked a minute ago says that, you are running a copy of the .exe without its
`D3D12\` folder; copy the whole set in the table below with it.

**A copy without `dxcompiler.dll` degrades gracefully instead**, because `dxcompiler.lib` is deliberately not on the
link line and the DLL is loaded by hand. The executable still starts, still draws, and says
`linear algebra: not available - dxcompiler.dll is not beside this executable ...` rather than failing in the loader.

## Reference

### What lands beside the executable

Every build of the option-ON viewer puts these beside `nntc_view_d3d12.exe`. Where each one sits is not a detail.

| file | where | why |
|---|---|---|
| `D3D12Core.dll` | `D3D12\` | the folder named by the executable's `D3D12SDKPath` export, which is how a process gets a Direct3D 12 runtime other than the system one |
| `d3d12SDKLayers.dll` | `D3D12\` | the preview debug layer, which must match that runtime. A Debug build without it loses the debug layer and nothing else |
| `d3d10warp.dll` | **beside the .exe, not in `D3D12\`** | a copy under `D3D12\` is **not** picked up: the run silently gets the system WARP, which answers Shader Model 6.8 and linear algebra tier 0. Beside the executable is also where the WARP package's own `.targets` file puts it |
| `dxcompiler.dll` | beside the .exe | the preview compiler. The linear-algebra shader is compiled at run time, so this is a runtime dependency |
| `dxil.dll` | beside the .exe | that compiler's validator and signing library. It is not what makes this path run -- a runtime in experimental shader models, which this path requires anyway, accepts unsigned DXIL, and with `dxil.dll` deleted the shader still compiles and draws byte-identical frames. It ships so that what DXC emits is validated and signed, as it would have to be for a runtime not in that mode |
| `dx\linalg.h` | `dx\` beside the .exe | what the shader includes, read by that compiler at run time from the path the `#include` spells |
| `nntc_view_linalg.hlsl` | beside the .exe | the second pixel shader, compiled at run time, so it can be edited beside the executable |
| `nntc_view.hlsl` | beside the .exe | the plain shader, copied for every build of this viewer, option or no option |

### The three packages

Configuring with the option on downloads three **preview** NuGet packages, at versions pinned in the root
CMakeLists (`NNTC_LINALG_AGILITY_VERSION`, `NNTC_LINALG_DXC_VERSION`, `NNTC_LINALG_WARP_VERSION`), and unpacks them into
`<build dir>/nntc_linalg_packages/`. A `.nupkg` is an ordinary zip archive, which is why `file(DOWNLOAD)` and
`file(ARCHIVE_EXTRACT)` are enough and `nuget.exe` -- a fourth download, for nothing -- is not used. They come from
Microsoft's own registry over HTTPS, they land only in the build directory, they are never installed onto the machine
and never vendored into git, and deleting the build directory removes every trace of them.

| package | version | what it is for |
|---|---|---|
| `Microsoft.Direct3D.D3D12` | `1.721.3-preview` | the Agility SDK: the preview `d3d12.h` the viewer compiles against, and the `D3D12Core.dll` runtime that answers the Shader Model 6.10 and linear-algebra queries at all |
| `Microsoft.Direct3D.DXC` | `1.10.2605.37-preview` | the preview shader compiler, its `dxil.dll` signing library, and `dx/linalg.h`, which the linear-algebra shader includes |
| `Microsoft.Direct3D.WARP` | `1.65535.20-preview` | the preview software rasteriser, which implements the whole feature in software and is therefore the reference on a machine whose driver does not |

The urls the build fetches:

```
https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.d3d12/1.721.3-preview/microsoft.direct3d.d3d12.1.721.3-preview.nupkg
https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.dxc/1.10.2605.37-preview/microsoft.direct3d.dxc.1.10.2605.37-preview.nupkg
https://api.nuget.org/v3-flatcontainer/microsoft.direct3d.warp/1.65535.20-preview/microsoft.direct3d.warp.1.65535.20-preview.nupkg
```

The unpacked packages are about 360 MB and are **kept**: the build skips the download whenever
`<package dir>/build/native` already exists, so re-running `cmake` does not fetch them again, and deleting the build
directory is what re-fetches them.

**Do not float the pins.** The runtime specification they implement is a 0.9 draft: a different Agility SDK may rename
an enumerator, and a different compiler may change what `dx/linalg.h` declares.

### Configuring without a network

Fetch the three `.nupkg` files on a machine that has one, from the urls above, and unpack each -- they are zip
archives, so any unzip tool will do -- into `<build dir>/nntc_linalg_packages/<id>.<version>/`, so that each package's
own `build/native/` sits directly under it. The existence of `build/native` is exactly what the build tests before it
decides to download, so a correctly placed set means the download never happens. Then configure again.

These are the seven files the build checks for, and `x64` is `arm64` on an arm64 build:

```
build_linalg\nntc_linalg_packages\
  microsoft.direct3d.d3d12.1.721.3-preview\build\native\include\d3d12.h
  microsoft.direct3d.d3d12.1.721.3-preview\build\native\bin\x64\D3D12Core.dll
  microsoft.direct3d.dxc.1.10.2605.37-preview\build\native\include\dxcapi.h
  microsoft.direct3d.dxc.1.10.2605.37-preview\build\native\include\hlsl\dx\linalg.h
  microsoft.direct3d.dxc.1.10.2605.37-preview\build\native\bin\x64\dxcompiler.dll
  microsoft.direct3d.dxc.1.10.2605.37-preview\build\native\bin\x64\dxil.dll
  microsoft.direct3d.warp.1.65535.20-preview\build\native\bin\x64\d3d10warp.dll
```

Unpacking a `.nupkg` gives all of this and a good deal more, and the rest is simply not read.

### Checking a download by hand

The build verifies the transfer and the status code and nothing else -- the `file(DOWNLOAD)` call carries no
`EXPECTED_HASH` -- so if you want to check what arrived, this is what these three pinned versions are, over the
`.nupkg` files in `<build dir>/nntc_linalg_packages/`. Use `sha256sum` or `certutil -hashfile <file> SHA256`.

| file | bytes | SHA-256 |
|---|---|---|
| `microsoft.direct3d.d3d12.1.721.3-preview.nupkg` | 35353560 | `0131bce1e4bace3fc08c03018c29a09ede2570b263721c6449b0ec75762ab22d` |
| `microsoft.direct3d.dxc.1.10.2605.37-preview.nupkg` | 49260377 | `8cf5ec1a8abb832de43251bae1ffcec89b22240677c599c38d30a18dee2f4de6` |
| `microsoft.direct3d.warp.1.65535.20-preview.nupkg` | 15934355 | `dc935a30e13e35ff4d4a270889b2c1282bdd5e3dee79a4bc7934fc87351111bc` |

### The specification is a 0.9 draft

The Direct3D 12 linear-algebra runtime feature is a **0.9 draft** as of June 2026, and the three pinned packages are
preview builds of it. That is why the versions are pinned rather than floated, why the granular query is spelled the
way this Agility SDK's header spells it rather than the way the specification writes it, and why one query the draft
describes is not called at all -- it is not in this header. When the specification reaches 1.0, these pins, this
document and the plan's section B.1.1 are re-read against it. Until then, a newer Agility SDK or a newer compiler is a
change to test, not a change to assume.

### Further reading

* [`viewer_d3d12/PHASE_B_NOTES.md`](../viewer_d3d12/PHASE_B_NOTES.md) -- the diagnosis guide, for a machine where the path does not work.
* [`D3D12_LINEAR_ALGEBRA_PLAN.md`](D3D12_LINEAR_ALGEBRA_PLAN.md) beside this file -- the design record.
* The runtime feature specification:
  <https://microsoft.github.io/DirectX-Specs/d3d/D3D12LinearAlgebraRuntimeFeatureSupport.html>
* HLSL proposal 0035, the matrix type and `MultiplyAdd`:
  <https://github.com/microsoft/hlsl-specs/blob/main/proposals/0035-linalg-matrix.md>
* AMD on its own neural texture compression through this feature, which is where the preview driver above is
  named: <https://gpuopen.com/learn/minidxnn-v040-interactive-neural-texture-compression/>
