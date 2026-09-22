# The NNTC viewer in the browser (WebGPU)

> This fork also includes a browser encoder at `encode.html`, powered by CUDA
> WebShader. Start it from the repository root with `npm ci` and `npm start`;
> initialize the pinned `vendor/cuda-webshader` submodule first. See
> [the encoder guide](../docs/BROWSER_ENCODER.md). The instructions below apply
> to the viewer by itself.

This page draws an NNTC asset the way a game would: two latent textures on a real hardware sampler, and one small
affine layer over two samples. It is the same decode the Direct3D 11, Direct3D 12 and Vulkan viewers run, ported to
WGSL. The controls are ordinary HTML on the left, the picture is WebGPU on the right, and a two-line strip is drawn
over it in the native viewers' own font.

Nothing is compiled. There is no WebAssembly, no bundler, no package manager and no third-party code of any kind -
plain JavaScript, plain HTML, plain CSS, and one small Python server. Clone the tree, serve this directory, open it.

## Run it

```
python webgpu/webserver.py
```

It prints the address it is on:

```
Serving <this directory> at http://localhost:8082 - open that in a browser with WebGPU
```

Open that, and the first bundled material appears by itself. The current directory does not matter: the server
roots itself at the directory the script is in, so `webgpu/` is the document root and nothing above it is reachable.

Under WSL 2 the command is `python3`, and the Windows tree is reached under `/mnt/c`:

```
cd /mnt/c/<wherever the tree is>/nntc
python3 webgpu/webserver.py
```

WSL 2 forwards `localhost`, so the page is at the same address in a Windows browser. Nothing else differs; the
server is Python's own `http.server` and needs no package on either side.

`file://` does not work in any browser. WebGPU needs a secure context, which means HTTPS or `localhost`, and
serving this directory is the cheapest way to get one.

## The words this page uses

| word | what it means here |
|---|---|
| **material** | one encoded asset: a `.json` descriptor and the two or three `.dds` files that belong to it |
| **descriptor** | `PREFIX_nntc.json` - the sizes, the dequantisation of both levels, and the decoder's weights |
| **level 0** | a full-resolution latent plane, 1-4 channels, normally BC4 / BC5 blocks, with its whole mip chain |
| **level 1** | a second latent plane at a quarter of the resolution per axis, with its own chain |
| **output texture** | what the material decodes to; one material can carry several, and the key `N` steps through them |

Decoding one pixel is two texture samples - one from each level, at the same coordinate - and then `W phi + b` over
the two sampled vectors and their products. That is the whole of the runtime: no library, no compute pass, no
unpacking step. `docs/FORMAT.md` is the specification.

## What is on screen

The left column is the panel and the right is the picture. The panel's groups, top to bottom:

* **Load** - the material pulldown with **Reload** beside it, the working-directory buttons, and "Your files" for
  opening or dropping your own. The rows underneath are the checklist: a tick means the page has that file,
  `needed` means it is still missing.
* **Loaded files** - what was loaded. The two latents side by side (names and bytes, the format as stored and what
  was bound on this GPU, size, channels, bits, mips, dequantisation), the four bitrates, and the decoder's shape
  with its weight count.
* **Show** - which output texture, which channels, the geometry, the row of objects, orbit, renormalise, and
  **Reset view**.
* **Sample** - the filter, anisotropy, mips, and level 1's LOD shift.
* **Camera** - X, Y, Z, yaw and pitch, as numbers you can type into.
* **Decode** - the adapter, whether `texture-compression-bc` was granted, which path drew level 0, the CPU-unpack
  checkbox and **Reload shader**.
* **Display** - the canvas colour space, and whether the strip is drawn.
* **Frame** - Save frame and Copy link.
* **Messages** - "Stop on a WebGPU error", and the log.

Every key the native viewers bind has a control here, and the two move together. That makes the panel the status
readout as well: a radio set to Trilinear *is* the filter readout, so there is no second copy of it to fall out of
step. The strip over the picture carries only the two things that change on their own - the frame rate and the
camera.

**Channels.** The channel pulldown in Show draws RGB, or R, G, B or A on its own, and it applies to all three views
(the decode, level 0 as stored, level 1 as stored). A single channel is drawn as grey, the same value in all three
outputs, so what you read is a magnitude and not a hue. A four-channel latent's fourth channel cannot be seen any
other way. In the decode it is applied after the tangent-space renormalisation, where alpha is 1.0 by construction,
so choosing A draws white.

**The bitrates** are the encoder's own arithmetic, mirrored from `src/main.cpp` rather than worked out a second
time. A block-compressed level costs whole 4x4 blocks, and computing it as `w * h * bytes_per_texel` reports
18.33 bpp where the files take 18.66. The six numbers in the panel are the six the encoder's `memory` lines print
for the same asset.

## Opening your own material

There are three ways in, and the first two work in every browser that runs the page:

* **"Open files..."** in the Load group. Select the `.json` and its `.dds` files together - open the folder,
  Ctrl+A, Open.
* **Drag them onto the page**, anywhere on it, together or a few at a time.
* **Give the page a folder**, which is the next section.

The page keeps every file you hand it, so a set can be assembled over several goes and from several folders: pick
the descriptor, then its `.dds` files, then a replacement for one of them. The material opens the moment the set is
complete, and until then the top of the panel names what is still wanted. Files that do not belong to the
descriptor are listed and ignored. The set is matched on file name and nothing else, and it is emptied whenever a
material opens, so a half-finished attempt cannot leak into the next one.

**A bare `PREFIX_nntc.json` cannot load on its own** unless the page has a folder. No browser hands a page a file's
siblings from a file handle - the File System Access specification defines no parent accessor - so the page names
the files it is short of rather than failing quietly.

## The working directory

One folder, granted once. This is what makes the encode-look-encode loop worth having.

* **Chrome and Edge** draw **"Select working directory..."** in the Load group.
* **Firefox and Safari** have no folder picker, so that button is not drawn at all and one quiet line stands in its
  place. **Drop the folder on the page** instead. It arrives through `webkitGetAsEntry`, which they have had for
  years, and becomes the same working directory. Dropping a folder works in Chrome and Edge too.

Which of the two you get is decided by `'showDirectoryPicker' in window` and never by the browser's name: a control
that cannot work is worse than an absent one.

Once the page has a folder, every material in it joins the **same pulldown** as the bundled ones, under its own
group, with the same line each bundled material has - `m1 - 512x512, 1 texture, level 0 BC5, 432 KB`. The bundled
materials stay where they are, because they are the reference you compare your own encode against. What gets listed
is every `.json` whose `format` is `nntc-dds-1` or `ntc-dds-1`; anything else - a `*_source_material.json`, a
`package.json` - is read once in the background and never shown. Names and byte counts appear first and each line
fills in as its descriptor is read, so a folder of hundreds of materials is not a wait.

A folder buys four things, and none of them works without one:

* **click a material and it opens.** Each `.dds` is resolved out of the folder by the name the descriptor uses,
  with no prompt per file.
* **a bare `.json` opens too**, picked or dropped. The page asks the folder where that file sits and takes its
  `.dds` files from the same directory, a subfolder as readily as the root.
* **Reload survives a re-encode.** `R` re-reads what is on disk now and leaves the camera and every mode where they
  were. It also re-lists the folder, so a material the encoder has just written joins the pulldown even with a
  bundled one on screen, and one that has gone leaves it with a sentence saying so.
* **a picked folder is remembered.** The handle goes into IndexedDB, and on the next visit `queryPermission()`
  usually answers `granted` and the folder is listed with no click at all. Where it answers `prompt` instead, the
  page offers **Reopen**: `requestPermission()` needs a user activation, so it cannot run on load and has to be a
  button. **Forget** drops the page's record of the folder - the permission itself is taken back in the browser's
  own site settings, not here.

A folder restored on load is listed and nothing else changes; the bundled material that has already opened stays on
screen. Picking, reopening or dropping a folder does open that folder's first material, because each of those is
someone saying "this folder, now" - and opening a material resets the view.

A **dropped** folder differs in two ways, and the status line says both. It is walked afresh on every listing,
which is what keeps Reload meaning the same thing here: a rewritten file comes back as a fresh `File` rather than
the stale snapshot the File API would refuse to read. And it cannot be stored, because a directory entry is not
structured-cloneable, so it does not survive a page reload and there is no Reopen for it. Only that one folder's
own files are read, with no recursion into subfolders, and at most 2,000 materials; past that the panel says the
folder holds more than it listed.

The page **only ever reads**. The picker is asked for `mode: "read"`, and nothing here writes a byte of your tree.

## Keys

Click the view first: the keys go to the canvas, so typing in the panel never moves the model. Every key below is
also a control in the left panel, bound both ways. **Help** at the top of the panel, or `F1`, shows the same table
in a dialog.

| key | what it does |
|---|---|
| arrows | move left / right / up / down |
| `W` / `S` | zoom in / out |
| `A` / `D` | yaw |
| `Q` / `E` | pitch |
| `Shift` | with any of the above: one third the speed |
| `Space` | reset the camera and every mode below, back to a fresh load |
| `C` | quad, cube, sphere |
| `G` | one object per output texture, in a row |
| `O` | turn the view by itself, 30 degrees a second |
| `N` | next output texture; with the row on, it slides which texture sits on which object |
| `1` / `2` | show level 0's texture as stored, or level 1's (each a toggle; one turns the other off) |
| `V` | renormalise the shown triple as a tangent-space normal |
| `P` / `B` / `T` | point / bilinear / trilinear filter |
| `X` | anisotropic filtering on / off (trilinear filtering only) |
| `M` | mips on / off (off holds the sampler at mip 0; nothing is reloaded) |
| `L` | level 1's LOD shift on / off |
| `R` | reload the asset, keeping the camera and every mode; `Shift+R` reloads the shader |
| `F1` | the help dialog |
| `Esc` | close the help dialog |
| drag, wheel | yaw and pitch; zoom |

That is the native viewers' table (`../viewer/README.md`) with three differences. `R` reloads the asset here and
the shader under `Shift+R`, which is the other way round from the native viewers, because the asset is the thing
that changes ten times an hour; `Ctrl+R` stays the browser's. `Esc` closes the help dialog rather than quitting,
because a page cannot close itself. And the native `4` and `5`-`8` are not bound at all: `4` draws level 0 from a
load-time BC4 / BC5 pack, which is not implemented here, and `5`-`8` are spare shader debug constants that nothing
in the shipped shader reads. The Direct3D 12 viewer's `K` has no counterpart either, since there is one WGSL decode
path.

**Reset and Reload are opposites, and the difference is the point.** `Space`, the **Reset view** button at the foot
of the Show group and the one in the help dialog all put the camera, the geometry, the filter, anisotropy, mips,
the level-1 shift, the as-stored views, renormalise, the channels and the shown texture back to a fresh load. They
leave the CPU unpack and the canvas colour space alone, because both are about the device rather than the view.
**Reload** (`R`) re-reads the same files and keeps every one of those. Opening a *different* material resets,
because a new material is a new thing to look at.

## URL fields

They mirror the native viewers' flag names, so a reader of those recognises them.

| field | what it does |
|---|---|
| `?asset=assets/PREFIX_nntc.json` | open a bundled material directly |
| `?z=` `?yaw=` `?pitch=` `?x=` `?y=` | the camera, in the native units |
| `?tex=N` | the output texture shown |
| `?cube` | the cube rather than the quad, as the native `--cube` |
| `?geom=` | `quad`, `cube` or `sphere`; this is what Copy link writes, since `?cube` cannot say sphere |
| `?row=1` | one object per output texture, side by side |
| `?orbit=1` | start with the view turning, 30 degrees a second |
| `?filter=` | `point`, `bilinear` or `trilinear` |
| `?chan=` | `rgb`, `r`, `g`, `b` or `a`; one channel is drawn as grey |
| `?nomips` `?nobias` `?noaniso` `?renorm` `?raw0` `?raw1` | the native flags of the same names |
| `?overlay=0` | draw the scene without the strip, as `--nooverlay` does |
| `?unpack=0` / `?unpack=1` | hardware BC, or the CPU unpack (below) |
| `?strict=1` | a WebGPU refusal stops the page instead of writing a line (below) |

An unknown value for `?geom=`, `?filter=` or `?chan=` is ignored rather than refused: a URL is something people
edit by hand, and the cost of a typo should be the default and not a page that will not open.

## Saving a frame

The **Frame** group writes what is on screen to a file. `Save frame (.bmp)` produces the same 24-bit bottom-up BGR
`.bmp` that `nntc_view --shot` writes, so `tools/frame_diff.py` can read one from this page and one from a native
viewer without being told which is which. `Save frame (.png)` is the same picture for looking at. **Size** takes
`WIDTHxHEIGHT`, and is empty for the canvas as it stands.

A capture is rendered away from the canvas, at the size you asked for, and **without the strip** - which is what
`--nooverlay` does natively and for the same reason: the strip names the file and its format, and that is exactly
what two otherwise identical renders differ in.

`Copy link` puts the page's address on the clipboard with the camera and every mode in it. Not the orbit: a link is
a view, and a view that starts turning the moment it opens is not the one that was copied. A bundled material goes
in by name; a material out of your own folder does not, because no link can hand a page a folder. The camera is
rounded to three decimals, so a link reopens on the same view rather than on the same bytes.

### Driving it from a script

`?shot=1` renders one frame and hands it over with nothing to click:

```
http://localhost:8082/?asset=assets/m1_nntc.json&shot=1&size=1280x720&noaniso=1&post=/frame
```

It loads a bundled material - nothing else is reachable from a URL - renders once, and either downloads the `.bmp`
or, with `&post=PATH`, POSTs it to the server that served the page. `document.title` becomes `done`, or
`shot failed: ...` with the reason, so a driver watching the title learns the outcome even if the POST never
arrives. An `?asset=` naming a material that is not bundled is refused rather than shot, because the page would
otherwise open the first bundled one and post a frame of the wrong material under a title of `done`. The strip is
off under `?shot=1` unless `&overlay=1` asks for it.

The page cannot write a file and cannot close the browser, so the two halves of a native `--shot` are split: the
page sends the bytes, and whatever launched the browser writes them and kills it.

### The gate arm

```
python tests/run_checks.py --webgpu-frames
```

That does the above for every bundled material: a headless browser renders one frame, `nntc_view_d3d12 --shot`
renders the same one, and `frame_diff` compares them. The Direct3D 12 viewer is the comparison because Dawn sits on
Direct3D 12 on Windows, so the two are one API two layers apart rather than two different ones. The page's frames
have measured very close to that viewer's; run the arm for the numbers on your own machine.

It is **opt-in** and prints a skip line otherwise, because it drives a browser rather than reading a file. It also
skips, by name, when the machine is not Windows, has no Chrome or Edge, or has no `nntc_view_d3d12` built.

## When something goes wrong

**WebGPU's validation is always on.** There is no layer to enable and no flag that turns it off. What a page can
get wrong is not *listening*: a `createTexture` or a `writeTexture` that is refused throws nothing, it marks the
object invalid and writes one line to a console nobody is reading. That is how a mip which never uploaded drew a
flat wrong colour here for three rounds of frame comparisons.

So the page listens in three places - an `uncapturederror` handler on the device, a `pushErrorScope('validation')`
around each file's upload, and the `device.lost` promise - and everything it hears reaches the same two places you
can see.

The **status line** at the top of the panel says what just happened, in red when it failed and in ordinary text
when it did not. The **message log** at the foot of the panel keeps the sequence behind it, numbered and
timestamped, as selectable text with Copy and Clear. Every line names its source: `[WebGPU]` for anything the
device or driver refused, `[page]` for the viewer's own. The log is always on the page and open, empty or not,
because a log you have to go looking for is one you learn about after you needed it.

**`?strict=1`, and the "Stop on a WebGPU error" checkbox beside the log**, change what a refusal *does*: the
message goes to a banner over the view and drawing halts on the last good frame, rather than a line that a moving
picture carries past your attention. It does not enable validation - nothing can, it is already on. The checkbox
and the URL field are the same switch, it takes effect without reloading the page, and turning it off releases a
page it has already stopped.

One level of checking this page cannot reach is Dawn's own deeper backend validation, which is a browser launch
flag and not something a page can ask for. Dawn documents it as `--enable-dawn-backend-validation`
([`webgpu-cts/README.md`](https://dawn.googlesource.com/dawn/+/HEAD/webgpu-cts/README.md), read 2026-09-20). It has
not been exercised in this tree, so it is the next place to look for a driver-level fault rather than a step anyone
here has run.

## How the decode works

`nntc_view.wgsl` is a transliteration of `viewer/bin/nntc_view.hlsl`, which the Direct3D 11 and Direct3D 12 viewers
share: two mipmapped samples (three when level 0 is in two files), dequantise **after** sampling, build phi from
the two sampled vectors and their products, and apply `W phi + b`. The uniform block is the Direct3D viewers' two
constant buffers laid end to end, 2032 bytes, the same block the Vulkan viewer packs; `renderer.js` asserts its
offsets before the first draw.

**Level 1 is sampled with its gradients scaled by 4**, not with a sampler LOD bias. That is the encoder's 1:1 mip
rule - output mip `m` reads mip `m` of both latents - and in a browser it is the only form the rule can take,
because WebGPU's sampler has no LOD bias field at all. Getting this wrong gives a picture that looks plausible at
1:1 and is wrong from mip 1 down.

**Testing a mip change.** Use `examples/npot360x200`, not one of the bundled materials. It is 360 x 200, so its
chain is 360x200, 180x100, 90x50, 45x25, 22x12 - three of those five levels are not multiples of the 4 x 4 block,
which is where a wrong copy extent, row pitch or block count shows. The bundled 512 and 1024 materials are
block-aligned at every level and will draw correctly with such a bug present. Open it with "Open files...":
`npot360x200_nntc.json` and its two `.dds` files together. It is not bundled because the bundled set is already
4.2 MB and is what every visitor downloads.

### The BC unpack, and `?unpack`

Where a GPU's WebGPU has no `texture-compression-bc` - phones, tablets, and any adapter that says so - the page
decodes BC4 and BC5 on load in plain JavaScript and uploads `r8unorm` / `rg8unorm` planes instead. **The shader
does not change**: a `bc5-rg-unorm` texture and an `rg8unorm` texture are the same `texture_2d<f32>` under the same
sampler, so only the `format` field of `createTexture` and the bytes handed to `writeTexture` differ.

`?unpack=1` forces that path on any device, including one with hardware BC, which is what makes it testable here.
`?unpack=0` asks for hardware BC and is refused by name on a device that has none, rather than silently falling
back - the shape of the native viewers' `--coopvec 0|1` and `--linalg 0|1`. The Decode group carries the same
switch as a checkbox, which changes the path without losing the camera.

A block-compressed **base** level must be a multiple of 4 on both axes: the tree's own rule, Direct3D's, and
WebGPU's, which enforces it at `createTexture`. The lower mip levels need not be, and usually are not. A `.dds`
whose base breaks the rule is refused by name before any device call, with the unpack named as the way to view it
anyway. `nntc_encode` pads the base, so only a hand-made or third-party file can hit this.

The unpack reproduces the format's specified values, rounded to the nearest byte. It is not byte-identical to any
particular GPU's decode and cannot be: a GPU may evaluate the palette interior with six-bit weights. The two paths
agree to within a count or two on the GPUs they have been run on.

## Where it runs

WebGPU is a secure context everywhere, and never `file://`.

| browser | WebGPU | `texture-compression-bc` | this page |
|---|---|---|---|
| Chrome / Edge, Windows, macOS, ChromeOS | 113+ | present | the hardware BC path |
| Chrome, Android | 121+ | absent | the CPU unpack |
| Firefox, Windows | 141+ | present | the hardware BC path |
| Firefox, macOS on Apple silicon | 145+ | present | the hardware BC path |
| Safari, macOS 26+, iOS 26+ | 26+ | present on macOS, absent on iOS | either, by feature detection |
| Firefox on Linux, Safari before 26, Chrome on most Linux | not shipped | - | the page says which case it hit |

The version numbers are `docs/WEBGPU_VIEWER_PLAN.md` section 3.1's, which cites them one by one against MDN's
browser-compat-data, the gpuweb wiki, Mozilla's shipping post and caniuse; none of them was measured here. Chrome
and Firefox on Windows have both been run. A browser with no WebGPU gets one sentence saying so rather than a blank
canvas.

The folder is a second axis and a separate feature test, because the folder and the button that asks for one are
different questions. The button is drawn where `showDirectoryPicker` exists, which is Chrome and Edge - and those
Chromium browsers that have not turned it off, which Brave has by default, so Brave gets the quiet line. Firefox
and Safari have both declined the API, so they get it too. A dropped folder is taken in all of them. Every other
way in is the same everywhere.

## Not here yet

`.zip` reading; a "Choose a folder..." dialog (`<input webkitdirectory>`) beside the drop, which is the one form of
folder intake still missing where there is no picker; and the 16-bit unpack target option.

## The files here

| file | what it is |
|---|---|
| `index.html`, `style.css` | the page and the two-column layout |
| `main.js` | the state object, the frame loop, the keys and the mouse |
| `renderer.js` | the adapter, the textures, the eight samplers, the uniform block, the camera |
| `nntc_view.wgsl` | the scene shader, the transliteration of `viewer/bin/nntc_view.hlsl` |
| `overlay.js`, `overlay.wgsl`, `font8x8.js` | the drawn strip and the tree's own 8x8 font, copied byte for byte |
| `dds.js` | the DX10 `.dds` reader, the twin of `shared/dds.h` |
| `bcdec.js` | BC4 and BC5 in plain JavaScript |
| `descriptor.js` | the descriptor, every refusal `viewer/main.cpp` makes, and the mirrored bitrates |
| `intake.js`, `panel.js` | the matcher behind every way in, and the left column |
| `workdir.js` | the working directory: the grant, the handle in IndexedDB, the folder listed |
| `capture.js` | a rendered frame as the native viewers' 24-bit bottom-up `.bmp`, or as a `.png` |
| `webserver.py` | the static server, rooted at its own directory |
| `assets/` | three encoded materials and the hand-written `assets.json` that lists them |

`assets/assets.json` is hand-written on purpose: a generator would be a feature bolted onto a server that is copied
because it is trusted, and a static host has no directory listing. The failure a hand-written list can have - a
material added and the line forgotten - is what `tests/run_checks.py`'s `webgpu_assets_check` catches, in both
directions, along with a listed descriptor whose `.dds` files are not beside it.

## Hosting

Copy this directory to any HTTPS host. There is nothing to rewrite, no header to configure and no server-side code;
the cost is about 4.2 MB of assets served on demand.

## Licence

The same as the rest of the tree (`LICENSE` at its root). The 8x8 font is basisu's `g_debug_font8x8_basic`; the
page and the shaders are this tree's.
