# Browser encoder using CUDA WebShader

This fork adds a working browser encoder at `webgpu/encode.html`. It compiles
the CUDA device kernels in `webgpu/encoder/kernels.cu` with CUDA WebShader and
executes them on WebGPU. The existing WebGPU viewer can open the exported files
or receive them directly from the encoder's **Open interactive viewer** button.
No CUDA installation or NVIDIA GPU is required by this path.

## Run

```sh
git clone --recurse-submodules --branch feat/cuda-webshader-encoder https://github.com/SamG-Coder/nntc.git
cd nntc
npm ci
npm start
```

Open http://127.0.0.1:5174. For an existing checkout, first run
`git submodule update --init --recursive`. Node 20+ is needed for the local
server and tests. The actual application is static JavaScript and needs only
HTTPS or localhost and a working WebGPU browser. A static deployment must serve
both `webgpu/` and `vendor/cuda-webshader/src/` with JavaScript MIME types; do not
serve only the `webgpu/` directory with the old viewer-only server.

Choose one to six same-sized images, or try the bundled material. The browser
reads RGB bytes, ignores alpha with a notice, and pads the right/bottom edges to
multiples of four with clamped source pixels. Choose the latent channel counts,
number of fitting passes, mipmaps, and BC or uncompressed storage. Download the
ZIP and extract the descriptor and DDS files together. The output descriptor is
the existing `nntc-dds-1` format, with an informational `encoder` field.

The app does not upload source images. Cancellation takes effect between GPU
dispatches and host stages. Images beyond the adapter's buffer/dispatch limits
or the encoder's estimated 1 GiB working-memory ceiling are refused before GPU
allocation. The UI also bounds source image memory before reading pixels.

## Implemented path

* One to six RGB maps; C0 and C1 each support one to four latent channels.
* Shared PCA initialization and an area-averaged source mip chain.
* The same ordered bilinear feature vector `[c; s; s*c]` and affine decoder.
* Center-only fitting or the center plus NNTC's four rotated-grid fractional
  positions. Both latents and the source use bilinear clamp sampling.
* Streaming GPU normal equations; the at-most-25-variable dense solve runs in
  JavaScript's double precision. The GPU never holds all observations at once.
* Four-color coordinate descent on each latent plane. Each texel gathers its
  affected filter sites and solves its channel vector's small quadratic.
* BC4/BC5 endpoint/selector fitting (split into two DDS files for C0=3 or 4), or
  R8/RG8/RGBA8 storage, plus quantized R8/RG8/RGBA8 for the quarter-size plane.
* A final decoder fit using the actual quantized/BC-decoded values that are
  exported. Ranges and decoder coefficients are shared by all stored mips.
* Existing viewer validation of every exported descriptor and DDS file, source
  comparison, base-level PSNR, ZIP download and interactive viewer handoff.

The GPU performs observation generation, normal-equation accumulation, latent
updates and reconstruction. JavaScript performs image intake, source mip/PCA
initialization, small dense decoder solves, quantization/BC packing and file I/O.
The `cuda-webshader` submodule is pinned to the commit that supplies the new
`NormalEquations` streaming operation.

## Differences from the native encoder

This is a **format-compatible browser implementation, not native encoder parity**.
The native C++/CUDA encoder remains available and unchanged. The browser path
does not implement its full command-line/material-JSON option set or reproduce
its exact initialization, global sparse latent solves, ridge acceptance ladder,
palette modes, per-map weighting, sRGB/normal-specific mip filters, or joint
decoder-objective BC block refinement. BC packing here minimizes latent-channel
error, then refits the decoder. Raising fitting passes does not add those native
algorithms.

GPU normal-equation products and sums use f32, with a fixed reduction order;
only the small host solve uses f64. Results are not promised to match native
CUDA bytes, PSNR or performance, or to be bit-identical across GPU vendors.
The four-color solver uses a bounded, ridge-regularized update, rather than the
native global sparse solve. Quality must be measured for the intended materials.

The source mip filter is a box/area filter on the supplied RGB bytes; it does not
renormalize normals or linearize sRGB. The reported PSNR compares unclamped
reconstruction with original RGB at base-level pixel centers, excluding padding.
It uses the BC specification's palette values; actual hardware BC interpolation
and the viewer's byte-unpack fallback can vary slightly. It is not a fractional
site or whole-mip-chain error measurement.

## Verification

```sh
npm test
npm run test:gpu
npm run test:ui
# Optional independent upstream decoder (requires numpy and Pillow):
python tests/check-web-exports.py
```

The Node tests compile every CUDA entry and check all 32 combinations of latent
layout and storage type through the existing viewer parser, partial BC blocks,
DDS mip sizes, malformed input rejection and ZIP CRCs. GPU tests exercise
one/six outputs, split BC planes, uncompressed RGB, odd sizes, mip chains,
constant input, repeatability, cancellation and buffer cleanup. An independent
JavaScript decoder reads the generated DDS bytes and descriptor, and checks its
reconstruction against the GPU. The UI test encodes the bundled single-map and
four-map materials, downloads ZIPs, loads the interactive viewer, and checks a
mobile viewport and browser errors.

The Python cross-check also validates ZIP CRCs and runs the original repository's
strict DDS reader and decoder on both downloaded assets. It reports saturated,
rounded RGB8 PSNR, which can differ from the browser's unclamped float PSNR.

On Windows the browser tests use installed Edge. Set `NNTC_BROWSER` to another
Playwright browser channel. On other platforms they use Playwright Chromium;
install it with `npx playwright install chromium`. `NNTC_SOFTWARE_GPU=1` requests
SwiftShader for the small numerical GPU suite. Reports, downloaded assets and
screenshots go to ignored `out/web-tests/`.

Measured on the development machine with Edge/WebGPU: the bundled 512×512
`m1.png`, default C0=2/C1=4, six fitting passes, fractional sites and mipmaps,
encoded in approximately 2.8 seconds at 35.2 dB, producing approximately
431.2 KiB including its JSON. This is a local smoke measurement, not a native
CUDA comparison or a claim for other hardware.

The four-map `m1.png` through `m4.png` material took approximately 3.5 seconds
with the same options, producing 436.0 KiB and per-map PSNR of 25.9, 33.9, 30.7
and 34.9 dB. The optional SwiftShader test could not obtain a WebGPU adapter in
the installed Edge browser; it is not counted as a passing software-adapter test.

## JavaScript API

```js
import {BrowserEncoder} from './webgpu/encoder/encoder.js';

const encoder = await BrowserEncoder.create();
try {
  const result = await encoder.encode([
    { name: 'albedo.png', width, height, data: rgbaUint8Array }
  ], { name: 'material', c0: 2, c1: 4, iterations: 6,
       mips: true, bc: true, sites: 5,
       signal: abortController.signal, onProgress: console.log });
  // result.files: Map<string, Uint8Array>, containing DDS and JSON files.
  // result.psnr: per-source-map PSNR; result.reconstructed: interleaved floats.
} finally {
  await encoder.dispose();
}
```

An existing CUDA WebShader runtime can be passed to `BrowserEncoder.create({runtime})`;
in that case the encoder does not own or dispose that runtime. One encoder accepts
one encode at a time; cancellation or failure releases all per-encode GPU buffers.
