# viewer: the asset on a real GPU sampler

A Windows / Direct3D 11 viewer for the asset `nntc_encode` writes (`../docs/FORMAT.md`): the two latent textures
with their full mip chains (`PREFIX_lat0.dds` -- or `PREFIX_lat0a.dds` and `PREFIX_lat0b.dds` -- and `PREFIX_lat1.dds`)
and `PREFIX_nntc.json` (the dequantisation and the decoder). The GPU takes one `Sample()` per texture with the
hardware sampler -- two, or three when level 0 is in two files (point, bilinear, trilinear or anisotropic, one LOD for
both) -- and the pixel shader (`bin/nntc_view.hlsl`) dequantises the samples, builds
`phi(z)` and applies the affine decoder. That is the whole runtime.

The window, device, swap chain, camera, quad and cube, samplers, constant buffers, debug overlay and frame loop are
the author's `shader_deblocking_d3d11` sample's (Apache 2.0, the text is `LICENSE` at the root), and `main.cpp` keeps
that sample's dense one-statement-per-line formatting rather than the encoder's. New here: the DX10 `.dds` loader (uncompressed 8-bit and BC4 / BC5, the levels base-first and
tightly packed, its parsing in `../shared/dds.h` and its Direct3D tail here), the JSON reading (sheredom's
single-header `json.h`, public domain, wrapped by `nntc_json.h`; both live in `../shared/` with everything else more
than one program in this tree reads), the decoder constant buffer and the shader.

## Building

`nntc_view` and `bc_check` are built by the repository's root CMakeLists on Windows. Neither needs CUDA: on a
machine without the toolkit the root CMakeLists skips the encoder and builds them alone, so the plain command is

```
cmake -B build -S . -G "Visual Studio 18 2026"
cmake --build build --config Release
```

and a machine that also builds the encoder adds the CUDA toolset (`-T cuda=13.4`, or `-G "Visual Studio 17 2022"
-T cuda=13.1`), as the root README's Building section describes.

`nntc_view.hlsl` is copied next to the executable by the build; the viewer compiles it at runtime from the working
directory or from beside the executable, so it can be edited and reloaded with `R`.

## Running

`nntc_view` returns 0 on success and 1 on any error: an unreadable descriptor, an unknown flag, a malformed value,
a Direct3D object that could not be created, or a `--shot` frame that could not be written.

```
build\Release\nntc_view.exe <PREFIX_nntc.json>
```

The `.dds` files are found through the JSON's `file` / `files` entries, beside the descriptor itself. The viewer
takes the descriptor's path exactly as given and derives no name of its own, so an asset written under any descriptor
name -- `PREFIX_nntc.json`, the `-o name.json` spelling, or the `PREFIX.json` of an older asset -- opens.

## Controls

Arrows move, `W`/`S` zoom, `A`/`D` yaw, `Q`/`E` pitch, `Shift` slow, `C` cube/quad, `P`/`B`/`T` point/bilinear/trilinear,
`V` renormalise the shown texture as a tangent-space normal (unpack, unit length, repack; grey-ish and black-ish / z < 0 pixels are left alone; off by default),
`X` anisotropic filtering on/off (on by default, `MaxAnisotropy` 8; it applies in TRILINEAR mode only, and in the other two the
overlay says the flag is remembered and idle),
`M` mips on/off (a second sampler set with MaxLOD = 0, so the hardware reads mip 0 of both textures; nothing is reloaded),
`L` level 1's LOD shift on/off (on by default, see below),
`N` next output texture of the material, cycling through all of them - up to SIX, which is what the shader's `W[108]`,
`bias[5]` and `outv[18]` carry and what `viewer/main.cpp` checks the decoder's shape against (`nout <= 18`),
`1` show level 0's texture as stored, `2` show level 1's, `4` level 0 from the BC4 / BC5 pack made
at load - which exists only when the file's level 0 is UNCOMPRESSED, and does nothing for the block-compressed default (below), `R` reload the shader, `Space` reset, `Esc` quit. Keys `5`-`8` set the shader's spare debug constants and do
nothing in the shader as it ships.

`--shot FILE.bmp` renders one frame and exits. Its state comes from `--cube`, `--z`, `--yaw`, `--pitch`, `--tex`, `--raw0`,
`--raw1`, `--bc`, `--nomips`, `--nobias`, `--noaniso` and `--renorm` (the key `V` state), and `--nooverlay` leaves the debug strip
off the frame so that two shots can be compared byte for byte. **`--shot` takes no keyboard input at all** - neither the held keys
nor the struck ones - so its bytes are a function of the command line and the asset and of nothing else. That is not cosmetic: the
frame used to read the live keyboard while its own freshly created window was the foreground one, so a keystroke anywhere on the
machine during the few milliseconds the process lives moved the camera or changed the filter for that one frame. It showed up as
one shot in ten differing from the others on identical binaries and identical assets; `tests/run_checks.py` now launches the same
shot five times and asserts the five are the same bytes before it compares any two assets.

Anisotropy costs the representation nothing: it is several trilinear samples along the footprint's long axis, the dequantisation
is affine and the decoder is affine in the samples, so a blend of latent samples decodes to the blend of the decodes - the same
argument that makes trilinear filtering free (`../docs/DESIGN.md`, section 5). The sampler set carries an anisotropic variant,
so the mips toggle keeps working under it (level 1's LOD shift is in the shader's gradients and needs no sampler).

The window is sized from `GetClientRect` after it is created, so the swap chain is the size the window actually got - Windows
clamps a 2560x1440 request to the work area - and a later `WM_SIZE` carrying that same size changes nothing. The first frame is
therefore the steady one and, with `--shot`'s keyboard reads gone as well (above), `--shot` writes the same bytes on every launch
and the release gate compares two assets' frames with no warm-up run.

Sampling: one standard mipmapped call per texture -- two, or three when level 0 is in two files. Level 1's is a `SampleGrad` whose two UV
derivatives are both multiplied by `2^lod_bias_level1` (4 at block 4), which raises the LOD the hardware computes by exactly `log2(block)`
before any clamp: level 1 has a quarter of the texels per axis, so the GPU's own LOD for it sits two mips finer than level 0's, while the
encoder fitted output mip m from mip m of BOTH latents (the 1:1 rule). Scaling both gradients lines the two up, and it is still one ordinary hardware sample (under anisotropic filtering it is a little blurrier on oblique surfaces than a sampler bias would be, because the longer gradients also lengthen the line the anisotropic taps are spread along: the root README's "The two ways to apply the level-1 mip shift"); at 1:1 on screen level 1 is still at its
mip 0. With `L` off (the gradients unscaled) the base is right but from mip 1 down the colour latent is the wrong plane and the result goes
splotchy. **No sampler here carries a `MipLODBias`.** Until v1.1.21 one did: level 1 was sampled through a sampler with a LOD bias of +2, and that was switched to scaled gradients because of mipmap LOD calculation differences on Intel parts against AMD and NVIDIA ones (the root README has the note). A sampler LOD bias of the same value is equivalent on hardware that keeps a negative base LOD under magnification (NVIDIA, AMD), but was observed to blur a magnified picture badly on Intel Xe integrated graphics, so the gradient form is the one to ship -- a 13th-gen Core i7 Intel Xe part
rendered the magnified quad VERY blurry with the biased sampler and sharp without it, in this viewer and in the Vulkan one alike, which is
what the gradient form fixes.

The overlay is four lines and the debug strip `--nooverlay` leaves off:

1. **the asset**: the source size, then each level's `WxHxchannels`, its stored format, its index bit depth and its mip count, then
   the block factor, then the GPU memory of what is bound. Level 0's format reads `BC5 (file)` or `BC5+BC4 (file)` when the file
   itself is block-compressed and nothing was packed here, `U8` when it is uncompressed and the load-time pack is not bound, and
   `BC5 lossless` or `BC5 41.2dB` when it is (key `4`). The memory is `GPU:10.00bpp/tex base (L0 8.00), 13.33 with mips`: the
   actual bytes of the bound textures - whole 4x4 blocks for a block-compressed one, `stored_channels * w * h` for an uncompressed
   one - over the **decode extent** (level 0's own base size, which is the padded size when the input was padded) times the
   material's texture count, so it reads as bits per source pixel per material texture. `(L0 x.xx)` is level 0's own share of the
   base figure, which is the number the file layout decides: 4 bpp for one BC4, 8 for one BC5, 12 for a BC5 and a BC4, 16 for two
   BC5s, divided by the texture count. It follows key `4` and is independent of the file's own byte size;
2. **the state**: quad or cube, the filter, the anisotropy (and whether it is idle, which it is outside trilinear), mips on or off,
   level 1's LOD shift on or off, which texture of the material is shown, whether the decode or a raw latent is shown, and the
   renormalise flag;
3. **the camera**: x, y, z, yaw, pitch;
4. **the keys**, as a reminder line.

## Level 0 as BC4 / BC5: what the file holds, and the load-time pack (key `4`)

**Under the encoder's default the asset already holds OPTIMISED BC4 / BC5 textures, and this program does nothing to
them.** `nntc_encode --l0 bc8` solves level 0 as a continuous 8-bit plane, encodes it as BC4 / BC5, and then goes on
optimising it *as BC*: every 4x4 block's endpoints and selectors are re-chosen to minimise the decoder's own output
error, a candidate taken only when that error falls, and the whole model is refitted and the plane repacked in outer
passes taken only when the shipped objective falls (`../docs/DESIGN.md` section 3.6, worth +0.11 to +2.97 dB over
packing once). So the blocks in the `.dds` are the representation that was measured, not a compressed copy of
something else. The viewer binds them, samples them with the hardware sampler and decodes in the pixel shader; a level 0
stored in two files binds its second file to `t2`, the overlay's format reads `(file)`, and **key `4` does nothing**.

The **load-time pack** below exists for the other case only: a level 0 that arrives **uncompressed** in the file, which
is `--l0 palette --bc0 0` or the `_u` twin `--bc0 both` writes. Then the viewer packs its whole mip chain with
`../shared/bc_pack.h`, the same encoder `nntc_encode` seeds its own pack with, and key `4` switches between the two
(`--bc` starts on the pack). That pack is the SEED pack - the stb_dxt-style choice the encoder starts its refinement
from, not the choice it ships - so it is a comparison and a validation of `bc_pack.h`, and not what a default asset
contains. The layout is the same either way:

| level-0 channels | textures | shader |
|---|---|---|
| 1 | one BC4 | `t0.r` |
| 2 | one BC5 | `t0.rg` |
| 3 | a BC5 and a BC4 (the BC4 holds channel 2) | `t0.rg`, `t2.r` |
| 4 | two BC5s | `t0.rg`, `t2.rg` |

The load-time pack is a plain rewrite of the stored index (again: this is the pack of an UNCOMPRESSED level 0, and it is the
seed the encoder's own refinement starts from, not the pack a default asset ships):

- **1-3 bits per channel: lossless.** Every block gets 0 and 255 as its endpoints (through the mode that provides them: 3 bits the
  eight-value mode with r0 = 255, r1 = 0, whose palette is exactly k/7; 2 bits r0 = 85, r1 = 170 in the six-value mode, whose palette
  holds 0, 85, 170, 255; 1 bit the six-value mode's fixed 0 and 255) and the index goes into the selectors. The loader decodes every
  packed block on the CPU with the format's palette and reports the largest error against the exact index value (float rounding only).
- **4 bits and more: lossy, typically.** The block's lowest and highest index values are the endpoints and each texel takes the nearest
  of the eight palette values, the way stb_dxt and most BC4 encoders do it. The loader prints the packing PSNR over the chain against
  the exact index values.

The shader is unchanged apart from where level 0's channels past the first two come from (`t2` when two textures are bound); the packed
textures use the same sampler and the same one `Sample()` call per texture. It reads `t2.rg` either way and uses only `.r` at three channels, which is
what a BC4 in that slot returns.

Measured on the GPU (`--shot`, packed against uncompressed, the frame below the overlay), one NVIDIA GPU:

| asset | level 0 | pack | CPU round trip | GPU: raw level-0 view | GPU: decoded view |
|---|---|---|---|---|---|
| a single 1152x1408 image | 2 x 3 bits | BC5 | lossless (largest error 6.5e-6 / 255) | max diff 4 / 255, mean 0.147 | max diff 5, mean 0.118 |
| a four-texture material | 2 x 4 bits | BC5 | 35.31 dB, largest error 17 / 255 | | max diff 19, mean 0.088 |

The 3-bit raw view is not bit-identical on the GPU even though the pack is exact in the index: in flat areas the packed texture samples
36, 71, 112, 143, 184, 219 for k = 1..6 where the uncompressed byte holds 36, 73, 109, 146, 182, 219. Those are the eight-value palette
evaluated with 6-bit weights (9, 18, 28, 36, 46, 55 over 64) rather than k/7: the GPU's BC4 decoder, within the API's tolerance, not the
encoder. The decoded texture differs by at most 5 / 255 from the uncompressed path on this asset.

### Validation against bcdec

`bc_check PREFIX_nntc.json [UNCOMPRESSED_PREFIX_nntc.json]` (built next to the viewer) validates `bc_pack.h` against iOrange's
[bcdec](https://github.com/iOrange/bcdec) (`bcdec.h`, MIT / Unlicense, built with `BCDEC_BC4BC5_PRECISE`). What it does depends on
what the asset's level 0 actually is, and both jobs are real:

* **an UNCOMPRESSED level 0**: every level is packed here exactly as the viewer packs it at load, and every block is then decoded
  three ways - bcdec's precise float decoder (the D3D spec's palette), bcdec's 8-bit integer decoder and `bc_pack.h`'s own
  `bc4_decode_block` - each against the exact index value k / (2^bits - 1). That is the table below;
* **a BLOCK-COMPRESSED level 0, which is what the encoder writes by default**: nothing is packed here. The file's own blocks are
  decoded by bcdec and by `bc_pack.h`, the two are compared, and at 1-3 bits the decoded byte must be the index's own byte again,
  PER CHANNEL - the statement that the quantisation index survived the block format. At the 8 bits of `--l0 bc8` there is no index
  to recover and no lossless claim is made.

The **second, optional argument** is the uncompressed twin of the same solve, which `--bc0 both` writes as `PREFIX_u_nntc.json`. Given
it, the decoded bytes of the compressed asset are compared texel by texel against that plane too, which is the check that the two
decode paths agree. It is used only where this asset's level 0 is compressed, since that is the only place there is a plane for the
blocks to be compared against.

```
build\Release\bc_check.exe out\mymat_nntc.json
build\Release\bc_check.exe out\mymat_nntc.json out\mymat_u_nntc.json
```

| input | pack | bcdec float vs exact | bcdec 8-bit vs exact | viewer decoder vs exact | bcdec float vs viewer decoder |
|---|---|---|---|---|---|
| a 2 x 3-bit level 0, 3,939,350 channel-texels | BC5 | 6.5e-6 / 255 (lossless) | 0.43 / 255 (the palette as bytes, rounded to nearest) | 6.5e-6 / 255 | 1.3e-5 / 255 |
| a 2 x 4-bit level 0, 699,008 | BC5 | 17 / 255, 35.31 dB | 17 / 255 | 17 / 255 | 2.6e-5 / 255 |
| a 4 x 8-bit level 1 (reported for reference; the viewer does not pack level 1; exercises the two-BC5 path) | BC5 + BC5 | 14.7 / 255, 40.30 dB | 15 / 255 | 14.7 / 255 | 2.9e-5 / 255 |
