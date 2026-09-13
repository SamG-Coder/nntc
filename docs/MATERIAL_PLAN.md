# MATERIAL_PLAN.md -- per-texture settings, the material JSON and the mip filters (BUILT)

Status 2026-09-12: owner-approved shape; reviewed once by a planning agent (Opus 5) and once by a review agent
(Fable 5.1) against the code, both sets of findings folded in below.

**BUILT**, in four stages, each green on `tests/run_checks.py` and tagged: `v0.10a-list-rule-six` (the per-texture
list rule, the settings rows, `--quiet` redefined, six textures), `v0.10b-mip-filters` (the per-texture source chain),
`v0.10c-material-json` (the material JSON, its precedence and `source.inputs`) and `v0.10d-material-docs` (the
documentation and the Linux build). The stage tags named in this file, `v0.10a-list-rule-six` through
`v0.10g.2-weight-clamp`, are tags of the predecessor tree this repository was copied from, not of this one, which
begins at `v1.0.0-nntc`; the stage names are kept because that is how the work is referred to elsewhere in these
documents. The description below is kept as the specification it was; `docs/DESIGN.md` 4.2,
`docs/FORMAT.md` section 4 and `README.md` are the documentation of what shipped.

Two departures from the plan as written, both of them about WHERE a gate landed rather than about what the code does:
gate cases 3 (its `srgb` arm), 7 and 9 of section 6 were listed under the filters but all three drive `srgb` or
`normal_map`, which only a material JSON can set, so they landed with the JSON in stage C. And gate 3's `box` arm is
not asserted to differ from `default`: `tests/tiny.png` is 64x64, so every level is an exact integer ratio and a box
is the same mean taken once or iterated -- the two cubics are what a mis-wired name shows up in. That last departure
has since been narrowed: `box` IS asserted to differ from `default` where the two can differ, which is the odd chains
of the review fixes below.

## Review fixes (v0.10e, v0.10f, v0.10g)

The four stages were then reviewed against the code and two more commits followed, both green on the gate.

**`v0.10e-review-fixes`, the code.** The one that mattered was data loss: with a material and no `-o`, or with `-o`
naming the material's own directory, the asset's `PREFIX.json` WAS the material, and writing it destroyed the file that
described the run; it is refused before anything is created. Beside it: a weight that overflows the float `cw` (`1e300`
made `E` a nan and shipped a garbage asset with exit 0) is refused at both ends of the float range; `--weights` names
the flag and the value instead of exiting 1 in silence; a material may name a non-ASCII file (its strings are UTF-8 and
the narrow CRT calls are ANSI, so a file with an accent in its name opened as a positional and not from a JSON, which
is as inconsistent as it sounds); `--quiet` means no
progress at all rather than about thirty lines of it; the loader skips a UTF-8 byte-order mark, reports a parse failure
with its line and column, and refuses a key repeated inside one entry; the reported seed direction is bounded by
`MAX_NOUT` rather than by the old cap of twelve, so a six-texture material no longer prints six zeros; a
`static_assert` holds the least-squares staging block inside the 48 KB a block may have; the claim that the chain's
clamp exists because `linear_to_srgb` of a negative is a nan was wrong and is replaced by the true reason, with
`--diag` printing every source level's range so the clamp is observable; paths in messages are normalised; and the
viewer's usage lists all fourteen of its flags with `--help` a flag of its own.

**`v0.10g-json-suffix`, the descriptor's name.** The refusal above stopped the data loss; it did not stop the
collision from being the everyday case. A material encoded in its own directory had to be refused at all only
because the descriptor was `PREFIX.json` and the prefix was the material's own stem. The descriptor is now
`PREFIX_nntc.json`, beside `PREFIX_lat0.dds` and `PREFIX_lat1.dds` (and `_lat0a` / `_lat0b`), so a stem derived
from an input can never name the input:

* no `-o`, or `-o DIR/`: the descriptor is `DIR/<stem>_nntc.json`; the `.dds` names are unchanged.
* `-o some/prefix` (the explicit-prefix rule of v0.10e): `some/prefix_nntc.json` beside `some/prefix_lat0.dds`.
* `-o some/name.json`: the descriptor IS that file, and the `.dds` files take `some/name` as their prefix, so
  `some/name_lat0.dds`. This is the way to choose the descriptor's name outright.
* `--bc0 both` writes its uncompressed twin as `PREFIX_u_nntc.json` beside `PREFIX_u_lat0.dds`, from the prefix
  and not from the chosen name: the twin is a second asset, not a second spelling of the first.

The refusal stays as the backstop for the one spelling that can still name the material -- `-o material.json`
itself -- and its message now names the descriptor spelling as a third way out. Every reader follows: the viewer
and `bc_check` take a path and derive nothing, `tools/dds_decode.py` opens `PREFIX_nntc.json` and falls back to
`PREFIX.json` with a printed note so that the assets already in `out/` still read, and the gate covers all four
`-o` spellings, the twin, the fallback and the surviving refusal.

**`v0.10f-gate-and-docs`, the gate and the documentation.** Three gate arms asserted nothing at all -- a `--shot` BMP
is a fixed ~10.9 MB, so `len(frame) < 1024` could not fail, and one arm grepped the ENCODER's stdout for a warning the
VIEWER writes to stderr. `shot()` returns stderr now, the six `--tex` frames are asserted pairwise different, `--tex 6`
must warn and fall back, and every frame comparison is against another frame. Added: the host-twin agreement at six
textures under both level-0 modes, which is the only proof the eighteen-output kernels are right, with determinism
there and a five-texture case; the sRGB chain's known answer (128 in the source encoding, 188 in linear light, exactly);
the edge mode; per-texture `rgb_weights` against the one global triple, byte for byte; every override warning by its
exact text and count; an escaped `type` through the asset's own JSON; eighteen refusals that had no gate; both `-o`
forms and the vanilla identity; and the odd chains 44 / 22 / 11 / 5 and 68 / 34 / 17 / 8, where the iterated box and a
direct resize agree at the exact ratios and part company at the odd halving. The documentation: `LICENSE`'s file list
followed the headers into `src/` -- `LICENSE` is now the Apache License 2.0 text alone, and the file list it used to
carry is the licence table at the end of `README.md` -- the cross-platform numbers of `DESIGN` 6 are re-measured and cited by log path, every
remaining "box-filtered source" says the run's own chain, and `RESULTS` says that byte-identical across operating
systems is a claim about the DEFAULT chain.

**`v1.0.16-auto-box`, the implied filter.** Stage 4's refusal of `srgb`, `edge` or `normal_map` beside the `default`
filter now applies only when `default` was ASKED for; a texture that sets one of the three and names no filter at all
is switched to `box` and its settings row says `filter box (implied by srgb)`.

## 0. Why

The 2x2 box source chain is a safe default and a poor mip generator. `src/stb_image_resize2.h` (v2.18, vendored) has
the filters and wrap/clamp edges. A material's textures need different treatment (an albedo is filtered in linear
light, a normal map is renormalised, a mask is neither), so the settings become per texture; the command line alone
is a burden at four textures, so a small JSON carries the advanced case. **The bare command line stays the vanilla
path**: four images, the built-in box chain, weights 1, RGB weights 1,1,1, no renormalisation, no JSON needed.

## 1. The two ways in

**Command line** (what exists today, per texture where it was not):

    nntc_encode a.png [b.png c.png d.png] [-o ...] [--weights W,W,..] [--mip-filter F,F,..] [--rgb-weights R,G,B] [--quiet] [..]

* `--weights` and `--mip-filter` take **exactly one entry per input texture** when given; any other count is an ERROR
  naming the flag, the count given and the texture count. Absent, every texture takes the default. Implementation:
  both option vectors default to EMPTY (empty = all default; non-empty must equal the texture count), so there is no
  `_given` flag to disagree with the vector. `--weights`' single-value broadcast is REMOVED; `--bits0` keeps its own
  (it is per level-0 channel, a count that is itself a flag) and `--help` says why.
* `--rgb-weights R,G,B` stays ONE triple applied to every texture on the command line; per-texture triples are a JSON
  matter. The existing checks (three values, none negative, not all zero) run on the merged per-texture values, so a
  JSON `"weight": -1` or `"rgb_weights": [0,0,0]` is refused by the same messages.
* `--mip-filter` names, first cut: `default` (the current iterated 2x2 box, byte-identical), `box`, `mitchell`,
  `catmullrom`. Not exposed: `point` (keeps one texel in 64 at 8:1 and hands the deep planes an aliased target),
  `triangle`, `cubicbspline`; the enum is there if wanted later.
* `--quiet` EXISTS today with the opposite sense (it suppresses the final report and keeps the round lines). It is
  redefined: suppress the banner, the settings rows and the per-round lines; never warnings, errors or the report.
  No gate passes it. `--help-advanced`'s line changes accordingly.

**Material JSON** (`nntc_encode material.json [-o ...] [flags]`): one array, texture order. Read during parsing,
before `validate_options`, and it populates the input list with the resolved paths. A JSON mixed with image
positionals, two JSONs, or an empty array is an ERROR. `file` relative to the JSON's directory (absolute taken as is;
forward slashes fine on Windows). **The asset's base name is the JSON's stem** (`-o dir/` writes
`dir/material_lat0.dds`), which is what `validate_options` derives from the first positional today.

```json
[
  { "file": "albedo.png", "type": "albedo", "filter": "mitchell", "srgb": true, "edge": "wrap", "weight": 2 },
  { "file": "normal.png", "type": "normal", "filter": "mitchell", "normal_map": true, "rgb_weights": [1, 1, 0.5] },
  { "file": "rough.png" }
]
```

| key | default | meaning |
|---|---|---|
| `file` | required | the image, relative to the JSON |
| `type` | `""` | a free label, echoed into the settings rows and the output JSON (escaped), never interpreted |
| `filter` | `"default"` | `default`, `box`, `mitchell`, `catmullrom` |
| `srgb` | `false` | derive the deep levels in linear light: bytes -> linear float -> resize -> sRGB float. PRESENT with `filter` `default` is an ERROR (the box path is untouched; a key that does nothing must not look like a setting) |
| `edge` | `"clamp"` | `clamp` or `wrap`, the FILTER's edge mode (`STBIR_EDGE_CLAMP` / `STBIR_EDGE_WRAP`). PRESENT with `default` is an ERROR. It affects the chain only: the fit's sampler stays clamped, as today |
| `normal_map` | `false` | tangent-space normal map: after the resize, renormalise by the viewer's rule (section 3). Implies linear; with `srgb` true it is an ERROR; PRESENT with `default` is an ERROR (same rule as the other two) |
| `weight` | `1` | the texture's weight in the objective |
| `rgb_weights` | `[1, 1, 1]` | the texture's own channel weights, normalised to mean 1 PER TEXTURE. One global triple normalised per texture gives exactly today's `cw`, so existing runs' numbers do not move |

Unknown keys are an ERROR. More than six entries, or images of differing size, are ERRORS as today. "Present" means
the key appears in the entry, whatever its value.

**Precedence.** The command line overrides the JSON per setting: `--weights` replaces every `weight`, `--mip-filter`
every `filter`, `--rgb-weights` every `rgb_weights`. For each texture whose entry HAD the key, one WARNING on stdout
naming the texture, the JSON value and the command-line value (equal values still warn: the JSON key was overridden).
A key the JSON omitted and the command line sets is not an override and does not warn. The `default`-with-`srgb`/
`edge`/`normal_map` errors are evaluated on the MERGED settings, and when the command line caused them the message
says so. With a JSON input the command-line lists must still give exactly one entry per texture in the JSON.

## 2. The settings rows

Today's banner (`nntc_encode: N textures, WxH source ...`) already prints the layout, the planes, the device, the
sites, rounds, tol, ridge, the mip weights and the init and BC flags. The per-texture rows go INSIDE that banner,
right after its first line, and only the missing globals are added there (the output prefix, `--png`, `--mip-min`,
the JSON path when there is one). One banner, not two. Each row: index, file (base name), type, weight, rgb
weights, filter, srgb, edge, normal_map, and **the source of each value**: `default`, `json` or `command line`.
`--quiet` suppresses the whole banner and the round lines; the padding and override WARNINGs print before it and
are never suppressed.

## 3. The mip filters

* The source chain is float end to end (today: `Image::v` is float, loaded as `byte / 255`; the only 8-bit roundings
  are the PNG writer's and the PSNR's). **One more touch is added: every stb-resized level is clamped to [0,1]**
  before the sRGB encode and before the renorm. Mitchell and Catmull-Rom have negative lobes, the objective reads the
  chain raw, and `linear_to_srgb` of a negative is NaN, which is also where stb's determinism ends.
* Under `default` for a texture, that texture's chain is the current iterated 2x2 box, byte-identical to today. When
  no texture has any non-default setting the whole existing path runs unchanged (one early return).
* Under any other filter, **every level `m >= 1` is resized directly from the padded base** to the level's own size
  (`w_m, h_m`, unchanged from today's floor halving with `--mip-min`), through `stbir_resize` with `STBIR_RGB`,
  `STBIR_TYPE_FLOAT`, the texture's edge mode and filter. The chain is one interleaved `3T` float buffer, so each
  texture is split out, resized and re-interleaved per level. A NULL return from stb is an ERROR.
* `srgb`: our own float conversion (sRGB -> linear before, linear -> sRGB after), never through 8 bits. The base
  `M0` is never converted; it is the source as loaded, and the objective fits the source's own byte encoding at every
  level. Only the derivation of the deeper targets is in linear light.
* `normal_map`: on the clamped float RGB of each resized level, the viewer's rule verbatim
  (`viewer/bin/nntc_view.hlsl`, key V): `n = rgb * 2 - 1`, `len = length(n)`; if `len > 0.5 && n.z > 0` then
  `rgb = saturate(n / len * 0.5 + 0.5)`, else left as filtered. Float arithmetic, `sqrtf`, the same thresholds.
  Object-space normals are out of scope and documented as such.
* Footprint: the current box drops the last column/row of an odd plane (a sub-rectangle of the base); a direct resize
  covers the whole base. Reachable whenever an odd plane is halved, e.g. 68 -> 34 -> 17 -> 8 at the default
  `--mip-min 8`. Documented in DESIGN 4.2 and gated with a 44x44 crop at `--mip-min 4` (planes 44, 22, 11, 5; 44 is
  divisible by 4, so the padding WARNING does not fire there).
* Refused by name: a non-default filter (or srgb / edge / normal_map) with `--mips 0`; an input too small to have any
  chain level with a non-default setting (known only after the load; `tiny.png --mip-min 64` reaches it).
  (Superseded at v1.0.15: a filter with no chain level is idle, not refused, the owner's rule.)
* A padded input (not divisible by 4) is a testing convenience: the filters, wrap included, treat the padded texture
  as the image. The existing padding WARNING gains that clause.

## 4. What the report and the asset say

* The report's `psnr texture t` lines keep `dB` right after the number (the gate's regex) and carry the filter,
  srgb and normal_map inside the trailing parenthetical. The `mip psnr` line keeps `M.. dB` and its trailing prose
  says "against the run's own source chain". The `--diag` header likewise.
* The chain the run generates is the golden reference: the objective, every level's PSNR and the `--png` source
  images are all against it. Nothing regenerates a reference.
* The output JSON's `source` object gains an informational **`inputs`** list (NOT `textures`, which is the two
  latent levels throughout FORMAT.md): one entry per input with `file` (base name only), `type`, `filter`, `srgb`,
  `edge`, `normal_map`, `weight`, `rgb_weights`. So a consumer can see whether an input was a normal map and whether
  its chain was derived in linear light, when known. Strings are JSON-escaped (the writer concatenates text today
  and has no escaper; one is added). Both readers ignore unknown keys. FORMAT section 4 lists the key as informational.
* Determinism: the gate keeps asserting byte-identical reruns on one machine. Nothing is promised across platforms or
  compilers; the stb chains are measured across Windows and WSL once and reported as PSNR agreement.

## 5. Code touch points

* `src/options.h`: `weights` and `mip_filter` default empty; `TextureSettings {file, type, filter, srgb, edge,
  normal_map, weight, rgb_weights[3], source of each}` and the merged per-texture list; the JSON path; `quiet`
  redefined.
* `src/main.cpp`: `parse_string_list`, the `--mip-filter` arm, the JSON loader, the list rule in `validate_options`,
  the override warnings, the settings rows inside the banner, `cw` from per-texture rgb weights, the report and
  `--diag` lines, the round-line and report `quiet` changes, `--help` / `--help-advanced`, the padding clause.
* The JSON reader: MOVE `viewer/json.h` and `viewer/nntc_json.h` to `src/` (both viewer targets already have `src/`
  on their include path; `VENDORED` matches by base name) and list them there; README's licence rows follow.
* `src/mips.cpp`, `src/image.h`: `build_source_chain(chain, planes, settings)`, the split / re-interleave helpers,
  `srgb_to_linear` / `linear_to_srgb` on floats, the clamp, `normal_renorm_rgb`; the stb implementation macro inside
  the warning-suppressed include block in `main.cpp`.
* `src/model.h`, `src/export.cpp`: the per-input settings carried on `Model`, the escaper, `source.inputs`.
* `CMakeLists.txt`: the executable's source list names no stb header today; list all three or none (all three).
* Docs: DESIGN 4.2 (the chain, the filters, the golden reference, the edge mode is the filter's, the footprint
  note), DESIGN 6 (what determinism promises), the "sharper source filter" future-work bullet rewritten, FORMAT
  section 4 (`source.inputs`), README (the JSON, `--mip-filter`, `--quiet`, the list rule), the root
  `PRIOR_ART_DISCLOSURE.md` section 10 item 6, `--help`.

## 6. Gate cases

1. One `tiny.png` encode per filter name (four), through the existing loop and level checks.
2. `--mip-filter default` byte-identical to no flag.
3. `tiny_src_M1.png` differs between `default`, `mitchell` and a `srgb` run (a mis-wired name cannot pass).
4. `--weights` count mismatch in both directions is an error naming the flag and both counts; a two-texture run with
   no `--weights` still works.
5. `--mip-filter` count mismatch; an unknown name (the message lists the four); a non-default filter with `--mips 0`
   refused, `default` with `--mips 0` accepted; `tiny.png --mip-min 64 --mip-filter mitchell` refused (no level).
   (Superseded at v1.0.15: a filter with no chain level is idle, not refused, the owner's rule.)
6. A mixed list (`default,mitchell`) on two inputs: `PREFIX_src_t0_M1.png` byte-identical to the all-default run's.
7. Determinism pairs for `mitchell` + `srgb` and for `normal_map`.
8. The 44x44 crop at `--mip-min 4` under `default` and `box`: both solve; both print planes 44, 22, 11, 5.
9. The normal renorm: an 8x8 synthetic RGB PNG (a new small 8-bit writer beside `write_grey16`) made of 4x4 uniform
   blocks: grey (128,128,128), black (0,0,0), z<0 (128,128,0), ordinary (200,140,220); `filter` `box` (exact 2x2
   mean at an integer ratio), `--mip-min 4`, `--png 1`. From `_src_M1.png` (4x4): the three excluded blocks equal
   the plain filter's output, the fourth is unit length within 0.01.
10. The JSON path: a material JSON with every key set (settings rows show `json`); the same plus `--weights` shows
    `command line` and prints the override WARNING on stdout; an unknown key refused; `srgb` with `default` refused;
    a relative `file` resolves against the JSON's directory; the asset is named after the JSON's stem; the output
    JSON carries `source.inputs` with one entry per input; `--quiet` prints no banner and no round lines but the
    report.
11. `tools/dds_decode.py --grid` and `nntc_view --shot` succeed on an asset written from a JSON material.

## 7. Up to six textures (owner-approved, same stage)

The four-texture cap is one constant in each of three places: `MAX_NOUT` (12) in `src/device.cuh` and the `dir[12]`
in `src/model.h`, the input-count and `--weights` checks in `src/main.cpp`, and the viewer shader's constant buffer
(`W[72]` float4 for 12 rows of 24 columns, `bias[3]`, `outv[12]`) with its shape check in `viewer/main.cpp`. All go
to 18 outputs, `W[108]`, `bias[5]`. The kernels' per-output arrays are sized by the cap at compile time; the cost in
registers is accepted as is (encoding is seconds; the before-and-after one-texture timing is logged as a number and
nothing is templated). The layout does not change, so six textures share the same four plus four channels; `--diag`
shows which one starves. Gate: a six-input case from crops of `tiny.png`, and the viewer's key `N` cycling six.
Inputs are whatever `stb_image` loads as 8-bit; nothing special for 16-bit files.

## 8. Test materials

Material JSONs for the sets already measured live beside their images in the previous tree (`*_material.json`):
`m`, `mb`, `mc`, `md`, `me`, `mf`, `mg`, each naming its normal map (found by measurement: m1, mb1, mc3, md4, me4,
mf4, mg4 are unit-length with z > 0) with `normal_map`, its colour texture with `mitchell` + `srgb`, and its masks
with `mitchell`; plus `m_vanilla_material.json`, the same four files with no keys but `file`, which must produce the
same asset as the bare command line.

**v0.10g.2-weight-clamp.** A weight that overflows the float `cw` is no longer refused: texture and channel weights are
CLAMPED to the sane range [0, 128] (the owner's call, "sane but permissive"), a positive value below 1e-6 is raised to
it, negative stays refused, and each clamp prints one WARNING naming the texture and the value; the settings row shows
the clamped weight. The gate's case asserts the clamp, the single warning and the absence of a nan objective.
