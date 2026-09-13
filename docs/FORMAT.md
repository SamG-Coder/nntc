# FORMAT.md -- the NNTC asset specification

Everything a consumer needs to decode an NNTC asset (the files `nntc_encode` writes), and nothing about how it was produced (that is
`docs/DESIGN.md`). The asset is three files - four when level 0 takes two - and no runtime library.

One thing to have in mind before the tables start: **under the default `--l0 bc8` the level-0 `.dds` holds BC4 / BC5
blocks the encoder OPTIMISED**, block by block against the decoder's own output error and then repacked in outer
passes (`docs/DESIGN.md` 3.6). Nothing about reading the asset changes for that - a refined block is an ordinary BC4
block with ordinary endpoints and selectors - but nothing has to be packed, unpacked or repaired at load either: a
consumer binds the file's own textures and samples them.

## 1. The files

| file | what it holds |
|---|---|
| `PREFIX_lat0.dds` | level 0: a full-resolution plane of 1-4 channels, 8 bits each under the default `--l0 bc8` and 1-4 under `--l0 palette`, with its whole mip chain |
| `PREFIX_lat1.dds` | level 1: a plane at a quarter of the resolution per axis, 1-4 channels, 4-8 bits each, with its chain |
| `PREFIX_nntc.json` | the sizes, the dequantisation of both levels and the affine decoder |

When level 0 has three or four channels and is block-compressed (the default), it takes **two** files,
`PREFIX_lat0a.dds` and `PREFIX_lat0b.dds`, in place of `PREFIX_lat0.dds`. Four channels put 0-1 in the first and 2-3 in
the second, both `BC5_UNORM`; **three put 0-1 in a `BC5_UNORM` and channel 2 alone in a `BC4_UNORM`**, so the two files
do not share a format and level 0 costs 12 bits per texel rather than the 16 a padded fourth channel would. Section 7
has the rule and the JSON says which case it is, per file. The encoder writes the `.dds` files beside the descriptor and names
them by base name alone; a reader takes a name that is an absolute path as written and any other name from the
descriptor's own directory, so a hand-authored descriptor may point anywhere.

The descriptor is `PREFIX_nntc.json`, not `PREFIX.json`: the prefix is usually an input's own base name, and a
descriptor of `PREFIX.json` written beside a material named `PREFIX.json` would have been written over it. `-o` may
name the descriptor outright instead -- `-o out/name.json` writes exactly `out/name.json` beside `out/name_lat0.dds`
-- so a consumer must take the descriptor's path as given and derive nothing from it; the `.dds` names come from the
file's own entries. Assets written before this rule carry `PREFIX.json`, and every reader in this tree still opens
them.

A pixel of output mip `m` is decoded from mip `m` of **both** textures. **Both textures are read through the hardware
filter** -- one ordinary `Sample()` each, at the same texture coordinate, with whatever filter the consumer selects:

```
z0 = dequantise0(sample of level 0 at (u, v))                            C0 values
z1 = dequantise1(sample of level 1 at (u, v))                            C1 values
phi = [ z1_0 .. z1_(C1-1) , z0_0 .. z0_(C0-1) , z0_i * z1_j (i outer, j inner) ]
out = W phi + b                                                          3T values, clamped to [0,1] on write-out
```

and nothing else. Level 0 is at the decoded extent, so at the centre of a base-level pixel its bilinear sample is
exactly that one texel's value; level 1, at a quarter of the extent, blends four texels there. Between texel centres
both are filtered, and the asset is fitted for that (section 5). Sections 5 and 6 are the two rules that are easy to
get wrong: the LOD bias on level 1, and the fact that the dequantisation happens **after** the hardware's blend.

## 2. The .dds byte layout

Both files are DX10 `.dds`: the 4-byte magic, a 124-byte `DDS_HEADER` (31 dwords) and a 20-byte `DDS_HEADER_DXT10`
(5 dwords) -- 148 bytes in all -- then the mip levels base first, tightly packed, with no padding or alignment between
them. Every dword is little-endian.

| offset | field | value |
|---|---|---|
| 0 | magic | `0x20534444` (`"DDS "`) |
| 4 | `dwSize` | 124 |
| 8 | `dwFlags` | `0x00001007` plus `DDSD_MIPMAPCOUNT` `0x20000` when more than one level is stored, plus `DDSD_PITCH` `0x8` (uncompressed) or `DDSD_LINEARSIZE` `0x00080000` (block-compressed) |
| 12 | `dwHeight` | the base level's height |
| 16 | `dwWidth` | the base level's width |
| 20 | `dwPitchOrLinearSize` | uncompressed: the base level's ROW PITCH, `width * channels_stored` bytes. Block-compressed: the base level's TOTAL byte size |
| 24 | `dwDepth` | 1 |
| 28 | `dwMipMapCount` | the number of stored levels, the base included |
| 32 | `dwReserved1[11]` | 0 |
| 76 | `ddspf.dwSize` | 32 |
| 80 | `ddspf.dwFlags` | `0x4` (`DDPF_FOURCC`) |
| 84 | `ddspf.dwFourCC` | `0x30315844` (`"DX10"`) |
| 88 | the rest of `ddspf` | 0 |
| 108 | `dwCaps` | `0x00401008` (TEXTURE, MIPMAP, COMPLEX) with a chain, `0x00001000` (TEXTURE alone) when one level is stored |
| 112 | `dwCaps2`, `dwCaps3`, `dwCaps4`, `dwReserved2` | 0 |
| 128 | `dxgiFormat` | the format id, section 4 |
| 132 | `resourceDimension` | 3 (`TEXTURE2D`) |
| 136 | `miscFlag` | 0 |
| 140 | `arraySize` | 1 |
| 144 | `miscFlags2` | 0 (`DDS_ALPHA_MODE_UNKNOWN`) |

`0x00001007` is `CAPS | HEIGHT | WIDTH | PIXELFORMAT`; the specification attaches `DDSD_PITCH` or `DDSD_LINEARSIZE` to
say which of the two things `dwPitchOrLinearSize` is, and `DDSD_MIPMAPCOUNT` only when there is a chain to count. So
the whole word reads `0x0002100F` uncompressed with mips, `0x000A1007` compressed with mips, and `0x0000100F` /
`0x00081007` for a single stored level -- which is what `--mips 0` writes, and which then carries `dwCaps`
`0x00001000` rather than the MIPMAP and COMPLEX bits. That is DirectXTex's own rule; `dwMipMapCount` is 1 either way.

**The levels.** Level `m`'s size is the base halved `m` times with floor halving and a floor of 1:
`w_m = max(1, w_(m-1) / 2)`. The `.json`'s `mip_sizes` lists every one of them, so a reader never has to derive them.
A level's bytes are:

* uncompressed: `w_m * h_m * channels_stored` bytes, rows top to bottom, `channels_stored` bytes per texel, channel 0
  first (R, then G, then B, then A);
* block-compressed: `ceil(w_m / 4) * ceil(h_m / 4)` blocks of 8 bytes (BC4) or 16 (BC5), in block rows, top to bottom.

The two files of a two-file level 0 hold the same sizes and the same level count. Their FORMATS may differ: at three
channels the first is a `BC5_UNORM` and the second a `BC4_UNORM` of 8 bytes per block, so a reader must take each
file's block size from its own header (and from its own `"files"` entry) rather than from the first one's.

## 3. The texel of an uncompressed plane

A stored channel is one byte and carries a quantisation index `k` of `bits` bits, `bits` from `bits_per_channel`. The
byte is the index in its top bits with the index **replicated** downward into the rest:

```
byte = 0;  for (sh = 8 - bits; sh > -bits; sh -= bits)  byte |= sh >= 0 ? (k << sh) : (k >> -sh);
```

So 3 bits give `k k k` (the last copy truncated to the two bits that are left), 4 bits give `k * 17`, and 8 bits give
`k` itself.

Two properties follow, and both are used:

* **the index is recoverable exactly**: `k = byte >> (8 - bits)`, whatever the bit depth. A reader that wants the
  index itself takes this path;
* **the byte can go straight to a UNORM sampler**: the replication makes index 0 the byte 0 and the top index the byte
  255, so `lo + (byte / 255) * (hi - lo)` is a dequantisation on the sampled float, which is what the GPU does.

**The value of an index is `rep(k) / 255` of the range, not `k / levels`.** The two are the same number at 1, 2, 4 and
8 bits and are not at 3, 5, 6 and 7 -- at 3 bits `rep(1)` is 36 and `255 / 7` is 36.43, half a byte apart -- and what a
sampler returns is the first. The encoder fits its grid on exactly that, so an asset's published values ARE the values
its consumer reads. A reader should therefore dequantise the byte and not the index; `palette[k]` (section 4) lists the
same numbers for level 0, and for level 1 the `note` spells the byte rule out.

A channel a level stores but the layout does not use -- the fourth channel of an `R8G8B8A8` plane carrying three --
holds 255.

## 4. The JSON keys

The file is one object. "Required" below means a decoder is wrong without it; "informational" means it repeats
something else in prose, or is carried for a reader written against a wider format.

**What `nntc_view` actually does with each of them**, since "required" is a statement about correctness and not about
this program. It REFUSES the file outright over: a `format` that is neither `nntc-dds-1` nor the older `ntc-dds-1`, a `textures` array that is not
exactly two entries, a `level` that is not its own index, a `bits_per_channel` entry outside 1..8 (level 0) or 4..8
(level 1), a `files` array that is not two names, a missing or too-short `lo` / `hi`, a `decoder.type` that is not
`bilinear`, a non-empty `hidden`, an `output` that is not `identity`, a `terms` list that is not `a`, `b`, `sc` in that
order, a `nin` that does not match those terms, and a `layers` array that is not one layer of exactly `nout x nin`. It
DEFAULTS quietly over: `source.width` / `source.height` (0, display only), `textures_out` (1), `block` (4),
`lod_bias_level1` (2 when `block` > 1), and a `bits_per_channel` entry that is absent (8). It CROSS-CHECKS against the
`.dds` headers rather than trusting: each `files` entry's `dxgi_format_id` and `channels_stored`, and
`channels_used` against what the files between them store; the extent, the level count and the formats come from the
headers themselves. And it never reads `dequantise.kind` at all: it takes `lo` and `hi` and applies
`lo + sample * (hi - lo)`, which is the right answer for the `range` kind and for a `palette` whose `lo` / `hi` are its
first and last entries, so both kinds decode from the same two arrays.

| key | kind | meaning |
|---|---|---|
| `format` | required | `"nntc-dds-1"`. A reader must refuse anything else, with one exception: `"ntc-dds-1"` is the same format under the name it had before the project was renamed, and the readers in this tree accept it with a note |
| `source.width`, `source.height` | required | the ORIGINAL image size: the extent worth displaying |
| `source.padded`, `source.padded_width`, `source.padded_height` | present only when the input was padded | the input was not a multiple of 4 and was padded by edge replication; the textures are the padded size, and everything past `source.width/height` is replication |
| `source.inputs` | informational | one entry per INPUT image, in texture order: entry `t` describes output channels `3t..3t+2`. Each carries `file` (the base name), `type` (a free label the encoder never interprets), `filter`, `srgb`, `edge`, `normal_map`, `weight` and `rgb_weights` -- what the input was and how its deeper source levels were derived (`DESIGN.md` 4.2). `weight` and `rgb_weights` are the values AS GIVEN, before the per-texture normalisation the objective applies: the three rgb weights are normalised to mean 1 inside each texture when `cw` is built, and what is published here is what was asked for, not what came out of that. It is `inputs` and not `textures` because `textures` is the two LATENT levels everywhere else in this format. A decoder needs none of it; it is published because whether an input was a tangent-space normal map, and whether its chain was derived in linear light, cannot be recovered from the two latent textures and both change what the asset means |
| `decode.width`, `decode.height` | required | the base level's decoded extent: the padded size when there was padding, the source size otherwise |
| `decode.textures_out` | required | `T`, the material's texture count, 1 to 6; the decoder has `nout = 3T` outputs and texture `t` takes outputs `3t..3t+2` as RGB |
| `block` | required | the resolution ratio between the levels, always 4 |
| `lod_bias_level1` | required | `log2(block)` = 2, the `MipLODBias` level 1's sampler must carry (section 5) |
| `sampling` | informational | sections 5 and 6 in prose |
| `textures` | required | exactly two entries, level 0 then level 1 |
| `decoder` | required | the one affine layer |

### `textures[l]`

| key | kind | meaning |
|---|---|---|
| `file` | required, unless `files` is present | the `.dds` file name, beside the descriptor |
| `files` | present INSTEAD of `file` for a two-file level 0 | two OBJECTS, `[{channels 0-1}, {the rest}]`, both beside the descriptor. Each carries `file`, `dxgi_format`, `dxgi_format_id` and `channels_stored` for that file alone, because the two need not match: at three channels the first is a BC5 storing 2 and the second a BC4 storing 1. A reader that knows only `file` finds no `file` and stops, rather than decoding half the plane. Assets written before the two files were allowed to differ in format carry a **bare string** in each slot instead of an object; every reader in this tree accepts both, taking the string as the name and cross-checking nothing |
| `level` | required | 0 or 1, and equal to the array index |
| `width`, `height` | required | this texture's base level size (level 1's is `decode.width / block`, floored) |
| `dxgi_format`, `dxgi_format_id` | required | `R8_UNORM` 61, `R8G8_UNORM` 49, `R8G8B8A8_UNORM` 28, `BC4_UNORM` 80, `BC5_UNORM` 83. With two files these describe the FIRST one; each file's own are in its `files` entry |
| `channels_used` | required | `C0` or `C1`: how many channels the decoder reads |
| `channels_stored` | required | how many channels ONE file stores: 4 for an `R8G8B8A8` plane carrying 3, 1 for BC4, 2 for BC5. With two files this is the first file's, and the second file's is in its `files` entry |
| `bc_palette` | present only on a block-compressed texture | which BC palette the plane was fitted against; `"standard"` (section 7) |
| `bits_per_channel` | required | one entry per USED channel: level 0's are all 8 under the default `--l0 bc8` and 1-4, possibly differing per channel, under `--l0 palette`; level 1's are all the same 4-8 |
| `mip_count`, `mip_sizes` | required | the stored level count, and every level's `[w, h]` base first |
| `texel` | informational | section 3, or section 7 for a block format, in prose |
| `dequantise` | required | below |

### `textures[l].dequantise`

Level 0 is `"kind": "range"` under the default `--l0 bc8` and is then read exactly as level 1's is (see the end of
this section); it is `"kind": "palette"` when the asset was written with `--l0 palette`. For the palette kind: `palette` is one array per used channel listing the `2^bits` values in index order,
`lo` and `hi` are each channel's first and last palette entry, and `levels` is `2^bits - 1` per channel. **The palette
is the value the SHIPPED FORMAT decodes to**, which is one of two things:

```
uncompressed        pal_c(k) = -1 + 2 rep(k) / 255              rep = the index replicated over the byte (section 3)
block-compressed    pal_c(k) = -1 + 2 k / (2^bits_c - 1)        the standard BC palette (section 7)
```

The two agree at 1, 2 and 4 bits and differ at 3, by up to half a byte of the span. Either way both of these give the
same number, which is the property the encoder fits and this file's `--grid` check asserts:

```
value = palette[c][byte >> (8 - bits_c)]                 from the index
value = lo[c] + sample * (hi[c] - lo[c])                 what a UNORM sampler feeds a shader
```

Level 1 is `"kind": "range"`: `lo` and `hi` are one value per channel and `levels` is the single integer
`2^bits1 - 1`. One `lo/hi` pair covers the whole mip chain, because the GPU applies one scale and one bias whatever mip
its sampler chose. Level 1 is always uncompressed, so its grid is the replicated byte:

```
value = lo[c] + sample * (hi[c] - lo[c])                 sample = the UNORM float the hardware returns
value = lo[c] + (rep(k) / 255) * (hi[c] - lo[c])         the same number, written on the index
```

`levels` is the index's own range and is what `k = byte >> (8 - bits1)` produces; `lo + (k / levels) * (hi - lo)` is
NOT the stored value at 5, 6 and 7 bits, where `rep(k) / 255` and `k / levels` part company. At 4 and 8 bits they are
the same. Both objects also carry an informational `note` saying which rule they follow.

**Level 0 under `--l0 bc8`** is `"kind": "range"` too, with `bits_per_channel` `[8, ...]`, `levels` the single integer
`255`, and one `lo/hi` pair per used channel covering the whole mip chain. There is no `palette` array, because there is
no index on a palette: the plane was solved as a continuous 8-bit plane, BC4 / BC5-encoded, and the pack then REFINED
block by block against the decoder's own error and repacked in outer passes, so the byte a sampler returns is a block
palette entry the encoder chose for what the decoder makes of it - not a quantisation index of anything. The rule is
the one rule:

```
value = lo[c] + sample * (hi[c] - lo[c])                 sample = the UNORM float the hardware returns
```

At 8 bits the index rule `lo + k / 255 * (hi - lo)` is the same number anyway, so a reader that recovers `k` with
`byte >> 0` and dequantises from it lands in the same place -- but the byte rule is the one the encoder fitted and the
one the `texel` string publishes. A reader must therefore look at `kind` and not at the level index: `kind` is the
authority on how a plane is dequantised, and both levels can carry either.

### `decoder`

| key | kind | meaning |
|---|---|---|
| `type` | required | `"bilinear"`. A reader must refuse anything else |
| `terms` | required | `"a b sc"`: the feature groups, whitespace-separated, IN ORDER (below). A reader must refuse a different order rather than reorder it |
| `C0`, `C1` | required | the two channel counts, equal to `textures[*].channels_used` |
| `features` | informational | the feature order in prose |
| `nin` | required | the feature count `C1 + C0 + C0 * C1`, and the layer's column count |
| `nout` | required | `3 * decode.textures_out`, and the layer's row count. At most 18, which is the six-texture cap; `nntc_view`'s shader refuses more |
| `output` | required | `"identity"`: the layer's result IS the output. A consumer writing 8-bit pixels clamps to [0,1] first |
| `hidden` | required to be empty | `[]`. Anything else would be a decoder this format does not describe, and a reader must refuse it |
| `activation`, `leak` | informational | `"leaky_relu"` and `0`, which with no hidden layer describe nothing. They are carried so that a reader written against a deeper decoder can parse this file, and both are ignored |
| `layers` | required | exactly one entry: `rows` = `nout`, `cols` = `nin`, `weights` a `rows * cols` array in ROW-MAJOR order (output `r`'s row is `weights[r * cols]` to `weights[r * cols + cols - 1]`), `bias` a `rows` array |

### The feature order

`terms` is `a b sc` and `phi` is built in that order, `nin` entries in all:

```
[a]   phi[0 .. C1-1]                      z1_j                    the level-1 channels
[b]   phi[C1 .. C1+C0-1]                  z0_i                    the level-0 channels
[sc]  phi[C1+C0 .. C1+C0+C0*C1-1]         z0_i * z1_j             i outer, j inner
```

This order is the contract between the encoder, `tools/dds_decode.py` and the viewer's shader. Getting it wrong gives
an image that is recognisable and wrong, so a reader should assert that `nin` is `C1 + C0 + C0 * C1`.

## 5. The sampling rule and the LOD bias

**One coordinate, two filtered samples.** Both textures are sampled at the same `(u, v)` with the same filter. The
bilinear rule is the standard one: `x = u * W - 0.5`, the two texel indices `floor(x)` and `floor(x) + 1` **clamped**
into `[0, W-1]` (so a position past the last texel centre has its two taps on the same texel and their weights add),
the same in `y`, and the four taps blended. Level 1 has a quarter of the texels per axis, so one pixel of movement in
the image is a quarter of a texel there.

**At a base-level texel centre.** At `u = (p.x + 0.5) / W`, `v = (p.y + 0.5) / H` level 0's bilinear sample lands
exactly on texel `p` -- level 0 is at the decoded extent, so the fractional parts are zero and the blend has weight 1
on one tap. That is why the decode at texel centres can be written with level 0's own stored value, and why the
encoder's centre sites call it a nearest read; it is a statement about that position, not about the sampler state.
Everywhere else level 0 is filtered like any other texture, and the encoder fits it at fixed fractional positions
precisely so that those filtered samples decode correctly -- that is the whole point of the representation.

**Down the chain.** Output mip `m` is decoded from mip `m` of **both** textures -- the 1:1 rule. The hardware picks
each texture's mip from that texture's own texel density, and level 1 has `1/block` of the texels per axis, so its LOD
sits `log2(block)` below level 0's: when level 0 is at mip 1, level 1 is still at mip 0, a plane that was fitted beside
level 0's mip 0. So:

> **Level 1's sampler must carry `MipLODBias = lod_bias_level1` (= 2).**

With the bias its LOD equals level 0's at every distance; at 1:1 on screen it goes from -2 to 0, still mip 0, so
nothing changes at the base. Without it the base is right and everything from mip 1 down is decoded from the wrong
colour plane, which looks like splotching at level-1 texel scale. No shader arithmetic is involved: it is one
sampler-state field. `nntc_view` sets it by default and key `L` turns it off, which is the quickest way to see what it
is for.

Both textures are otherwise sampled with the same filter and the same addressing. A trilinear filter is fine and needs
nothing else, and so is an anisotropic one: anisotropy is several trilinear samples along the footprint's long axis,
the dequantisation is affine and the decoder is affine in the samples, so a blend of latent samples decodes to the
blend of the decodes (section 6). `nntc_view` uses anisotropic filtering by default in its trilinear mode.

## 6. Dequantisation after sampling

The dequantisation of both levels is **affine** and is applied **after** the sampler's blend, never before. That is the
invariant the whole representation rests on. For level 1, with tap weights `w_t` summing to 1,

```
dequantise( sum_t w_t * byte_t / 255 )  =  lo + (sum_t w_t * byte_t / 255) * (hi - lo)
                                        =  sum_t w_t * ( lo + (byte_t / 255) * (hi - lo) )
                                        =  sum_t w_t * dequantise( byte_t / 255 )
```

-- the hardware's blend of the quantised texels IS the blend of the values. That is why a quantised latent can be given
to a bilinear sampler at all, and why the encoder's model of the GPU is exact rather than approximate. A consumer must
therefore sample first and dequantise afterwards; dequantising per texel and blending by hand gives the same answer,
but anything non-affine applied to the bytes before the blend does not.

The decoder's product terms `z0_i * z1_j` are of course not affine -- but they are formed after both samples have been
taken and dequantised, so the invariant is untouched.

## 7. Level 0 as BC4 / BC5

Level 0 is a full-resolution plane of one to four channels and is block-compressed in every default configuration.
Under `--l0 palette` a channel carries one to four bits, so a byte per channel would spend most of what it stores on
nothing, and `--bc0 0` writes that uncompressed plane of sections 2 and 3 instead. Under the default `--l0 bc8` a
channel is a continuous 8-bit value, `--bc0 0` does not apply and is refused by name, and the block format is the whole
point of the mode rather than a saving applied to it. Level 1 is always uncompressed: its four to eight bits per
channel would not survive a block palette.

**The formats and the files.** BC5 is the widest block format that carries independent UNORM channels, so the channel
count decides how many files level 0 takes:

| level-0 channels | files | format | bytes per level |
|---|---|---|---|
| 1 | `PREFIX_lat0.dds` | `BC4_UNORM`, DXGI 80 | `ceil(w/4) * ceil(h/4) * 8` |
| 2 | `PREFIX_lat0.dds` | `BC5_UNORM`, DXGI 83 | `ceil(w/4) * ceil(h/4) * 16` |
| 3 | `PREFIX_lat0a.dds` (channels 0-1) and `PREFIX_lat0b.dds` (channel 2) | `BC5_UNORM` then `BC4_UNORM` | `ceil(w/4) * ceil(h/4) * 16`, then the same `* 8` |
| 4 | `PREFIX_lat0a.dds` (channels 0-1) and `PREFIX_lat0b.dds` (channels 2-3) | two `BC5_UNORM` | `ceil(w/4) * ceil(h/4) * 16`, twice |

Both files of a two-file level 0 carry the whole mip chain, at the same sizes. Every channel in them is a channel the
layout uses: a three-channel level 0 puts its odd channel in a BC4 rather than padding a second BC5, so no file holds a
constant plane nobody reads, and the layout costs 4, 8, 12 or 16 bits per texel at one to four channels.

**The header** is the header of section 2 with two fields changed, as the DDS specification requires for a compressed
format: `dwFlags` carries `DDSD_LINEARSIZE` (`0x00080000`) instead of `DDSD_PITCH` (`0x8`), so the whole word is
`0x000A1007`; and `dwPitchOrLinearSize` is the base level's byte size rather than a row pitch. The levels still follow
the 148-byte header base-first, tightly packed, and a level's rows are block rows of `ceil(w/4)` blocks.

**The block.** A BC4 block is 8 bytes covering 4x4 texels of one channel: two 8-bit endpoints `r0`, `r1` and then
sixteen 3-bit selectors packed LSB-first, texel 0 in the lowest bits, row-major. The selector indexes an eight-value
palette evaluated from the endpoints:

```
r0 >  r1    p = [r0, r1, (6 r0 + r1) / 7, (5 r0 + 2 r1) / 7, ... , (r0 + 6 r1) / 7]
r0 <= r1    p = [r0, r1, (4 r0 + r1) / 5, (3 r0 + 2 r1) / 5, (2 r0 + 3 r1) / 5, (r0 + 4 r1) / 5, 0, 255]
```

A BC5 block is 16 bytes: the red channel's BC4 block and then the green channel's.

**`bc_palette`.** `textures[0].bc_palette` says which palette the plane was fitted against, and is `"standard"`: the
`k / 7` interpolation above, which is what the Direct3D functional specification defines and what every decoder --
hardware or software, on any vendor -- is required to return. Some hardware evaluates the interior of the eight-value
palette with six-bit weights (`n / 64`), which lands within half a byte of `k / 7`; fitting against that instead would
be a vendor-specific mode and would make the asset wrong everywhere else, so it is not what this field ever says today.
It exists so that such a mode could name itself if one were ever added.

**Why 1-3 bits are exact.** With the endpoints 255 and 0 the eight-value palette IS the 3-bit grid: `(8 - i) * 255 / 7`
for `i = 1..7`, together with 255, gives `k * 255 / 7` for `k = 7..0`. So a 3-bit plane packs by relabelling the index
as a selector and nothing is lost; two bits use the six-value mode with the endpoints 85 and 170, whose palette holds
0, 85, 170, 255; one bit uses that mode's fixed 0 and 255. Rounded to bytes those are 255, 219, 182, 146, 109, 73, 36,
0 -- the same bytes the uncompressed plane's replication produces -- so `byte >> (8 - bits)` still recovers `k`
exactly, which is what `bc_check` asserts per channel.

What the block format does NOT preserve is the exact value: a sampler returns `k / 7` of full scale here and
`rep(k) / 255` from the uncompressed plane, and at 3 bits those differ by up to half a byte. That is why `dequantise`
publishes the palette of the format actually shipped, and why `--bc0 both` writes two assets that decode to two
slightly different pictures from one solve rather than to the same one.

At four bits and more it cannot: sixteen or more distinct values do not fit in one block's eight palette entries. The
block's lowest and highest index values become the endpoints and each texel takes the nearest palette value, and the
encoder prints the packing PSNR against the exact index values.

**`--l0 bc8`, the default.** The file is the same file -- the same BC4 / BC5 blocks, the same standard palette, the
same `bc_palette: "standard"` -- and only what the encoder put in it differs: the plane was solved as a *continuous*
8-bit plane, packed, and then **kept being optimised as BC**. Every 4x4 block's endpoints and selectors are re-chosen
to minimise the decoder's own output error, a candidate taken only when that error falls, and the whole model is
refitted and the plane repacked in outer passes taken only when the shipped objective falls (`docs/DESIGN.md` section
3.6). So a block's eight values are fitted along a line chosen for what the decoder makes of them rather than
approximating a fixed 16-level grid, and what the file holds is the representation that was measured -- not a
compressed copy of something else. Nothing about how the file is READ changes: a refined block is an
ordinary BC4 block with ordinary endpoints and selectors. There is no index to recover and nothing is lossless: `bits_per_channel` is `[8, ...]`, `dequantise` is
`"kind": "range"` (section 4), `texel` says so, and `bc_check` reports the packing PSNR without a lossless claim. A
consumer does nothing different: it samples the block-compressed texture and applies `lo + sample * (hi - lo)`, which
is the same thing it does for every other asset this format describes.

**What the JSON says.** `textures[0]` carries `"file"` for one file and `"files": [a, b]` -- two objects, and then no
`"file"` -- for two; `dxgi_format` / `dxgi_format_id` name the block format of the entry (the first file, when there are
two) and each `files` entry repeats them for its own file; `channels_stored` is what ONE file stores, which is 1 for
BC4 and 2 for BC5, while `channels_used` stays the layout's channel count; `bc_palette` is as above; and `texel`
describes the block rather than the byte. `dequantise` keeps its shape -- the palette, `lo`, `hi` and `levels` of the
index -- but the palette itself is the BC one, `-1 + 2k / (2^bits - 1)`, because that is what this file decodes to.
Under `--l0 bc8` `dequantise` is `"kind": "range"` instead and carries no palette at all (section 4).

## 8. Reading the asset without this tree

`tools/dds_decode.py` is a complete second reader in about four hundred lines of Python: it parses the headers of
section 2, decodes the BC blocks of section 7, dequantises the sampled byte by section 6 (the index shift of section 3
survives in it only as a check that the block format kept the index), builds `phi` in the order of section 4 and
applies the one layer. `--grid` is the smallest check of all and needs no image: it asserts, index by index, that the
value the descriptor publishes is the value a sampler returns. It shares no code with the encoder, and the release gate
runs it on every asset it writes. It is the shortest answer to "what exactly does a consumer have to do".
