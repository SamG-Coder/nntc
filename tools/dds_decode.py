"""An independent reader of the shipped asset.

This script shares no code with the encoder: it parses the two DX10 .dds headers itself, recovers the quantisation index
from each byte, dequantises by the JSON, samples level 1 bilinearly at the pixel centres with clamp-to-edge addressing,
builds phi in the JSON's own term order, applies the single affine layer and rounds to 8 bits. It is the check that the
asset says what the encoder meant, with nothing of the encoder trusted.

    python tools/dds_decode.py PREFIX --ref PREFIX_recon
        decode every stored level and report max |diff| against the encoder's own reconstruction PNGs.

    python tools/dds_decode.py PREFIX --grid
        the grid check, from the descriptor alone: the value published for every quantisation index is the value a
        sampler returns for it, which is what the encoder fitted against.

    PREFIX is the asset's prefix, not a file name: the descriptor is PREFIX_nntc.json and the .dds files it names sit
    beside it. The descriptor path is accepted in its place - the other checkers take that spelling - and stripped back
    to the prefix. An asset written before the _nntc suffix, whose descriptor is PREFIX.json, is read too, with a note,
    as is a descriptor whose "format" is the older ntc-dds-1, the same format under the name it had before the rename.

    python tools/dds_decode.py PREFIX --psnr source.png [source2.png ...]
        decode level 0 and report the PSNR of each texture against its source over the original extent.

    python tools/dds_decode.py PREFIX --psnr-levels [source0.png source1.png ...]
        decode EVERY stored level and report the PSNR of each texture at each level against the encoder's
        PREFIX_src_M<m>.png chain, which is the run's own source chain that level was fitted against. The base is
        compared against the original source images when they are named, and against PREFIX_src_M0.png -- the same
        picture, cropped to the original extent -- when they are not. The two summary lines are the encoder's own
        `psnr` and `mip psnr` report lines recomputed from the asset alone, so the two can be compared decibel for
        decibel: nothing of the encoder's own arithmetic is trusted, only the bytes it wrote.

    A named source is read the way the ENCODER reads it (load_rgb8): a 16-bit image is reduced to its top byte, as
    stb_image does, so a 16-bit greyscale height map is compared against the same eight bits the encoder was fitted to.
"""

import json
import os
import struct
import sys

import numpy as np
from PIL import Image


def decode_bc4(blocks, w, h):
    """One BC4 plane: two endpoint bytes and sixteen 3-bit selectors per 4x4 block, LSB-first and row-major.

    The palette a decoder evaluates from the endpoints is [r0, r1] and then six or four interpolants: with r0 > r1 the
    six are ((8 - i) r0 + (i - 1) r1) / 7, and with r0 <= r1 the four are the same over 5 with 0 and 255 in the last
    two slots. That is the palette of the Direct3D functional specification, and the one the encoder fits against.

    The result is kept in FLOAT, not rounded to a byte: the palette entry of a 3-bit index is 255 k / 7, which is not a
    whole number, and what a sampler hands the shader is that value over 255 and not its rounding. Rounding here would
    put the reader a third of a byte away from both the hardware and the encoder.
    """
    bx, by = (w + 3) // 4, (h + 3) // 4
    blk = np.frombuffer(blocks, dtype=np.uint8).reshape(by, bx, 8)
    r0, r1 = blk[:, :, 0].astype(np.float64), blk[:, :, 1].astype(np.float64)
    sel = np.zeros((by, bx), dtype=np.uint64)
    for i in range(6):
        sel |= blk[:, :, 2 + i].astype(np.uint64) << np.uint64(8 * i)
    pal = np.zeros((by, bx, 8), dtype=np.float64)
    pal[:, :, 0], pal[:, :, 1] = r0, r1
    wide = r0 > r1
    for i in range(2, 8):
        eight = ((8 - i) * r0 + (i - 1) * r1) / 7.0
        if i < 6:
            narrow = ((6 - i) * r0 + (i - 1) * r1) / 5.0
        else:
            narrow = np.zeros_like(r0) if i == 6 else np.full_like(r0, 255.0)
        pal[:, :, i] = np.where(wide, eight, narrow)
    out = np.zeros((by * 4, bx * 4), dtype=np.float64)
    for t in range(16):
        k = ((sel >> np.uint64(3 * t)) & np.uint64(7)).astype(np.intp)
        out[t // 4::4, t % 4::4] = np.take_along_axis(pal, k[:, :, None], axis=2)[:, :, 0]
    return out[:h, :w]


def decode_bc(blocks, w, h, channels):
    """A BC4 level (one channel, 8 bytes a block) or a BC5 level (two channels, their BC4 blocks in order)."""
    if channels == 1:
        return decode_bc4(blocks, w, h)[:, :, None]
    bx, by = (w + 3) // 4, (h + 3) // 4
    pairs = np.frombuffer(blocks, dtype=np.uint8).reshape(by * bx, 16)
    return np.stack([decode_bc4(pairs[:, 8 * c:8 * c + 8].tobytes(), w, h) for c in range(2)], axis=-1)


def read_dds(path):
    """Parse a DX10 .dds, asserting every header field the format fixes, and return the levels and the channel count.

    An uncompressed level is width * height * channels bytes, with DDSD_PITCH set and dwPitchOrLinearSize the base
    level's row pitch. A block-compressed one is ceil(w / 4) * ceil(h / 4) blocks of 8 (BC4) or 16 (BC5) bytes, with
    DDSD_LINEARSIZE set instead and dwPitchOrLinearSize the base level's byte size. Either way the levels follow the
    148-byte header base-first with nothing between them, and the file ends exactly where the last one does.
    """
    b = open(path, 'rb').read()
    magic, size, flags, height, width, pitch, depth, nmip = struct.unpack_from('<IIIIIIII', b, 0)
    assert magic == 0x20534444 and size == 124, 'not a .dds header'
    pfsize, pfflags, fourcc = struct.unpack_from('<III', b, 76)
    assert pfsize == 32 and pfflags == 4 and fourcc == 0x30315844, 'not a DX10 pixel format'
    caps, = struct.unpack_from('<I', b, 108)
    dxgi, dim, misc, arr, misc2 = struct.unpack_from('<IIIII', b, 128)
    assert dim == 3 and arr == 1 and misc == 0 and misc2 == 0, 'not a plain 2D texture'
    assert depth == 1, 'depth'
    compressed = dxgi in (80, 83)
    if dxgi not in (61, 49, 28, 80, 83):
        raise SystemExit('%s: DXGI format %d is not one this format uses (R8 61, R8G8 49, R8G8B8A8 28, BC4 80, BC5 83)'
                         % (path, dxgi))
    channels = {61: 1, 49: 2, 28: 4, 80: 1, 83: 2}[dxgi]
    block_bytes = 8 if dxgi == 80 else 16
    assert 1 <= width <= 16384 and 1 <= height <= 16384, (width, height)
    assert 1 <= nmip <= 15 and nmip <= 1 + max(width, height).bit_length() - 1, (nmip, width, height)
    # A single stored level carries neither DDSD_MIPMAPCOUNT (0x20000) nor DDSCAPS_MIPMAP | COMPLEX, which is
    # DirectXTex's own rule and what the writer follows.
    chain = 0x00020000 if nmip > 1 else 0
    assert caps == (0x00401008 if nmip > 1 else 0x00001000), hex(caps)
    if compressed:
        assert flags == 0x00081007 | chain, hex(flags)
        assert pitch == ((width + 3) // 4) * ((height + 3) // 4) * block_bytes, pitch
    else:
        assert flags == 0x0000100F | chain, hex(flags)
        assert pitch == width * channels, (pitch, width * channels)
    off, levels, w, h = 148, [], width, height
    for _ in range(nmip):
        n = ((w + 3) // 4) * ((h + 3) // 4) * block_bytes if compressed else w * h * channels
        raw = b[off:off + n]
        assert len(raw) == n, 'truncated'
        levels.append(decode_bc(raw, w, h, channels) if compressed
                      else np.frombuffer(raw, dtype=np.uint8).reshape(h, w, channels))
        off += n
        w, h = max(1, w >> 1), max(1, h >> 1)
    assert off == len(b), (off, len(b))
    return levels, channels, dxgi


def dequantise(levels, tex, compressed=False):
    """sample -> value, exactly as the GPU does it: value = lo + sample * (hi - lo) with sample = byte / 255.

    That one rule covers both levels and both formats, because the byte is whatever the sampler returns: the stored
    byte of an uncompressed plane (the quantisation index replicated, so the value is lo + rep(k) / 255 * (hi - lo)),
    or the BC palette entry of a block-compressed one (which at 1-3 bits is 255 k / (2 ^ bits - 1) exactly). The
    encoder fits the grid of the format it ships, so this is also the grid it fitted -- and the `palette` array of a
    level-0 entry lists these same numbers, index by index.

    The index shift survives only as a CHECK: a block-compressed level 0 at 1-3 bits is required to be exact, because
    the pack puts the mode's own extremes in the endpoints and the quantisation index in the selector, so the decoded
    value rounds to the index's own byte again. It is not how the value is computed.
    """
    dq = tex['dequantise']
    used = tex['channels_used']
    bits = tex['bits_per_channel']
    lo = np.asarray(dq['lo'], dtype=np.float64)[:used]
    hi = np.asarray(dq['hi'], dtype=np.float64)[:used]
    out = []
    for lv in levels:
        byte = lv[:, :, :used].astype(np.float64)
        if compressed:
            for c in range(used):
                if bits[c] <= 3:
                    rounded = np.floor(byte[:, :, c] + 0.5)
                    k = rounded.astype(np.int64) >> (8 - bits[c])
                    exact = np.floor(k * 255.0 / ((1 << bits[c]) - 1) + 0.5)
                    assert np.array_equal(exact, rounded), 'the BC pack lost the index of channel %d' % c
        out.append(lo + byte / 255.0 * (hi - lo))
    return out


def bilinear(plane, w, h):
    """The hardware rule at the pixel centres: x = (px + 0.5) / w * wp - 0.5, the two taps clamped into the plane."""
    hp, wp, _ = plane.shape
    xs = (np.arange(w) + 0.5) / w * wp - 0.5
    ys = (np.arange(h) + 0.5) / h * hp - 0.5
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    fx = xs - x0
    fy = ys - y0
    x1 = np.clip(x0 + 1, 0, wp - 1)
    x0 = np.clip(x0, 0, wp - 1)
    y1 = np.clip(y0 + 1, 0, hp - 1)
    y0 = np.clip(y0, 0, hp - 1)
    a = plane[y0][:, x0]
    b = plane[y0][:, x1]
    c = plane[y1][:, x0]
    d = plane[y1][:, x1]
    fx = fx[None, :, None]
    fy = fy[:, None, None]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def read_texture(directory, tex):
    """The levels of one JSON texture entry, resolved next to the .json.

    A level named by "file" is one .dds. A level named by "files" is two, which is how three or four level-0 channels
    are stored: BC5 carries two channels, so channels 0-1 are in the first file and what is left in the second, and the
    two are put back side by side here exactly as the shader reads them from two samplers. The second file is NOT
    necessarily the first one's format -- three channels are a BC5 and then a BC4 holding channel 2 alone, which is
    what makes that layout 12 bits per texel rather than 16 -- so each entry of "files" is an object carrying its own
    dxgi_format_id and channels_stored, and those are checked against the header rather than trusted. A bare string is
    accepted there too, which is what assets written before the BC4 second file carry.
    """
    entries = tex['files'] if 'files' in tex else [tex['file']]
    names = [e['file'] if isinstance(e, dict) else e for e in entries]
    parts = [read_dds(os.path.join(directory, n)) for n in names]
    compressed = parts[0][2] in (80, 83)
    for _, _, dxgi in parts:
        assert (dxgi in (80, 83)) == compressed, 'a level is block-compressed in every file or in none'
    for e, (_, channels, dxgi) in zip(entries, parts):
        if isinstance(e, dict):
            assert e.get('dxgi_format_id', dxgi) == dxgi, (e.get('dxgi_format_id'), dxgi)
            assert e.get('channels_stored', channels) == channels, (e.get('channels_stored'), channels)
    if len(parts) == 1:
        levels = parts[0][0]
    else:
        levels = [np.concatenate(pair, axis=-1) for pair in zip(*[p[0] for p in parts])]
    stored = sum(p[1] for p in parts)
    assert stored >= tex['channels_used'], (stored, tex['channels_used'])
    return dequantise(levels, tex, compressed), len(levels)


def prefix_from_argument(arg):
    """The asset prefix meant by ARG, which is the prefix itself or the descriptor path.

    The first argument is documented as the prefix, but the descriptor is the name the user has in front of them - it is
    what the viewer and bc_check take - and handing it over used to end in a traceback about 'NAME_nntc.json_nntc.json'.
    A descriptor spelling is stripped back to the prefix instead: '_nntc.json' first, since that is what the encoder
    writes, and a plain '.json' otherwise, which covers a descriptor named outright by -o.
    """
    for suffix in ('_nntc.json', '.json'):
        if arg.endswith(suffix):
            return arg[:-len(suffix)]
    return arg


def descriptor_path(prefix, named=None):
    """The descriptor of the asset at PREFIX: PREFIX_nntc.json, as the encoder writes it.

    NAMED is the argument itself when it ends in .json and is a file on disk. The caller then asked for that exact
    descriptor and it is opened, silently: the note below is about a name this tool went looking for and did not find,
    and printing it over a file the caller pointed straight at read as though something were wrong with the asset.

    PREFIX.json is accepted as a fallback, with that note, so that an asset written before the _nntc suffix still
    reads. Nothing is guessed when the _nntc name is there.

    Neither existing is a mistyped argument rather than a broken asset, so it is reported as one line naming both names
    that were tried; the caller exits on the None.
    """
    if named is not None:
        return named
    path = prefix + '_nntc.json'
    if not os.path.isfile(path):
        old = prefix + '.json'
        if os.path.isfile(old):
            print('note: %s does not exist; reading %s instead' % (path, old))
            return old
        print('ERROR: no descriptor at prefix %r: neither %s nor %s exists' % (prefix, path, old))
        return None
    return path


def load_asset(prefix, named=None):
    path = descriptor_path(prefix, named)
    if path is None:
        return None
    meta = json.load(open(path))
    # Assets written before the rename say 'ntc-dds-1'; the format is the same and they are read as such.
    if meta['format'] == 'ntc-dds-1':
        print("note: format 'ntc-dds-1' is the older name of nntc-dds-1; reading it as such")
    else:
        assert meta['format'] == 'nntc-dds-1', meta.get('format')
    tex = meta['textures']
    assert len(tex) == 2 and tex[0]['level'] == 0 and tex[1]['level'] == 1
    directory = os.path.dirname(path)
    z0, n0 = read_texture(directory, tex[0])
    z1, n1 = read_texture(directory, tex[1])
    assert n0 == n1 == tex[0]['mip_count'] == tex[1]['mip_count']
    return meta, z0, z1


def decode_level(meta, z0, z1, level):
    """out = W phi + b, saturated and rounded to 8 bits; the same arithmetic the pixel shader performs."""
    dec = meta['decoder']
    assert dec['type'] == 'bilinear' and dec['output'] == 'identity'
    assert not dec.get('hidden')
    terms = dec['terms'].split()
    c0, c1 = dec['C0'], dec['C1']
    layer = dec['layers'][0]
    w = np.asarray(layer['weights'], dtype=np.float64).reshape(layer['rows'], layer['cols'])
    b = np.asarray(layer['bias'], dtype=np.float64)
    s = z0[level]
    h, width = s.shape[:2]
    c = bilinear(z1[level], width, h)
    phi = []
    if 'a' in terms:
        phi += [c[:, :, j] for j in range(c1)]
    if 'b' in terms:
        phi += [s[:, :, i] for i in range(c0)]
    if 'sc' in terms:
        phi += [s[:, :, i] * c[:, :, j] for i in range(c0) for j in range(c1)]
    phi = np.stack(phi, axis=-1)
    assert phi.shape[-1] == dec['nin'], (phi.shape[-1], dec['nin'])
    out = phi @ w.T + b
    # floor(x + 0.5), not np.rint: rint rounds a half to even and the encoder's lroundf rounds it away from zero, and
    # the two differ on exactly the values a flat region lands on.
    return np.floor(np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_rgb8(path):
    """One reference image as the ENCODER sees it: three 8-bit channels.

    The encoder reads every input through stb_image with req_comp 3, so what it is actually fitted against is what stb
    produces, and this has to be the same bytes or the psnr it reports and the psnr computed here are of two different
    sources. Two of stb's reductions matter:

      16 bits per channel  stb keeps THE TOP BYTE (stbi__convert_16_to_8 is `orig[i] >> 8`), so a 16-bit greyscale
                           height map is fitted as its high byte and nothing else. Pillow's own .convert("RGB") of an
                           I;16 image clips instead -- every sample of a map whose values run past 255 comes back 255 --
                           so reading it that way compares the decode against a white image and reports nonsense.
      one channel          replicated into three, which is what .convert("RGB") does for mode L anyway.

    Anything already 8-bit goes through Pillow untouched.
    """
    im = Image.open(path)
    a = np.asarray(im)
    if a.dtype != np.uint8:
        # 16 bits per sample, whatever Pillow calls the mode (I;16, I;16B, or a 32-bit I holding 16-bit samples).
        a = (a.astype(np.uint32) >> 8).astype(np.uint8)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, axis=2)
        return a[:, :, :3]
    if im.mode != 'RGB':
        return np.asarray(im.convert('RGB'))
    return a


def png_name(ref_prefix, textures, texture, level):
    if textures > 1:
        return '%s_t%d_M%d.png' % (ref_prefix, texture, level)
    return '%s_M%d.png' % (ref_prefix, level)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    prefix = prefix_from_argument(sys.argv[1])
    mode = sys.argv[2]
    # A .json argument that is a file on disk IS the descriptor to read; only a name this tool derived itself can be
    # the older spelling the note is about.
    named = sys.argv[1] if sys.argv[1].endswith('.json') and os.path.isfile(sys.argv[1]) else None
    loaded = load_asset(prefix, named)
    if loaded is None:
        return 1
    meta, z0, z1 = loaded
    textures = meta['decode'].get('textures_out', 1)
    src_w, src_h = meta['source']['width'], meta['source']['height']

    if mode == '--ref':
        if len(sys.argv) != 4:
            print('--ref needs one reference prefix')
            return 2
        ref_prefix = sys.argv[3]
        worst = 0
        for level in range(len(z0)):
            img = decode_level(meta, z0, z1, level)
            for t in range(textures):
                ref = load_rgb8(png_name(ref_prefix, textures, t, level))
                cw = min(ref.shape[1], img.shape[1])
                ch = min(ref.shape[0], img.shape[0])
                a = img[:ch, :cw, 3 * t:3 * t + 3].astype(int)
                d = np.abs(a - ref[:ch, :cw].astype(int))
                worst = max(worst, int(d.max()))
                print('M%d texture %d: %dx%d from the .dds + .json, max |diff| %d, values off by more than 1: %d of %d'
                      % (level, t, cw, ch, d.max(), int((d > 1).sum()), d.size))
        print('worst max |diff| over every level: %d' % worst)
        return 0 if worst <= 1 else 1

    if mode == '--grid':
        # The statement the encoder's fit rests on: the value the asset PUBLISHES for index k is the value a sampler
        # RETURNS for it, dequantised. For an uncompressed plane that is lo + rep(k) / 255 * (hi - lo), rep being the
        # index replicated over the byte; for a block-compressed one it is the standard BC palette, which at 1-3 bits
        # decodes index k to 255 k / (2 ^ bits - 1) exactly. Nothing here is read from the image: it is the grid alone.
        #
        # A level-0 entry of kind `range` is the --l0 bc8 mode: level 0 is then a continuous 8-bit plane on a
        # per-channel lo/hi, exactly as level 1 is, and there is no palette to publish or to check. What is checked
        # there is the same thing level 1's entry is checked for -- that the byte rule and the index rule are the same
        # number, which at 8 bits they are exactly -- and it is reported per level rather than pooled, so a mode that
        # published no palette cannot clear the palette statement by having nothing to say.
        worst, palettes = 0.0, 0
        gap = {}
        for tex in meta['textures']:
            dq = tex['dequantise']
            used, bits = tex['channels_used'], tex['bits_per_channel']
            compressed = tex['dxgi_format_id'] in (80, 83)
            lo = np.asarray(dq['lo'], dtype=np.float64)
            hi = np.asarray(dq['hi'], dtype=np.float64)
            # A `range` entry publishes no palette, so the only thing there is to hold it to is its own shape: the
            # index range is the one 8-bit continuous plane's 255, every used channel is at 8 bits, and lo / hi carry
            # at least one entry per used channel with hi above lo -- without which `lo + byte / 255 * (hi - lo)`
            # would not be a grid at all. This used to be checked for nothing on a bc8 asset.
            if dq['kind'] == 'range':
                assert dq['levels'] == (1 << bits[0]) - 1, (dq['levels'], bits[0])
                assert all(b == bits[0] for b in bits[:used]), bits[:used]
                assert len(dq['lo']) >= used and len(dq['hi']) >= used, (len(dq['lo']), len(dq['hi']), used)
                assert all(hi[c] > lo[c] for c in range(used)), (dq['lo'], dq['hi'])
                if tex['level'] == 0:
                    assert dq['levels'] == 255 and bits[0] == 8, (dq['levels'], bits)
                print('level %d: kind range, %d levels, bits %s, lo/hi per channel %s'
                      % (tex['level'], dq['levels'], ','.join(str(b) for b in bits[:used]),
                         ' '.join('[%g, %g]' % (lo[c], hi[c]) for c in range(used))))
            for c in range(used):
                b = bits[c]
                levels = (1 << b) - 1
                for k in range(levels + 1):
                    if compressed:
                        byte = 255.0 * k / levels          # the BC palette entry, in float as the hardware returns it
                    else:
                        byte = 0                           # the index replicated over the byte
                        sh = 8 - b
                        while sh > -b:
                            byte |= (k << sh) if sh >= 0 else (k >> -sh)
                            sh -= b
                        byte &= 0xFF
                    sampled = lo[c] + byte / 255.0 * (hi[c] - lo[c])
                    if dq['kind'] == 'palette':
                        worst = max(worst, abs(sampled - dq['palette'][c][k]))
                        palettes += 1
                    else:
                        # An entry of kind `range` publishes no palette, only lo, hi and the index's range, and its
                        # note says the rule is on the BYTE. The gap below is what reading it as
                        # lo + k / levels * (hi - lo) instead would cost -- zero at 4 and 8 bits, half a byte of the
                        # span at 5, 6 and 7 -- and is reported rather than asserted, because the byte rule is the one
                        # the encoder fits and the one published.
                        naive = lo[c] + k / dq['levels'] * (hi[c] - lo[c])
                        gap[tex['level']] = max(gap.get(tex['level'], 0.0), abs(sampled - naive))
            print('level %d: %d channel(s) at %s bits, %s, dequantise kind %s'
                  % (tex['level'], used, ','.join(str(b) for b in bits[:used]),
                     'block-compressed' if compressed else 'uncompressed', dq['kind']))
        if palettes:
            print('worst |published palette - sampled value| over every index of every channel: %.3e' % worst)
        else:
            print('no entry publishes a palette: nothing to check against one')
        for level in sorted(gap):
            print('level %d (kind range): the byte rule against the index rule, over the same set: %.3e of the '
                  'channel span (0 at 4 and 8 bits)' % (level, gap[level]))
        return 0 if worst < 1e-6 else 1

    if mode == '--psnr-levels':
        # The encoder's own two quality lines, from the files alone: the base against the source over the original
        # extent, and every chain level against the run's own source chain at that level's own resolution. A level's
        # number is the worst of its textures, which is what the encoder's `mip psnr` line quotes.
        sources = sys.argv[3:]
        if sources and len(sources) != textures:
            print('--psnr-levels takes no sources or %d of them, %d given' % (textures, len(sources)))
            return 2
        worst_of_level = []
        for level in range(len(z0)):
            img = decode_level(meta, z0, z1, level)
            worst = None
            for t in range(textures):
                if level == 0 and sources:
                    path = sources[t]
                    crop_w, crop_h = src_w, src_h
                else:
                    path = png_name(prefix + '_src', textures, t, level)
                    crop_w, crop_h = None, None
                # The mode measures against the run's OWN source chain, which is written only when the run was asked
                # for PNGs. Under --png 0 there is no chain on disk and the mode has nothing to compare against, which
                # used to arrive as a traceback out of the PNG reader; it is one line naming the first file that is
                # not there, and what would have written it.
                if not os.path.isfile(path):
                    print('ERROR: %s does not exist; --psnr-levels measures against the run\'s own source chain, which '
                          'is written by --png 1' % path)
                    return 1
                ref = load_rgb8(path).astype(np.float64)
                cw = min(ref.shape[1], img.shape[1]) if crop_w is None else min(crop_w, ref.shape[1], img.shape[1])
                ch = min(ref.shape[0], img.shape[0]) if crop_h is None else min(crop_h, ref.shape[0], img.shape[0])
                a = img[:ch, :cw, 3 * t:3 * t + 3].astype(np.float64)
                mse = float(np.mean((a - ref[:ch, :cw]) ** 2))
                psnr = 10.0 * np.log10(255.0 * 255.0 / mse) if mse > 0 else 100.0
                worst = psnr if worst is None else min(worst, psnr)
                print('M%d texture %d: %dx%d against %s, psnr %.2f dB' % (level, t, cw, ch, path, psnr))
            worst_of_level.append(worst)
        print('psnr M0 %.2f dB' % worst_of_level[0])
        if len(worst_of_level) > 1:
            print('mip psnr' + ''.join(' M%d %.2f' % (i, worst_of_level[i])
                                       for i in range(1, len(worst_of_level))) + '  dB')
        return 0

    if mode == '--psnr':
        sources = sys.argv[3:]
        if len(sources) != textures:
            print('--psnr needs %d source image(s), %d given' % (textures, len(sources)))
            return 2
        img = decode_level(meta, z0, z1, 0)
        for t, path in enumerate(sources):
            ref = load_rgb8(path).astype(np.float64)
            a = img[:src_h, :src_w, 3 * t:3 * t + 3].astype(np.float64)
            mse = float(np.mean((a - ref[:src_h, :src_w]) ** 2))
            psnr = 10.0 * np.log10(255.0 * 255.0 / mse) if mse > 0 else 100.0
            print('texture %d: %dx%d against %s, psnr %.2f dB' % (t, src_w, src_h, path, psnr))
        return 0

    print('unknown mode %s' % mode)
    return 2


if __name__ == '__main__':
    sys.exit(main())
