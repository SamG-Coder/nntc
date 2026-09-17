"""Two viewer frames, compared.

    python tools/frame_diff.py A.bmp B.bmp [--max-diff N] [--min-psnr X] [--quiet]

--quiet prints NOTHING on a pass -- the exit status is the whole answer, which is what a script wants; a failure still
prints its one ERROR line to stderr, quiet or not.

The two viewers in this tree draw the same asset through two graphics APIs, and the question "are they the same
picture" has to be a number rather than an opinion. This reads the 24-bit bottom-up .bmp both of them write and reports
the largest absolute difference and the mean absolute difference PER CHANNEL, and the PSNR between the two frames over
all three channels at once.

What it does NOT do is assert that the two are identical, because that is not a reasonable requirement across two APIs
and should never be written into a gate: the rasteriser's fill rule at a triangle edge, the bilinear weight precision
(both APIs specify a minimum number of fractional bits, not an exact value), the block-compression palette evaluation
(docs/FORMAT.md section 7 records that one NVIDIA GPU evaluates the BC4 interior with 6-bit weights rather than k/7)
and anisotropic tap placement are all free to differ. What IS asserted, by --max-diff and --min-psnr, is that the
difference is small enough to be one of those and not a bug: a wrong UV orientation, a missed y flip, a dropped LOD
bias, a feature vector in the wrong order or a dequantisation applied before the sample would every one of them be far
larger than that.

Exit status is 0 when the frames were read and every threshold given was met, and 1 otherwise, with one line beginning
ERROR saying which.

It needs numpy, which tools/dds_decode.py already needs, and nothing else.
"""

import math
import struct
import sys

import numpy as np


def read_bmp(path):
    """A 24-bit bottom-up .bmp as an (height, width, 3) uint8 array of r, g, b, top row first.

    Only the shape the two viewers write is accepted. A .bmp this cannot read is an error and not something to guess
    at: a comparison of two frames read differently is worse than no comparison at all.
    """
    try:
        data = open(path, 'rb').read()
    except OSError as exc:
        raise SystemExit('ERROR: cannot read %s (%s)' % (path, exc))
    if len(data) < 54 or data[:2] != b'BM':
        raise SystemExit('ERROR: %s is not a .bmp' % path)
    offset = struct.unpack_from('<I', data, 10)[0]
    width, height = struct.unpack_from('<ii', data, 18)
    planes, depth = struct.unpack_from('<HH', data, 26)
    compression = struct.unpack_from('<I', data, 30)[0]
    # A pixel array that starts inside the 54-byte header is not a file this can read: the rows would overlap the
    # header they are described by. It is checked before the offset is used to slice anything.
    if offset < 54:
        raise SystemExit('ERROR: %s says its pixels start at byte %d, which is inside its own 54-byte header'
                         % (path, offset))
    # The message names all three of the fields it tested, because a .bmp with two planes used to be reported as
    # whatever its depth happened to be, which named the wrong problem.
    if planes != 1 or depth != 24 or compression != 0:
        raise SystemExit('ERROR: %s has %d plane(s) and is %d-bit with compression %d; this reads uncompressed 24-bit '
                         'single-plane frames only' % (path, planes, depth, compression))
    if width <= 0 or height <= 0:
        raise SystemExit('ERROR: %s is %dx%d; this reads bottom-up frames of positive size only'
                         % (path, width, height))
    pitch = (width * 3 + 3) & ~3   # every row of a .bmp is padded to four bytes
    if len(data) < offset + pitch * height:
        raise SystemExit('ERROR: %s is truncated (%d bytes, %d expected)' % (path, len(data), offset + pitch * height))
    rows = np.frombuffer(data, dtype=np.uint8, count=pitch * height, offset=offset).reshape(height, pitch)
    pixels = rows[:, :width * 3].reshape(height, width, 3)
    return pixels[::-1, :, ::-1]   # bottom-up to top-first, and b, g, r to r, g, b


def compare(path_a, path_b):
    """The per-channel maxima and means and the PSNR, as a dict."""
    a = read_bmp(path_a)
    b = read_bmp(path_b)
    if a.shape != b.shape:
        raise SystemExit('ERROR: %s is %dx%d and %s is %dx%d; two frames of different sizes cannot be compared'
                         % (path_a, a.shape[1], a.shape[0], path_b, b.shape[1], b.shape[0]))
    # int16 first: the difference of two uint8 arrays wraps, which would report 255 where the answer is 1.
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    return {
        'width': int(a.shape[1]), 'height': int(a.shape[0]),
        'max': [int(diff[:, :, c].max()) for c in range(3)],
        'mean': [float(diff[:, :, c].mean()) for c in range(3)],
        'mse': mse,
        'psnr': float('inf') if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse),
    }


def main():
    args = sys.argv[1:]
    paths = []
    max_diff = None
    min_psnr = None
    quiet = False
    i = 0
    while i < len(args):
        a = args[i]
        if a in ('--help', '-h'):
            print(__doc__)
            return 0
        if a == '--quiet':
            quiet = True
        elif a in ('--max-diff', '--min-psnr'):
            if i + 1 >= len(args):
                raise SystemExit('ERROR: %s needs a number' % a)
            try:
                value = float(args[i + 1])
            except ValueError:
                raise SystemExit("ERROR: %s needs a number, not '%s'" % (a, args[i + 1]))
            if a == '--max-diff':
                max_diff = value
            else:
                min_psnr = value
            i += 1
        elif a.startswith('-'):
            raise SystemExit("ERROR: unknown option '%s'" % a)
        else:
            paths.append(a)
        i += 1
    if len(paths) != 2:
        raise SystemExit('ERROR: two .bmp frames are needed, not %d' % len(paths))

    r = compare(paths[0], paths[1])
    if not quiet:
        print('%s vs %s: %dx%d' % (paths[0], paths[1], r['width'], r['height']))
        for c, name in enumerate('rgb'):
            print('  %s: max |diff| %d, mean |diff| %.4f' % (name, r['max'][c], r['mean'][c]))
        print('  PSNR %s' % ('inf dB (the frames are identical)' if r['mse'] == 0.0 else '%.2f dB' % r['psnr']))

    failed = []
    if max_diff is not None and max(r['max']) > max_diff:
        failed.append('the largest difference is %d, over the %g allowed' % (max(r['max']), max_diff))
    if min_psnr is not None and r['psnr'] < min_psnr:
        failed.append('the PSNR is %.2f dB, under the %g asked for' % (r['psnr'], min_psnr))
    if failed:
        print('ERROR: %s and %s differ more than allowed: %s' % (paths[0], paths[1], '; '.join(failed)),
              file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
