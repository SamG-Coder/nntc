#!/usr/bin/env python3
"""crop_set.py - build a "torture test" multi-texture material from unrelated images.

Takes 1 to 6 PNG files and writes one 8-bit PNG per input, all at the same
width x height:

  * an axis longer than the target is CROPPED (anchor: center or top-left,
    optionally shifted by --offset X,Y);
  * an axis shorter than the target is EXPANDED by replicating the source's
    last column / last row (edge clamp: no scaling, mirroring or tiling).
    The source stays at the left / top edge on an expanded axis, whatever
    the anchor.

Pixel formats: L stays L, LA stays LA, RGB stays RGB, RGBA stays RGBA.
Palette images become RGBA if they carry transparency, else RGB. 1-bit
images become L. 16-bit grayscale is reduced to 8-bit by keeping the high
byte (v >> 8). Anything else (CMYK, YCbCr, ...) is converted to RGB.
With --gray every output is written as L instead (luma, alpha dropped); with
--gray-first only the first one is, which makes a material with a grayscale first
texture.

Usage examples:
  python crop_set.py a.png b.png c.png --size 1024x1024 -o out --prefix t
      -> out/t1.png out/t2.png out/t3.png
  python crop_set.py a.png b.png --size 512x768 --anchor topleft -o out
  python crop_set.py a.png --size 1024x1024 --offset 32,-16 -o out \\
      --json t_source_material.json
"""

import argparse
import json
import os
import re

import numpy as np
from PIL import Image

MAX_INPUTS = 6


def parse_pair(text, sep, what, form):
    parts = text.lower().split(sep)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"bad {what} '{text}' (expected {form})")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad {what} '{text}' (expected integers)")


def parse_size(text):
    w, h = parse_pair(text, "x", "size", "WIDTHxHEIGHT")
    if w < 1 or h < 1:
        raise argparse.ArgumentTypeError(f"bad size '{text}' (must be positive)")
    return w, h


def parse_offset(text):
    return parse_pair(text, ",", "offset", "X,Y")


def load_8bit(path):
    """Open a PNG; return (uint8 array HxW or HxWxC, source mode, output mode)."""
    im = Image.open(path)
    im.load()
    src = im.mode
    if src in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im).astype(np.int64)
        return np.clip(a >> 8, 0, 255).astype(np.uint8), src, "L"
    if im.mode == "P":
        im = im.convert("RGBA" if im.has_transparency_data else "RGB")
    elif im.mode == "1":
        im = im.convert("L")
    elif im.mode not in ("L", "LA", "RGB", "RGBA"):
        im = im.convert("RGB")
    return np.asarray(im), src, im.mode


def fit_axis(n, target, anchor, offset):
    """Return (start, stop, pad, action) for one axis of length n."""
    if n > target:
        start = (n - target) // 2 if anchor == "center" else 0
        start = min(max(start + offset, 0), n - target)
        return start, start + target, 0, f"crop {n}->{target} @{start}"
    if n < target:
        return 0, n, target - n, f"expand {n}->{target}"
    return 0, n, 0, "as-is"


def main():
    ap = argparse.ArgumentParser(
        description="Crop / edge-extend 1-6 PNGs to one common size (8-bit PNG output).",
        epilog="An axis shorter than the target keeps the image at the left/top and "
               "repeats its last column/row, for either anchor. --offset shifts the "
               "crop window (clamped to the image) on cropped axes only.")
    ap.add_argument("inputs", nargs="*", help="1-6 input PNG files")
    ap.add_argument("--size", type=parse_size, required=True, help="target WIDTHxHEIGHT, e.g. 1024x1024")
    ap.add_argument("-o", "--outdir", required=True, help="output directory (created if missing)")
    ap.add_argument("--prefix", default="t", help="output name prefix: PREFIX1.png .. PREFIXN.png (default t)")
    ap.add_argument("--anchor", choices=("center", "topleft"), default="center",
                    help="crop anchor (default center)")
    ap.add_argument("--offset", type=parse_offset, default=(0, 0),
                    help="X,Y shift of the crop window from the anchor (default 0,0); write --offset=-8,0 for a negative X")
    ap.add_argument("--json", metavar="NAME",
                    help="also write a source-material JSON of the outputs into the output directory")
    ap.add_argument("--gray", action="store_true",
                    help="write every output as single-channel 8-bit grayscale (Pillow's L: ITU-R 601 luma, alpha dropped)")
    ap.add_argument("--gray-first", action="store_true",
                    help="as --gray, for the FIRST output only; the rest keep their formats")
    args = ap.parse_args()

    if not 1 <= len(args.inputs) <= MAX_INPUTS:
        ap.error(f"need 1 to {MAX_INPUTS} input PNGs, got {len(args.inputs)}")
    for path in args.inputs:
        if not os.path.isfile(path):
            ap.error(f"input not found: {path}")

    tw, th = args.size
    os.makedirs(args.outdir, exist_ok=True)
    outputs = []
    for i, path in enumerate(args.inputs, 1):
        a, src_mode, mode = load_8bit(path)
        if (args.gray or (args.gray_first and i == 1)) and mode != "L":
            a, mode = np.asarray(Image.fromarray(a).convert("L")), "L"
        h, w = a.shape[:2]
        x0, x1, px, xact = fit_axis(w, tw, args.anchor, args.offset[0])
        y0, y1, py, yact = fit_axis(h, th, args.anchor, args.offset[1])
        a = a[y0:y1, x0:x1]
        if px or py:
            a = np.pad(a, [(0, py), (0, px)] + [(0, 0)] * (a.ndim - 2), mode="edge")
        name = f"{args.prefix}{i}.png"
        out = os.path.join(args.outdir, name)
        Image.fromarray(np.ascontiguousarray(a)).save(out)
        outputs.append(name)
        fmt = mode if src_mode == mode else f"{src_mode}->{mode}"
        print(f"{os.path.basename(path)} {w}x{h} {fmt}: x {xact}, y {yact} -> {out}")

    if args.json:
        material = [{"file": n, "type": "", "filter": "mitchell", "edge": "clamp",
                     "weight": 1, "rgb_weights": [1, 1, 1]} for n in outputs]
        jpath = os.path.join(args.outdir, args.json)
        text = json.dumps(material, indent=2)
        text = re.sub(r"\[\s*1,\s*1,\s*1\s*\]", "[1, 1, 1]", text)  # one line, as in the examples
        with open(jpath, "w", newline="\n") as f:
            f.write(text + "\n")
        print(f"material JSON -> {jpath}")


if __name__ == "__main__":
    main()
