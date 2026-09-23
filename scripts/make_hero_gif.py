"""scripts/make_hero_gif.py — tile one rollout per hand into a labelled montage GIF/MP4.

    python scripts/make_hero_gif.py --frames robotis_sh5=a.npy tesollo=b.npy ... --cols 3 --out docs/media/hero.gif

Each ``<label>=<npy>`` is a ``(T, H, W, 3)`` uint8 array as written by
``render_rollout.py --save-frames``. Frames are resized to ``--tile-width``,
labelled in the top-left corner and laid out in a grid; the shortest clip
sets the length.
"""
from __future__ import annotations

import argparse
import os
import subprocess

import numpy as np


def _even(frames):
    """Pad frames to even width/height (libx264 yuv420p requirement)."""
    import numpy as _np
    h, w = frames[0].shape[:2]
    if h % 2 == 0 and w % 2 == 0:
        return frames
    return [_np.pad(f, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge") for f in frames]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--frames", nargs="+", required=True, metavar="LABEL=NPY")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--tile-width", type=int, default=256)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--every", type=int, default=1, help="keep every k-th frame")
    p.add_argument("--label", action=argparse.BooleanOptionalAction, default=True,
                   help="draw the LABEL of each clip in its corner (off when the clips are already labelled)")
    p.add_argument("--out", required=True, help=".gif or .mp4 (a .mp4 is always written next to a .gif)")
    args = p.parse_args()
    from PIL import Image, ImageDraw, ImageFont
    clips = []
    for spec in args.frames:
        label, path = spec.split("=", 1)
        arr = np.load(path)
        clips.append((label, arr))
    T = min(a.shape[0] for _, a in clips)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    tiles_t = []
    for t in range(0, T, args.every):
        tiles = []
        for label, arr in clips:
            im = Image.fromarray(arr[t])
            w = args.tile_width; h = int(round(im.height * w / im.width))
            im = im.resize((w, h), Image.LANCZOS)
            if args.label:
                dr = ImageDraw.Draw(im)
                tw = dr.textlength(label, font=font)
                dr.rectangle([4, 4, 12 + tw, 30], fill=(0, 0, 0))
                dr.text((8, 6), label, fill=(255, 255, 255), font=font)
            tiles.append(np.asarray(im))
        n = len(tiles); rows = (n + args.cols - 1) // args.cols
        h, w, c = tiles[0].shape
        canvas = np.full((rows * h, args.cols * w, c), 255, dtype=np.uint8)
        for i, tile in enumerate(tiles):
            r, q = divmod(i, args.cols)
            canvas[r * h:(r + 1) * h, q * w:(q + 1) * w] = tile
        tiles_t.append(canvas)
    import imageio.v2 as imageio
    mp4 = os.path.splitext(args.out)[0] + ".mp4"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    imageio.mimsave(mp4, _even(tiles_t), fps=args.fps, macro_block_size=1)
    print("wrote", mp4)
    if args.out.endswith(".gif"):
        vf = f"fps={args.fps},split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", mp4, "-vf", vf, args.out], check=True)
        print("wrote", args.out, f"({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
