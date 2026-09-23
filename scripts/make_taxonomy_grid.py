"""scripts/make_taxonomy_grid.py — grasp-taxonomy × hand montage from render_rollout clips.

    python scripts/render_rollout.py ... --taxonomy-out clips/          # once per hand
    python scripts/make_taxonomy_grid.py --clips clips --hands tesollo shadow allegro \\
        --taxonomies medium_diameter power_sphere tripod --out docs/media/taxonomy_grid.gif

Rows are grasp taxonomies (the classes the policy is asked to imitate via the
finger-pose target), columns are hands; every cell is that hand's rollout for a
world that sampled that taxonomy (green frame = success). ``--taxonomies``
defaults to the taxonomies present for EVERY listed hand.
"""
from __future__ import annotations

import argparse
import json
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


def per_hand_grid(args):
    from PIL import Image
    import imageio.v2 as imageio
    meta = json.load(open(os.path.join(args.clips, f"{args.per_hand}.json")))
    taxes = args.taxonomies or sorted(meta["taxonomies"])
    if args.require_success:
        taxes = [t for t in taxes if meta["taxonomies"][t]["success"]]
    clips = [np.load(os.path.join(args.clips, f"{args.per_hand}__{t}.npy")) for t in taxes]
    T = min(c.shape[0] for c in clips); w = args.tile_width
    h0, w0 = clips[0].shape[1:3]; hgt = int(round(h0 * w / w0))
    rows = (len(clips) + args.cols - 1) // args.cols
    frames = []
    for t in range(0, T, args.every):
        canvas = np.full((rows * hgt, args.cols * w, 3), 255, dtype=np.uint8)
        for i, c in enumerate(clips):
            r, q = divmod(i, args.cols)
            canvas[r * hgt:(r + 1) * hgt, q * w:(q + 1) * w] = np.asarray(Image.fromarray(c[t]).resize((w, hgt), Image.LANCZOS))
        frames.append(canvas)
    base = os.path.splitext(args.out)[0]
    os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
    imageio.mimsave(base + ".mp4", _even(frames), fps=args.fps, macro_block_size=1); print("wrote", base + ".mp4", f"({len(taxes)} taxonomies)")
    if args.out.endswith(".gif"):
        vf = f"fps={args.fps},split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".mp4", "-vf", vf, base + ".gif"], check=True)
        print("wrote", base + ".gif", f"({os.path.getsize(base + '.gif')/1e6:.1f} MB)")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", required=True, help="directory written by render_rollout.py --taxonomy-out")
    p.add_argument("--hands", nargs="+", required=True)
    p.add_argument("--taxonomies", nargs="*", default=None)
    p.add_argument("--tile-width", type=int, default=160)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--out", required=True, help=".gif or .mp4 (mp4 always written)")
    p.add_argument("--max-rows", type=int, default=0, help="split into several files of at most this many rows (0 = one file)")
    p.add_argument("--transpose", action="store_true", help="rows = hands, columns = taxonomies (README layout)")
    p.add_argument("--require-success", action="store_true",
                   help="only use taxonomies whose clip succeeded for EVERY listed hand (grid) / drop failed clips (per-hand)")
    p.add_argument("--per-hand", default=None, metavar="HAND",
                   help="instead of the taxonomy × hand grid, tile ALL taxonomy clips of this one hand "
                        "(labels are burned into the clips) in a --cols wide grid")
    p.add_argument("--cols", type=int, default=6)
    args = p.parse_args()
    if args.per_hand:
        return per_hand_grid(args)
    from PIL import Image, ImageDraw, ImageFont
    meta = {h: json.load(open(os.path.join(args.clips, f"{h}.json"))) for h in args.hands}
    common = None
    for h in args.hands:
        s = set(meta[h]["taxonomies"]); common = s if common is None else common & s
    if args.require_success:
        common = {t for t in common if all(meta[h]["taxonomies"][t]["success"] for h in args.hands)}
    taxes = args.taxonomies or sorted(common)
    if args.require_success:
        bad = [t for t in taxes if t not in common]
        assert not bad, f"not successful for every hand: {bad}; successful everywhere: {sorted(common)}"
    missing = [(h, t) for h in args.hands for t in taxes if t not in meta[h]["taxonomies"]]
    assert not missing, f"missing clips: {missing}"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    clips = {(h, t): np.load(os.path.join(args.clips, f"{h}__{t}.npy")) for h in args.hands for t in taxes}
    T = min(a.shape[0] for a in clips.values())
    w = args.tile_width; h0 = next(iter(clips.values())).shape[1]; w0 = next(iter(clips.values())).shape[2]
    hgt = int(round(h0 * w / w0)); LEFT = 150; TOP = 28
    def cell(hd, tax, t):
        im = Image.fromarray(clips[(hd, tax)][t]).resize((w, hgt), Image.LANCZOS)
        ok = meta[hd]["taxonomies"][tax]["success"]
        ImageDraw.Draw(im).rectangle([0, 0, w - 1, hgt - 1], outline=(40, 200, 80) if ok else (220, 60, 60), width=3)
        return im
    def render_frame(t, rows):
        if args.transpose:      # rows = hands, cols = taxonomies
            canvas = Image.new("RGB", (LEFT + w * len(taxes), TOP + hgt * len(rows)), (255, 255, 255))
            dr = ImageDraw.Draw(canvas)
            for j, tax in enumerate(taxes):
                dr.text((LEFT + j * w + 4, 6), tax.replace("_", " "), fill=(0, 0, 0), font=font)
            for i, hd in enumerate(rows):
                y = TOP + i * hgt
                dr.text((6, y + hgt // 2 - 8), meta[hd]["label"], fill=(0, 0, 0), font=font)
                for j, tax in enumerate(taxes):
                    canvas.paste(cell(hd, tax, t), (LEFT + j * w, y))
            return np.asarray(canvas)
        canvas = Image.new("RGB", (LEFT + w * len(args.hands), TOP + hgt * len(rows)), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for j, hd in enumerate(args.hands):
            dr.text((LEFT + j * w + 4, 6), meta[hd]["label"], fill=(0, 0, 0), font=font)
        for i, tax in enumerate(rows):
            y = TOP + i * hgt
            dr.text((6, y + hgt // 2 - 8), tax.replace("_", " "), fill=(0, 0, 0), font=font)
            for j, hd in enumerate(args.hands):
                canvas.paste(cell(hd, tax, t), (LEFT + j * w, y))
        return np.asarray(canvas)
    row_items = args.hands if args.transpose else taxes
    groups = [row_items] if not args.max_rows else [row_items[i:i + args.max_rows] for i in range(0, len(row_items), args.max_rows)]
    import imageio.v2 as imageio
    for gi, rows in enumerate(groups):
        base = os.path.splitext(args.out)[0] + (f"_{gi+1}" if len(groups) > 1 else "")
        frames = [render_frame(t, rows) for t in range(0, T, args.every)]
        os.makedirs(os.path.dirname(os.path.abspath(base)) or ".", exist_ok=True)
        imageio.mimsave(base + ".mp4", _even(frames), fps=args.fps, macro_block_size=1); print("wrote", base + ".mp4")
        if args.out.endswith(".gif"):
            vf = f"fps={args.fps},split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", base + ".mp4", "-vf", vf, base + ".gif"], check=True)
            print("wrote", base + ".gif", f"({os.path.getsize(base + '.gif')/1e6:.1f} MB)")
    print("taxonomies:", taxes)


if __name__ == "__main__":
    main()
