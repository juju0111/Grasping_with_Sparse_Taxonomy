"""scripts/make_gallery.py — per-clip videos + browsable gallery pages from render_rollout clips.

    python scripts/make_gallery.py --clips clips --out docs/media/clips --pages docs/gallery

Writes for every ``<hand>__<taxonomy>.npy`` clip an MP4 (and a small GIF), then

* ``docs/gallery/index.html`` — pick a hand and a grasp taxonomy from two drop-downs and
  the matching clip plays (static page: open locally or serve with GitHub Pages);
* ``docs/gallery/<hand>.md``  — one collapsible ``<details>`` block per taxonomy with the
  GIF inside (GitHub renders these, so the choice works inside plain Markdown);
* ``docs/GALLERY.md``         — the index linking both.
"""
from __future__ import annotations

import argparse
import glob
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


def write_video(frames, mp4, gif=None, fps=12, gif_width=256, gif_every=2, gif_max_frames=60):
    import imageio.v2 as imageio
    imageio.mimsave(mp4, _even(frames), fps=fps, macro_block_size=1)
    if gif:
        sub = frames[::gif_every][:gif_max_frames]
        tmp = mp4 + ".gif.mp4"
        imageio.mimsave(tmp, _even(sub), fps=max(1, fps // gif_every), macro_block_size=1)
        vf = f"scale={gif_width}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-vf", vf, gif], check=True)
        os.remove(tmp)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--clips", required=True)
    p.add_argument("--out", required=True, help="directory for the per-clip mp4/gif files (inside the repo)")
    p.add_argument("--pages", required=True, help="directory for the gallery pages (docs/gallery)")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--hands", nargs="*", default=None, help="hand order for the index (default: alphabetical)")
    p.add_argument("--gif-width", type=int, default=200)
    p.add_argument("--gif-every", type=int, default=3, help="keep every k-th frame in the GIF")
    p.add_argument("--gif-max-frames", type=int, default=40)
    p.add_argument("--include-failures", action="store_true", help="also publish clips whose world failed (default: successes only)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True); os.makedirs(args.pages, exist_ok=True)
    metas = {}
    for f in sorted(glob.glob(os.path.join(args.clips, "*.json"))):
        d = json.load(open(f)); metas[d["hand"]] = d
    hands = [h for h in (args.hands or sorted(metas)) if h in metas]
    rel = os.path.relpath(args.out, args.pages)           # media path relative to the pages
    index = {}
    for h in hands:
        for tax, info in sorted(metas[h]["taxonomies"].items()):
            npy = os.path.join(args.clips, f"{h}__{tax}.npy")
            if not os.path.exists(npy) or (not info["success"] and not args.include_failures):
                continue
            mp4 = os.path.join(args.out, f"{h}__{tax}.mp4"); gif = os.path.join(args.out, f"{h}__{tax}.gif")
            if not os.path.exists(mp4) or not os.path.exists(gif):
                write_video(list(np.load(npy)), mp4, gif, fps=args.fps, gif_width=args.gif_width,
                            gif_every=args.gif_every, gif_max_frames=args.gif_max_frames)
            index.setdefault(h, {})[tax] = dict(info, mp4=f"{rel}/{h}__{tax}.mp4", gif=f"{rel}/{h}__{tax}.gif")
        print(h, len(index.get(h, {})), "clips")
    # ── per-hand markdown with <details> per taxonomy
    for h in hands:
        label = metas[h]["label"]; rows = index[h]
        n_all = len(metas[h]["taxonomies"])
        md = [f"# {label} — grasp taxonomies\n",
              f"Teacher policy `pretrained/{h}_grasping_teacher`, one successful world per taxonomy "
              f"({len(rows)} of the {n_all} taxonomies sampled in the rollout had a successful world; "
              "re-grasp successes count). Click a taxonomy to expand its clip; the MP4 link opens the full-rate video.\n",
              "[← gallery](../GALLERY.md) · [interactive picker](index.html)\n"]
        for tax, v in rows.items():
            mark = "✅" if v["success"] else "❌"
            md.append(f'<details><summary><b>{tax.replace("_", " ")}</b> {mark} — {v["object"]} '
                      f'(<a href="{v["mp4"]}">mp4</a>)</summary>\n\n<img src="{v["gif"]}" width="400"/>\n\n</details>\n')
        open(os.path.join(args.pages, f"{h}.md"), "w").write("\n".join(md))
    # ── interactive html
    data = {h: {t: {"mp4": v["mp4"], "success": v["success"], "object": v["object"]} for t, v in index[h].items()} for h in hands}
    labels = {h: metas[h]["label"] for h in hands}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>grit_share gallery</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;color:#222}} select{{font-size:16px;padding:4px 8px;margin-right:12px}}
video{{max-width:100%;border:4px solid #ccc;border-radius:6px}} .ok{{border-color:#2ac850}} .fail{{border-color:#dc3c3c}} #info{{margin:8px 0 12px;color:#555}}</style></head>
<body><h2>grit_share — pick a hand and a grasp taxonomy</h2>
<label>hand <select id="hand"></select></label><label>taxonomy <select id="tax"></select></label>
<div id="info"></div><video id="v" autoplay loop muted playsinline controls></video>
<p><a href="../GALLERY.md">back to the gallery index</a></p>
<script>
const DATA={json.dumps(data)}; const LABEL={json.dumps(labels)};
const hs=document.getElementById('hand'), ts=document.getElementById('tax'), v=document.getElementById('v'), info=document.getElementById('info');
for (const h of Object.keys(DATA)) hs.add(new Option(LABEL[h], h));
function fillTax(){{ const cur=ts.value; ts.innerHTML=''; for (const t of Object.keys(DATA[hs.value])) ts.add(new Option(t.replaceAll('_',' '), t)); if (cur && DATA[hs.value][cur]) ts.value=cur; }}
function show(){{ const d=DATA[hs.value][ts.value]; v.src=d.mp4; v.className=d.success?'ok':'fail'; info.textContent=`${{LABEL[hs.value]}} · ${{ts.value.replaceAll('_',' ')}} · object: ${{d.object}} · ${{d.success?'success':'failure'}}`; v.play(); }}
hs.onchange=()=>{{fillTax(); show();}}; ts.onchange=show; fillTax(); show();
</script></body></html>"""
    open(os.path.join(args.pages, "index.html"), "w").write(html)
    # ── gallery index
    lines = ["# Gallery\n",
             "Every clip: one parallel world, teacher policy, no scripted lift assist (the policy lifts by itself), "
             "label = hand · grasp taxonomy · object, green frame = success.\n",
             "**Interactive picker** (hand × taxonomy drop-downs, plays the MP4): [docs/gallery/index.html](gallery/index.html) — "
             "open it locally or enable GitHub Pages on the `docs/` folder; GitHub's Markdown view cannot run it.\n",
             "## Per hand\n", "| hand | rollout | taxonomies (collapsible, Markdown) | all taxonomies in one video |", "|---|---|---|---|"]
    for h in hands:
        lines.append(f"| {labels[h]} | [gif](media/{h}.gif) · [mp4](media/{h}.mp4) | [gallery/{h}.md](gallery/{h}.md) ({len(index[h])} clips) | [media/taxonomy_{h}.mp4](media/taxonomy_{h}.mp4) |")
    lines += ["", "## Same taxonomy, every hand\n",
              "Rows = hands, columns = three representative taxonomies (successful worlds only):\n",
              "<img src=\"media/taxonomy_grid_1.gif\" width=\"100%\"/>\n<img src=\"media/taxonomy_grid_2.gif\" width=\"100%\"/>\n<img src=\"media/taxonomy_grid_3.gif\" width=\"100%\"/>\n",
              "Every taxonomy common to the seven hands (rows) × hands: `media/taxonomy_grid_full_1.mp4`, `media/taxonomy_grid_full_2.mp4`.\n",
              ]
    open(os.path.join(os.path.dirname(args.pages.rstrip("/")), "GALLERY.md"), "w").write("\n".join(lines))
    print("gallery written:", args.pages)


if __name__ == "__main__":
    main()
