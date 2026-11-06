"""Condition C: export a few labelled TRAINING images for the LLM to look at.

    python -m cxr.examples --n_per_class 8 --out astra/examples

Picks n pneumonia and n normal images from the training split, one per child,
with MATCHED original aspect ratios (so the black padding bars - the geometry
shortcut found in A1 - look the same in both classes), and saves each one
exactly as the network receives it: 224 x 224 grayscale, padded, 0 = black.
Also saves a labelled 4 x 4 contact sheet. Validation and test are never used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .dataset import load_index


def pick(df, n_per_class, seed, band):
    tr = df[(df.split == "train") & (df.cache_idx >= 0)].copy()
    tr["aspect"] = tr.width / tr.height
    tr = tr.drop_duplicates("patient_key")                      # one image per child
    lo, hi = band
    inband = tr[(tr.aspect >= lo) & (tr.aspect <= hi)]
    rng = np.random.default_rng(seed)
    out = []
    for label in (1, 0):
        part = inband[inband.label == label]
        if label == 1 and {"bacteria", "virus"} <= set(part.subtype):     # half bacterial, half viral
            k = n_per_class // 2
            for st, kk in zip(("bacteria", "virus"), (k, n_per_class - k)):
                sub = part[part.subtype == st]
                out.append(sub.iloc[rng.choice(len(sub), kk, replace=False)])
        else:
            out.append(part.iloc[rng.choice(len(part), n_per_class, replace=False)])
    sel = __import__("pandas").concat(out)
    return sel.iloc[rng.permutation(len(sel))].reset_index(drop=True)   # interleave the classes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="astra/examples")
    ap.add_argument("--n_per_class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--band", type=float, nargs=2, default=[1.25, 1.45], help="original width/height range used for both classes")
    a = ap.parse_args(argv)
    df, images = load_index(a.data, 224)
    sel = pick(df, a.n_per_class, a.seed, tuple(a.band))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta, tiles = [], []
    for i, r in sel.iterrows():
        name = "PNEUMONIA" if r.label == 1 else "NORMAL"
        fn = f"example_{i + 1:02d}_{name}.png"
        arr = np.asarray(images[int(r.cache_idx)])
        Image.fromarray(arr).save(out / fn)
        tiles.append((arr, f"{i + 1:02d} {name}"))
        meta.append(dict(file=fn, label=name, aspect=round(float(r.aspect), 3)))
    # contact sheet, 4 columns, each tile 224 px + caption
    cols = 4; rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("L", (cols * 232, rows * 252), 255)
    d = ImageDraw.Draw(sheet)
    for k, (arr, cap) in enumerate(tiles):
        x, y = (k % cols) * 232 + 4, (k // cols) * 252 + 4
        sheet.paste(Image.fromarray(arr), (x, y))
        d.text((x + 2, y + 228), cap, fill=0)
    sheet.save(out / "examples_contact_sheet.png")
    (out / "examples.json").write_text(json.dumps(dict(seed=a.seed, aspect_band=a.band, split="train", images=meta), indent=2))
    asp = {lab: [m["aspect"] for m in meta if m["label"] == lab] for lab in ("PNEUMONIA", "NORMAL")}
    print(f"[examples] {len(meta)} training images -> {out}  (contact sheet: examples_contact_sheet.png)")
    for lab, v in asp.items():
        print(f"  {lab:9s} n={len(v)}  original aspect ratio mean {np.mean(v):.2f} (range {min(v):.2f}-{max(v):.2f})")


if __name__ == "__main__":
    main()
