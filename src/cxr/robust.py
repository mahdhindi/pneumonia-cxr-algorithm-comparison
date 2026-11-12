"""Robustness check for the project: does a model rank pneumonia by the lungs, or by image size?

A1 found an acquisition shortcut in this dataset: the ORIGINAL image height alone separates the classes
(test AUROC 0.92; pneumonia X-rays are smaller files) and aspect ratio survives pad-to-square (0.70).
Condition F's LLM-written network scores "less fine detail" as pneumonia, which smaller originals could
produce after resizing. This check removes the size cue and re-scores every model:

  size-matched test subset: each NORMAL test image is paired (greedy, random order, seed 0, without
  replacement) with the PNEUMONIA test image whose original width and height are both within a factor
  exp(caliper) of its own (default caliper 0.05, i.e. about 5 %). On this subset original size and
  aspect ratio no longer separate the classes (AUROC ~0.5, reported as reference rows), so a model that
  ranks by size falls toward 0.5 while a model that ranks by the lungs keeps its AUROC.

Also reported on the full test set: Spearman correlation between each model's logit and log(original
height) WITHIN each class (a size-driven model scores small normals as pneumonia: negative rho).

    python -m cxr.robust run                  # trains any missing checkpoints, then writes results/robust_size.{json,md}
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .dataset import load_index
from .metrics import compute_metrics
from .models import build_model, count_params
from .written import build as build_written, load_weights, scores

ASTRA = [("Astra, practice (B)", "astra/weights_practice_final.py"),
         ("Astra, images + practice (D)", "astra/weights_examples_practice_final.py"),
         ("Astra, images + practice + channel report (F, run 1)", "astra/practice_F1/weights_diag_final.py"),
         ("Astra, images + practice + channel report (F, run 2)", "astra/practice_F2/weights_diag_final.py"),
         ("Astra, images + practice + channel report (F, run 3)", "astra/practice_F3/weights_diag_final.py"),
         ("Fable 5.2, images + practice + channel report (F, run 1)", "fable/practice_F1/weights_diag_final.py"),
         ("Fable 5.2, images + practice + channel report (F, run 2)", "fable/practice_F2/weights_diag_final.py"),
         ("Fable 5.2, images + practice + channel report (F, run 3)", "fable/practice_F3/weights_diag_final.py")]
TRAINED = [("Same CNN, trained (seed 42)", "writtencnn", 42), ("Same CNN, trained (seed 43)", "writtencnn", 43),
           ("ResNet-50, trained (seed 42)", "resnet50", 42)]


def size_matched_pairs(t: pd.DataFrame, caliper: float, seed: int = 0) -> list[tuple[int, int]]:
    lw, lh, y = np.log(t.width.values.astype(float)), np.log(t.height.values.astype(float)), t.label.values
    pos = np.flatnonzero(y == 1); used = np.zeros(len(t), bool); pairs = []
    for i in np.random.default_rng(seed).permutation(np.flatnonzero(y == 0)):
        d = np.maximum(np.abs(lw[pos] - lw[i]), np.abs(lh[pos] - lh[i]))
        for k in np.argsort(d, kind="stable"):
            if d[k] > caliper:
                break
            if not used[pos[k]]:
                used[pos[k]] = True; pairs.append((int(i), int(pos[k]))); break
    return pairs


def ensure_checkpoint(model: str, seed: int, runs: str, data: str) -> Path:
    ck = Path(runs) / f"{model}__abl-none__frac-1__seed-{seed}" / "best.pt"
    if not ck.exists():
        print(f"[robust] no checkpoint {ck}: training {model} seed {seed} (A1 protocol, results to results/retrain/)")
        subprocess.run([sys.executable, "-m", "cxr.train", "--model", model, "--seed", str(seed), "--epochs", "12",
                        "--img_size", "224", "--data", data, "--runs", runs, "--results", "results/retrain"], check=True)
    return ck


def trained_scores(model: str, ck: Path, images, rows):
    m = build_model(model, 224, pretrained=False)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from torch.utils.data import DataLoader
    from .dataset import CachedCXR, build_transform
    ds = CachedCXR(images, rows, build_transform(224, False, None))
    p = []
    m = m.to(dev).eval()
    with torch.no_grad():
        for x, _ in DataLoader(ds, 64, shuffle=False):
            p.append(torch.sigmoid(m(x.to(dev))).squeeze(1).float().cpu().numpy())
    return np.concatenate(p), count_params(m)


def cmd_run(a):
    df, images = load_index(a.data, 224)
    t = df[df.split == "test"].reset_index(drop=True)
    y = t.label.values.astype(int)
    pairs = size_matched_pairs(t, a.caliper)
    sub = np.array([k for pr in pairs for k in pr])
    pairs_wide = size_matched_pairs(t, a.caliper_wide)
    sub_wide = np.array([k for pr in pairs_wide for k in pr])
    print(f"[robust] test n={len(t)}; size-matched subset: {len(pairs)} pairs ({len(sub)} images) at caliper {a.caliper}, "
          f"{len(pairs_wide)} pairs at caliper {a.caliper_wide}")

    models = []
    for name, f in ASTRA:
        if Path(f).exists():
            m = build_written(load_weights(f))
            p, yy = scores(m, images, t)
            assert (yy == y).all()
            models.append((name, "LLM-written, no training", p, count_params(m)))
    for name, model, seed in TRAINED:
        ck = ensure_checkpoint(model, seed, a.runs, a.data)
        p, n = trained_scores(model, ck, images, t)
        models.append((name, "gradient descent, A1 protocol", p, n))

    def row(name, how, s, n_params=None, is_prob=True):
        full = compute_metrics(y, s, n_boot=0)["auroc"] if is_prob else float(roc_auc_score(y, s))
        ms = compute_metrics(y[sub], s[sub], n_boot=a.n_boot) if is_prob else None
        matched = ms["auroc"] if is_prob else float(roc_auc_score(y[sub], s[sub]))
        ci = ms["ci95"]["auroc"] if is_prob else None
        wide = float(roc_auc_score(y[sub_wide], s[sub_wide]))
        logit = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))) if is_prob else s
        lh = np.log(t.height.values.astype(float))
        rho = ({c: float(spearmanr(logit[y == k], lh[y == k]).correlation) for c, k in (("normal", 0), ("pneumonia", 1))}
               if is_prob else None)
        return dict(model=name, how=how, n_params=n_params, auroc_full=full, auroc_matched=matched, auroc_matched_ci=ci,
                    auroc_matched_wide=wide, drop=full - matched, spearman_logit_vs_log_height=rho)

    rows = [row("Original image height alone (smaller = pneumonia)", "acquisition shortcut, reference", -t.height.values.astype(float), is_prob=False),
            row("Aspect ratio alone (wider = pneumonia)", "acquisition shortcut, reference", (t.width / t.height).values.astype(float), is_prob=False)]
    rows += [row(name, how, p, n) for name, how, p, n in models]
    out = dict(caliper=a.caliper, caliper_wide=a.caliper_wide, n_test=int(len(t)), n_pairs=len(pairs), n_pairs_wide=len(pairs_wide),
               matched_images=[str(t.path.values[k]) for k in sub], rows=rows)
    R = Path(a.results); R.mkdir(parents=True, exist_ok=True)
    (R / "robust_size.json").write_text(json.dumps(out, indent=2))
    lines = [f"# Robustness check: size-matched test subset ({len(pairs)} normal/pneumonia pairs, original width and height within "
             f"~{100 * (np.exp(a.caliper) - 1):.0f} %; {len(sub)} of {len(t)} test images)", "",
             "| Model | How | Full test AUROC (n=618) | Size-matched AUROC [95% CI] | Drop | Matched, wider caliper "
             f"({len(pairs_wide)} pairs) | Spearman(logit, log height) normal / pneumonia |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = f" [{r['auroc_matched_ci'][0]:.3f}, {r['auroc_matched_ci'][1]:.3f}]" if r["auroc_matched_ci"] else ""
        rho = r["spearman_logit_vs_log_height"]
        rho_s = f"{rho['normal']:+.2f} / {rho['pneumonia']:+.2f}" if rho else "–"
        lines.append(f"| {r['model']} | {r['how']} | {r['auroc_full']:.3f} | {r['auroc_matched']:.3f}{ci} | {r['drop']:+.3f} | "
                     f"{r['auroc_matched_wide']:.3f} | {rho_s} |")
    md = "\n".join(lines) + "\n"
    (R / "robust_size.md").write_text(md); print("\n" + md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--data", default="data"); p.add_argument("--runs", default="runs"); p.add_argument("--results", default="results")
    p.add_argument("--caliper", type=float, default=0.05); p.add_argument("--caliper_wide", type=float, default=0.10)
    p.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args(argv)
    {"run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
