"""Condition G: the LLM writes the final layer (2,048 weights + bias) on top of a FROZEN ImageNet ResNet-50.

The backbone was trained on everyday photographs only (torchvision IMAGENET1K_V2) and never on X-rays.
Its 2,048 pooled features are fixed; the LLM writes the linear head by reasoning, with the same practice
loop and per-unit report as condition F (here one row per feature instead of per channel).

    python -m cxr.head features                            # once: frozen features for train / practice (val) / test
    python -m cxr.head practice                            # PASS 0: zero head, feature report on slice 1 (not counted)
    python -m cxr.head practice --head astra/head/head_v1.py   # one counted practice pass (next 256 practice images)
    python -m cxr.head evaluate --head astra/head/head_final.py  # once: Astra's head vs rivals, test + size-matched subset

A head file is Python that defines make_head() -> (weight, bias): 2,048 numbers and one float, from constants
and formulas only (numpy and math; no files, no randomness), checked statically and run in an isolated process.
Rivals in `evaluate` (all on the same frozen features):
  * matched random search: best of N random heads (PyTorch default init), each scored on one practice slice
    in Astra's slice order, N = the passes Astra used; 100 repetitions;
  * selection rule: the mechanical version of "read the report, keep the features that separate": top-k
    features by |separation - 0.5| on the practice images Astra saw, weight sign / (k * sd), k in {8, 32, 128, 512}
    chosen on those same images; no gradients;
  * linear probe: logistic regression trained on the 4,424 training images (gradient-based), C chosen on the
    practice set; the strong reference for "a head on these features".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from .metrics import compute_metrics
from .written import static_check

FEAT = Path("astra/head/features.npz")
SLICE = 256


def cmd_features(a):
    import torch
    from torch.utils.data import DataLoader
    from .dataset import CachedCXR, build_transform, load_index
    from .models import resnet50
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = resnet50(pretrained=not a.no_pretrained)
    m.fc = torch.nn.Identity()
    m = m.to(dev).eval()
    df, images = load_index(a.data, 224)
    parts = {"train": df[df.split == "train"],
             "val": df[df.split == "val"].sample(frac=1.0, random_state=0),     # the practice order used in conditions B-F
             "test": df[df.split == "test"]}
    out = {}
    for name, rows in parts.items():
        rows = rows.reset_index(drop=True)
        ds = CachedCXR(images, rows, build_transform(224, False, None))
        f = []
        with torch.no_grad():
            for x, _ in DataLoader(ds, 64, shuffle=False):
                f.append(m(x.to(dev)).float().cpu().numpy())
        out[f"{name}_x"] = np.concatenate(f).astype(np.float32)
        out[f"{name}_y"] = rows.label.values.astype(int)
        out[f"{name}_cidx"] = rows.cache_idx.values.astype(int)
    FEAT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEAT, **out)
    print(f"[head] frozen ImageNet ResNet-50 features: " + ", ".join(f"{k} {out[k + '_x'].shape}" for k in parts) + f" -> {FEAT}")


def load_head(path):
    src = Path(path).read_text()
    bad = static_check(src, required="make_head")
    if bad:
        raise ValueError(f"{path}: rejected - " + "; ".join(bad))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "h.py").write_text(src)
        runner = ("import sys, numpy as np; sys.path.insert(0, '.'); import h; w, b = h.make_head(); "
                  "np.savez('out.npz', w=np.asarray(w, dtype=np.float64).reshape(-1), b=np.array([float(b)]))")
        r = subprocess.run([sys.executable, "-I", "-c", runner], cwd=d, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"{path}: make_head() failed:\n{r.stderr[-2000:]}")
        o = np.load(Path(d) / "out.npz")
        w, b = o["w"], float(o["b"][0])
    if w.shape != (2048,) or not np.all(np.isfinite(w)) or not np.isfinite(b):
        raise ValueError(f"{path}: make_head must return 2,048 finite weights and a finite bias")
    return w, b


def prob(x, w, b):
    return 1 / (1 + np.exp(-np.clip(x @ w + b, -50, 50)))


def metrics(y, x, w, b, n_boot=0):
    """Threshold metrics from P at 0.5; AUROC (and its CI) from the RANK of the logit, which orders images
    exactly like P but never saturates (hand-written heads on ImageNet features can push P to exactly 1.0)."""
    from scipy.stats import rankdata
    z = x @ w + b
    m = compute_metrics(y, prob(x, w, b), n_boot=0)
    r = compute_metrics(y, rankdata(z) / len(z), n_boot=n_boot)
    m["auroc"] = r["auroc"]
    if n_boot:
        m["ci95"] = r["ci95"]
    return m


def separation(x, y):
    """AUROC of every column on its own (0.5 = none; > 0.5 higher on pneumonia). Rank-based, vectorised."""
    from scipy.stats import rankdata
    r = rankdata(x, axis=0)
    n1, n0 = int(y.sum()), int((y == 0).sum())
    sep = (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    sep[np.ptp(x, axis=0) < 1e-12] = 0.5
    return sep


def slice_idx(k, n):
    return np.array([(k * SLICE + i) % n for i in range(SLICE)])


def report(x, y, w, b):
    sep = separation(x, y)
    mn, mp = x[y == 0].mean(0), x[y == 1].mean(0)
    silent = (x <= 0).mean(0)
    order = np.argsort(-sep, kind="stable")
    lines = ["", "FEATURE REPORT (same practice images; label 1 = PNEUMONIA), sorted by separation, highest first",
             "separation = AUROC of that feature alone (0.5 = no separation; above 0.5 = higher on pneumonia, below 0.5 = higher on normal). "
             "silent = share of images where the feature is 0. contrib_* = weight x mean feature value, per class.",
             f"summary: separation > 0.70: {int((sep > 0.7).sum())} features; < 0.30: {int((sep < 0.3).sum())}; "
             f"between 0.45 and 0.55: {int(((sep > 0.45) & (sep < 0.55)).sum())}. Head now uses {int((np.abs(w) > 0).sum())} of 2,048 features, bias {b:+.4f}.",
             "feature,mean_normal,mean_pneumonia,separation,silent,contrib_normal,contrib_pneumonia"]
    for i in order:
        lines.append(f"f{i},{mn[i]:.4f},{mp[i]:.4f},{sep[i]:.3f},{silent[i]:.2f},{w[i] * mn[i]:+.4f},{w[i] * mp[i]:+.4f}")
    return "\n".join(lines)


def cmd_practice(a):
    z = np.load(FEAT)
    xv, yv, cv = z["val_x"], z["val_y"], z["val_cidx"]
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    used = sum(1 for _ in log.open()) if log.exists() else 0
    if a.head is None:
        w, b, sl = np.zeros(2048), 0.0, slice_idx(0, len(yv))
        label = "PASS 0 (zero head: every image gets P = 0.5; feature report only, not counted)"
    else:
        if used >= a.budget:
            sys.exit(f"budget exhausted: {used}/{a.budget}")
        w, b = load_head(a.head)
        sl = slice_idx(used, len(yv))
        label = f"PRACTICE PASS {used + 1}/{a.budget}"
    x, y = xv[sl], yv[sl]
    p = prob(x, w, b)
    m = metrics(y, x, w, b)
    print(f"{label}  (n={len(y)} practice images; label 1 = PNEUMONIA)")
    print(f"AUROC {m['auroc']:.4f} | accuracy {m['accuracy']:.4f} | sensitivity {m['recall']:.4f} | specificity {m['specificity']:.4f}")
    print(f"mean P(pneumonia): pneumonia images {p[y == 1].mean():.4f}, normal images {p[y == 0].mean():.4f}")
    if a.head is not None:
        with log.open("a") as f:
            f.write(json.dumps(dict(pass_no=used + 1, budget=a.budget, head=str(a.head), n=int(len(y)), auroc=m["auroc"],
                                    accuracy=m["accuracy"], recall=m["recall"], specificity=m["specificity"],
                                    features_used=int((np.abs(w) > 0).sum()), bias=b, time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print("practice_image_id,label,p_pneumonia")
    for c, yi, pi in zip(cv[sl], y, p):
        print(f"P{int(c):05d},{yi},{pi:.4f}")
    print(report(x, y, w, b))


# ------------------------------------------------------------------ rivals and scoring
def random_search(xv, yv, xt, yt, passes, reps):
    bound = 1 / np.sqrt(2048)
    runs = []
    for r in range(reps):
        best = None
        for k in range(passes):
            rng = np.random.default_rng(200000 + r * passes + k)
            w, b = rng.uniform(-bound, bound, 2048), float(rng.uniform(-bound, bound))
            sl = slice_idx(k, len(yv))
            au = metrics(yv[sl], xv[sl], w, b)["auroc"]
            if best is None or au > best[0]:
                best = (au, w, b)
        runs.append(dict(practice_auroc=best[0], test_auroc=metrics(yt, xt, best[1], best[2])["auroc"],
                         w=best[1], b=best[2]))
    return runs


def selection_rule(xv, yv, passes):
    seen = np.unique(np.concatenate([slice_idx(k, len(yv)) for k in range(max(passes, 1))]))
    x, y = xv[seen], yv[seen]
    sep = separation(x, y); sd = x.std(0) + 1e-6
    best = None
    for k in (8, 32, 128, 512):
        top = np.argsort(-np.abs(sep - 0.5))[:k]
        w = np.zeros(2048); w[top] = np.sign(sep[top] - 0.5) / (k * sd[top])
        b = -float((x @ w).mean())
        au = metrics(y, x, w, b)["auroc"]
        if best is None or au > best[0]:
            best = (au, k, w, b)
    return dict(k=best[1], practice_auroc=best[0], w=best[2], b=best[3], n_seen=int(len(seen)))


def linear_probe(z):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(z["train_x"])
    xtr, xv = sc.transform(z["train_x"]), sc.transform(z["val_x"])
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        t0 = time.time()
        lr = LogisticRegression(C=C, max_iter=3000).fit(xtr, z["train_y"])
        au = compute_metrics(z["val_y"], lr.predict_proba(xv)[:, 1], n_boot=0)["auroc"]
        if best is None or au > best[0]:
            best = (au, C, lr, time.time() - t0)
    lr = best[2]
    w = lr.coef_[0] / sc.scale_; b = float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())
    return dict(C=best[1], val_auroc=best[0], fit_seconds=best[3], w=w, b=b)


def cmd_evaluate(a):
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from .robust import size_matched_pairs
    z = np.load(FEAT)
    xv, yv, xt, yt = z["val_x"], z["val_y"], z["test_x"], z["test_y"]
    df = pd.read_csv(Path(a.data) / "index.csv")
    t = df[df.split == "test"].reset_index(drop=True)
    assert (t.cache_idx.values == z["test_cidx"]).all(), "index.csv does not match features.npz"
    sub = np.array([k for pr in size_matched_pairs(t, 0.05) for k in pr])
    passes = a.passes if a.passes else (sum(1 for _ in Path(a.log).open()) if Path(a.log).exists() else 0)
    rows = []

    def add(name, how, w, b, extra=None):
        m = metrics(yt, xt, w, b, n_boot=1000)
        z = xt @ w + b
        rows.append(dict(model=name, how=how, auroc=m["auroc"], auroc_ci=m["ci95"]["auroc"], accuracy=m["accuracy"],
                         sensitivity=m["recall"], specificity=m["specificity"], features_used=int((np.abs(w) > 1e-12).sum()),
                         auroc_size_matched=float(roc_auc_score(yt[sub], z[sub])), **(extra or {})))

    w, b = load_head(a.head)
    add(f"Head written by {a.llm}", f"no training; {passes} practice passes + feature report", w, b)
    sel = selection_rule(xv, yv, passes)
    add("Selection rule (mechanical rival)", f"top-{sel['k']} features by separation on the {sel['n_seen']} practice images seen; no gradients",
        sel["w"], sel["b"], dict(k=sel["k"]))
    lp = linear_probe(z)
    add("Linear probe (trained)", f"logistic regression on 4,424 training images, C={lp['C']} chosen on practice; fit {lp['fit_seconds']:.1f} s",
        lp["w"], lp["b"], dict(C=lp["C"]))
    runs = random_search(xv, yv, xt, yt, max(passes, 1), a.reps)
    au = np.array([r["test_auroc"] for r in runs])
    sm = np.array([roc_auc_score(yt[sub], xt[sub] @ r["w"] + r["b"]) for r in runs])
    search = dict(passes=max(passes, 1), reps=a.reps, median=float(np.median(au)), p05=float(np.percentile(au, 5)),
                  p95=float(np.percentile(au, 95)), max=float(au.max()), share_ge_astra=float((au >= rows[0]["auroc"]).mean()),
                  size_matched_median=float(np.median(sm)), practice_auroc_median=float(np.median([r["practice_auroc"] for r in runs])))
    out = dict(backbone="torchvision ResNet-50 IMAGENET1K_V2, frozen, never trained on X-rays", head_file=str(a.head), llm=a.llm,
               practice_passes=passes, n_test=int(len(yt)), n_size_matched=int(len(sub)), rows=rows, random_search=search)
    R = Path(a.results); R.mkdir(parents=True, exist_ok=True)
    tag = f"_{a.tag}" if a.tag else ""
    (R / f"head_resnet50{tag}.json").write_text(json.dumps(out, indent=2))
    lines = ["| Head on frozen ImageNet ResNet-50 | How | Test AUROC [95% CI] | Sens. | Spec. | Features used | Size-matched AUROC (140 img) |",
             "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['how']} | {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}] | {r['sensitivity']:.3f} | "
                     f"{r['specificity']:.3f} | {r['features_used']} | {r['auroc_size_matched']:.3f} |")
    lines.append(f"| Best of {search['passes']} random heads | matched random search, {a.reps} repetitions: median [5th-95th pct]; "
                 f"max {search['max']:.3f}; share >= Astra {search['share_ge_astra']:.0%} | {search['median']:.3f} [{search['p05']:.3f}, {search['p95']:.3f}] | | | 2048 | "
                 f"{search['size_matched_median']:.3f} (median) |")
    md = f"# Condition G: linear head on a frozen ImageNet ResNet-50 (official test folder, n = {len(yt)})\n\n" + "\n".join(lines) + "\n"
    (R / f"head_resnet50{tag}.md").write_text(md); print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features"); p.add_argument("--data", default="data"); p.add_argument("--no_pretrained", action="store_true")
    p = sub.add_parser("practice"); p.add_argument("--head", default=None); p.add_argument("--log", default="astra/practice_log_G1.jsonl")
    p.add_argument("--budget", type=int, default=20)
    p = sub.add_parser("evaluate"); p.add_argument("--head", required=True); p.add_argument("--llm", default="Astra")
    p.add_argument("--log", default="astra/practice_log_G1.jsonl"); p.add_argument("--passes", type=int, default=0)
    p.add_argument("--reps", type=int, default=100); p.add_argument("--data", default="data"); p.add_argument("--results", default="results")
    p.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    {"features": cmd_features, "practice": cmd_practice, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
