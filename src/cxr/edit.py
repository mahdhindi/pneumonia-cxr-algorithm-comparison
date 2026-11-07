"""Condition E: an LLM EDITS the final layer of our trained ResNet-50, with no gradients.

After Takehi et al. (2026), "edit" mode: the network already has the ability (test AUROC 0.97)
but something hides it (specificity 0.64: a third of healthy children flagged). The LLM reads
the final layer's 2,049 numbers and practice results, and edits them by reasoning.

    python -m cxr.edit features                          # once: backbone features of practice (val) + test images
    python -m cxr.edit export                            # writes astra/edit/resnet50_fc.txt for the LLM to read
    python -m cxr.edit practice --edit astra/edit/edit_v1.py      # one practice pass (val images only)
    python -m cxr.edit practice                          # pass 0: the unedited model (free, not counted)

Practice pool (default --pool test_half): the official test folder is split by patient (seed 0) into a
practice half and a scoring half. Our validation set cannot be the practice pool: it comes from the
training collection, where the model is already near-perfect (AUROC 0.9995), so the collection-shift
problem the edit is meant to fix is invisible there. The threshold rival uses the same practice half.
    python -m cxr.edit evaluate --edit astra/edit/edit_final.py   # once: edited vs unedited vs threshold rival, on test

An edit file is Python that defines edit(weight, bias) -> (new_weight, new_bias), where weight is
a numpy array of 2,048 numbers and bias a float; numpy and math only, no files, no randomness.
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
import torch
from torch.utils.data import DataLoader

from .dataset import CachedCXR, build_transform, load_index
from .metrics import compute_metrics
from .models import resnet50
from .written import static_check

CKPT = "runs/resnet50__abl-none__frac-1__seed-42/best.pt"
FEAT = Path("astra/edit/features.npz")


def load_model(ckpt):
    m = resnet50(pretrained=False)                      # architecture only; every weight comes from our checkpoint
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.eval()


def cmd_features(a):
    """Penultimate (2,048-d, after global pooling) features for the practice set and the test set."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = load_model(a.ckpt).to(dev)
    fc = m.fc; m.fc = torch.nn.Identity()
    df, images = load_index(a.data, 224)
    out = {}
    for split in ("val", "test"):
        rows = df[df.split == split].sample(frac=1.0, random_state=0).reset_index(drop=True) if split == "val" else df[df.split == split]
        ds = CachedCXR(images, rows, build_transform(224, False, None))
        f = []
        with torch.no_grad():
            for x, _ in DataLoader(ds, 64, shuffle=False):
                f.append(m(x.to(dev)).float().cpu().numpy())
        out[f"{split}_x"] = np.concatenate(f); out[f"{split}_y"] = ds.labels.astype(int)
        out[f"{split}_id"] = np.array([f"P{int(c):05d}" for c in rows.cache_idx]) if split == "val" else np.array([], dtype=str)
    out["w"] = fc.weight.detach().cpu().numpy().reshape(-1); out["b"] = np.array([float(fc.bias.detach().cpu())])
    FEAT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(FEAT, **out)
    # sanity: features + fc reproduce the model's own test AUROC
    p = 1 / (1 + np.exp(-(out["test_x"] @ out["w"] + out["b"][0])))
    print(f"[edit] features: practice {out['val_x'].shape}, test {out['test_x'].shape}; unedited test AUROC {compute_metrics(out['test_y'], p, n_boot=0)['auroc']:.4f}")


def cmd_export(a):
    z = np.load(FEAT)
    w, b = z["w"], float(z["b"][0])
    lines = [f"# Final layer of our trained ResNet-50: logit = sum_i weight[i] * feature[i] + bias;  P(pneumonia) = sigmoid(logit)",
             f"# features are the 2,048 outputs of the network's global average pool (each >= 0, after ReLU)",
             f"# bias = {b:.6f}",
             f"# weight summary: mean {w.mean():+.6f}, sd {w.std():.6f}, min {w.min():+.6f} (index {w.argmin()}), max {w.max():+.6f} (index {w.argmax()}), "
             f"{int((w > 0).sum())} positive, {int((w < 0).sum())} negative",
             "index,weight"] + [f"{i},{v:.6f}" for i, v in enumerate(w)]
    Path("astra/edit/resnet50_fc.txt").write_text("\n".join(lines) + "\n")
    print(f"[edit] wrote astra/edit/resnet50_fc.txt (2,048 weights + bias)")


def apply_edit(path, w, b):
    if path is None:
        return w.copy(), b
    src = Path(path).read_text()
    bad = static_check(src, required="edit")
    if bad:
        raise ValueError(f"{path}: rejected - " + "; ".join(bad))
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "e.py").write_text(src)
        np.savez(Path(d) / "in.npz", w=w, b=np.array([b]))
        runner = ("import sys, numpy as np; sys.path.insert(0, '.'); import e; z = np.load('in.npz'); "
                  "nw, nb = e.edit(z['w'].copy(), float(z['b'][0])); np.savez('out.npz', w=np.asarray(nw, dtype=np.float64).reshape(-1), b=np.array([float(nb)]))")
        r = subprocess.run([sys.executable, "-I", "-c", runner], cwd=d, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"{path}: edit() failed:\n{r.stderr[-2000:]}")
        o = np.load(Path(d) / "out.npz")
    nw, nb = o["w"], float(o["b"][0])
    if nw.shape != w.shape or not np.all(np.isfinite(nw)) or not np.isfinite(nb):
        raise ValueError(f"{path}: edit must return 2,048 finite weights and a finite bias")
    return nw, nb


def pools(z, mode, data="data"):
    """Practice and scoring sets. 'test_half' (default): the official test folder split by PATIENT (seed 0)
    into a practice half and a scoring half, so practice shows the same collection shift the model meets at
    scoring time. 'val': our validation set as practice, whole test folder as scoring (no shift visible)."""
    if mode == "val":
        return (z["val_x"], z["val_y"], z["val_id"]), (z["test_x"], z["test_y"])
    import pandas as pd
    df = pd.read_csv(Path(data) / "index.csv")
    t = df[df.split == "test"].reset_index(drop=True)
    assert len(t) == len(z["test_y"]) and (t.label.values == z["test_y"]).all(), "index.csv does not match features.npz"
    pats = np.array(sorted(t.patient_key.unique()))
    rng = np.random.default_rng(0)
    prac_p = set(rng.permutation(pats)[: len(pats) // 2])
    is_prac = t.patient_key.isin(prac_p).values
    pi = np.flatnonzero(is_prac); pi = pi[np.random.default_rng(1).permutation(len(pi))]   # mix the classes
    si = np.flatnonzero(~is_prac)
    ids = np.array([f"T{int(c):05d}" for c in t.cache_idx.values[pi]])
    return (z["test_x"][pi], z["test_y"][pi], ids), (z["test_x"][si], z["test_y"][si])


def prob(x, w, b):
    return 1 / (1 + np.exp(-np.clip(x @ w + b, -50, 50)))


def cmd_practice(a):
    z = np.load(FEAT)
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    used = sum(1 for _ in log.open()) if log.exists() else 0
    (px, py, pid), _ = pools(z, a.pool)
    if a.edit is None:
        sl = np.arange(len(py)); label = f"PASS 0 (unedited model, all {len(sl)} practice images, not counted)"
    else:
        if used >= a.budget:
            sys.exit(f"budget exhausted: {used}/{a.budget}")
        if a.pool == "val":
            n = len(py); start = (used * 256) % n
            sl = np.array([(start + i) % n for i in range(256)])
        else:
            sl = np.arange(len(py))                          # the whole practice half every pass
        label = f"PRACTICE PASS {used + 1}/{a.budget}"
    w, b = apply_edit(a.edit, z["w"], float(z["b"][0]))
    p, y = prob(px[sl], w, b), py[sl]
    m = compute_metrics(y, p, n_boot=0)
    changed = int((np.abs(w - z["w"]) > 1e-9).sum())
    print(f"{label}  (n={len(y)} practice images; label 1 = PNEUMONIA)")
    print(f"edit: {changed} of 2,048 weights changed; bias {float(z['b'][0]):+.4f} -> {b:+.4f}")
    print(f"AUROC {m['auroc']:.4f} | accuracy {m['accuracy']:.4f} | sensitivity {m['recall']:.4f} | specificity {m['specificity']:.4f}")
    print(f"mean P(pneumonia): pneumonia images {p[y == 1].mean():.4f}, normal images {p[y == 0].mean():.4f}")
    if a.edit is not None:
        with log.open("a") as f:
            f.write(json.dumps(dict(pass_no=used + 1, edit=str(a.edit), n=int(len(y)), auroc=m["auroc"], accuracy=m["accuracy"],
                                    recall=m["recall"], specificity=m["specificity"], weights_changed=changed, bias=b, pool=a.pool,
                                    time=time.strftime("%Y-%m-%d %H:%M:%S"))) + "\n")
    print("practice_image_id,label,p_pneumonia")
    for i, pi, yi in zip(pid[sl], p, y):
        print(f"{i},{yi},{pi:.4f}")


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def cmd_evaluate(a):
    z = np.load(FEAT)
    w0, b0 = z["w"], float(z["b"][0])
    w1, b1 = apply_edit(a.edit, w0, b0)
    (xv, yv, _), (xt, yt) = pools(z, a.pool)
    rows = []

    def add(name, how, p_test, thr=0.5):
        m = compute_metrics(yt, p_test)
        pred = (p_test >= thr).astype(int)
        acc = float((pred == yt).mean()); sens = float(pred[yt == 1].mean()); spec = float(1 - pred[yt == 0].mean())
        rows.append(dict(model=name, how=how, threshold=thr, auroc=m["auroc"], auroc_ci=m["ci95"]["auroc"], accuracy=acc,
                         sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2))

    p0 = prob(xt, w0, b0)
    add("ResNet-50, trained (unedited)", "gradient descent, A1 protocol, seed 42", p0)
    t = youden_threshold(yv, prob(xv, w0, b0))
    add("ResNet-50 + threshold moved on practice set", f"rival: one number changed (threshold {t:.3f}, Youden on the {len(yv)} practice images)", p0, t)
    p1 = prob(xt, w1, b1)
    changed = int((np.abs(w1 - w0) > 1e-9).sum())
    add(f"ResNet-50 edited by {a.llm}", f"no gradients; {changed} of 2,048 weights changed, bias {b0:+.3f} -> {b1:+.3f}", p1)
    out = dict(edit_file=str(a.edit), llm=a.llm, pool=a.pool, n_practice=int(len(yv)), n_scoring=int(len(yt)), checkpoint=CKPT, practice_passes=sum(1 for _ in Path(a.log).open()) if Path(a.log).exists() else 0,
               weights_changed=changed, rows=rows)
    R = Path(a.results); R.mkdir(exist_ok=True, parents=True)
    (R / "edit_resnet50.json").write_text(json.dumps(out, indent=2))
    lines = ["| Model | How | Test AUROC [95% CI] | Accuracy | Sensitivity | Specificity | Balanced acc. |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['how']} | {r['auroc']:.3f} [{r['auroc_ci'][0]:.3f}, {r['auroc_ci'][1]:.3f}] | {r['accuracy']:.3f} | "
                     f"{r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['balanced_accuracy']:.3f} |")
    where = "official test folder, n = 618" if a.pool == "val" else f"scoring half of the official test folder (patients not in the practice half), n = {len(yt)}"
    md = f"# Condition E: LLM edit of the trained ResNet-50's final layer ({where})\n\n" + "\n".join(lines) + "\n"
    (R / "edit_resnet50.md").write_text(md); print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("features"); p.add_argument("--data", default="data"); p.add_argument("--ckpt", default=CKPT)
    sub.add_parser("export")
    p = sub.add_parser("practice"); p.add_argument("--pool", choices=["test_half", "val"], default="test_half"); p.add_argument("--edit", default=None); p.add_argument("--log", default="astra/edit/practice_log_E.jsonl")
    p.add_argument("--budget", type=int, default=20)
    p = sub.add_parser("evaluate"); p.add_argument("--pool", choices=["test_half", "val"], default="test_half"); p.add_argument("--edit", required=True); p.add_argument("--llm", default="Astra")
    p.add_argument("--log", default="astra/edit/practice_log_E.jsonl"); p.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    {"features": cmd_features, "export": cmd_export, "practice": cmd_practice, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
