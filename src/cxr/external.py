"""External test on a different hospital and population: RSNA Pneumonia Detection Challenge.

    # 1. download only what is needed from the Kaggle competition archive and cache it
    python -m cxr.external prepare --zip rsna-pneumonia-detection-challenge.zip --n_per_class 500
    # 2. score every model on it (trained checkpoints from runs/, LLM-written weights from astra/)
    python -m cxr.external evaluate

RSNA 2018 (Kaggle) labels each frontal chest X-ray of an adult as 'Normal', 'Lung Opacity'
(pneumonia) or 'No Lung Opacity / Not Normal'. We use 'Lung Opacity' as PNEUMONIA and 'Normal'
as NORMAL, drop the ambiguous third class, and take a fixed random, balanced sample (seed 0).
Images go through exactly the same preprocessing as the Kermany images (grayscale, pad to
square, 224 px). Nothing here is used for training or for any choice of weights.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .dataset import CachedCXR, build_transform
from .metrics import compute_metrics
from .models import build_model, count_params

LABELS = "stage_2_train_labels.csv"
CLASSES = "stage_2_detailed_class_info.csv"
IMGDIR = "stage_2_train_images/"


def to_square_224(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    a = (a - a.min()) / max(float(a.max() - a.min()), 1e-6) * 255.0
    im = Image.fromarray(a.astype(np.uint8), "L")
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(canvas.resize((224, 224), Image.LANCZOS), dtype=np.uint8)


def cmd_prepare(a):
    import pydicom
    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    find = lambda suffix: next(n for n in names if n.endswith(suffix))
    lab = pd.read_csv(z.open(find(LABELS))).groupby("patientId", as_index=False)["Target"].max()
    cls = pd.read_csv(z.open(find(CLASSES))).drop_duplicates("patientId")
    df = lab.merge(cls, on="patientId")
    df = df[df["class"].isin(["Normal", "Lung Opacity"])].copy()
    df["label"] = (df["class"] == "Lung Opacity").astype(int)
    rng = np.random.default_rng(0)
    take = []
    for label in (1, 0):
        part = df[df.label == label]
        take.append(part.iloc[rng.choice(len(part), min(a.n_per_class, len(part)), replace=False)])
    sel = pd.concat(take).sort_values("patientId").reset_index(drop=True)
    first = next(n for n in names if IMGDIR in n and n.endswith(".dcm"))       # archives may omit folder entries
    imgdir = first[: first.index(IMGDIR) + len(IMGDIR)]
    images, rows = [], []
    for i, r in sel.iterrows():
        ds = pydicom.dcmread(io.BytesIO(z.read(imgdir + r.patientId + ".dcm")))
        arr = ds.pixel_array
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        images.append(to_square_224(arr))
        rows.append(dict(patient_key=r.patientId, label=int(r.label), class_name=r["class"], subtype=r["class"], split="test", cache_idx=i,
                         age=getattr(ds, "PatientAge", ""), view=getattr(ds, "ViewPosition", "")))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(sel)} images")
    out = Path(a.out); (out / "cache").mkdir(parents=True, exist_ok=True)
    np.save(out / "cache" / "images_224.npy", np.stack(images))
    idx = pd.DataFrame(rows); idx.to_csv(out / "index.csv", index=False)
    ages = pd.to_numeric(idx.age, errors="coerce")
    print(f"[external] RSNA: {len(idx)} images ({idx.label.sum()} Lung Opacity, {(idx.label == 0).sum()} Normal), "
          f"age median {ages.median():.0f} (min {ages.min():.0f}), views {idx.view.value_counts().to_dict()} -> {out}")


def _scores(model, images, rows, device, batch=64):
    ds = CachedCXR(images, rows, build_transform(224, False, None))
    p = []
    model = model.to(device).eval()
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch, shuffle=False):
            p.append(torch.sigmoid(model(x.to(device))).squeeze(1).float().cpu().numpy())
    return np.concatenate(p), ds.labels.astype(int)


def cmd_evaluate(a):
    from .written import build as build_written, load_weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ext = pd.read_csv(Path(a.ext) / "index.csv")
    ext_img = np.load(Path(a.ext) / "cache" / "images_224.npy", mmap_mode="r")
    R = Path(a.results)
    entries = []
    for label, f in [("Astra zero-shot (A)", "astra/weights_zeroshot.py"), ("Astra practice (B)", "astra/weights_practice_final.py"),
                     ("Astra examples (C)", "astra/weights_examples.py")]:
        if Path(f).exists():
            entries.append((label, "LLM-written, no training", lambda f=f: build_written(load_weights(f)), f))
    for name, label in [("writtencnn", "Same CNN, trained"), ("smallcnn", "Small CNN, trained"), ("resnet50", "ResNet-50, trained")]:
        for seed in a.seeds:
            ck = Path(a.runs) / f"{name}__abl-none__frac-1__seed-{seed}" / "best.pt"
            if ck.exists():
                def mk(name=name, ck=ck):
                    m = build_model(name, 224); m.load_state_dict(torch.load(ck, map_location="cpu")); return m
                entries.append((f"{label} (seed {seed})", "gradient descent, A1 protocol", mk, str(ck)))
            else:
                print(f"  (no checkpoint {ck}; run scripts/run_external.sh to train it)")
    out = []
    for label, how, make, src in entries:
        m = make()
        p, y = _scores(m, ext_img, ext, device)
        met = compute_metrics(y, p)
        out.append(dict(model=label, how=how, source=src, n_params=count_params(m), external=met))
        print(f"[external] {label:32s} AUROC {met['auroc']:.3f} [{met['ci95']['auroc'][0]:.3f}, {met['ci95']['auroc'][1]:.3f}]  "
              f"acc {met['accuracy']:.3f}  sens {met['recall']:.3f}  spec {met['specificity']:.3f}")
    # internal test AUROC next to it, from the existing result files
    internal = {"Astra zero-shot (A)": "written__astra-zeroshot.json", "Astra practice (B)": "written__astra-practice.json",
                "Astra examples (C)": "written__astra-examples.json"}
    for r in out:
        f = internal.get(r["model"])
        if f is None:
            name = {"Same CNN": "writtencnn", "Small CNN": "smallcnn", "ResNet-50": "resnet50"}[r["model"].split(",")[0]]
            seed = r["model"].split("seed ")[1].rstrip(")")
            f = f"{name}__abl-none__frac-1__seed-{seed}.json"
        for d in (R, R / "retrain"):
            if (d / f).exists():
                r["internal_test_auroc"] = json.loads((d / f).read_text())["test"]["auroc"]; break
    (R / "external_rsna.json").write_text(json.dumps(dict(dataset="RSNA Pneumonia Detection Challenge 2018 (stage 2 train), "
        "Lung Opacity vs Normal, balanced random sample, seed 0", n=int(len(ext)), results=out), indent=2))
    lines = ["| Model | Weights | Internal test AUROC (Kermany, n=618) | External AUROC (RSNA adults) [95% CI] | Accuracy | Sensitivity | Specificity |",
             "|---|---|---|---|---|---|---|"]
    for r in out:
        e = r["external"]; it = r.get("internal_test_auroc")
        lines.append(f"| {r['model']} | {r['how']} | {it:.3f} |" if it is not None else f"| {r['model']} | {r['how']} | – |")
        lines[-1] += f" {e['auroc']:.3f} [{e['ci95']['auroc'][0]:.3f}, {e['ci95']['auroc'][1]:.3f}] | {e['accuracy']:.3f} | {e['recall']:.3f} | {e['specificity']:.3f} |"
    md = f"# External test: RSNA Pneumonia (adults, different hospitals), n = {len(ext)}\n\n" + "\n".join(lines) + "\n"
    (R / "external_rsna.md").write_text(md); print("\n" + md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--zip", required=True); p.add_argument("--out", default="data_ext/rsna")
    p.add_argument("--n_per_class", type=int, default=500)
    p = sub.add_parser("evaluate"); p.add_argument("--ext", default="data_ext/rsna"); p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results"); p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    a = ap.parse_args(argv)
    {"prepare": cmd_prepare, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    main()
