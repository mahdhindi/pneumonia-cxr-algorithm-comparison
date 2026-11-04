"""Assignment 2 (paper critique) - reality-check tables and figure.

    python -m cxr.report_a2 [--results results]

Compares, on the SAME pipeline, (a) the A1 patient-grouped protocol and
(b) the image-level random 80/10/10 split used by Singh et al. (2024),
for every model that was run under both, and puts the paper's reported
numbers in the same table.  Writes results/a2_reality_check.{md,csv} and
results/figures/a2_split_effect.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .report import LABEL, ORDER, agg, load_runs, md_table

# Singh et al., Sci. Rep. 14:2487 (2024), Tables 5 and 6, test n = 586 (random 10 % by image).
PAPER = {
    "vit_deit_b":  dict(accuracy=0.9761, recall=0.949, specificity=0.981, auroc=0.966, f1=0.952),
    "resnet50":    dict(accuracy=0.8237, f1=0.8281),     # backbone frozen, 200,706 trainable params
    "densenet121": dict(accuracy=0.9182, f1=0.9171),     # backbone frozen, 100,354 trainable params
}


def fmt(r, m):
    return f"{r[m]:.3f}" + (f" ± {r[m + '_std']:.3f}" if m + "_std" in r and r["n_seeds"] > 1 else "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    R = Path(a.results); F = R / "figures"; F.mkdir(exist_ok=True, parents=True)
    df = load_runs(R)
    df = df[(df.resize == "pad") & (df.ablation == "none") & (df.frac == 1.0)]
    order = [m for m in ORDER if m in set(df.model)]

    rows = []
    for m in order:
        for split in ("patient", "image"):
            for tag, part in df[(df.model == m) & (df.split == split)].groupby("tag"):
                g = agg(part, ["model"]).iloc[0]
                rows.append(dict(model=m, split=split, tag=tag, lr=part["lr"].iloc[0], n_seeds=int(g["n_seeds"]),
                                 n_train=int(g["n_train"]), auroc=g["auroc"], auroc_std=g["auroc_std"],
                                 accuracy=g["accuracy"], accuracy_std=g["accuracy_std"],
                                 recall=g["recall"], specificity=g["specificity"], balanced_acc=g["balanced_acc"],
                                 params_M=g["params_M"], train_min=g["train_time_s"] / 60))
    ours = pd.DataFrame(rows)
    ours.to_csv(R / "a2_reality_check.csv", index=False)

    def source(r):
        s = "A1 protocol: patient-grouped split, official test folder (n=618)" if r.split == "patient" \
            else "Singh et al. protocol: random 80/10/10 by image"
        return s + (f", {r.tag}" if r.tag else "")

    show = pd.DataFrame({
        "Model": ours["model"].map(LABEL), "Setup": [source(r) for _, r in ours.iterrows()],
        "LR": ours["lr"].map("{:g}".format), "Seeds": ours["n_seeds"],
        "AUROC": [fmt(r, "auroc") for _, r in ours.iterrows()],
        "Accuracy": [fmt(r, "accuracy") for _, r in ours.iterrows()],
        "Sensitivity": ours["recall"].map("{:.3f}".format), "Specificity": ours["specificity"].map("{:.3f}".format),
        "Balanced acc.": ours["balanced_acc"].map("{:.3f}".format),
        "Params (M)": ours["params_M"].map("{:.1f}".format), "Train (min)": ours["train_min"].map("{:.1f}".format),
    })
    paper_rows = pd.DataFrame([dict(
        Model=LABEL[m], Setup="Reported by Singh et al. (2024), test n=586, 1 run", LR="1e-05" if m == "vit_deit_b" else "n/a",
        Seeds=1, AUROC=f"{v['auroc']:.3f}" if "auroc" in v else "not reported",
        Accuracy=f"{v['accuracy']:.3f}", Sensitivity=f"{v['recall']:.3f}" if "recall" in v else "not reported",
        Specificity=f"{v['specificity']:.3f}" if "specificity" in v else "not reported",
        **{"Balanced acc.": "n/a", "Params (M)": "85.8" if m == "vit_deit_b" else "head only",
           "Train (min)": "not reported"}) for m, v in PAPER.items() if m in order])
    table = pd.concat([paper_rows, show], ignore_index=True)
    (R / "a2_reality_check.md").write_text(
        "# Assignment 2 reality check - Singh et al. (2024) vs this pipeline\n\n" + md_table(table) + "\n")

    # figure: accuracy under the two splits, per model, with the paper's bar
    fig, ax = plt.subplots(figsize=(8, 4))
    base = ours[ours.tag == ""]
    models = [m for m in order if m in set(base.model)]
    x = np.arange(len(models)); w = 0.27
    for k, (split, col, lab) in enumerate([("patient", "#34a853", "patient-grouped split (A1)"),
                                           ("image", "#ea4335", "random split by image (paper protocol)")]):
        vals = [base[(base.model == m) & (base.split == split)]["accuracy"].mean() for m in models]
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(x[i] + (k - 1) * w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    vals = [PAPER.get(m, {}).get("accuracy", np.nan) for m in models]
    ax.bar(x + w, vals, w, label="reported in the paper", color="#9aa0a6")
    for i, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(x[i] + w, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models], rotation=12, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.02); ax.set_ylabel("Test accuracy"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Same pipeline, two splits: how much of the reported accuracy is the split?")
    fig.tight_layout(); fig.savefig(F / "a2_split_effect.png", dpi=200); plt.close(fig)
    print(md_table(table)); print(f"\n[report_a2] wrote {R}/a2_reality_check.md/csv, figures/a2_split_effect.png")


if __name__ == "__main__":
    main(sys.argv[1:])
