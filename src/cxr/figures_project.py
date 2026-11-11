"""Figures for the course-project report and slides (LLM-written weights).

    python -m cxr.figures_project          # writes results/figures/project_{ladder,robust}.{pdf,png}

Every number is read from results/*.json; nothing is typed in by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path("results")
OUT = R / "figures"
ASTRA, TRAINED, REF = "#2a78d6", "#eb6834", "#8a8984"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0, "pdf.fonttype": 42, "savefig.dpi": 300})


def j(name):
    return json.loads((R / name).read_text())


def ladder():
    rows = []   # (label, value, lo, hi, colour, marker, group)
    for lab, f in [("A  nothing (zero-shot)", "written__astra-zeroshot.json"), ("C  16 example images", "written__astra-examples.json"),
                   ("B  practice passes", "written__astra-practice.json"), ("D  images + practice", "written__astra-examples-practice.json"),
                   ("F  + channel report, run 1", "written__astra-diagnose-r1.json"), ("F  + channel report, run 2", "written__astra-diagnose-r2.json"),
                   ("F  + channel report, run 3", "written__astra-diagnose-r3.json")]:
        t = j(f)["test"]
        rows.append((lab, t["auroc"], *t["ci95"]["auroc"], ASTRA, "o", "Astra writes the weights (no training)"))
    fl, s4, s10 = j("written__floor.json"), j("written__search4.json"), j("written__search10.json")
    rows.append(("Random weights, no selection", fl["median"], fl["p05"], fl["p95"], REF, "D", "Rivals and reference"))
    rows.append(("Best of 4 random nets (F budget)", s4["median"], s4["p05"], s4["p95"], REF, "D", "Rivals and reference"))
    rows.append(("Best of 10 random nets (B budget)", s10["median"], s10["p05"], s10["p95"], REF, "D", "Rivals and reference"))
    for s in (42, 43):
        t = j(f"writtencnn__abl-none__frac-1__seed-{s}.json")["test"]
        rows.append((f"Same CNN, gradient-trained (seed {s})", t["auroc"], *t["ci95"]["auroc"], TRAINED, "s", "Rivals and reference"))
    fig, ax = plt.subplots(figsize=(3.45, 3.0))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, v, lo, hi, c, m, _) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.4, solid_capstyle="round", alpha=0.55 if c == REF else 0.8)
        ax.plot([v], [yi], marker=m, ms=5, color=c, mec="white", mew=0.8, zorder=3)
        ax.text(min(hi + 0.012, 0.985), yi, f"{v:.3f}", va="center", ha="left", fontsize=6.3, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.axhline(y[6] - 0.5, color=GRID, lw=0.8)
    ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(0.2, 1.0); ax.set_xlabel("Test AUROC (official test folder, n = 618)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    fig.text(0.01, 0.005, "Bars: 95% bootstrap CI (Astra, trained); 5th-95th percentile over 100-200 draws (random rows).",
             fontsize=5.6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def robust():
    d = j("robust_size.json")
    pick = [("F run 1", "(F, run 1)", ASTRA, "o"), ("F run 2", "(F, run 2)", ASTRA, "o"), ("F run 3", "(F, run 3)", ASTRA, "o"),
            ("Same CNN, trained s42", "(seed 42)", TRAINED, "s"), ("Same CNN, trained s43", "(seed 43)", TRAINED, "s"),
            ("ResNet-50, trained", "ResNet-50", TRAINED, "s"), ("Image height alone", "height alone", REF, "D")]
    rows = []
    for lab, key, c, m in pick:
        r = next(r for r in d["rows"] if key in r["model"] and (key != "(seed 42)" or "Same CNN" in r["model"]))
        rows.append((lab, r["auroc_full"], r["auroc_matched"], c, m))
    fig, ax = plt.subplots(figsize=(3.45, 2.05))
    y = list(range(len(rows)))[::-1]
    for yi, (lab, full, mat, c, m) in zip(y, rows):
        ax.annotate("", xy=(mat, yi), xytext=(full, yi), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.1, alpha=0.7, shrinkA=3, shrinkB=3, mutation_scale=7))
        ax.plot([full], [yi], marker=m, ms=5, mfc="white", mec=c, mew=1.1, zorder=3)
        ax.plot([mat], [yi], marker=m, ms=5, color=c, mec="white", mew=0.8, zorder=3)
        ax.text(max(full, mat) + 0.03, yi, f"{full:.3f} → {mat:.3f}", va="center", fontsize=6.3, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0.5, color=GRID, lw=0.8, zorder=0)
    ax.set_xlim(0.45, 1.12); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel(f"AUROC: all 618 test images (open) → size-matched subset, {2 * d['n_pairs']} images (filled)")
    ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    fig.tight_layout()
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, make in (("project_ladder", ladder), ("project_robust", robust)):
        fig = make()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"wrote {OUT / name}.pdf/.png")


if __name__ == "__main__":
    main()
