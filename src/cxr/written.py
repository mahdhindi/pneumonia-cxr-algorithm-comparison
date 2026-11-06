"""Project experiment: weights WRITTEN by an LLM, no training.

After Takehi, Teoh, Zhang & Khattab, "Sparks of Weight-Space Programming"
(MIT CSAIL, 2026), Exp. 5 (one-shot write) and Exp. 6 (write with a budget of
practice forward passes).  The LLM writes every weight of WrittenCNN
(src/cxr/models.py, 5,569 parameters); nothing is fitted to data.

    # 1. check a weights file (shapes, finiteness, no forbidden code) - no data touched
    python -m cxr.written check   --weights astra/weights_zeroshot.py
    # 2. (condition B only) one practice forward pass on the PRACTICE set (= our validation set)
    python -m cxr.written practice --weights astra/weights_practice_v3.py --log astra/practice_log.jsonl --budget 50
    # 3. final, one-time evaluation on the official test folder
    python -m cxr.written evaluate --weights astra/weights_zeroshot.py --label zeroshot
    # 4. comparison table (written vs trained vs untrained vs A1 models)
    python -m cxr.written report

A weights file is either
  * .py   - defines make_weights() -> {name: array-like}; may import only numpy and math,
            must not read files, use the network, or randomness without a fixed formula; or
  * .json - {name: nested lists}; or
  * .npz  - arrays saved under the parameter names.
The .py file is executed in a separate, isolated Python process in an empty temp directory.
"""
from __future__ import annotations

import argparse
import ast
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
from .models import WrittenCNN, count_params

ALLOWED_IMPORTS = {"numpy", "math"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "getattr", "setattr"}


# ------------------------------------------------------------------ loading
def static_check(src: str) -> list[str]:
    """Reject code that could read data, the network or the file system."""
    problems = []
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of '{a.name}' (only numpy and math are allowed)")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from '{node.module}' (only numpy and math are allowed)")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"use of '{node.id}'")
        elif isinstance(node, ast.Attribute) and node.attr in {"load", "loadtxt", "genfromtxt", "fromfile", "memmap", "save", "savez", "tofile"}:
            problems.append(f"file access '.{node.attr}'")
        elif isinstance(node, ast.Attribute) and node.attr == "random":
            problems.append("np.random (weights must be written, not sampled)")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "make_weights" for n in tree.body):
        problems.append("no top-level function make_weights()")
    return problems


def load_weights(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    if path.suffix == ".npz":
        raw = dict(np.load(path))
    elif path.suffix == ".json":
        raw = {k: np.asarray(v, dtype=np.float32) for k, v in json.loads(path.read_text()).items()}
    elif path.suffix == ".py":
        src = path.read_text()
        bad = static_check(src)
        if bad:
            raise ValueError(f"{path}: rejected - " + "; ".join(bad))
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "w.py").write_text(src)
            runner = ("import sys, numpy as np; sys.path.insert(0, '.'); import w; "
                      "out = w.make_weights(); np.savez('out.npz', **{k: np.asarray(v, dtype=np.float32) for k, v in out.items()})")
            r = subprocess.run([sys.executable, "-I", "-c", runner], cwd=d, capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                raise RuntimeError(f"{path}: make_weights() failed:\n{r.stderr[-2000:]}")
            raw = dict(np.load(Path(d) / "out.npz"))
    else:
        raise ValueError("weights file must be .py, .json or .npz")
    problems = []
    for k, shape in WrittenCNN.SHAPES.items():
        if k not in raw:
            problems.append(f"missing '{k}' {shape}")
        elif tuple(np.shape(raw[k])) != shape:
            problems.append(f"'{k}' has shape {tuple(np.shape(raw[k]))}, expected {shape}")
        elif not np.all(np.isfinite(raw[k])):
            problems.append(f"'{k}' contains NaN or inf")
    extra = set(raw) - set(WrittenCNN.SHAPES)
    if extra:
        problems.append(f"unexpected keys {sorted(extra)}")
    if problems:
        raise ValueError(f"{path}: " + "; ".join(problems))
    return {k: raw[k].astype(np.float32) for k in WrittenCNN.SHAPES}


def build(weights: dict[str, np.ndarray]) -> WrittenCNN:
    m = WrittenCNN()
    m.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    return m.eval()


def scores(model, images, rows, batch=64):
    ds = CachedCXR(images, rows, build_transform(224, False, None))
    p = []
    with torch.no_grad():
        for x, _ in DataLoader(ds, batch, shuffle=False):
            p.append(torch.sigmoid(model(x)).squeeze(1).numpy())
    return np.concatenate(p), ds.labels.astype(int)


# ------------------------------------------------------------------ commands
def cmd_check(a):
    w = load_weights(a.weights)
    m = build(w)
    print(f"OK: {a.weights} -> {count_params(m):,} parameters, all shapes correct, all finite.")
    for k, v in w.items():
        print(f"  {k:14s} {str(v.shape):16s} min {v.min():+.4f}  max {v.max():+.4f}  mean {v.mean():+.4f}")


def cmd_practice(a):
    """One practice forward pass on the PRACTICE set (our patient-grouped validation set),
    which the LLM may see. The official test folder is never used here."""
    log = Path(a.log); log.parent.mkdir(parents=True, exist_ok=True)
    used = sum(1 for _ in log.open()) if log.exists() else 0
    if used >= a.budget:
        sys.exit(f"budget exhausted: {used}/{a.budget} practice passes already used ({log})")
    df, images = load_index(a.data, 224)
    # fixed shuffled order (seed 0) so every slice mixes both classes; the index file is sorted by class.
    # (Pass 1 of the first Astra run used the unshuffled order: 199 normal + 57 pneumonia. Recorded in astra/README.md.)
    va = df[df.split == "val"].sample(frac=1.0, random_state=0).reset_index(drop=True)
    start = (used * a.n) % len(va) if a.rotate else 0
    rows = va.iloc[[(start + i) % len(va) for i in range(min(a.n, len(va)))]]
    p, y = scores(build(load_weights(a.weights)), images, rows)
    m = compute_metrics(y, p, n_boot=0)
    rec = dict(pass_no=used + 1, budget=a.budget, weights=str(a.weights), n=int(len(y)), time=time.strftime("%Y-%m-%d %H:%M:%S"),
               auroc=m["auroc"], accuracy=m["accuracy"], recall=m["recall"], specificity=m["specificity"])
    with log.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"PRACTICE PASS {used + 1}/{a.budget}  (n={len(y)} practice images; label 1 = PNEUMONIA)")
    print(f"AUROC {m['auroc']:.4f} | accuracy {m['accuracy']:.4f} | sensitivity {m['recall']:.4f} | specificity {m['specificity']:.4f}")
    print(f"mean P(pneumonia): pneumonia images {p[y == 1].mean():.4f}, normal images {p[y == 0].mean():.4f}")
    if a.per_image:
        # ids, labels and scores only - no image size or other metadata (A1 showed size leaks the label)
        print("practice_image_id,label,p_pneumonia")
        for (_, r), pi, yi in zip(rows.iterrows(), p, y):
            print(f"P{int(r['cache_idx']):05d},{yi},{pi:.4f}")


def cmd_evaluate(a):
    """Final, one-time evaluation. Writes results/written__<label>.json in the A1 result format."""
    df, images = load_index(a.data, 224)
    w = load_weights(a.weights); model = build(w)
    out = dict(model="writtencnn", source="LLM-written weights, no training", label=a.label, llm=a.llm,
               weights_file=str(a.weights), n_params=count_params(model), params_M=count_params(model) / 1e6,
               train_time_s=0.0, split="patient", resize="pad", ablation="none", train_fraction=1.0, seed=0,
               practice_passes_used=sum(1 for _ in Path(a.log).open()) if a.log and Path(a.log).exists() else 0)
    for split in ("val", "test"):
        p, y = scores(model, images, df[df.split == split])
        out[split] = compute_metrics(y, p)
    R = Path(a.results); R.mkdir(parents=True, exist_ok=True)
    f = R / f"written__{a.label}.json"
    f.write_text(json.dumps(out, indent=2))
    t = out["test"]
    print(f"[written:{a.label}] TEST n={t['n']}  AUROC {t['auroc']:.4f}  accuracy {t['accuracy']:.4f}  "
          f"sensitivity {t['recall']:.4f}  specificity {t['specificity']:.4f}  -> {f}")


def cmd_untrained(a):
    """Floor: the same architecture with PyTorch's default random initialisation, no training."""
    df, images = load_index(a.data, 224)
    R = Path(a.results)
    for seed in a.seeds:
        torch.manual_seed(seed)
        model = WrittenCNN().eval()
        out = dict(model="writtencnn", source="random init, no training", label=f"untrained-seed{seed}", seed=seed,
                   n_params=count_params(model), params_M=count_params(model) / 1e6, train_time_s=0.0)
        for split in ("val", "test"):
            p, y = scores(model, images, df[df.split == split])
            out[split] = compute_metrics(y, p)
        (R / f"written__untrained-seed{seed}.json").write_text(json.dumps(out, indent=2))
        print(f"[untrained seed {seed}] TEST AUROC {out['test']['auroc']:.4f} accuracy {out['test']['accuracy']:.4f}")


def _practice_order(df):
    return df[df.split == "val"].sample(frac=1.0, random_state=0).reset_index(drop=True)


def _test_auroc_of_seed(seed, images, rows_te, cache):
    if seed not in cache:
        torch.manual_seed(seed)
        p, y = scores(WrittenCNN().eval(), images, rows_te, batch=128)
        cache[seed] = compute_metrics(y, p, n_boot=0)
    return cache[seed]


def cmd_floor(a):
    """Distribution of test AUROC over many random, untrained networks (no feedback at all)."""
    df, images = load_index(a.data, 224)
    te = df[df.split == "test"]
    vals = []
    for s in range(1000, 1000 + a.n):
        torch.manual_seed(s)
        p, y = scores(WrittenCNN().eval(), images, te, batch=128)
        m = compute_metrics(y, p, n_boot=0)
        vals.append(dict(seed=s, auroc=m["auroc"], accuracy=m["accuracy"], specificity=m["specificity"]))
        if (s - 999) % 25 == 0:
            print(f"  {s - 999}/{a.n} random networks scored")
    au = np.array([v["auroc"] for v in vals])
    out = dict(kind="random-init floor", n=a.n, seeds=[1000, 999 + a.n], median=float(np.median(au)),
               p05=float(np.percentile(au, 5)), p95=float(np.percentile(au, 95)), max=float(au.max()),
               frac_ge={str(t): float((au >= t).mean()) for t in a.thresholds}, runs=vals)
    Path(a.results, "written__floor.json").write_text(json.dumps(out, indent=2))
    print(f"[floor] {a.n} random untrained networks: test AUROC median {out['median']:.3f}, "
          f"5-95 % [{out['p05']:.3f}, {out['p95']:.3f}], max {out['max']:.3f}")
    for t, f in out["frac_ge"].items():
        print(f"  share with AUROC >= {t}: {f:.1%}")


def cmd_search(a):
    """Matched-budget rival to condition B: draw `passes` random networks, score each on one
    256-image practice slice (the same slice schedule Astra got), keep the best by practice AUROC,
    and score that one on the test set. Repeated `reps` times. No training, no gradients."""
    df, images = load_index(a.data, 224)
    va, te = _practice_order(df), df[df.split == "test"]
    slices = [va.iloc[[(i * 256 + j) % len(va) for j in range(256)]] for i in range(a.passes)]
    cache, picked = {}, []
    for r in range(a.reps):
        best = None
        for k in range(a.passes):
            seed = 100000 + r * a.passes + k
            torch.manual_seed(seed)
            p, y = scores(WrittenCNN().eval(), images, slices[k], batch=128)
            au = compute_metrics(y, p, n_boot=0)["auroc"]
            if best is None or au > best[1]:
                best = (seed, au)
        m = _test_auroc_of_seed(best[0], images, te, cache)
        picked.append(dict(rep=r, seed=best[0], practice_auroc=best[1], test_auroc=m["auroc"], test_accuracy=m["accuracy"],
                           test_specificity=m["specificity"]))
        if (r + 1) % 10 == 0:
            print(f"  {r + 1}/{a.reps} repetitions")
    au = np.array([x["test_auroc"] for x in picked])
    out = dict(kind=f"random search, best of {a.passes} on practice slices", reps=a.reps, passes=a.passes,
               median=float(np.median(au)), p05=float(np.percentile(au, 5)), p95=float(np.percentile(au, 95)),
               frac_ge={str(t): float((au >= t).mean()) for t in a.thresholds}, runs=picked)
    Path(a.results, f"written__search{a.passes}.json").write_text(json.dumps(out, indent=2))
    print(f"[search] best-of-{a.passes} random networks (chosen on practice), {a.reps} repetitions: test AUROC median "
          f"{out['median']:.3f}, 5-95 % [{out['p05']:.3f}, {out['p95']:.3f}]")
    for t, f in out["frac_ge"].items():
        print(f"  share with AUROC >= {t}: {f:.1%}")


def cmd_report(a):
    R = Path(a.results); rows = []

    def add(name, how, recs):
        if not recs:
            return
        t = [r["test"] for r in recs]
        mean = lambda k: float(np.mean([x[k] for x in t]))
        sd = lambda k: float(np.std([x[k] for x in t])) if len(t) > 1 else None
        rows.append(dict(model=name, how=how, runs=len(t), auroc=mean("auroc"), auroc_sd=sd("auroc"), acc=mean("accuracy"),
                         sens=mean("recall"), spec=mean("specificity"), params=recs[0].get("n_params") or recs[0].get("params") or recs[0].get("params_M", 0) * 1e6,
                         train_min=float(np.mean([r.get("train_time_s", 0) for r in recs])) / 60))

    load = lambda pat: [json.loads(f.read_text()) for f in sorted(R.glob(pat))]
    for f in sorted(R.glob("written__*.json")):
        r = json.loads(f.read_text())
        if r.get("source", "").startswith("LLM"):
            n_pass = r.get("practice_passes_before_final", r.get("practice_passes_used", 0))
            ci = r["test"].get("ci95", {}).get("auroc")
            add(f"WrittenCNN, weights written by {r.get('llm', 'LLM')} ({r['label']})",
                f"no training; {n_pass} practice passes" + (f"; AUROC 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""), [r])
    add("WrittenCNN, random init", "no training (floor)", load("written__untrained-seed*.json"))
    add("WrittenCNN, trained", "gradient descent, A1 protocol", load("writtencnn__abl-none__frac-1__seed-4[23].json"))
    for m, name in [("logreg", "Logistic regression (pixels)"), ("smallcnn", "Small CNN (scratch)"), ("resnet50", "ResNet-50 (ImageNet)"), ("majority", "Majority class")]:
        add(name, "A1 result", load(f"{m}__abl-none__frac-1__seed-4[23].json"))
    lines = ["| Model | How the weights were obtained | Runs | Test AUROC | Accuracy | Sensitivity | Specificity | Params | Train (min) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        au = f"{r['auroc']:.3f}" + (f" ± {r['auroc_sd']:.3f}" if r["auroc_sd"] is not None else "")
        lines.append(f"| {r['model']} | {r['how']} | {r['runs']} | {au} | {r['acc']:.3f} | {r['sens']:.3f} | {r['spec']:.3f} | {r['params']:,.0f} | {r['train_min']:.1f} |")
    for name, label in [("written__floor.json", "Random untrained networks (no feedback)"), ("written__search10.json", "Best of 10 random networks, chosen on practice slices (Astra's budget)")]:
        f = R / name
        if f.exists():
            d = json.loads(f.read_text()); n = d.get("n", d.get("reps"))
            ge = ", ".join(f"{float(t):.3f}: {v:.0%}" for t, v in d["frac_ge"].items())
            lines.append(f"| {label} | {n} draws; median [5th-95th pct]; share >= {ge} | {n} | {d['median']:.3f} [{d['p05']:.3f}, {d['p95']:.3f}] | | | | 5,569 | 0.0 |")
    md = "# LLM-written weights vs trained weights (patient-grouped split, official test folder, n = 618)\n\n" + "\n".join(lines) + "\n"
    (R / "written_table.md").write_text(md)
    print(md)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "practice", "evaluate"):
        p = sub.add_parser(name); p.add_argument("--weights", required=True); p.add_argument("--data", default="data")
    sub.choices["practice"].add_argument("--log", default="astra/practice_log.jsonl")
    sub.choices["practice"].add_argument("--budget", type=int, default=50)
    sub.choices["practice"].add_argument("--n", type=int, default=256, help="practice images per pass")
    sub.choices["practice"].add_argument("--rotate", action="store_true", help="use a different slice of the practice set each pass")
    sub.choices["practice"].add_argument("--per_image", action="store_true", help="also print per-image scores")
    sub.choices["evaluate"].add_argument("--label", required=True)
    sub.choices["evaluate"].add_argument("--llm", default="Astra")
    sub.choices["evaluate"].add_argument("--log", default="", help="practice log, to record how many passes were used")
    sub.choices["evaluate"].add_argument("--results", default="results")
    p = sub.add_parser("untrained"); p.add_argument("--data", default="data"); p.add_argument("--results", default="results")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    for name in ("floor", "search"):
        p = sub.add_parser(name); p.add_argument("--data", default="data"); p.add_argument("--results", default="results")
        p.add_argument("--thresholds", type=float, nargs="+", default=[0.519, 0.737, 0.882])
    sub.choices["floor"].add_argument("--n", type=int, default=200)
    sub.choices["search"].add_argument("--reps", type=int, default=100)
    sub.choices["search"].add_argument("--passes", type=int, default=10)
    p = sub.add_parser("report"); p.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    {"check": cmd_check, "practice": cmd_practice, "evaluate": cmd_evaluate, "untrained": cmd_untrained,
     "floor": cmd_floor, "search": cmd_search, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
