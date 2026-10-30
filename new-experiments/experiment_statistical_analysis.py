from __future__ import annotations
"""
experiment_statistical_analysis.py — Experiment 7: Statistical Reliability.

Are the accuracy differences among the five models real, or just sampling noise on
the test partition? Using per-clip predictions from all five models on the SAME
clips, this computes:
  * 95% confidence intervals for accuracy and macro-F1 (paired bootstrap resampling)
  * McNemar's exact test comparing the BEST model against each of the other four
  * Holm correction across those four comparisons
  * confidence intervals for the accuracy DIFFERENCES (best - other)
  * a table flagging which differences are statistically significant

No training, no re-inference needed if Experiment 1's test_predictions.csv exists
(the five models must have been evaluated on the same clips — this script checks).

Outputs (into --out, default: experiment7/):
  statistical_analysis_results.csv   per model: acc + 95% CI, macroF1 + 95% CI, adj p-value
  pairwise_significance_results.csv  best vs each other: McNemar p, Holm p, significant?
  confidence_interval_comparison.png accuracy with 95% CI error bars per model

Run:
  python3 experiment_statistical_analysis.py                  # all five
  python3 experiment_statistical_analysis.py --bootstrap 5000 # more resamples
  python3 experiment_statistical_analysis.py --device cuda    # force GPU (live inference)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import cec_infer as CI

ALL_CLASSES = sorted(CI.STROKE_CLASSES + CI.OUTCOME_CLASSES)
IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
SEED = 42


def mcnemar_exact(best_correct, other_correct):
    """Exact two-sided McNemar on paired boolean correctness arrays.
    b = best right & other wrong; c = best wrong & other right."""
    b = int(np.sum(best_correct & ~other_correct))
    c = int(np.sum(~best_correct & other_correct))
    n = b + c
    if n == 0:
        return 1.0, b, c
    try:
        from scipy.stats import binomtest
        p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    except Exception:                                        # noqa: BLE001
        # normal approximation with continuity correction
        from math import erfc, sqrt
        chi = (abs(b - c) - 1) ** 2 / n
        p = erfc(sqrt(chi / 2))
    return float(p), b, c


def sig(p: float) -> float:
    """p-value to 4 significant digits; round(p, 5) would save any p < 5e-6 as 0.0."""
    return float(f"{p:.4g}")


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description="Experiment 7 — statistical reliability.")
    ap.add_argument("--models", nargs="*", default=list(CI.CONFIGS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--predictions", default="test_eval/test_predictions.csv")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples.")
    ap.add_argument("--out", default="experiment7")
    args = ap.parse_args()

    device = CI.pick_device(args.device)
    preds, source = CI.gather_predictions(device, args.models, pred_csv=args.predictions)
    print(f"Predictions source: {source}  |  models: {list(preds)}")

    # ---- align every model on EXACTLY the same clips ----
    sets = {m: set(preds[m]["videos"]) for m in preds}
    common = set.intersection(*sets.values()) if sets else set()
    clips = sorted(common)
    n = len(clips)
    if n == 0:
        raise SystemExit("no common clips across the models — cannot compare.")
    for m in preds:
        extra = len(sets[m]) - n
        if extra:
            print(f"  [warn] model {m} has {extra} clip(s) not shared by all; "
                  f"using the {n} common clips.")
    print(f"Comparing on {n} clips shared by all {len(preds)} models.")

    # per-model aligned true/pred index arrays (true must match across models)
    def aligned(m):
        d = {v: (t, p) for v, t, p in zip(preds[m]["videos"], preds[m]["true"], preds[m]["pred"])}
        yt = np.array([IDX[d[v][0]] for v in clips])
        yp = np.array([IDX[d[v][1]] for v in clips])
        return yt, yp

    y_true = None
    ypred = {}
    for m in args.models:
        if m not in preds:
            continue
        yt, yp = aligned(m)
        if y_true is None:
            y_true = yt
        elif not np.array_equal(y_true, yt):
            print(f"  [warn] true labels for {m} differ from the reference — check clip ids.")
        ypred[m] = yp
    models = [m for m in args.models if m in ypred]

    obs_acc = {m: float((y_true == ypred[m]).mean()) for m in models}
    obs_f1 = {m: float(f1_score(y_true, ypred[m], labels=list(range(15)),
                                average="macro", zero_division=0)) for m in models}
    best = max(models, key=lambda m: obs_acc[m])
    print(f"Best model: {best} (acc {obs_acc[best]*100:.2f}%)")

    # ---- paired bootstrap (same resample indices for every model) ----
    rng = np.random.default_rng(SEED)
    B = args.bootstrap
    acc_s = {m: np.empty(B) for m in models}
    f1_s = {m: np.empty(B) for m in models}
    diff_s = {m: np.empty(B) for m in models}      # best - m, accuracy
    for b in range(B):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        accs = {}
        for m in models:
            yp = ypred[m][idx]
            a = (yt == yp).mean()
            accs[m] = a
            acc_s[m][b] = a
            f1_s[m][b] = f1_score(yt, yp, labels=list(range(15)), average="macro", zero_division=0)
        for m in models:
            diff_s[m][b] = accs[best] - accs[m]

    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    # ---- McNemar best-vs-rest + Holm ----
    best_correct = (y_true == ypred[best])
    others = [m for m in models if m != best]
    raw_p, bc_counts = {}, {}
    for m in others:
        p, b_, c_ = mcnemar_exact(best_correct, (y_true == ypred[m]))
        raw_p[m] = p
        bc_counts[m] = (b_, c_)
    adj = holm([raw_p[m] for m in others]) if others else []
    adj_p = {m: adj[i] for i, m in enumerate(others)}

    # ---- write results ----
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in models:
        a_lo, a_hi = ci(acc_s[m])
        f_lo, f_hi = ci(f1_s[m])
        is_best = (m == best)
        rows.append({
            "model": m, "arch": CI.human_arch(CI.CONFIGS[m]),
            "num_frames": CI.CONFIGS[m]["num_frames"], "sampling": CI.CONFIGS[m]["sampling"],
            "n_clips": n,
            "accuracy": round(obs_acc[m], 4),
            "acc_ci95_low": round(a_lo, 4), "acc_ci95_high": round(a_hi, 4),
            "f1_macro": round(obs_f1[m], 4),
            "f1_ci95_low": round(f_lo, 4), "f1_ci95_high": round(f_hi, 4),
            "role": "reference (best)" if is_best else "compared to best",
            "mcnemar_p": "" if is_best else sig(raw_p[m]),
            "holm_adj_p": "" if is_best else sig(adj_p[m]),
            "significant_vs_best_0.05": "" if is_best else ("yes" if adj_p[m] < 0.05 else "no"),
        })
    with (out / "statistical_analysis_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "statistical_analysis_results.json").write_text(json.dumps(rows, indent=2))

    pair_rows = []
    for m in others:
        d_lo, d_hi = ci(diff_s[m])
        b_, c_ = bc_counts[m]
        pair_rows.append({
            "best_model": best, "other_model": m,
            "acc_diff_best_minus_other": round(obs_acc[best] - obs_acc[m], 4),
            "diff_ci95_low": round(d_lo, 4), "diff_ci95_high": round(d_hi, 4),
            "mcnemar_b_best_right_other_wrong": b_,
            "mcnemar_c_best_wrong_other_right": c_,
            "mcnemar_p": sig(raw_p[m]), "holm_adj_p": sig(adj_p[m]),
            "significant_0.05": "yes" if adj_p[m] < 0.05 else "no",
        })
    with (out / "pairwise_significance_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows
                           else ["best_model"])
        w.writeheader(); w.writerows(pair_rows)

    # ---- plot: accuracy with 95% CI error bars ----
    ms = [r["model"] for r in rows]
    acc = [r["accuracy"] for r in rows]
    lo = [r["accuracy"] - r["acc_ci95_low"] for r in rows]
    hi = [r["acc_ci95_high"] - r["accuracy"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    colors = ["#235f3c" if m == best else "#2f7d4f" for m in ms]
    ax.bar(ms, acc, color=colors, yerr=[lo, hi], capsize=6)
    for i, a in enumerate(acc):
        ax.text(i, a + max(hi) + 0.01, f"{a:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy with 95% bootstrap CIs  (best = {best})", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(out / "confidence_interval_comparison.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved to {out}/: statistical_analysis_results.csv/json, "
          f"pairwise_significance_results.csv, confidence_interval_comparison.png")
    print(f"   (McNemar + Holm quantify uncertainty on THIS test partition, not across "
          f"training seeds — see the guide.)")


if __name__ == "__main__":
    main()
