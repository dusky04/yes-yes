from __future__ import annotations
"""
experiment_semantic_error_analysis.py — Experiment 6: Semantic-Layer & Misclassification.

For each of the five trained models (A-E), on the CricketEC TEST partition:
  * accuracy + macro-F1 separately for the 10 batting-stroke classes and the
    5 match-outcome classes, and overall
  * a full 15x15 confusion matrix (saved as PNG + in the confusion CSV)
  * the most frequently confused class pairs
  * how often errors stay WITHIN a semantic group vs cross BETWEEN groups
  * the list of misclassified clips (for qualitative review)

This is a subgroup ANALYSIS of the existing 15-class task — no retraining, and no
new 2-class model. It reuses the five checkpoints via cec_infer, and will read
Experiment 1's test_predictions.csv if present (so it needs no GPU in that case).

Outputs (into --out, default: experiment6/):
  semantic_layer_results.csv      per model: stroke/outcome/overall acc + F1, error split
  class_confusion_analysis.csv    per model: top confused (true -> pred) pairs with counts
  semantic_group_comparison.png   grouped bars: stroke vs outcome accuracy per model
  semantic_cm_<MODEL>.png         15x15 confusion matrix per model
  misclassified_clips.csv         every misclassified clip (model, true, pred, group split)

Run:
  python3 experiment_semantic_error_analysis.py                 # all five
  python3 experiment_semantic_error_analysis.py --models A D    # a subset
  python3 experiment_semantic_error_analysis.py --device cuda   # force GPU (live inference)
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import cec_infer as CI

ALL_CLASSES = sorted(CI.STROKE_CLASSES + CI.OUTCOME_CLASSES)   # 15, alphabetical
IDX = {c: i for i, c in enumerate(ALL_CLASSES)}
STROKE_IDX = [IDX[c] for c in CI.STROKE_CLASSES]
OUTCOME_IDX = [IDX[c] for c in CI.OUTCOME_CLASSES]


def group_metrics(yt, yp, group_idx):
    """accuracy + macro-F1 over clips whose TRUE label is in group_idx."""
    mask = np.isin(yt, group_idx)
    n = int(mask.sum())
    if n == 0:
        return 0.0, 0.0, 0
    acc = accuracy_score(yt[mask], yp[mask])
    f1 = f1_score(yt[mask], yp[mask], labels=group_idx, average="macro", zero_division=0)
    return float(acc), float(f1), n


def analyse_model(model_id, rec):
    yt = np.array([IDX[n] for n in rec["true"]])
    yp = np.array([IDX[n] for n in rec["pred"]])
    videos = rec["videos"]

    overall_acc = float(accuracy_score(yt, yp))
    overall_f1 = float(f1_score(yt, yp, labels=list(range(15)), average="macro", zero_division=0))
    s_acc, s_f1, n_s = group_metrics(yt, yp, STROKE_IDX)
    o_acc, o_f1, n_o = group_metrics(yt, yp, OUTCOME_IDX)

    # within- vs cross-group errors
    wrong = yt != yp
    within = cross = 0
    mis_rows = []
    for i in np.where(wrong)[0]:
        tg = CI.semantic_of(ALL_CLASSES[yt[i]])
        pg = CI.semantic_of(ALL_CLASSES[yp[i]])
        if tg == pg:
            within += 1
        else:
            cross += 1
        mis_rows.append({
            "model": model_id, "video": videos[i],
            "true": ALL_CLASSES[yt[i]], "pred": ALL_CLASSES[yp[i]],
            "true_group": tg, "pred_group": pg,
            "error_type": "within_group" if tg == pg else "cross_group",
        })
    n_err = within + cross
    cm = confusion_matrix(yt, yp, labels=list(range(15)))

    # top confused pairs (off-diagonal)
    pairs = []
    for a in range(15):
        for b in range(15):
            if a != b and cm[a, b] > 0:
                pairs.append((ALL_CLASSES[a], ALL_CLASSES[b], int(cm[a, b])))
    pairs.sort(key=lambda x: -x[2])

    summary = {
        "model": model_id, "arch": CI.human_arch(CI.CONFIGS[model_id]),
        "num_frames": CI.CONFIGS[model_id]["num_frames"],
        "sampling": CI.CONFIGS[model_id]["sampling"],
        "overall_accuracy": round(overall_acc, 4), "overall_f1_macro": round(overall_f1, 4),
        "stroke_accuracy": round(s_acc, 4), "stroke_f1_macro": round(s_f1, 4), "n_stroke": n_s,
        "outcome_accuracy": round(o_acc, 4), "outcome_f1_macro": round(o_f1, 4), "n_outcome": n_o,
        "n_errors": n_err, "within_group_errors": within, "cross_group_errors": cross,
        "within_group_error_rate": round(within / n_err, 4) if n_err else 0.0,
        "cross_group_error_rate": round(cross / n_err, 4) if n_err else 0.0,
    }
    return summary, cm, pairs, mis_rows


def plot_cm(cm, model_id, path):
    fig, ax = plt.subplots(figsize=(8.2, 7))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(15)); ax.set_xticklabels(ALL_CLASSES, rotation=90, fontsize=7)
    ax.set_yticks(range(15)); ax.set_yticklabels(ALL_CLASSES, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — model {model_id} "
                 f"({CI.human_arch(CI.CONFIGS[model_id])})", fontsize=10, fontweight="bold")
    thr = cm.max() / 2 if cm.max() else 0
    for a in range(15):
        for b in range(15):
            v = cm[a, b]
            if v:
                ax.text(b, a, str(v), ha="center", va="center", fontsize=6,
                        color="white" if v > thr else "black")
    # separate strokes from outcomes visually
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def plot_group_comparison(rows, path):
    rows = sorted(rows, key=lambda r: r["model"])
    models = [r["model"] for r in rows]
    s = [r["stroke_accuracy"] for r in rows]
    o = [r["outcome_accuracy"] for r in rows]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w / 2, s, w, label="Batting strokes (10)", color="#2f7d4f")
    ax.bar(x + w / 2, o, w, label="Match outcomes (5)", color="#c07a2b")
    for i, (sv, ov) in enumerate(zip(s, o)):
        ax.text(i - w / 2, sv + 0.01, f"{sv:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, ov + 0.01, f"{ov:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Stroke vs. outcome accuracy per model", fontweight="bold")
    ax.legend(); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Experiment 6 — semantic-layer & error analysis.")
    ap.add_argument("--models", nargs="*", default=list(CI.CONFIGS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--predictions", default="test_eval/test_predictions.csv",
                    help="reuse Experiment 1 predictions if present (else live inference).")
    ap.add_argument("--top-pairs", type=int, default=10, help="confused pairs to report per model.")
    ap.add_argument("--out", default="experiment6")
    args = ap.parse_args()

    device = CI.pick_device(args.device)
    preds, source = CI.gather_predictions(device, args.models, pred_csv=args.predictions)
    print(f"Predictions source: {source}   |  models: {list(preds)}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summaries, conf_rows, mis_all = [], [], []
    for mid in args.models:
        if mid not in preds:
            print(f"[skip] no predictions for {mid}")
            continue
        summary, cm, pairs, mis_rows = analyse_model(mid, preds[mid])
        summaries.append(summary)
        plot_cm(cm, mid, out / f"semantic_cm_{mid}.png")
        for t, p, n in pairs[:args.top_pairs]:
            conf_rows.append({"model": mid, "true": t, "pred": p, "count": n,
                              "true_group": CI.semantic_of(t), "pred_group": CI.semantic_of(p)})
        mis_all.extend(mis_rows)
        print(f"  {mid}: overall {summary['overall_accuracy']*100:.1f}% | "
              f"stroke {summary['stroke_accuracy']*100:.1f}% | "
              f"outcome {summary['outcome_accuracy']*100:.1f}% | "
              f"errors within/cross = {summary['within_group_errors']}/{summary['cross_group_errors']}")

    if summaries:
        with (out / "semantic_layer_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader(); w.writerows(summaries)
        (out / "semantic_layer_results.json").write_text(json.dumps(summaries, indent=2))
        with (out / "class_confusion_analysis.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "true", "pred", "count",
                                              "true_group", "pred_group"])
            w.writeheader(); w.writerows(conf_rows)
        with (out / "misclassified_clips.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "video", "true", "pred",
                                              "true_group", "pred_group", "error_type"])
            w.writeheader(); w.writerows(mis_all)
        plot_group_comparison(summaries, out / "semantic_group_comparison.png")
        print(f"\nSaved to {out}/: semantic_layer_results.csv/json, class_confusion_analysis.csv, "
              f"misclassified_clips.csv, semantic_group_comparison.png, semantic_cm_<MODEL>.png")


if __name__ == "__main__":
    main()
