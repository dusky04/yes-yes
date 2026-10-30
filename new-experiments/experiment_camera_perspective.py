from __future__ import annotations
"""
experiment_camera_perspective.py — Experiment 4: Camera-Perspective Robustness.

Do the trained models perform consistently across broadcast camera angles? Using the
camera_perspective annotations, this groups each model's test predictions by
(semantic layer, camera perspective) and reports accuracy / macro P-R-F1 and clip
count per group, then flags the viewpoints with the most errors.

IMPORTANT (per the spec): clips without a camera_perspective annotation are REPORTED
as missing, never auto-assigned. The perspective labels are read from YOUR annotation
file (the merged dataset JSON, or a CSV with file_path + camera_perspective), matched
to test clips by the video file name (a stable clip id).

No training. Reuses the five checkpoints via cec_infer, or Experiment 1's predictions.

Outputs (into --out, default: experiment4/):
  camera_perspective_results.csv        model x (layer, perspective): n, acc, P/R/F1
  camera_perspective_error_analysis.csv per group: error rate + coverage/missing report
  camera_perspective_comparison.png     accuracy by perspective, per model

Run:
  python3 experiment_camera_perspective.py --annotations annotations/merged.json
  python3 experiment_camera_perspective.py --annotations annotations/manual_annotations.csv --device cuda
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

import cec_infer as CI


def clip_key(path: str) -> str:
    """<class folder>/<file name>: unique, unlike the bare file name (some releases
    reuse a name across class folders, e.g. four/six_103.avi and six/six_103.avi)."""
    return "/".join(Path(path).parts[-2:])


def load_perspectives(path: Path):
    """clip-filename -> (semantic_layer, camera_perspective). Accepts the merged
    dataset JSON (list of records) or a CSV with file_path + camera_perspective."""
    m = {}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        records = data if isinstance(data, list) else (
            data.get("clips") or data.get("annotations") or data.get("data") or [])
        for r in records:
            fp = r.get("file_path") or r.get("video") or r.get("path") or ""
            if fp:
                m[clip_key(fp)] = ((r.get("semantic_layer") or "").strip(),
                                    (r.get("camera_perspective") or "").strip())
    else:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                fp = r.get("file_path") or r.get("video") or r.get("path") or ""
                if fp:
                    m[clip_key(fp)] = ((r.get("semantic_layer") or "").strip(),
                                        (r.get("camera_perspective") or "").strip())
    return m


def group_scores(items):
    """items: list of (true_idx, pred_idx). returns acc, macroP, macroR, macroF1, n."""
    if not items:
        return 0.0, 0.0, 0.0, 0.0, 0
    yt = np.array([i for i, _ in items]); yp = np.array([j for _, j in items])
    acc = float((yt == yp).mean())
    labels = sorted(set(yt.tolist()))
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, labels=labels,
                                                  average="macro", zero_division=0)
    return acc, float(p), float(r), float(f1), len(items)


def main():
    ap = argparse.ArgumentParser(description="Experiment 4 — camera-perspective robustness.")
    ap.add_argument("--annotations", required=True,
                    help="merged dataset JSON, or a CSV with file_path + camera_perspective.")
    ap.add_argument("--models", nargs="*", default=list(CI.CONFIGS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--predictions", default="test_eval/test_predictions.csv")
    ap.add_argument("--out", default="experiment4")
    args = ap.parse_args()

    ann_path = Path(args.annotations)
    if not ann_path.exists():
        raise SystemExit(f"annotations file not found: {ann_path}")
    persp = load_perspectives(ann_path)
    print(f"Loaded perspective labels for {len(persp)} clips from {ann_path.name}")

    device = CI.pick_device(args.device)
    preds, source = CI.gather_predictions(device, args.models, pred_csv=args.predictions)
    print(f"Predictions source: {source}  |  models: {list(preds)}")

    ALL = sorted(CI.STROKE_CLASSES + CI.OUTCOME_CLASSES)
    IDX = {c: i for i, c in enumerate(ALL)}

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    result_rows, error_rows = [], []
    coverage_report = []

    for mid in args.models:
        if mid not in preds:
            continue
        rec = preds[mid]
        groups = defaultdict(list)       # (layer, perspective) -> [(true_idx, pred_idx)]
        n_total = len(rec["videos"])
        n_missing = 0
        missing_examples = []
        for v, t, p in zip(rec["videos"], rec["true"], rec["pred"]):
            name = clip_key(v)
            if name in persp and persp[name][1]:
                layer, pv = persp[name]
                if not layer:                      # derive layer from the true class
                    layer = CI.semantic_of(t)
                groups[(layer, pv)].append((IDX[t], IDX[p]))
            else:
                n_missing += 1
                if len(missing_examples) < 5:
                    missing_examples.append(name)

        covered = n_total - n_missing
        coverage_report.append({
            "model": mid, "n_test_clips": n_total, "with_perspective": covered,
            "missing_perspective": n_missing,
            "coverage_pct": round(100 * covered / n_total, 1) if n_total else 0.0,
            "missing_examples": "; ".join(missing_examples),
        })
        print(f"  [{mid}] {covered}/{n_total} test clips have a perspective label "
              f"({n_missing} missing)")

        for (layer, pv), items in sorted(groups.items()):
            acc, pr, rc, f1, n = group_scores(items)
            result_rows.append({
                "model": mid, "semantic_layer": layer, "camera_perspective": pv,
                "n_clips": n, "accuracy": round(acc, 4),
                "precision_macro": round(pr, 4), "recall_macro": round(rc, 4),
                "f1_macro": round(f1, 4),
            })
            error_rows.append({
                "model": mid, "semantic_layer": layer, "camera_perspective": pv,
                "n_clips": n, "n_errors": int(round((1 - acc) * n)),
                "error_rate": round(1 - acc, 4),
            })

    if not result_rows:
        print("No grouped results — check that the annotation clip names match the "
              "test video names, and that camera_perspective is filled.")
        # still write the coverage report so the mismatch is visible
    # write outputs
    if result_rows:
        with (out / "camera_perspective_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result_rows[0].keys()))
            w.writeheader(); w.writerows(result_rows)
        (out / "camera_perspective_results.json").write_text(json.dumps(result_rows, indent=2))
    with (out / "camera_perspective_error_analysis.csv").open("w", newline="") as f:
        fields = ["model", "semantic_layer", "camera_perspective", "n_clips",
                  "n_errors", "error_rate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(error_rows)
    with (out / "camera_perspective_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(coverage_report[0].keys()))
        w.writeheader(); w.writerows(coverage_report)

    if result_rows:
        _plot(result_rows, out / "camera_perspective_comparison.png")
        # worst viewpoints
        worst = sorted(result_rows, key=lambda r: r["accuracy"])[:5]
        print("\n  viewpoints with most errors (lowest accuracy groups):")
        for r in worst:
            print(f"    {r['model']} | {r['semantic_layer']}/{r['camera_perspective']}: "
                  f"acc {r['accuracy']*100:.1f}% (n={r['n_clips']})")

    print(f"\nSaved to {out}/: camera_perspective_results.csv/json, "
          f"camera_perspective_error_analysis.csv, camera_perspective_coverage.csv, "
          f"camera_perspective_comparison.png")
    print("   NOTE: clips without a perspective label are reported as missing, not assigned. "
          "A low score for a viewpoint may also reflect its class mix / small sample — "
          "read alongside n_clips.")


def _plot(rows, path):
    models = sorted(set(r["model"] for r in rows))
    persps = sorted(set(f"{r['semantic_layer']}/{r['camera_perspective']}" for r in rows))
    data = {(r["model"], f"{r['semantic_layer']}/{r['camera_perspective']}"): r["accuracy"]
            for r in rows}
    x = np.arange(len(persps)); w = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(persps) * 1.2), 5))
    for i, m in enumerate(models):
        vals = [data.get((m, p), 0) for p in persps]
        ax.bar(x + (i - len(models) / 2) * w + w / 2, vals, w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(persps, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy by camera perspective (per model)", fontweight="bold")
    ax.legend(fontsize=8, ncol=min(len(models), 5)); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
