from __future__ import annotations
"""
evaluate_test.py — Experiment 1: test-set evaluation of the five benchmark models (A-E).

Backed by the benchmark pipeline (benchmark/paper_configs.py): the models are the
A-E checkpoints in <runs dir>/<A..E>/best.pt, trained on the fixed 2,638-clip split
(data/CricketEC_split). Each training run already scored its best-val checkpoint on
the test set (with test-time augmentation), so this script collects those per-clip
predictions into one file instead of re-running inference.

It also provides the interface that cec_infer.py (and through it Experiments 4-7)
imports: CONFIGS, NUM_CLASSES, DATASET_NAME, pick_device, ensure_pixel_index_cache,
load_model, make_dataset, EVAL_TRANSFORM, SAMPLING.

Outputs (into --out, default: test_eval/):
  test_predictions.csv   model, video, true_label, pred_label, correct   (read by Exp 4/6/7)
  test_summary.csv       per model: architecture, frames, sampling, accuracy, macro P/R/F1

Run from the repo root:
  PYTHONPATH=.:new-experiments uv run --frozen python new-experiments/evaluate_test.py
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmark"))
import paper_configs as P  # noqa: E402

RUNS_DIR = Path(os.environ.get("CEC_RUNS_DIR", ROOT / "runs_v3"))
DATASET_NAME = os.environ.get("CEC_DATA_ROOT", str(ROOT / "data" / "CricketEC_split"))
MODEL_IDS = ["A", "B", "C", "D", "E"]
NUM_CLASSES = 15

# arch ids understood by cec_infer.ARCH_HUMAN / backbone_name / rnn_name
_ARCH = {("resnet34", "gru"): "resnet34_gru", ("resnet34", "lstm"): "resnet34_lstm",
         ("effnet_b0", "gru"): "effnet_b0_gru"}
CONFIGS = {
    m: {"arch": _ARCH[(P.CONFIGS[m].backbone, P.CONFIGS[m].rnn)],
        "num_frames": P.CONFIGS[m].num_frames,
        "sampling": "uniform" if P.CONFIGS[m].sampling == "uniform" else "pixel_intensity",
        "id": m}
    for m in MODEL_IDS
}
# The pipeline does its own decoding/normalisation; these exist for interface compatibility.
EVAL_TRANSFORM = None
SAMPLING = {"uniform": "uniform", "pixel_intensity": "pixel"}


def pick_device(choice: str = "auto") -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_pixel_index_cache(num_frames: int) -> None:
    """No cache needed: pixel-intensity frames are selected on the fly (Algorithm 1)."""


def checkpoint_path(model_id: str) -> Path:
    return RUNS_DIR / model_id / "best.pt"


def load_model(cfg: dict, device):
    """Return (model, c) for one of A-E, where c carries BATCH_SIZE / NUM_WORKERS /
    NUM_FRAMES like the original evaluate_test did."""
    model_id = cfg["id"]
    ckpt = torch.load(checkpoint_path(model_id), map_location=device, weights_only=False)
    pc = P.Config(**ckpt["config"])
    a = ckpt["args"]
    model = P.CNNRNN(pc, len(ckpt["classes"]), a.get("rnn_layers", 1), a.get("rnn_dropout", 0.3),
                     a.get("fc_dropout", 0.4), a.get("feat_dropout", 0.0))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    c = SimpleNamespace(BATCH_SIZE=pc.batch_size, NUM_WORKERS=4, NUM_FRAMES=pc.num_frames,
                        pipeline_config=pc, classes=ckpt["classes"], args=a)
    return model, c


def make_dataset(split: str, model_id: str):
    """The pipeline's deterministic eval dataset for one model (same sampling as testing)."""
    pc = P.CONFIGS[model_id]
    root = Path(DATASET_NAME)
    classes = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    return P.CricketECClips(root / split, classes, pc, 224, 64)


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 1 — collect A-E test predictions.")
    ap.add_argument("--runs", type=Path, default=RUNS_DIR, help="folder with A..E/ results")
    ap.add_argument("--out", type=Path, default=Path("test_eval"))
    ap.add_argument("--models", nargs="*", default=MODEL_IDS, choices=MODEL_IDS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    pred_rows, summary = [], []
    for m in args.models:
        res, preds = args.runs / m / "results.json", args.runs / m / "test_predictions.csv"
        if not res.exists():
            raise SystemExit(f"model {m} has no results yet ({res}); train it first")
        r = json.loads(res.read_text())
        with open(preds, newline="") as f:
            for row in csv.DictReader(f):
                pred_rows.append({"model": m, "video": row["clip"], "true_label": row["true"],
                                  "pred_label": row["pred"], "correct": row["correct"]})
        # runs from before test_macro_avg was recorded: average the per-class scores
        macro = r.get("test_macro_avg") or {
            k: sum(v[k] for v in r["test_per_class"].values()) / len(r["test_per_class"])
            for k in ("precision", "recall", "f1-score")}
        head = r.get("test_tta") or r["test"]
        summary.append({
            "model": m, "arch": CONFIGS[m]["arch"], "frames": CONFIGS[m]["num_frames"],
            "sampling": CONFIGS[m]["sampling"], "best_epoch": r["best_epoch"],
            "val_accuracy": round(r["val"]["accuracy"], 4),
            "test_accuracy": round(head["accuracy"], 4),
            "test_precision_macro": round(macro["precision"], 4),
            "test_recall_macro": round(macro["recall"], 4),
            "test_f1_macro": round(macro["f1-score"], 4),
            "predictions": r.get("test_per_class_source", "test"),
            "test_set_sha256": r["test_set"]["sha256"][:16],
        })
        print(f"  {m}: {CONFIGS[m]['arch']} {CONFIGS[m]['num_frames']}f {CONFIGS[m]['sampling']} | "
              f"test acc {100 * head['accuracy']:.2f}% | macro F1 {macro['f1-score']:.4f}")
    if len({s["test_set_sha256"] for s in summary}) != 1:
        raise SystemExit("the five models were not scored on the same test set")

    for name, rows in (("test_predictions.csv", pred_rows), ("test_summary.csv", summary)):
        with open(args.out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"Saved {args.out}/test_predictions.csv ({len(pred_rows)} rows) and test_summary.csv")


if __name__ == "__main__":
    main()
