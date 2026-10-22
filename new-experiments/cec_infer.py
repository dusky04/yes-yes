from __future__ import annotations
"""
cec_infer.py — shared inference helper for Experiments 4-7.

Reuses the PROVEN 5-model loading from evaluate_test.py (the checkpoint-size
auto-detection, the pixel-intensity cache, the exact A-E configs) so every later
experiment loads the same five trained models the same way. Import this; don't
re-implement model loading.

The five models are evaluate_test.CONFIGS (A-E) — the actual trained checkpoints.
Each experiment reports every model's REAL architecture/frames/sampling (read from
the config), so results stay correct regardless of how a spec table labelled them.
"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import evaluate_test as E          # proven loader: CONFIGS, load_model, pick_device, ...
from dataset import CricketEC
from utils import get_classes

CONFIGS = E.CONFIGS
NUM_CLASSES = E.NUM_CLASSES
DATASET_NAME = E.DATASET_NAME
pick_device = E.pick_device

# semantic groups (match the Experiment 6 spec)
STROKE_CLASSES = ["pull", "cover", "defense", "flick", "hook",
                  "late_cut", "lofted", "straight", "square_cut", "sweep"]
OUTCOME_CLASSES = ["six", "four", "wicket", "catch", "lbw"]

# arch id -> human-readable name (the real architecture of each checkpoint)
ARCH_HUMAN = {
    "cec_resnet34_motion": "ResNet-34 + LSTM",
    "cec_effnet_motion":   "EfficientNet-B0 + LSTM",
    "effnet_b0_gru":       "EfficientNet-B0 + GRU",
    "resnet34_gru":        "ResNet-34 + GRU",
    "resnet34_lstm":       "ResNet-34 + LSTM",
    "resnet18_lstm":       "ResNet-18 + LSTM",
}


def human_arch(cfg: dict) -> str:
    return ARCH_HUMAN.get(cfg["arch"], cfg["arch"])


def backbone_name(cfg: dict) -> str:
    a = cfg["arch"]
    if "effnet" in a:
        return "EfficientNet-B0"
    if "resnet18" in a:
        return "ResNet-18"
    return "ResNet-34"


def rnn_name(cfg: dict) -> str:
    # A/B (motion) are LSTM; names ending _gru are GRU; _lstm are LSTM
    a = cfg["arch"]
    return "GRU" if a.endswith("gru") else "LSTM"


def load_one(model_id: str, device, want_pixel_cache=True):
    """Load one trained model (A-E) exactly as evaluate_test does. Returns (model, c)."""
    cfg = CONFIGS[model_id]
    if want_pixel_cache and cfg["sampling"] == "pixel_intensity":
        E.ensure_pixel_index_cache(cfg["num_frames"])
    return E.load_model(cfg, device)


def predict_split(model_id: str, device, split: str = "test", keep_model: bool = False):
    """Run model `model_id` over CricketEC/<split> and return per-clip predictions.

    Returns a dict:
      model, arch, backbone, rnn, num_frames, sampling,
      paths (list[str]), y_true (np.int), y_pred (np.int), class_names (list[str]),
      [model_obj, config]  (only if keep_model=True)
    Order is stable (shuffle=False) so paths[i] <-> y_true[i] <-> y_pred[i].
    """
    cfg = CONFIGS[model_id]
    model, c = load_one(model_id, device)

    d = Path(DATASET_NAME) / split
    if not d.exists():
        raise FileNotFoundError(f"missing partition: {d}")
    class_names, _ = get_classes(d)
    if len(class_names) != NUM_CLASSES:
        raise ValueError(f"{d} has {len(class_names)} classes, expected {NUM_CLASSES}")

    if hasattr(E, "make_dataset"):      # benchmark-pipeline evaluate_test
        ds = E.make_dataset(split, model_id)
        ds.video_paths = ds.paths
    else:
        ds = CricketEC(c=c, dir=d, transform=E.EVAL_TRANSFORM,
                       sampling=E.SAMPLING[cfg["sampling"]])
    loader = DataLoader(ds, batch_size=c.BATCH_SIZE, shuffle=False,
                        num_workers=c.NUM_WORKERS, pin_memory=(device.type == "cuda"))

    y_true, y_pred = [], []
    with torch.inference_mode():
        for videos, labels in loader:
            logits = model(videos.to(device, non_blocking=True))
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(labels.numpy().tolist())

    out = {
        "model": model_id, "arch": human_arch(cfg),
        "backbone": backbone_name(cfg), "rnn": rnn_name(cfg),
        "num_frames": cfg["num_frames"], "sampling": cfg["sampling"],
        "paths": [str(p).replace("\\", "/") for p in ds.video_paths],
        "y_true": np.array(y_true), "y_pred": np.array(y_pred),
        "class_names": class_names,
    }
    if keep_model:
        out["model_obj"], out["config"] = model, c
    else:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


def semantic_of(class_name: str) -> str:
    if class_name in STROKE_CLASSES:
        return "stroke"
    if class_name in OUTCOME_CLASSES:
        return "outcome"
    return "unknown"


# --------------------------------------------------------------------------- #
#  unified predictions for the analysis experiments (4, 6, 7)
# --------------------------------------------------------------------------- #
def _load_predictions_csv(path: Path):
    """Read evaluate_test.py's test_predictions.csv ->
       {model_id: {"videos":[...], "true":[names], "pred":[names]}}."""
    import csv
    data = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            m = r["model"]
            data.setdefault(m, {"videos": [], "true": [], "pred": []})
            data[m]["videos"].append(r["video"].replace("\\", "/"))
            data[m]["true"].append(r["true_label"])
            data[m]["pred"].append(r["pred_label"])
    return data


def gather_predictions(device, model_ids=None,
                       pred_csv="test_eval/test_predictions.csv", split="test"):
    """Per-clip predictions (as class NAMES) for the requested models.

    Prefers an existing test_predictions.csv (from Experiment 1 / evaluate_test.py)
    so the analysis experiments need no GPU and stay consistent with Exp 1. Falls
    back to live inference (needs the checkpoints + GPU) when the CSV is absent or
    is missing a requested model.

    Returns: ({model_id: {"videos","true","pred"}}, source_str)
    """
    model_ids = model_ids or list(CONFIGS)
    csv_path = Path(pred_csv)
    from_csv = {}
    if csv_path.exists():
        try:
            from_csv = _load_predictions_csv(csv_path)
        except Exception:                                    # noqa: BLE001
            from_csv = {}

    have_all = all(m in from_csv and from_csv[m]["videos"] for m in model_ids)
    if have_all:
        return {m: from_csv[m] for m in model_ids}, f"saved predictions ({csv_path})"

    # fall back to live inference for the missing ones
    out = {}
    source = "live inference"
    for m in model_ids:
        if m in from_csv and from_csv[m]["videos"]:
            out[m] = from_csv[m]
            continue
        r = predict_split(m, device, split=split)
        names = r["class_names"]
        out[m] = {
            "videos": r["paths"],
            "true": [names[i] for i in r["y_true"]],
            "pred": [names[i] for i in r["y_pred"]],
        }
    return out, source
