from __future__ import annotations
"""
experiment_sampling_ablation.py — Experiment 2: Controlled Frame-Sampling Ablation.

Determines whether UNIFORM or PIXEL-INTENSITY frame sampling classifies the CricketEC
clips better, for BOTH a GRU and an LSTM temporal head, with everything else held
fixed (ResNet-34 backbone, 16 frames, identical training settings, one fixed seed).

Four configurations (2 x 2):
    E2-A  ResNet-34 + GRU  + Uniform          E2-B  ResNet-34 + GRU  + Pixel-Intensity
    E2-C  ResNet-34 + LSTM + Uniform          E2-D  ResNet-34 + LSTM + Pixel-Intensity

Pipeline per configuration:
  1. train on CricketEC/train for up to --epochs epochs (default 40), fixed seed
  2. after each epoch, score CricketEC/val; keep the checkpoint with the best val acc
  3. evaluate that best checkpoint on CricketEC/test
  4. record accuracy, macro precision, macro recall, macro F1

Reuses the existing code: dataset.py (both samplings), models/gru.py (ResNet34+GRU),
models/cnn.py (ResNet34+LSTM), precompute_indices.py (pixel-intensity selection).

Outputs (into --out, default: experiment2/):
  sampling_ablation_results.csv      one row per configuration, all metrics
  sampling_ablation_comparison.png   grouped bars: sampling effect, GRU vs LSTM
  weights/<CONFIG>_best.pth          the selected (best-val) checkpoint per config

CUDA is used automatically when available (force with --device). Run:
  python3 experiment_sampling_ablation.py                 # all four, 40 epochs, GPU
  python3 experiment_sampling_ablation.py --epochs 2      # quick pipeline check
  python3 experiment_sampling_ablation.py --configs E2-A E2-C
"""
import argparse
import csv
import json
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset import CricketEC, FrameSampling
from utils import get_classes
from models.cnn import resnet34_lstm_model
from models.gru import gru_resnet34_lstm_model
from precompute_indices import calculate_indices_for_video
from decord import VideoReader

# ----------------------------------------------------------------------------- #
SEED = 42
DATASET_NAME = "CricketEC"
NUM_CLASSES = 15
NUM_FRAMES = 16

# Training settings held IDENTICAL across all four configurations.
BASE = SimpleNamespace(
    DATASET_NAME=DATASET_NAME, NUM_CLASSES=NUM_CLASSES, NUM_FRAMES=NUM_FRAMES,
    BATCH_SIZE=16, LSTM_HIDDEN_DIM=64, LSTM_NUM_LAYERS=2, LSTM_DROPOUT=0.3,
    FC_DROPOUT=0.4, NUM_WORKERS=4, PREFETCH_FACTOR=2,
    LR=1e-3, BACKBONE_LR=1e-4, WEIGHT_DECAY=3e-3,
)

CONFIGS = {
    "E2-A": dict(cnn="ResNet-34", rnn="gru",  sampling="uniform"),
    "E2-B": dict(cnn="ResNet-34", rnn="gru",  sampling="pixel_intensity"),
    "E2-C": dict(cnn="ResNet-34", rnn="lstm", sampling="uniform"),
    "E2-D": dict(cnn="ResNet-34", rnn="lstm", sampling="pixel_intensity"),
}
BUILDERS = {"gru": gru_resnet34_lstm_model, "lstm": resnet34_lstm_model}
SAMPLING = {"uniform": FrameSampling.UNIFORM,
            "pixel_intensity": FrameSampling.PIXEL_INTENSITY}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(choice: str) -> torch.device:
    if choice != "auto":
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_pixel_index_cache(workers: int):
    """Pixel-intensity sampling needs CricketEC/pixel_intensity_indices_16_frames.pkl
    keyed as '<split>/<class>/<video>.avi'. Build it here if it is missing, so the
    pixel-intensity configs never fail with FileNotFoundError."""
    root = Path(DATASET_NAME)
    cache = root / f"pixel_intensity_indices_{NUM_FRAMES}_frames.pkl"
    if cache.exists():
        print(f"[cache] pixel-intensity indices present: {cache}")
        return
    vids = (list(root.glob("train/*/*.avi")) + list(root.glob("val/*/*.avi"))
            + list(root.glob("test/*/*.avi")))
    if not vids:
        raise FileNotFoundError(f"no videos under {root}/train|val|test")
    print(f"[cache] building {cache} for {len(vids)} videos "
          f"(one-off; {NUM_FRAMES} frames each)...")
    index_map = {}
    failed = []
    for vp in tqdm(vids, desc="pixel-intensity indices"):
        key = str(vp.relative_to(root)).replace("\\", "/")
        try:
            vr = VideoReader(str(vp))
            index_map[key] = calculate_indices_for_video(vr, NUM_FRAMES)
        except Exception as e:                                   # noqa: BLE001
            failed.append((key, repr(e)))
    # Fail fast and clearly NOW, rather than letting a missing key crash training
    # with a cryptic KeyError many minutes in. Do NOT save an incomplete cache
    # (a saved partial cache would be reused and keep failing).
    if failed:
        preview = "\n  ".join(f"{k}: {err}" for k, err in failed[:10])
        raise RuntimeError(
            f"pixel-intensity cache build FAILED for {len(failed)}/{len(vids)} "
            f"video(s); not saving a partial cache. First few:\n  {preview}\n"
            f"Fix or remove those clips, then re-run (the cache will rebuild)."
        )
    with cache.open("wb") as f:
        pickle.dump(index_map, f)
    print(f"[cache] saved {len(index_map)} entries -> {cache}")


def make_config() -> SimpleNamespace:
    return SimpleNamespace(**vars(BASE))


def make_transforms():
    train_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomApply([transforms.RandomRotation(15)], p=0.4),
        transforms.RandomApply([transforms.RandomAffine(0, translate=(0.15, 0.15))], p=0.4),
        transforms.RandomApply([transforms.ColorJitter(0.3, 0.3, 0.3, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(7, 7))], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def make_loader(c, split: str, transform, sampling, workers: int, shuffle: bool):
    d = CricketEC(c=c, dir=Path(DATASET_NAME) / split, transform=transform,
                  sampling=SAMPLING[sampling])
    return DataLoader(d, batch_size=c.BATCH_SIZE, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available()), d


def build_optimizer(model, c):
    """All TRAINABLE params are optimized (the backbone at a lower LR). The existing
    train_resnet34.py optimised only the backbone, which would leave the RNN+head
    untrained — fixed here so the ablation is valid."""
    backbone = [p for p in model.feature_extractor.parameters() if p.requires_grad]
    backbone_ids = {id(p) for p in model.feature_extractor.parameters()}
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
    return torch.optim.AdamW(
        [{"params": backbone, "lr": c.BACKBONE_LR}, {"params": rest, "lr": c.LR}],
        weight_decay=c.WEIGHT_DECAY,
    )


@torch.inference_mode()
def score(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for videos, labels in loader:
        logits = model(videos.to(device, non_blocking=True))
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
        y_true.extend(labels.numpy().tolist())
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    return acc, float(p), float(r), float(f1)


def run_config(cid: str, spec: dict, device, epochs: int, workers: int, out_dir: Path):
    print(f"\n{'='*64}\n  {cid}: ResNet-34 + {spec['rnn'].upper()} + {spec['sampling']} "
          f"({NUM_FRAMES} frames)\n{'='*64}")
    set_seed(SEED)                       # identical init + data order for every config
    c = make_config()
    train_tf, eval_tf = make_transforms()

    train_loader, _ = make_loader(c, "train", train_tf, spec["sampling"], workers, True)
    val_loader, _ = make_loader(c, "val", eval_tf, spec["sampling"], workers, False)
    test_loader, test_ds = make_loader(c, "test", eval_tf, spec["sampling"], workers, False)

    model = BUILDERS[spec["rnn"]](c).to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = build_optimizer(model, c)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = weights_dir / f"{cid}_best.pth"
    best_val, best_epoch = -1.0, -1
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        for videos, labels in tqdm(train_loader, desc=f"{cid} epoch {epoch:02d}/{epochs}"):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(videos)
            loss = loss_fn(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * videos.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()

        val_acc, *_ = score(model, val_loader, device)
        print(f"  {cid} epoch {epoch:02d} | train_loss {running/max(total,1):.4f} "
              f"| train_acc {100*correct/max(total,1):.2f}% | val_acc {100*val_acc:.2f}%")
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_ckpt)
            print(f"    -> new best val_acc {100*best_val:.2f}% (saved {best_ckpt.name})")

    # evaluate the BEST-VAL checkpoint on the test partition
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    acc, p, r, f1 = score(model, test_loader, device)
    dt = time.time() - t0
    print(f"  {cid} TEST | acc {100*acc:.2f}% | macroP {p:.4f} | macroR {r:.4f} "
          f"| macroF1 {f1:.4f} | best_epoch {best_epoch} | {dt/60:.1f} min")

    return {
        "config": cid, "cnn": spec["cnn"], "rnn": spec["rnn"].upper(),
        "sampling": spec["sampling"], "frames": NUM_FRAMES,
        "best_val_acc": round(best_val, 4), "best_epoch": best_epoch,
        "test_accuracy": round(acc, 4), "test_precision_macro": round(p, 4),
        "test_recall_macro": round(r, 4), "test_f1_macro": round(f1, 4),
        "n_test": len(test_ds), "train_minutes": round(dt / 60, 1),
        "checkpoint": str(best_ckpt).replace("\\", "/"),
    }


def plot_comparison(rows, path: Path):
    """Grouped bars: for GRU and for LSTM, Uniform vs Pixel-Intensity, on accuracy
    and macro-F1 — so the effect of the sampling strategy is read off directly."""
    by = {(r["rnn"], r["sampling"]): r for r in rows}
    rnns = ["GRU", "LSTM"]
    samplings = [("uniform", "Uniform"), ("pixel_intensity", "Pixel-Intensity")]
    colors = {"uniform": "#2f7d4f", "pixel_intensity": "#c07a2b"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (metric, title) in zip(axes, [("test_accuracy", "Test Accuracy"),
                                          ("test_f1_macro", "Test Macro-F1")]):
        x = np.arange(len(rnns)); w = 0.36
        for i, (skey, slabel) in enumerate(samplings):
            vals = [by.get((rnn, skey), {}).get(metric, 0) for rnn in rnns]
            bars = ax.bar(x + (i - 0.5) * w, vals, w, label=slabel, color=colors[skey])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(rnns)
        ax.set_ylim(0, 1.05); ax.set_ylabel(title); ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(title="Sampling", loc="upper right", fontsize=8)
    fig.suptitle("Experiment 2 — Frame-Sampling Ablation (ResNet-34, 16 frames)",
                 fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Experiment 2 — sampling ablation.")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS),
                    help="which configs to run (default: all four).")
    ap.add_argument("--epochs", type=int, default=40, help="max epochs per config.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers (use 0 if Windows multiprocessing errors).")
    ap.add_argument("--out", default="experiment2", help="output directory.")
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print(f"Seed: {SEED} | epochs: {args.epochs} | configs: {args.configs}")

    # sanity: the three partitions must exist with 15 classes
    for split in ("train", "val", "test"):
        d = Path(DATASET_NAME) / split
        if not d.exists():
            raise FileNotFoundError(f"missing partition: {d}")
        names, _ = get_classes(d)
        if len(names) != NUM_CLASSES:
            raise ValueError(f"{d} has {len(names)} classes, expected {NUM_CLASSES}")

    # build the pixel-intensity cache up front if any chosen config needs it
    if any(CONFIGS[c]["sampling"] == "pixel_intensity"
           for c in args.configs if c in CONFIGS):
        ensure_pixel_index_cache(args.workers)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid in args.configs:
        if cid not in CONFIGS:
            print(f"[skip] unknown config {cid!r}; known: {list(CONFIGS)}")
            continue
        rows.append(run_config(cid, CONFIGS[cid], device, args.epochs,
                               args.workers, out_dir))

    if rows:
        with (out_dir / "sampling_ablation_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (out_dir / "sampling_ablation_results.json").write_text(json.dumps(rows, indent=2))
        plot_comparison(rows, out_dir / "sampling_ablation_comparison.png")
        print(f"\nSaved to {out_dir}/: sampling_ablation_results.csv/json, "
              f"sampling_ablation_comparison.png, weights/<CONFIG>_best.pth")


if __name__ == "__main__":
    main()
