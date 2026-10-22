from __future__ import annotations
"""
experiment_frame_sensitivity.py — Experiment 3: Frame-Count Sensitivity.

Measures how the NUMBER of sampled frames affects classification accuracy and
computational cost on CricketEC, with everything else held fixed (ResNet-34
backbone, GRU temporal head, UNIFORM sampling, identical training settings and
one fixed seed). Only the frame count varies.

Four configurations:
    E3-A  ResNet-34 + GRU + Uniform +  8 frames
    E3-B  ResNet-34 + GRU + Uniform + 16 frames
    E3-C  ResNet-34 + GRU + Uniform + 24 frames
    E3-D  ResNet-34 + GRU + Uniform + 32 frames

Pipeline per configuration (identical protocol for a strictly controlled study):
  1. train on CricketEC/train for up to --epochs epochs (default 40), fixed seed
  2. after each epoch, score CricketEC/val; keep the checkpoint with the best val acc
  3. evaluate that best checkpoint on CricketEC/test
  4. record accuracy, macro precision, macro recall, macro F1
  5. measure per-clip inference latency (ms) and peak GPU memory (MB)

Reuses the existing code: dataset.py (uniform sampling), models/gru.py
(ResNet34+GRU). No pixel-intensity cache is needed (uniform sampling only).

Outputs (into --out, default: experiment3/):
  frame_sensitivity_results.csv / .json   one row per frame count, all metrics + cost
  frame_count_vs_accuracy.png             frames vs test accuracy and macro-F1
  frame_count_vs_latency.png              frames vs per-clip latency and peak GPU memory
  weights/<CONFIG>_best.pth               the selected (best-val) checkpoint per config

CUDA is used automatically when available (force with --device). Run:
  python3 experiment_frame_sensitivity.py                 # all four, 40 epochs, GPU
  python3 experiment_frame_sensitivity.py --epochs 2      # quick pipeline check
  python3 experiment_frame_sensitivity.py --configs E3-A E3-D
"""
import argparse
import csv
import json
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
from models.gru import gru_resnet34_lstm_model

# ----------------------------------------------------------------------------- #
SEED = 42
DATASET_NAME = "CricketEC"
NUM_CLASSES = 15

# Everything below is held IDENTICAL across the four configurations; only the
# frame count (the dict value) changes.
CONFIGS = {
    "E3-A": 8,
    "E3-B": 16,
    "E3-C": 24,
    "E3-D": 32,
}

# Training settings held IDENTICAL across all four configurations.
BASE = SimpleNamespace(
    DATASET_NAME=DATASET_NAME, NUM_CLASSES=NUM_CLASSES,
    BATCH_SIZE=16, LSTM_HIDDEN_DIM=64, LSTM_NUM_LAYERS=2, LSTM_DROPOUT=0.3,
    FC_DROPOUT=0.4, NUM_WORKERS=4, PREFETCH_FACTOR=2,
    LR=1e-3, BACKBONE_LR=1e-4, WEIGHT_DECAY=3e-3,
)

# latency timing: time a capped number of single-clip forward passes so the
# number reflects per-clip inference cost, isolated from batching.
LATENCY_SAMPLE = 100
LATENCY_WARMUP = 3


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


def make_config(num_frames: int) -> SimpleNamespace:
    c = SimpleNamespace(**vars(BASE))
    c.NUM_FRAMES = num_frames
    return c


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


def make_loader(c, split: str, transform, workers: int, shuffle: bool, batch_size=None):
    d = CricketEC(c=c, dir=Path(DATASET_NAME) / split, transform=transform,
                  sampling=FrameSampling.UNIFORM)
    return DataLoader(d, batch_size=batch_size or c.BATCH_SIZE, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available()), d


def build_optimizer(model, c):
    """All TRAINABLE params are optimized (the backbone at a lower LR), matching
    the Experiment 2 protocol so the two studies are directly comparable."""
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


@torch.inference_mode()
def measure_cost(model, dataset, device):
    """Per-clip inference latency (ms) and peak GPU memory (MB). Times single-clip
    forward passes (batch size 1) over a capped sample, after warm-up, so the
    latency reflects the frame count rather than batch throughput. Peak memory is
    read from CUDA's allocator for the same single-clip forward."""
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    it = iter(loader)

    # warm-up (kernel autotune / cache) — not timed
    for _ in range(LATENCY_WARMUP):
        try:
            v, _ = next(it)
        except StopIteration:
            it = iter(loader); v, _ = next(it)
        model(v.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)

    times = []
    for i, (v, _) in enumerate(loader):
        if i >= LATENCY_SAMPLE:
            break
        v = v.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(v)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_ms = 1000.0 * (sum(times) / max(len(times), 1))
    peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
               if device.type == "cuda" else float("nan"))
    return round(latency_ms, 2), (round(peak_mb, 1) if peak_mb == peak_mb else None), len(times)


def run_config(cid: str, num_frames: int, device, epochs: int, workers: int, out_dir: Path):
    print(f"\n{'='*64}\n  {cid}: ResNet-34 + GRU + Uniform + {num_frames} frames"
          f"\n{'='*64}")
    set_seed(SEED)                       # identical init + data order for every config
    c = make_config(num_frames)
    train_tf, eval_tf = make_transforms()

    train_loader, _ = make_loader(c, "train", train_tf, workers, True)
    val_loader, _ = make_loader(c, "val", eval_tf, workers, False)
    test_loader, test_ds = make_loader(c, "test", eval_tf, workers, False)

    model = gru_resnet34_lstm_model(c).to(device)
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
    latency_ms, peak_mb, n_timed = measure_cost(model, test_ds, device)
    dt = time.time() - t0
    print(f"  {cid} TEST | acc {100*acc:.2f}% | macroF1 {f1:.4f} "
          f"| latency {latency_ms} ms/clip | peak GPU {peak_mb} MB "
          f"| best_epoch {best_epoch} | {dt/60:.1f} min")

    return {
        "config": cid, "cnn": "ResNet-34", "rnn": "GRU", "sampling": "uniform",
        "frames": num_frames,
        "best_val_acc": round(best_val, 4), "best_epoch": best_epoch,
        "test_accuracy": round(acc, 4), "test_precision_macro": round(p, 4),
        "test_recall_macro": round(r, 4), "test_f1_macro": round(f1, 4),
        "latency_ms_per_clip": latency_ms, "peak_gpu_mem_mb": peak_mb,
        "n_latency_clips": n_timed, "n_test": len(test_ds),
        "train_minutes": round(dt / 60, 1),
        "checkpoint": str(best_ckpt).replace("\\", "/"),
    }


def plot_accuracy(rows, path: Path):
    rows = sorted(rows, key=lambda r: r["frames"])
    frames = [r["frames"] for r in rows]
    acc = [r["test_accuracy"] for r in rows]
    f1 = [r["test_f1_macro"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(frames, acc, "-o", color="#2f7d4f", label="Test accuracy")
    ax.plot(frames, f1, "-s", color="#c07a2b", label="Test macro-F1")
    for x, y in zip(frames, acc):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xlabel("Number of sampled frames")
    ax.set_ylabel("Score")
    ax.set_xticks(frames)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Frame count vs. classification accuracy (ResNet-34 + GRU, uniform)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cost(rows, path: Path):
    rows = sorted(rows, key=lambda r: r["frames"])
    frames = [r["frames"] for r in rows]
    latency = [r["latency_ms_per_clip"] for r in rows]
    mem = [r["peak_gpu_mem_mb"] for r in rows]
    have_mem = all(m is not None for m in mem)

    fig, ax1 = plt.subplots(figsize=(7, 4.6))
    ax1.plot(frames, latency, "-o", color="#2f7d4f", label="Latency (ms/clip)")
    ax1.set_xlabel("Number of sampled frames")
    ax1.set_ylabel("Inference latency (ms / clip)", color="#2f7d4f")
    ax1.tick_params(axis="y", labelcolor="#2f7d4f")
    ax1.set_xticks(frames)
    ax1.grid(alpha=0.25)
    for x, y in zip(frames, latency):
        ax1.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#235f3c")

    if have_mem:
        ax2 = ax1.twinx()
        ax2.plot(frames, mem, "-s", color="#b0472b", label="Peak GPU memory (MB)")
        ax2.set_ylabel("Peak GPU memory (MB)", color="#b0472b")
        ax2.tick_params(axis="y", labelcolor="#b0472b")
        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="upper left", fontsize=9)
    else:
        ax1.legend(loc="upper left", fontsize=9)

    ax1.set_title("Frame count vs. computational cost (ResNet-34 + GRU, uniform)",
                  fontsize=10, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Experiment 3 — frame-count sensitivity.")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS),
                    help="which configs to run (default: all four).")
    ap.add_argument("--epochs", type=int, default=40, help="max epochs per config.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers (use 0 if Windows multiprocessing errors).")
    ap.add_argument("--out", default="experiment3", help="output directory.")
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
        with (out_dir / "frame_sensitivity_results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (out_dir / "frame_sensitivity_results.json").write_text(json.dumps(rows, indent=2))
        plot_accuracy(rows, out_dir / "frame_count_vs_accuracy.png")
        plot_cost(rows, out_dir / "frame_count_vs_latency.png")
        print(f"\nSaved to {out_dir}/: frame_sensitivity_results.csv/json, "
              f"frame_count_vs_accuracy.png, frame_count_vs_latency.png, "
              f"weights/<CONFIG>_best.pth")


if __name__ == "__main__":
    main()
