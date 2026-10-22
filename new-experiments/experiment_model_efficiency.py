from __future__ import annotations
"""
experiment_model_efficiency.py — Experiment 5: Computational Efficiency & Complexity.

Profiles the five trained CNN-RNN models (A-E) on the SAME hardware, same batch size,
with warm-up and GPU synchronization, and pairs the cost with classification accuracy
so the accuracy/cost trade-off is explicit.

Per model it reports:
  * total and trainable parameters (millions)
  * GFLOPs per video (via thop, if installed; otherwise left blank with a note)
  * model-only inference latency (ms/video) and throughput (videos/sec)
  * peak GPU memory during inference (MB)
  * frame extraction + sampling time per video (end-to-end vs model-only)
  * test accuracy (reused from Experiment 1 if available, else computed)

No training. Reuses the five checkpoints via cec_infer.

Outputs (into --out, default: experiment5/):
  model_efficiency_results.csv     one row per model, all measurements + accuracy
  accuracy_vs_latency.png          accuracy vs model-only latency
  accuracy_vs_model_complexity.png accuracy vs parameter count

Run:
  python3 experiment_model_efficiency.py --device cuda          # profile all five on GPU
  python3 experiment_model_efficiency.py --models A --device cuda
Optional (better FLOPs):  pip install thop
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cec_infer as CI
from dataset import CricketEC

LATENCY_ITERS = 50
WARMUP = 5


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_flops(model, num_frames, device):
    """GFLOPs per video via thop, if available. Returns (gflops or None, note)."""
    try:
        from thop import profile
    except Exception:                                        # noqa: BLE001
        return None, "thop not installed (pip install thop)"
    x = torch.randn(1, num_frames, 3, 224, 224, device=device)
    try:
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) * 2 / 1e9, "ok"      # MACs*2 = FLOPs
    except Exception as e:                                   # noqa: BLE001
        return None, f"thop failed: {e}"


@torch.inference_mode()
def measure_latency(model, num_frames, device):
    """Model-only latency (ms/video) + peak GPU memory (MB), batch size 1, warmed up."""
    x = torch.randn(1, num_frames, 3, 224, 224, device=device)
    for _ in range(WARMUP):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    t = []
    for _ in range(LATENCY_ITERS):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    latency_ms = 1000.0 * float(np.mean(t))
    peak_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
               if device.type == "cuda" else None)
    return latency_ms, peak_mb


def measure_extraction(model_id, c, sampling, n_clips=20):
    """Average frame-extraction + sampling time per clip (seconds), measured on real
    clips from the test set using the same dataset/sampling as inference."""
    d = Path(CI.DATASET_NAME) / "test"
    if not d.exists():
        return None
    if hasattr(CI.E, "make_dataset"):   # benchmark-pipeline evaluate_test
        ds = CI.E.make_dataset("test", model_id)
    else:
        ds = CricketEC(c=c, dir=d, transform=CI.E.EVAL_TRANSFORM, sampling=sampling)
    n = min(n_clips, len(ds))
    if n == 0:
        return None
    t0 = time.perf_counter()
    for i in range(n):
        _ = ds[i]                       # reads + samples + transforms one clip
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser(description="Experiment 5 — model efficiency & complexity.")
    ap.add_argument("--models", nargs="*", default=list(CI.CONFIGS))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--predictions", default="test_eval/test_predictions.csv",
                    help="reuse Experiment 1 predictions for the accuracy column if present.")
    ap.add_argument("--no-extraction", action="store_true",
                    help="skip the frame-extraction timing (faster).")
    ap.add_argument("--out", default="experiment5")
    args = ap.parse_args()

    device = CI.pick_device(args.device)
    print(f"Device: {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # accuracy per model (reuse Exp 1 predictions when available)
    acc = {}
    try:
        preds, src = CI.gather_predictions(device, args.models, pred_csv=args.predictions)
        for m, r in preds.items():
            yt = np.array(r["true"]); yp = np.array(r["pred"])
            acc[m] = float((yt == yp).mean())
        print(f"Accuracy from: {src}")
    except Exception as e:                                   # noqa: BLE001
        print(f"   (accuracy unavailable: {e}; leaving blank)")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for mid in args.models:
        if mid not in CI.CONFIGS:
            print(f"[skip] unknown model {mid}")
            continue
        cfg = CI.CONFIGS[mid]
        print(f"\n[{mid}] {CI.human_arch(cfg)}  {cfg['num_frames']}f {cfg['sampling']}")
        model, c = CI.load_one(mid, device)
        model.eval()

        total, trainable = count_params(model)
        gflops, flop_note = measure_flops(model, cfg["num_frames"], device)
        latency_ms, peak_mb = measure_latency(model, cfg["num_frames"], device)
        throughput = (1000.0 / latency_ms) if latency_ms > 0 else None
        extract_s = None if args.no_extraction else measure_extraction(
            mid, c, CI.E.SAMPLING[cfg["sampling"]])

        row = {
            "model": mid, "arch": CI.human_arch(cfg),
            "backbone": CI.backbone_name(cfg), "rnn": CI.rnn_name(cfg),
            "num_frames": cfg["num_frames"], "sampling": cfg["sampling"],
            "accuracy": round(acc[mid], 4) if mid in acc else "",
            "params_total_M": round(total / 1e6, 3),
            "params_trainable_M": round(trainable / 1e6, 3),
            "gflops_per_video": round(gflops, 3) if gflops is not None else "",
            "latency_ms_per_video": round(latency_ms, 2),
            "throughput_videos_per_s": round(throughput, 2) if throughput else "",
            "peak_gpu_mem_mb": round(peak_mb, 1) if peak_mb is not None else "",
            "extract_sample_s_per_video": round(extract_s, 4) if extract_s is not None else "",
            "end_to_end_ms_per_video": (round(latency_ms + extract_s * 1000, 2)
                                        if extract_s is not None else ""),
            "flops_note": flop_note,
        }
        rows.append(row)
        print(f"   params {row['params_total_M']}M | GFLOPs {row['gflops_per_video']} | "
              f"latency {latency_ms:.1f}ms | throughput {row['throughput_videos_per_s']}/s | "
              f"peak {row['peak_gpu_mem_mb']}MB")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        return
    with (out / "model_efficiency_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    (out / "model_efficiency_results.json").write_text(json.dumps(rows, indent=2))

    # plots (only for models that have an accuracy value)
    pr = [r for r in rows if r["accuracy"] != ""]
    if pr:
        _scatter(pr, "latency_ms_per_video", "Model-only latency (ms/video)",
                 out / "accuracy_vs_latency.png", "Accuracy vs. inference latency")
        _scatter(pr, "params_total_M", "Parameters (millions)",
                 out / "accuracy_vs_model_complexity.png", "Accuracy vs. model complexity")
    print(f"\nSaved to {out}/: model_efficiency_results.csv/json, "
          f"accuracy_vs_latency.png, accuracy_vs_model_complexity.png")


def _scatter(rows, xkey, xlabel, path, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in rows:
        x, y = r[xkey], r["accuracy"]
        ax.scatter(x, y, s=70, color="#2f7d4f", zorder=3)
        ax.annotate(f"{r['model']}", (x, y), textcoords="offset points",
                    xytext=(6, 5), fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel("Test accuracy")
    ax.grid(alpha=0.25); ax.set_title(title, fontweight="bold")
    fig.tight_layout(); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
