"""
paper_configs.py — the five CNN–RNN benchmark configurations (A–E) from the
CricketEC paper (Table 7), trained and evaluated on ONE fixed train/val/test split.

Every configuration reads the same <data-root>/{train,val,test} folders (built by
make_split.py), so all five are scored on the identical test set. Each run writes a
fingerprint of the test file list into results.json so this can be checked afterwards.

Pipeline (paper Sec. 4):
    clip -> sample T frames (uniform Eq. 9 / pixel-intensity Algorithm 1)
         -> centre crop + resize to 224x224, ImageNet normalisation (Table 6)
         -> ImageNet-pretrained CNN, partially fine-tuned (Sec. 4.3.3)
         -> GRU / LSTM, final hidden state h (Sec. 4.4)
         -> Dropout(0.4) -> Linear(H, 15) (Sec. 4.5)
    loss: cross-entropy, optimiser: AdamW (Sec. 4.5.3)

Anti-overfitting additions, NOT in the paper (all on by default; --paper-strict
turns every one of them off to reproduce the paper's protocol exactly):
    * train-time augmentation, identical across a clip's frames: random resized
      crop (70–100% area), horizontal flip, mild colour jitter, random erasing
    * temporal jitter: training clips get different frames each epoch (random
      frame per uniform segment / random T of the top 1.5T pixel-intensity frames)
    * dropout 0.2 on the CNN features before the RNN
    * label smoothing 0.1
    * fine-tuned backbone layers train at 0.1x the Table 7 learning rate
    * early stopping after 8 epochs without a val-accuracy improvement

Per configuration, written to runs/<config>/:
    best.pt         weights with the best validation accuracy (used for test)
    last.pt         final-epoch weights + optimiser state (used for --resume)
    history.csv     per-epoch train/val loss, accuracy, macro F1
    results.json    val + test metrics, per-class report, hyperparameters
    test_confusion_matrix.csv, test_predictions.csv

Run (from the repo root, inside the uv environment):
    uv run benchmark/paper_configs.py --config A          # one config
    uv run benchmark/paper_configs.py --config A B C D E  # all five, in order
    uv run benchmark/paper_configs.py --summarize         # rebuild runs/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
from decord import VideoReader, cpu
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet34_Weights,
    efficientnet_b0,
    resnet34,
)
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)  # ITU-R BT.601 luma


# --- Configurations (paper Table 7) -------------------------------------------
@dataclass(frozen=True)
class Config:
    name: str
    backbone: str      # "resnet34" | "effnet_b0"
    rnn: str           # "gru" | "lstm"
    sampling: str      # "uniform" | "pixel"
    num_frames: int
    batch_size: int
    hidden_dim: int
    epochs: int
    lr: float
    weight_decay: float


CONFIGS = {
    "A": Config("A", "resnet34",  "gru",  "uniform", 16, 12, 128, 20, 1e-3, 1e-4),
    "B": Config("B", "resnet34",  "gru",  "uniform", 24, 12, 128, 20, 1e-3, 1e-4),
    "C": Config("C", "effnet_b0", "gru",  "uniform", 24, 12, 128, 20, 1e-3, 1e-4),
    "D": Config("D", "resnet34",  "lstm", "pixel",   24,  8,  64, 40, 1e-3, 3e-3),
    "E": Config("E", "resnet34",  "lstm", "pixel",   16, 16,  64, 40, 3e-3, 3e-3),
}


# --- Frame sampling (paper Sec. 4.2) -----------------------------------------
def read_all_frames(path: Path, size: int) -> np.ndarray:
    """Decode every frame as RGB; other resolutions are resized (short side = size)
    and centre-cropped to size x size, never stretched."""
    vr = VideoReader(str(path), ctx=cpu(0))
    if len(vr) == 0:
        raise RuntimeError(f"Could not decode any frames from {path}")
    h, w = vr[0].shape[:2]
    if (h, w) != (size, size):
        scale = size / min(h, w)
        vr = VideoReader(str(path), ctx=cpu(0),
                         width=max(size, round(w * scale)), height=max(size, round(h * scale)))
    frames = vr.get_batch(list(range(len(vr)))).asnumpy()  # [N, H', W', 3] uint8
    top, left = (frames.shape[1] - size) // 2, (frames.shape[2] - size) // 2
    return np.ascontiguousarray(frames[:, top:top + size, left:left + size])


def pad_indices(n_frames: int, t: int) -> list[int]:
    """Fewer than T frames: keep them all and repeat the last one (Sec. 4.2)."""
    return list(range(n_frames)) + [n_frames - 1] * (t - n_frames)


def uniform_indices(n_frames: int, t: int) -> list[int]:
    """Eq. 9: i_k = round(k (N-1) / (T-1)), k = 0..T-1."""
    if n_frames < t:
        return pad_indices(n_frames, t)
    if t == 1:
        return [0]
    return [int(math.floor(k * (n_frames - 1) / (t - 1) + 0.5)) for k in range(t)]


def jittered_uniform_indices(n_frames: int, t: int) -> list[int]:
    """Training-time variant of uniform sampling: split the clip into T equal
    segments and take a random frame from each, so every epoch sees new frames."""
    if n_frames < t:
        return pad_indices(n_frames, t)
    bounds = [round(k * n_frames / t) for k in range(t + 1)]
    return [random.randrange(bounds[k], max(bounds[k] + 1, bounds[k + 1])) for k in range(t)]


def pixel_intensity_indices(frames: np.ndarray, t: int, bins: int, pool: float = 1.0) -> list[int]:
    """Algorithm 1: top-T frames by L1 distance between consecutive normalised
    grayscale histograms, returned in chronological order. With pool > 1
    (training-time jitter) T frames are drawn at random from the top pool*T."""
    n_frames = len(frames)
    # Algorithm 1 pads when N < T; at N == T there are only T-1 distances, so
    # every frame is kept as well.
    if n_frames <= t:
        return pad_indices(n_frames, t)
    gray = (frames.astype(np.float32) @ GRAY_WEIGHTS).clip(0, 255).astype(np.int64)  # [N, H, W]
    bin_idx = gray.reshape(n_frames, -1) * bins // 256 + np.arange(n_frames)[:, None] * bins
    hists = np.bincount(bin_idx.ravel(), minlength=n_frames * bins).reshape(n_frames, bins)
    hists = hists / hists.sum(axis=1, keepdims=True)  # normalised so each sums to 1
    dists = np.abs(hists[1:] - hists[:-1]).sum(axis=1)  # d_t for t = 1..N-1
    # stable sort so ties resolve to the earlier frame, keeping runs reproducible
    top = (np.argsort(-dists, kind="stable") + 1).tolist()
    if pool > 1:
        return sorted(random.sample(top[:max(t, round(pool * t))], t))
    return sorted(top[:t])


# --- Dataset -------------------------------------------------------------------
class CricketECClips(Dataset):
    """CricketEC/<split>/<class>/<clip>.avi -> (uint8 [T, 3, H, W], label)."""

    def __init__(self, split_dir: Path, classes: list[str], cfg: Config,
                 size: int, hist_bins: int, augment: v2.Transform | None = None,
                 temporal_jitter: bool = False) -> None:
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.paths = sorted(p for p in split_dir.glob("*/*.avi"))
        unknown = {p.parent.name for p in self.paths} - set(classes)
        if unknown:
            raise ValueError(f"{split_dir} has classes not in train: {sorted(unknown)}")
        if not self.paths:
            raise FileNotFoundError(f"No .avi clips under {split_dir}")
        self.cfg, self.size, self.hist_bins, self.augment = cfg, size, hist_bins, augment
        self.temporal_jitter = temporal_jitter

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        frames = read_all_frames(path, self.size)
        t = self.cfg.num_frames
        if self.cfg.sampling == "uniform":
            indices = (jittered_uniform_indices(len(frames), t) if self.temporal_jitter
                       else uniform_indices(len(frames), t))
        else:
            indices = pixel_intensity_indices(frames, t, self.hist_bins,
                                              pool=1.5 if self.temporal_jitter else 1.0)
        clip = torch.from_numpy(np.ascontiguousarray(frames[indices])).permute(0, 3, 1, 2)
        if self.augment is not None:
            # v2 transforms draw one set of random parameters for the whole
            # [T, 3, H, W] tensor, so every frame of the clip gets the same crop/flip
            clip = self.augment(clip)
        return clip, self.class_to_idx[path.parent.name]


def train_augmentation(size: int) -> v2.Transform:
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
        # a mirrored right-hander's stroke is the same stroke played left-handed
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomApply([v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.8),
        # same patch blanked in every frame, so no single region can be relied on
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15), value=0),
    ])


def split_fingerprint(paths: list[Path], split_dir: Path) -> str:
    rel = "\n".join(p.relative_to(split_dir).as_posix() for p in sorted(paths))
    return hashlib.sha256(rel.encode()).hexdigest()


# --- Model (paper Sec. 4.3–4.5) -----------------------------------------------
def build_backbone(name: str) -> tuple[nn.Module, int]:
    if name == "resnet34":
        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        for p in net.parameters():
            p.requires_grad = False
        # layer1, layer2 (and stem) frozen; layer3, layer4 fine-tuned
        for stage in (net.layer3, net.layer4):
            for p in stage.parameters():
                p.requires_grad = True
        dim = net.fc.in_features                       # 512
        body = nn.Sequential(*list(net.children())[:-1])  # drop fc, keep avgpool
    elif name == "effnet_b0":
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in net.parameters():
            p.requires_grad = False
        # features[0] = stem, features[1..7] = MBConv1..MBConv7, features[8] = head conv.
        # Stem + MBConv1–4 frozen; MBConv5–7 (and the head conv after them) fine-tuned.
        for stage in net.features[5:]:
            for p in stage.parameters():
                p.requires_grad = True
        dim = net.classifier[1].in_features            # 1280
        body = nn.Sequential(net.features, net.avgpool)
    else:
        raise ValueError(name)
    return body, dim


class CNNRNN(nn.Module):
    def __init__(self, cfg: Config, num_classes: int, rnn_layers: int,
                 rnn_dropout: float, fc_dropout: float, feat_dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone, feat_dim = build_backbone(cfg.backbone)
        self.feat_dropout = nn.Dropout(feat_dropout)
        rnn_cls = nn.GRU if cfg.rnn == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=feat_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(cfg.hidden_dim, num_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: uint8 [B, T, 3, H, W]; normalising here keeps the dataloader cheap
        x = (x.float() / 255.0 - self.mean) / self.std
        b, t = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).flatten(1)  # [B*T, D]
        feats = self.feat_dropout(feats.view(b, t, -1))     # [B, T, D]
        _, hidden = self.rnn(feats)
        h = hidden[0] if isinstance(hidden, tuple) else hidden  # LSTM -> (h, c)
        return self.fc(self.dropout(h[-1]))                 # final hidden state


# --- Train / evaluate ----------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_ctx(device: torch.device, enabled: bool):
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled and device.type == "cuda")


@torch.inference_mode()
def evaluate(model, loader, loss_fn, device, amp: bool):
    model.eval()
    loss_sum, preds, labels = 0.0, [], []
    for clips, y in tqdm(loader, desc="  eval", leave=False, disable=None):
        clips, y = clips.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx(device, amp):
            out = model(clips)
            loss = loss_fn(out.float(), y)
        loss_sum += loss.item() * y.size(0)
        preds.append(out.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": loss_sum / len(labels),
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "preds": preds,
        "labels": labels,
    }


def train_one_epoch(model, loader, loss_fn, optimizer, scaler, device, amp, clip_grad, epoch, total):
    model.train()
    loss_sum, correct, seen = 0.0, 0, 0
    bar = tqdm(loader, desc=f"  epoch {epoch:02d}/{total}", leave=False, disable=None)
    for clips, y in bar:
        clips, y = clips.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, amp):
            out = model(clips)
            loss = loss_fn(out.float(), y)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        seen += y.size(0)
        bar.set_postfix(loss=f"{loss_sum / seen:.3f}", acc=f"{correct / seen:.3f}")
    return {"loss": loss_sum / seen, "accuracy": correct / seen}


def make_loader(ds, batch_size, shuffle, workers, seed):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed),
    )


def run_config(cfg: Config, args, device: torch.device) -> dict:
    out_dir = args.out_dir / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    data_root = args.data_root
    classes = sorted(p.name for p in (data_root / "train").iterdir() if p.is_dir())
    augment = train_augmentation(args.img_size) if args.augment else None
    datasets = {s: CricketECClips(data_root / s, classes, cfg, args.img_size, args.hist_bins,
                                  augment if s == "train" else None,
                                  temporal_jitter=args.temporal_jitter and s == "train")
                for s in ("train", "val", "test")}
    test_fp = split_fingerprint(datasets["test"].paths, data_root / "test")
    print(f"\n=== Config {cfg.name}: {cfg.backbone} + {cfg.rnn.upper()} | {cfg.sampling} "
          f"T={cfg.num_frames} | bs={cfg.batch_size} H={cfg.hidden_dim} E={cfg.epochs} "
          f"lr={cfg.lr:g} wd={cfg.weight_decay:g} ===")
    print(f"    classes={len(classes)}  train={len(datasets['train'])}  val={len(datasets['val'])}  "
          f"test={len(datasets['test'])}  test-set sha256={test_fp[:16]}")

    train_dl = make_loader(datasets["train"], cfg.batch_size, True, args.workers, args.seed)
    val_dl = make_loader(datasets["val"], cfg.batch_size, False, args.workers, args.seed)
    test_dl = make_loader(datasets["test"], cfg.batch_size, False, args.workers, args.seed)

    model = CNNRNN(cfg, len(classes), args.rnn_layers, args.rnn_dropout, args.fc_dropout,
                   args.feat_dropout).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"    trainable params: {sum(p.numel() for p in trainable):,} / "
          f"{sum(p.numel() for p in model.parameters()):,}")
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    optimizer = torch.optim.AdamW([
        {"params": [p for p in trainable if id(p) in backbone_ids], "lr": cfg.lr * args.backbone_lr_mult},
        {"params": [p for p in trainable if id(p) not in backbone_ids], "lr": cfg.lr},
    ], weight_decay=cfg.weight_decay)
    use_scaler = args.amp and device.type == "cuda" and not torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    eval_loss_fn = nn.CrossEntropyLoss()  # plain CE so val/test losses stay comparable
    print(f"    augment={args.augment}  temporal_jitter={args.temporal_jitter}  "
          f"feat_dropout={args.feat_dropout}  label_smoothing={args.label_smoothing}  "
          f"backbone_lr={cfg.lr * args.backbone_lr_mult:g}  early_stop_patience={args.patience or 'off'}")

    start_epoch, best, history = 1, {"accuracy": -1.0, "macro_f1": -1.0, "epoch": 0}, []
    last_path, best_path = out_dir / "last.pt", out_dir / "best.pt"
    if args.resume and last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch, best, history = ckpt["epoch"] + 1, ckpt["best"], ckpt["history"]
        print(f"    resumed from epoch {ckpt['epoch']} (best val acc {best['accuracy']:.4f})")

    meta = {"config": asdict(cfg), "classes": classes, "args": vars_for_json(args)}
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, train_dl, loss_fn, optimizer, scaler, device,
                             args.amp, args.clip_grad, epoch, cfg.epochs)
        va = evaluate(model, val_dl, eval_loss_fn, device, args.amp)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_acc": tr["accuracy"],
               "val_loss": va["loss"], "val_acc": va["accuracy"], "val_macro_f1": va["macro_f1"],
               "seconds": round(time.time() - t0, 1)}
        history.append(row)
        improved = (va["accuracy"], va["macro_f1"]) > (best["accuracy"], best["macro_f1"])
        if improved:
            best = {"accuracy": va["accuracy"], "macro_f1": va["macro_f1"], "epoch": epoch}
            torch.save({**meta, "epoch": epoch, "model": model.state_dict()}, best_path)
        torch.save({**meta, "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "history": history}, last_path)
        if args.save_every_epoch:
            torch.save({**meta, "epoch": epoch, "model": model.state_dict()},
                       out_dir / f"epoch{epoch:02d}.pt")
        print(f"    epoch {epoch:02d}/{cfg.epochs} | train loss {tr['loss']:.4f} acc {tr['accuracy']:.4f} | "
              f"val loss {va['loss']:.4f} acc {va['accuracy']:.4f} F1 {va['macro_f1']:.4f} | "
              f"{row['seconds']:.0f}s{'  *best*' if improved else ''}")
        if args.patience and epoch - best["epoch"] >= args.patience:
            print(f"    early stop: no val-accuracy improvement for {args.patience} epochs")
            break

    write_csv(out_dir / "history.csv", history)

    # Final-epoch test score (model is still at the last epoch's weights)
    test_last = evaluate(model, test_dl, eval_loss_fn, device, args.amp)
    # Test score of the checkpoint picked on validation — the headline number
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    val_best = evaluate(model, val_dl, eval_loss_fn, device, args.amp)
    test_best = evaluate(model, test_dl, eval_loss_fn, device, args.amp)

    labels, preds = test_best["labels"], test_best["preds"]
    idx = list(range(len(classes)))
    report = classification_report(labels, preds, labels=idx, target_names=classes,
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=idx)
    with open(out_dir / "test_confusion_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for name, r in zip(classes, cm):
            w.writerow([name, *r.tolist()])
    write_csv(out_dir / "test_predictions.csv", [
        {"clip": p.relative_to(data_root / "test").as_posix(),
         "true": classes[t], "pred": classes[q], "correct": int(t == q)}
        for p, t, q in zip(datasets["test"].paths, labels, preds)
    ])

    results = {
        **meta,
        "best_epoch": best["epoch"],
        "val": {"accuracy": val_best["accuracy"], "macro_f1": val_best["macro_f1"], "loss": val_best["loss"]},
        "test": {"accuracy": test_best["accuracy"], "macro_f1": test_best["macro_f1"], "loss": test_best["loss"]},
        "test_last_epoch": {"accuracy": test_last["accuracy"], "macro_f1": test_last["macro_f1"]},
        "test_per_class": {c: report[c] for c in classes},
        "test_set": {"num_clips": len(datasets["test"]), "sha256": test_fp},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"    -> best epoch {best['epoch']}: val acc {val_best['accuracy']:.4f} F1 {val_best['macro_f1']:.4f} | "
          f"TEST acc {test_best['accuracy']:.4f} F1 {test_best['macro_f1']:.4f}")
    print(f"    saved to {out_dir}")
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def vars_for_json(args) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}


def summarize(out_dir: Path) -> None:
    rows, fingerprints = [], set()
    for name in CONFIGS:
        f = out_dir / name / "results.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        c = r["config"]
        fingerprints.add(r["test_set"]["sha256"])
        rows.append({
            "config": name, "backbone": c["backbone"], "rnn": c["rnn"], "sampling": c["sampling"],
            "frames": c["num_frames"], "best_epoch": r["best_epoch"],
            "val_acc": round(100 * r["val"]["accuracy"], 2), "val_macro_f1": round(r["val"]["macro_f1"], 4),
            "test_acc": round(100 * r["test"]["accuracy"], 2), "test_macro_f1": round(r["test"]["macro_f1"], 4),
        })
    if not rows:
        print(f"No results under {out_dir}")
        return
    write_csv(out_dir / "summary.csv", rows)
    print("\n" + " | ".join(f"{k:>13}" for k in rows[0]))
    for r in rows:
        print(" | ".join(f"{str(v):>13}" for v in r.values()))
    same = "yes" if len(fingerprints) == 1 else f"NO ({len(fingerprints)} different test sets!)"
    print(f"\nAll configs evaluated on the same test set: {same}")
    print(f"Summary written to {out_dir / 'summary.csv'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", choices=list(CONFIGS), default=[],
                    help="configuration(s) to run, in order")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "CricketEC_split",
                    help="folder containing train/ val/ test/ class subfolders (see make_split.py)")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--hist-bins", type=int, default=64,
                    help="K in Algorithm 1 (not stated in the paper)")
    ap.add_argument("--rnn-layers", type=int, default=1)
    ap.add_argument("--rnn-dropout", type=float, default=0.3,
                    help="between stacked RNN layers; no effect when --rnn-layers 1")
    ap.add_argument("--fc-dropout", type=float, default=0.4, help="Eq. 11")
    ap.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm, 0 disables")
    ap.add_argument("--no-augment", dest="augment", action="store_false",
                    help="disable train-time augmentation")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1,
                    help="learning-rate multiplier for the fine-tuned CNN layers")
    ap.add_argument("--no-temporal-jitter", dest="temporal_jitter", action="store_false",
                    help="train on the same frames every epoch")
    ap.add_argument("--feat-dropout", type=float, default=0.2,
                    help="dropout on CNN features before the RNN")
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stopping patience in epochs, 0 disables")
    ap.add_argument("--paper-strict", action="store_true",
                    help="exact paper protocol: no augmentation, no label smoothing, "
                         "one learning rate, no early stopping")
    ap.add_argument("--no-amp", dest="amp", action="store_false",
                    help="disable mixed precision (slower, full fp32)")
    ap.add_argument("--resume", action="store_true", help="continue from runs/<config>/last.pt")
    ap.add_argument("--save-every-epoch", action="store_true", help="also keep epochNN.pt files")
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (smoke tests only)")
    ap.add_argument("--summarize", action="store_true", help="only rebuild runs/summary.csv")
    args = ap.parse_args()
    if args.paper_strict:
        args.augment, args.label_smoothing, args.backbone_lr_mult, args.patience = False, 0.0, 1.0, 0
        args.temporal_jitter, args.feat_dropout = False, 0.0

    if args.summarize or not args.config:
        summarize(args.out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: no CUDA device found, running on CPU")

    for name in args.config:
        cfg = CONFIGS[name]
        if args.epochs:
            cfg = replace(cfg, epochs=args.epochs)
        run_config(cfg, args, device)
        summarize(args.out_dir)


if __name__ == "__main__":
    main()
