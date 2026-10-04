from __future__ import annotations
"""
train_baseline.py — a simple, honest PyTorch baseline for CricketEC.

Design choice: a *late-fusion frame baseline*. We sample N frames per clip
(via extract_frames.py), run each through a ResNet-18, average the per-frame
features, and classify. This is deliberately NOT a fancy video model — a
benchmark needs a floor that future temporal models (I3D, SlowFast, VideoMAE)
must beat. If a frame-average baseline already scores high, the task may be
solvable from single frames and the "temporal" framing is weaker than claimed —
a result worth knowing before building anything heavier.

It reads the match-disjoint splits, so the reported val accuracy is the leak-free
number. It prints overall accuracy plus per-class accuracy (the honest view when
classes are imbalanced — overall accuracy can hide a class the model never gets).

This runs on CPU for a smoke test and uses CUDA automatically if present.

Run:
    python3 baselines/train_baseline.py \
        --train annotations/train_split_match_disjoint.json \
        --val   annotations/val_split_match_disjoint.json \
        --num-frames 16 --strategy uniform --epochs 5 --batch-size 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "baselines"))
from extract_frames import extract_clip_frames   # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ClipFrameDataset(Dataset):
    """Loads N sampled frames per clip as a tensor [N, 3, H, W]."""

    def __init__(self, split_json, num_frames, strategy, size=224, cache=True):
        self.clips = json.loads(Path(split_json).read_text())["clips"]
        self.num_frames = num_frames
        self.strategy = strategy
        self.size = size
        self.cache = {} if cache else None
        # stable class_id -> contiguous target index (0..num_classes-1)
        ids = sorted({c["class_id"] for c in self.clips})
        self.id_to_target = {cid: t for t, cid in enumerate(ids)}
        self.target_to_name = {
            self.id_to_target[c["class_id"]]: c["class_name"] for c in self.clips}
        self.num_classes = len(ids)

    def __len__(self):
        return len(self.clips)

    def _load(self, clip):
        vp = ROOT / clip["file_path"]
        frames = extract_clip_frames(vp, self.num_frames, self.strategy,
                                     (self.size, self.size))
        if not frames:
            frames = [np.zeros((self.size, self.size, 3), np.uint8)] * self.num_frames
        arr = np.stack(frames).astype(np.float32) / 255.0      # [N,H,W,3] BGR
        arr = arr[..., ::-1]                                   # BGR->RGB
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.ascontiguousarray(arr.transpose(0, 3, 1, 2))  # [N,3,H,W]
        return torch.from_numpy(arr)

    def __getitem__(self, i):
        clip = self.clips[i]
        if self.cache is not None and i in self.cache:
            x = self.cache[i]
        else:
            x = self._load(clip)
            if self.cache is not None:
                self.cache[i] = x
        y = self.id_to_target[clip["class_id"]]
        return x, y


class FrameAverageResNet(nn.Module):
    """ResNet-18 backbone, mean-pool over frames, linear head."""

    def __init__(self, num_classes):
        super().__init__()
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            from torchvision.models import resnet18
            backbone = resnet18(weights=None)
        self.feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):                       # x: [B, N, 3, H, W]
        b, n = x.shape[:2]
        x = x.flatten(0, 1)                     # [B*N, 3, H, W]
        f = self.backbone(x)                    # [B*N, D]
        f = f.view(b, n, -1).mean(dim=1)        # average over frames -> [B, D]
        return self.head(f)


def run_epoch(model, loader, device, criterion, optim=None):
    train = optim is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optim.zero_grad()
                loss.backward()
                optim.step()
        loss_sum += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def per_class_accuracy(model, loader, device, target_to_name):
    model.eval()
    n = len(target_to_name)
    hit, cnt = np.zeros(n), np.zeros(n)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        y = y.numpy()
        for p, t in zip(pred, y):
            cnt[t] += 1
            hit[t] += (p == t)
    return {target_to_name[i]: (hit[i] / cnt[i] if cnt[i] else float("nan"))
            for i in range(n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--strategy", choices=["uniform", "pixel"], default="uniform")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}  frames={args.num_frames}  "
          f"strategy={args.strategy}  epochs={args.epochs}")

    train_ds = ClipFrameDataset(args.train, args.num_frames, args.strategy, args.size)
    val_ds = ClipFrameDataset(args.val, args.num_frames, args.strategy, args.size)
    # val must use the training set's label mapping
    val_ds.id_to_target = train_ds.id_to_target
    val_ds.target_to_name = train_ds.target_to_name
    val_ds.num_classes = train_ds.num_classes

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = FrameAverageResNet(train_ds.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best = 0.0
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, train_dl, device, criterion, optim)
        vl, va = run_epoch(model, val_dl, device, criterion, None)
        best = max(best, va)
        print(f"[train] epoch {ep:>2}  train_loss {tl:.3f} acc {ta:.3f}  |  "
              f"val_loss {vl:.3f} acc {va:.3f}")

    print(f"[train] best val accuracy: {best:.3f}")
    print("[train] per-class val accuracy:")
    for name, acc in per_class_accuracy(model, val_dl, device,
                                        train_ds.target_to_name).items():
        print(f"         {name:<16} {acc:.3f}")


if __name__ == "__main__":
    main()
