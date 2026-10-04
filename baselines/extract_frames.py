"""
extract_frames.py — sample a fixed number of frames from each clip.

Two strategies (the professor's "Uniform / Pixel-Intensity"):

  uniform   : pick N frames at evenly spaced indices across the clip. Simple,
              unbiased baseline. Good default for a clean action-recognition set.

  pixel     : pick the N frames with the largest pixel-intensity change from the
              previous frame (mean absolute difference). This biases sampling
              toward moments of motion — bat swing, ball contact — which is
              often where the discriminative signal in a cricket stroke lives.
              The trade-off: it can over-sample camera cuts or crowd motion, so
              it isn't strictly better; it's a different inductive bias.

Output: one folder of JPEGs per clip under --out, mirroring the class layout:
    <out>/<class>/<clip_stem>/frame_00.jpg ...

This is a utility: stage/baseline scripts call extract_clip_frames(), or you can
run it standalone to materialise frames for all clips referenced by a split JSON.

Run:
    python3 baselines/extract_frames.py \
        --split annotations/train_split_match_disjoint.json \
        --out frames/train --num-frames 16 --strategy uniform
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _uniform_indices(total: int, n: int) -> list[int]:
    if total <= 0:
        return []
    if total <= n:
        return list(range(total))
    # evenly spaced, inclusive of first and last
    return list(np.linspace(0, total - 1, n).round().astype(int))


def _pixel_change_indices(path: Path, n: int) -> list[int]:
    """Return indices of the n frames with largest mean-abs-diff from previous."""
    cap = cv2.VideoCapture(str(path))
    prev, diffs, idx = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append((float(np.mean(cv2.absdiff(gray, prev))), idx))
        prev = gray
        idx += 1
    cap.release()
    if not diffs:
        return _uniform_indices(idx, n)
    diffs.sort(reverse=True)                       # largest change first
    chosen = sorted(i for _, i in diffs[:n])
    return chosen or _uniform_indices(idx, n)


def extract_clip_frames(video_path: Path, num_frames: int = 16,
                        strategy: str = "uniform",
                        size: tuple[int, int] | None = (224, 224)) -> list[np.ndarray]:
    """Return a list of sampled frames (BGR np arrays) for one clip."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if strategy == "pixel":
        cap.release()
        indices = _pixel_change_indices(video_path, num_frames)
        cap = cv2.VideoCapture(str(video_path))
    else:
        indices = _uniform_indices(total, num_frames)

    wanted = set(indices)
    grabbed: dict[int, np.ndarray] = {}
    i = 0
    while wanted - grabbed.keys():
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            if size:
                frame = cv2.resize(frame, size)
            grabbed[i] = frame
        i += 1
    cap.release()

    frames = [grabbed[i] for i in indices if i in grabbed]
    # pad by repeating last frame if the clip was shorter than num_frames
    while frames and len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="a split JSON from annotations/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--strategy", choices=["uniform", "pixel"], default="uniform")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    clips = json.loads(Path(args.split).read_text())["clips"]
    out_root = Path(args.out)
    done = 0
    for c in clips:
        vp = ROOT / c["file_path"]
        if not vp.exists():
            print(f"[frames] missing {vp}")
            continue
        frames = extract_clip_frames(vp, args.num_frames, args.strategy,
                                     (args.size, args.size))
        dst = out_root / c["class_name"] / vp.stem
        dst.mkdir(parents=True, exist_ok=True)
        for j, fr in enumerate(frames):
            cv2.imwrite(str(dst / f"frame_{j:02d}.jpg"), fr)
        done += 1
    print(f"[frames] extracted {args.num_frames} {args.strategy} frames for "
          f"{done}/{len(clips)} clips -> {out_root}")


if __name__ == "__main__":
    main()
