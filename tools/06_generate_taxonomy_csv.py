from __future__ import annotations
"""
Stage 06 — build annotations/full_taxonomy_metadata.csv (the paper's Table 2).

One row per class, columns exactly as the professor defined them:
    class_id, class_name, semantic_layer, total_clips, train_clips, val_clips,
    total_duration_sec, mean_duration_sec, std_duration_sec, fps,
    dominant_perspective

Everything numeric is aggregated straight from the two JSONs, so the CSV can
never disagree with the annotations — same source. dominant_perspective depends
on camera_perspective, which is an annotation field, so it's left blank until
that pass is done; the column exists so Table 2 has its shape from day one.

Run:  python3 tools/06_generate_taxonomy_csv.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C


def load_clips(path):
    if not path.exists():
        sys.exit(f"[06] missing {path.name}. Run tools/05 first.")
    return json.loads(path.read_text())["clips"]


def main():
    train = load_clips(C.TRAIN_JSON)
    val = load_clips(C.VAL_JSON)
    all_clips = train + val

    # index by class for counts/durations
    train_count = Counter(c["class_name"] for c in train)
    val_count = Counter(c["class_name"] for c in val)
    by_class = {}
    for c in all_clips:
        by_class.setdefault(c["class_name"], []).append(c)

    rows = []
    for cid in sorted(C.ID_TO_NAME):
        name = C.ID_TO_NAME[cid]
        clips = by_class.get(name, [])
        durations = pd.Series([c["duration_sec"] for c in clips], dtype="float64")
        fpss = [c["fps"] for c in clips]
        dominant_fps = Counter(fpss).most_common(1)[0][0] if fpss else ""
        # dominant_perspective: annotation field -> blank until filled
        persps = [c.get("camera_perspective") for c in clips
                  if c.get("camera_perspective")]
        dominant_persp = Counter(persps).most_common(1)[0][0] if persps else ""
        rows.append({
            "class_id": cid,
            "class_name": name,
            "semantic_layer": C.NAME_TO_LAYER[name],
            "total_clips": len(clips),
            "train_clips": train_count.get(name, 0),
            "val_clips": val_count.get(name, 0),
            "total_duration_sec": round(float(durations.sum()), 2) if len(clips) else 0.0,
            "mean_duration_sec": round(float(durations.mean()), 2) if len(clips) else 0.0,
            "std_duration_sec": round(float(durations.std(ddof=0)), 2) if len(clips) else 0.0,
            "fps": dominant_fps,
            "dominant_perspective": dominant_persp,
        })

    df = pd.DataFrame(rows, columns=[
        "class_id", "class_name", "semantic_layer", "total_clips",
        "train_clips", "val_clips", "total_duration_sec", "mean_duration_sec",
        "std_duration_sec", "fps", "dominant_perspective"])
    df.to_csv(C.TAXONOMY_CSV, index=False)

    # Console summary incl. a crude class-imbalance ratio.
    nonzero = df[df.total_clips > 0]["total_clips"]
    print(f"[06] wrote {C.TAXONOMY_CSV.name} — {len(df)} classes, "
          f"{int(df.total_clips.sum())} clips")
    if len(nonzero):
        print(f"[06] class imbalance (max/min clips): "
              f"{nonzero.max()}/{nonzero.min()} = {nonzero.max()/nonzero.min():.1f}x")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
