from __future__ import annotations
"""
Stage 05 — emit the two annotation JSONs in the professor's exact schema.

    annotations/train_split_match_disjoint.json
    annotations/val_split_match_disjoint.json

Root object:
    split_name, dataset_name, version, total_clips, clips[]
Each clip:
    clip_id, file_path, class_name, class_id, semantic_layer,
    source_match_id, duration_sec, num_frames, fps, resolution,
    chirality, camera_perspective

Fields this script fills from real data: clip_id, file_path, class_name,
class_id, semantic_layer, source_match_id, duration_sec, num_frames, fps,
resolution. The two purely-human-judgement fields — chirality (batter's
handedness) and camera_perspective — are written as null, because they can't
be derived from the file and belong to the annotation pass. They're present in
the schema so nothing downstream breaks; they just wait to be filled.

clip_id is assigned per split in sorted order: CEC_TR_0001.., CEC_VAL_0001..

Run:  python3 tools/05_generate_annotations.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SPLIT_CACHE = C.ANNOTATIONS_DIR / "_split_assignments.csv"


def load_rows():
    if not C.METADATA_CACHE.exists():
        sys.exit("[05] missing metadata cache. Run tools/03 first.")
    if not SPLIT_CACHE.exists():
        sys.exit("[05] missing split assignments. Run tools/04 first.")
    with C.METADATA_CACHE.open() as f:
        meta = {r["file_path"]: r for r in csv.DictReader(f)}
    split, match = {}, {}
    with SPLIT_CACHE.open() as f:
        for r in csv.DictReader(f):
            split[r["clip_stem"]] = r["split"]
            match[r["clip_stem"]] = r["source_match_id"]
    return meta, split, match


def build_clip(clip_id, r, source_match_id):
    return {
        "clip_id": clip_id,
        "file_path": r["file_path"],
        "class_name": r["class_name"],
        "class_id": int(r["class_id"]),
        "semantic_layer": r["semantic_layer"],
        "source_match_id": source_match_id,
        "duration_sec": float(r["duration_sec"]),
        "num_frames": int(r["num_frames"]),
        "fps": float(r["fps"]),
        "resolution": [int(r["width"]), int(r["height"])],
        # annotation pass fills these:
        "chirality": None,
        "camera_perspective": None,
        "start_time_of_stroke": None,   # stroke clips only (seconds from clip start)
        "end_time_of_stroke": None,     # stroke clips only (seconds from clip start)
    }


def write_split(split_name, split_key, meta, split, match):
    stems = sorted(s for s, v in split.items() if v == split_key)
    # map stem -> metadata row
    by_stem = {Path(r["file_path"]).stem: r for r in meta.values()}
    clips = []
    for i, stem in enumerate(stems, start=1):
        clip_id = f"{C.CLIPID_PREFIX[split_key]}{i:04d}"
        mid = match.get(stem) or None      # empty string -> null
        clips.append(build_clip(clip_id, by_stem[stem], mid))
    doc = {
        "split_name": split_name,
        "dataset_name": C.DATASET_NAME,
        "version": C.DATASET_VERSION,
        "total_clips": len(clips),     # always equals the real array length
        "clips": clips,
    }
    out = C.SPLIT_JSON[split_key]
    with out.open("w") as f:
        json.dump(doc, f, indent=2)
    assert doc["total_clips"] == len(doc["clips"])
    print(f"[05] {out.name}: {len(clips)} clips")
    return out


def main():
    meta, split, match = load_rows()
    write_split("train_match_disjoint", "train", meta, split, match)
    write_split("val_match_disjoint", "val", meta, split, match)
    write_split("test_match_disjoint", "test", meta, split, match)
    print("[05] annotations written. chirality & camera_perspective left null "
          "for the annotation pass.")


if __name__ == "__main__":
    main()
