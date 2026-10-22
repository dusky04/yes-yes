from __future__ import annotations
"""
add_lbw_clips.py — add the LBW clips to the existing split JSONs WITHOUT
re-splitting the other 14 classes.

Reads the LBW videos in videos/outcome_classes/lbw/, extracts their real metadata
(num_frames, fps, resolution, duration), splits them into train/val/test with the
SAME seeded, stratified logic the pipeline uses (so it's reproducible and the
existing classes are untouched), and appends them to the three split JSONs as
class_id 14 (semantic_layer "outcome"). Annotation fields are left null — run
merge_annotations.py afterwards to fill chirality / perspective / timing /
bowler_handedness / qc_status from lbw_annotations.csv.

Idempotent: existing lbw clips in the JSONs are removed and re-added, so you can
re-run safely.

Run:  python3 add_lbw_clips.py
Then: python3 merge_annotations.py
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C

LBW_DIR = C.OUTCOME_DIR / "lbw"
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}
ANNO_KEYS = ["chirality", "camera_perspective", "start_frame", "end_frame",
             "start_time_ms", "end_time_ms", "start_time_of_stroke",
             "end_time_of_stroke", "bowler_handedness", "qc_status"]


def video_meta(path: Path):
    """num_frames, fps, [width, height] from the real file."""
    try:
        from decord import VideoReader
        vr = VideoReader(str(path))
        n = len(vr)
        fps = float(vr.get_avg_fps() or 0) or 25.0
        h, w = vr[0].shape[0], vr[0].shape[1]
        return n, fps, [int(w), int(h)]
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 224)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 224)
        cap.release()
        return n, fps, [w, h]


def main():
    if "lbw" not in C.OUTCOME_CLASSES:
        sys.exit("[add_lbw] config.py OUTCOME_CLASSES has no 'lbw' — add it first (id 14).")
    if not LBW_DIR.exists():
        sys.exit(f"[add_lbw] no LBW folder at {LBW_DIR}")
    vids = sorted(p for p in LBW_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        sys.exit(f"[add_lbw] no videos in {LBW_DIR}")
    lbw_id = C.NAME_TO_ID["lbw"]
    print(f"[add_lbw] {len(vids)} lbw videos, class_id={lbw_id}")

    # seeded stratified split of the lbw clips (same seed/fractions as the pipeline)
    stems = [v.stem for v in vids]
    rng = random.Random(C.SPLIT_SEED)
    rng.shuffle(stems)
    n = len(stems)
    n_val = max(1, round(C.VAL_FRACTION * n))
    n_test = max(1, round(C.TEST_FRACTION * n))
    n_val = min(n_val, n - 1)
    n_test = min(n_test, n - 1 - n_val)
    assign = {}
    for i, s in enumerate(stems):
        assign[s] = "val" if i < n_val else ("test" if i < n_val + n_test else "train")

    by_name = {v.stem: v for v in vids}
    added = {"train": 0, "val": 0, "test": 0}
    for split_key, jf in C.SPLIT_JSON.items():
        if not jf.exists():
            print(f"[add_lbw] {jf.name} missing — skip")
            continue
        doc = json.loads(jf.read_text())
        # drop any existing lbw clips (idempotent re-run)
        doc["clips"] = [c for c in doc["clips"] if "/lbw/" not in c["file_path"]]
        for s, v in by_name.items():
            if assign[s] != split_key:
                continue
            n_f, fps, res = video_meta(v)
            num = re.sub(r"[^0-9]", "", s) or s
            rec = {
                "clip_id": f"CEC_LBW_{num}",
                "file_path": f"videos/outcome_classes/lbw/{v.name}",
                "class_name": "lbw", "class_id": lbw_id, "semantic_layer": "outcome",
                "source_match_id": None,
                "duration_sec": round(n_f / fps, 2) if fps else None,
                "num_frames": n_f, "fps": fps, "resolution": res,
            }
            for k in ANNO_KEYS:
                rec[k] = None
            doc["clips"].append(rec)
            added[split_key] += 1
        doc["total_clips"] = len(doc["clips"])
        jf.write_text(json.dumps(doc, indent=2))
        print(f"[add_lbw] {jf.name}: +{added[split_key]} lbw  (total now {doc['total_clips']})")

    print(f"[add_lbw] added {sum(added.values())} lbw clips. "
          f"Now run: python3 merge_annotations.py")


if __name__ == "__main__":
    main()
