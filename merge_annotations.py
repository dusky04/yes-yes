from __future__ import annotations
"""
merge_annotations.py — fold manual_annotations.csv AND lbw_annotations.csv into
the three split JSONs.

For every clip whose file_path appears in an annotation CSV, this fills:
  chirality, camera_perspective,
  start_frame, end_frame, start_time_ms, end_time_ms,
  start_time_of_stroke, end_time_of_stroke  (seconds, derived from *_time_ms)
and, for LBW clips only:
  bowler_handedness, qc_status

The annotation CSVs use the frame/millisecond schema produced by annotate.py v3
and annotate_lbw.py (start_frame/end_frame/start_time_ms/end_time_ms). The
seconds fields (start_time_of_stroke/end_time_of_stroke) are kept for backward
compatibility and computed from the millisecond values.

Every clip is guaranteed to have all these keys (null when not yet annotated),
so you can merge partway through and again later, and older JSONs are upgraded.

Run:  python3 merge_annotations.py
Then: git add -A && git commit -m "Add annotations" && git push
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C

MANUAL_CSV = C.ANNOTATIONS_DIR / "manual_annotations.csv"
LBW_CSV = C.ANNOTATIONS_DIR / "lbw_annotations.csv"

# keys ensured on every clip (null until filled)
ANNO_KEYS = ["chirality", "camera_perspective",
             "start_frame", "end_frame", "start_time_ms", "end_time_ms",
             "start_time_of_stroke", "end_time_of_stroke",
             "bowler_handedness", "qc_status"]


def _int(x):
    x = (x or "").strip()
    if not x:
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def _ms_to_sec(x):
    v = _int(x)
    return round(v / 1000.0, 3) if v is not None else None


def _str(x):
    return (x or "").strip() or None


def _row(r, is_lbw):
    return {
        "chirality": _str(r.get("chirality")),
        "camera_perspective": _str(r.get("camera_perspective")),
        "start_frame": _int(r.get("start_frame")),
        "end_frame": _int(r.get("end_frame")),
        "start_time_ms": _int(r.get("start_time_ms")),
        "end_time_ms": _int(r.get("end_time_ms")),
        "start_time_of_stroke": _ms_to_sec(r.get("start_time_ms")),
        "end_time_of_stroke": _ms_to_sec(r.get("end_time_ms")),
        "bowler_handedness": _str(r.get("bowler_handedness")) if is_lbw else None,
        "qc_status": _str(r.get("qc_status")) if is_lbw else None,
    }


def load_csv(path, is_lbw):
    ann = {}
    if path.exists():
        for r in csv.DictReader(path.open()):
            fp = (r.get("file_path") or "").strip().replace("\\", "/")
            if fp:
                ann[fp] = _row(r, is_lbw)
    return ann


def main():
    if not MANUAL_CSV.exists() and not LBW_CSV.exists():
        sys.exit(f"[merge] no {MANUAL_CSV.name} or {LBW_CSV.name} yet.")

    ann = {}
    ann.update(load_csv(MANUAL_CSV, is_lbw=False))
    ann.update(load_csv(LBW_CSV, is_lbw=True))      # lbw file paths are distinct
    print(f"[merge] loaded {len(ann)} annotated clip(s) (manual + lbw).")

    total_filled = 0
    for jf in (C.TRAIN_JSON, C.VAL_JSON, C.TEST_JSON):
        if not jf.exists():
            continue
        doc = json.loads(jf.read_text())
        filled = 0
        for c in doc["clips"]:
            for k in ANNO_KEYS:                     # ensure keys exist on every clip
                c.setdefault(k, None)
            fp = c["file_path"].replace("\\", "/")
            if fp in ann:
                a = ann[fp]
                for k in ANNO_KEYS:
                    c[k] = a[k]
                if any(v is not None for v in a.values()):
                    filled += 1
        jf.write_text(json.dumps(doc, indent=2))
        total_filled += filled
        print(f"[merge] {jf.name}: filled {filled} clip(s)")

    print(f"[merge] total {total_filled} clip-annotations applied.")

    tax = ROOT / "tools" / "06_generate_taxonomy_csv.py"
    if tax.exists():
        r = subprocess.run([sys.executable, str(tax)])
        if r.returncode == 0:
            print("[merge] taxonomy CSV regenerated.")
    print("[merge] done. Now: git add -A && git commit -m 'Add annotations + LBW' && git push")


if __name__ == "__main__":
    main()
