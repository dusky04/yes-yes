from __future__ import annotations
"""
reset_class.py — undo the annotations for one or more classes.

Use it to discard a test/throwaway class (e.g. cover) so it is NOT in the dataset:
it removes that class's rows from annotations/manual_annotations.csv AND sets the
class's clips back to null in the split JSONs (chirality, camera_perspective and
all timing fields). Other classes are untouched.

USAGE
  python3 reset_class.py --classes cover            # reset one class
  python3 reset_class.py --classes cover,defense    # reset several

A timestamped backup of the CSV and each JSON is written next to the original
before anything is changed, so you can roll back if needed.
"""
import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C

CSV_PATH = C.ANNOTATIONS_DIR / "manual_annotations.csv"
NULL_KEYS = ["chirality", "camera_perspective",
             "start_frame", "end_frame", "start_time_ms", "end_time_ms"]
OLD_KEYS = ["start_time_of_stroke", "end_time_of_stroke"]
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def backup(p: Path):
    b = p.with_suffix(p.suffix + f".bak-{STAMP}")
    shutil.copy2(p, b)
    return b.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", required=True,
                    help="comma-separated class names to reset (e.g. cover).")
    args = ap.parse_args()
    names = {s.strip() for s in args.classes.split(",") if s.strip()}
    valid = set(C.ALL_CLASSES)
    bad = names - valid
    if bad:
        sys.exit(f"[reset] unknown class name(s): {', '.join(sorted(bad))}\n"
                 f"        pick from: {', '.join(C.ALL_CLASSES)}")
    folders = tuple(f"/{n}/" for n in names)

    # 1) drop the class's rows from the CSV
    if CSV_PATH.exists():
        rows = list(csv.DictReader(CSV_PATH.open()))
        keep = [r for r in rows if not any(f in r.get("file_path", "") for f in folders)]
        dropped = len(rows) - len(keep)
        if dropped:
            print(f"[reset] CSV backup: {backup(CSV_PATH)}")
            with CSV_PATH.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(keep)
        print(f"[reset] manual_annotations.csv: removed {dropped} row(s) for {', '.join(sorted(names))}")

    # 2) null the class's clips in each split JSON
    for jf in (C.TRAIN_JSON, C.VAL_JSON, C.TEST_JSON):
        if not jf.exists():
            continue
        doc = json.loads(jf.read_text())
        n = 0
        for c in doc["clips"]:
            if c.get("class_name") in names:
                for k in OLD_KEYS:
                    c.pop(k, None)
                for k in NULL_KEYS:
                    c[k] = None
                n += 1
        if n:
            print(f"[reset] JSON backup: {backup(jf)}")
            jf.write_text(json.dumps(doc, indent=2))
        print(f"[reset] {jf.name}: reset {n} clip(s) to null")

    print(f"[reset] done. {', '.join(sorted(names))} is no longer in the dataset.")


if __name__ == "__main__":
    main()
