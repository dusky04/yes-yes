from __future__ import annotations
"""
merge_annotations.py — fold manual_annotations.csv into the three split JSONs.

Reads annotations/manual_annotations.csv (produced by annotate.py), and for
every clip whose file_path appears there, fills in `chirality`,
`camera_perspective`, `start_time_of_stroke` and `end_time_of_stroke` (blank
cells become null; time cells are parsed as floats). Clips not yet annotated keep
their null values, so you can merge partway through and again later. The two time
keys and the two annotation keys are ensured to exist on every clip, so a single
merge run also upgrades older JSONs that predate these fields.

Then it regenerates full_taxonomy_metadata.csv so `dominant_perspective` fills in.

Run:  python3 merge_annotations.py
Then commit & push:  git add -A && git commit -m "Add annotations" && git push
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


def _num(x):
    """Parse a time cell: blank -> None, else float."""
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def main():
    if not MANUAL_CSV.exists():
        sys.exit(f"[merge] no {MANUAL_CSV.name} yet. Run annotate.py first.")

    ann = {}
    for r in csv.DictReader(MANUAL_CSV.open()):
        ann[r["file_path"]] = {
            "chirality": (r.get("chirality") or "").strip() or None,
            "camera_perspective": (r.get("camera_perspective") or "").strip() or None,
            "start_time_of_stroke": _num(r.get("start_time_of_stroke")),
            "end_time_of_stroke": _num(r.get("end_time_of_stroke")),
        }

    total_filled = 0
    for jf in (C.TRAIN_JSON, C.VAL_JSON, C.TEST_JSON):
        if not jf.exists():
            continue
        doc = json.loads(jf.read_text())
        filled = 0
        for c in doc["clips"]:
            # ensure the keys exist on every clip (upgrades older JSONs)
            c.setdefault("start_time_of_stroke", None)
            c.setdefault("end_time_of_stroke", None)
            if c["file_path"] in ann:
                a = ann[c["file_path"]]
                c["chirality"] = a["chirality"]
                c["camera_perspective"] = a["camera_perspective"]
                c["start_time_of_stroke"] = a["start_time_of_stroke"]
                c["end_time_of_stroke"] = a["end_time_of_stroke"]
                if any(a.values()):
                    filled += 1
        jf.write_text(json.dumps(doc, indent=2))
        total_filled += filled
        print(f"[merge] {jf.name}: filled {filled} clip(s)")

    print(f"[merge] total {total_filled} clip-annotations applied.")

    # refresh the taxonomy CSV so dominant_perspective recomputes
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "06_generate_taxonomy_csv.py")])
    if r.returncode == 0:
        print("[merge] taxonomy CSV regenerated. Now: git add -A && git commit && git push")


if __name__ == "__main__":
    main()
