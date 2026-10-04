from __future__ import annotations
"""
merge_annotations.py — fold manual_annotations.csv into the three split JSONs.

Reads annotations/manual_annotations.csv (produced by annotate.py), and for
every clip whose file_path appears there, fills in `chirality` and
`camera_perspective` (blank cells become null). Clips not yet annotated keep
their null values, so you can merge partway through and again later.

Then it regenerates full_taxonomy_metadata.csv so `dominant_perspective` fills in.

Run:  python3 merge_annotations.py
Then commit & push:  git add -A && git commit -m "Add chirality/camera_perspective annotations" && git push
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


def main():
    if not MANUAL_CSV.exists():
        sys.exit(f"[merge] no {MANUAL_CSV.name} yet. Run annotate.py first.")

    ann = {}
    for r in csv.DictReader(MANUAL_CSV.open()):
        ann[r["file_path"]] = (
            (r.get("chirality") or "").strip() or None,
            (r.get("camera_perspective") or "").strip() or None,
        )

    total_filled = 0
    for jf in (C.TRAIN_JSON, C.VAL_JSON, C.TEST_JSON):
        if not jf.exists():
            continue
        doc = json.loads(jf.read_text())
        filled = 0
        for c in doc["clips"]:
            if c["file_path"] in ann:
                ch, persp = ann[c["file_path"]]
                c["chirality"] = ch
                c["camera_perspective"] = persp
                if ch or persp:
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
