from __future__ import annotations
"""
annotate.py — watch each clip and label camera_perspective (+ chirality for strokes).

TWO CATEGORIES, each with its own camera_perspective menu:
  * STROKE clips (the 10 batting shots): asks chirality AND a batting-view
    perspective.
  * OUTCOME clips (catch/four/six/wicket): asks only an outcome-view perspective;
    chirality stays null (handedness is scoped to batting shots).

The script picks the correct menu automatically from each clip's semantic_layer,
so the professor only ever sees valid options and can't mislabel across categories.

Saves after every clip — press q to stop and resume later; done clips are skipped.

HOW THE PROFESSOR USES IT:
  python3 annotate.py        (on Windows this may be:  python annotate.py)
Opens each video in the default player (Windows/macOS/Linux), then asks:
  chirality?            r = right-handed  l = left-handed  u = unknown   (strokes only)
  camera_perspective?   pick a number from the menu shown  u = unknown
Shortcuts:  [Enter] = reuse the previous answer (per category)   s = skip   q = save & quit

Output: annotations/manual_annotations.csv. Run merge_annotations.py afterwards.

------------------------------------------------------------------------------
PROFESSOR: confirm these two lists match your categories. Edit here if needed —
keep spelling EXACT; a typo becomes a new, separate category.
"""
STROKE_PERSPECTIVES = [
    "front_on",        # bowler's-end main camera, batsman facing
    "side_on",         # square of the wicket — best for swing mechanics
    "reverse_angle",   # from behind the batsman (keeper's end)
    "high_angle",      # elevated broadcast view
]
OUTCOME_PERSPECTIVES = [
    "outfield_wide",
    "main_pitch",
    "close_up_replay",
    "side_on",
]
# ------------------------------------------------------------------------------

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C

OUT_CSV = C.ANNOTATIONS_DIR / "manual_annotations.csv"
CHIRALITY = {"r": "right-handed", "l": "left-handed", "u": None}


def open_video(path: Path):
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)])
        elif sys.platform.startswith("win"):
            os.startfile(str(path))                       # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", str(path)])
    except Exception as e:                                # noqa: BLE001
        print(f"   (couldn't auto-open — open it manually: {path}  [{e}])")


def all_clips():
    """Every clip across the three splits, de-duplicated by file_path, sorted."""
    seen, clips = set(), []
    for jf in (C.TRAIN_JSON, C.VAL_JSON, C.TEST_JSON):
        if not jf.exists():
            continue
        for c in json.loads(jf.read_text())["clips"]:
            if c["file_path"] in seen:
                continue
            seen.add(c["file_path"])
            clips.append({"clip_id": c["clip_id"], "file_path": c["file_path"],
                          "class_name": c["class_name"],
                          "layer": c["semantic_layer"]})
    clips.sort(key=lambda x: x["file_path"])
    return clips


def load_done():
    done = {}
    if OUT_CSV.exists():
        for r in csv.DictReader(OUT_CSV.open()):
            done[r["file_path"]] = r
    return done


def append_row(row, write_header):
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "file_path",
                                          "chirality", "camera_perspective"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def ask(prompt, valid, prev):
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("q", "s"):
            return ans
        if ans == "" and prev is not None:
            return prev
        if ans in valid:
            return ans
        print(f"   please type one of: {'/'.join(sorted(valid))}  (or Enter/s/q)")


def main():
    clips = all_clips()
    if not clips:
        sys.exit("No clips found. Run the pipeline first (need the JSONs).")
    done = load_done()
    todo = [c for c in clips if c["file_path"] not in done]
    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    n_stroke = sum(1 for c in todo if c["layer"] == C.SEMANTIC_STROKE)
    print(f"{len(clips)} clips total, {len(done)} done, {len(todo)} to go "
          f"({n_stroke} stroke / {len(todo)-n_stroke} outcome).")

    # per-category memory for [Enter] reuse
    prev = {"ch": None, C.SEMANTIC_STROKE: None, C.SEMANTIC_OUTCOME: None}
    quit_now = False

    for i, c in enumerate(todo, 1):
        if quit_now:
            break
        path = ROOT / c["file_path"]
        is_stroke = c["layer"] == C.SEMANTIC_STROKE
        persp_list = STROKE_PERSPECTIVES if is_stroke else OUTCOME_PERSPECTIVES
        print(f"\n[{i}/{len(todo)}] {c['clip_id']}  ({c['class_name']}, {c['layer']})"
              f"  {c['file_path']}")
        if path.exists():
            open_video(path)
        else:
            print("   (video not found on disk — labelling blind)")

        chir_val = ""   # outcomes: stays blank -> null
        if is_stroke:
            ch = ask(f"   chirality? r/l/u [{prev['ch'] or '-'}]: ",
                     set(CHIRALITY), prev["ch"])
            if ch == "q":
                break
            if ch == "s":
                continue
            prev["ch"] = ch
            chir_val = CHIRALITY[ch] or ""

        menu = "  ".join(f"{j+1}={p}" for j, p in enumerate(persp_list))
        print(f"   perspective menu ({c['layer']}): {menu}   u=unknown")
        p = ask(f"   camera_perspective? 1-{len(persp_list)}/u [{prev[c['layer']] or '-'}]: ",
                {str(j + 1) for j in range(len(persp_list))} | {"u"}, prev[c["layer"]])
        if p == "q":
            break
        if p == "s":
            continue
        prev[c["layer"]] = p
        persp = "" if p == "u" else persp_list[int(p) - 1]

        append_row({"clip_id": c["clip_id"], "file_path": c["file_path"],
                    "chirality": chir_val, "camera_perspective": persp},
                   write_header=not OUT_CSV.exists())

    print(f"\nSaved to {OUT_CSV}. Re-run to continue; then: python3 merge_annotations.py")


if __name__ == "__main__":
    main()
