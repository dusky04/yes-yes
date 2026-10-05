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
Opens each video in the default player (Windows/macOS/Linux), then asks (strokes):
  chirality?             r = right-handed  l = left-handed  u = unknown
  camera_perspective?    pick a number from the menu shown  u = unknown
  start_time_of_stroke?  seconds from the clip start (0 .. clip duration)
  end_time_of_stroke?    seconds, greater than start, up to the clip duration
Outcome clips are asked only for camera_perspective (the other fields stay null).
Shortcuts:  [Enter] = reuse the previous answer (where offered)   s = skip   q = save & quit

Output: annotations/manual_annotations.csv. Run merge_annotations.py afterwards.

------------------------------------------------------------------------------
PROFESSOR: confirm these two lists match your categories. Edit here if needed —
keep spelling EXACT; a typo becomes a new, separate category.

Camera-perspective vocabulary (as per the paper):
  * Stroke clips are batsman-oriented views.
  * Outcome clips are field / event-oriented views.
"""
STROKE_PERSPECTIVES = [
    "Batsman-Centric",
    "Batsman / High-Angle Tracking",
]
OUTCOME_PERSPECTIVES = [
    "High-Angle Ball Tracking",
    "Ground Field Tracking",
    "Pitch / Stump Close-up",
    "Fielder / Boundary View",
    "Pitch / Umpire Perspective",
]
# ------------------------------------------------------------------------------

import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as C

OUT_CSV = C.ANNOTATIONS_DIR / "manual_annotations.csv"
PROGRESS_FILE = C.ANNOTATIONS_DIR / "annotation_progress.json"
CHIRALITY = {"r": "right-handed", "l": "left-handed", "u": None}

# CSV schema (order matters). start/end time are stroke-only; blank -> null.
CSV_FIELDS = ["clip_id", "file_path", "chirality", "camera_perspective",
              "start_time_of_stroke", "end_time_of_stroke"]


def migrate_csv():
    """If an older manual_annotations.csv (without the time columns) exists,
    rewrite it to the current 6-column schema, leaving the new fields blank for
    rows already recorded. Preserves every existing annotation."""
    if not OUT_CSV.exists():
        return
    rows = list(csv.DictReader(OUT_CSV.open()))
    header_ok = rows == [] or all(f in (rows[0].keys()) for f in CSV_FIELDS)
    if header_ok:
        return
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"[migrate] upgraded {OUT_CSV.name} to include start/end time columns "
          f"({len(rows)} existing row(s) preserved).")


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
    """Every clip across the three splits, de-duplicated by file_path, sorted.
    Includes duration_sec so the time prompts can show/validate the clip length."""
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
                          "layer": c["semantic_layer"],
                          "duration_sec": float(c.get("duration_sec") or 0.0)})
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
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_progress(clips, done):
    """Persist a progress record so completion can be checked without re-reading
    every file. Written after each clip and on quit."""
    done_fps = set(done)
    done_stroke = sum(1 for c in clips
                      if c["file_path"] in done_fps and c["layer"] == C.SEMANTIC_STROKE)
    total = len(clips)
    n_done = sum(1 for c in clips if c["file_path"] in done_fps)
    rec = {
        "total_clips": total,
        "completed": n_done,
        "remaining": total - n_done,
        "percent_complete": round(100 * n_done / total, 1) if total else 0.0,
        "completed_stroke": done_stroke,
        "completed_outcome": n_done - done_stroke,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }
    PROGRESS_FILE.write_text(json.dumps(rec, indent=2))
    return rec


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


def ask_time(prompt, lo, hi):
    """Ask for a time in seconds within [lo, hi]. Returns a float, or
    's'/'q'/'' (unknown). Validates numeric input and the range."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("q", "s"):
            return ans
        if ans in ("", "u"):          # unknown -> blank
            return ""
        try:
            v = float(ans)
        except ValueError:
            print("   please enter a number of seconds (or u/s/q)")
            continue
        if v < lo or (hi > 0 and v > hi + 1e-6):
            print(f"   must be between {lo:.2f} and {hi:.2f} seconds")
            continue
        return v


def main():
    clips = all_clips()
    if not clips:
        sys.exit("No clips found. Run the pipeline first (need the JSONs).")
    done = load_done()
    todo = [c for c in clips if c["file_path"] not in done]
    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    migrate_csv()   # upgrade an older 4-column CSV in place, if present

    # progress banner on load (also persisted to annotation_progress.json)
    rec = save_progress(clips, done)
    n_stroke_todo = sum(1 for c in todo if c["layer"] == C.SEMANTIC_STROKE)
    print("=" * 56)
    print(f"  ANNOTATION PROGRESS")
    print(f"  completed : {rec['completed']}/{rec['total_clips']} "
          f"({rec['percent_complete']}%)   "
          f"[stroke {rec['completed_stroke']} / outcome {rec['completed_outcome']}]")
    print(f"  remaining : {rec['remaining']}  "
          f"({n_stroke_todo} stroke / {len(todo)-n_stroke_todo} outcome)")
    if rec["completed"]:
        print(f"  last saved: {rec['last_updated']}  (resuming where you left off)")
    print("=" * 56)
    if not todo:
        print("All clips are already annotated. Nothing to do.")
        return

    # per-category memory for [Enter] reuse
    prev = {"ch": None, C.SEMANTIC_STROKE: None, C.SEMANTIC_OUTCOME: None}

    for i, c in enumerate(todo, 1):
        path = ROOT / c["file_path"]
        is_stroke = c["layer"] == C.SEMANTIC_STROKE
        persp_list = STROKE_PERSPECTIVES if is_stroke else OUTCOME_PERSPECTIVES
        print(f"\n[{i}/{len(todo)}  ·  {rec['completed']+i-1}/{rec['total_clips']} overall]"
              f"  {c['clip_id']}  ({c['class_name']}, {c['layer']})  {c['file_path']}")
        if path.exists():
            open_video(path)
        else:
            print("   (video not found on disk — labelling blind)")

        chir_val = ""          # outcomes: stays blank -> null
        start_val = ""         # start/end time: stroke-only -> blank for outcomes
        end_val = ""
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

        # start/end time of the stroke (stroke clips only), in seconds from the
        # clip start, within [0, clip duration].
        if is_stroke:
            dur = c["duration_sec"]
            rng = f"0–{dur:.2f}s" if dur else "seconds"
            start_val = ask_time(
                f"   start_time_of_stroke ({rng})? [u=unknown, s=skip, q=quit]: ",
                0.0, dur)
            if start_val == "q":
                break
            if start_val == "s":
                continue
            lo_end = (start_val + 1e-6) if isinstance(start_val, float) else 0.0
            end_val = ask_time(
                f"   end_time_of_stroke (>start, ≤{dur:.2f}s)? "
                f"[u=unknown, s=skip, q=quit]: ", lo_end, dur)
            if end_val == "q":
                break
            if end_val == "s":
                continue

        append_row({"clip_id": c["clip_id"], "file_path": c["file_path"],
                    "chirality": chir_val, "camera_perspective": persp,
                    "start_time_of_stroke": start_val,
                    "end_time_of_stroke": end_val},
                   write_header=not OUT_CSV.exists())
        done[c["file_path"]] = True
        save_progress(clips, done)   # update the persisted record after every clip

    final = save_progress(clips, done)
    print(f"\nSaved {final['completed']}/{final['total_clips']} "
          f"({final['percent_complete']}%) to {OUT_CSV.name}.")
    print(f"Progress recorded in {PROGRESS_FILE.name}. "
          f"Re-run to continue; then: python3 merge_annotations.py")


if __name__ == "__main__":
    main()
