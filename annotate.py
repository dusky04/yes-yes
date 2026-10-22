from __future__ import annotations
"""
annotate.py — label each clip's chirality, camera_perspective, and (for strokes)
the stroke's temporal boundaries by FRAME, from which millisecond times derive.

TWO CATEGORIES:
  * STROKE clips (the batting shots): chirality + a batting-view perspective +
    the start/end FRAME of the stroke, marked with a frame-accurate scrubber.
  * OUTCOME clips (catch/four/six/wicket — match events): a field/event-view
    perspective only. Chirality and the timing/frame fields stay null. The
    scrubber never opens for outcome clips — timing is not mixed into events.

Why frames, not typed seconds: a batting stroke lasts only a few hundred
milliseconds. A human cannot eyeball that precisely in a normal player, and a
float in seconds already encodes ms, so the limit is the *capture*, not the unit.
The frame is the video's native quantum (at 30 fps, 33 ms each), so marking the
exact start/end frame gives the finest precision the clip contains, and the
millisecond times are derived exactly as  round(frame * 1000 / fps).

LBW is NOT handled here — it is a match event with its own batsman/stump-only
perspective vocabulary, annotated by the separate annotate_lbw.py. Any lbw clip is
skipped by this script so the two tools never write conflicting perspectives.

Saves after every clip — press q to stop and resume later; done clips are skipped.

HOW TO USE:
  python3 annotate.py                       annotate every remaining clip
  python3 annotate.py --classes cover,pull  annotate only these classes (divide work)
  python3 annotate.py --list                per-class done / total / remaining, then exit
  python3 annotate.py --stats               camera_perspective + chirality counts, then exit
  (on Windows these may be:  python annotate.py ...)
For a STROKE clip it asks chirality and camera_perspective in the terminal, then
opens a frame scrubber window:
  a / d        step one frame back / forward
  b / f        jump 5 frames back / forward
  space        play / pause
  s            mark the current frame as the stroke START
  e            mark the current frame as the stroke END
  Enter        confirm (needs both start and end marked)
  u            unknown (leave timing blank)   q  quit session
If no display is available (e.g. a headless machine), it falls back to typing the
start/end time in MILLISECONDS instead.

Output: annotations/manual_annotations.csv. Run merge_annotations.py afterwards.

------------------------------------------------------------------------------
PROFESSOR: confirm these two lists match your categories. Edit here if needed —
keep spelling EXACT; a typo becomes a new, separate category.

Camera-perspective vocabulary (as per the paper):
  * Stroke clips are batsman-oriented views.
  * Outcome clips are field / event-oriented views.
"""
VERSION = "3.0  (scrubber-first — clip opens BEFORE the questions)"

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

import argparse
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

# CSV schema (order matters). Frame/time fields are stroke-only; blank -> null.
CSV_FIELDS = ["clip_id", "file_path", "chirality", "camera_perspective",
              "start_frame", "end_frame", "start_time_ms", "end_time_ms"]


def migrate_csv():
    """Upgrade an older manual_annotations.csv to the current schema in place,
    preserving clip_id/file_path/chirality/camera_perspective for rows already
    recorded and leaving the new frame/time columns blank. Handles the original
    4-column file and the earlier seconds-based 6-column file."""
    if not OUT_CSV.exists():
        return
    rows = list(csv.DictReader(OUT_CSV.open()))
    header_ok = rows == [] or all(f in rows[0].keys() for f in CSV_FIELDS)
    if header_ok:
        return
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"[migrate] upgraded {OUT_CSV.name} to frame/millisecond schema "
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
    """Every clip across the three splits, de-duplicated by file_path.
    Carries fps / num_frames / duration (for the scrubber + ms derivation) AND the
    annotation values already present in the JSONs (camera_perspective, chirality),
    so clips finished in an earlier round — e.g. the outcome clips already merged
    into the JSONs — are recognised as done without needing the per-annotator CSV.
    STROKES are served before outcomes: this is the batting tool, so batting comes
    first instead of making the annotator wade through the finished events."""
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
                          "fps": float(c.get("fps") or 0.0),
                          "num_frames": int(c.get("num_frames") or 0),
                          "duration_sec": float(c.get("duration_sec") or 0.0),
                          # annotation state already in the JSON (may be null)
                          "json_cp": c.get("camera_perspective"),
                          "json_ch": c.get("chirality")})
    # strokes first (layer rank 0), then outcomes (1); stable within each by path
    rank = {C.SEMANTIC_STROKE: 0, C.SEMANTIC_OUTCOME: 1}
    clips.sort(key=lambda x: (rank.get(x["layer"], 9), x["file_path"]))
    return clips


def json_annotated(clip):
    """True if this clip is already labelled in the JSON (so it should be skipped).
    Outcome: a camera_perspective is enough. Stroke: needs chirality AND perspective
    (timing may legitimately be 'unknown', so it is not required)."""
    cp = (clip.get("json_cp") or "")
    ch = (clip.get("json_ch") or "")
    if clip["layer"] == C.SEMANTIC_STROKE:
        return bool(str(cp).strip()) and bool(str(ch).strip())
    return bool(str(cp).strip())


def is_done(clip, done):
    """Done = recorded in this annotator's CSV, OR already labelled in the JSON."""
    return clip["file_path"] in done or json_annotated(clip)


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
    done_stroke = sum(1 for c in clips
                      if is_done(c, done) and c["layer"] == C.SEMANTIC_STROKE)
    total = len(clips)
    n_done = sum(1 for c in clips if is_done(c, done))
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


def _ask_ms(prompt, lo, hi):
    """Ask for an integer millisecond value in [lo, hi]. Returns int or
    's'/'q'/'' (unknown)."""
    while True:
        ans = input(prompt).strip().lower()
        if ans in ("q", "s"):
            return ans
        if ans in ("", "u"):
            return ""
        try:
            v = int(round(float(ans)))
        except ValueError:
            print("   please enter a whole number of milliseconds (or u/s/q)")
            continue
        if v < lo or (hi > 0 and v > hi):
            print(f"   must be between {lo} and {hi} ms")
            continue
        return v


def manual_ms_fallback(dur_ms):
    """No-GUI path: type start/end time in milliseconds. Returns one of
    ("ms", start_ms, end_ms) | ("unknown",) | ("skip",) | ("quit",)."""
    print("   [no video window available — enter the times in MILLISECONDS]")
    s = _ask_ms(f"   start_time_ms (0–{dur_ms})? [u=unknown, s=skip, q=quit]: ",
                0, dur_ms)
    if s == "q":
        return ("quit",)
    if s == "s":
        return ("skip",)
    if s == "":
        return ("unknown",)
    e = _ask_ms(f"   end_time_ms (>{s}, ≤{dur_ms})? [u=unknown, s=skip, q=quit]: ",
                s + 1, dur_ms)
    if e == "q":
        return ("quit",)
    if e == "s":
        return ("skip",)
    if e == "":
        return ("unknown",)
    return ("ms", s, e)


def _has_display():
    """True if a GUI window can be opened. On Windows/macOS desktops this is
    assumed; on Linux it requires an X11/Wayland display. This gate is checked
    BEFORE any cv2 GUI call, because on a headless Linux box with the GUI OpenCV
    build, cv2.imshow can hard-crash the process via Qt rather than raise."""
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def scrub_frames(video_path: Path, fps: float, duration_sec: float):
    """Frame-accurate scrubber for one stroke clip. Returns one of:
       ("ok", start_frame, end_frame) | ("ms", start_ms, end_ms)
       | ("unknown",) | ("skip",) | ("quit",).
    Falls back to millisecond typing when a display / OpenCV GUI is unavailable."""
    dur_ms = int(round((duration_sec or 0.0) * 1000))
    try:
        import cv2
    except Exception:                                     # noqa: BLE001
        return manual_ms_fallback(dur_ms)

    if not _has_display():
        return manual_ms_fallback(dur_ms)

    frames = []
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    if not frames:
        print("   (video not readable here — using millisecond entry)")
        return manual_ms_fallback(dur_ms)

    n = len(frames)
    eff_fps = fps if fps > 0 else (n / duration_sec if duration_sec else 30.0)
    win = "Stroke scrubber"
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, frames[0])
        cv2.waitKey(1)
    except Exception:                                     # noqa: BLE001
        try:
            cv2.destroyAllWindows()
        except Exception:                                 # noqa: BLE001
            pass
        return manual_ms_fallback(dur_ms)

    idx, start, end, playing = 0, None, None, False

    def render():
        img = frames[idx].copy()
        t_ms = int(round(idx * 1000 / eff_fps))
        lines = [
            f"frame {idx+1}/{n}   t={t_ms} ms   fps={eff_fps:.1f}",
            f"START={start if start is not None else '-'}   "
            f"END={end if end is not None else '-'}",
            "a/d step  b/f +-5  space play  s=start  e=end  Enter=ok  u=unknown  q=quit",
        ]
        y = 22
        for ln in lines:
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)
            y += 22
        cv2.imshow(win, img)

    result = ("unknown",)
    while True:
        render()
        k = cv2.waitKey(30 if playing else 0) & 0xFF
        if playing and k == 255:
            idx = min(idx + 1, n - 1)
            if idx >= n - 1:
                playing = False
            continue
        if k in (ord('d'), 83):            # forward
            idx = min(idx + 1, n - 1)
        elif k in (ord('a'), 81):          # back
            idx = max(idx - 1, 0)
        elif k == ord('f'):
            idx = min(idx + 5, n - 1)
        elif k == ord('b'):
            idx = max(idx - 5, 0)
        elif k == 32:                      # space
            playing = not playing
        elif k == ord('s'):
            start = idx
        elif k == ord('e'):
            end = idx
        elif k == ord('u'):
            result = ("unknown",)
            break
        elif k == ord('q'):
            result = ("quit",)
            break
        elif k in (13, 10):                # Enter
            if start is None or end is None:
                print("   mark both START (s) and END (e) before Enter")
                continue
            a, b = sorted((start, end))
            result = ("ok", a, b)
            break
    try:
        cv2.destroyWindow(win)
        cv2.waitKey(1)
    except Exception:                                     # noqa: BLE001
        pass
    return result


def _assignable_classes():
    """Every class annotate.py can serve: all strokes + outcomes except lbw
    (lbw is handled by the separate annotate_lbw.py)."""
    return C.STROKE_CLASSES + [o for o in C.OUTCOME_CLASSES if o != "lbw"]


def print_class_table(clips, done):
    """Per-class remaining counts, so the work can be divided up."""
    order = {name: i for i, name in enumerate(C.ALL_CLASSES)}
    by_class = {}
    for c in clips:
        d = by_class.setdefault(c["class_name"], {"total": 0, "done": 0, "layer": c["layer"]})
        d["total"] += 1
        if is_done(c, done):
            d["done"] += 1
    print(f"{'class':<14}{'layer':<9}{'done':>6}{'total':>7}{'left':>7}")
    print("-" * 43)
    for name in sorted(by_class, key=lambda x: order.get(x, 99)):
        d = by_class[name]
        print(f"{name:<14}{d['layer']:<9}{d['done']:>6}{d['total']:>7}"
              f"{d['total']-d['done']:>7}")


def print_perspective_stats(clips):
    """camera_perspective + chirality distribution, split by layer.

    Counts from the JSONs (the merged source of truth — so finished outcome clips
    already merged in are included) and then OVERLAYS any rows from this annotator's
    CSV that are newer/not-yet-merged. This is why the finished events show up even
    when the local CSV is fresh. This is the count the paper's figure needs.
    """
    from collections import Counter, defaultdict
    layer_of = {c["file_path"]: c["layer"] for c in clips}

    # 1) start from the JSON annotation state carried on each clip
    label = {}   # file_path -> (layer, camera_perspective, chirality)
    for c in clips:
        cp = (c.get("json_cp") or "").strip()
        ch = (c.get("json_ch") or "").strip()
        if cp or ch:
            label[c["file_path"]] = [c["layer"], cp, ch]

    # 2) overlay the CSV (an annotator's latest work, maybe not merged yet)
    if OUT_CSV.exists():
        for r in csv.DictReader(OUT_CSV.open()):
            fp = r.get("file_path", "")
            lay = layer_of.get(fp)
            if lay is None:
                lay = "outcome (lbw)" if "outcome_classes/lbw/" in fp else "unknown"
            label[fp] = [lay,
                         (r.get("camera_perspective") or "").strip(),
                         (r.get("chirality") or "").strip()]

    persp = defaultdict(Counter)
    chir = defaultdict(Counter)
    for lay, cp, ch in label.values():
        persp[lay][cp or "(blank/null)"] += 1
        chir[lay][ch or "(blank)"] += 1
    n_rows = len(label)

    print(f"camera_perspective + chirality counts  ({n_rows} annotated clip(s))")
    if not n_rows:
        print("  (nothing annotated yet)")
        return
    for lay in ("stroke", "outcome", "outcome (lbw)", "unknown"):
        if lay not in persp:
            continue
        print(f"\n=== {lay.upper()} — camera_perspective ===")
        for v, n in persp[lay].most_common():
            print(f"  {n:>4}  {v}")
        # chirality is only meaningful for strokes; show it where present
        nonblank_ch = {k: v for k, v in chir[lay].items() if k != "(blank)"}
        if nonblank_ch:
            print(f"  -- {lay} chirality --")
            for v, n in chir[lay].most_common():
                print(f"  {n:>4}  {v}")


def main():
    ap = argparse.ArgumentParser(
        description="CricketEC batting/outcome annotation (ms stroke timing).")
    ap.add_argument("--classes", help="comma-separated class names to annotate "
                    "(e.g. cover,pull,sweep). Default: all except lbw.")
    ap.add_argument("--list", action="store_true",
                    help="print per-class done/remaining counts and exit.")
    ap.add_argument("--stats", action="store_true",
                    help="print camera_perspective + chirality counts and exit.")
    ap.add_argument("--type-ms", dest="type_ms", action="store_true",
                    help="skip the scrubber window; type the stroke start/end time "
                    "in milliseconds instead (use if the scrubber keys don't work).")
    args = ap.parse_args()

    print(f"annotate.py  version {VERSION}")

    clips = all_clips()
    if not clips:
        sys.exit("No clips found. Run the pipeline first (need the JSONs).")

    # --stats runs on the FULL set (incl. lbw) so every annotated row is counted.
    if args.stats:
        print_perspective_stats(clips)
        return

    # LBW is a match-event handled by the SEPARATE annotate_lbw.py (its camera
    # vocabulary is Batsman-Centric/Unknown only, not the field-event angles this
    # script offers). Exclude lbw clips here so the two tools never collide.
    clips = [c for c in clips if c["class_name"] != "lbw"]

    if args.list:
        print_class_table(clips, load_done())
        return

    # Optional: restrict to a chosen set of classes (for dividing the work).
    if args.classes:
        wanted = [s.strip() for s in args.classes.split(",") if s.strip()]
        valid = set(_assignable_classes())
        bad = [w for w in wanted if w not in valid]
        if bad:
            sys.exit(f"[classes] unknown class name(s): {', '.join(bad)}\n"
                     f"          pick from: {', '.join(_assignable_classes())}")
        wanted_set = set(wanted)
        clips = [c for c in clips if c["class_name"] in wanted_set]
        print(f"[classes] restricted to: {', '.join(wanted)}  "
              f"({len(clips)} clip(s))")

    done = load_done()
    # a clip is skipped if THIS annotator already did it (CSV) OR it is already
    # labelled in the JSON (e.g. the outcome clips merged in an earlier round).
    todo = [c for c in clips if not is_done(c, done)]
    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    migrate_csv()

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

    prev = {"ch": None, C.SEMANTIC_STROKE: None, C.SEMANTIC_OUTCOME: None}

    for i, c in enumerate(todo, 1):
        path = ROOT / c["file_path"]
        is_stroke = c["layer"] == C.SEMANTIC_STROKE
        persp_list = STROKE_PERSPECTIVES if is_stroke else OUTCOME_PERSPECTIVES
        print(f"\n[{i}/{len(todo)}  ·  {rec['completed']+i-1}/{rec['total_clips']} overall]"
              f"  {c['clip_id']}  ({c['class_name']}, {c['layer']})  {c['file_path']}")
        chir_val = ""
        sf = ef = sms = ems = ""     # stroke-only frame/time fields

        if is_stroke:
            # VIEWER FIRST, THEN the questions — so you always see the clip before
            # labelling it. The scrubber uses OpenCV to decode the clip itself, so
            # QuickTime is never involved for strokes (these .avi files often will
            # not open in QuickTime). With --type-ms the clip opens in your default
            # player instead and you type the start/end time in milliseconds.
            if args.type_ms:
                if path.exists():
                    open_video(path)
                res = manual_ms_fallback(int(round((c["duration_sec"] or 0.0) * 1000)))
            else:
                print("   a scrubber window will open with the clip — watch it, then "
                      "mark start/end frame (click the window first so the keys work)...")
                res = scrub_frames(path, c["fps"], c["duration_sec"])
            tag = res[0]
            if tag == "quit":
                break
            if tag == "skip":
                continue
            if tag == "ok":
                sf, ef = res[1], res[2]
                eff = c["fps"] if c["fps"] > 0 else (
                    (c["num_frames"] / c["duration_sec"]) if c["duration_sec"] else 30.0)
                sms = int(round(sf * 1000 / eff))
                ems = int(round(ef * 1000 / eff))
            elif tag == "ms":
                sms, ems = res[1], res[2]      # frames stay blank
            # tag == "unknown" -> timing stays blank

            # now that you've seen the clip in the scrubber, the handedness is easy
            ch = ask(f"   chirality? r/l/u [{prev['ch'] or '-'}]: ",
                     set(CHIRALITY), prev["ch"])
            if ch == "q":
                break
            if ch == "s":
                continue
            prev["ch"] = ch
            chir_val = CHIRALITY[ch] or ""
        else:
            # outcome clips: open in the default player, no scrubber / no timing
            if path.exists():
                open_video(path)
            else:
                print("   (video not found on disk — labelling blind)")

        # camera_perspective — asked for both layers, after the clip has been seen
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
                    "chirality": chir_val, "camera_perspective": persp,
                    "start_frame": sf, "end_frame": ef,
                    "start_time_ms": sms, "end_time_ms": ems},
                   write_header=not OUT_CSV.exists())
        done[c["file_path"]] = True
        save_progress(clips, done)

    final = save_progress(clips, done)
    print(f"\nSaved {final['completed']}/{final['total_clips']} "
          f"({final['percent_complete']}%) to {OUT_CSV.name}.")
    print(f"Progress recorded in {PROGRESS_FILE.name}. "
          f"Re-run to continue; then: python3 merge_annotations.py")


if __name__ == "__main__":
    main()
