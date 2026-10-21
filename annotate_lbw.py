from __future__ import annotations
"""
annotate_lbw.py — LBW annotation, SINGLE-WINDOW flow.

One OpenCV window stays open for the whole clip. You scrub to mark start/end,
then answer chirality, bowler handedness, camera perspective and qc_status with
one keypress each WHILE the clip keeps looping in front of you — no switching to
the terminal, no window closing between questions.

Records per LBW clip:
  start/end frame + millisecond timing (frame scrubber)
  chirality          (batter: right / left / unknown)
  bowler_handedness  (right-arm / left-arm / unknown)
  camera_perspective (3 = Pitch / Stump Close-up, 5 = Pitch / Umpire Perspective, u)
  qc_status          (PASS / FAIL_GRAPHICS / FAIL_OCCLUSION)

Writes annotations/lbw_annotations.csv (its own file; batting CSV untouched).

First time — import the raw LBW videos into the dataset (renames to lbw_0001..):
    python3 annotate_lbw.py --src "/path/to/LBW"
Then annotate:
    python3 annotate_lbw.py
    python3 annotate_lbw.py --type-ms   # no-GUI fallback: type everything in terminal
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import config as C

ROOT = Path(__file__).resolve().parent

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
LBW_DIR = C.OUTCOME_DIR / "lbw"
OUT_CSV = C.ANNOTATIONS_DIR / "lbw_annotations.csv"
PROGRESS_FILE = C.ANNOTATIONS_DIR / "lbw_progress.json"

CSV_FIELDS = ["clip_id", "file_path", "chirality", "bowler_handedness",
              "camera_perspective", "start_frame", "end_frame",
              "start_time_ms", "end_time_ms", "qc_status"]

# key -> stored value. Perspective numbers 3/5 match the outcome-class menu.
LBW_PERSPECTIVES = {"3": "Pitch / Stump Close-up", "5": "Pitch / Umpire Perspective"}
CHIRALITY = {"r": "right-handed", "l": "left-handed", "u": None}
BOWLER_HAND = {"r": "right-arm", "l": "left-arm", "u": None}
QC = {"p": "PASS", "g": "FAIL_GRAPHICS", "o": "FAIL_OCCLUSION"}


# --------------------------------------------------------------------------- #
#  video import / dataset helpers
# --------------------------------------------------------------------------- #
def probe(path: Path):
    """fps and frame-count for one clip, via OpenCV."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return float(fps), n
    except Exception:                                     # noqa: BLE001
        return 0.0, 0


def import_src(src: str):
    """Copy raw LBW videos into videos/outcome_classes/lbw/ as lbw_0001.avi...,
    in sorted order. Originals untouched. Idempotent."""
    src_dir = Path(src)
    if not src_dir.exists():
        sys.exit(f"[lbw] --src folder not found: {src_dir}")
    vids = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not vids:
        sys.exit(f"[lbw] no video files under {src_dir}")
    LBW_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for i, p in enumerate(vids, 1):
        dst = LBW_DIR / f"lbw_{i:04d}{p.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(str(p), str(dst))
            copied += 1
    print(f"[lbw] imported {len(vids)} clip(s) into {LBW_DIR} "
          f"({copied} newly copied, originals untouched).")


def lbw_clips():
    if not LBW_DIR.exists():
        return []
    vids = sorted(p for p in LBW_DIR.glob("*") if p.suffix.lower() in VIDEO_EXTS)
    out = []
    for i, p in enumerate(vids, 1):
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        out.append({"clip_id": f"CEC_LBW_{i:04d}", "file_path": rel})
    return out


def load_done():
    done = set()
    if OUT_CSV.exists():
        for r in csv.DictReader(OUT_CSV.open()):
            done.add(r["file_path"])
    return done


def append_row(row):
    new = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def save_progress(total, done_n):
    PROGRESS_FILE.write_text(json.dumps({
        "class": "lbw", "total_lbw_clips": total, "completed": done_n,
        "remaining": total - done_n,
        "percent_complete": round(100 * done_n / total, 1) if total else 0.0,
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }, indent=2))


# --------------------------------------------------------------------------- #
#  single-window annotation
# --------------------------------------------------------------------------- #
def _read_frames(path: Path):
    import cv2
    frames = []
    if path.exists():
        cap = cv2.VideoCapture(str(path))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    return frames


def _draw(cv2, win, base_frame, lines, active_line=None):
    img = base_frame.copy()
    y = 24
    for i, ln in enumerate(lines):
        color = (0, 255, 255) if i == active_line else (0, 255, 0)
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 1, cv2.LINE_AA)
        y += 24
    cv2.imshow(win, img)


def annotate_clip_window(cv2, win, frames, eff_fps, header, prev):
    """Run the full timing + 4-question flow for ONE clip inside `win`.
    Returns a dict of answers, or ('__skip__'|'__quit__')."""
    n = len(frames)
    rng = lambda: (0, n - 1) if (start is None or end is None) else tuple(sorted((start, end)))

    stage = "timing"
    idx = 0                      # scrub cursor (timing)
    play = idx                   # playback cursor (questions)
    playing_timing = False
    start = end = None
    ans = {"ch": None, "bowl": None, "persp": None, "qc": None}

    def timing_lines():
        t_ms = int(round(idx * 1000 / eff_fps))
        return [
            header,
            f"TIMING  frame {idx+1}/{n}  t={t_ms}ms   START={start if start is not None else '-'}  END={end if end is not None else '-'}",
            "a/d step   b/f +-5   space play   s=START (last frame BEFORE release)   e=END",
            "Enter=confirm timing   u=unknown timing   n=skip clip   q=save & quit",
        ]

    def q_lines(title, options, active_key_hint):
        return [
            header,
            f"{title}   [{active_key_hint}]",
            options,
            f"chosen -> chir:{ans['ch'] or '-'}  bowl:{ans['bowl'] or '-'}  "
            f"persp:{ans['persp'] or '-'}  qc:{ans['qc'] or '-'}"
            + (f"   (Enter=reuse prev)" if False else ""),
            "z=back   n=skip clip   q=save & quit",
        ]

    QUESTION_STAGES = ["chirality", "bowler", "perspective", "qc"]

    while True:
        # ---------------- TIMING ----------------
        if stage == "timing":
            _draw(cv2, win, frames[idx], timing_lines())
            k = cv2.waitKey(30 if playing_timing else 0) & 0xFF
            if playing_timing and k == 255:
                idx = min(idx + 1, n - 1)
                if idx >= n - 1:
                    playing_timing = False
                continue
            if k in (ord('d'), 83):
                idx = min(idx + 1, n - 1)
            elif k in (ord('a'), 81):
                idx = max(idx - 1, 0)
            elif k == ord('f'):
                idx = min(idx + 5, n - 1)
            elif k == ord('b'):
                idx = max(idx - 5, 0)
            elif k == 32:
                playing_timing = not playing_timing
            elif k == ord('s'):
                start = idx
            elif k == ord('e'):
                end = idx
            elif k == ord('u'):
                start = end = None
                stage = "chirality"
                play = rng()[0]
            elif k == ord('n'):
                return "__skip__"
            elif k == ord('q'):
                return "__quit__"
            elif k in (13, 10):
                if start is None or end is None:
                    continue
                stage = "chirality"
                play = rng()[0]
            continue

        # ---------------- QUESTION STAGES (clip keeps looping) ----------------
        lo, hi = rng()
        if play < lo or play > hi:
            play = lo

        if stage == "chirality":
            lines = q_lines("chirality (BATTER)?",
                            "r = right-handed    l = left-handed    u = unknown",
                            f"prev {prev.get('ch') or '-'}")
            mapping, key_name = CHIRALITY, "ch"
        elif stage == "bowler":
            lines = q_lines("bowler_handedness?",
                            "r = right-arm    l = left-arm    u = unknown",
                            f"prev {prev.get('bowl') or '-'}")
            mapping, key_name = BOWLER_HAND, "bowl"
        elif stage == "perspective":
            lines = q_lines("camera_perspective?",
                            "3 = Pitch / Stump Close-up    5 = Pitch / Umpire Perspective    u = unknown",
                            f"prev {prev.get('persp') or '-'}")
            mapping, key_name = {**LBW_PERSPECTIVES, "u": None}, "persp"
        elif stage == "qc":
            lines = q_lines("qc_status?",
                            "p = PASS (clean out given)    g = FAIL_GRAPHICS    o = FAIL_OCCLUSION",
                            f"prev {prev.get('qc') or '-'}")
            mapping, key_name = QC, "qc"
        else:
            lines = None

        if stage in QUESTION_STAGES:
            _draw(cv2, win, frames[play], lines)
            play = lo + ((play - lo + 1) % (hi - lo + 1))     # loop the segment
            k = cv2.waitKey(45) & 0xFF
            ch = chr(k) if 32 <= k < 127 else ""
            if ch in mapping:
                ans[key_name] = ch
                _advance = QUESTION_STAGES.index(stage) + 1
                stage = QUESTION_STAGES[_advance] if _advance < len(QUESTION_STAGES) else "confirm"
            elif k in (13, 10) and prev.get(key_name):
                ans[key_name] = prev[key_name]
                _advance = QUESTION_STAGES.index(stage) + 1
                stage = QUESTION_STAGES[_advance] if _advance < len(QUESTION_STAGES) else "confirm"
            elif k == ord('z'):
                i = QUESTION_STAGES.index(stage)
                stage = "timing" if i == 0 else QUESTION_STAGES[i - 1]
            elif k == ord('n'):
                return "__skip__"
            elif k == ord('q'):
                return "__quit__"
            continue

        # ---------------- CONFIRM ----------------
        if stage == "confirm":
            sf, ef = ("", "")
            if start is not None and end is not None:
                sf, ef = sorted((start, end))
            lines = [
                header,
                "CONFIRM:  chir=%s  bowl=%s  persp=%s  qc=%s" % (
                    ans['ch'] or '-', ans['bowl'] or '-', ans['persp'] or '-', ans['qc'] or '-'),
                f"timing: START={sf if sf != '' else 'unknown'}  END={ef if ef != '' else 'unknown'}",
                "Enter = SAVE & next clip      z = back to qc      n = skip      q = save & quit",
            ]
            _draw(cv2, win, frames[play], lines)
            play = lo + ((play - lo + 1) % (hi - lo + 1))
            k = cv2.waitKey(45) & 0xFF
            if k in (13, 10):
                eff = eff_fps if eff_fps > 0 else 30.0
                return {
                    "ch": ans["ch"], "bowl": ans["bowl"], "persp": ans["persp"], "qc": ans["qc"],
                    "sf": sf, "ef": ef,
                    "sms": (int(round(sf * 1000 / eff)) if sf != "" else ""),
                    "ems": (int(round(ef * 1000 / eff)) if ef != "" else ""),
                }
            elif k == ord('z'):
                stage = "qc"
            elif k == ord('n'):
                return "__skip__"
            elif k == ord('q'):
                return "__quit__"
            continue


# --------------------------------------------------------------------------- #
#  terminal fallback (no display / --type-ms)
# --------------------------------------------------------------------------- #
def ask(prompt, mapping, prev):
    valid = set(mapping)
    while True:
        a = input(prompt).strip().lower()
        if a in ("q", "s"):
            return a
        if a == "" and prev is not None:
            return prev
        if a in valid:
            return a
        print("   type one of: " + "/".join(sorted(valid)) + "  (or Enter/s/q)")


def annotate_clip_terminal(path, header, prev):
    print("\n" + header)
    fps, n = probe(path)
    dur_ms = int(round(n * 1000 / fps)) if fps > 0 else 0
    print(f"   (terminal mode) clip length ~{dur_ms} ms")
    try:
        sms = input("   start_time_ms (blank=unknown): ").strip()
        ems = input("   end_time_ms  (blank=unknown): ").strip()
    except EOFError:
        return "__quit__"
    ch = ask(f"   chirality r/l/u [{prev.get('ch') or '-'}]: ", CHIRALITY, prev.get("ch"))
    if ch in ("q", "s"):
        return "__quit__" if ch == "q" else "__skip__"
    bowl = ask(f"   bowler r/l/u [{prev.get('bowl') or '-'}]: ", BOWLER_HAND, prev.get("bowl"))
    if bowl in ("q", "s"):
        return "__quit__" if bowl == "q" else "__skip__"
    persp = ask(f"   perspective 3/5/u [{prev.get('persp') or '-'}]: ",
                {**LBW_PERSPECTIVES, "u": None}, prev.get("persp"))
    if persp in ("q", "s"):
        return "__quit__" if persp == "q" else "__skip__"
    qc = ask(f"   qc p/g/o [{prev.get('qc') or '-'}]: ", QC, prev.get("qc"))
    if qc in ("q", "s"):
        return "__quit__" if qc == "q" else "__skip__"
    return {"ch": ch, "bowl": bowl, "persp": persp, "qc": qc,
            "sf": "", "ef": "", "sms": sms, "ems": ems}


# --------------------------------------------------------------------------- #
def _has_display():
    if sys.platform == "darwin" or sys.platform.startswith("win"):
        return True
    return bool(os.environ.get("DISPLAY"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="raw LBW video folder to import into the dataset")
    ap.add_argument("--type-ms", dest="type_ms", action="store_true",
                    help="no-GUI fallback: type every field in the terminal")
    args = ap.parse_args()

    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    if args.src:
        import_src(args.src)

    clips = lbw_clips()
    if not clips:
        sys.exit(f"[lbw] no clips in {LBW_DIR}. Run once with "
                 f'--src "/path/to/LBW" to import the videos first.')

    done = load_done()
    todo = [c for c in clips if c["file_path"] not in done]
    done_n = len(clips) - len(todo)
    save_progress(len(clips), done_n)

    print("=" * 64)
    print("  LBW ANNOTATION — single window: scrub, then r/l/u · r/l/u · 3/5/u · p/g/o")
    print(f"  completed {done_n}/{len(clips)}   remaining {len(todo)}")
    print("=" * 64)
    if not todo:
        print("All LBW clips already annotated. Nothing to do.")
        return

    # decide GUI vs terminal
    use_gui = (not args.type_ms) and _has_display()
    cv2 = win = None
    if use_gui:
        try:
            import cv2 as _cv2
            cv2 = _cv2
            win = "LBW annotation"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        except Exception as e:                               # noqa: BLE001
            print(f"   (no OpenCV window available — terminal mode: {e})")
            use_gui = False

    prev = {"ch": None, "bowl": None, "persp": None, "qc": None}
    for i, c in enumerate(todo, 1):
        path = ROOT / c["file_path"]
        header = f"[{i}/{len(todo)} · {done_n+i}/{len(clips)}]  {c['clip_id']}  (lbw)  {Path(c['file_path']).name}"

        res = "__skip__"
        if use_gui:
            frames = _read_frames(path)
            if not frames:
                print(f"   {c['clip_id']}: video not readable, skipping -> {path}")
                continue
            fps, _n = probe(path)
            eff_fps = fps if fps > 0 else 30.0
            print(f"\n{header}  (window open — press keys in the window)")
            res = annotate_clip_window(cv2, win, frames, eff_fps, header, prev)
        else:
            res = annotate_clip_terminal(path, header, prev)

        if res == "__quit__":
            break
        if res == "__skip__":
            continue

        append_row({
            "clip_id": c["clip_id"], "file_path": c["file_path"],
            "chirality": CHIRALITY.get(res["ch"], "") or "",
            "bowler_handedness": BOWLER_HAND.get(res["bowl"], "") or "",
            "camera_perspective": LBW_PERSPECTIVES.get(res["persp"], "") or "",
            "start_frame": res["sf"], "end_frame": res["ef"],
            "start_time_ms": res["sms"], "end_time_ms": res["ems"],
            "qc_status": QC.get(res["qc"], ""),
        })
        prev.update({k: res[k] for k in ("ch", "bowl", "persp", "qc")})
        done_n += 1
        save_progress(len(clips), done_n)
        print(f"   saved {c['clip_id']}  ->  chir={res['ch']} bowl={res['bowl']} "
              f"persp={res['persp']} qc={res['qc']}")

    if use_gui and cv2 is not None:
        try:
            cv2.destroyAllWindows(); cv2.waitKey(1)
        except Exception:                                    # noqa: BLE001
            pass
    save_progress(len(clips), done_n)
    print(f"\nSaved {done_n}/{len(clips)} LBW clips to {OUT_CSV.name}. "
          f"Re-run to continue; then: python3 merge_annotations.py")


if __name__ == "__main__":
    main()
