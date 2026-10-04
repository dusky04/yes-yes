"""
Stage 03 — probe every sorted clip for its real physical properties.

We never hand-type duration/fps/frames — we read them from the file so the
annotations can't drift from reality. Primary probe is ffprobe (fast, exact);
if ffprobe is missing we fall back to OpenCV.

Output: annotations/_metadata_cache.csv with one row per clip:
    class_name, class_id, semantic_layer, file_path (relative to ROOT),
    duration_sec, num_frames, fps, width, height

A consistency check flags clips where num_frames deviates a lot from
duration*fps (usually a variable-frame-rate or truncated file worth a look).

Run:  python3 tools/03_extract_metadata.py
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
HAVE_FFPROBE = shutil.which("ffprobe") is not None


def probe_ffprobe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,avg_frame_rate,nb_frames,duration:format=duration",
           "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    st = data["streams"][0]
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    fps = (float(num) / float(den)) if float(den) else 0.0
    duration = float(st.get("duration")
                     or data.get("format", {}).get("duration") or 0.0)
    nb = st.get("nb_frames")
    frames = int(nb) if nb and nb.isdigit() else int(round(duration * fps))
    return {"duration_sec": round(duration, 2), "num_frames": frames,
            "fps": round(fps, 3), "width": int(st["width"]),
            "height": int(st["height"])}


def probe_opencv(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration = (frames / fps) if fps else 0.0
    return {"duration_sec": round(duration, 2), "num_frames": frames,
            "fps": round(fps, 3), "width": w, "height": h}


def probe(path: Path) -> dict:
    try:
        return probe_ffprobe(path) if HAVE_FFPROBE else probe_opencv(path)
    except Exception as e:                    # noqa: BLE001
        print(f"[03] ffprobe failed on {path.name} ({e}); trying OpenCV")
        return probe_opencv(path)


def main():
    rows, flags = [], []
    for base in (C.STROKE_DIR, C.OUTCOME_DIR):
        for class_dir in sorted(p for p in base.glob("*") if p.is_dir()):
            cls = class_dir.name
            if cls not in C.NAME_TO_ID:
                print(f"[03] WARNING: unknown class folder '{cls}' — skipped")
                continue
            for clip in sorted(class_dir.glob("*")):
                if clip.suffix.lower() not in VIDEO_EXTS:
                    continue
                m = probe(clip)
                expected = m["duration_sec"] * m["fps"]
                if expected and abs(m["num_frames"] - expected) > 0.1 * expected:
                    flags.append((clip.name, m["num_frames"], round(expected, 1)))
                rows.append({
                    "class_name": cls,
                    "class_id": C.NAME_TO_ID[cls],
                    "semantic_layer": C.NAME_TO_LAYER[cls],
                    "file_path": str(clip.relative_to(C.ROOT)).replace("\\", "/"),
                    **m,
                })

    if not rows:
        sys.exit("[03] no clips found under videos/. Run stages 01–02 first.")

    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["class_name", "class_id", "semantic_layer", "file_path",
            "duration_sec", "num_frames", "fps", "width", "height"]
    with C.METADATA_CACHE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"[03] probed {len(rows)} clip(s) "
          f"via {'ffprobe' if HAVE_FFPROBE else 'OpenCV'} -> {C.METADATA_CACHE.name}")
    if flags:
        print(f"[03] {len(flags)} clip(s) where num_frames != duration*fps "
              f"(VFR or truncated — worth a glance):")
        for name, got, exp in flags[:10]:
            print(f"        {name}: {got} frames vs ~{exp} expected")


if __name__ == "__main__":
    main()
