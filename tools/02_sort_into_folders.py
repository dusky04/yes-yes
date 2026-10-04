"""
Stage 02 — move named clips into the final class folders.

Target layout (the professor's structure):
    videos/stroke_classes/<class>/<class>_NNNN.mp4     for the 10 strokes
    videos/outcome_classes/<class>/<class>_NNNN.mp4    for the 5 outcomes

The class is read from the filename prefix produced by stage 01, so this stage
is pure bookkeeping: it never guesses. If a clip's prefix isn't a known class,
it's left where it is and reported, rather than silently misfiled.

Run:  python3 tools/02_sort_into_folders.py
      python3 tools/02_sort_into_folders.py --copy   # copy instead of move
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def class_from_name(stem: str) -> str | None:
    # stage 01 produced "<class>_NNNN"; recover <class> by stripping the number.
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and parts[0] in C.NAME_TO_ID:
        return parts[0]
    # tolerate names that are just "<class>" or already correct
    if stem in C.NAME_TO_ID:
        return stem
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.RAW_CLIPS_DIR))
    ap.add_argument("--copy", action="store_true", help="copy rather than move")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    srcs = [p for p in raw_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS] \
        if raw_dir.exists() else []
    # Also pick up clips already sitting loose in videos/ (idempotent re-runs).
    srcs += [p for p in C.VIDEOS_DIR.rglob("*")
             if p.suffix.lower() in VIDEO_EXTS and p.parent.name not in C.NAME_TO_ID]

    if not srcs:
        sys.exit(f"[02] no clips to sort under {raw_dir} or {C.VIDEOS_DIR}")

    moved, skipped = 0, []
    for p in sorted(set(srcs)):
        cls = class_from_name(p.stem)
        if cls is None:
            skipped.append(p)
            continue
        dest_dir = C.layer_dir(cls) / cls
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / p.name
        if dest.resolve() == p.resolve():
            continue
        (shutil.copy2 if args.copy else shutil.move)(str(p), str(dest))
        moved += 1

    print(f"[02] {'copied' if args.copy else 'moved'} {moved} clip(s) into class folders.")
    for layer_name, base in (("stroke", C.STROKE_DIR), ("outcome", C.OUTCOME_DIR)):
        for d in sorted(base.glob("*")):
            if d.is_dir():
                n = len([x for x in d.iterdir() if x.suffix.lower() in VIDEO_EXTS])
                print(f"       {layer_name:<8} {d.name:<16} {n}")
    if skipped:
        print(f"[02] WARNING: {len(skipped)} clip(s) had no recognisable class prefix:")
        for p in skipped[:10]:
            print(f"        {p.name}")


if __name__ == "__main__":
    main()
