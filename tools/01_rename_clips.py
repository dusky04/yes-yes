"""
Stage 01 — rename raw clips to  <class>_<4-digit>.mp4

Why first (the professor's "pehle clips naming karo"): a canonical, zero-padded
name per class is what every later stage keys off. Zero-padding (0012 not 12)
keeps lexical order == numeric order, so listings and the clip_ids stay sorted.

How a clip's class is decided (two modes, auto-detected):
  A) FOLDERED : raw_clips/<class>/<anything>.mp4   -> class is the parent folder
  B) PREFIXED : raw_clips/<class>_<anything>.mp4   -> class is the name prefix
This script does NOT move files between class groups; it only renames in place.
Stage 02 handles moving into videos/stroke_classes vs outcome_classes.

Output: a rename_manifest.csv recording old_path -> new_name for every clip, so
the operation is reproducible and reversible. We never destroy information we
can't reconstruct.

Run:  python3 tools/01_rename_clips.py            # real run
      python3 tools/01_rename_clips.py --dry-run  # show what would happen
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def detect_class(path: Path) -> str | None:
    """Return the class name for a raw clip, or None if undetectable."""
    # Mode A: parent folder is a known class
    parent = path.parent.name
    if parent in C.NAME_TO_ID:
        return parent
    # Mode B: filename starts with "<class>_" or "<class>-" or "<class> "
    stem = path.stem
    for name in sorted(C.ALL_CLASSES, key=len, reverse=True):  # longest first
        for sep in ("_", "-", " "):
            if stem.lower().startswith(name.lower() + sep):
                return name
    # Mode B fallback: the whole stem up to first separator equals a class
    head = stem.split("_")[0].split("-")[0].split(" ")[0].lower()
    if head in C.NAME_TO_ID:
        return head
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(C.RAW_CLIPS_DIR),
                    help="folder containing the downloaded clips")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    raw_dir = Path(args.raw)
    if not raw_dir.exists():
        sys.exit(f"[01] raw clips folder not found: {raw_dir}\n"
                 f"     Drop the Drive clips there (flat or in per-class folders).")

    clips = [p for p in raw_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    if not clips:
        sys.exit(f"[01] no video files under {raw_dir}")

    # Group by class, then assign zero-padded running numbers within each class.
    per_class: dict[str, list[Path]] = {}
    unknown: list[Path] = []
    for p in sorted(clips):
        cls = detect_class(p)
        (per_class.setdefault(cls, []) if cls else unknown).append(p)

    if unknown:
        print(f"[01] WARNING: {len(unknown)} clip(s) have no detectable class "
              f"(not in a class folder, no class prefix). They are skipped:")
        for p in unknown[:10]:
            print(f"        {p}")
        if len(unknown) > 10:
            print(f"        ... and {len(unknown) - 10} more")

    manifest = []
    for cls in sorted(per_class):
        for i, old in enumerate(sorted(per_class[cls]), start=1):
            new_name = f"{cls}_{i:04d}{old.suffix.lower()}"  # keep original ext
            new_path = old.with_name(new_name)
            manifest.append((str(old), cls, new_name))
            if args.dry_run:
                print(f"  {old.name:40s} -> {new_name}")
            else:
                if new_path.exists() and new_path != old:
                    print(f"[01] skip (target exists): {new_name}")
                    continue
                old.rename(new_path)

    out = raw_dir / "rename_manifest.csv"
    if not args.dry_run:
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["old_path", "class_name", "new_name"])
            w.writerows(manifest)

    print(f"\n[01] {'DRY RUN — ' if args.dry_run else ''}"
          f"{len(manifest)} clip(s) across {len(per_class)} class(es).")
    for cls in sorted(per_class):
        print(f"       {cls:<16} {len(per_class[cls])}")
    if not args.dry_run:
        print(f"[01] manifest -> {out}")


if __name__ == "__main__":
    main()
