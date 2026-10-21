"""
make_split.py — turn the flat CricketEC release (one folder per class, no split)
into the fixed train/val/test layout that paper_configs.py reads.

    <src>/<Class>/<clip>.avi  ->  <out>/{train,val,test}/<class>/<clip>.avi  (symlinks)

* Byte-identical clips (same video under two names) are reduced to one copy, so
  no clip can appear in both train and test.
* Val and test get --per-class clips from every class (default 25, i.e. 375 each,
  as in the paper's evaluation protocol, Sec. 5.2); everything else is train.
* Class folder names are lower-cased (LBW -> lbw) to match the others.
* The assignment is seeded and written to <out>/split.json, so the split can be
  rebuilt or audited later. Re-running with the same inputs gives the same split.

Run:
    uv run benchmark/make_split.py --src data/raw/CricketEC --out data/CricketEC_split
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="folder with one subfolder per class")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-class", type=int, default=25, help="val and test clips per class")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src, out = args.src.resolve(), args.out.resolve()
    class_dirs = sorted(d for d in src.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders under {src}")

    seen: dict[str, str] = {}
    dropped: list[dict] = []
    clips: dict[str, list[Path]] = {}
    for d in class_dirs:
        name = d.name.lower()
        kept = []
        for p in sorted(d.glob("*.avi")):
            digest = md5(p)
            if digest in seen:
                dropped.append({"clip": p.relative_to(src).as_posix(), "duplicate_of": seen[digest]})
                continue
            seen[digest] = p.relative_to(src).as_posix()
            kept.append(p)
        clips[name] = kept

    rng = random.Random(args.seed)
    assignment: dict[str, dict[str, list[str]]] = {s: {} for s in SPLITS}
    for name, paths in clips.items():
        if len(paths) < 2 * args.per_class + 1:
            raise SystemExit(f"Class {name} has only {len(paths)} unique clips, "
                             f"need more than {2 * args.per_class}")
        shuffled = paths[:]
        rng.shuffle(shuffled)
        k = args.per_class
        parts = {"val": shuffled[:k], "test": shuffled[k:2 * k], "train": shuffled[2 * k:]}
        for split, ps in parts.items():
            assignment[split][name] = sorted(p.relative_to(src).as_posix() for p in ps)

    if out.exists():
        shutil.rmtree(out)
    for split, by_class in assignment.items():
        for name, rels in by_class.items():
            (out / split / name).mkdir(parents=True, exist_ok=True)
            for rel in rels:
                (out / split / name / Path(rel).name).symlink_to(src / rel)

    counts = {s: sum(len(v) for v in assignment[s].values()) for s in SPLITS}
    (out / "split.json").write_text(json.dumps({
        "source": str(src), "seed": args.seed, "per_class_val_test": args.per_class,
        "counts": counts, "dropped_duplicates": dropped, "splits": assignment,
    }, indent=2))

    print(f"{len(class_dirs)} classes, {sum(len(v) for v in clips.values())} unique clips "
          f"({len(dropped)} duplicate files dropped)")
    for name in clips:
        print(f"  {name:<12} train {len(assignment['train'][name]):4d}  "
              f"val {len(assignment['val'][name]):3d}  test {len(assignment['test'][name]):3d}")
    print(f"total        train {counts['train']:4d}  val {counts['val']:3d}  test {counts['test']:3d}")
    print(f"written to {out} (assignment in split.json)")


if __name__ == "__main__":
    main()
