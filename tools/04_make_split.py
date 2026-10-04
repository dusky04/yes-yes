from __future__ import annotations
"""
Stage 04 — assign every clip to train or val, WITHOUT leaking a match.

The core idea (and the reason the files are named *_match_disjoint):
clips from the same broadcast match share players, kit, pitch, camera and
lighting. If some clips of a match go to train and others to val, the model can
score well on val by recognising the *match* instead of the *stroke*. That is
data leakage, and it makes the reported accuracy a lie. So we split at the
MATCH level: every clip of a given match goes entirely to one side.

Input: annotations/match_mapping.csv  with columns  clip_stem,source_match_id
       (clip_stem is the filename without extension, e.g. cover_0012)
Each clip MUST have a match id — that's the one thing a script cannot infer
from the pixels. If the mapping is missing, we stop and tell you exactly which
clips need an id.

We also stratify as well as match-disjointness allows: matches are greedily
assigned so each class gets close to VAL_FRACTION of its clips in val, which
keeps rare classes from vanishing out of either split.

Output: annotations/_split_assignments.csv  (clip_stem,source_match_id,split)

Run:  python3 tools/04_make_split.py
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

SPLIT_CACHE = C.ANNOTATIONS_DIR / "_split_assignments.csv"


def load_metadata():
    if not C.METADATA_CACHE.exists():
        sys.exit("[04] missing metadata cache. Run tools/03 first.")
    with C.METADATA_CACHE.open() as f:
        return list(csv.DictReader(f))


def load_match_mapping(clip_stems):
    if not C.MATCH_MAPPING_CSV.exists():
        _write_mapping_template(clip_stems)
        sys.exit(
            f"[04] no match mapping found. A template listing all "
            f"{len(clip_stems)} clips was written to:\n"
            f"       {C.MATCH_MAPPING_CSV}\n"
            f"     Fill the source_match_id column (which broadcast match each "
            f"clip came from) and re-run. This is the one input only you have.")
    mapping = {}
    with C.MATCH_MAPPING_CSV.open() as f:
        for row in csv.DictReader(f):
            stem = (row.get("clip_stem") or "").strip()
            mid = (row.get("source_match_id") or "").strip()
            if stem:
                mapping[stem] = mid
    missing = [s for s in clip_stems if not mapping.get(s)]
    if missing:
        sys.exit(f"[04] {len(missing)} clip(s) have no source_match_id in "
                 f"{C.MATCH_MAPPING_CSV.name}:\n       "
                 + ", ".join(missing[:15])
                 + (" ..." if len(missing) > 15 else "")
                 + "\n     Fill them in and re-run.")
    return mapping


def _write_mapping_template(clip_stems):
    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    with C.MATCH_MAPPING_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_stem", "source_match_id"])
        for s in sorted(clip_stems):
            w.writerow([s, ""])


def stratified_clip_split(meta):
    """Fallback when no match data exists: stratified per-class clip split.
    NOT leak-free; source_match_id is left empty. Clearly flagged."""
    by_class = defaultdict(list)
    for r in meta:
        by_class[r["class_name"]].append(Path(r["file_path"]).stem)
    assign = {}
    rng = random.Random(C.SPLIT_SEED)
    for cls, stems in by_class.items():
        rng.shuffle(stems)
        n_val = max(1, round(C.VAL_FRACTION * len(stems))) if len(stems) > 1 else 0
        for i, s in enumerate(stems):
            assign[s] = "val" if i < n_val else "train"
    with SPLIT_CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_stem", "source_match_id", "split"])
        for s, sp in assign.items():
            w.writerow([s, "", sp])
    n_val = sum(1 for v in assign.values() if v == "val")
    print(f"[04] NO MATCH DATA: stratified clip-level split (NOT match-disjoint).")
    print(f"[04] source_match_id left empty. {len(assign)-n_val} train / {n_val} val")
    print(f"[04] WARNING: tell the professor this split can leak until a "
          f"clip->match mapping is provided; then re-run without --allow-no-match.")
    print(f"[04] -> {SPLIT_CACHE.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-match", action="store_true",
                    help="produce a stratified clip-level split when no match "
                         "mapping exists (NOT leak-free)")
    args = ap.parse_args()

    meta = load_metadata()
    if args.allow_no_match and not C.MATCH_MAPPING_CSV.exists():
        stratified_clip_split(meta)
        return

    stem_of = {r["file_path"]: Path(r["file_path"]).stem for r in meta}
    clip_stems = [stem_of[r["file_path"]] for r in meta]
    mapping = load_match_mapping(clip_stems)

    # Group clips by match; remember each match's class composition.
    match_clips = defaultdict(list)            # match_id -> [row, ...]
    for r in meta:
        stem = Path(r["file_path"]).stem
        match_clips[mapping[stem]].append(r)

    classes = C.ALL_CLASSES
    total_per_class = defaultdict(int)
    for r in meta:
        total_per_class[r["class_name"]] += 1

    # Greedy stratified, match-disjoint assignment.
    # Shuffle matches reproducibly, then send each match to whichever side is
    # currently furthest *below* its val target, summed over the classes in it.
    matches = list(match_clips)
    random.Random(C.SPLIT_SEED).shuffle(matches)

    val_target = {c: C.VAL_FRACTION * total_per_class[c] for c in classes}
    val_have = defaultdict(int)
    assign = {}

    for mid in matches:
        comp = defaultdict(int)
        for r in match_clips[mid]:
            comp[r["class_name"]] += 1
        # deficit if we were to NOT add this match to val, per class
        pull_to_val = sum(max(0.0, val_target[c] - val_have[c]) * n
                          for c, n in comp.items())
        room_in_val = sum(max(0.0, val_target[c] - val_have[c]) for c in comp)
        if room_in_val > 0 and pull_to_val > 0:
            assign[mid] = "val"
            for c, n in comp.items():
                val_have[c] += n
        else:
            assign[mid] = "train"

    # Write per-clip assignments.
    with SPLIT_CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_stem", "source_match_id", "split"])
        for mid, rows in match_clips.items():
            for r in rows:
                w.writerow([Path(r["file_path"]).stem, mid, assign[mid]])

    # Report + leakage assertion.
    n_val = sum(len(match_clips[m]) for m in matches if assign[m] == "val")
    n_train = len(meta) - n_val
    train_matches = {m for m in matches if assign[m] == "train"}
    val_matches = {m for m in matches if assign[m] == "val"}
    assert not (train_matches & val_matches), "LEAKAGE: a match is in both splits"

    print(f"[04] {len(matches)} matches -> "
          f"{len(train_matches)} train / {len(val_matches)} val")
    print(f"[04] {len(meta)} clips -> {n_train} train / {n_val} val "
          f"({n_val/len(meta):.0%} val; target {C.VAL_FRACTION:.0%})")
    print(f"[04] leakage check passed: no match on both sides")
    # Per-class val coverage, so you can see nothing disappeared.
    print("[04] per-class  train/val:")
    for c in classes:
        v = val_have[c]
        print(f"       {c:<16} {total_per_class[c]-v:>4}/{v:<4}")
    print(f"[04] -> {SPLIT_CACHE.name}")


if __name__ == "__main__":
    main()
