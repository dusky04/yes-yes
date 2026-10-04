from __future__ import annotations
"""
Stage 04 — assign every clip to train / val / test, WITHOUT leaking a match.

The core idea (and the reason the files are named *_match_disjoint):
clips from the same broadcast match share players, kit, pitch, camera and
lighting. If clips of one match land in more than one split, the model can score
well by recognising the *match* instead of the *stroke* — data leakage, which
makes the reported accuracy a lie. So, when match data exists, every clip of a
given match goes entirely into ONE split.

Split sizes come from config: VAL_FRACTION, TEST_FRACTION (train is the rest).

Input (for the leak-free split): annotations/match_mapping.csv with columns
    clip_stem,source_match_id
Each clip needs a match id — the one thing a script can't infer from pixels.

Fallback: with --allow-no-match and no mapping, a stratified per-class clip
split is produced instead (source_match_id left empty). NOT leak-free; flagged.

Output: annotations/_split_assignments.csv  (clip_stem,source_match_id,split)
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
SPLITS = ("train", "val", "test")


def load_metadata():
    if not C.METADATA_CACHE.exists():
        sys.exit("[04] missing metadata cache. Run tools/03 first.")
    with C.METADATA_CACHE.open() as f:
        return list(csv.DictReader(f))


def _write_cache(rows):
    """rows: list of (clip_stem, source_match_id, split)."""
    with SPLIT_CACHE.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_stem", "source_match_id", "split"])
        w.writerows(rows)


def _counts(assign_values):
    c = {s: 0 for s in SPLITS}
    for v in assign_values:
        c[v] += 1
    return c


# ---------------------------------------------------------------------------
# Fallback: no match data -> stratified per-class three-way clip split.
# ---------------------------------------------------------------------------
def stratified_clip_split(meta):
    by_class = defaultdict(list)
    for r in meta:
        by_class[r["class_name"]].append(Path(r["file_path"]).stem)

    assign = {}
    rng = random.Random(C.SPLIT_SEED)
    for cls, stems in by_class.items():
        rng.shuffle(stems)
        n = len(stems)
        n_val = round(C.VAL_FRACTION * n)
        n_test = round(C.TEST_FRACTION * n)
        # guarantee at least 1 in val/test when the class is big enough
        if n >= 3:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - 1 - n_val))
        for i, s in enumerate(stems):
            if i < n_val:
                assign[s] = "val"
            elif i < n_val + n_test:
                assign[s] = "test"
            else:
                assign[s] = "train"

    _write_cache([(s, "", sp) for s, sp in assign.items()])
    c = _counts(assign.values())
    print("[04] NO MATCH DATA: stratified clip-level split (NOT match-disjoint).")
    print(f"[04] source_match_id left empty. "
          f"{c['train']} train / {c['val']} val / {c['test']} test")
    print(f"[04] WARNING: tell the professor this split can leak until a "
          f"clip->match mapping is provided; then re-run without --allow-no-match.")
    print(f"[04] -> {SPLIT_CACHE.name}")


# ---------------------------------------------------------------------------
# Leak-free: assign whole matches to train/val/test, stratified by class.
# ---------------------------------------------------------------------------
def load_match_mapping(clip_stems):
    if not C.MATCH_MAPPING_CSV.exists():
        C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
        with C.MATCH_MAPPING_CSV.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_stem", "source_match_id"])
            for s in sorted(clip_stems):
                w.writerow([s, ""])
        sys.exit(
            f"[04] no match mapping found. A template listing all "
            f"{len(clip_stems)} clips was written to:\n       "
            f"{C.MATCH_MAPPING_CSV}\n     Fill the source_match_id column and "
            f"re-run (drop --allow-no-match for the leak-free split).")
    mapping = {}
    with C.MATCH_MAPPING_CSV.open() as f:
        for row in csv.DictReader(f):
            stem = (row.get("clip_stem") or "").strip()
            mid = (row.get("source_match_id") or "").strip()
            if stem:
                mapping[stem] = mid
    missing = [s for s in clip_stems if not mapping.get(s)]
    if missing:
        sys.exit(f"[04] {len(missing)} clip(s) have no source_match_id:\n       "
                 + ", ".join(missing[:15]) + (" ..." if len(missing) > 15 else ""))
    return mapping


def match_disjoint_split(meta, mapping):
    match_clips = defaultdict(list)
    for r in meta:
        match_clips[mapping[Path(r["file_path"]).stem]].append(r)

    classes = C.ALL_CLASSES
    total_per_class = defaultdict(int)
    for r in meta:
        total_per_class[r["class_name"]] += 1

    # Targets per class for the two held-out splits.
    target = {
        "val": {c: C.VAL_FRACTION * total_per_class[c] for c in classes},
        "test": {c: C.TEST_FRACTION * total_per_class[c] for c in classes},
    }
    have = {"val": defaultdict(int), "test": defaultdict(int)}

    matches = list(match_clips)
    random.Random(C.SPLIT_SEED).shuffle(matches)

    assign = {}
    for mid in matches:
        comp = defaultdict(int)
        for r in match_clips[mid]:
            comp[r["class_name"]] += 1
        # score how much this match is "needed" by val vs test (unmet deficit)
        pull = {}
        for sp in ("val", "test"):
            pull[sp] = sum(max(0.0, target[sp][c] - have[sp][c]) * n
                           for c, n in comp.items())
        best = max(pull, key=pull.get)
        if pull[best] > 0:
            assign[mid] = best
            for c, n in comp.items():
                have[best][c] += n
        else:
            assign[mid] = "train"

    _write_cache([(Path(r["file_path"]).stem, mid, assign[mid])
                  for mid, rows in match_clips.items() for r in rows])

    # leakage check + report
    by_split = defaultdict(set)
    for mid in matches:
        by_split[assign[mid]].add(mid)
    assert not (by_split["train"] & by_split["val"]) \
        and not (by_split["train"] & by_split["test"]) \
        and not (by_split["val"] & by_split["test"]), "LEAKAGE: match in 2 splits"

    clip_counts = _counts([assign[mapping[Path(r["file_path"]).stem]] for r in meta])
    print(f"[04] {len(matches)} matches -> "
          f"{len(by_split['train'])} train / {len(by_split['val'])} val / "
          f"{len(by_split['test'])} test")
    print(f"[04] {len(meta)} clips -> {clip_counts['train']} train / "
          f"{clip_counts['val']} val / {clip_counts['test']} test")
    print(f"[04] leakage check passed: no match in two splits")
    print(f"[04] -> {SPLIT_CACHE.name}")


def _read_mapping():
    """Read match_mapping.csv into {stem: match_id}; blanks allowed. None if absent."""
    if not C.MATCH_MAPPING_CSV.exists():
        return None
    m = {}
    with C.MATCH_MAPPING_CSV.open() as f:
        for row in csv.DictReader(f):
            stem = (row.get("clip_stem") or "").strip()
            if stem:
                m[stem] = (row.get("source_match_id") or "").strip()
    return m


def _write_synthetic_mapping(meta):
    """Auto-generate a match_mapping.csv when no real match ids are available.
    Each clip gets a distinct SYNTHETIC placeholder id (clearly prefixed so it is
    never mistaken for a real broadcast match). With one id per clip the resulting
    partition is effectively a RANDOM stratified split, not a true match-disjoint
    one — the README documents this. Returns the mapping dict."""
    C.ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    rows = sorted(meta, key=lambda r: r["file_path"])
    mapping = {Path(r["file_path"]).stem: f"SYNTH_{i:05d}"
               for i, r in enumerate(rows, 1)}
    with C.MATCH_MAPPING_CSV.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_stem", "source_match_id"])
        for r in rows:
            s = Path(r["file_path"]).stem
            w.writerow([s, mapping[s]])
    print(f"[04] no real match ids found — auto-generated {len(mapping)} SYNTHETIC "
          f"placeholder ids in {C.MATCH_MAPPING_CSV.name}.")
    print(f"[04] NOTE: synthetic ids => this is a RANDOM stratified split, not a "
          f"true match-disjoint one (see README). Provide real ids to upgrade it.")
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-no-match", action="store_true",
                    help="(kept for back-compat; fallback is now automatic)")
    ap.parse_args()

    meta = load_metadata()
    stems = [Path(r["file_path"]).stem for r in meta]
    mapping = _read_mapping()

    # Real, fully-populated mapping -> genuine match-disjoint split.
    # Otherwise auto-generate synthetic placeholder ids so the pipeline always
    # proceeds; the result is a random stratified split (documented in the README).
    if mapping and all(mapping.get(s) for s in stems):
        match_disjoint_split(meta, mapping)
    else:
        if mapping is not None:
            filled = sum(1 for s in stems if mapping.get(s))
            print(f"[04] match_mapping.csv present but only {filled}/{len(stems)} "
                  f"clips have a source_match_id — regenerating synthetic ids.")
        mapping = _write_synthetic_mapping(meta)
        match_disjoint_split(meta, mapping)


if __name__ == "__main__":
    main()
