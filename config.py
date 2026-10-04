"""
config.py — the single source of truth for the CricketEC dataset pipeline.

Every script in tools/ reads from this file. Change the taxonomy or the split
ratio here, in ONE place, and the whole pipeline follows. Nothing downstream
hard-codes a class name or an id.

------------------------------------------------------------------------------
!!! ACTION REQUIRED (Tapan): confirm the class lists below with the professor !!!
------------------------------------------------------------------------------
We know for certain from his screenshots:
    cover  -> class_id 0   (stroke)
    six    -> class_id 10  (outcome)
    wicket -> class_id 12  (outcome)
    "defense" and "four" also exist.
Everything else below is a SENSIBLE PLACEHOLDER. The scripts do not care what
the names are — they derive everything from these lists — so you can swap any
name and re-run. But the DELIVERED dataset must use his real names, so get the
exact 10 strokes and 5 outcomes from him and edit the two lists below.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# Resolve everything relative to this file, so the pipeline works no matter
# what directory you run it from.
ROOT = Path(__file__).resolve().parent

VIDEOS_DIR          = ROOT / "videos"
STROKE_DIR          = VIDEOS_DIR / "stroke_classes"
OUTCOME_DIR         = VIDEOS_DIR / "outcome_classes"
ANNOTATIONS_DIR     = ROOT / "annotations"

# Where you drop the clips you downloaded from the professor's Drive, before
# sorting. Can be a flat folder or already-foldered; see tools/02 for both modes.
RAW_CLIPS_DIR       = ROOT / "raw_clips"

# Input you provide: clip stem -> source broadcast match. Needed for a
# leak-free (match-disjoint) split. See annotations/match_mapping.csv.
MATCH_MAPPING_CSV   = ANNOTATIONS_DIR / "match_mapping.csv"

# Generated outputs
TRAIN_JSON          = ANNOTATIONS_DIR / "train_split_match_disjoint.json"
VAL_JSON            = ANNOTATIONS_DIR / "val_split_match_disjoint.json"
TAXONOMY_CSV        = ANNOTATIONS_DIR / "full_taxonomy_metadata.csv"
METADATA_CACHE      = ANNOTATIONS_DIR / "_metadata_cache.csv"   # intermediate

# --- Dataset identity --------------------------------------------------------
DATASET_NAME        = "CricketEC"
DATASET_VERSION     = "1.0"

# --- Taxonomy ----------------------------------------------------------------
# ORDER MATTERS: the index in each list is appended to the offset below to make
# class_id. Strokes occupy ids 0..9, outcomes 10..14. Do not reorder casually —
# re-running will relabel clips if you do.
STROKE_CLASSES = [
    "cover",          # 0   (confirmed)
    "defense",        # 1   (confirmed it exists)
    "pull",           # 2   placeholder
    "hook",           # 3   placeholder
    "cut",            # 4   placeholder
    "sweep",          # 5   placeholder
    "straight_drive", # 6   placeholder
    "flick",          # 7   placeholder
    "square_drive",   # 8   placeholder
    "late_cut",       # 9   placeholder
]

OUTCOME_CLASSES = [
    "six",            # 10  (confirmed)
    "four",           # 11  placeholder position (folder shows four next to six)
    "wicket",         # 12  (confirmed)
    "dot_ball",       # 13  placeholder
    "single",         # 14  placeholder
]

STROKE_ID_OFFSET  = 0
OUTCOME_ID_OFFSET = len(STROKE_CLASSES)   # 10

SEMANTIC_STROKE   = "stroke"
SEMANTIC_OUTCOME  = "outcome"

# --- Split -------------------------------------------------------------------
# Fraction of *matches* (not clips) held out for validation. The split is made
# at the match level so no match appears in both train and val (no leakage).
VAL_FRACTION      = 0.20
SPLIT_SEED        = 42        # reproducible shuffling of matches

# Clip-id prefixes per split (matches the professor's CEC_TR_0001 / CEC_VAL_0001)
CLIPID_PREFIX = {"train": "CEC_TR_", "val": "CEC_VAL_"}

# --- Derived lookups (do not edit; built from the lists above) ---------------
def _build_maps():
    name_to_id, id_to_name, name_to_layer = {}, {}, {}
    for i, name in enumerate(STROKE_CLASSES):
        cid = STROKE_ID_OFFSET + i
        name_to_id[name] = cid
        id_to_name[cid] = name
        name_to_layer[name] = SEMANTIC_STROKE
    for i, name in enumerate(OUTCOME_CLASSES):
        cid = OUTCOME_ID_OFFSET + i
        name_to_id[name] = cid
        id_to_name[cid] = name
        name_to_layer[name] = SEMANTIC_OUTCOME
    return name_to_id, id_to_name, name_to_layer

NAME_TO_ID, ID_TO_NAME, NAME_TO_LAYER = _build_maps()
ALL_CLASSES = STROKE_CLASSES + OUTCOME_CLASSES

def layer_dir(class_name: str) -> Path:
    """Return the videos/ subfolder a class lives in."""
    return STROKE_DIR if NAME_TO_LAYER[class_name] == SEMANTIC_STROKE else OUTCOME_DIR


if __name__ == "__main__":
    # Quick sanity print so you can eyeball the id assignment.
    print(f"{DATASET_NAME} v{DATASET_VERSION} — {len(ALL_CLASSES)} classes\n")
    for cid in sorted(ID_TO_NAME):
        print(f"  {cid:>2}  {ID_TO_NAME[cid]:<16} {NAME_TO_LAYER[ID_TO_NAME[cid]]}")
