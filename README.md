# CricketEC Dataset

A benchmark for cricket **E**vent & stroke **C**lassification from short broadcast
video clips. Clips are labelled across two semantic layers:

- **Stroke classes** (10) — biomechanical batting strokes (`cover`, `defense`, …)
- **Outcome classes** (5) — match-event outcomes (`six`, `four`, `wicket`, …)

> **Status note.** The video clips are distributed via Google Drive (see
> [Hosting the videos](#hosting-the-videos)); this repository ships the dataset
> *structure*, the *metadata/annotations*, the *build pipeline*, and the
> *baseline code*. The `chirality` and `camera_perspective` annotation fields,
> and `dominant_perspective` in the taxonomy table, are produced by a separate
> annotation pass and are left empty/null until that pass is complete.

---

## Repository layout

```
CricketEC_Dataset/
├── README.md                      # this file
├── config.py                      # SINGLE SOURCE OF TRUTH: classes, paths, split ratio
├── requirements.txt
├── run_pipeline.py                # runs the full build (stages 01–06)
├── annotations/
│   ├── train_split_match_disjoint.json   # generated
│   ├── val_split_match_disjoint.json     # generated
│   ├── full_taxonomy_metadata.csv        # generated (paper Table 2)
│   └── match_mapping.csv                 # YOU fill: clip_stem -> source_match_id
├── videos/
│   ├── stroke_classes/<class>/<class>_NNNN.mp4
│   └── outcome_classes/<class>/<class>_NNNN.mp4
├── baselines/
│   ├── extract_frames.py          # uniform / pixel-intensity frame sampling
│   └── train_baseline.py          # PyTorch frame-average ResNet baseline
└── tools/                         # the build pipeline (run via run_pipeline.py)
    ├── 01_rename_clips.py
    ├── 02_sort_into_folders.py
    ├── 03_extract_metadata.py
    ├── 04_make_split.py
    ├── 05_generate_annotations.py
    └── 06_generate_taxonomy_csv.py
```

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Put the downloaded clips in ./raw_clips/  (flat, or in per-class subfolders)
# 2. Build everything:
python3 run_pipeline.py
```

If a stage needs input from you it stops with a clear message (most commonly the
match mapping in stage 04). Fix it and resume:

```bash
python3 run_pipeline.py --from 04
```

---

## The build pipeline (what each stage does)

| Stage | Script | Does |
|------|--------|------|
| 01 | `rename_clips.py` | Renames raw clips to `<class>_NNNN.mp4`, writes a reversible `rename_manifest.csv`. |
| 02 | `sort_into_folders.py` | Moves clips into `videos/stroke_classes/<class>/` or `.../outcome_classes/<class>/`. |
| 03 | `extract_metadata.py` | Reads real `duration`, `num_frames`, `fps`, `resolution` per clip (ffprobe → OpenCV fallback). |
| 04 | `make_split.py` | **Match-disjoint** train/val split (see below). Needs `match_mapping.csv`. |
| 05 | `generate_annotations.py` | Writes the two split JSONs in the required schema. |
| 06 | `generate_taxonomy_csv.py` | Aggregates `full_taxonomy_metadata.csv` (Table 2) + a class-imbalance check. |

Everything keys off `config.py`. Change a class name or the val fraction there,
in one place, and re-run.

---

## Why the split is "match-disjoint" (and why it matters)

Clips from the same broadcast match share players, kit, pitch, lighting and
camera setup. If clips from one match land in *both* train and val, a model can
get a high val score by recognising the **match** rather than the **stroke** —
the score looks great and means nothing. That is **data leakage**.

So the split is made at the **match** level: every clip of a given match goes
entirely to train or entirely to val. This is exactly why the schema carries
`source_match_id`, and why the files are named `*_match_disjoint`. Stage 04 also
asserts, after assignment, that no match appears on both sides.

`match_mapping.csv` (clip → match) is the one input a script cannot derive from
the pixels, so you provide it. Stage 04 writes a blank template listing every
clip the first time you run it.

---

## Annotation schema (split JSONs)

```jsonc
{
  "split_name": "train_match_disjoint",
  "dataset_name": "CricketEC",
  "version": "1.0",
  "total_clips": 2114,
  "clips": [
    {
      "clip_id": "CEC_TR_0001",
      "file_path": "videos/stroke_classes/cover/cover_0012.mp4",
      "class_name": "cover",
      "class_id": 0,
      "semantic_layer": "stroke",
      "source_match_id": "IND_vs_AUS_2023_T20_M01",
      "duration_sec": 2.72,
      "num_frames": 82,
      "fps": 30.0,
      "resolution": [1920, 1080],
      "chirality": null,            // annotation pass
      "camera_perspective": null    // annotation pass
    }
  ]
}
```

`total_clips` always equals the real length of `clips`. `class_id`: strokes
`0–9`, outcomes `10–14`.

---

## Baselines

```bash
# Frame-average ResNet-18, leak-free match-disjoint splits
python3 baselines/train_baseline.py \
    --train annotations/train_split_match_disjoint.json \
    --val   annotations/val_split_match_disjoint.json \
    --num-frames 16 --strategy uniform --epochs 5 --batch-size 8
```

`extract_frames.py` offers two sampling strategies: `uniform` (evenly spaced) and
`pixel` (frames with the largest pixel-intensity change, biased toward motion).
The baseline prints overall **and per-class** val accuracy — the honest view when
classes are imbalanced.

The baseline is intentionally simple: a benchmark needs a floor for future
temporal models to beat. If a frame-average model already scores high, the task
may be largely solvable from single frames.

---

## Hosting the videos

The clips are **not** committed to git (`.gitignore` excludes them): GitHub
rejects files over 100 MB and the repo would bloat. Two supported options:

1. **Drive + structure (default).** Keep clips in Drive, link them here, commit
   only the structure, scripts and metadata.
2. **Git LFS.** `git lfs install && git lfs track "videos/**/*.mp4"`, commit
   `.gitattributes`, then add the videos. Use this only if the videos must be
   versioned in-repo.

---

## Reproducing the dataset from scratch

```bash
python3 config.py            # sanity-check the taxonomy / class ids
python3 run_pipeline.py      # 01 → 06
```

Splits are reproducible: match shuffling is seeded (`SPLIT_SEED` in `config.py`).
