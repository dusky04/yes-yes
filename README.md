# CricketEC: A Benchmark for Cricket Event & Stroke Classification

CricketEC is a video classification benchmark built from short broadcast cricket
clips. Each clip is labelled into one of **14 classes** spanning two semantic
layers — the *biomechanical batting stroke* being played, and the *match-event
outcome* of the delivery — enabling study of fine-grained action recognition and
event understanding in a single, consistently annotated corpus.

## Dataset at a glance

| | |
|---|---|
| Total clips | **2,488** |
| Classes | **14** (10 stroke + 4 outcome) |
| Splits | train / validation / test (≈ 70 / 15 / 15) |
| Clip format | `.avi`, native broadcast resolution and frame rate |
| Per-clip metadata | duration, frame count, fps, resolution, chirality, camera perspective, source match |

## Taxonomy

**Stroke classes (10)** — the batting shot played:
`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`, `square_cut`,
`straight`, `sweep`

**Outcome classes (4)** — the match-event outcome:
`six`, `four`, `wicket`, `catch`

Class ids are `0–9` for strokes and `10–13` for outcomes, fixed in `config.py`.

---

## How the dataset is organized

Two distinct notions of "grouping" are used, and they are represented
differently — this distinction matters when working with the data:

- **By class (stroke vs outcome)** — represented as **folders** under `videos/`.
  Every clip lives in exactly one class folder.
- **By split (train / validation / test)** — represented as **JSON files** in
  `annotations/`, each listing the clips that belong to that split. Clips are
  **not** duplicated into split folders; a single copy of each clip is referenced
  by the split files. Re-splitting therefore only rewrites the JSONs and never
  moves a video.

```
CricketEC_Dataset/
├── videos/
│   ├── stroke_classes/<class>/<class>_NNNN.avi     # 10 stroke folders
│   └── outcome_classes/<class>/<class>_NNNN.avi    # 4 outcome folders
├── annotations/
│   ├── train_split_match_disjoint.json             # training split (clip list)
│   ├── val_split_match_disjoint.json               # validation split
│   ├── test_split_match_disjoint.json              # test split
│   └── full_taxonomy_metadata.csv                  # per-class summary table
├── tools/                                          # dataset build pipeline (01–06)
├── baselines/                                      # reference benchmark code
├── annotate.py                                     # interactive annotation tool
├── merge_annotations.py                            # writes annotations into the JSONs
├── config.py                                       # taxonomy, paths, split ratios
└── run_pipeline.py                                 # runs the full build
```

> The video files are distributed separately (see **Obtaining the videos**); the
> repository ships the taxonomy, annotations, build pipeline and baseline code.

---

## Annotation schema

Each split JSON has top-level metadata (`split_name`, `dataset_name`, `version`,
`total_clips`) and a `clips` array. Each clip:

```jsonc
{
  "clip_id": "CEC_TR_0001",                 // unique id, prefixed per split
  "file_path": "videos/stroke_classes/cover/cover_0012.avi",
  "class_name": "cover",
  "class_id": 0,
  "semantic_layer": "stroke",               // "stroke" or "outcome"
  "source_match_id": "IND_vs_AUS_2023_T20_M01",  // source broadcast match (for leakage control)
  "duration_sec": 2.72,
  "num_frames": 82,
  "fps": 30.0,
  "resolution": [1920, 1080],
  "chirality": "right-handed",              // batter handedness — stroke clips
  "camera_perspective": "side_on"           // camera angle (see Annotation)
}
```

`full_taxonomy_metadata.csv` aggregates, per class: `total_clips`, `train_clips`,
`val_clips`, `test_clips`, duration statistics, dominant fps and dominant camera
perspective.

---

## Annotation

Beyond the automatically extracted metadata, each clip carries two expert-assigned
attributes:

- **`chirality`** — batter handedness (`right-handed` / `left-handed`), annotated
  for stroke clips.
- **`camera_perspective`** — the dominant camera angle, annotated for all clips,
  using a layer-specific vocabulary:

  | Layer | `camera_perspective` vocabulary |
  |---|---|
  | Stroke | `front_on`, `side_on`, `reverse_angle`, `high_angle` |
  | Outcome | `outfield_wide`, `main_pitch`, `close_up_replay`, `side_on` |

Labels were assigned by frame-by-frame review of each clip. The labelling interface
(`annotate.py`) and the routine that writes labels into the split files
(`merge_annotations.py`) are included for reproducibility and extension.

---

## Obtaining the videos

The clips are distributed as a separate archive (video data is kept out of version
control for size). Place the per-class folders under `videos/stroke_classes/` and
`videos/outcome_classes/` so the paths match the `file_path` entries in the split
JSONs. Alternatively, rebuild the organized `videos/` tree from the raw clips with
the pipeline below.

---

## Reproducing / building the dataset

The `tools/` pipeline turns raw clips into the organized dataset. It is driven
entirely by `config.py` (taxonomy, paths, split ratios in one place).

```bash
pip install pandas opencv-python numpy
# with raw clips under raw_clips/ (per-class subfolders):
python run_pipeline.py
```

| Stage | Script | Function |
|------|--------|----------|
| 01 | `rename_clips.py` | Rename clips to `<class>_NNNN`, with a reversible manifest |
| 02 | `sort_into_folders.py` | Sort clips into their stroke / outcome class folders |
| 03 | `extract_metadata.py` | Read duration, frame count, fps, resolution (ffprobe → OpenCV) |
| 04 | `make_split.py` | Produce the match-disjoint train / val / test split |
| 05 | `generate_annotations.py` | Emit the three split JSONs |
| 06 | `generate_taxonomy_csv.py` | Aggregate the per-class summary table |

### Match-disjoint splitting

Clips from one broadcast match share players, kit, pitch, lighting and camera
setup; allowing a match to appear in more than one split lets a model exploit
match identity rather than the target class, inflating reported accuracy. Splits
are therefore **match-disjoint**: using `source_match_id`, every clip of a given
match is assigned entirely to a single split, and the pipeline asserts that no
match spans two splits. The split file names reflect this (`*_match_disjoint`).

---

## Baselines

`baselines/` provides reference code: frame sampling (`extract_frames.py`,
uniform or motion/pixel-intensity based) and a frame-averaged ResNet classifier
(`train_baseline.py`) that reads the split JSONs and reports overall and per-class
accuracy on the leak-controlled splits.

```bash
pip install torch torchvision
python baselines/train_baseline.py \
    --train annotations/train_split_match_disjoint.json \
    --val   annotations/val_split_match_disjoint.json \
    --num-frames 16 --strategy uniform
```

---

## Reproducibility

Splits are deterministic given a fixed seed (`SPLIT_SEED` in `config.py`), and the
full dataset — class organization, metadata, splits and summary table — can be
regenerated from the raw clips with `run_pipeline.py`. The taxonomy, split ratios
and paths are all defined in `config.py`.
