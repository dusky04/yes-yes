# CricketEC: A Benchmark for Cricket Event & Stroke Classification

CricketEC is a video classification benchmark built from short broadcast cricket
clips. Each clip is labelled into one of **15 classes** spanning two semantic
layers — the *biomechanical batting stroke* being played, and the *match-event
outcome* of the delivery — enabling study of fine-grained action recognition and
event understanding in a single, consistently annotated corpus.

The repository ships the taxonomy, the full annotations, the dataset build
pipeline, the CNN-RNN models, and the complete experiment suite (Experiments
1–7). The video data and trained weights are distributed separately (links below).

---

## Downloads

| Resource | Description | Link |
|---|---|---|
| **Annotated database** | Full annotated corpus (2,638 clips) with all per-clip metadata and the match-disjoint train/val/test split. | [Google Drive](https://drive.google.com/drive/folders/1w4ykJM8tRzlQFMJb1CVLBqjBiGqJRqbl?usp=sharing) |
| **Experimental database** | The exact partition used to run Experiments 1–7 (2,623 clips; fixed seed, 25 val + 25 test per class). Ships with `split_canonical.json`. | [Google Drive](https://drive.google.com/drive/folders/1rhzw17lMDRh0QKeSYnZ7gtHmnjnlBTxw?usp=sharing) |
| **Pretrained weights** | The five trained CNN-RNN checkpoints (Configurations A–E) evaluated in the paper. | [Google Drive](https://drive.google.com/drive/folders/18Crru0dQAubrQppi8_mi6CuFmeJna3q4?usp=sharing) |

Both databases are laid out identically:

```
<database>/<split>/<stroke_classes|outcome_classes>/<class>/<clip>.avi
```

so any training or evaluation script can point at either one unchanged.

---

## Dataset at a glance

| | Annotated database | Experimental database |
|---|---|---|
| Total clips | **2,638** | **2,623** |
| Train / val / test | 1,852 / 393 / 393 | 1,873 / 375 / 375 |
| Classes | **15** (10 stroke + 5 outcome) | **15** (10 stroke + 5 outcome) |
| Split protocol | match-disjoint (stratified) | fixed seed, 25 val + 25 test per class |
| Clip format | `.avi`, native broadcast resolution and frame rate | same |
| Per-clip metadata | duration, frames, fps, resolution, chirality, camera perspective, stroke timing, bowler handedness, QC status | same |

The **annotated database** is the complete, fully labelled release. The
**experimental database** is the frozen partition on which every reported result
(Experiments 1–7) was produced; it is provided so the experiments are exactly
reproducible and a reviewer can cross-check each clip against `split_canonical.json`.

---

## Taxonomy

**Stroke classes (10)** — the batting shot played:
`cover`, `defense`, `flick`, `hook`, `late_cut`, `lofted`, `pull`, `square_cut`,
`straight`, `sweep`

**Outcome classes (5)** — the match-event outcome:
`six`, `four`, `wicket`, `catch`, `lbw`

Class ids are `0–9` for strokes and `10–14` for outcomes, fixed in `config.py`
(`lbw` = `14`). The id ordering is stable, so the alphabetical class→index mapping
used by the loaders (`utils.get_classes`) is deterministic across runs.

---

## How the dataset is organized

Two notions of grouping are used, and they are represented differently:

- **By class (stroke vs outcome)** — represented as **folders**. Every clip lives
  in exactly one class folder, nested under its semantic layer
  (`stroke_classes/` or `outcome_classes/`).
- **By split (train / val / test)** — represented both as the top-level split
  folders in the distributed databases and as **JSON files** in `annotations/`,
  each listing the clips in that split. Re-splitting only rewrites the JSONs.

```
CricketEC/
├── train/
│   ├── stroke_classes/<class>/<class>_NNNN.avi      # 10 stroke folders
│   └── outcome_classes/<class>/<class>_NNNN.avi     # 5 outcome folders
├── val/   …                                         # same structure
└── test/  …                                         # same structure
```

Repository layout (code + annotations; videos distributed separately):

```
CricketEC/
├── annotations/
│   ├── train_split_match_disjoint.json              # annotated-DB split (clip list)
│   ├── val_split_match_disjoint.json
│   ├── test_split_match_disjoint.json
│   ├── split_canonical.json                         # experimental-DB split map
│   └── full_taxonomy_metadata.csv                   # per-class summary table
├── models/                                          # CNN-RNN architectures
├── tools/                                           # dataset build pipeline (01–06)
├── baselines/                                       # reference benchmark code
├── annotate.py / annotate_lbw.py                    # interactive annotation tools
├── merge_annotations.py                             # writes annotations into the JSONs
├── evaluate_test.py                                 # test-set benchmark (Experiment 1)
├── experiment_*.py                                  # Experiments 2–7
├── config.py                                        # taxonomy, paths, split ratios
└── run_pipeline.py                                  # runs the full dataset build
```

---

## Annotation schema

Each split JSON has top-level metadata (`split_name`, `dataset_name`, `version`,
`total_clips`) and a `clips` array. Every clip carries the full field set (values
are `null` where an attribute does not apply):

```jsonc
{
  "clip_id": "CEC_TR_0107",
  "file_path": "videos/stroke_classes/cover/cover_0002.avi",
  "class_name": "cover",
  "class_id": 0,
  "semantic_layer": "stroke",                 // "stroke" or "outcome"
  "source_match_id": null,                    // source broadcast match (leakage control)
  "duration_sec": 2.08,
  "num_frames": 52,
  "fps": 25.0,
  "resolution": [224, 224],
  "chirality": "left-handed",                 // batter handedness
  "camera_perspective": "Batsman-Centric",    // dominant camera angle (see below)
  "start_frame": 1,                           // action extent, in frames …
  "end_frame": 17,
  "start_time_ms": 40,                        // … and in milliseconds
  "end_time_ms": 680,
  "start_time_of_stroke": 0.04,               // seconds from clip start
  "end_time_of_stroke": 0.68,
  "bowler_handedness": null,                  // "right-arm" / "left-arm"
  "qc_status": null                           // quality-control flag
}
```

**Attributes**

- **`chirality`** — batter handedness: `right-handed` / `left-handed`.
- **`camera_perspective`** — the dominant broadcast camera angle, from a
  layer-aware vocabulary:

  | Perspective | Typical layer |
  |---|---|
  | `Batsman-Centric` | batting strokes |
  | `Pitch / Umpire Perspective` | outcomes (bowler's-end view) |
  | `Ground Field Tracking` | outcomes |
  | `High-Angle Ball Tracking` | outcomes |
  | `Pitch / Stump Close-up` | outcomes |
  | `Fielder / Boundary View` | outcomes |

- **`start_frame` / `end_frame` / `start_time_ms` / `end_time_ms`** — the temporal
  extent of the action within the clip, given in both frames and milliseconds;
  `start_time_of_stroke` / `end_time_of_stroke` carry the same extent in seconds.
- **`bowler_handedness`** — `right-arm` / `left-arm`.
- **`qc_status`** — expert quality-control flag (e.g. `PASS`, `FAIL_OCCLUSION`).

`full_taxonomy_metadata.csv` aggregates, per class: `total_clips`, `train_clips`,
`val_clips`, `test_clips`, duration statistics, dominant fps and dominant camera
perspective.

Labels were assigned by frame-by-frame review of each clip. The labelling
interfaces (`annotate.py`, `annotate_lbw.py`) and the routine that writes labels
into the split files (`merge_annotations.py`) are included for reproducibility and
extension.

---

## Models

The classifiers follow a **CNN backbone → temporal head → attention pooling → FC**
design (`models/`):

- **Spatial backbone** — `ResNet-18`, `ResNet-34`, or `EfficientNet-B0`, pretrained
  on ImageNet, with the upper blocks fine-tuned and a per-frame `BatchNorm`.
- **Temporal head** — a **bidirectional GRU or LSTM** over the per-frame features,
  followed by `LayerNorm` (`models/gru.py`, `models/cnn.py`).
- **Pooling** — learned **attention pooling** across time, with a parameter-free
  **mean-pooling** variant also provided.
- **Classifier head** — `Linear → BatchNorm → ReLU → Dropout → Linear` over the 15
  classes.

Each clip is represented by a fixed number of sampled frames (uniform or
pixel-intensity sampling; see `dataset.py`).

---

## Experiments

All reported results are produced on the **experimental database** with the five
trained checkpoints (A–E). Experiment 1 is the headline benchmark; 2–7 analyse it
from complementary angles.

| # | Script | What it measures |
|---|--------|------------------|
| 1 | `evaluate_test.py` | Held-out **test-set benchmark** of Configurations A–E (accuracy, macro/weighted P-R-F1, per-class scores). No training, no re-splitting. |
| 2 | `experiment_sampling_ablation.py` | **Frame-sampling ablation** — uniform vs pixel-intensity, for both GRU and LSTM heads, everything else fixed. |
| 3 | `experiment_frame_sensitivity.py` | **Frame-count sensitivity** — how accuracy and cost vary with the number of sampled frames. |
| 4 | `experiment_camera_perspective.py` | **Camera-perspective robustness** — accuracy grouped by (semantic layer, camera perspective); flags the hardest viewpoints. |
| 5 | `experiment_model_efficiency.py` | **Computational efficiency & complexity** — parameters, GFLOPs, latency, throughput, peak memory, paired with accuracy. |
| 6 | `experiment_semantic_error_analysis.py` | **Semantic-layer & misclassification analysis** — per-layer accuracy, 15×15 confusion matrix, most-confused pairs, within- vs cross-layer errors. |
| 7 | `experiment_statistical_analysis.py` | **Statistical reliability** — bootstrap confidence intervals, McNemar's exact test (best vs each other model), Holm correction. |

`cec_infer.py` is the shared inference helper used by Experiments 4–7 (it reuses
Experiment 1's predictions when available, and falls back to live inference).

### Running the experiments

```bash
# 1 — test-set benchmark (all five configs)
python evaluate_test.py --configs A B C D E \
    --test annotations/test_split_match_disjoint.json \
    --data CricketEC_experiment --weights weights/

# 2 / 3 — ablations (configs run individually and accumulate into one results file)
python experiment_sampling_ablation.py --configs E2-A
python experiment_frame_sensitivity.py --configs E3-A

# 4–7 — analyses over the trained models
python experiment_camera_perspective.py
python experiment_model_efficiency.py
python experiment_semantic_error_analysis.py
python experiment_statistical_analysis.py
```

---

## Pretrained weights

Download the checkpoints from the [weights folder](https://drive.google.com/drive/folders/18Crru0dQAubrQppi8_mi6CuFmeJna3q4?usp=sharing)
and place them under `weights/` using the names in `evaluate_test.py`.

| Config | Backbone | Temporal head | Frames | Sampling |
|--------|----------|---------------|:------:|----------|
| **A** | ResNet-34 | LSTM | 16 | uniform |
| **B** | EfficientNet-B0 | LSTM | 16 | uniform |
| **C** | EfficientNet-B0 | GRU | 16 | uniform |
| **D** | ResNet-34 | GRU | 24 | pixel-intensity |
| **E** | ResNet-34 | LSTM | 16 | pixel-intensity |

All five were trained on the 15-class taxonomy. `evaluate_test.py` auto-detects
each checkpoint's RNN hidden size / layer count from its weights, so the correct
architecture is rebuilt automatically at load time.

---

## Reproducing / building the dataset

The `tools/` pipeline turns raw clips into the organized dataset. It is driven
entirely by `config.py` (taxonomy, paths, split ratios in one place).

```bash
pip install -r requirements.txt
# with raw clips under raw_clips/ (flat or per-class subfolders):
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

Annotations are folded into the split JSONs with `merge_annotations.py` (reads the
annotation CSVs and writes chirality, camera perspective, timing, bowler handedness
and QC status onto every clip).

### Splitting protocol

The intended partition is **match-disjoint**: clips from one broadcast match share
players, kit, pitch, lighting and camera setup, so letting a match appear in more
than one split lets a model exploit match identity rather than the target class.
Using `source_match_id`, every clip of a match is assigned to a single split and
the pipeline asserts that no match spans two splits. Where reliable per-clip match
identifiers are unavailable, `source_match_id` is a synthetic placeholder and the
partition is a seeded, class-stratified split; supplying real match identifiers in
`annotations/match_mapping.csv` upgrades it to genuine match-disjointness with no
other change. Splits are deterministic given `SPLIT_SEED` in `config.py`.

---

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `opencv-python`,
`scikit-learn`, `matplotlib` (plus optional `thop` for FLOP counting in Experiment
5 and `decord` for fast video decoding). `ffmpeg` is used by the metadata stage
when available, with OpenCV as the fallback.

---

## Baselines

`baselines/` provides reference code: frame sampling (`extract_frames.py`, uniform
or motion/pixel-intensity based) and a frame-averaged ResNet classifier
(`train_baseline.py`) that reads the split JSONs and reports overall and per-class
accuracy on the leak-controlled splits.

---

## Citation

If you use CricketEC, please cite the accompanying paper. A BibTeX entry will be
added here on publication.
