#!/bin/bash
# ====================================================================
# One-click dataset build for macOS.
# Double-click this file in Finder (if macOS blocks it: right-click ->
# Open -> Open). It finds your CricketEC clips, installs the few Python
# packages needed, and builds the whole annotated dataset.
# Your original clips are COPIED, never moved or changed.
# ====================================================================
set -e
cd "$(dirname "$0")"          # run from the repo folder, wherever it lives
echo "==> Working in: $(pwd)"

# 1. locate the raw clips (the CricketEC folder with class subfolders)
CLIPS=""
for guess in "$HOME/Downloads/CricketEC" "$HOME/Desktop/CricketEC" "../CricketEC" "./CricketEC"; do
  if [ -d "$guess" ]; then CLIPS="$guess"; break; fi
done
if [ -z "$CLIPS" ]; then
  echo "!! Could not find your CricketEC clips folder."
  echo "   Put the unzipped CricketEC folder in your Downloads, then run again."
  read -p "Press Return to close." ; exit 1
fi
echo "==> Found clips: $CLIPS"

# 2. python + packages (only what the build needs; torch is only for baselines)
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
  echo "!! python3 not found. Install it from https://www.python.org/downloads/ and run again."
  read -p "Press Return to close." ; exit 1
fi
echo "==> Installing Python packages (pandas, opencv, numpy)…"
$PY -m pip install --quiet --user pandas opencv-python numpy || \
  $PY -m pip install --quiet --break-system-packages pandas opencv-python numpy

# 3. copy clips into raw_clips/ (non-destructive) and run the pipeline
echo "==> Copying clips into raw_clips/ (originals untouched)…"
mkdir -p raw_clips
cp -R "$CLIPS"/* raw_clips/

echo "==> Running the pipeline…"
$PY run_pipeline.py --allow-no-match

echo ""
echo "==> DONE. Your annotated dataset is in this folder:"
echo "    annotations/train_split_match_disjoint.json"
echo "    annotations/val_split_match_disjoint.json"
echo "    annotations/full_taxonomy_metadata.csv"
echo "    videos/stroke_classes/… and videos/outcome_classes/…"
read -p "Press Return to close."
