#!/usr/bin/env bash
# Run Experiment 2 (sampling ablation) and Experiment 3 (frame-count sensitivity)
# through the benchmark pipeline (benchmark/paper_configs.py): the paper's CNN-RNN,
# the fixed 2,638-clip split in data/CricketEC_split/, and the same regularisation
# as the A-E benchmark runs. The two experiments run in parallel on the GPU,
# detached from SSH; each resumes where it stopped if started again.
#
#   bash new-experiments/run_experiments.sh
#   tail -f experiments/exp2/E2-B.log      # per-epoch progress of one config
#
# Results: experiments/exp2/<config>/, experiments/exp3/<config>/ and a summary.csv
# in each. Extra flags for paper_configs.py go in $EXTRA_ARGS.
set -euo pipefail
cd "$(dirname "$0")/.."

ARGS="--workers 6 ${EXTRA_ARGS:-}"
OUT_DIR=experiments/exp2 EXTRA_ARGS="$ARGS" bash benchmark/run_all.sh E2-B E2-C E2-D E2-A
OUT_DIR=experiments/exp3 EXTRA_ARGS="$ARGS" bash benchmark/run_all.sh E3-A E3-B E3-C E3-D
