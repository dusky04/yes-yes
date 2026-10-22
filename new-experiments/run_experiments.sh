#!/usr/bin/env bash
# Run the training experiments (2: sampling ablation, 3: frame-count sensitivity)
# on the server, detached from the SSH session. The two run in parallel on the GPU.
#
#   bash new-experiments/run_experiments.sh
#   tail -f experiments_out/ex2.log experiments_out/ex3.log
#
# Needs the old-split dataset at CricketEC/{train,val,test} (2250/375/375 clips,
# 15 classes), which these scripts read. Outputs go to $OUT (default experiments_out/).
# Extra flags for both scripts go in $EXTRA_ARGS, e.g. EXTRA_ARGS="--epochs 2" for a check.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
# the scripts import dataset.py / models/ from the repo root and cec_infer from their folder
export PYTHONPATH="$PWD:$PWD/new-experiments"
OUT="${OUT:-experiments_out}"
mkdir -p "$OUT"

if [ "${_RUN_EXP_DETACHED:-}" != "1" ]; then
    _RUN_EXP_DETACHED=1 OUT="$OUT" nohup bash "$0" >> "$OUT/run_experiments.log" 2>&1 < /dev/null &
    echo "Started in background (pid $!). Logs: $OUT/ex2.log  $OUT/ex3.log  $OUT/run_experiments.log"
    exit 0
fi

run() {
    local name=$1; shift
    echo "---- $(date) | starting $name"
    # shellcheck disable=SC2086
    uv run --frozen python -u "$@" --device cuda --workers 6 ${EXTRA_ARGS:-} > "$OUT/$name.log" 2>&1 \
        && echo "---- $(date) | $name finished" \
        || echo "!!!! $(date) | $name FAILED, see $OUT/$name.log"
}

echo "==== $(date) | experiments 2 + 3 ===="
run ex2 new-experiments/experiment_sampling_ablation.py \
    --configs E2-B E2-C E2-D E2-A --out "$OUT/experiment2" &
run ex3 new-experiments/experiment_frame_sensitivity.py \
    --configs E3-A E3-B E3-C E3-D --out "$OUT/experiment3" &
wait
echo "==== $(date) | all done ===="
