#!/usr/bin/env bash
# Experiments 1 and 4-7 on the five benchmark models A-E (runs_v3/<A..E>/best.pt).
# 1 collects the per-clip test predictions; 4, 6 and 7 analyse them (no GPU);
# 5 profiles the five models on the GPU. Run on the server once A-E have finished:
#
#   bash new-experiments/run_analysis.sh
#   bash new-experiments/run_analysis.sh --models B     # quick check on a subset
#
# Outputs: experiments/exp1_test_eval, exp4_camera_perspective, exp5_model_efficiency,
#          exp6_semantic_error_analysis, exp7_statistical_analysis (+ one log each).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$PWD:$PWD/new-experiments"
export CEC_RUNS_DIR="${CEC_RUNS_DIR:-$PWD/runs_v3}"
E=experiments
MODELS=()
[ "${1:-}" = "--models" ] && { shift; MODELS=("$@"); }
M=()
[ ${#MODELS[@]} -gt 0 ] && M=(--models "${MODELS[@]}")
PRED="$E/exp1_test_eval/test_predictions.csv"
mkdir -p "$E"

step() {
    local name=$1; shift
    echo "---- $(date +%T) | $name"
    if uv run --frozen "$@" > "$E/$name.log" 2>&1; then
        tail -n 4 "$E/$name.log"
    else
        echo "!!!! $name FAILED, see $E/$name.log"; tail -n 15 "$E/$name.log"
    fi
}

step exp1_test_eval python new-experiments/evaluate_test.py --out "$E/exp1_test_eval" "${M[@]}"
step exp4_camera_perspective python new-experiments/experiment_camera_perspective.py \
    --annotations annotations/manual_annotations.csv --predictions "$PRED" \
    --out "$E/exp4_camera_perspective" "${M[@]}"
step exp5_model_efficiency --with thop python new-experiments/experiment_model_efficiency.py \
    --device cuda --predictions "$PRED" --out "$E/exp5_model_efficiency" "${M[@]}"
step exp6_semantic_error_analysis python new-experiments/experiment_semantic_error_analysis.py \
    --predictions "$PRED" --out "$E/exp6_semantic_error_analysis" "${M[@]}"
step exp7_statistical_analysis python new-experiments/experiment_statistical_analysis.py \
    --bootstrap 5000 --predictions "$PRED" --out "$E/exp7_statistical_analysis" "${M[@]}"
echo "==== $(date +%T) | analysis done"
