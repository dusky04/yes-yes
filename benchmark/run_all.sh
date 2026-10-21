#!/usr/bin/env bash
# Train + test configurations A–E one after another, detached from the SSH
# session, so closing the terminal does not stop training.
#
#   bash benchmark/run_all.sh              # all five
#   bash benchmark/run_all.sh D E          # just some
#   tail -f runs/run_all.log               # watch progress
#
# Re-running resumes each config from runs/<config>/last.pt, and a config whose
# results.json already exists is skipped, so it is safe to start again after a
# pause or crash. Extra flags for paper_configs.py go in $EXTRA_ARGS,
# e.g.  EXTRA_ARGS="--workers 4" bash benchmark/run_all.sh
# Results go to $OUT_DIR (default runs/), e.g.  OUT_DIR=runs_v2 bash benchmark/run_all.sh B
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"  # where the uv installer puts uv

CONFIGS=("$@")
[ ${#CONFIGS[@]} -eq 0 ] && CONFIGS=(A B C D E)
OUT_DIR="${OUT_DIR:-runs}"
export OUT_DIR
mkdir -p "$OUT_DIR"

if [ "${_RUN_ALL_DETACHED:-}" != "1" ]; then
    _RUN_ALL_DETACHED=1 nohup bash "$0" "${CONFIGS[@]}" >> "$OUT_DIR/run_all.log" 2>&1 &
    echo "Started in background (pid $!) for configs: ${CONFIGS[*]}"
    echo "Log:   tail -f $OUT_DIR/run_all.log"
    echo "Stop:  kill $! && pkill -f paper_configs.py"
    exit 0
fi

echo "==== $(date) | configs: ${CONFIGS[*]} ===="
for cfg in "${CONFIGS[@]}"; do
    if [ -f "$OUT_DIR/$cfg/results.json" ]; then
        echo "---- $cfg already finished, skipping (delete $OUT_DIR/$cfg to retrain)"
        continue
    fi
    echo "---- $(date) | starting config $cfg"
    # shellcheck disable=SC2086
    PYTHONUNBUFFERED=1 uv run --frozen benchmark/paper_configs.py --config "$cfg" --resume --out-dir "$OUT_DIR" ${EXTRA_ARGS:-} \
        > "$OUT_DIR/$cfg.log" 2>&1 || echo "!!!! config $cfg FAILED, see $OUT_DIR/$cfg.log"
    tail -n 3 "$OUT_DIR/$cfg.log"
done
uv run --frozen benchmark/paper_configs.py --summarize --out-dir "$OUT_DIR"
echo "==== $(date) | all done ===="
