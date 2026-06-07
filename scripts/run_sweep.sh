#!/usr/bin/env bash
# Multi-seed overnight sweep driver for the DASC paper.
#
# Runs ``make all`` for each (variant, seed) combination across both GPUs.
# Trainings and analyses on cuda:0 happen in one lane; those on cuda:1 in
# another; the two lanes do not synchronize so the wall-clock is
# dominated by the slower (scratch) lane. Failures in one seed do not
# halt the others.
#
# Usage:
#   SEEDS="0 1 2 3 4" DEVICE=cuda:0 DEVICE2=cuda:1 ./scripts/run_sweep.sh
#   DRY=1 ./scripts/run_sweep.sh                     # single-lane CPU smoke
set -uo pipefail

# Make sure juliaup's julia is on PATH for make's Julia targets.
[ -d "$HOME/.juliaup/bin" ] && export PATH="$HOME/.juliaup/bin:$PATH"

SEEDS=${SEEDS:-"0 1 2 3 4"}
DEVICE=${DEVICE:-cuda:0}
DEVICE2=${DEVICE2:-cuda:1}

mkdir -p logs

run_lane() {
    local variant=$1
    local device=$2
    for seed in $SEEDS; do
        tag="${variant}_seed${seed}"
        log="logs/${tag}.log"
        echo "[SWEEP $tag] $(date -Is) on $device → $log"
        make all VARIANT="$variant" SEED="$seed" DEVICE="$device" \
             > "$log" 2>&1 \
            || echo "[SWEEP $tag] FAILED (see $log)"
    done
}

if [[ -n "${DRY:-}" ]]; then
    # Local CPU smoke: just one lane, one seed.
    run_lane pretrained cpu
    exit 0
fi

run_lane pretrained "$DEVICE" &
PID0=$!
run_lane scratch "$DEVICE2" &
PID1=$!
echo "[SWEEP] lanes launched pre=$PID0 scr=$PID1  $(date -Is)"
wait $PID0 $PID1
echo "[SWEEP] all lanes finished $(date -Is)"
