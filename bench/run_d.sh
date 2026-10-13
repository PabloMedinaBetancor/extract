#!/bin/bash
# RUN-D: the development pages read with the frozen engine, then pass 2 with L (16 lines, latin+wide),
# Z (40 zones, deep) and ZC (40 zones, latin+wide: the ablation of the readers in zones needs it). Resumable: pass 1 skips pages already read with the same fingerprint, pass 2 the ones with the
# same readers' key. Launch it as a service with a memory cap so it outlives the session that started it:
#     LOG=... EXTRA_MODELS_DIR=... systemd-run --user --unit=run-d --setenv=LOG --setenv=EXTRA_MODELS_DIR --collect -p MemoryHigh=2500M -p MemoryMax=3G bash bench/run_d.sh
# Progress: $LOG (required, set by the caller), one line per step; the stage in $LOG.stage.
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export EXTRACT_MODELS_DIR="${EXTRACT_MODELS_DIR:+$EXTRACT_MODELS_DIR:}${EXTRA_MODELS_DIR:?set EXTRA_MODELS_DIR to the folder with the extra reader models}"
LOG="${LOG:?set LOG to the progress log file}"
mkdir -p "$(dirname "$LOG")"
SETS=(--set train --set val --set test --set clean --set unseen)
step() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; echo "$1" > "$LOG.stage"; }
step "pass1 start"
python3 -m bench.run_corpus "${SETS[@]}" --second-reading >> "$LOG.out" 2>&1 || { step "pass1 FAILED rc=$?"; exit 1; }
step "pass1 done"
python3 -m bench.run_pass2 "${SETS[@]}" --cell L >> "$LOG.out" 2>&1 || { step "L FAILED rc=$?"; exit 1; }
step "L done"
python3 -m bench.run_pass2 "${SETS[@]}" --cell Z >> "$LOG.out" 2>&1 || { step "Z FAILED rc=$?"; exit 1; }
step "Z done"
python3 -m bench.run_pass2 "${SETS[@]}" --cell ZC >> "$LOG.out" 2>&1 || { step "ZC FAILED rc=$?"; exit 1; }
step "ZC done"
step "RUN-D DONE"
