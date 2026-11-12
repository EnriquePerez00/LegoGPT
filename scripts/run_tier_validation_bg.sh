#!/usr/bin/env bash
# =============================================================================
# LegoGPT — Tier Validation en background (probe → fix → scale progresivo)
# =============================================================================
# Uso:
#   bash scripts/run_tier_validation_bg.sh                 # probe → fix → scale(5)
#   bash scripts/run_tier_validation_bg.sh probe           # solo probe
#   bash scripts/run_tier_validation_bg.sh all 10          # scale con 10 sets/tier
#
# El proceso queda en background. Log: outputs/tier_validation/run.log
# Estado:   outputs/tier_validation/RUNNING
# =============================================================================

set -uo pipefail

PROJECT_DIR="/Users/I764690/Code_personal/LegoGPT"
OUT_DIR="$PROJECT_DIR/outputs/tier_validation"
LOG_FILE="$OUT_DIR/run.log"
PID_FILE="$OUT_DIR/RUNNING"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

PHASE="${1:-all}"
PER_TIER="${2:-5}"

mkdir -p "$OUT_DIR"

if [[ ! -x "$VENV_PY" ]]; then
    echo "[ERROR] No se encuentra el intérprete del venv: $VENV_PY" >&2
    exit 1
fi

cd "$PROJECT_DIR" || exit 1

echo "[TierValidation] Iniciando fase='$PHASE' per_tier=$PER_TIER"
echo "[TierValidation] Log: $LOG_FILE"

# --phase probe|fix|scale|all ; --per-tier N
nohup "$VENV_PY" -u scripts/tier_validation.py \
    --phase "$PHASE" \
    --per-tier "$PER_TIER" \
    --output-dir "$OUT_DIR" \
    > "$LOG_FILE" 2>&1 &

BG_PID=$!
echo "$BG_PID" > "$PID_FILE"
echo "[TierValidation] PID background: $BG_PID"
echo "[TierValidation] Monitor: tail -f $LOG_FILE"
