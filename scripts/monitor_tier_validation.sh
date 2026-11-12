#!/usr/bin/env bash
# =============================================================================
# LegoGPT — Monitor de validación por tiers (resultados por pantalla)
# =============================================================================
# Muestra el progreso del run en background: estado, última línea del log,
# y (cuando existen) los resultados por tier extraídos del reporte JSON.
#
# Uso:
#   bash scripts/monitor_tier_validation.sh          # snapshot
#   bash scripts/monitor_tier_validation.sh watch    # refresco continuo (10s)
# =============================================================================

set -uo pipefail

PROJECT_DIR="/Users/I764690/Code_personal/LegoGPT"
OUT_DIR="$PROJECT_DIR/outputs/tier_validation"
LOG_FILE="$OUT_DIR/run.log"
PID_FILE="$OUT_DIR/RUNNING"
VENV_PY="$PROJECT_DIR/.venv/bin/python"

MODE="${1:-snapshot}"

snapshot() {
    echo "══════════════════════════════════════════════════════════════════"
    echo "  TIER VALIDATION — $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════════════════════════"

    if [[ -f "$PID_FILE" ]]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            echo "  Estado: 🟢 EN CURSO (PID $PID)"
        else
            echo "  Estado: ⚪  TERMINADO (PID $PID)"
        fi
    else
        echo "  Estado: ⚪  sin run activo"
    fi

    echo ""
    echo "─── ÚLTIMAS LÍNEAS DEL LOG ───"
    if [[ -f "$LOG_FILE" ]]; then
        tail -25 "$LOG_FILE"
    else
        echo "  (sin log todavía)"
    fi

    echo ""
    echo "─── RESUMEN POR TIER ───"
    "$VENV_PY" - <<'PYEOF' 2>/dev/null || echo "  (reportes aún no disponibles)"
import glob, json, os

out_dir = "outputs/tier_validation"
reports = []
for pattern in ("probe_report.json", "scale_report_*.json"):
    reports.extend(sorted(glob.glob(os.path.join(out_dir, pattern))))

if not reports:
    raise SystemExit(1)

for rp in reports:
    with open(rp) as f:
        data = json.load(f)
    s = data.get("summary", {})
    print(f"\n  ▸ {os.path.basename(rp)}  ({data.get('phase')}, "
          f"{data.get('items_evaluated')} sets)")
    print(f"    Global: {s.get('passed')}/{s.get('total')} "
          f"({s.get('pass_rate_percent', 0):.1f}%) | "
          f"{s.get('total_elapsed_sec', 0):.1f}s | "
          f"{s.get('total_incidents', 0)} incidentes")
    for key, agg in (s.get("by_combo") or {}).items():
        flag = "OK " if agg["pass_rate_percent"] == 100 else "ERR"
        print(f"      [{flag}] {key:<26} {agg['passed']}/{agg['n']} "
              f"({agg['pass_rate_percent']:>5.1f}%) | "
              f"survival {agg['mean_survival_ratio']*100:>5.1f}% | "
              f"{agg['mean_elapsed_sec']:>6.2f}s/set")
PYEOF

    echo ""
    echo "─── CAUSAS RAÍZ (top) ───"
    "$VENV_PY" - <<'PYEOF' 2>/dev/null || true
import glob, json, os
out_dir = "outputs/tier_validation"
for rp in sorted(glob.glob(os.path.join(out_dir, "probe_report.json"))):
    with open(rp) as f:
        data = json.load(f)
    rc = data.get("root_causes", {})
    causes = rc.get("causes", {})
    if causes:
        print(f"  {os.path.basename(rp)}:")
        for cause, cnt in list(causes.items())[:6]:
            print(f"    {cnt:>4}x  {cause}")
PYEOF
    echo ""
}

if [[ "$MODE" == "watch" ]]; then
    while true; do
        clear
        snapshot
        sleep 10
    done
else
    snapshot
fi
