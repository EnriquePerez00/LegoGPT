---
name: agentic-io-pipeline
description: Ejecuta el pipeline independiente multi-agente de inferencia de grafos 3D (.io) a partir de manuales LEGO (.pdf), con auditoría causal contra Ground Truth (.io) y enrutamiento óptimo de modelos LLM por coste (flash_lite, flash, pro).
---

# Skill: Pipeline Multi-Agente de Inferencia PDF $\rightarrow$ Grafo `.io`

Utiliza esta skill cuando el usuario solicite inferir un grafo de construcción 3D (`.io` / `.ldr`) desde un manual `.pdf`, auditar errores de colocación frente a archivos Ground Truth `.io` (como el corpus de ~300 parejas `.pdf` + `.io`), o coordinar los 6 agentes especializados desde Antigravity.

> **Independencia garantizada**: Este pipeline reside en `src/agentic_pipeline/` y `scripts/run_agentic_pipeline.py`, manteniendo intacto el pipeline clásico en `src/manual_connection_graph.py`.

---

## 1. Enrutamiento Óptimo de Modelos LLM en Antigravity (`invoke_subagent`)

Al invocar subagentes desde Antigravity mediante `invoke_subagent`, utiliza siempre el parámetro `Model` asignado en esta tabla para minimizar el coste:

| Subagente (`TypeName`) | `Model` Óptimo | Rol |
| :--- | :--- | :--- |
| `lego-context-agent` | `flash_lite` | Analiza metadatos de la página PDF (multiplicadores `2x`/`4x`, *callouts*, iconos de rotación/inversión `flip`) y pose inicial de cámara. |
| `lego-identifier-agent` | `flash` | Cruza las piezas añadidas en la página con la geometría 3D LDraw (`LDrawGeometryResolver`) y los conectores libres en $G_{t-1}$ (`ConnectorOccupancyMap`) para proponer hipótesis 3D. |
| `lego-guardian-agent` | `flash_lite` | Valida determinísticamente `Stud-to-Socket`, cero colisión sólida OBB (`SolidCollisionValidator`), exclusividad de studs y regla de suelo ($Y \le 0.0$, máx 4 piezas base). |
| `lego-placer-agent` | `flash` | Monta los grafos candidatos, los renderiza en 3D (`StepAxonometricRenderer`) y compara la **Zona de Interés (ROI) con margen de holgura** (incluyendo piezas vecinas) contra la imagen 2D del PDF en GPU MPS. |
| `lego-supervisor-agent` | `flash` | Normaliza el suelo y empaqueta el archivo `.io` final para BrickLink Studio (no implementa beam search entre pasos). |
| `lego-gt-auditor-agent` | `pro` | Compara el grafo inferido contra el grafo Ground Truth `.io`, diagnostica la causa raíz de cualquier error (`IDENTIFIER_RECALL_MISS`, `GUARDIAN_FALSE_REJECTION`, `PLACER_ROI_RANKING_ERROR`, `FLIP_OR_CALLOUT_DESYNC`) y guarda el *fix* en `data/agentic_fixes/learned_rules.json`. |

---

## 2. Comandos de Ejecución Directa

### A. Autotest y Verificación End-to-End de los 6 Agentes
```bash
python3 scripts/run_agentic_pipeline.py --self-test
```

### B. Inferencia de un Manual `.pdf` hacia `.io`
```bash
python3 scripts/run_agentic_pipeline.py --set-num 75295-1 --pdf data/manuals/75295-1.pdf --output-dir outputs/agentic_pipeline
```

### C. Inferencia + Auditoría Causal con Ground Truth `.io` (Auto-Fix Activo)
```bash
python3 scripts/run_agentic_pipeline.py --set-num 75295-1 --pdf data/manuals/75295-1.pdf --gt-io data/mocs/75295-1.io --output-dir outputs/agentic_pipeline
```

### D. Procesamiento en Lote del Corpus (~300 Parejas `.pdf` + `.io`)
```bash
python3 scripts/run_agentic_pipeline.py --corpus-dir data/mocs --max-pairs 300 --output-dir outputs/agentic_pipeline
```
