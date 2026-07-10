# /pipeline-status - Ver estado del pipeline de datos

Muestra el estado actual del pipeline de ingesta, catalogo y clasificacion.

## Uso
`/pipeline-status` — muestra resumen completo del estado

## Instrucciones

Ejecuta los siguientes comandos en orden y presenta un resumen:

### 1. Estado de ingesta
```bash
cat data/ingestion_progress.json | python -m json.tool
```

### 2. Estadisticas del catalogo SQLite
```bash
sqlite3 data/catalog/models_catalog.db "
SELECT
  COUNT(*) as total_sets,
  SUM(CASE WHEN parts_count > 0 THEN 1 ELSE 0 END) as sets_con_piezas,
  MIN(parts_count) as min_piezas,
  MAX(parts_count) as max_piezas,
  ROUND(AVG(parts_count), 1) as avg_piezas
FROM sets WHERE source = 'BrickLink';
"
```

### 3. Estado de clasificacion
```bash
sqlite3 data/catalog/models_catalog.db "
SELECT classification_status, COUNT(*) as count
FROM sets
WHERE source = 'BrickLink'
GROUP BY classification_status
ORDER BY count DESC;
"
```

### 4. Sets pendientes de clasificacion (ordenados por tamano ASC)
```bash
sqlite3 data/catalog/models_catalog.db "
SELECT set_id, name, parts_count
FROM sets
WHERE source = 'BrickLink'
  AND (classification_status IS NULL OR classification_status = 'vlm_raw')
ORDER BY parts_count ASC
LIMIT 10;
"
```

### 5. Archivos LDR/MPD disponibles localmente
```bash
find data/ -name "*.ldr" -o -name "*.mpd" -o -name "*.mbx" | wc -l
find data/ -name "*_sequence.ldr" | sort -t_ -k1 | head -10
```

### 6. Modelos entrenados disponibles
```bash
ls -la models/*.pt 2>/dev/null || echo "No hay checkpoints entrenados"
ls -la models/*.json 2>/dev/null | head -5
```

### 7. Estadisticas de BrickLink scraping
```bash
cat data/bricklink_stats.json | python -m json.tool 2>/dev/null || echo "bricklink_stats.json no disponible"
```

## Interpretacion del estado

| Estado | Significado | Accion |
|---|---|---|
| `vlm_raw` | Clasificado por VLM automaticamente | Revisar y verificar manualmente |
| `human_verified` | Verificado por humano | Listo para entrenamiento |
| `NULL` | Sin clasificar | Correr `classify_pending_parallel.py` |

## Siguiente paso recomendado
- Si hay sets sin clasificar: `python scripts/classify_pending_parallel.py`
- Si faltan secuencias LDR: `python scripts/generate_sequences.py`
- Si el catalogo esta vacio: `python scripts/initialize_catalog.py`
