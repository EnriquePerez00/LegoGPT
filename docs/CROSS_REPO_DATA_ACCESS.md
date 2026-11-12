 # Cross-Repo Data Access: LegoGPT ↔ LdrawNova
 
 **Objetivo:** Centralizar toda la data de sets/MOCs en LegoGPT; LdrawNova consume sin duplicar.
 
 ---
 
 ## Arquitectura de datos
 
 ### LegoGPT — Fuente única de verdad (Fase 1 completada ✅)
 
 ```
 /Users/I764690/Code_personal/LegoGPT/
 └── data/
     ├── normalized_io/           (298 MB, 927 .io files)
     │   ├── OMR-1234-1_normalized.io
     │   ├── RB-MOC-12345_normalized.io
     │   └── ...
     ├── catalog/
     │   └── models_catalog.db    (929 pairs: 733 OMR sets + 196 MOCs)
     ├── omr_raw/                 (157 MB, 733 .mpd official sets)
     ├── manuals/                 (50 MB PDF manuals)
     └── ldr_extracted/           (preparado para LDraw extraído)
 ```
 
 **Contenido:**
 - **normalized_io**: Grafos de construcción 3D step-by-step (927 archivos .io)
 - **models_catalog.db**: Metadatos de sets y MOCs (pair_id, designer, parts, steps, URLs)
 - **omr_raw**: Ficheros .mpd originales de sets oficiales LEGO
 - **manuals**: Instrucciones en PDF de sets y MOCs
 
 ### LdrawNova — Consumidor sin duplicación
 
 ```
 /Users/I764690/Code_personal/LdrawNova/
 ├── scripts/bench/
 │   ├── config.py               ← _resolve_io_source_dir() → LegoGPT
 │   └── io_loader.py            ← consume .io files via config
 ├── data/
 │   ├── ldraw-info.db           (26 MB, metadatos LDraw piezas)
 │   ├── offLibShadow/           (shadow library de partes)
 │   └── categories/             (paletas de colores)
 └── .env.example
     └── LDRAW_NOVA_IO_DIR       (documentado, auto-resuelve)
 ```
 
 **Nota:** Los .io files NO están en LdrawNova; se leen directamente de LegoGPT.
 
 ---
 
 ## Módulo de acceso: LegoGPT/src/data_locator.py
 
 Interfaz pública para consumir datos de LegoGPT desde cualquier repo:
 
 ```python
 from LegoGPT.src.data_locator import (
     get_normalized_io_dir,        # → Path a 927 .io files
     get_catalog_db,               # → Path a models_catalog.db
     get_omr_raw_dir,              # → Path a 733 .mpd files
     get_manuals_dir,              # → Path a PDFs
     validate_data_sources,        # → dict con status
     get_io_by_pair_id,            # → Path a .io específico por pair_id
 )
 ```
 
 ### Ejemplo de uso
 
 ```python
 # En cualquier script en LdrawNova:
 import sys
 sys.path.insert(0, "../LegoGPT")
 from src.data_locator import get_normalized_io_dir, validate_data_sources
 
 # Validar acceso
 status = validate_data_sources()
 if not status["ok"]:
     raise RuntimeError(f"LegoGPT data missing: {status['missing']}")
 
 # Usar
 io_dir = get_normalized_io_dir()
 io_files = list(io_dir.glob("*.io"))
 ```
 
 ---
 
 ## Configuración en LdrawNova
 
 ### Resolución automática (recomendada)
 
 Si LegoGPT y LdrawNova son hermanos en el mismo directorio:
 
 ```bash
 ~/Code_personal/
 ├── LegoGPT/
 │   └── data/normalized_io/  (927 .io files)
 └── LdrawNova/
     └── scripts/bench/config.py
 ```
 
 **No se requiere configuración extra.** El archivo `config.py` auto-detecta:
 
 ```python
 # scripts/bench/config.py: _resolve_io_source_dir()
 # Busca en orden:
 # 1. Env var LDRAW_NOVA_IO_DIR (si está set)
 # 2. Sibling ../LegoGPT/data/normalized_io (auto-detect)
 # 3. Fallback hardcoded: /Users/I764690/Code_personal/LegoGPT/data/normalized_io
 ```
 
 ### Configuración manual (LegoGPT en otro lugar)
 
 Si LegoGPT está en otra ruta:
 
 ```bash
 # Opción 1: Env var en shell
 export LDRAW_NOVA_IO_DIR="/ruta/a/LegoGPT/data/normalized_io"
 python scripts/bench/smoke.py
 
 # Opción 2: En LdrawNova/.env
 cat >> .env << EOF
 LDRAW_NOVA_IO_DIR="/ruta/a/LegoGPT/data/normalized_io"
 EOF
 ```
 
 ### Validación
 
 ```bash
 # Verificar que LdrawNova ve los .io files
 cd /path/to/ldraw-nova
 python -c "
   from scripts.bench.config import IO_SOURCE_DIR, validate_io_source
   validate_io_source()
   print(f'✅ IO corpus: {IO_SOURCE_DIR}')
   print(f'✅ Found: {len(list(IO_SOURCE_DIR.glob(\"*.io\")))} .io files')
 "
 ```
 
 ---
 
 ## Matriz de resolución (config.py)
 
 | Escenario | LDRAW_NOVA_IO_DIR | ../LegoGPT? | Resultado |
 |-----------|-------------------|------------|-----------|
 | Sibling hermano | No set | Existe | ✅ Auto-detecta |
 | Custom path | Set | Ignorado | ✅ Usa env var |
 | LegoGPT no existe | No set | No | ❌ Usa fallback, falla si no existe |
 | Aislado / testing | Set local | N/A | ✅ Usa env var |
 
 ---
 
 ## Acceso a la BD desde LdrawNova
 
 Si necesitas metadatos de la BD `models_catalog.db`:
 
 ```python
 import sys, sqlite3
 sys.path.insert(0, "../LegoGPT")
 from src.data_locator import get_catalog_db
 
 db = get_catalog_db()
 conn = sqlite3.connect(db)
 cursor = conn.cursor()
 
 # Ejemplo: obtener todos los pair_ids con sus .io paths
 cursor.execute("SELECT pair_id, normalized_io_path FROM ground_truth_pairs")
 for pair_id, io_path in cursor.fetchall():
     print(f"{pair_id}: {io_path}")
 ```
 
 ---
 
 ## Troubleshooting
 
 | Problema | Solución |
 |----------|----------|
 | `IO_SOURCE_DIR not found` | Verificar que LegoGPT Fase 1 completó (`ls LegoGPT/data/normalized_io/*.io`) |
 | `Found only N .io files < 900` | Fase 1 falló parcialmente; re-ejecutar `scripts/recover_missing_data.py` |
 | `No module named 'LegoGPT'` | Agregar `sys.path.insert(0, "../LegoGPT")` antes de imports |
 | Symlink roto | Verificar `ls -la LegoGPT/data/normalized_io/` |
 
 ---
 
 ## Próximos pasos
 
 **Fase 3 (Validación progresiva):**
 - Crear eval sets de test con subconjuntos de MICRO tier
 - Ejecutar validación con timeout realista
 - Verificar que los fixes sintetizados vienen de incidentes reales (no timeouts)
 - Escalar a SMALL, MEDIUM, etc.
 
 ---
 
 **Creado:** Nov 12, 2026  
 **Relacionado:** Fase 1 (data recovery), Fase 2 (contrato de acceso)