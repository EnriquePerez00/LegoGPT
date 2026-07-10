# LegoGPT — Claude Code Context

## Descripcion del Proyecto

**LegoGPT** es un motor de IA generativa que traduce prompts de lenguaje natural en modelos 3D LEGO estructuralmente validos y ensamblables, emitiendo archivos **LDraw (.ldr/.mpd)** en lugar de pixeles. El sistema comprende la topologia de construccion LEGO mediante Graph Neural Networks entrenadas sobre secuencias de manuales reales.

### Tres subsistemas principales:
1. **Data Pipeline** (`scripts/`, `src/ingestion_pipeline.py`) — Ingesta de manuales BrickLink/OMR a grafos PyG
2. **Core AI** (`src/model.py`, `src/generator.py`, `generative/`) — GNN multi-tarea que predice (part_id, color, transform_matrix)
3. **Validator** (`src/validator.py`, `src/graph_validator.py`) — Motor determinista de colisiones y conectividad

---

## Hardware Profile — REGLAS DE OPTIMIZACION OBLIGATORIAS

**Maquina local: Apple Silicon M4 · 48 GB RAM unificada · 12 cores**

### Reglas estrictas para todo codigo generado:

**1. Device MPS-first — NUNCA usar device='cuda':**
```python
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

**2. Limpiar cache MPS tras operaciones intensivas:**
```python
del tensor_grande
torch.mps.empty_cache()
```

**3. Paralelismo maximo — usar todos los cores:**
```python
import multiprocessing
NUM_WORKERS = multiprocessing.cpu_count()  # 12 en esta maquina
# DataLoaders: num_workers=min(8, NUM_WORKERS)
# Scripts batch:
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    results = list(executor.map(process_fn, items))
```

**4. Tests siempre en paralelo:**
```bash
pytest tests/ -n auto -v
```

**5. Batch sizes agresivos con 48GB RAM:**
- Entrenamiento GNN: batch_size=32 minimo
- Inferencia: batch_size=64+

**6. Monitorizar memoria MPS:**
```python
torch.mps.current_allocated_memory()
```

**7. Compilar modelos cuando sea posible:**
```python
model = torch.compile(model)
```

---

## Regla de Ordenacion 3D — OBLIGATORIA

**Cuando se trabaje con listas de disenos/sets/modelos 3D, SIEMPRE ordenar por numero de piezas ascendente antes de iterar.**

Razon: fail-fast, validacion rapida en sets pequenos antes de invertir tiempo en sets grandes.

```python
# Listas Python:
sets_sorted = sorted(sets, key=lambda s: s.get("parts_count", 0))
for set_data in sets_sorted:
    process(set_data)

# SQLite — siempre ORDER BY parts_count ASC:
cursor.execute("SELECT * FROM sets WHERE source='BrickLink' ORDER BY parts_count ASC")

# Archivos LDR/MPD — ordenar por tamano como proxy:
files_sorted = sorted(ldr_files, key=lambda f: os.path.getsize(f))
```

---

## Arquitectura del Codebase

```
LegoGPT/
├── src/                        # Nucleo importable del proyecto
│   ├── api.py                  # FastAPI: endpoints REST + WebSocket training
│   ├── model.py                # LegoGNN, LegoGraphTransformer, HierarchicalSoftmax
│   ├── parser.py               # LDraw parser, ALLOWED_PARTS, ALLOWED_COLORS
│   ├── generator.py            # LegoGenerator: beam search autoregresivo
│   ├── validator.py            # check_connectivity_and_gravity, check_collisions
│   ├── graph_validator.py      # Validacion de grafos PyG
│   ├── writer.py               # ParsedPart dataclass + write_ldraw_file
│   ├── mpd_parser.py           # flatten_mpd: submodelos MPD anidados
│   ├── ingestion_pipeline.py   # Pipeline principal de ingesta
│   ├── classification_pipeline.py
│   ├── vlm_parser_pipeline.py  # Procesamiento VLM para etiquetas
│   ├── mosaic_generator.py     # Mosaicos LEGO desde imagen
│   └── sequence_visualizer.py
├── scripts/                    # Utilidades standalone (NO importar desde src/)
│   ├── adaptive_bricklink_crawler.py
│   ├── classify_pending_parallel.py
│   ├── generate_sequences.py
│   ├── harvest_*.py            # Scrapers de imagenes y metadatos BrickLink
│   └── resolve_*.py            # Resolucion de IDs y nombres BrickLink
├── generative/                 # Pipeline LLM experimental
│   ├── voxelizer.py            # Mesh 3D a LEGO bricks
│   └── llm_pipeline/
│       ├── rl_loop.py
│       ├── sequence_planner.py
│       ├── standardizer.py
│       └── tokenizer_ldr.py
├── tests/                      # Suite pytest TDD
├── data/
│   ├── catalog/
│   │   └── models_catalog.db   # PROTEGIDO: SQLite principal
│   ├── bricklink_stats.json    # PROTEGIDO: estadisticas de scraping
│   ├── ingestion_progress.json # Estado del pipeline
│   └── *.ldr, *.mbx
├── models/                     # PROTEGIDO: checkpoints PyTorch *.pt
├── public/                     # Frontend estatico servido por FastAPI
├── prd/                        # Documentacion y requirements
└── generate_build.py           # CLI principal de generacion
```

---

## Stack Tecnologico

| Capa | Tecnologia |
|---|---|
| ML Core | PyTorch 2.2+ (MPS), PyTorch Geometric (PyG) |
| 3D/LDraw | Trimesh, python-ldraw, NetworkX |
| API | FastAPI + Uvicorn + WebSockets |
| Base de datos | SQLite3 (data/catalog/models_catalog.db) |
| Tests | pytest + pytest-mock + pytest-xdist (-n auto) |
| Agentes locales | Ollama (LLM ligero para VLM embeddings) |
| Render | Blender headless (path: $BLENDER_PATH) |

---

## Formato de Datos Clave

### ParsedPart (dataclass canonico)
```python
@dataclass
class ParsedPart:
    part_id: str          # e.g. "3003.dat" (Brick 2x2)
    color: int            # LDraw color code: 14=Yellow, 4=Red, 1=Blue
    transform: np.ndarray # Shape (4,4) float32
    step_id: int          # Paso de ensamblaje (0-indexed)
```

### JSON de Assembly
```json
{
  "step_id": 14,
  "action": "add_node",
  "part_id": "3003.dat",
  "color_code": 14,
  "transform_matrix": [1,0,0, 0,1,0, 0,0,1, 40,-24,20],
  "edges_formed": ["node_8_stud_1", "node_9_stud_2"]
}
```

### Sistema de Coordenadas LDraw
- El eje Y esta INVERTIDO respecto a motores 3D estandar (Y aumenta hacia abajo)
- Unidad LDraw = 0.4mm, 1 stud = 20 LDU, 1 placa = 8 LDU, 1 brick = 24 LDU
- Formato linea: `1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part.dat>`
- Matriz de rotacion 3x3 serializada aplanada (9 valores)

---

## Comandos de Desarrollo Frecuentes

```bash
# Activar entorno virtual
source legogpt_env/bin/activate

# Arrancar la API
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

# Todos los tests en paralelo (12 cores M4)
pytest tests/ -n auto -v

# Tests de un modulo especifico
pytest tests/test_validator.py -v

# Generar modelo desde CLI
python generate_build.py --prompt "Torre azul" --num_pieces 8

# Pipeline de ingesta
python scripts/generate_sequences.py

# Clasificacion paralela
python scripts/classify_pending_parallel.py

# Estado del pipeline
cat data/ingestion_progress.json | python -m json.tool

# Consultar catalogo SQLite (ordenado ASC)
sqlite3 data/catalog/models_catalog.db "SELECT set_id, name, parts_count FROM sets ORDER BY parts_count ASC LIMIT 20"
```

---

## Convenciones de Codigo OBLIGATORIAS

1. **Type Hints completos** en todas las firmas (Python 3.10+)
2. **Docstrings** en clases y funciones publicas
3. **TDD estricto**: tests primero, implementacion despues, pasar tests antes de continuar
4. **Imports absolutos** (nunca relativos): `from src.parser import ParsedPart`
5. **Variables de entorno para rutas externas**: `os.getenv("BLENDER_PATH", "blender")`
6. **No silenciar excepciones**: nunca `except: pass`
7. **Logging** en lugar de print en produccion: `logger = logging.getLogger(__name__)`

---

## Archivos Protegidos — NO MODIFICAR SIN CONFIRMAR

| Recurso | Razon |
|---|---|
| `data/catalog/models_catalog.db` | Catalogo principal — miles de horas de scraping |
| `models/*.pt` | Checkpoints de modelos entrenados |
| `data/bricklink_stats.json` | Estadisticas de scraping acumuladas |
| `data/ingestion_progress.json` | Estado del pipeline de ingesta |

---

## Vocabulario del Dominio

| Termino | Significado |
|---|---|
| LDU | LDraw Unit = 0.4mm |
| Stud | Clavija superior LEGO (conexion macho) |
| Tube | Tubo inferior LEGO (conexion hembra) |
| MPD | Multi-Part Document (LDraw con submodelos anidados) |
| OMR | Official LEGO Model Repository |
| BrickLink | Marketplace/base de datos de referencia de sets LEGO |
| Rebrickable | Base de datos de inventarios de sets LEGO |
| PyG | PyTorch Geometric |
| MPS | Metal Performance Shaders (GPU Apple Silicon) |
| VLM | Vision Language Model |
| OBB | Oriented Bounding Box (colision 3D) |
| ALLOWED_PARTS | Vocabulario controlado de part_ids en src/parser.py |
| ALLOWED_COLORS | Vocabulario controlado de color codes en src/parser.py |

---

## Taxonomia de Clasificacion de Sets (SQLite)

- `level_1_entorno` > `level_2_proposito` > `level_3_clase` > `level_4_escala`
- Para animales: `animal_level_1_habitat` > `animal_level_2_categoria` > `animal_level_3_especie` > `animal_level_4_estilo`
- `classification_status`: `vlm_raw` | `human_verified`
