# /run-api - Arrancar el servidor FastAPI de LegoGPT

Inicia el servidor de desarrollo con recarga automatica.

## Uso
`/run-api` — arranca en localhost:8000

## Instrucciones

### Arranque estandar (desarrollo)
```bash
source legogpt_env/bin/activate
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### Verificar que esta corriendo
```bash
curl http://localhost:8000/config/parts | python -m json.tool
```

### Endpoints disponibles

| Metodo | Ruta | Descripcion |
|---|---|---|
| GET | `/` | Frontend estatico (public/) |
| POST | `/train/start` | Inicia entrenamiento GNN via WebSocket |
| POST | `/train/stop` | Detiene entrenamiento |
| POST | `/generate` | Genera estructura desde modelo entrenado |
| POST | `/generate-prompt` | Genera desde prompt de texto |
| POST | `/upload-mpd` | Carga y parsea archivo MPD |
| POST | `/validate-graph` | Valida conectividad y colisiones |
| POST | `/voxelize` | Convierte mesh 3D a LEGO |
| POST | `/generate-mosaic` | Genera mosaico desde imagen |
| POST | `/sequence-assembly` | Planifica secuencia de ensamblaje |
| POST | `/render-eevee` | Render con Blender headless |
| GET | `/models` | Lista modelos entrenados disponibles |
| GET | `/themes` | Lista temas de sets disponibles |
| GET | `/api/vlm/sets` | Lista sets del catalogo para clasificacion |
| POST | `/api/vlm/sets/classify` | Clasifica un set manualmente |
| WS | `/ws` | WebSocket para logs de entrenamiento en tiempo real |

### Variables de entorno utiles
```bash
export BLENDER_PATH="/Applications/Blender.app/Contents/MacOS/Blender"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTHONPATH="."
```

### Troubleshooting
- **ImportError src.parser**: asegurar `PYTHONPATH=.` o ejecutar desde raiz del repo
- **MPS error**: exportar `PYTORCH_ENABLE_MPS_FALLBACK=1`
- **Puerto ocupado**: `lsof -i :8000 | grep LISTEN` y matar el proceso
- **data/catalog/models_catalog.db not found**: el catalogo SQLite no existe aun, correr scripts de ingesta primero
