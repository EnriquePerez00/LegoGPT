# /test - Ejecutar tests del modulo actual

Ejecuta los tests relevantes para el archivo o modulo en el que estoy trabajando actualmente.

## Uso
`/test` - corre los tests del modulo actual
`/test <modulo>` - corre tests de un modulo especifico

## Instrucciones

1. Identifica el modulo actual (ej: si estas en `src/validator.py`, el test es `tests/test_validator.py`)
2. Ejecuta con parallelismo maximo aprovechando los 12 cores del M4:

```bash
pytest tests/test_$MODULO.py -v --tb=short
```

Si quieres correr TODOS los tests:
```bash
pytest tests/ -n auto -v --tb=short
```

3. Si hay fallos, analiza el traceback y propone un fix antes de continuar.

## Notas
- Usar siempre `-n auto` para tests completos (12 cores disponibles)
- Si un test falla por MPS, verificar que el codigo usa `get_device()` correctamente
- Los tests de modelos PyTorch pueden requerir `PYTORCH_ENABLE_MPS_FALLBACK=1`
