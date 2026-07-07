# Directrices de Desarrollo para Agente IA (Proyecto LegoGPT)

## Contexto de Hardware
Todo el código generado debe estar optimizado para ejecutarse localmente en una máquina con arquitectura Apple Silicon y 48GB de memoria RAM unificada. 

## Reglas de Codificación Estrictas
1. **PyTorch Device:** No uses `device = 'cuda'`. Implementa una función robusta de selección de dispositivo que priorice `mps` (Metal Performance Shaders), seguido de `cpu`.
2. **Memory Management:** Dado que trabajaremos con grafos pesados y mallas 3D, limpia los tensores innecesarios de la memoria usando `del` y vacía la caché de MPS (`torch.mps.empty_cache()`) después de operaciones de entrenamiento intensivas.
3. **Flujo de Trabajo TDD:** 
   - PASO 1: Escribe las pruebas en `pytest` para la validación física (colisiones usando `trimesh` y conectividad de grafos usando `networkx`).
   - PASO 2: Escribe la implementación del pipeline de datos.
   - PASO 3: Ejecuta y pasa las pruebas antes de escribir la arquitectura GNN.
4. **Formato de Salida LDraw:** Todas las matrices de transformación generadas deben traducirse estrictamente a la matriz de rotación de 9 valores requerida por el estándar LDraw, respetando su sistema de coordenadas (el eje Y en LDraw está invertido respecto a motores 3D estándar).
5. **Tipado:** Usa Type Hints completos de Python 3.10+ en todas las firmas de funciones para mantener el contexto del código legible.