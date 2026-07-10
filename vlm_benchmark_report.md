# Reporte Comparativo de Modelos VLM (Ollama)
Fecha de ejecución: 2026-07-09 15:44:19
Sets validados evaluados: 33

## Resumen de Métricas Globales

| Modelo | Precisión L1 (Entorno) | Precisión L2 (Propósito) | Precisión Exacta Completa | Tiempo Promedio/Set | Tiempo Promedio/Imagen |
|---|---|---|---|---|---|
| **qwen2.5vl:3b** | 21.9% (7/32) | 21.9% (7/32) | 18.8% (6/32) | 41.99s | 19.91s |
| **qwen2.5vl:latest** | 46.9% (15/32) | 21.9% (7/32) | 18.8% (6/32) | 52.02s | 26.38s |
| **llama3.2-vision:11b** | 62.5% (20/32) | 34.4% (11/32) | 3.1% (1/32) | 1.66s | 1.36s |

## Detalle de Clasificaciones por Set y Modelo

### Set 1: Obsidian monarch (`Obsidian monarch`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Coche | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 12.07s | 3.02s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 33.24s | 8.31s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.71s | 0.43s |

---

### Set 2: 4th of July (`4th of July`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 11.16s | 5.58s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 18.69s | 9.34s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.17s | 0.59s |

---

### Set 3: micro cart 3 (`micro cart 3`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Microscale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 15.85s | 3.17s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 29.11s | 5.82s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.44s | 0.29s |

---

### Set 4: Phantom Belle (`Phantom Belle`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 15.58s | 3.90s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 27.47s | 6.87s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.24s | 0.31s |

---

### Set 5: Classic Police Car (`Classic Police Car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Militar/Defensa | Coche | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 890.30s | 296.77s |
| qwen2.5vl:latest | `Terrestre | Competicion/Deportes | Vehículo | Minifig-scale` | ❌ NO | 48.04s | 16.01s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.40s | 0.47s |

---

### Set 6: Rally Car (`Rally Car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Coche | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 32.40s | 6.48s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 781.75s | 156.35s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.75s | 0.35s |

---

### Set 7: police sports car (`police sports car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Coche | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 24.45s | 4.89s |
| qwen2.5vl:latest | `Terrestre | Competicion/Deportes | Vehículo | Minifig-scale` | ❌ NO | 50.09s | 10.02s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 2.40s | 0.48s |

---

### Set 8: Stormblade (`563116`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Coche | Microscale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 0.00s | 0.00s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 0.00s | 0.00s |
| llama3.2-vision:11b | `Otros | Otros | Otros | Otros` | ❌ NO | 0.00s | 0.00s |

---

### Set 9: 50s van (`50s van`)
**Etiquetas Reales (Human Verified):** `Terrestre | Carga/Trabajo | Furgoneta | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 14.89s | 14.89s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 26.79s | 26.79s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.20s | 1.20s |

---

### Set 10: Fire Hydrant (`Fire Hydrant`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 12.39s | 12.39s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 23.86s | 23.86s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.29s | 1.29s |

---

### Set 11: Heavy Pacific (`Heavy Pacific`)
**Etiquetas Reales (Human Verified):** `Terrestre | Carga/Trabajo | Tren | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 12.65s | 12.65s |
| qwen2.5vl:latest | `Terrestre | Militar/Combate | Vehículo | Minifig-scale` | ❌ NO | 38.55s | 38.55s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.12s | 1.12s |

---

### Set 12: Pulse (`Pulse`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 14.57s | 14.57s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 29.68s | 29.68s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.43s | 1.43s |

---

### Set 13: Lunar Cruiser (`Lunar Cruiser`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.69s | 13.69s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 30.45s | 30.45s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.75s | 1.75s |

---

### Set 14: Tall Tree (`Tall Tree`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 0.00s | 0.00s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 0.00s | 0.00s |
| llama3.2-vision:11b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 0.00s | 0.00s |

---

### Set 15: Train Set (`Train Set`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Tren | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.80s | 13.80s |
| qwen2.5vl:latest | `Terrestre | Militar/Combate | Vehículo | Minifig-scale` | ❌ NO | 27.13s | 27.13s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.62s | 1.62s |

---

### Set 16: airline (`airline`)
**Etiquetas Reales (Human Verified):** `Aéreo | Civil/Pasajeros | Avión | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 14.00s | 14.00s |
| qwen2.5vl:latest | `Aéreo | Militar/Combate | Avión | Minifig-scale` | ❌ NO | 25.38s | 25.38s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.56s | 1.56s |

---

### Set 17: formula (`formula`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 11.82s | 11.82s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 22.23s | 22.23s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.29s | 1.29s |

---

### Set 18: policecar (`policecar`)
**Etiquetas Reales (Human Verified):** `Terrestre | Militar/Defensa | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.29s | 13.29s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 24.91s | 24.91s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.15s | 1.15s |

---

### Set 19: scissor lift (`scissor lift`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 12.62s | 12.62s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 28.99s | 28.99s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.25s | 1.25s |

---

### Set 20: skeletal train (`skeletal train`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Tren | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.71s | 13.71s |
| qwen2.5vl:latest | `Terrestre | Militar/Combate | Vehículo | Minifig-scale` | ❌ NO | 29.50s | 29.50s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.24s | 1.24s |

---

### Set 21: steampunk airplane (`steampunk airplane`)
**Etiquetas Reales (Human Verified):** `Aéreo | Militar/Defensa | Avión | Minifig-scale`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 31.61s | 3.16s |
| qwen2.5vl:latest | `Aéreo | Militar/Combate | Avión | Minifig-scale` | ❌ NO | 50.03s | 5.00s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.71s | 0.17s |

---

### Set 22: truck (`truck`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Camión | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.61s | 13.61s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 21.67s | 21.67s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.42s | 1.42s |

---

### Set 23: vintage Police Car (`vintage Police Car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 15.58s | 15.58s |
| qwen2.5vl:latest | `Terrestre | Competicion/Deportes | Vehiculo | Microscale` | ❌ NO | 28.40s | 28.40s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 2.26s | 2.26s |

---

### Set 24: Air Taxi (`Air Taxi`)
**Etiquetas Reales (Human Verified):** `Aéreo | Civil/Pasajeros | Helicóptero | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 13.11s | 13.11s |
| qwen2.5vl:latest | `Terrestre | Competicion/Deportes | Vehiculo | Microscale` | ❌ NO | 30.33s | 30.33s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.64s | 1.64s |

---

### Set 25: Bubble Car (`Bubble Car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 14.11s | 14.11s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 26.48s | 26.48s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.43s | 1.43s |

---

### Set 26: Commercial Crab Boat (`Commercial Crab Boat`)
**Etiquetas Reales (Human Verified):** `Marítimo | Civil/Pasajeros | Barco | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | *Error/Sin datos* | - | - | - |
| qwen2.5vl:latest | *Error/Sin datos* | - | - | - |
| llama3.2-vision:11b | *Error/Sin datos* | - | - | - |

---

### Set 27: Delivery Truck (`Delivery Truck`)
**Etiquetas Reales (Human Verified):** `Terrestre | Carga/Trabajo | Camión | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 14.26s | 14.26s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 26.68s | 26.68s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.31s | 1.31s |

---

### Set 28: Level Crossing (`Level Crossing`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 15.09s | 15.09s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 26.77s | 26.77s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.43s | 1.43s |

---

### Set 29: Little Car (`Little Car`)
**Etiquetas Reales (Human Verified):** `Terrestre | Civil/Pasajeros | Coche | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 15.42s | 15.42s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 15.32s | 15.32s |
| llama3.2-vision:11b | `Otros | Otros | Otros | Otros` | ❌ NO | 9.76s | 9.76s |

---

### Set 30: Tractor (`Tractor`)
**Etiquetas Reales (Human Verified):** `Terrestre | Carga/Trabajo | Maquinaria | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 15.30s | 15.30s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 49.92s | 49.92s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.96s | 1.96s |

---

### Set 31: hotrod (`hotrod`)
**Etiquetas Reales (Human Verified):** `Terrestre | Deportivo/Carreras | Camión | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 16.37s | 16.37s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 33.68s | 33.68s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.49s | 1.49s |

---

### Set 32: Brush 13 (Fire Dept) (`Brush 13 (Fire Dept)`)
**Etiquetas Reales (Human Verified):** `Otros | Otros | Otros | Otros`

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ✅ SÍ | 13.80s | 13.80s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ✅ SÍ | 33.86s | 33.86s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.45s | 1.45s |

---

### Set 33: Steamboat MOC (`Steamboat MOC`)
**Etiquetas Reales (Human Verified):** `Marítimo | Recreativo/Fantasía | Otros | `

| Modelo | Predicción VLM | ¿Acierto Completo? | Tiempo Total | Tiempo/Imagen |
|---|---|---|---|---|
| qwen2.5vl:3b | `Otros | Otros | Otros | Otros` | ❌ NO | 16.24s | 16.24s |
| qwen2.5vl:latest | `Otros | Otros | Otros | Otros` | ❌ NO | 25.79s | 25.79s |
| llama3.2-vision:11b | `Terrestre | Civil/Pasajeros | Desconocido | Minifig-scale` | ❌ NO | 1.37s | 1.37s |

---
