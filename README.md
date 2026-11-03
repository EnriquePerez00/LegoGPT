# LegoGPT: Generación y Validación de Modelos 3D Estructurales LEGO

LegoGPT es una plataforma de IA generativa y validación física orientada al diseño y ensamblaje de modelos 3D basados en el estándar físico LDraw/MPD. El sistema procesa instrucciones, inventarios y geometrías en forma de grafos espaciales garantizando construcciones física y estructuralmente viables.

---

## 1. Fuentes de Datos y Catálogos Oficiales

El repositorio utiliza **Rebrickable** como fuente principal y oficial de datos de sets, temas, inventarios y piezas de LEGO, complementada con el repositorio abierto **OMR (Official Model Repository de LDraw)** y **BrickLink Studio**.

### Catálogo Oficial (Rebrickable)
- **Base de Datos Local:** SQLite en `data/catalog/models_catalog.db`.
- **Pipeline de Ingesta:** `scripts/import_rebrickable.py` y `scripts/initialize_catalog.py`.
- **Criterio de Ingesta de Sets Oficiales:**
  - Año de lanzamiento: $\ge 2004$.
  - Rango de piezas: $> 10$ y $< 150$ piezas.
  - Almacenamiento en tabla principal `sets` con metadatos de nombre, tema, piezas, URL de imagen y referencia oficial.

### Manuales Oficiales
Almacenados en `data/manuals/` en formato PDF de alta resolución descargados directamente del servicio de instrucciones oficial de LEGO.

### Modelos y Diseños MOC / OMR
Almacenados en `data/mocs/` en formato nativo LDraw `.mpd` y `.ldr`, listos para validación con el motor determinista de conectividad física.

---

## 2. Estructura de la Base de Datos (`models_catalog.db`)

### Tablas Principales

* **`sets`:** Catálogo consolidado con 28 campos (metadatos, popularidad, rutas de archivos y clasificación taxonómica multiescala).
* **`rb_sets`:** Catálogo oficial completo descargado de los dumps de Rebrickable (`set_num`, `name`, `year`, `theme_id`, `num_parts`, `img_url`).
* **`rb_themes`:** Árbol jerárquico de temáticas oficiales LEGO.
* **`rb_parts` y `rb_part_categories`:** Catálogo exhaustivo de piezas oficiales y sus categorías funcionales.
* **`parts_inventory`:** Desglose pieza por pieza y color para cada set o creación.
* **`subassemblies`:** Subestructuras y módulos detectados con verificación de anclaje al suelo.
* **`bricklink_scraping_queue`:** Cola adaptativa de recolección de creaciones MOC de la comunidad.

---

## 3. Arquitectura y Módulos del Sistema

* **`src/parser.py`, `src/mpd_parser.py`:** Procesamiento y lectura de archivos 3D LDraw (`.ldr`, `.mpd`), extrayendo matrices de transformación y generando grafos para PyTorch Geometric (PyG).
* **`src/validator.py`, `src/graph_validator.py`:** Comprobación estricta de restricciones físicas:
  - Regla Stud-to-Socket (macho/hembra).
  - Ausencia de piezas flotantes (recorrido DFS conectado al suelo $Y \le 0.0$).
  - Prevención total de colisiones geométricas (OBB / Trimesh).
* **`src/model.py`, `src/generator.py`:** Redes neuronales de grafos (GNN) y políticas de generación autorregresiva.
* **`src/api.py`:** Servidor FastAPI con WebSockets para inferencia 3D en tiempo real y recolección de feedback humano.
* **`src/sequence_visualizer.py`:** Visualizador interactivo 3D basado en Three.js.
