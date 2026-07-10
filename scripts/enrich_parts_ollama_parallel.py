import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Configuration
DB_PATH = "data/catalog/models_catalog.db"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"
MAX_WORKERS = 4  # Targets half of available CPU resources, leaves plenty of RAM, allows Ollama concurrency

db_lock = threading.Lock()

THEME_DESCRIPTIONS = {
    "Classic Town": "Entornos urbanos clásicos de LEGO de los años 80 y 90, vehículos de emergencia, servicios públicos y escenas cotidianas de ciudad.",
    "City": "Línea de LEGO moderna que representa escenas urbanas realistas, aeropuertos, trenes, estaciones de policía, bomberos y vehículos de transporte público.",
    "Friends": "Entorno suburbano centrado en Heartlake City, con cafeterías, tiendas, casas detalladas, clínicas veterinarias y actividades sociales.",
    "Technic": "Modelos mecánicos de ingeniería avanzada con engranajes, transmisiones, sistemas neumáticos y suspensión móvil.",
    "Creator 3-in-1": "Modelos creativos y modulares de vehículos, casas o animales construidos con bloques tradicionales que admiten tres reconstrucciones diferentes.",
    "Police": "Subtemática de acción urbana de LEGO City centrada en persecuciones, comisarías, vehículos de patrulla y operativos de rescate policial.",
    "Universal Building Set": "Sets clásicos de construcción libre que contienen una amplia variedad de ladrillos básicos de diferentes colores y propósitos generales.",
    "Jack Stone": "Línea de acción de LEGO de principios de los años 2000 orientada a rescates rápidos de héroes con minifiguras grandes y vehículos modulares.",
    "Space": "Sets retro y de ciencia ficción con naves espaciales, bases lunares, exploradores galácticos y tecnología futurista.",
    "Castle": "Entornos de fantasía medieval con castillos, caballeros, armaduras, dragones y batallas de asedio."
}

def check_ollama():
    global MODEL_NAME
    print(f"Verificando conexión con Ollama en {OLLAMA_URL}...")
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = [m["name"] for m in data.get("models", [])]
            if MODEL_NAME in models or f"{MODEL_NAME}:latest" in models or any(m.startswith(MODEL_NAME) for m in models):
                print(f"Confirmado: El modelo {MODEL_NAME} está disponible.")
                return True
            else:
                matching = [m for m in models if "7b" in m]
                if matching:
                    MODEL_NAME = matching[0]
                    print(f"Advertencia: Usando fallback {MODEL_NAME}.")
                    return True
                print("Error: No se encontró modelo 7B.")
                return False
    except Exception as e:
        print(f"Error crítico al conectar a Ollama: {e}")
        return False

def init_db():
    with db_lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rb_parts_enriched (
            part_num TEXT PRIMARY KEY,
            formal_description TEXT,
            category_group TEXT,
            utility_labels TEXT,
            enriched_at TEXT,
            FOREIGN KEY (part_num) REFERENCES rb_parts(part_num)
        )
        """)
        conn.commit()
        conn.close()

def get_theme_and_description(conn, part_num):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT t.name
        FROM rb_inventory_parts ip
        JOIN rb_inventories i ON ip.inventory_id = i.id
        JOIN rb_sets s ON i.set_num = s.set_num
        JOIN rb_themes t ON s.theme_id = t.id
        WHERE ip.part_num = ?
        GROUP BY t.name
        ORDER BY count(*) DESC
        LIMIT 1
    """, (part_num,))
    row = cursor.fetchone()
    if row:
        theme_name = row[0]
        desc = THEME_DESCRIPTIONS.get(theme_name, f"Línea de construcción creativa basada en la temática oficial de LEGO '{theme_name}'.")
        return theme_name, desc
    return "Universal LEGO Catalog", "Catálogo universal de piezas generales reutilizadas en todo tipo de creaciones de LEGO."

def build_prompt(part_num, name, theme_name, theme_description):
    return f"""[SYSTEM]
Eres un sistema experto en visión artificial, semántica y catalogación geométrica, especializado EXCLUSIVAMENTE en el ecosistema de piezas de LEGO. Tu objetivo es analizar una pieza dentro de su contexto temático original para extraer metadatos puros basados en su realidad física, utilidad y propósito en el set.

[CONTEXTO DE LA SERIE / SET]
Estás analizando una pieza que pertenece a la siguiente línea temática de LEGO:
- Serie/Tema: {theme_name}
- Descripción del entorno de la serie: {theme_description}

[RESTRICCIONES CRÍTICAS]
1. Usa el contexto de la serie para precisar la función, pero NO contamines la pieza con elementos que no le corresponden (ej: si la serie es "City (Policía)", un parabrisas sigue siendo un "parabrisas" o "cristal", NO es una "esposa" ni una "sirena").
2. Evita asociaciones abstractas o circunstanciales. Las etiquetas deben describir lo que el objeto REALMENTE REPRESENTA en ese entorno.
3. Devuelve estrictamente un objeto JSON. No incluyas marcas de bloque ```json ... ```, ni introducciones ni notas.

[TAXONOMÍA DE ENTRADA]
Clasifica la pieza en uno de estos grupos de categoría principales ('category_group'):
- "Transportation" (Piezas integrales de vehículos: ruedas, chasis, parabrisas, volantes, motores, alas).
- "Minifigure Accessories" (Objetos que una minifigura puede sostener, vestir o usar directamente: tazas, herramientas, armas, mochilas).
- "Construction / Bricks" (Bloques, placas, pendientes y elementos estructurales o arquitectónicos puros).
- "Technic / Mechanics" (Engranajes, ejes, conectores, vigas perforadas y piezas mecánicas).
- "Home & Decor / Interior" (Mobiliario, vajilla fija, electrodomésticos, comida, vegetación o elementos de ambientación interna).

[REGLAS PARA LAS UTILITY_LABELS]
Debes generar entre 3 y 5 etiquetas en minúsculas que respondan estrictamente a:
1. NATURALEZA (¿Qué es el objeto físicamente? Ej: "contenedor", "herramienta", "cristal", "asiento").
2. FUNCIÓN (¿Para qué sirve en la construcción/juego según su serie? Ej: "conducción", "protección", "alimentación", "decoración").
3. TIPO DE OBJETO (Especificidad del catálogo. Ej: "volante", "parabrisas", "taza", "panel").

[EJEMPLO DE REFERENCIA (CONTEXTUALIZADO)]
Entrada: 
- ID: 3823 | Nombre: Windshield 2 x 4 x 2
- Serie: City (Tráfico)
- Entorno de serie: Entorno urbano con coches de policía, camiones de bomberos, heladerías y vehículos civiles modernos.
Salida: {{"formal_description": "Parabrisas translúcido para vehículo urbano", "category_group": "Transportation", "utility_labels": ["parabrisas", "protección", "ventana", "automóvil"]}}

[PROCESAR AHORA]
Analiza la siguiente pieza con su contexto asignado:
- ID de Pieza: {part_num}
- Nombre/Descripción original: {name}
- Serie de LEGO: {theme_name}
- Descripción del entorno: {theme_description}

Respuesta en JSON estricto:"""

def query_ollama(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        # Long timeout to allow concurrent slots scheduling
        with urllib.request.urlopen(req, timeout=60) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_response = res_data.get("response", "").strip()
            return json.loads(raw_response)
    except Exception:
        return None

def process_part(part_num, name):
    conn = sqlite3.connect(DB_PATH)
    try:
        theme_name, theme_description = get_theme_and_description(conn, part_num)
    finally:
        conn.close()
        
    prompt = build_prompt(part_num, name, theme_name, theme_description)
    res = query_ollama(prompt)
    if res:
        formal_desc = res.get("formal_description", name)
        cat_group = res.get("category_group", "Unknown")
        labels_list = res.get("utility_labels", [])
        labels_json = json.dumps(labels_list, ensure_ascii=False)
        now_str = datetime.now().isoformat()
        
        # Thread-safe database insert
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO rb_parts_enriched (part_num, formal_description, category_group, utility_labels, enriched_at)
                VALUES (?, ?, ?, ?, ?)
            """, (part_num, formal_desc, cat_group, labels_json, now_str))
            conn.commit()
            conn.close()
        return True, part_num, theme_name, cat_group, labels_list
    return False, part_num, None, None, None

def main():
    if not check_ollama():
        sys.exit(1)
    init_db()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Target themes
    print("Recuperando listado de piezas de las series: Speed Champions, Creator 3-in-1, Friends, Icons...")
    cursor.execute("""
        SELECT DISTINCT p.part_num, p.name
        FROM rb_parts p
        JOIN rb_inventory_parts ip ON p.part_num = ip.part_num
        JOIN rb_inventories i ON ip.inventory_id = i.id
        JOIN rb_sets s ON i.set_num = s.set_num
        JOIN rb_themes t ON s.theme_id = t.id
        LEFT JOIN rb_parts_enriched e ON p.part_num = e.part_num
        WHERE e.part_num IS NULL
          AND (t.name LIKE '%Speed Champions%'
               OR t.name LIKE '%Creator 3-in-1%'
               OR t.name LIKE '%Friends%'
               OR t.name LIKE '%Icons%')
        ORDER BY p.part_num ASC
    """)
    parts_to_process = cursor.fetchall()
    conn.close()
    
    total_parts = len(parts_to_process)
    print(f"Encontradas {total_parts} piezas pendientes de enriquecimiento.")
    if total_parts == 0:
        print("No hay piezas pendientes de procesamiento para estas series.")
        return
        
    print(f"Iniciando ThreadPoolExecutor con {MAX_WORKERS} hilos concurrentes.")
    print("Procesando exclusivamente piezas de las series seleccionadas. El job finalizará al terminar.")
    
    completed = 0
    success_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_part, p[0], p[1]): p for p in parts_to_process}
        
        for future in as_completed(futures):
            try:
                success, part_num, theme_name, cat_group, labels = future.result()
                completed += 1
                if success:
                    success_count += 1
                    print(f"[{completed}/{total_parts}] [ÉXITO] {part_num} ({theme_name}) -> {cat_group} {labels}")
                else:
                    print(f"[{completed}/{total_parts}] [FALLO] {part_num}")
            except Exception as e:
                completed += 1
                # Retrieve part info from futures dict mapping to print context
                part_info = futures[future]
                print(f"[{completed}/{total_parts}] [EXCEPCIÓN] Error procesando {part_info[0]}: {e}")

if __name__ == "__main__":
    main()
