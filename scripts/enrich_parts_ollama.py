import os
import sys
import json
import sqlite3
import urllib.request
import urllib.error
import argparse
from datetime import datetime

# Configuration
DB_PATH = "data/catalog/models_catalog.db"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

PILOT_PARTS = [
    "16091",   # Steering Wheel Small, 2 x 2
    "10053",   # Weapon Sword Small
    "3001",    # Brick 2 x 4
    "10928",   # Technic Gear 8 Tooth
    "003381"   # Sticker Sheet for Set 663-1
]

def check_ollama():
    global MODEL_NAME
    print(f"Verificando conexión con Ollama en {OLLAMA_URL}...")
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode('utf-8'))
            models = [m["name"] for m in data.get("models", [])]
            print(f"Modelos de Ollama detectados: {models}")
            if MODEL_NAME in models or f"{MODEL_NAME}:latest" in models or any(m.startswith(MODEL_NAME) for m in models):
                print(f"Confirmado: El modelo {MODEL_NAME} está disponible.")
                return True
            else:
                # Find best 7b fallback or matching model
                matching = [m for m in models if "7b" in m]
                if matching:
                    MODEL_NAME = matching[0]
                    print(f"Advertencia: Modelo {MODEL_NAME} seleccionado automáticamente como fallback de 7B.")
                    return True
                print(f"Error: No se encontró un modelo 7B disponible en Ollama.")
                return False
    except Exception as e:
        print(f"Error crítico: No se pudo conectar a Ollama. Asegúrate de que el servicio esté corriendo en el puerto 11434. Detalles: {e}")
        return False

def init_db():
    print(f"Inicializando esquema de base de datos en {DB_PATH}...")
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
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            raw_response = res_data.get("response", "").strip()
            return json.loads(raw_response)
    except Exception as e:
        print(f"Error de inferencia/parseo con Ollama: {e}")
        return None

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

def run_pilot():
    print("\n--- INICIANDO PRUEBA PILOTO (5 Piezas) ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    placeholders = ",".join("?" for _ in PILOT_PARTS)
    cursor.execute(f"SELECT part_num, name FROM rb_parts WHERE part_num IN ({placeholders})", PILOT_PARTS)
    parts = cursor.fetchall()
    
    if not parts:
        print("Error: No se encontraron las piezas de prueba en la tabla rb_parts.")
        conn.close()
        return
        
    for part_num, name in parts:
        theme_name, theme_description = get_theme_and_description(conn, part_num)
        print(f"\nProcesando pieza: {part_num} - '{name}' (Tema: {theme_name})...")
        prompt = build_prompt(part_num, name, theme_name, theme_description)
        res = query_ollama(prompt)
        if res:
            print("Resultado JSON exitoso:")
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print("Fallo al procesar la pieza.")
    conn.close()

def run_production(batch_size=50, target_parts=None):
    print("\n--- INICIANDO MODO PRODUCCIÓN ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if target_parts:
        placeholders = ",".join("?" for _ in target_parts)
        cursor.execute(f"""
            SELECT part_num, name 
            FROM rb_parts 
            WHERE part_num IN ({placeholders})
        """, target_parts)
    else:
        # Select parts that have not been enriched yet
        cursor.execute("""
            SELECT p.part_num, p.name 
            FROM rb_parts p
            LEFT JOIN rb_parts_enriched e ON p.part_num = e.part_num
            WHERE e.part_num IS NULL
        """)
        
    pending_parts = cursor.fetchall()
    total_pending = len(pending_parts)
    print(f"Total de piezas a procesar: {total_pending}")
    
    if total_pending == 0:
        print("No hay piezas para procesar.")
        conn.close()
        return
        
    batch = []
    processed_count = 0
    
    for part_num, name in pending_parts:
        theme_name, theme_description = get_theme_and_description(conn, part_num)
        print(f"Procesando {part_num} (Tema: {theme_name})...")
        prompt = build_prompt(part_num, name, theme_name, theme_description)
        res = query_ollama(prompt)
        
        if res:
            formal_desc = res.get("formal_description", name)
            cat_group = res.get("category_group", "Unknown")
            labels_list = res.get("utility_labels", [])
            labels_json = json.dumps(labels_list, ensure_ascii=False)
            now_str = datetime.now().isoformat()
            
            # Print output for visibility
            print(f"  [+] Enriquecido: {part_num} -> Category: {cat_group}, Labels: {labels_list}")
            
            batch.append((part_num, formal_desc, cat_group, labels_json, now_str))
            processed_count += 1
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT OR REPLACE INTO rb_parts_enriched (part_num, formal_description, category_group, utility_labels, enriched_at)
                    VALUES (?, ?, ?, ?, ?)
                """, batch)
                conn.commit()
                batch = []
                print(f"Progreso: {processed_count}/{total_pending} piezas procesadas y guardadas.")
        else:
            print(f"Omitiendo pieza {part_num} por fallo de inferencia.")
            
    if batch:
        cursor.executemany("""
            INSERT OR REPLACE INTO rb_parts_enriched (part_num, formal_description, category_group, utility_labels, enriched_at)
            VALUES (?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        print(f"Progreso final: {processed_count}/{total_pending} piezas procesadas y guardadas.")
        
    conn.close()
    print("Modo producción completado exitosamente.")

def main():
    parser = argparse.ArgumentParser(description="Enriquecimiento de piezas LEGO usando Ollama")
    parser.add_argument("--mode", type=str, choices=["pilot", "production"], required=True, help="Modo de ejecución: pilot o production")
    parser.add_argument("--batch-size", type=int, default=50, help="Tamaño de lote para el modo de producción")
    parser.add_argument("--parts", type=str, nargs="+", help="Lista de referencias específicas a procesar")
    args = parser.parse_args()
    
    if not check_ollama():
        sys.exit(1)
        
    init_db()
    
    if args.mode == "pilot":
        run_pilot()
    elif args.mode == "production":
        run_production(args.batch_size, args.parts)

if __name__ == "__main__":
    main()
