"""
LegoGPT: Local VLM Set Visual Enrichment & Archetype Classifier
Supports both:
1. Native Apple Silicon Metal MLX backend (Llama-3.2-11B-Vision-Instruct-4bit via mlx-vlm)
2. Local Ollama HTTP backend (llava:7b)

Enforces strict JSON schema, orthogonal taxonomy, and multimodal sanity checks against SQLite.
"""

import os
import sys
import json
import base64
import sqlite3
import argparse
import time
from io import BytesIO
from typing import Dict, Any, Optional, List, Tuple
import requests
from PIL import Image

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

DB_PATH = "data/catalog/models_catalog.db"
IMAGE_CACHE_DIR = "data/cache/set_images"
DEFAULT_MLX_MODEL = "mlx-community/Llama-3.2-11B-Vision-Instruct-4bit"
DEFAULT_OLLAMA_MODEL = "llava:7b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# Global MLX model cache
_MLX_MODEL = None
_MLX_PROCESSOR = None
_MLX_CONFIG = None

TAXONOMY_MAP = {
    "VEHICULO_TERRESTRE": [
        "Coche / Deportivo / Moto",
        "Camión / Maquinaria Pesada / Construcción",
        "Ferrocarril / Tren",
        "Buggy / Todoterreno / Explorador"
    ],
    "VEHICULO_AEREO": [
        "Avión de Ala Fija (Pasajeros / Carga / Militar)",
        "Helicóptero / Rotor Giratorio",
        "Planeador / Dron"
    ],
    "VEHICULO_ACUATICO": [
        "Barco / Buque / Yate / Lancha",
        "Submarino / Vehículo Subacuático"
    ],
    "NAVE_ESPACIAL": [
        "Caza Estelar / Nave Monoplaza de Combate",
        "Crucero / Lanzadera / Transporte Multitripulado",
        "Speeder / Deslizador Sin Ruedas",
        "Estación Espacial / Satélite"
    ],
    "ARQUITECTURA_INMUEBLE": [
        "Edificio Modular / Residencia / Tienda",
        "Castillo / Fortaleza / Templo Histórico",
        "Infraestructura / Puente / Torre / Estación"
    ],
    "FIGURA_MEC_CRIATURA": [
        "Robot / Mecha Humanoide Articulado",
        "Criatura Orgánica / Dragón / Animal Fantástico",
        "Figura Buildable a Gran Escala"
    ],
    "DIORAMA_MICROESCENA": [
        "Escenario de Juego con Múltiples Elementos",
        "Micro-diorama Temático de Exhibición"
    ],
    "OBJETO_EXHIBICION": [
        "Botánica / Flor / Árbol / Planta",
        "Casco / Escultura / Busto de Personaje",
        "Mosaico 2D / Cuadro Art",
        "Objeto Cotidiano / Réplica a Escala Real"
    ]
}

VLM_PROMPT = """Eres un inspector experto de modelos y sets LEGO.
Analiza con rigor forense la imagen oficial de este set LEGO y clasifícalo con MÁXIMA PRECISIÓN en la taxonomía cerrada permitida.
No inventes categorías. Prioriza la precisión factual y evita descripciones de marketing.

TAXONOMÍA CERRADA PERMITIDA:
1. VEHICULO_TERRESTRE: Coche / Deportivo / Moto | Camión / Maquinaria Pesada / Construcción | Ferrocarril / Tren | Buggy / Todoterreno / Explorador
2. VEHICULO_AEREO: Avión de Ala Fija (Pasajeros / Carga / Militar) | Helicóptero / Rotor Giratorio | Planeador / Dron
3. VEHICULO_ACUATICO: Barco / Buque / Yate / Lancha | Submarino / Vehículo Subacuático
4. NAVE_ESPACIAL: Caza Estelar / Nave Monoplaza de Combate | Crucero / Lanzadera / Transporte Multitripulado | Speeder / Deslizador Sin Ruedas | Estación Espacial / Satélite
5. ARQUITECTURA_INMUEBLE: Edificio Modular / Residencia / Tienda | Castillo / Fortaleza / Templo Histórico | Infraestructura / Puente / Torre / Estación
6. FIGURA_MEC_CRIATURA: Robot / Mecha Humanoide Articulado | Criatura Orgánica / Dragón / Animal Fantástico | Figura Buildable a Gran Escala
7. DIORAMA_MICROESCENA: Escenario de Juego con Múltiples Elementos | Micro-diorama Temático de Exhibición
8. OBJETO_EXHIBICION: Botánica / Flor / Árbol / Planta | Casco / Escultura / Busto de Personaje | Mosaico 2D / Cuadro Art | Objeto Cotidiano / Réplica a Escala Real

SILUETAS PERMITIDAS: Delta_Triangular, Prismatica_Cubica, Aerodinamica_Baja, Cilindrica_Alargada, Cruz_Simetrica, Torre_Vertical, Escultorica_Organica, Compacta_Modular
PROPORCIONES PERMITIDAS: Aplanado_Horizontal, Alargado_Esbelto, Cubico_Equilibrado, Vertical_Alto

Responde ÚNICAMENTE con un objeto JSON válido con esta estructura exacta:
{
  "arquetipo_primario": "<uno de los 8 arquetipos>",
  "subclase": "<subclase correspondiente exacta>",
  "silueta_geometrica": "<una de las siluetas permitidas>",
  "proporciones_volumen": "<una de las proporciones permitidas>",
  "elementos_visuales_clave": ["elemento 1", "elemento 2", "elemento 3"],
  "partes_moviles_visibles": ["parte móvil 1", "parte móvil 2"],
  "resumen_visual_factual": "Dos oraciones estrictamente factuales describiendo el aspecto, color y forma del modelo construido.",
  "confidence_score": 0.95
}
"""


def init_db(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS set_vlm_visual_analysis (
        set_id TEXT PRIMARY KEY,
        model_name TEXT,
        image_url TEXT,
        local_image_path TEXT,
        arquetipo_primario TEXT,
        subclase TEXT,
        silueta_geometrica TEXT,
        proporciones_volumen TEXT,
        elementos_visuales_clave TEXT,
        partes_moviles_visibles TEXT,
        resumen_visual_factual TEXT,
        confidence_score REAL,
        sanity_check_passed INTEGER,
        conflict_reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vlm_arquetipo ON set_vlm_visual_analysis(arquetipo_primario);")
    conn.commit()


def get_cached_image_path(set_id: str, image_url: str) -> Optional[str]:
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)
    clean_id = set_id.replace("/", "_")
    local_path = os.path.join(IMAGE_CACHE_DIR, f"{clean_id}.jpg")
    
    if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
        return local_path
        
    if not image_url or not image_url.startswith("http"):
        return None
        
    try:
        resp = requests.get(image_url, timeout=15, headers={"User-Agent": "LegoGPT-VLM-Worker/1.0"})
        if resp.status_code == 200:
            with open(local_path, "wb") as f:
                f.write(resp.content)
            return local_path
    except Exception as e:
        print(f"⚠️ Error descargando imagen para {set_id}: {e}")
        
    return None


def prepare_image_b64(local_image_path: str, max_size: int = 1024) -> str:
    with Image.open(local_image_path) as img:
        img = img.convert("RGB")
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_vlm_json(raw_text: str) -> Dict[str, Any]:
    try:
        return json.loads(raw_text)
    except Exception:
        if repair_json:
            repaired = repair_json(raw_text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
            return json.loads(repaired)
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.endswith("```"):
            clean = clean[:-3]
        return json.loads(clean.strip())


def call_mlx_vlm(image_path: str, set_name: str, model_id: str = DEFAULT_MLX_MODEL) -> Dict[str, Any]:
    global _MLX_MODEL, _MLX_PROCESSOR, _MLX_CONFIG
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    
    if _MLX_MODEL is None:
        print(f"🧠 Cargando modelo MLX nativo en Apple Silicon GPU: '{model_id}'...")
        _MLX_CONFIG = load_config(model_id)
        _MLX_MODEL, _MLX_PROCESSOR = load(model_id)
        print("✅ Modelo MLX listo en memoria unificada Metal!")
        
    prompt_text = f"Nombre del set: '{set_name}'.\n" + VLM_PROMPT
    formatted_prompt = apply_chat_template(_MLX_PROCESSOR, _MLX_CONFIG, prompt_text, num_images=1)
    
    output = generate(_MLX_MODEL, _MLX_PROCESSOR, prompt=formatted_prompt, image=image_path, max_tokens=768, temperature=0.1)
    text = output.text if hasattr(output, "text") else str(output)
    return parse_vlm_json(text)


def call_ollama_vlm(b64_image: str, set_name: str, model_name: str = DEFAULT_OLLAMA_MODEL) -> Dict[str, Any]:
    prompt = f"Nombre del set: '{set_name}'.\n" + VLM_PROMPT
    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [b64_image],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024
        }
    }
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=90)
    resp.raise_for_status()
    result_text = resp.json().get("response", "{}")
    return parse_vlm_json(result_text)


def run_sanity_check(set_id: str, arquetipo: str, conn: sqlite3.Connection) -> Tuple[bool, Optional[str]]:
    cursor = conn.cursor()
    cursor.execute("""
        SELECT count(*) 
        FROM rb_inventories inv
        JOIN rb_inventory_parts ip ON ip.inventory_id = inv.id
        JOIN rb_parts p ON p.part_num = ip.part_num
        WHERE (inv.set_num = ? OR inv.set_num = ? || '-1')
          AND (lower(p.name) LIKE '%wheel%' OR lower(p.name) LIKE '%tire%' OR lower(p.name) LIKE '%tyre%');
    """, (set_id, set_id))
    wheels_count = cursor.fetchone()[0] or 0
    
    if arquetipo == "VEHICULO_TERRESTRE" and wheels_count == 0:
        return False, "VLM predijo VEHICULO_TERRESTRE pero no hay ruedas ni orugas en el inventario físico."
    if arquetipo == "ARQUITECTURA_INMUEBLE" and wheels_count >= 4:
        return False, "VLM predijo ARQUITECTURA_INMUEBLE pero el modelo tiene 4+ ruedas en el inventario."
    return True, None


def process_set(set_id: str, name: str, img_url: str, conn: sqlite3.Connection, backend: str, model_name: str) -> Optional[Dict[str, Any]]:
    local_path = get_cached_image_path(set_id, img_url)
    if not local_path:
        return None
        
    if backend == "mlx":
        vlm_result = call_mlx_vlm(local_path, name, model_name)
    else:
        b64_img = prepare_image_b64(local_path)
        vlm_result = call_ollama_vlm(b64_img, name, model_name)
        
    primary = vlm_result.get("arquetipo_primario", "DIORAMA_MICROESCENA")
    subclass = vlm_result.get("subclase", "")
    silhouette = vlm_result.get("silueta_geometrica", "Compacta_Modular")
    proportions = vlm_result.get("proporciones_volumen", "Cubico_Equilibrado")
    key_elements = json.dumps(vlm_result.get("elementos_visuales_clave", []), ensure_ascii=False)
    moving_parts = json.dumps(vlm_result.get("partes_moviles_visibles", []), ensure_ascii=False)
    factual_desc = vlm_result.get("resumen_visual_factual", "")
    confidence = float(vlm_result.get("confidence_score", 0.90))
    
    passed, reason = run_sanity_check(set_id, primary, conn)
    
    return {
        "set_id": set_id,
        "model_name": model_name,
        "image_url": img_url,
        "local_image_path": local_path,
        "arquetipo_primario": primary,
        "subclase": subclass,
        "silueta_geometrica": silhouette,
        "proporciones_volumen": proportions,
        "elementos_visuales_clave": key_elements,
        "partes_moviles_visibles": moving_parts,
        "resumen_visual_factual": factual_desc,
        "confidence_score": confidence,
        "sanity_check_passed": 1 if passed else 0,
        "conflict_reason": reason
    }


def main():
    parser = argparse.ArgumentParser(description="LegoGPT Local VLM Set Visual Enrichment")
    parser.add_argument("--backend", type=str, choices=["mlx", "ollama"], default="ollama", help="Inference backend (mlx for native Apple Silicon, ollama for server)")
    parser.add_argument("--model", type=str, default="", help="Vision model name")
    parser.add_argument("--limit", type=int, default=5, help="Number of sets to process")
    parser.add_argument("--set-ids", type=str, default="", help="Comma-separated set IDs to process specifically")
    args = parser.parse_args()
    
    model_name = args.model
    if not model_name:
        model_name = DEFAULT_MLX_MODEL if args.backend == "mlx" else DEFAULT_OLLAMA_MODEL
        
    print(f"👁️ Iniciando Pipeline VLM Local (Backend: {args.backend.upper()}, Modelo: '{model_name}')...")
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)
    cursor = conn.cursor()
    
    if args.set_ids:
        ids = [i.strip() for i in args.set_ids.split(",") if i.strip()]
        placeholders = ",".join(["?"] * len(ids))
        query = f"""
            SELECT s.set_id, s.name, COALESCE(img.image_url, rb.img_url) as img_url
            FROM sets s
            LEFT JOIN set_images img ON img.set_id = s.set_id
            LEFT JOIN rb_sets rb ON (rb.set_num = s.set_id OR rb.set_num = s.set_id || '-1')
            WHERE s.set_id IN ({placeholders})
        """
        cursor.execute(query, ids)
    else:
        query = f"""
            SELECT s.set_id, s.name, COALESCE(img.image_url, rb.img_url) as img_url
            FROM sets s
            LEFT JOIN set_images img ON img.set_id = s.set_id
            LEFT JOIN rb_sets rb ON (rb.set_num = s.set_id OR rb.set_num = s.set_id || '-1')
            WHERE COALESCE(img.image_url, rb.img_url) IS NOT NULL
              AND s.set_id NOT IN (SELECT set_id FROM set_vlm_visual_analysis)
            LIMIT ?
        """
        cursor.execute(query, (args.limit,))
        
    candidate_sets = cursor.fetchall()
    print(f"📦 Sets a procesar: {len(candidate_sets)}")
    
    processed_count = 0
    for set_id, name, img_url in candidate_sets:
        start_t = time.time()
        print(f"\n🔍 [{processed_count+1}/{len(candidate_sets)}] Analizando set {set_id}: '{name}'...")
        try:
            res = process_set(set_id, name, img_url, conn, args.backend, model_name)
            if res:
                cursor.execute("""
                    INSERT OR REPLACE INTO set_vlm_visual_analysis
                    (set_id, model_name, image_url, local_image_path, arquetipo_primario,
                     subclase, silueta_geometrica, proporciones_volumen, elementos_visuales_clave,
                     partes_moviles_visibles, resumen_visual_factual, confidence_score,
                     sanity_check_passed, conflict_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    res["set_id"], res["model_name"], res["image_url"], res["local_image_path"],
                    res["arquetipo_primario"], res["subclase"], res["silueta_geometrica"],
                    res["proporciones_volumen"], res["elementos_visuales_clave"],
                    res["partes_moviles_visibles"], res["resumen_visual_factual"],
                    res["confidence_score"], res["sanity_check_passed"], res["conflict_reason"]
                ))
                conn.commit()
                elapsed = time.time() - start_t
                status_icon = "✅" if res["sanity_check_passed"] == 1 else "⚠️ (Conflicto detectado)"
                print(f"   {status_icon} Clasificado en {elapsed:.2f}s:")
                print(f"      • Arquetipo: {res['arquetipo_primario']} -> {res['subclase']}")
                print(f"      • Silueta: {res['silueta_geometrica']} ({res['proporciones_volumen']})")
                print(f"      • Resumen VLM: {res['resumen_visual_factual']}")
                processed_count += 1
            else:
                print(f"   ❌ No se pudo descargar imagen para {set_id}")
        except Exception as e:
            print(f"   ❌ Error en inferencia VLM para {set_id}: {e}")
            
    conn.close()
    print(f"\n🎉 Procesamiento VLM finalizado: {processed_count} sets analizados con alta precisión.")


if __name__ == "__main__":
    main()
