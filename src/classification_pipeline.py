import os
import sys
import sqlite3
import json
import base64
import requests
from typing import Dict, Any, Optional, Literal, List
from pydantic import BaseModel
import ollama

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification_models import LegoTaxonomy, ClassificationResult, LegoAnimalTaxonomy, AnimalClassificationResult
from src.mpd_parser import flatten_mpd
from src.vehicle_rules import evaluate_vehicle_topology

DB_PATH = "data/catalog/models_catalog.db"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5vl:latest"

# Fallback text model if vision is not loaded
TEXT_MODEL_NAME = "qwen2.5:7b"

# --- Structured Output Schema (GBNF Constraint) ---
L0_Type = Literal['Vehículo', 'Animal', 'Popular', 'Espacial', 'Mecha', 'Edificio', 'Personaje', 'Estacional', 'Vida', 'Otros']
L1_Type = Literal['Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno', 'Otros']
L2_Type = Literal['Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia', 'Otros']
L3_Type = Literal['Coche', 'Moto', 'Tren', 'Barco', 'Avión', 'Helicóptero', 'Caza_Estelar', 'Mech_Caminante', 'Desconocido', 'Otros']
L4_Type = Literal['Microscale', 'Minifig-scale', 'UCS/Gran Escala', 'Otros']

class Category(BaseModel):
    L0: L0_Type
    L1: L1_Type
    L2: L2_Type
    L3: L3_Type
    L4: L4_Type

class SetClassification(BaseModel):
    image_1: Category
    image_2: Category
    image_3: Category

def verify_and_pull_model(model_name: str):
    """Verifies if the model exists locally in Ollama; if not, triggers download with progress log."""
    print(f"Verificando modelo '{model_name}' en Ollama local...")
    try:
        models_list = ollama.list()
        installed_models = [m.model for m in getattr(models_list, 'models', [])]
        
        # Check direct match or tag match
        model_exists = False
        for m in installed_models:
            if m == model_name or m.split(':')[0] == model_name.split(':')[0]:
                model_exists = True
                break
                
        if not model_exists:
            print(f"Modelo '{model_name}' no encontrado localmente. Iniciando pull...")
            current_status = ""
            for progress in ollama.pull(model_name, stream=True):
                status = progress.get('status', '')
                completed = progress.get('completed')
                total = progress.get('total')
                if status != current_status:
                    print(f"\n[Ollama Pull] Status: {status}", end="", flush=True)
                    current_status = status
                if total is not None and total > 0 and completed is not None:
                    percent = (completed / total) * 100
                    print(f"\r[Ollama Pull] {status}: {percent:.2f}% ({completed}/{total} bytes)", end="", flush=True)
            print(f"\n[+] Modelo '{model_name}' descargado exitosamente.")
        else:
            print(f"[+] Modelo '{model_name}' verificado y disponible.")
    except Exception as e:
        print(f"[-] Error verificando o descargando modelo '{model_name}': {e}")



# Manual JSON Schema to avoid $defs issues in Ollama structured outputs
OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "set_id": {"type": "string"},
        "source": {"type": "string", "enum": ["Official", "BrickLink", "OMR"]},
        "taxonomy_proposal": {
            "type": "object",
            "properties": {
                "Level_1_Entorno": {"type": "string", "enum": ["Terrestre", "Acuático", "Aéreo", "Espacial", "Multientorno", "Otros"]},
                "Level_2_Proposito": {"type": "string", "enum": ["Civil/Pasajeros", "Carga/Comercial", "Emergencias/Servicios", "Construccion/Industrial", "Competicion/Deportes", "Militar/Combate", "Ficcion/Fantasia", "Otros"]},
                "Level_3_Clase": {"type": "string"},
                "Level_4_Escala": {"type": "string", "enum": ["Microscale", "Minifig-scale", "UCS/Gran Escala", "Otros"]}
            },
            "required": ["Level_1_Entorno", "Level_2_Proposito", "Level_3_Clase", "Level_4_Escala"]
        },
        "confidence_score": {"type": "number"},
        "reasoning_notes": {"type": "string"},
        "needs_human_review": {"type": "boolean"}
    },
    "required": ["set_id", "source", "taxonomy_proposal", "confidence_score", "reasoning_notes", "needs_human_review"]
}

OLLAMA_ANIMAL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "set_id": {"type": "string"},
        "source": {"type": "string", "enum": ["Official", "BrickLink", "OMR"]},
        "taxonomy_proposal": {
            "type": "object",
            "properties": {
                "Level_1_Habitat": {"type": "string", "enum": ["Terrestre", "Acuático", "Aéreo", "Anfibio/Multientorno", "Extinto/Prehistórico", "Mitológico/Fantasía"]},
                "Level_2_Categoria": {"type": "string", "enum": ["Mamífero", "Ave", "Reptil/Anfibio", "Pez/Vida Marina", "Insecto/Invertebrado", "Dinosaurio", "Criatura Fantástica", "Otros"]},
                "Level_3_Especie": {"type": "string"},
                "Level_4_Estilo": {"type": "string", "enum": ["Escala Minifig", "Escultura/Exhibición", "Brick-built (Pequeña escala)", "Otros"]}
            },
            "required": ["Level_1_Habitat", "Level_2_Categoria", "Level_3_Especie", "Level_4_Estilo"]
        },
        "confidence_score": {"type": "number"},
        "reasoning_notes": {"type": "string"},
        "needs_human_review": {"type": "boolean"}
    },
    "required": ["set_id", "source", "taxonomy_proposal", "confidence_score", "reasoning_notes", "needs_human_review"]
}


# --- Extractor Layer ---

class BaseExtractor:
    def extract(self, set_id: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class OfficialLEGOExtractor(BaseExtractor):
    """
    Ingests official sets from rb_sets (Rebrickable database)
    """
    def extract(self, set_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Search by set_id in rb_sets and join with rb_themes
        cursor.execute("""
        SELECT s.set_num, s.name, s.year, t.name as theme_name, s.num_parts, s.img_url
        FROM rb_sets s
        LEFT JOIN rb_themes t ON s.theme_id = t.id
        WHERE s.set_num = ? OR s.set_num LIKE ?
        """, (set_id, f"{set_id}-%"))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        return {
            "set_id": row[0],
            "name": row[1],
            "year": row[2],
            "theme": row[3] or "Unknown",
            "parts_count": row[4],
            "image_url": row[5],
            "tags": "",
            "description": f"Official LEGO Set {row[0]}: {row[1]} under theme {row[3]}.",
            "source": "Official",
            "metadata_3d": {}
        }


class BricklinkGalleryExtractor(BaseExtractor):
    """
    Ingests MOCs from the sets table where source = 'BrickLink'
    """
    def extract(self, set_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT set_id, name, theme, year, description, parts_count, tags, image_url, source_url, file_path
        FROM sets
        WHERE set_id = ? AND source = 'BrickLink'
        """, (set_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        file_path = row[9]
        parts_count = row[5] or 0
        metadata_3d = {}
        
        # Calculate parts count dynamically from LDraw file if missing in DB
        if file_path and os.path.exists(file_path):
            try:
                parts = flatten_mpd(file_path)
                metadata_3d = evaluate_vehicle_topology(parts)
                if parts_count == 0 and len(parts) > 0:
                    parts_count = len(parts)
                    # Update database cache
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE sets SET parts_count = ? WHERE set_id = ?", (parts_count, set_id))
                    conn.commit()
                    conn.close()
                    print(f"  [+] Recalculadas piezas para BrickLink {set_id}: {parts_count} piezas.")
            except Exception as e:
                print(f"Warning parsing 3D file for BrickLink model {set_id}: {e}")
                
        return {
            "set_id": row[0],
            "name": row[1],
            "theme": row[2] or "Studio MOC",
            "year": row[3],
            "description": row[4] or "",
            "parts_count": parts_count,
            "tags": row[6] or "",
            "image_url": row[7],
            "source_url": row[8],
            "source": "BrickLink",
            "metadata_3d": metadata_3d
        }


class OMRExtractor(BaseExtractor):
    """
    Ingests models from sets table where source = 'OMR' and parses LDraw metadata
    """
    def extract(self, set_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
        SELECT set_id, name, theme, year, description, parts_count, tags, image_url, source_url, file_path
        FROM sets
        WHERE set_id = ? AND source = 'OMR'
        """, (set_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
            
        file_path = row[9]
        parts_count = row[5] or 0
        metadata_3d = {}
        
        # Parse physical pieces from LDraw file to get 3D metadata and recount parts if missing
        if file_path and os.path.exists(file_path):
            try:
                parts = flatten_mpd(file_path)
                metadata_3d = evaluate_vehicle_topology(parts)
                if parts_count == 0 and len(parts) > 0:
                    parts_count = len(parts)
                    # Update database cache
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE sets SET parts_count = ? WHERE set_id = ?", (parts_count, set_id))
                    conn.commit()
                    conn.close()
                    print(f"  [+] Recalculadas piezas para OMR {set_id}: {parts_count} piezas.")
            except Exception as e:
                print(f"Warning parsing 3D file for OMR model {set_id}: {e}")
                
        return {
            "set_id": row[0],
            "name": row[1],
            "theme": row[2] or "Unknown",
            "year": row[3],
            "description": row[4] or "",
            "parts_count": parts_count,
            "tags": row[6] or "",
            "image_url": row[7],
            "source_url": row[8],
            "source": "OMR",
            "metadata_3d": metadata_3d
        }


# --- Classifier Agent ---

class VehicleClassifierAgent:
    def __init__(self, model_name: str = MODEL_NAME, fallback_model: str = TEXT_MODEL_NAME):
        self.model_name = model_name
        self.fallback_model = fallback_model
        # Verify model is available locally at initialization
        verify_and_pull_model(self.model_name)
        
    def _download_image_base64(self, image_url: str) -> Optional[str]:
        if not image_url:
            return None
            
        from PIL import Image
        import io
        
        try:
            # Check if it is a local file first
            if os.path.exists(image_url):
                img = Image.open(image_url)
                file_size = os.path.getsize(image_url)
            else:
                if image_url.startswith("//"):
                    image_url = "https:" + image_url
                headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
                res = requests.get(image_url, headers=headers, timeout=10)
                if res.status_code == 200:
                    img = Image.open(io.BytesIO(res.content))
                    file_size = len(res.content)
                else:
                    return None
                
            # Resize strictly to 1024x1024
            img = img.resize((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to RGB if RGBA/PNG transparent
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            compressed_bytes = buffered.getvalue()
            
            print(f"  [+] Imagen redimensionada a 1024x1024 y optimizada de {file_size/1024:.1f}KB a {len(compressed_bytes)/1024:.1f}KB")
            return base64.b64encode(compressed_bytes).decode("utf-8")
        except Exception as e:
            print(f"Warning downloading/processing image {image_url}: {e}")
        return None

    def classify_design(self, metadata: dict):
        set_id = metadata["set_id"]
        source = metadata["source"]
        
        # 0. Early Exit Heuristic / Animal Routing
        name_lower = metadata.get("name", "").lower()
        desc_lower = metadata.get("description", "").lower()
        tags_lower = metadata.get("tags", "").lower()
        theme_lower = metadata.get("theme", "").lower()
        
        meta_3d = metadata.get("metadata_3d", {})
        wheel_count = meta_3d.get("wheel_count", 0)
        
        animal_keywords = [
            "dinosaur", "animal", "creature", "dino", "dragon", "pet", "cat", "dog", "bear",
            "shark", "bird", "eagle", "duck", "owl", "horse", "snake", "crocodile", "lion",
            "tiger", "elephant", "fish", "whale", "spider", "scorpion", "crab", "lizard",
            "monster", "beast", "pato"
        ]
        
        is_animal = False
        for kw in animal_keywords:
            if (kw in name_lower or kw in desc_lower or kw in tags_lower or kw in theme_lower) and wheel_count == 0:
                is_animal = True
                break
                
        matched_l0 = None
        matched_kw = None
        
        if not is_animal:
            # Map keyword categories
            l0_keyword_map = {
                "Edificio": [
                    "building", "house", "castle", "temple", "tower", "bridge", 
                    "lighthouse", "shop", "market", "modular", "diorama", "display", "station", "garden"
                ],
                "Mecha": [
                    "mech", "robot"
                ],
                "Personaje": [
                    "minifig", "figure", "statue", "brickheadz", "portrait", "painting", "helmet", "mask"
                ],
                "Estacional": [
                    "christmas", "halloween", "easter", "holiday", "seasonal"
                ],
                "Vida": [
                    "flower", "tree", "plant", "vegetation", "furniture", "chair", "table", "clock", 
                    "camera", "piano", "instrument", "organ", "plate", "tile"
                ],
                "Otros": [
                    "box", "case", "cup", "mug", "keychain", "magnet", "sculpture", 
                    "weapon", "sword", "shield", "gun"
                ]
            }
            
            for l0_cat, kws in l0_keyword_map.items():
                for kw in kws:
                    if (kw in name_lower or kw in desc_lower or kw in tags_lower or kw in theme_lower) and wheel_count == 0:
                        matched_l0 = l0_cat
                        matched_kw = kw
                        break
                if matched_l0:
                    break
                    
        if matched_l0:
            print(f"  [Early Exit] Filtro rápido detectó {matched_l0} (coincidencia con '{matched_kw}' y 0 ruedas). Omitiendo VLM.")
            taxonomy_proposal = LegoTaxonomy(
                Level_0_Categoria=matched_l0,
                Level_1_Entorno="Otros",
                Level_2_Proposito="Otros",
                Level_3_Clase="Otros",
                Level_4_Escala="Otros"
            )
            return ClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=taxonomy_proposal,
                confidence_score=0.10,
                reasoning_notes=f"Filtro rápido de texto y física 3D identificó el diseño como {matched_l0} (con palabra clave '{matched_kw}' y 0 ruedas).",
                needs_human_review=True,
                review_table_payload={
                    "columns": ["Set/MOC", "Origen", "Confianza", "Propuesta Nivel 0", "Propuesta Nivel 1", "Propuesta Nivel 2", "Motivo de Baja Confianza", "Acción"],
                    "row_data": {
                        "set_name": metadata.get("name", set_id),
                        "source": source,
                        "score": "10%",
                        "prop_L0": matched_l0,
                        "prop_L1": "Otros",
                        "prop_L2": "Otros",
                        "conflict_alert": f"Filtro rápido detectó {matched_l0} (palabra clave '{matched_kw}' y 0 ruedas)."
                    },
                    "editable_fields": ["prop_L0", "prop_L1", "prop_L2", "escala"]
                }
            )
            
        # 1. Download and convert available images to Base64 (prioritizing primary image, limit to 3)
        image_base64_list = []
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT image_url FROM set_images WHERE set_id = ?", (set_id,))
        img_rows = cursor.fetchall()
        conn.close()
        
        image_urls = [r[0] for r in img_rows]
        primary_url = metadata.get("image_url")
        
        if primary_url:
            image_urls = [primary_url] + [url for url in image_urls if url != primary_url]
            
        # Deduplicate URLs preserving order
        seen = set()
        image_urls = [x for x in image_urls if not (x in seen or seen.add(x))]
        
        # Limit to a maximum of 3 images to avoid massive VLM queues
        if len(image_urls) > 3:
            image_urls = image_urls[:3]
            
        if image_urls:
            print(f"  [+] Descargando y codificando {len(image_urls)} imágenes para análisis del VLM...")
            for url in image_urls:
                b64 = self._download_image_base64(url)
                if b64:
                    image_base64_list.append(b64)
                    
        # If no images, default fallback
        if not image_base64_list:
            print(f"  [-] No hay imágenes disponibles para {set_id}. Utilizando clasificación por defecto.")
            return self._get_fallback_result(set_id, source, metadata, is_animal, "Sin imágenes disponibles")

        # Group images into batches of exactly 3
        batches = []
        for i in range(0, len(image_base64_list), 3):
            batch = image_base64_list[i:i+3]
            if len(batch) > 0:
                while len(batch) < 3:
                    batch.append(batch[-1])  # Pad last batch by replicating last image
                batches.append(batch)

        # Determine VLM model to use, with fallback support if target model is not found
        model_to_use = self.model_name
        try:
            models_list = ollama.list()
            installed_models = [m.model for m in getattr(models_list, 'models', [])]
            installed_match = False
            for m in installed_models:
                if m == self.model_name or m.split(':')[0] == self.model_name.split(':')[0]:
                    installed_match = True
                    model_to_use = m
                    break
            if not installed_match:
                # Look for other vision models
                vlm_fallbacks = [m for m in installed_models if 'vl' in m.lower() or 'vision' in m.lower()]
                if vlm_fallbacks:
                    model_to_use = vlm_fallbacks[0]
                    print(f"  [!] Modelo '{self.model_name}' no disponible. Usando fallback VLM: '{model_to_use}'")
        except Exception as e:
            print(f"  [!] Error comprobando fallbacks en Ollama: {e}")

        predictions = []
        
        system_prompt = (
            "Eres un Arquitecto de Software Senior y experto en IA Multimodal y taxonomías de LEGO.\n"
            "Tu tarea es analizar detalladamente las imágenes de entrada correspondientes al diseño LEGO "
            "y asignar una etiqueta taxonómica estructurada de L0 a L4.\n\n"
            "Guías taxonómicas:\n"
            "- L0 (Categoría Principal): 'Vehículo', 'Animal', 'Popular', 'Espacial', 'Mecha', 'Edificio', 'Personaje', 'Estacional', 'Vida', 'Otros'\n"
            "- L1 (Entorno/Hábitat): 'Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno', 'Otros'\n"
            "- L2 (Propósito/Categoría): 'Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia', 'Otros'\n"
            "- L3 (Clase/Especie): El tipo específico (ej. Coche, Moto, Tren, Barco, Avión, Helicóptero, Mecha, Casa, Castillo, Flor, Dragon, etc. El modelo debe inferirlo)\n"
            "- L4 (Escala/Estilo): 'Microscale', 'Minifig-scale', 'UCS/Gran Escala', 'Escultura/Exhibición', 'Otros'"
        )

        user_prompt = (
            f"Clasifica por favor estas 3 imágenes del set LEGO ID {set_id}.\n"
            f"Nombre: {metadata.get('name')}\n"
            f"Temática: {metadata.get('theme')}\n"
            f"Descripción: {metadata.get('description')}\n"
        )

        for b_idx, batch in enumerate(batches):
            print(f"  [+] Ejecutando inferencia VLM en lote {b_idx + 1}/{len(batches)} con '{model_to_use}'...")
            try:
                response = ollama.chat(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt, "images": batch}
                    ],

                    format=SetClassification.model_json_schema(),
                    options={
                        "temperature": 0.1
                    }
                )
                raw_response = response.get("message", {}).get("content", "").strip()
                parsed = SetClassification.model_validate_json(raw_response)
                predictions.append(parsed.image_1)
                predictions.append(parsed.image_2)
                predictions.append(parsed.image_3)
            except Exception as e:
                print(f"  [-] Error en inferencia VLM del lote {b_idx + 1}: {e}")
                
        if not predictions:
            return self._get_fallback_result(set_id, source, metadata, is_animal, "Error en inferencia de Ollama")

        # 4. Aggregate results (voting)
        from collections import Counter
        l0_votes = [p.L0 for p in predictions]
        l1_votes = [p.L1 for p in predictions]
        l2_votes = [p.L2 for p in predictions]
        l3_votes = [p.L3 for p in predictions]
        l4_votes = [p.L4 for p in predictions]

        l0_val = Counter(l0_votes).most_common(1)[0][0] if l0_votes else "Otros"
        l1_val = Counter(l1_votes).most_common(1)[0][0] if l1_votes else "Otros"
        l2_val = Counter(l2_votes).most_common(1)[0][0] if l2_votes else "Otros"
        l3_val = Counter(l3_votes).most_common(1)[0][0] if l3_votes else "Otros"
        l4_val = Counter(l4_votes).most_common(1)[0][0] if l4_votes else "Otros"

        is_animal_predicted = (l0_val == "Animal")
        confidence = 0.90 if len(predictions) > 0 else 0.10
        reasoning = f"Clasificación agregada de {len(predictions)} imágenes. Votos L0: {Counter(l0_votes)}."

        if is_animal_predicted:
            taxonomy_proposal = LegoAnimalTaxonomy(
                Level_1_Habitat=l1_val if l1_val in ['Terrestre', 'Acuático', 'Aéreo', 'Anfibio/Multientorno', 'Extinto/Prehistórico', 'Mitológico/Fantasía'] else 'Terrestre',
                Level_2_Categoria=l2_val if l2_val in ['Mamífero', 'Ave', 'Reptil/Anfibio', 'Pez/Vida Marina', 'Insecto/Invertebrado', 'Dinosaurio', 'Criatura Fantástica', 'Otros'] else 'Otros',
                Level_3_Especie=l3_val,
                Level_4_Estilo=l4_val if l4_val in ['Escala Minifig', 'Escultura/Exhibición', 'Brick-built (Pequeña escala)', 'Otros'] else 'Otros'
            )
            needs_human = confidence <= 0.80
            review_payload = {
                "columns": ["Set/MOC", "Origen", "Confianza", "Hábitat N1", "Categoría N2", "Especie N3", "Motivo de Baja Confianza", "Acción"],
                "row_data": {
                    "set_name": metadata.get("name", set_id),
                    "source": source,
                    "score": f"{int(confidence * 100)}%",
                    "prop_L1": taxonomy_proposal.Level_1_Habitat,
                    "prop_L2": taxonomy_proposal.Level_2_Categoria,
                    "prop_L3": taxonomy_proposal.Level_3_Especie,
                    "conflict_alert": reasoning
                },
                "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala"]
            }
            return AnimalClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=taxonomy_proposal,
                confidence_score=confidence,
                reasoning_notes=reasoning,
                needs_human_review=needs_human,
                review_table_payload=review_payload
            )
        else:
            taxonomy_proposal = LegoTaxonomy(
                Level_0_Categoria=l0_val,
                Level_1_Entorno=l1_val if l1_val in ['Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno', 'Otros'] else 'Otros',
                Level_2_Proposito=l2_val if l2_val in ['Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia', 'Otros'] else 'Otros',
                Level_3_Clase=l3_val,
                Level_4_Escala=l4_val if l4_val in ['Microscale', 'Minifig-scale', 'UCS/Gran Escala', 'Otros'] else 'Otros'
            )
            needs_human = confidence <= 0.80
            review_payload = {
                "columns": ["Set/MOC", "Origen", "Confianza", "Propuesta Nivel 1", "Propuesta Nivel 2", "Propuesta Nivel 3", "Motivo de Baja Confianza", "Acción"],
                "row_data": {
                    "set_name": metadata.get("name", set_id),
                    "source": source,
                    "score": f"{int(confidence * 100)}%",
                    "prop_L1": taxonomy_proposal.Level_1_Entorno,
                    "prop_L2": taxonomy_proposal.Level_2_Proposito,
                    "prop_L3": taxonomy_proposal.Level_3_Clase,
                    "conflict_alert": reasoning
                },
                "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala"]
            }
            return ClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=taxonomy_proposal,
                confidence_score=confidence,
                reasoning_notes=reasoning,
                needs_human_review=needs_human,
                review_table_payload=review_payload
            )

    def _get_fallback_result(self, set_id: str, source: str, metadata: dict, is_animal: bool, error_msg: str):
        if is_animal:
            fallback_taxonomy = LegoAnimalTaxonomy(
                Level_1_Habitat="Terrestre",
                Level_2_Categoria="Otros",
                Level_3_Especie="Desconocido",
                Level_4_Estilo="Otros"
            )
            return AnimalClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=fallback_taxonomy,
                confidence_score=0.1,
                reasoning_notes=f"Inference failure: {error_msg}",
                needs_human_review=True,
                review_table_payload={
                    "columns": ["Set/MOC", "Origen", "Confianza", "Hábitat N1", "Categoría N2", "Especie N3", "Motivo de Baja Confianza", "Acción"],
                    "row_data": {
                        "set_name": metadata.get("name", set_id),
                        "source": source,
                        "score": "10%",
                        "prop_L1": "Terrestre",
                        "prop_L2": "Otros",
                        "prop_L3": "Desconocido",
                        "conflict_alert": f"Fallo: {error_msg}"
                    },
                    "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala"]
                }
            )
        else:
            fallback_taxonomy = LegoTaxonomy(
                Level_0_Categoria="Vehículo",
                Level_1_Entorno="Terrestre",
                Level_2_Proposito="Civil/Pasajeros",
                Level_3_Clase="Desconocido",
                Level_4_Escala="Minifig-scale"
            )
            return ClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=fallback_taxonomy,
                confidence_score=0.1,
                reasoning_notes=f"Inference failure: {error_msg}",
                needs_human_review=True,
                review_table_payload={
                    "columns": ["Set/MOC", "Origen", "Confianza", "Propuesta Nivel 1", "Propuesta Nivel 2", "Propuesta Nivel 3", "Motivo de Baja Confianza", "Acción"],
                    "row_data": {
                        "set_name": metadata.get("name", set_id),
                        "source": source,
                        "score": "10%",
                        "prop_L1": "Terrestre",
                        "prop_L2": "Civil/Pasajeros",
                        "prop_L3": "Desconocido",
                        "conflict_alert": f"Fallo de conexión o parseo: {error_msg}"
                    },
                    "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala"]
                }
            )



# --- Database Integrator ---

def save_classification_to_db(result):
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    cursor = conn.cursor()
    
    tax = result.taxonomy_proposal
    payload_str = json.dumps(result.review_table_payload, ensure_ascii=False) if result.review_table_payload else None
    
    # Check if the set exists in sets table
    cursor.execute("SELECT set_id FROM sets WHERE set_id = ?", (result.set_id,))
    exists = cursor.fetchone()
    
    # Is this animal or vehicle?
    is_animal = hasattr(tax, "Level_1_Habitat") or getattr(tax, "Level_0_Categoria", None) == "Animal"
    l0_val = getattr(tax, "Level_0_Categoria", "Otros")
    if is_animal:
        l0_val = "Animal"
    
    if exists:
        if is_animal:
            cursor.execute("""
            UPDATE sets
            SET level_0_categoria = ?,
                animal_level_1_habitat = ?,
                animal_level_2_categoria = ?,
                animal_level_3_especie = ?,
                animal_level_4_estilo = ?,
                animal_confidence_score = ?,
                animal_reasoning_notes = ?,
                level_1_entorno = 'Otros',
                level_2_proposito = 'Otros',
                level_3_clase = 'Otros',
                level_4_escala = 'Otros',
                confidence_score = ?,
                reasoning_notes = ?,
                needs_human_review = ?,
                review_table_payload = ?,
                classification_status = 'vlm_classified'
            WHERE set_id = ?
            """, (
                l0_val,
                tax.Level_1_Habitat,
                tax.Level_2_Categoria,
                tax.Level_3_Especie,
                tax.Level_4_Estilo,
                result.confidence_score,
                result.reasoning_notes,
                result.confidence_score,
                result.reasoning_notes,
                1 if result.needs_human_review else 0,
                payload_str,
                result.set_id
            ))
        else:
            cursor.execute("""
            UPDATE sets
            SET level_0_categoria = ?,
                level_1_entorno = ?,
                level_2_proposito = ?,
                level_3_clase = ?,
                level_4_escala = ?,
                confidence_score = ?,
                reasoning_notes = ?,
                needs_human_review = ?,
                review_table_payload = ?,
                classification_status = 'vlm_classified'
            WHERE set_id = ?
            """, (
                l0_val,
                tax.Level_1_Entorno,
                tax.Level_2_Proposito,
                tax.Level_3_Clase,
                tax.Level_4_Escala,
                result.confidence_score,
                result.reasoning_notes,
                1 if result.needs_human_review else 0,
                payload_str,
                result.set_id
            ))
    else:
        # Create a new record in sets for this Official set
        cursor.execute("SELECT name, year, img_url, num_parts FROM rb_sets WHERE set_num = ?", (result.set_id,))
        rb_info = cursor.fetchone()
        name = rb_info[0] if rb_info else result.set_id
        year = rb_info[1] if rb_info else None
        img_url = rb_info[2] if rb_info else None
        num_parts = rb_info[3] if rb_info else 0
        
        if is_animal:
            cursor.execute("""
            INSERT INTO sets (
                set_id, name, source, year, image_url, parts_count,
                level_0_categoria,
                animal_level_1_habitat, animal_level_2_categoria, animal_level_3_especie, animal_level_4_estilo,
                animal_confidence_score, animal_reasoning_notes,
                level_1_entorno, level_2_proposito, level_3_clase, level_4_escala,
                confidence_score, reasoning_notes, needs_human_review, review_table_payload, classification_status
            )
            VALUES (?, ?, 'Official', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'Otros', 'Otros', 'Otros', 'Otros', ?, ?, ?, ?, 'vlm_classified')
            """, (
                result.set_id,
                name,
                year,
                img_url,
                num_parts,
                l0_val,
                tax.Level_1_Habitat,
                tax.Level_2_Categoria,
                tax.Level_3_Especie,
                tax.Level_4_Estilo,
                result.confidence_score,
                result.reasoning_notes,
                result.confidence_score,
                result.reasoning_notes,
                1 if result.needs_human_review else 0,
                payload_str
            ))
        else:
            cursor.execute("""
            INSERT INTO sets (
                set_id, name, source, year, image_url, parts_count,
                level_0_categoria,
                level_1_entorno, level_2_proposito, level_3_clase, level_4_escala,
                confidence_score, reasoning_notes, needs_human_review, review_table_payload, classification_status
            )
            VALUES (?, ?, 'Official', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'vlm_classified')
            """, (
                result.set_id,
                name,
                year,
                img_url,
                num_parts,
                l0_val,
                tax.Level_1_Entorno,
                tax.Level_2_Proposito,
                tax.Level_3_Clase,
                tax.Level_4_Escala,
                result.confidence_score,
                result.reasoning_notes,
                1 if result.needs_human_review else 0,
                payload_str
            ))
            
    conn.commit()
    conn.close()
