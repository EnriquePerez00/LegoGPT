import os
import sys
import sqlite3
import json
import base64
import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.classification_models import LegoTaxonomy, ClassificationResult
from src.mpd_parser import flatten_mpd
from src.vehicle_rules import evaluate_vehicle_topology

DB_PATH = "data/catalog/models_catalog.db"
OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "qwen2.5:7b"

# Fallback text model if vision is not loaded
TEXT_MODEL_NAME = "qwen2.5:7b"

# Manual JSON Schema to avoid $defs issues in Ollama structured outputs
OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "set_id": {"type": "string"},
        "source": {"type": "string", "enum": ["Official", "BrickLink", "OMR"]},
        "taxonomy_proposal": {
            "type": "object",
            "properties": {
                "Level_1_Entorno": {"type": "string", "enum": ["Terrestre", "Acuático", "Aéreo", "Espacial", "Multientorno"]},
                "Level_2_Proposito": {"type": "string", "enum": ["Civil/Pasajeros", "Carga/Comercial", "Emergencias/Servicios", "Construccion/Industrial", "Competicion/Deportes", "Militar/Combate", "Ficcion/Fantasia"]},
                "Level_3_Clase": {"type": "string"},
                "Level_4_Escala": {"type": "string", "enum": ["Microscale", "Minifig-scale", "UCS/Gran Escala"]},
                "Level_4_Motorizacion": {"type": "string", "enum": ["Estatico", "Ruedas Libres", "Pull-back", "Motorizado"]},
                "Level_4_Licencia": {"type": "string"}
            },
            "required": ["Level_1_Entorno", "Level_2_Proposito", "Level_3_Clase", "Level_4_Escala", "Level_4_Motorizacion", "Level_4_Licencia"]
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
        
    def _download_image_base64(self, image_url: str) -> Optional[str]:
        if not image_url:
            return None
            
        if image_url.startswith("//"):
            image_url = "https:" + image_url
            
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
            res = requests.get(image_url, headers=headers, timeout=10)
            if res.status_code == 200:
                from PIL import Image
                import io
                
                # Open image from bytes
                img = Image.open(io.BytesIO(res.content))
                # Resize keeping aspect ratio
                img.thumbnail((448, 448), Image.Resampling.LANCZOS)
                
                # Convert to RGB if RGBA/PNG transparent
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                    
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=80)
                compressed_bytes = buffered.getvalue()
                
                print(f"  [+] Imagen redimensionada y optimizada de {len(res.content)/1024:.1f}KB a {len(compressed_bytes)/1024:.1f}KB")
                return base64.b64encode(compressed_bytes).decode("utf-8")
        except Exception as e:
            print(f"Warning downloading image {image_url}: {e}")
        return None

    def classify_design(self, metadata: dict) -> ClassificationResult:
        set_id = metadata["set_id"]
        source = metadata["source"]
        
        # 1. Download image and convert to Base64 if available
        image_base64 = None
        if metadata.get("image_url"):
            print(f"  [+] Descargando y codificando imagen para {set_id}...")
            image_base64 = self._download_image_base64(metadata["image_url"])
            
        # 2. Build system and user prompt
        system_prompt = (
            "Eres un Arquitecto de Software Senior y experto en IA Multimodal y taxonomías de LEGO.\n"
            "Tu tarea es analizar la información textual (título, temática, tags, descripción) junto con "
            "los metadatos físicos 3D (número de ruedas, estabilidad, simetría) y la imagen suministrada (si la hay) "
            "para clasificar un vehículo LEGO dentro de la taxonomía oficial.\n\n"
            "Reglas de Confianza:\n"
            "- Si el texto dice una cosa pero la imagen muestra otra (ej. el texto indica 'Coche' pero la imagen "
            "muestra un helicóptero o no tiene ruedas), la confianza ('confidence_score') DEBE ser muy baja (<= 0.50).\n"
            "- Si no se suministra una imagen, reduce la confianza automáticamente (ej. max 0.80) debido a la falta "
            "de confirmación visual, forzando la revisión humana.\n"
            "- La confianza debe ser un float entre 0.0 y 1.0.\n"
            "- Justifica tu respuesta brevemente en 'reasoning_notes'.\n"
        )
        
        user_content = (
            f"Set ID: {set_id}\n"
            f"Origen: {source}\n"
            f"Nombre: {metadata.get('name')}\n"
            f"Temática: {metadata.get('theme')}\n"
            f"Año: {metadata.get('year')}\n"
            f"Descripción: {metadata.get('description')}\n"
            f"Tags del usuario: {metadata.get('tags')}\n"
            f"Cantidad de piezas: {metadata.get('parts_count')}\n"
            f"Metadatos 3D del diseño: {json.dumps(metadata.get('metadata_3d', {}))}\n"
        )
        
        if image_base64:
            user_content += "\n[IMAGEN DE ENTRADA DISPONIBLE Y PROCESADA MULTIMODALMENTE]"
        else:
            user_content += "\n[ATENCIÓN: NO HAY IMAGEN DISPONIBLE. Confianza máxima reducida por falta de visual.]"

        # Construct messages payload for Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        use_vision = "vision" in self.model_name.lower() and image_base64 is not None
        if use_vision:
            messages[1]["images"] = [image_base64]
            model_to_use = self.model_name
        else:
            model_to_use = self.fallback_model
            
        print(f"  [+] Enviando consulta a Ollama usando modelo '{model_to_use}'...")
        
        payload = {
            "model": model_to_use,
            "messages": messages,
            "format": OLLAMA_JSON_SCHEMA,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        }
        
        try:
            res = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
            if res.status_code == 200:
                res_json = res.json()
                raw_response = res_json.get("message", {}).get("content", "").strip()
                
                # Parse structured JSON from response
                parsed_data = json.loads(raw_response)
                
                # Enforce fields and types
                taxonomy_proposal = LegoTaxonomy(
                    Level_1_Entorno=parsed_data["taxonomy_proposal"]["Level_1_Entorno"],
                    Level_2_Proposito=parsed_data["taxonomy_proposal"]["Level_2_Proposito"],
                    Level_3_Clase=parsed_data["taxonomy_proposal"]["Level_3_Clase"],
                    Level_4_Escala=parsed_data["taxonomy_proposal"]["Level_4_Escala"],
                    Level_4_Motorizacion=parsed_data["taxonomy_proposal"]["Level_4_Motorizacion"],
                    Level_4_Licencia=parsed_data["taxonomy_proposal"]["Level_4_Licencia"]
                )
                
                confidence = float(parsed_data.get("confidence_score", 0.5))
                reasoning = parsed_data.get("reasoning_notes", "No notes provided.")
                
                # HITL Logic
                needs_human = confidence <= 0.80
                review_payload = None
                
                if needs_human:
                    score_percent = f"{int(confidence * 100)}%"
                    review_payload = {
                        "columns": ["Set/MOC", "Origen", "Confianza", "Propuesta Nivel 1", "Propuesta Nivel 2", "Propuesta Nivel 3", "Motivo de Baja Confianza", "Acción"],
                        "row_data": {
                            "set_name": metadata.get("name", set_id),
                            "source": source,
                            "score": score_percent,
                            "prop_L1": taxonomy_proposal.Level_1_Entorno,
                            "prop_L2": taxonomy_proposal.Level_2_Proposito,
                            "prop_L3": taxonomy_proposal.Level_3_Clase,
                            "conflict_alert": reasoning
                        },
                        "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala", "motorizacion", "licencia"]
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
            else:
                raise Exception(f"Ollama returned HTTP status {res.status_code}")
                
        except Exception as e:
            print(f"  [-] Fallo en la inferencia con Ollama: {e}. Creando clasificación por defecto para HITL.")
            # Fallback placeholder if LLM connection fails or returns invalid format
            fallback_taxonomy = LegoTaxonomy(
                Level_1_Entorno="Terrestre",
                Level_2_Proposito="Civil/Pasajeros",
                Level_3_Clase="Desconocido",
                Level_4_Escala="Minifig-scale",
                Level_4_Motorizacion="Ruedas Libres",
                Level_4_Licencia="Genérico"
            )
            return ClassificationResult(
                set_id=set_id,
                source=source,
                taxonomy_proposal=fallback_taxonomy,
                confidence_score=0.1,
                reasoning_notes=f"Inference failure: {str(e)}",
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
                        "conflict_alert": f"Fallo de conexión o parseo: {str(e)}"
                    },
                    "editable_fields": ["prop_L1", "prop_L2", "prop_L3", "escala", "motorizacion", "licencia"]
                }
            )


# --- Database Integrator ---

def save_classification_to_db(result: ClassificationResult):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Store classification details in sets table (or insert/replace if Official)
    # We update sets table for BrickLink and OMR, and update/insert for Official sets since they might only be in rb_sets
    tax = result.taxonomy_proposal
    payload_str = json.dumps(result.review_table_payload, ensure_ascii=False) if result.review_table_payload else None
    
    # Check if the set exists in sets table
    cursor.execute("SELECT set_id FROM sets WHERE set_id = ?", (result.set_id,))
    exists = cursor.fetchone()
    
    if exists:
        # Update existing record
        cursor.execute("""
        UPDATE sets
        SET level_1_entorno = ?,
            level_2_proposito = ?,
            level_3_clase = ?,
            level_4_escala = ?,
            level_4_motorizacion = ?,
            level_4_licencia = ?,
            confidence_score = ?,
            reasoning_notes = ?,
            needs_human_review = ?,
            review_table_payload = ?
        WHERE set_id = ?
        """, (
            tax.Level_1_Entorno,
            tax.Level_2_Proposito,
            tax.Level_3_Clase,
            tax.Level_4_Escala,
            tax.Level_4_Motorizacion,
            tax.Level_4_Licencia,
            result.confidence_score,
            result.reasoning_notes,
            1 if result.needs_human_review else 0,
            payload_str,
            result.set_id
        ))
    else:
        # Create a new record in sets for this Official set
        # Get details from rb_sets first
        cursor.execute("SELECT name, year, img_url, num_parts FROM rb_sets WHERE set_num = ?", (result.set_id,))
        rb_info = cursor.fetchone()
        name = rb_info[0] if rb_info else result.set_id
        year = rb_info[1] if rb_info else None
        img_url = rb_info[2] if rb_info else None
        num_parts = rb_info[3] if rb_info else 0
        
        cursor.execute("""
        INSERT INTO sets (
            set_id, name, source, year, image_url, parts_count,
            level_1_entorno, level_2_proposito, level_3_clase, level_4_escala, level_4_motorizacion, level_4_licencia,
            confidence_score, reasoning_notes, needs_human_review, review_table_payload
        )
        VALUES (?, ?, 'Official', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.set_id,
            name,
            year,
            img_url,
            num_parts,
            tax.Level_1_Entorno,
            tax.Level_2_Proposito,
            tax.Level_3_Clase,
            tax.Level_4_Escala,
            tax.Level_4_Motorizacion,
            tax.Level_4_Licencia,
            result.confidence_score,
            result.reasoning_notes,
            1 if result.needs_human_review else 0,
            payload_str
        ))
        
    conn.commit()
    conn.close()
