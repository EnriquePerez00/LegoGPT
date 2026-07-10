import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import time
import json
import os
import torch
import numpy as np
import csv

# Import from existing project
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS, build_pyg_graph
from src.writer import ParsedPart
from src.model import LegoGNN, get_device
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Create models directory
os.makedirs("models", exist_ok=True)

# Load theme mapping on startup
themes = {}
sets = {}

def init_theme_mapping():
    global themes, sets
    themes_path = "scratch/themes.csv"
    sets_path = "scratch/sets.csv"
    if not os.path.exists(themes_path) or not os.path.exists(sets_path):
        print("WARNING: themes.csv or sets.csv not found in scratch/. Theme filtering will be disabled.")
        return
    
    try:
        with open(themes_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                themes[row['id']] = {
                    'name': row['name'],
                    'parent_id': row['parent_id']
                }
        
        with open(sets_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sets[row['set_num']] = {
                    'name': row['name'],
                    'theme_id': row['theme_id'],
                    'num_parts': int(row['num_parts'])
                }
        print("Loaded Rebrickable LEGO themes and sets mapping successfully.")
    except Exception as e:
        print(f"Error loading theme mapping: {e}")

def get_top_theme(theme_id):
    curr = themes.get(theme_id)
    if not curr:
        return "Unknown"
    path = [curr['name']]
    while curr and curr['parent_id']:
        curr = themes.get(curr['parent_id'])
        if curr:
            path.append(curr['name'])
    return path[-1] if path else "Unknown"

init_theme_mapping()

class TrainConfig(BaseModel):
    model_name: str
    theme: str = "All"
    max_pieces: int = 100
    max_epochs: int = 100
    early_stopping_patience: int = 15

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# Global training state
training_task = None
training_active = False

class LegoDataset(torch.utils.data.Dataset):
    def __init__(self, assembly_files: list[str], allowed_parts: list[str]):
        self.examples = []
        self.allowed_parts = allowed_parts
        from src.parser import ParsedPart
        for f_path in assembly_files:
            try:
                with open(f_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                parts_data = data.get("parts", [])
                if len(parts_data) < 2:
                    continue
                parsed_parts = []
                for p in parts_data:
                    t = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
                    parsed_parts.append(ParsedPart(
                        part_id=p["part_id"],
                        color=p["color"],
                        transform=t,
                        step_id=p["step_id"]
                    ))
                # Predict next piece incrementally
                for i in range(1, len(parsed_parts)):
                    self.examples.append((parsed_parts[:i], parsed_parts[i]))
            except Exception:
                pass

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        sub_parts, target_part = self.examples[idx]
        from src.parser import build_pyg_graph
        graph_data = build_pyg_graph(sub_parts, self.allowed_parts)
        
        try:
            part_idx = self.allowed_parts.index(target_part.part_id)
        except ValueError:
            part_idx = 0
            
        color_idx = target_part.color
        translation = target_part.transform[:3, 3].tolist()
        rotation = target_part.transform[:3, :3].flatten().tolist()
        target_transform = torch.tensor(translation + rotation, dtype=torch.float32)
        
        return graph_data, torch.tensor(part_idx, dtype=torch.long), torch.tensor(color_idx, dtype=torch.long), target_transform

def pyg_collate(batch):
    from torch_geometric.data import Batch
    graphs, parts, colors, transforms = zip(*batch)
    return Batch.from_data_list(graphs), torch.stack(parts), torch.stack(colors), torch.stack(transforms)

async def training_loop(config: TrainConfig):
    global training_active
    epoch = 0
    device = torch.device(get_device())
    
    import glob
    # Filter to use synthetic models for training the MVP sandbox
    assembly_files = glob.glob("data/processed/9999*_assembly.json")
    if not assembly_files:
        assembly_files = glob.glob("data/processed/*_assembly.json")
        
    allowed_parts = ALLOWED_PARTS
    num_parts = len(allowed_parts)
    
    await manager.broadcast(json.dumps({
        "type": "info",
        "message": f"Iniciando entrenamiento real con {len(assembly_files)} sets. Vocabulario de piezas: {num_parts}."
    }))
    
    # Initialize real dataset and dataloader
    dataset = LegoDataset(assembly_files, allowed_parts)
    if len(dataset) == 0:
        await manager.broadcast(json.dumps({
            "type": "info",
            "message": "Error: No hay datos de entrenamiento procesados. Canceling."
        }))
        training_active = False
        return
        
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=pyg_collate)
    
    model = LegoGNN(
        num_part_classes=num_parts,
        num_color_classes=len(ALLOWED_COLORS),
        hidden_dim=32
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    while training_active:
        epoch += 1
        model.train()
        epoch_loss = 0.0
        batches_count = 0
        
        # Give control back to event loop for a brief moment
        await asyncio.sleep(0.01)
        
        for batch_graph, batch_parts, batch_colors, batch_transforms in loader:
            if not training_active:
                break
                
            batch_graph = batch_graph.to(device)
            batch_parts = batch_parts.to(device)
            batch_colors = batch_colors.to(device)
            batch_transforms = batch_transforms.to(device)
            
            optimizer.zero_grad()
            
            part_logits, color_logits, transform_preds = model(
                batch_graph.x, batch_graph.edge_index, batch_graph.batch
            )
            
            loss_part = torch.nn.functional.cross_entropy(part_logits, batch_parts)
            loss_color = torch.nn.functional.cross_entropy(color_logits, batch_colors)
            loss_trans = torch.nn.functional.mse_loss(transform_preds, batch_transforms)
            
            loss = loss_part + loss_color + loss_trans
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batches_count += 1
            
        if not training_active:
            break
            
        avg_loss = epoch_loss / max(1, batches_count)
        loss_val = round(float(avg_loss), 4)
        
        log_msg = json.dumps({
            "type": "log",
            "epoch": epoch,
            "loss": loss_val
        })
        await manager.broadcast(log_msg)
        
        # Save checkpoints
        if epoch % 5 == 0 or epoch >= config.max_epochs:
            checkpoint_path = f"models/{config.model_name}.pt"
            metadata_path = f"models/{config.model_name}.json"
            torch.save(model.state_dict(), checkpoint_path)
            with open(metadata_path, "w") as f:
                json.dump({
                    "num_parts": num_parts,
                    "allowed_parts": allowed_parts,
                    "theme": config.theme,
                    "max_pieces": config.max_pieces,
                    "epoch": epoch
                }, f)
            
        # Early Stopping Logic
        if config.early_stopping_patience > 0:

            if loss_val < best_loss - 0.001:
                best_loss = loss_val
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= config.early_stopping_patience:
                training_active = False
                await manager.broadcast(json.dumps({
                    "type": "info", 
                    "message": f"Entrenamiento detenido por Early Stopping en época {epoch} (paciencia = {config.early_stopping_patience})."
                }))
                break
                
        if epoch >= config.max_epochs:
            training_active = False
            await manager.broadcast(json.dumps({"type": "info", "message": f"Training finished at {config.max_epochs} epochs."}))
            break


@app.post("/train/start")
async def start_training(config: TrainConfig):
    global training_task, training_active
    if training_active:
        return {"status": "already running"}
    
    training_active = True
    training_task = asyncio.create_task(training_loop(config))
    return {"status": "started"}

@app.post("/train/stop")
async def stop_training():
    global training_active
    training_active = False
    return {"status": "stopped"}

class GenerateRequest(BaseModel):
    model_name: str

@app.post("/generate")
async def generate_structure(req: GenerateRequest):
    try:
        from src.generator import LegoGenerator
        
        device = torch.device(get_device())
        
        # Load the requested model metadata and checkpoint
        num_parts = len(ALLOWED_PARTS)
        current_allowed_parts = ALLOWED_PARTS
        
        checkpoint_path = f"models/{req.model_name}.pt"
        metadata_path = f"models/{req.model_name}.json"
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                num_parts = meta.get("num_parts", num_parts)
                current_allowed_parts = meta.get("allowed_parts", current_allowed_parts)
        else:
            return {"status": "error", "message": f"Modelo '{req.model_name}' no encontrado. Asegúrate de haber completado su entrenamiento."}
            
        model = LegoGNN(num_part_classes=num_parts, num_color_classes=len(ALLOWED_COLORS), hidden_dim=32).to(device)
        
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        else:
            return {"status": "error", "message": f"Archivo de checkpoint {checkpoint_path} no encontrado."}
        model.eval()
        
        # Determine target number of pieces from metadata with fallback
        target_num_pieces = 8
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    meta = json.load(f)
                    target_num_pieces = int(meta.get("max_pieces", 8))
            except Exception:
                pass

        generator = LegoGenerator(model, current_allowed_parts, ALLOWED_COLORS, device=device)
        generated_parts = generator.generate_beam_search(
            target_num_pieces=target_num_pieces,
            beam_width=3,
            max_candidates=5
        )
        
        out_parts = []
        for p in generated_parts:
            out_parts.append({
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist()
            })
            
        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/config/parts")
async def get_parts_config():
    return {"parts": ALLOWED_PARTS}

@app.get("/local-sets")
async def get_local_sets():
    import glob
    files = glob.glob("data/omr_raw/*.mpd") + glob.glob("data/omr_raw/*.ldr")
    return {"sets": sorted([os.path.basename(f) for f in files])}

@app.get("/themes")
async def get_themes():
    import glob
    import re
    assembly_files = glob.glob("data/processed/*_assembly.json")
    unique_themes = set()
    for f_path in assembly_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                set_name = data.get("set_name", "")
                match = re.match(r'^([0-9]+-[0-9]+)', set_name)
                set_num = match.group(1) if match else set_name
                set_info = sets.get(set_num) or sets.get(f"{set_num}-1")
                if set_info:
                    unique_themes.add(get_top_theme(set_info['theme_id']))
        except Exception:
            pass
    return {"themes": sorted(list(unique_themes))}

@app.get("/models")
async def get_models():
    import glob
    metadata_files = glob.glob("models/*.json")
    model_list = []
    for f_path in metadata_files:
        try:
            with open(f_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                model_name = os.path.splitext(os.path.basename(f_path))[0]
                model_list.append({
                    "name": model_name,
                    "theme": meta.get("theme", "All"),
                    "max_pieces": meta.get("max_pieces", 100),
                    "num_parts": meta.get("num_parts", 0),
                    "epoch": meta.get("epoch", 0),
                    "allowed_parts": meta.get("allowed_parts", [])
                })
        except Exception:
            pass
    return {"models": model_list}

@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    try:
        pt_path = f"models/{model_name}.pt"
        json_path = f"models/{model_name}.json"
        deleted = False
        if os.path.exists(pt_path):
            os.remove(pt_path)
            deleted = True
        if os.path.exists(json_path):
            os.remove(json_path)
            deleted = True
            
        if deleted:
            return {"status": "success", "message": f"Modelo {model_name} eliminado."}
        else:
            return {"status": "error", "message": "Modelo no encontrado."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class LocalFileRequest(BaseModel):
    filename: str

@app.post("/upload-local")
async def upload_local(req: LocalFileRequest):
    try:
        file_path = os.path.join("data/omr_raw", req.filename)
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"Archivo {req.filename} no encontrado."}
            
        from src.mpd_parser import flatten_mpd
        parts = flatten_mpd(file_path)
            
        out_parts = []
        for idx, p in enumerate(parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/upload-mpd")
async def upload_mpd(file: UploadFile = File(...)):
    try:
        temp_path = "temp_uploaded.mpd"
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        from src.mpd_parser import flatten_mpd
        parts = flatten_mpd(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        out_parts = []
        for idx, p in enumerate(parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PromptGenerateRequest(BaseModel):
    prompt: str
    num_pieces: int = 12

@app.post("/generate-prompt")
async def generate_prompt(req: PromptGenerateRequest):
    try:
        # Run generate_build.py CLI as a subprocess to keep generation clean
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean prompt for file name
        safe_name = req.prompt.lower().replace(" ", "_")
        
        cmd = [
            "./legogpt_env/bin/python",
            "generate_build.py",
            "--prompt", req.prompt,
            "--output_dir", output_dir,
            "--num_pieces", str(req.num_pieces)
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            return {"status": "error", "message": f"CLI execution failed: {res.stderr}"}
            
        # Parse the generated LDraw file to return the parts to the frontend
        ldr_path = os.path.join(output_dir, f"{safe_name}.ldr")
        if not os.path.exists(ldr_path):
            return {"status": "error", "message": "LDraw file was not created."}
            
        from src.parser import parse_ldraw_file
        parts = parse_ldraw_file(ldr_path)
        
        out_parts = []
        for idx, p in enumerate(parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        png_path = f"/outputs/{safe_name}.png"
        has_render = os.path.exists(os.path.join(output_dir, f"{safe_name}.png"))
        
        return {
            "status": "success", 
            "parts": out_parts,
            "render_url": png_path if has_render else None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/voxelize")
async def voxelize_file(file: UploadFile = File(...)):
    try:
        os.makedirs("scratch", exist_ok=True)
        temp_path = os.path.join("scratch", f"uploaded_{file.filename}")
        
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        from generative.voxelizer import voxelize_mesh_to_lego
        parts = voxelize_mesh_to_lego(temp_path)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        out_parts = []
        for idx, p in enumerate(parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/generate-mosaic")
async def generate_mosaic(file: UploadFile = File(...), size: int = 32, use_dithering: bool = True):
    try:
        os.makedirs("scratch", exist_ok=True)
        temp_path = os.path.join("scratch", f"mosaic_uploaded_{file.filename}")
        
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)
            
        from src.mosaic_generator import generate_mosaic_from_image
        parts = generate_mosaic_from_image(temp_path, size=size, use_dithering=use_dithering)
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        out_parts = []
        for idx, p in enumerate(parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p.part_id,
                "color": p.color,
                "transform": p.transform.flatten().tolist(),
                "step_id": p.step_id
            })
            
        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

class PartsListRequest(BaseModel):
    parts: list[dict]

class RenderRequest(BaseModel):
    parts: list[dict]
    name: str = "temp_render"

@app.post("/validate-graph")
async def validate_graph(req: PartsListRequest):
    try:
        from src.parser import ParsedPart
        from src.validator import check_connectivity_and_gravity, check_collisions
        from src.vehicle_rules import evaluate_vehicle_topology
        import numpy as np

        parsed_parts = []
        for p in req.parts:
            tf = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
            parsed_parts.append(ParsedPart(
                part_id=p["part_id"],
                color=p["color"],
                transform=tf,
                step_id=p.get("step_id", 0)
            ))

        is_stable = check_connectivity_and_gravity(parsed_parts)
        collisions = check_collisions(parsed_parts)
        topology = evaluate_vehicle_topology(parsed_parts)

        return {
            "status": "success",
            "is_stable": is_stable,
            "collisions_count": len(collisions),
            "topology": topology
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/sequence-assembly")
async def sequence_assembly(req: PartsListRequest):
    try:
        from generative.llm_pipeline.sequence_planner import plan_disassembly_sequence
        import numpy as np

        parts_dicts = []
        for idx, p in enumerate(req.parts):
            tf = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
            parts_dicts.append({
                "part_id": p["part_id"],
                "color": p["color"],
                "transform": tf,
                "step_id": p.get("step_id", 0),
                "original_index": idx
            })

        ordered_parts = plan_disassembly_sequence(parts_dicts)

        out_parts = []
        for idx, p in enumerate(ordered_parts):
            out_parts.append({
                "sequence_index": idx,
                "part_id": p["part_id"],
                "color": p["color"],
                "transform": p["transform"].flatten().tolist(),
                "step_id": p["step_id"]
            })

        return {"status": "success", "parts": out_parts}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/render-eevee")
async def render_eevee(req: RenderRequest):
    try:
        import subprocess
        import numpy as np
        from src.parser import ParsedPart

        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        parsed_parts = []
        for p in req.parts:
            tf = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
            parsed_parts.append(ParsedPart(
                part_id=p["part_id"],
                color=p["color"],
                transform=tf,
                step_id=p.get("step_id", 0)
            ))

        parts_sorted = sorted(parsed_parts, key=lambda p: p.transform[1, 3], reverse=True)
        ldr_path = os.path.join(output_dir, f"{req.name}.ldr")
        
        lines = ["0 LegoGPT Render LDraw", f"0 Name: {req.name}"]
        for p in parts_sorted:
            color = p.color
            x, y, z = p.transform[:3, 3]
            rot = p.transform[:3, :3].flatten()
            rot_str = " ".join(f"{val:.6f}" for val in rot)
            lines.append(f"1 {color} {x:.4f} {y:.4f} {z:.4f} {rot_str} {p.part_id}")
            
        with open(ldr_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
        png_path = os.path.join(output_dir, f"{req.name}.png")

        if os.path.exists(blender_path):
            cmd = [
                blender_path,
                "-b",
                "-P", "scratch/render_helper.py",
                "--",
                "--filepath", ldr_path,
                "--output", png_path
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                return {"status": "error", "message": f"Blender execution failed: {res.stderr}"}
            has_render = True
        else:
            has_render = False

        png_url = f"/outputs/{req.name}.png"
        return {
            "status": "success" if has_render else "warning",
            "render_url": png_url if has_render else None,
            "message": "Render completed." if has_render else "Blender not found at expected path. Render skipped."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

class ClassifyRequest(BaseModel):
    set_id: str
    level_1_entorno: Optional[str] = None
    level_2_proposito: Optional[str] = None
    level_3_clase: Optional[str] = None
    level_4_escala: Optional[str] = None
    is_animal: Optional[bool] = False
    animal_level_1_habitat: Optional[str] = None
    animal_level_2_categoria: Optional[str] = None
    animal_level_3_especie: Optional[str] = None
    animal_level_4_estilo: Optional[str] = None

@app.get("/api/vlm/sets")
async def get_vlm_sets():
    try:
        import sqlite3
        conn = sqlite3.connect("data/catalog/models_catalog.db")
        cursor = conn.cursor()
        
        # Select all sets from BrickLink where parts_count > 0, including VLM tracking columns
        cursor.execute("""
            SELECT 
                set_id, name, theme, parts_count, source_url, image_url, 
                level_1_entorno, level_2_proposito, level_3_clase, level_4_escala,
                confidence_score, reasoning_notes, subassemblies_count,
                vlm_level_1_entorno, vlm_level_2_proposito, vlm_level_3_clase, vlm_level_4_escala,
                classification_status,
                animal_level_1_habitat, animal_level_2_categoria, animal_level_3_especie, animal_level_4_estilo,
                animal_confidence_score, animal_reasoning_notes
            FROM sets 
            WHERE source = 'BrickLink'
            ORDER BY set_id ASC
        """)
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for r in rows:
            results.append({
                "set_id": r[0],
                "name": r[1],
                "theme": r[2],
                "parts_count": r[3],
                "source_url": r[4],
                "image_url": r[5],
                "level_1_entorno": r[6] or "",
                "level_2_proposito": r[7] or "",
                "level_3_clase": r[8] or "",
                "level_4_escala": r[9] or "",
                "confidence_score": r[10] or 0.0,
                "reasoning_notes": r[11] or "",
                "subassemblies_count": r[12] or 0,
                "vlm_level_1_entorno": r[13] or "",
                "vlm_level_2_proposito": r[14] or "",
                "vlm_level_3_clase": r[15] or "",
                "vlm_level_4_escala": r[16] or "",
                "classification_status": r[17] or "vlm_raw",
                "animal_level_1_habitat": r[18] or "",
                "animal_level_2_categoria": r[19] or "",
                "animal_level_3_especie": r[20] or "",
                "animal_level_4_estilo": r[21] or "",
                "animal_confidence_score": r[22] or 0.0,
                "animal_reasoning_notes": r[23] or ""
            })
        return {"status": "success", "sets": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/vlm/sets/classify")
async def classify_set(req: ClassifyRequest):
    try:
        import sqlite3
        import shutil
        
        # 1. Perform database backup
        db_src = "data/catalog/models_catalog.db"
        db_dst = "data/catalog/clasificacion VLM 08.07.2026.db"
        try:
            shutil.copy(db_src, db_dst)
            print(f"Database backup created successfully at: {db_dst}")
        except Exception as backup_err:
            print(f"WARNING: Database backup failed: {backup_err}")
            
        # 2. Update set classification details
        conn = sqlite3.connect(db_src)
        cursor = conn.cursor()
        
        if req.is_animal:
            cursor.execute("""
                UPDATE sets 
                SET animal_level_1_habitat = ?, 
                    animal_level_2_categoria = ?, 
                    animal_level_3_especie = ?, 
                    animal_level_4_estilo = ?, 
                    level_1_entorno = 'Otros',
                    level_2_proposito = 'Otros',
                    level_3_clase = 'Otros',
                    level_4_escala = 'Otros',
                    needs_human_review = 0,
                    classification_status = 'human_verified'
                WHERE set_id = ?
            """, (
                req.animal_level_1_habitat,
                req.animal_level_2_categoria,
                req.animal_level_3_especie,
                req.animal_level_4_estilo,
                req.set_id
            ))
        else:
            cursor.execute("""
                UPDATE sets 
                SET level_1_entorno = ?, 
                    level_2_proposito = ?, 
                    level_3_clase = ?, 
                    level_4_escala = ?, 
                    animal_level_1_habitat = NULL,
                    animal_level_2_categoria = NULL,
                    animal_level_3_especie = NULL,
                    animal_level_4_estilo = NULL,
                    needs_human_review = 0,
                    classification_status = 'human_verified'
                WHERE set_id = ?
            """, (
                req.level_1_entorno,
                req.level_2_proposito,
                req.level_3_clase,
                req.level_4_escala,
                req.set_id
            ))
            
        conn.commit()
        conn.close()
        
        # 3. Regenerate HTML reports automatically
        import subprocess
        try:
            subprocess.run(["python3", "scripts/generate_classification_report.py"], check=True)
            subprocess.run(["python3", "scripts/generate_discarded_report.py"], check=True)
            subprocess.run(["python3", "scripts/generate_animal_report.py"], check=True)
            print("HTML reports successfully regenerated after user classification update.")
        except Exception as report_err:
            print(f"WARNING: Report regeneration failed: {report_err}")
            
        return {"status": "success", "message": f"Set {req.set_id} successfully classified."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ---------------------------------------------------------------------------
# OpenWiki Tool — langchain-ai/openwiki
# Endpoint REST que expone Wikipedia como herramienta de busqueda.
# Referencia: https://github.com/langchain-ai/openwiki
# ---------------------------------------------------------------------------

class WikiSearchRequest(BaseModel):
    q: str
    lang: str = "es"
    top_k: int = 2
    max_chars: int = 1500


@app.get("/wiki/search")
async def wiki_search(q: str, lang: str = "es", top_k: int = 2, max_chars: int = 1500):
    """
    Busca en Wikipedia usando OpenWiki (langchain-ai/openwiki).

    Args:
        q: Termino a buscar en Wikipedia.
        lang: Codigo ISO del idioma (por defecto "es").
        top_k: Numero maximo de resultados (por defecto 2).
        max_chars: Longitud maxima del resultado en caracteres (por defecto 1500).

    Returns:
        JSON con el resultado de Wikipedia o mensaje de error.

    Example:
        GET /wiki/search?q=LEGO+Technic&lang=es&top_k=2
    """
    try:
        from src.tools.openwiki_tool import OpenWikiTool
        tool = OpenWikiTool(lang=lang, top_k_results=top_k, doc_content_chars_max=max_chars)
        result = tool.search(q)
        return {
            "status": "success",
            "query": q,
            "lang": lang,
            "results": result,
        }
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Error inesperado: {exc}"}


@app.post("/wiki/search")
async def wiki_search_post(req: WikiSearchRequest):
    """
    Busca en Wikipedia usando OpenWiki (POST, para queries largas).

    Body JSON:
        { "q": "LEGO Technic", "lang": "es", "top_k": 2, "max_chars": 1500 }
    """
    try:
        from src.tools.openwiki_tool import OpenWikiTool
        tool = OpenWikiTool(
            lang=req.lang,
            top_k_results=req.top_k,
            doc_content_chars_max=req.max_chars,
        )
        result = tool.search(req.q)
        return {
            "status": "success",
            "query": req.q,
            "lang": req.lang,
            "results": result,
        }
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    except RuntimeError as exc:
        return {"status": "error", "message": str(exc)}
    except Exception as exc:
        return {"status": "error", "message": f"Error inesperado: {exc}"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

import subprocess
os.makedirs("outputs", exist_ok=True)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
app.mount("/", StaticFiles(directory="public", html=True), name="public")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)

