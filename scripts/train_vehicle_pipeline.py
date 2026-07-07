import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import psutil
from src.model import LegoGNN
from src.parser import ParsedPart
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS
from src.vehicle_rules import get_vehicle_rl_reward

def check_resources():
    """Ensures RAM usage doesn't exceed 80% of system memory."""
    mem = psutil.virtual_memory()
    if mem.percent > 80.0:
        print(f"Advertencia: El uso de memoria es alto ({mem.percent}%). Liberando caché.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def get_device() -> str:
    """Prioritizes Apple Silicon MPS for training."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def build_pyg_data_for_vocab(parts: list[ParsedPart], vocab: list[str], colors: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Builds node features (x) and edge indices (edge_index) using custom vocab and colors.
    """
    from src.validator import check_connection_optimized
    
    num_nodes = len(parts)
    node_features = []
    
    for part in parts:
        # Part ID One-hot encoding
        part_one_hot = [1.0 if part.part_id == p else 0.0 for p in vocab]
        if sum(part_one_hot) == 0.0:
            # Fallback to standard 1x1 brick if not in vocab
            part_one_hot[0] = 1.0
            
        # Color One-hot encoding
        color_one_hot = [1.0 if part.color == c else 0.0 for c in colors]
        if sum(color_one_hot) == 0.0:
            color_one_hot[0] = 1.0
            
        # Translation vector
        translation = part.transform[:3, 3].tolist()
        # Rotation vector
        rotation = part.transform[:3, :3].flatten().tolist()
        
        feature_vector = part_one_hot + color_one_hot + translation + rotation
        node_features.append(feature_vector)
        
    x = torch.tensor(node_features, dtype=torch.float32)
    
    edge_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                continue
            if check_connection_optimized(parts[i], parts[j]):
                edge_list.append([i, j])
                
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        
    return x, edge_index

def prepare_dataset(assembly_files: list[str]) -> list[dict]:
    """Loads assemblies and formats them as sequential training graphs."""
    dataset = []
    for filepath in assembly_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        parts = []
        for p_json in data["parts"]:
            transform = np.array(p_json["transform"], dtype=np.float32).reshape(4, 4)
            parts.append(ParsedPart(
                part_id=p_json["part_id"],
                color=p_json["color"],
                transform=transform,
                step_id=p_json["step_id"]
            ))
            
        # Add sequential steps
        for k in range(1, len(parts)):
            input_state = parts[:k]
            target_part = parts[k]
            dataset.append({
                "input_state": input_state,
                "target_part": target_part,
                "set_name": data["set_name"]
            })
            
    return dataset

def train_supervised_epoch(model, optimizer, dataset, device, stage=1) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    
    # Process in chunks to prevent unified memory spikes on macOS
    check_resources()
    
    for sample in dataset:
        input_state = sample["input_state"]
        target_part = sample["target_part"]
        
        # Build graph using custom vehicle vocabulary
        x, edge_index = build_pyg_data_for_vocab(input_state, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS)
        x, edge_index = x.to(device), edge_index.to(device)
        batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        
        optimizer.zero_grad()
        part_out, color_logits, transform_preds = model(x, edge_index, batch)
        
        # Supervised target labels
        try:
            part_target = VEHICLE_ALLOWED_PARTS.index(target_part.part_id)
        except ValueError:
            part_target = 0
            
        if stage == 1:
            # Stage 1: Functional chassis. Mask color target to index 0 to ignore aesthetics.
            color_target = 0
        else:
            try:
                color_target = VEHICLE_ALLOWED_COLORS.index(target_part.color)
            except ValueError:
                color_target = 0
                
        transform_target = torch.tensor(target_part.transform[:3].flatten(), dtype=torch.float32, device=device)
        
        # Compute loss
        loss_part = F.cross_entropy(part_out, torch.tensor([part_target], device=device))
        loss_color = F.cross_entropy(color_logits, torch.tensor([color_target], device=device))
        loss_tf = F.mse_loss(transform_preds[0], transform_target)
        
        loss = loss_part + 0.1 * loss_color + 0.01 * loss_tf
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        count += 1
        
    return total_loss / max(1, count)

def main():
    device = torch.device(get_device())
    print(f"Usando dispositivo acelerado: {device}")
    
    # Locate all processed vehicle assemblies
    assembly_files = glob.glob("data/processed_vehicles/*_assembly.json")
    print(f"Encontrados {len(assembly_files)} archivos de vehículos para el entrenamiento.")
    
    if not assembly_files:
        print("No hay datos. Abortando.")
        return
        
    # Group assemblies into themes (Stage 2 specialization)
    city_files = []
    sw_files = []
    
    city_kws = ["city", "town", "racer", "truck", "buggy", "car", "police", "fire", "formula1", "kart", "roadster", "jeep", "auto", "tractor"]
    sw_kws = ["star", "wars", "fighter", "space", "rover", "ship", "grey", "gray", "31066"]
    
    for f in assembly_files:
        name_lower = os.path.basename(f).lower()
        if any(kw in name_lower for kw in city_kws):
            city_files.append(f)
        elif any(kw in name_lower for kw in sw_kws):
            sw_files.append(f)
        else:
            # Distribute general builds to both to enrich the subsets
            city_files.append(f)
            sw_files.append(f)
            
    print(f"  Subconjunto City: {len(city_files)} modelos")
    print(f"  Subconjunto Star Wars: {len(sw_files)} modelos")
    
    # -------------------------------------------------------------
    # STAGE 1: Train Base Chassis Model
    # -------------------------------------------------------------
    print("\n--- INICIANDO FASE 1: Entrenamiento del Chasis Funcional Base ---")
    base_dataset = prepare_dataset(assembly_files)
    print(f"Total de muestras de secuencia de chasis: {len(base_dataset)}")
    
    # Initialize GNN with allowed_parts=None to avoid hardcoded HierarchicalSoftmax error
    model = LegoGNN(
        num_part_classes=len(VEHICLE_ALLOWED_PARTS),
        num_color_classes=len(VEHICLE_ALLOWED_COLORS),
        hidden_dim=64,
        allowed_parts=None
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Supervised base training
    for epoch in range(15):
        loss = train_supervised_epoch(model, optimizer, base_dataset, device, stage=1)
        print(f"  Epoch {epoch+1}/15 | Pérdida Supervisada: {loss:.4f}")
        
    # RL Fine-tuning on base chassis to reinforce symmetry, wheels, and stability
    print("\n  Ajuste Fino de Refuerzo (RL) para el Chasis Base...")
    from src.rl_train import train_rl_step
    rl_optimizer = optim.Adam(model.parameters(), lr=0.001)
    for step in range(50):
        rl_loss = train_rl_step(model, rl_optimizer, device=device, mode="vehicle")
        if (step+1) % 10 == 0:
            print(f"    Paso de RL {step+1}/50 | Pérdida de Política: {rl_loss:.4f}")
            
    # Save base chassis checkpoint
    os.makedirs("models", exist_ok=True)
    base_path = "models/vehicle_base_chassis.pt"
    torch.save(model.state_dict(), base_path)
    print(f"Modelo base de chasis guardado en: {base_path}")
    
    # -------------------------------------------------------------
    # STAGE 2: Fine-Tuning Aesthetic Bodyworks
    # -------------------------------------------------------------
    # A. Lego City Fine-Tuning
    if city_files:
        print("\n--- INICIANDO FASE 2: Fine-Tuning Estético - LEGO CITY ---")
        city_dataset = prepare_dataset(city_files)
        city_model = LegoGNN(
            num_part_classes=len(VEHICLE_ALLOWED_PARTS),
            num_color_classes=len(VEHICLE_ALLOWED_COLORS),
            hidden_dim=64,
            allowed_parts=None
        ).to(device)
        city_model.load_state_dict(torch.load(base_path))
        city_optimizer = optim.Adam(city_model.parameters(), lr=0.002)
        
        for epoch in range(10):
            loss = train_supervised_epoch(city_model, city_optimizer, city_dataset, device, stage=2)
            print(f"  Epoch {epoch+1}/10 | Pérdida City: {loss:.4f}")
            
        city_path = "models/vehicle_fine_city.pt"
        torch.save(city_model.state_dict(), city_path)
        print(f"Modelo especializado LEGO City guardado en: {city_path}")
        
    # B. Lego Star Wars / Space Fine-Tuning
    if sw_files:
        print("\n--- INICIANDO FASE 2: Fine-Tuning Estético - LEGO STAR WARS ---")
        sw_dataset = prepare_dataset(sw_files)
        sw_model = LegoGNN(
            num_part_classes=len(VEHICLE_ALLOWED_PARTS),
            num_color_classes=len(VEHICLE_ALLOWED_COLORS),
            hidden_dim=64,
            allowed_parts=None
        ).to(device)
        sw_model.load_state_dict(torch.load(base_path))
        sw_optimizer = optim.Adam(sw_model.parameters(), lr=0.002)
        
        for epoch in range(10):
            loss = train_supervised_epoch(sw_model, sw_optimizer, sw_dataset, device, stage=2)
            print(f"  Epoch {epoch+1}/10 | Pérdida Star Wars: {loss:.4f}")
            
        sw_path = "models/vehicle_fine_starwars.pt"
        torch.save(sw_model.state_dict(), sw_path)
        print(f"Modelo especializado LEGO Star Wars guardado en: {sw_path}")

if __name__ == "__main__":
    main()
