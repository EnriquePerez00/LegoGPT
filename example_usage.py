import os
import numpy as np
import torch
from torch_geometric.data import Data, Batch

from src.parser import parse_ldraw_file, build_pyg_graph, ALLOWED_PARTS, ALLOWED_COLORS
from src.model import LegoGNN, get_device
from src.validator import check_collisions, check_connectivity_and_gravity, validate_rules
from src.writer import write_ldraw_file, ParsedPart

def setup_dummy_dataset() -> str:
    """Helper to write a dummy LDraw file for demonstration."""
    dummy_path = "demo_model.ldr"
    content = """0 LegoGPT Demo
0 STEP
1 14 0 0 0 1 0 0 0 1 0 0 0 1 3005.dat
0 STEP
1 14 0 -24 0 1 0 0 0 1 0 0 0 1 3005.dat
"""
    with open(dummy_path, "w") as f:
        f.write(content)
    return dummy_path

def run_data_parsing_demo(file_path: str) -> Data:
    print("\n--- 1. DEMO: Parseo de datos y conversión a Grafo PyG ---")
    # 1.1 Parsear archivo LDraw
    parts = parse_ldraw_file(file_path)
    print(f"Leídas {len(parts)} piezas desde {file_path}:")
    for idx, p in enumerate(parts):
        print(f"  Pieza {idx}: ID={p.part_id}, Color={p.color}, Posición={p.transform[:3, 3].tolist()}, Paso={p.step_id}")
        
    # 1.2 Validaciones físicas básicas iniciales
    collisions = check_collisions(parts)
    is_stable = check_connectivity_and_gravity(parts)
    rules_ok = validate_rules(parts)
    
    print(f"  ¿Colisiones?: {len(collisions) > 0} (Colisiones: {collisions})")
    print(f"  ¿Estable y conectado a la base?: {is_stable}")
    print(f"  ¿Cumple reglas de vocabulario?: {rules_ok}")
    
    # 1.3 Convertir a PyTorch Geometric Graph
    graph_data = build_pyg_graph(parts)
    print(f"  Grafo PyG creado con éxito:")
    print(f"    Nodos (piezas): {graph_data.num_nodes}")
    print(f"    Características de nodos x: {graph_data.x.shape}")
    print(f"    Conexiones edge_index: {graph_data.edge_index.shape}")
    return graph_data

def run_training_demo(graph_data: Data):
    print("\n--- 2. DEMO: Entrenamiento del Modelo (Un Paso de Optimización) ---")
    device = torch.device(get_device())
    print(f"Usando dispositivo: {device.type.upper()}")
    
    # 2.1 Inicializar Modelo
    model = LegoGNN(
        num_part_classes=len(ALLOWED_PARTS),
        num_color_classes=len(ALLOWED_COLORS),
        hidden_dim=32
    ).to(device)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 2.2 Preparar Lote (Batch) de entrenamiento
    # En entrenamiento real, usarás un DataLoader de PyG
    # pyg_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch_data = Batch.from_data_list([graph_data, graph_data]).to(device) # Lote con 2 grafos iguales
    
    # 2.3 Paso Forward
    part_logits, color_logits, transform_preds = model(
        batch_data.x, batch_data.edge_index, batch_data.batch
    )
    
    # 2.4 Objetivos simulados (etiquetas reales)
    # Por cada grafo del batch (batch_size=2), el target es la siguiente pieza a colocar:
    target_part = torch.tensor([3, 4], dtype=torch.long, device=device) # Índices en ALLOWED_PARTS
    target_color = torch.tensor([14, 14], dtype=torch.long, device=device) # Códigos de color
    target_transform = torch.randn(2, 12, device=device) # 3 trans + 9 rot reales
    
    # 2.5 Calcular Pérdida (Loss)
    loss_part = torch.nn.functional.cross_entropy(part_logits, target_part)
    loss_color = torch.nn.functional.cross_entropy(color_logits, target_color)
    loss_trans = torch.nn.functional.mse_loss(transform_preds, target_transform)
    
    loss = loss_part + loss_color + loss_trans
    
    # 2.6 Backpropagation y Actualización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"  Paso de entrenamiento completado. Pérdida total: {loss.item():.4f}")
    
    # 2.7 Guardar checkpoint
    checkpoint_path = "legogpt_checkpoint.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  Checkpoint guardado en: {checkpoint_path}")
    
    # Limpieza
    del loss, loss_part, loss_color, loss_trans
    if device.type == "mps":
        torch.mps.empty_cache()

def run_inference_and_rollback_demo():
    print("\n--- 3. DEMO: Inferencia con Control Físico (Rollback) ---")
    device = torch.device(get_device())
    
    # 3.1 Cargar modelo entrenado
    model = LegoGNN(
        num_part_classes=len(ALLOWED_PARTS),
        num_color_classes=len(ALLOWED_COLORS),
        hidden_dim=32
    ).to(device)
    
    checkpoint_path = "legogpt_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("  Modelo cargado correctamente.")
    model.eval()
    
    # 3.2 Representación del modelo actual (Base: 1 brick 1x1 en el origen)
    transform_base = np.eye(4, dtype=np.float32)
    transform_base[:3, 3] = [0.0, 0.0, 0.0]
    current_parts = [
        ParsedPart(part_id="3005.dat", color=14, transform=transform_base, step_id=0)
    ]
    
    # 3.3 Construir grafo del estado actual
    graph_data = build_pyg_graph(current_parts).to(device)
    batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
    
    # 3.4 Inferencia: Predicción de la siguiente acción
    with torch.no_grad():
        part_logits, color_logits, transform_preds = model(
            graph_data.x, graph_data.edge_index, batch
        )
        
    # Obtener el part_id y color más probables
    pred_part_idx = torch.argmax(part_logits[0]).item()
    pred_part_id = ALLOWED_PARTS[pred_part_idx]
    pred_color = torch.argmax(color_logits[0]).item()
    
    # Para transform: tomamos predicciones y las decodificamos
    # transform_preds[0] es un tensor de 12 elementos (3 trans, 9 rot)
    pred_t_rot_flat = transform_preds[0].cpu().numpy()
    pred_t = pred_t_rot_flat[:3]
    pred_R = pred_t_rot_flat[3:].reshape(3, 3)
    
    # Simulación de dos predicciones del modelo: una correcta y otra con colisión (falla)
    print("  Modelo predijo:")
    print(f"    Siguiente Pieza: {pred_part_id}, Color: {pred_color}")
    
    # Predicción A (VÁLIDA): Colocar encima de la base sin colisionar
    transform_valid = np.eye(4, dtype=np.float32)
    transform_valid[:3, :3] = np.eye(3)
    transform_valid[:3, 3] = [0.0, -24.0, 0.0] # y = -24 (arriba de y=0)
    part_valid = ParsedPart(part_id=pred_part_id, color=pred_color, transform=transform_valid, step_id=1)
    
    # Predicción B (INVÁLIDA - COLISIÓN): Colocar superpuesta en la base
    transform_invalid = np.eye(4, dtype=np.float32)
    transform_invalid[:3, :3] = np.eye(3)
    transform_invalid[:3, 3] = [0.0, -10.0, 0.0] # Solapa volumétricamente con la base de altura 24!
    part_invalid = ParsedPart(part_id=pred_part_id, color=pred_color, transform=transform_invalid, step_id=1)
    
    # 3.5 Evaluación de las predicciones en el Validador
    for name, candidate in [("Predicción A (Válida)", part_valid), ("Predicción B (Con Colisión)", part_invalid)]:
        print(f"\n  Evaluando: {name}...")
        test_assembly = current_parts + [candidate]
        
        # Comprobación de colisiones y estabilidad
        collisions = check_collisions(test_assembly)
        is_stable = check_connectivity_and_gravity(test_assembly)
        
        if len(collisions) > 0:
            print(f"    [RECHAZADA] Colisión detectada en {collisions}. ¡Ejecutando Rollback!")
        elif not is_stable:
            print("    [RECHAZADA] Pieza flotante o inestable. ¡Ejecutando Rollback!")
        else:
            print("    [APROBADA] Físicamente correcta. Añadiendo al modelo.")
            current_parts.append(candidate)
            
    # 3.6 Guardar el resultado en formato LDraw
    output_ldr = "output_generated.ldr"
    write_ldraw_file(current_parts, output_ldr)
    print(f"\n  Modelo resultante exportado a LDraw: {output_ldr}")

if __name__ == "__main__":
    dummy_file = setup_dummy_dataset()
    try:
        # 1. Parseo
        graph = run_data_parsing_demo(dummy_file)
        # 2. Entrenamiento
        run_training_demo(graph)
        # 3. Inferencia y Rollback
        run_inference_and_rollback_demo()
    finally:
        # Limpieza de archivos temporales generados
        for f in [dummy_file, "legogpt_checkpoint.pt", "output_generated.ldr"]:
            if os.path.exists(f):
                os.remove(f)
        print("\nPrueba completada y archivos temporales de demostración eliminados.")
