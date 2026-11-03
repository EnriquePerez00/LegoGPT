import os
import sys
import json
import torch
import numpy as np
from src.io_parser import StudioIOParser
from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph, ParsedPart
from src.validator import check_connection_optimized, check_connectivity_and_gravity

def process_model_to_sequence_graph(input_path: str, output_pt_dir: str = "data/sequences") -> dict:
    """
    Takes a .io, .mpd, or .ldr file with steps and produces:
    1. A PyTorch Geometric graph tensor (.pt) with temporal step annotations.
    2. Summary metadata of parts, steps, and connectivity.
    """
    os.makedirs(output_pt_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    # 1. Handle .io vs .ldr/.mpd
    if input_path.lower().endswith(".io"):
        io_parser = StudioIOParser(input_path)
        step_metrics = io_parser.analyze_steps()
        temp_ldr_path = os.path.join("/tmp", f"{base_name}_extracted.ldr")
        io_parser.save_ldr_to_disk(temp_ldr_path)
        parse_target = temp_ldr_path
    else:
        parse_target = input_path
        step_metrics = {"step_count": 1, "parts_count": 0}
        
    # 2. Flatten hierarchical MPD/LDR with step resolution
    parts = flatten_mpd(parse_target)
    
    if not parts:
        return {
            "success": False,
            "error": "No physical parts parsed from model",
            "model": base_name
        }
        
    total_parts = len(parts)
    max_step = max(p.step_id for p in parts)
    
    # 3. Build Graph
    # Extract part IDs and colors present in the model
    unique_parts = list(set(p.part_id for p in parts))
    unique_colors = list(set(p.color for p in parts))
    
    pyg_graph = build_pyg_graph(parts, allowed_parts=unique_parts, allowed_colors=unique_colors)
    
    # Attach step attributes (temporal sequence)
    step_tensor = torch.tensor([p.step_id for p in parts], dtype=torch.long)
    pyg_graph.step_id = step_tensor
    
    # 4. Physical checks
    is_fully_connected = check_connectivity_and_gravity(parts)
    
    # Save graph tensor
    out_pt_path = os.path.join(output_pt_dir, f"{base_name}.pt")
    torch.save(pyg_graph, out_pt_path)
    
    # Clean up temp file
    if input_path.lower().endswith(".io") and os.path.exists(temp_ldr_path):
        os.remove(temp_ldr_path)
        
    return {
        "success": True,
        "model": base_name,
        "parts_count": total_parts,
        "steps_count": max_step,
        "has_discrete_steps": max_step > 1,
        "edges_count": int(pyg_graph.edge_index.size(1)) if pyg_graph.edge_index is not None else 0,
        "is_fully_connected": is_fully_connected,
        "graph_path": out_pt_path
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_sequence_graph.py <path_to_model.io_or_ldr>")
        # Test on existing sample if present
        sample = "data/mocs/5761-1_Mini_Digger.mpd"
        if os.path.exists(sample):
            print(f"Testing on existing sample {sample}...")
            res = process_model_to_sequence_graph(sample)
            print(json.dumps(res, indent=2))
        sys.exit(0)
        
    input_file = sys.argv[1]
    res = process_model_to_sequence_graph(input_file)
    print(json.dumps(res, indent=2))
