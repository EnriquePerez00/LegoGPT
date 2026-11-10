import os
import sys
import json
import torch
import numpy as np
from src.io_parser import StudioIOParser, export_parts_to_io
from src.mpd_parser import flatten_mpd
from src.parser import build_pyg_graph, ParsedPart
from src.validator import check_connection_optimized, check_connectivity_and_gravity
from src.homogenize_assembly import save_homogenized_assembly
from generative.llm_pipeline.sequence_planner import schedule_topological_steps

def process_model_to_sequence_graph(
    input_path: str,
    output_pt_dir: str = "data/sequences",
    export_io: bool = True,
    export_homogenized: bool = True,
) -> dict:
    """
    Takes a .io, .mpd, .ldr, .mbx, or .zmbx file and produces:
    1. Ground-normalized and topologically stepped sequence (synthesizing '0 STEP' if flat).
    2. A PyTorch Geometric graph tensor (.pt) with temporal step annotations.
    3. A homogenized port-to-port relational sequence JSON (*_homogenized.json).
    4. A compliant BrickLink Studio 2.0 archive (*.io) with step markers.
    """
    os.makedirs(output_pt_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    lower_path = input_path.lower()
    temp_ldr_path = None

    # 1. Handle .io, .mbx/.zmbx, and .ldr/.mpd
    if lower_path.endswith(".io"):
        io_parser = StudioIOParser(input_path)
        temp_ldr_path = os.path.join("/tmp", f"{base_name}_extracted.ldr")
        io_parser.save_ldr_to_disk(temp_ldr_path)
        parts = flatten_mpd(temp_ldr_path)
    elif lower_path.endswith((".mbx", ".zmbx")):
        from src.mecabricks_converter import parse_zmbx_file
        parts = parse_zmbx_file(input_path)
    else:
        parts = flatten_mpd(input_path)

    if temp_ldr_path and os.path.exists(temp_ldr_path):
        os.remove(temp_ldr_path)

    if not parts:
        return {
            "success": False,
            "error": "No physical parts parsed from model",
            "model": base_name
        }

    # 2. Ground normalization & topological step synthesis if flat
    raw_steps = len({p.step_id for p in parts})
    synthesized_steps = raw_steps <= 1
    parts = schedule_topological_steps(parts)

    total_parts = len(parts)
    unique_steps_count = len({p.step_id for p in parts})

    # 3. Build PyG Graph
    unique_parts = list(set(p.part_id for p in parts))
    unique_colors = list(set(p.color for p in parts))

    pyg_graph = build_pyg_graph(parts, allowed_parts=unique_parts, allowed_colors=unique_colors)

    # Attach step attributes (temporal sequence)
    step_tensor = torch.tensor([p.step_id for p in parts], dtype=torch.long)
    pyg_graph.step_id = step_tensor

    # 4. Physical checks
    is_fully_connected = check_connectivity_and_gravity(parts)

    # 5. Save outputs (.pt, _homogenized.json, .io)
    out_pt_path = os.path.join(output_pt_dir, f"{base_name}.pt")
    torch.save(pyg_graph, out_pt_path)

    out_homogenized_path = None
    if export_homogenized:
        out_homogenized_path = os.path.join(output_pt_dir, f"{base_name}_homogenized.json")
        save_homogenized_assembly(parts, set_name=base_name, output_path=out_homogenized_path)

    out_io_path = None
    if export_io:
        out_io_path = os.path.join(output_pt_dir, f"{base_name}.io")
        export_parts_to_io(parts, out_io_path, model_title=base_name)

    return {
        "success": True,
        "model": base_name,
        "parts_count": total_parts,
        "steps_count": unique_steps_count,
        "synthesized_steps": synthesized_steps,
        "has_discrete_steps": unique_steps_count > 1,
        "edges_count": int(pyg_graph.edge_index.size(1)) if pyg_graph.edge_index is not None else 0,
        "is_fully_connected": is_fully_connected,
        "graph_path": out_pt_path,
        "homogenized_path": out_homogenized_path,
        "io_path": out_io_path,
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
