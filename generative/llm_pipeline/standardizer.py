import os
import numpy as np

def parse_mpd_submodels(file_path: str) -> dict[str, list[str]]:
    """
    Parses a Multi-Part Document (.mpd) and returns a dict mapping
    submodel names to their raw LDraw line strings.
    """
    submodels = {}
    current_model = None
    current_lines = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue
                
            tokens = line_str.split()
            if len(tokens) >= 3 and tokens[0] == "0" and tokens[1].upper() == "FILE":
                # Save previous
                if current_model and current_lines:
                    submodels[current_model] = current_lines
                current_model = " ".join(tokens[2:]).lower()
                current_lines = []
            elif current_model:
                current_lines.append(line_str)
                
    if current_model and current_lines:
        submodels[current_model] = current_lines
        
    return submodels

def flatten_ldr_parts(lines: list[str], submodels: dict[str, list[str]], parent_transform: np.ndarray = None) -> list[dict]:
    """
    Recursively flattens LDraw parts, resolving submodel calls into global coordinates.
    """
    if parent_transform is None:
        parent_transform = np.eye(4, dtype=np.float32)
        
    parts = []
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue
            
        cmd = tokens[0]
        if cmd == "1" and len(tokens) >= 15:
            color = int(tokens[1])
            tx, ty, tz = float(tokens[2]), float(tokens[3]), float(tokens[4])
            rot = [float(val) for val in tokens[5:14]]
            part_name = " ".join(tokens[14:]).lower()
            
            # Construct child transform
            child_tf = np.eye(4, dtype=np.float32)
            child_tf[:3, 3] = [tx, ty, tz]
            child_tf[:3, :3] = np.array(rot).reshape(3, 3)
            
            # Combine transforms
            global_tf = parent_transform @ child_tf
            
            if part_name in submodels:
                # Recursively expand submodel call
                child_parts = flatten_ldr_parts(submodels[part_name], submodels, global_tf)
                parts.extend(child_parts)
            else:
                parts.append({
                    "part_id": part_name,
                    "color": color,
                    "transform": global_tf
                })
    return parts

def standardize_and_order_ldr(input_path: str, output_path: str) -> bool:
    """
    Flattens any submodels, sorts parts bottom-up (LDraw Y increases downwards),
    injects 0 STEP commands, and writes a clean sequential LDraw file.
    """
    if not os.path.exists(input_path):
        return False
        
    # Check if file has FILE headers (MPD format)
    submodels = parse_mpd_submodels(input_path)
    
    # If no OMR submodels were found, treat the whole file as a single model
    if not submodels:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = [line.strip() for line in f]
        # Main model name
        main_model = "main"
        submodels = {main_model: raw_lines}
    else:
        # The main model in MPD is typically the first model
        main_model = list(submodels.keys())[0]
        
    # Flatten parts
    flat_parts = flatten_ldr_parts(submodels[main_model], submodels)
    
    if not flat_parts:
        return False
        
    # Plan physical sequence bottom-up using disassembly planning
    from generative.llm_pipeline.sequence_planner import plan_disassembly_sequence
    flat_parts_sorted = plan_disassembly_sequence(flat_parts)
    
    # Group parts into step layers (different Y levels or max 5 parts per step)
    lines = ["0 LegoGPT Standardized Sequence"]
    
    current_y = None
    parts_in_current_step = 0
    for p in flat_parts_sorted:
        y_val = p["transform"][1, 3]
        
        # Trigger new step if height shifts significantly OR if we reach 5 parts in a step
        trigger_step = False
        if current_y is not None and abs(y_val - current_y) > 8.0:
            trigger_step = True
        if parts_in_current_step >= 5:
            trigger_step = True
            
        if trigger_step:
            lines.append("0 STEP")
            parts_in_current_step = 0
            
        current_y = y_val
        parts_in_current_step += 1
        
        # Format part line
        color = p["color"]
        x, y, z = p["transform"][:3, 3]
        rot = p["transform"][:3, :3].flatten()
        rot_str = " ".join(f"{val:.6f}" for val in rot)
        lines.append(f"1 {color} {x:.4f} {y:.4f} {z:.4f} {rot_str} {p['part_id']}")
        
    # Save file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    return True
