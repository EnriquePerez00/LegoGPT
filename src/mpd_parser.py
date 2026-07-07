import os
import re
import numpy as np
from src.parser import ParsedPart

def parse_mpd_to_submodels(file_path: str) -> dict[str, list[str]]:
    """
    Reads an LDraw file (either LDR or MPD) and splits it into its constituent
    submodel files.
    
    Returns:
        A dictionary mapping lowercase submodel filenames to their lines.
    """
    submodels = {}
    current_model = None
    current_lines = []
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        
    has_file_headers = any(line.strip().startswith("0 FILE") for line in lines)
    
    if not has_file_headers:
        # It's a single flat LDR model file
        clean_lines = [line.strip() for line in lines if line.strip()]
        submodels["__main__"] = clean_lines
        return submodels
        
    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        tokens = line_str.split()
        if len(tokens) >= 3 and tokens[0] == "0" and tokens[1].upper() == "FILE":
            # Save previous model
            if current_model is not None:
                submodels[current_model.lower()] = current_lines
            current_model = " ".join(tokens[2:])
            current_lines = []
        elif current_model is not None:
            current_lines.append(line_str)
            
    if current_model is not None:
        submodels[current_model.lower()] = current_lines
        
    return submodels

def flatten_mpd(file_path: str) -> list[ParsedPart]:
    """
    Parses an LDraw (.mpd or .ldr) file and flattens its hierarchical structure
    recursively, returning a list of physical parts (leaves) with global 3D transforms
    and resolved step identifiers.
    """
    submodels = parse_mpd_to_submodels(file_path)
    if not submodels:
        return []
        
    # Main model is the first defined model (or "__main__" if it was a flat file)
    main_model_name = "__main__" if "__main__" in submodels else list(submodels.keys())[0]
    
    physical_parts = []
    
    def resolve_model(model_name: str, current_transform: np.ndarray, current_step_offset: int, visited=None):
        if visited is None:
            visited = set()
            
        model_name_lower = model_name.lower()
        if model_name_lower in visited:
            # Recursion guard to prevent infinite loops in cyclic dependencies
            print(f"Advertencia: Bucle de recursión detectado en {model_name}. Deteniendo recursión.")
            return
            
        if model_name_lower not in submodels:
            print(f"Advertencia: Submodelo {model_name} no encontrado en el archivo.")
            return
            
        visited.add(model_name_lower)
        local_step = 0
        lines = submodels[model_name_lower]
        
        for line in lines:
            tokens = line.split()
            if not tokens:
                continue
                
            cmd_type = tokens[0]
            
            if cmd_type == "0":
                # Step separator
                if len(tokens) >= 2 and tokens[1].upper() == "STEP":
                    local_step += 1
            elif cmd_type == "1":
                # Part reference:
                # 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part_name>
                if len(tokens) >= 15:
                    color = int(tokens[1])
                    # Position
                    x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                    # Rotation matrix (LDraw format)
                    a, b, c = float(tokens[5]), float(tokens[6]), float(tokens[7])
                    d, e, f = float(tokens[8]), float(tokens[9]), float(tokens[10])
                    g, h, i = float(tokens[11]), float(tokens[12]), float(tokens[13])
                    part_name = " ".join(tokens[14:]).lower()
                    
                    # Local 4x4 matrix
                    # Note: LDraw's rotation matrix is row-major:
                    # [ [a, b, c], [d, e, f], [g, h, i] ]
                    local_transform = np.array([
                        [a, b, c, x],
                        [d, e, f, y],
                        [g, h, i, z],
                        [0.0, 0.0, 0.0, 1.0]
                    ], dtype=np.float32)
                    
                    # Global transformation matrix
                    global_transform = current_transform @ local_transform
                    global_step = current_step_offset + local_step
                    
                    # Check if referenced part name is a submodel in our dictionary
                    if part_name in submodels:
                        # Recurse submodel, passing a copy of the visited set
                        resolve_model(part_name, global_transform, global_step, visited.copy())
                    else:
                        # It is a physical part (leaf)
                        physical_parts.append(ParsedPart(
                            part_id=part_name,
                            color=color,
                            transform=global_transform,
                            step_id=global_step
                        ))
                        
    # Start recursive flattening from the root main model
    resolve_model(main_model_name, np.eye(4, dtype=np.float32), 0)
    return physical_parts
