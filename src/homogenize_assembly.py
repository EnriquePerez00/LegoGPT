import json
import numpy as np
from src.parser import ParsedPart
from src.validator import get_studs_and_sockets_world, check_connection_optimized

def homogenize_parts_list(parts: list[ParsedPart]) -> list[dict]:
    """
    Parses a list of ParsedParts and returns a homogenized list of dictionaries
    detailing explicit connection points, parent indices, and relative transforms.
    """
    homogenized = []
    
    for idx, part in enumerate(parts):
        part_record = {
            "part_index": idx,
            "part_id": part.part_id,
            "color_code": part.color,
            "step_id": part.step_id,
            "absolute_transform": part.transform.flatten().tolist(),
            "connections": []
        }
        
        # Identify parent connections from previously placed parts (j < idx)
        if idx > 0:
            for j in range(idx):
                parent = parts[j]
                if check_connection_optimized(parent, part):
                    # Found mechanical connection. Determine which anchors align.
                    studs_parent, sockets_parent = get_studs_and_sockets_world(parent)
                    studs_child, sockets_child = get_studs_and_sockets_world(part)
                    
                    parent_anchor_name = "unknown"
                    child_anchor_name = "unknown"
                    
                    # Check parent stud -> child socket
                    found_anchor = False
                    if len(studs_parent) > 0 and len(sockets_child) > 0:
                        for s_idx, sp in enumerate(studs_parent):
                            for c_idx, sc in enumerate(sockets_child):
                                if np.linalg.norm(sp - sc) < 5.0:
                                    parent_anchor_name = f"stud_{s_idx}"
                                    child_anchor_name = f"socket_{c_idx}"
                                    found_anchor = True
                                    break
                            if found_anchor:
                                break
                                
                    # Check parent socket -> child stud
                    if not found_anchor and len(sockets_parent) > 0 and len(studs_child) > 0:
                        for s_idx, sp in enumerate(sockets_parent):
                            for c_idx, sc in enumerate(studs_child):
                                if np.linalg.norm(sp - sc) < 5.0:
                                    parent_anchor_name = f"socket_{s_idx}"
                                    child_anchor_name = f"stud_{c_idx}"
                                    found_anchor = True
                                    break
                            if found_anchor:
                                break
                                
                    # Compute relative transform
                    R_rel = parent.transform[:3, :3].T @ part.transform[:3, :3]
                    t_rel = part.transform[:3, 3] - parent.transform[:3, 3]
                    
                    rel_matrix = np.eye(4, dtype=np.float32)
                    rel_matrix[:3, :3] = R_rel
                    rel_matrix[:3, 3] = t_rel
                    
                    part_record["connections"].append({
                        "parent_part_index": j,
                        "parent_anchor": parent_anchor_name,
                        "child_anchor": child_anchor_name,
                        "relative_transform": rel_matrix.flatten().tolist()
                    })
                    
        homogenized.append(part_record)
        
    return homogenized

def save_homogenized_assembly(parts: list[ParsedPart], set_name: str, output_path: str):
    """Homogenizes and saves a sequence to output_path."""
    sequence = homogenize_parts_list(parts)
    output_data = {
        "set_name": set_name,
        "num_parts": len(parts),
        "sequence": sequence
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
