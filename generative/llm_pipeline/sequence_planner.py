import numpy as np
import os
from src.validator import get_part_dimensions

def get_part_corners_world(part_id: str, transform: np.ndarray) -> np.ndarray:
    """Computes the 8 corners of the part's OBB in world space."""
    dim = get_part_dimensions(part_id)
    w, h, d = dim[0], dim[1], dim[2]
    
    # 8 corners in local space (with bottom-center at local origin)
    local_corners = np.array([
        [-w/2, 0,    -d/2, 1],
        [ w/2, 0,    -d/2, 1],
        [-w/2, 0,     d/2, 1],
        [ w/2, 0,     d/2, 1],
        [-w/2, -h,   -d/2, 1],
        [ w/2, -h,   -d/2, 1],
        [-w/2, -h,    d/2, 1],
        [ w/2, -h,    d/2, 1]
    ], dtype=np.float32)
    
    # Project to world space
    world_corners = (transform @ local_corners.T).T[:, :3]
    return world_corners

def is_blocked_upwards_cached(bounds_a: dict, bounds_b: dict) -> bool:
    """
    Returns True if part B is physically above part A and overlaps its horizontal projection,
    using pre-calculated bounds.
    """
    # Check if 2D projections overlap (with 0.1 LDU tolerance)
    overlap_x = (bounds_a["min_x"] - 0.1 <= bounds_b["max_x"]) and (bounds_a["max_x"] + 0.1 >= bounds_b["min_x"])
    overlap_z = (bounds_a["min_z"] - 0.1 <= bounds_b["max_z"]) and (bounds_a["max_z"] + 0.1 >= bounds_b["min_z"])
    
    if not (overlap_x and overlap_z):
        return False
        
    # Check vertical alignment (In LDraw, Y decreases upwards, so B is physically higher than A if Y_b < Y_a)
    if bounds_b["min_y"] < bounds_a["max_y"] - 0.1:
        return True
        
    return False

def plan_disassembly_sequence(parts: list[dict]) -> list[dict]:
    """
    Plans a physical disassembly sequence using a Directed Acyclic Graph (DAG) 
    and dependency resolution for ultra-fast O(N^2) execution.
    """
    if not parts:
        return []
        
    # 1. Pre-calculate bounds for all parts
    part_bounds = {}
    parts_map = {}
    for p in parts:
        p_id = id(p)
        parts_map[p_id] = p
        corners = get_part_corners_world(p["part_id"], p["transform"])
        part_bounds[p_id] = {
            "min_x": np.min(corners[:, 0]),
            "max_x": np.max(corners[:, 0]),
            "min_y": np.min(corners[:, 1]),
            "max_y": np.max(corners[:, 1]),
            "min_z": np.min(corners[:, 2]),
            "max_z": np.max(corners[:, 2]),
            "mean_y": np.mean(corners[:, 1])
        }
        
    # 2. Build Dependency Graph
    # We track:
    # - blocked_by[B] = set of parts A that are blocked by B (A cannot be removed before B)
    # - in_degree[A] = count of parts B blocking A
    blocked_by = {id(p): set() for p in parts}
    in_degree = {id(p): 0 for p in parts}
    
    for i in range(len(parts)):
        id_a = id(parts[i])
        bounds_a = part_bounds[id_a]
        for j in range(len(parts)):
            if i == j:
                continue
            id_b = id(parts[j])
            bounds_b = part_bounds[id_b]
            
            # If B blocks A (B is above A and overlaps in 2D)
            if is_blocked_upwards_cached(bounds_a, bounds_b):
                if id_a not in blocked_by[id_b]:
                    blocked_by[id_b].add(id_a)
                    in_degree[id_a] += 1
                    
    # 3. Disassembly loop using a queue of unblocked parts (in-degree == 0)
    # in-degree == 0 means nothing is above this part blocking it.
    removable = [p_id for p_id, deg in in_degree.items() if deg == 0]
    disassembly_order = []
    
    # Track which parts have been removed
    removed_set = set()
    
    # We process until all parts are removed
    while len(disassembly_order) < len(parts):
        if not removable:
            # Fallback for cycles (interlocking parts): find the remaining part physically highest
            remaining_ids = [id(p) for p in parts if id(p) not in removed_set]
            if not remaining_ids:
                break
            remaining_ids.sort(key=lambda p_id: part_bounds[p_id]["mean_y"])
            # Remove the highest part
            highest_id = remaining_ids[0]
            removable.append(highest_id)
            
        current_id = removable.pop(0)
        if current_id in removed_set:
            continue
            
        disassembly_order.append(parts_map[current_id])
        removed_set.add(current_id)
        
        # Decrement in-degree for all parts that were blocked by the removed part
        for blocked_id in blocked_by[current_id]:
            if blocked_id in removed_set:
                continue
            in_degree[blocked_id] -= 1
            if in_degree[blocked_id] <= 0:
                removable.append(blocked_id)
                
    # The build sequence is the reverse of disassembly sequence
    build_order = list(reversed(disassembly_order))
    return build_order
