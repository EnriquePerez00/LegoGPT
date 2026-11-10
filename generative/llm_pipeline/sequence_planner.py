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


def normalize_parts_to_ground(parts: list) -> list:
    """
    Normalizes a list of ParsedPart objects so that the lowest physical point
    of the assembly aligns with ground level Y = 0.0 (all parts reside at Y <= 0.0
    in standard LDraw coordinates where +Y points downward).
    """
    if not parts:
        return []

    from src.parser import ParsedPart

    # Compute the maximum Y translation (lowest physical bottom in LDraw coords)
    max_y = max(float(p.transform[1, 3]) for p in parts)
    normalized = []
    for p in parts:
        tf = np.array(p.transform, dtype=np.float32).copy()
        tf[1, 3] = float(tf[1, 3]) - max_y
        # Snap very small float drift near ground to 0.0
        if abs(float(tf[1, 3])) < 1e-3:
            tf[1, 3] = 0.0
        normalized.append(
            ParsedPart(
                part_id=p.part_id,
                color=int(p.color),
                transform=tf,
                step_id=int(p.step_id),
            )
        )
    return normalized


def schedule_topological_steps(
    parts: list,
    max_parts_per_step: int = 4,
    force_reschedule: bool = False,
) -> list:
    """
    Takes a list of ParsedPart objects (e.g., from a flat LDraw/MPD/IO/MBX file without steps)
    and schedules physically valid, ground-up '0 STEP' construction steps:
      1. Normalizes ground level so max(Y_bottom) == 0.0 (all Y <= 0.0).
      2. Builds physical port/stud connectivity graph (check_connection_optimized).
      3. Orders parts from ground (Y=0) upwards (-Y) ensuring each added part
         connects to already-placed parts (connected BFS + vertical support priority).
      4. Groups bilateral symmetric pairs (X <-> -X) and same-layer parts into discrete step_ids.
    """
    if not parts:
        return []

    norm_parts = normalize_parts_to_ground(parts)
    unique_steps = {p.step_id for p in norm_parts}
    if len(unique_steps) > 1 and not force_reschedule:
        # Already has discrete steps; ensure step_ids are contiguous 0..S-1
        sorted_steps = sorted(unique_steps)
        step_remap = {old_s: new_s for new_s, old_s in enumerate(sorted_steps)}
        for p in norm_parts:
            p.step_id = step_remap[p.step_id]
        return sorted(norm_parts, key=lambda p: p.step_id)

    from src.validator import check_connection_optimized

    n = len(norm_parts)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if check_connection_optimized(norm_parts[i], norm_parts[j]):
                adj[i].add(j)
                adj[j].add(i)

    # Identify bilateral symmetry pairs (same part_id, same Y and Z, mirrored X)
    mirror_partner: dict[int, int] = {}
    for i in range(n):
        if i in mirror_partner:
            continue
        pi = norm_parts[i]
        xi, yi, zi = float(pi.transform[0, 3]), float(pi.transform[1, 3]), float(pi.transform[2, 3])
        if abs(xi) < 5.0:
            continue
        for j in range(i + 1, n):
            if j in mirror_partner:
                continue
            pj = norm_parts[j]
            if pi.part_id != pj.part_id:
                continue
            xj, yj, zj = float(pj.transform[0, 3]), float(pj.transform[1, 3]), float(pj.transform[2, 3])
            if abs(xi + xj) < 4.0 and abs(yi - yj) < 2.0 and abs(zi - zj) < 4.0:
                mirror_partner[i] = j
                mirror_partner[j] = i
                break

    # Seed with ground parts (|Y| <= 1.0), up to 4 foundation parts
    ground_indices = [
        i for i in range(n) if abs(float(norm_parts[i].transform[1, 3])) <= 1.0
    ]
    if not ground_indices:
        # Fallback: lowest physical part (largest Y)
        max_y = max(float(p.transform[1, 3]) for p in norm_parts)
        ground_indices = [i for i in range(n) if abs(float(norm_parts[i].transform[1, 3]) - max_y) <= 1.0]

    placed: set[int] = set()
    ordered_steps: list[list[int]] = []

    # Step 0: Foundation parts (at most 4 touching ground)
    step0 = sorted(ground_indices, key=lambda idx: (norm_parts[idx].part_id, float(norm_parts[idx].transform[0, 3])))[:4]
    ordered_steps.append(step0)
    placed.update(step0)

    while len(placed) < n:
        # Find unplaced parts connected to already placed parts
        frontier = [
            i for i in range(n)
            if i not in placed and any(nb in placed for nb in adj[i])
        ]
        if not frontier:
            # Disconnected subassembly: pick lowest remaining part (largest Y)
            remaining = [i for i in range(n) if i not in placed]
            remaining.sort(key=lambda idx: (-float(norm_parts[idx].transform[1, 3]), idx))
            frontier = [remaining[0]]

        # Sort frontier from lowest physical level (largest Y) upwards, then by Z and X
        frontier.sort(
            key=lambda idx: (
                -round(float(norm_parts[idx].transform[1, 3]) / 8.0),
                norm_parts[idx].part_id,
                float(norm_parts[idx].transform[2, 3]),
                float(norm_parts[idx].transform[0, 3]),
            )
        )

        lead_idx = frontier[0]
        lead_layer = round(float(norm_parts[lead_idx].transform[1, 3]) / 8.0)
        current_step_batch = [lead_idx]
        placed.add(lead_idx)

        # If lead_idx has a symmetric partner that is also available or on the same layer, pair it
        partner = mirror_partner.get(lead_idx)
        if partner is not None and partner not in placed:
            current_step_batch.append(partner)
            placed.add(partner)

        # Add same-layer, same-part_id frontier items up to max_parts_per_step
        for cand_idx in frontier[1:]:
            if len(current_step_batch) >= max_parts_per_step:
                break
            if cand_idx in placed:
                continue
            cand_layer = round(float(norm_parts[cand_idx].transform[1, 3]) / 8.0)
            if cand_layer == lead_layer and norm_parts[cand_idx].part_id == norm_parts[lead_idx].part_id:
                current_step_batch.append(cand_idx)
                placed.add(cand_idx)
                c_partner = mirror_partner.get(cand_idx)
                if c_partner is not None and c_partner not in placed and len(current_step_batch) < max_parts_per_step:
                    current_step_batch.append(c_partner)
                    placed.add(c_partner)

        ordered_steps.append(current_step_batch)

    result = []
    for step_id, indices in enumerate(ordered_steps):
        for idx in indices:
            p = norm_parts[idx]
            p.step_id = step_id
            result.append(p)

    return result
