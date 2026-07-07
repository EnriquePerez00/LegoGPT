import numpy as np
from src.parser import ParsedPart
from src.validator import get_part_dimensions

# Lowercase wheel/tire part names
WHEEL_PARTS = {
    "3139.dat", "42610.dat", "56902.dat", "30027.dat", "18976.dat", "18977.dat", 
    "55981.dat", "55982.dat", "30285.dat", "6014.dat", "6015.dat", "56890.dat",
    "56891.dat", "30285b.dat", "6014b.dat"
}

CABIN_PARTS = {
    "3823.dat",  # Windscreen
    "3829.dat",  # Steering column
    "3829c01.dat"
}

def evaluate_vehicle_topology(parts: list[ParsedPart]) -> dict:
    """
    Evaluates vehicle structural features and returns a dictionary of metrics.
    Metrics evaluated:
    - wheel_count: Total number of wheels/tires.
    - wheels_at_bottom: True if all wheels are at the lowest layer (max Y).
    - symmetry_score: Proportion of parts that have a mirrored counterpart on the opposite side of X-axis.
    - cabin_height_valid: True if cabin parts are higher (lower Y) than the wheels.
    - cabin_centered: True if cabin parts are centered on the X-axis.
    """
    metrics = {
        "wheel_count": 0,
        "wheels_at_bottom": True,
        "symmetry_score": 1.0,
        "cabin_height_valid": True,
        "cabin_centered": True,
        "has_cabin_parts": False
    }
    
    if not parts:
        return metrics
        
    # 1. Identify wheels and cabin components
    wheels = []
    cabin_parts = []
    other_parts = []
    
    for p in parts:
        p_id = p.part_id.lower()
        if p_id in WHEEL_PARTS:
            wheels.append(p)
        elif p_id in CABIN_PARTS:
            cabin_parts.append(p)
        else:
            other_parts.append(p)
            
    metrics["wheel_count"] = len(wheels)
    metrics["has_cabin_parts"] = len(cabin_parts) > 0
    
    # 2. Check wheel layout
    if wheels:
        # Ground level is the maximum Y value (in LDraw, Y increases downwards)
        max_y = max(p.transform[1, 3] for p in parts)
        for w in wheels:
            w_y = w.transform[1, 3]
            # Wheels should be near the bottom (within 15 LDU of max_y)
            if abs(w_y - max_y) > 15.0:
                metrics["wheels_at_bottom"] = False
                
    # 3. Check cabin components position
    if wheels and cabin_parts:
        wheel_avg_y = np.mean([w.transform[1, 3] for w in wheels])
        for c in cabin_parts:
            # Cabin Y should be smaller (higher up) than wheels
            if c.transform[1, 3] >= wheel_avg_y - 8.0:
                metrics["cabin_height_valid"] = False
            # Cabin should be centered along X-axis (width)
            if abs(c.transform[0, 3]) > 10.0:
                metrics["cabin_centered"] = False
                
    # 4. Compute bilateral symmetry score across X-axis (left vs right)
    # For each part at X != 0, check if there is a matching counterpart of the same type at -X
    symmetric_count = 0
    total_to_check = 0
    
    matched_indices = set()
    for i, p1 in enumerate(parts):
        x1 = p1.transform[0, 3]
        if abs(x1) < 6.0:
            # Self-symmetric part (near center line)
            symmetric_count += 1
            continue
            
        total_to_check += 1
        # Search for a match in other parts
        found_match = False
        for j, p2 in enumerate(parts):
            if i == j or j in matched_indices:
                continue
            if p1.part_id != p2.part_id:
                continue
                
            x2 = p2.transform[0, 3]
            y1, y2 = p1.transform[1, 3], p2.transform[1, 3]
            z1, z2 = p1.transform[2, 3], p2.transform[2, 3]
            
            # Check mirror reflection on X-axis: x1 + x2 should be close to 0
            if abs(x1 + x2) < 8.0 and abs(y1 - y2) < 5.0 and abs(z1 - z2) < 5.0:
                found_match = True
                matched_indices.add(j)
                break
                
        if found_match:
            symmetric_count += 1
            
    if len(parts) > 0:
        metrics["symmetry_score"] = symmetric_count / len(parts)
        
    return metrics

def get_vehicle_rl_reward(parts: list[ParsedPart], base_stability_reward: float = 10.0) -> float:
    """
    Computes RL reward tailored specifically for 4-wheeled vehicles.
    
    Reward Components:
    - Base stability: +10 if connected and gravity check passes.
    - Wheel count:
      - Exactly 4 wheels: +15.0
      - 2 wheels: +4.0
      - Otherwise: -10.0
    - Wheel placement:
      - Wheels at bottom: +5.0, else -5.0
    - Bilateral Symmetry:
      - +10.0 * symmetry_score
    - Cabin placement (if cabin parts are present):
      - Correct height and centered: +5.0
      - Incorrect height or off-center: -5.0
    """
    from src.validator import check_connectivity_and_gravity
    
    if not parts:
        return -20.0
        
    # Check baseline stability
    is_stable = check_connectivity_and_gravity(parts)
    if not is_stable:
        return -10.0
        
    reward = base_stability_reward
    metrics = evaluate_vehicle_topology(parts)
    
    # Wheel count reward
    wc = metrics["wheel_count"]
    if wc == 4:
        reward += 15.0
    elif wc == 2:
        reward += 4.0
    else:
        reward -= 10.0
        
    # Wheel position reward
    if wc > 0:
        if metrics["wheels_at_bottom"]:
            reward += 5.0
        else:
            reward -= 5.0
            
    # Symmetry reward
    reward += 10.0 * metrics["symmetry_score"]
    
    # Cabin reward
    if metrics["has_cabin_parts"]:
        if metrics["cabin_height_valid"] and metrics["cabin_centered"]:
            reward += 5.0
        else:
            reward -= 5.0
            
    return reward
