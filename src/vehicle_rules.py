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

AESTHETIC_PARTS = {
    "3069b.dat", "2431.dat", "87079.dat", "3070b.dat", "63864.dat", "6636.dat",
    "50950.dat", "61678.dat", "85970.dat", "54200.dat", "85984.dat", "6091.dat",
    "11477.dat"
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
    - blocked_aesthetic_count: Number of aesthetic parts blocked by having parts stacked directly on top.
    """
    metrics = {
        "wheel_count": 0,
        "wheels_at_bottom": True,
        "symmetry_score": 1.0,
        "cabin_height_valid": True,
        "cabin_centered": True,
        "has_cabin_parts": False,
        "blocked_aesthetic_count": 0
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
        
    # 5. Check for blocked aesthetic parts (structural block)
    blocked_aesthetic = 0
    for p1 in parts:
        p1_id = p1.part_id.lower()
        if p1_id in AESTHETIC_PARTS:
            x1, y1, z1 = p1.transform[0, 3], p1.transform[1, 3], p1.transform[2, 3]
            dim1 = get_part_dimensions(p1.part_id)
            w1, d1 = dim1[0], dim1[2]
            
            # Check if any other part is directly above p1
            for p2 in parts:
                if p2 is p1:
                    continue
                x2, y2, z2 = p2.transform[0, 3], p2.transform[1, 3], p2.transform[2, 3]
                
                # In LDraw coordinates, Y decreases upwards. y2 < y1 means p2 is higher.
                dy = y1 - y2
                if 0.0 < dy <= 25.0:
                    # Check horizontal overlap with simple bounding box
                    dx = abs(x2 - x1)
                    dz = abs(z2 - z1)
                    if dx < (w1 / 2.0 + 4.0) and dz < (d1 / 2.0 + 4.0):
                        blocked_aesthetic += 1
                        break  # Count once per blocked aesthetic part
                        
    metrics["blocked_aesthetic_count"] = blocked_aesthetic
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
    elif wc == 2 or wc == 6:
        reward += 8.0
    elif wc == 3 or wc == 8:
        reward += 3.0
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
            
    # Penalize structural block (blocked aesthetic parts)
    reward -= 5.0 * metrics.get("blocked_aesthetic_count", 0)
    
    return reward


# ---------------------------------------------------------------------------
# FASE 3: Reference-similarity enriched reward
# ---------------------------------------------------------------------------

def _histogram_similarity(parts: list, refs: list) -> float:
    """
    Computes normalised histogram similarity between the generated assembly
    and the closest reference model.

    Similarity metric: 1 - (L1 distance / max_possible_L1_distance)
    where histograms are normalised part-frequency vectors.

    Returns:
        float in [0, 1] — 1.0 = identical histogram to closest ref.
    """
    if not refs or not parts:
        return 0.0

    from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS
    vocab = VEHICLE_ALLOWED_PARTS
    n = len(vocab)

    # Build generated histogram (normalised)
    gen_counts = {}
    for p in parts:
        gen_counts[p.part_id] = gen_counts.get(p.part_id, 0) + 1
    total_gen = max(len(parts), 1)
    gen_hist = np.array([gen_counts.get(pid, 0) / total_gen for pid in vocab], dtype=np.float32)

    best_sim = 0.0
    for ref in refs:
        ref_counts = ref.inventory if hasattr(ref, "inventory") else {}
        total_ref = max(sum(ref_counts.values()), 1) if ref_counts else 1
        ref_hist = np.array([ref_counts.get(pid, 0) / total_ref for pid in vocab], dtype=np.float32)
        l1_dist = np.sum(np.abs(gen_hist - ref_hist))
        sim = 1.0 - (l1_dist / 2.0)  # L1 in [0,2] for normalised histograms
        if sim > best_sim:
            best_sim = sim

    return float(np.clip(best_sim, 0.0, 1.0))


def _bigram_compliance_score(parts: list, design_priors) -> float:
    """
    Computes the average connectivity bigram compliance for the assembly.
    For each pair of consecutive parts (in step order), looks up
    P(part_j | part_i) in the design priors bigrams.

    Returns:
        float in [0, 1] — higher means the part-connection sequence matches
        patterns from reference models.
    """
    if design_priors is None or design_priors.n_reference_models == 0:
        return 0.5  # neutral, no signal
    if len(parts) < 2:
        return 0.5

    import math
    scores = []
    for i in range(len(parts) - 1):
        src = parts[i].part_id
        dst = parts[i + 1].part_id
        # get_connectivity_log_prior returns log(P(dst|src))
        log_p = design_priors.get_connectivity_log_prior(src, dst, smoothing=1e-4)
        # Convert to [0,1]: exp(log_p) in (0, 1], max is ~1 for known frequent pairs
        scores.append(min(math.exp(log_p), 1.0))

    return float(np.mean(scores))


def get_vehicle_rl_reward_with_refs(
    parts: list,
    refs: list = None,
    design_priors=None,
    base_stability_reward: float = 10.0,
    ref_similarity_weight: float = 5.0,
    bigram_compliance_weight: float = 3.0,
) -> float:
    """
    Enhanced RL reward for vehicle generation, enriched with:
    - Reference model histogram similarity (how close is the part distribution
      to the closest known-good vehicle reference)
    - Bigram compliance score (how well do part transitions match reference patterns)

    Falls back to get_vehicle_rl_reward() when refs/priors are not available.

    Args:
        parts:                    List[ParsedPart] — generated assembly.
        refs:                     Optional List[RefModel] — reference vehicles.
        design_priors:            Optional DesignPriors — for bigram compliance.
        base_stability_reward:    Base reward for structural validity.
        ref_similarity_weight:    Weight for histogram similarity bonus (default 5.0).
        bigram_compliance_weight: Weight for bigram compliance bonus (default 3.0).

    Returns:
        float: Total reward.

    Reward breakdown (approximate ranges):
      Base (get_vehicle_rl_reward):  [-20, +45]
      + ref_similarity: [0, +5]
      + bigram_compliance: [0, +3]
      Total range: [-20, +53]
    """
    # Base reward (topology + stability)
    base_reward = get_vehicle_rl_reward(parts, base_stability_reward=base_stability_reward)

    # Early exit: no enrichment if base reward is catastrophically bad
    if base_reward <= -10.0:
        return base_reward

    # Reference histogram similarity bonus
    sim_bonus = 0.0
    if refs:
        sim = _histogram_similarity(parts, refs)
        sim_bonus = ref_similarity_weight * sim

    # Bigram compliance bonus
    bigram_bonus = 0.0
    if design_priors is not None and design_priors.n_reference_models > 0:
        compliance = _bigram_compliance_score(parts, design_priors)
        bigram_bonus = bigram_compliance_weight * compliance

    return base_reward + sim_bonus + bigram_bonus
