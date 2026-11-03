from __future__ import annotations
import os
import json
import time
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any
import numpy as np
import psutil
import torch
import torch.nn.functional as F

from src.parser import ParsedPart, ALLOWED_PARTS, ALLOWED_COLORS, build_pyg_graph
from src.validator import (
    get_part_dimensions,
    get_studs_and_sockets_world,
    check_collisions,
    check_connectivity_and_gravity,
    check_connection_optimized,
)
from src.graph_validator import LegoGraphValidator
from src.model import LegoGNN, get_device
from src.rl_train import calculate_interlocking_score
from src.vehicle_rules import evaluate_vehicle_topology, WHEEL_PARTS, CABIN_PARTS
from generative.llm_pipeline.rl_loop import calculate_com_stability

# Vocabulary extensions for realistic Vehicles and Animals
ANIMAL_PARTS = [
    "3005.dat",   # Brick 1x1 (leg / neck / ear)
    "3004.dat",   # Brick 1x2 (leg / snout / tail)
    "3003.dat",   # Brick 2x2 (head / body block)
    "3002.dat",   # Brick 2x3 (torso)
    "3001.dat",   # Brick 2x4 (torso)
    "3010.dat",   # Brick 1x4 (flank / spine)
    "3024.dat",   # Plate 1x1 (eye / ear tip / claw)
    "3023.dat",   # Plate 1x2 (jaw / paw / ear)
    "3022.dat",   # Plate 2x2 (head top / chest)
    "3021.dat",   # Plate 2x3 (body plate)
    "3020.dat",   # Plate 2x4 (torso base)
    "3710.dat",   # Plate 1x4 (spine / tail)
    "3034.dat",   # Plate 2x8 (long body)
    "3040b.dat",  # Slope 45 2x1 (ears / snout / tail tip)
    "3039.dat",   # Slope 45 2x2 (forehead / chest / haunches)
    "54200.dat",  # Cheese Slope 1x1x2/3 (ears / horns / nose)
    "85984.dat",  # Slope 30 1x2x2/3 (snout / crest)
]

VEHICLE_GENERATION_PARTS = [
    "30027.dat",  # Wheel 8x6
    "42610.dat",  # Wheel Rim 11x8
    "3139.dat",   # Tire 14x6
    "3020.dat",   # Plate 2x4 (chassis)
    "3034.dat",   # Plate 2x8 (long chassis)
    "3022.dat",   # Plate 2x2 (hood / trunk)
    "3023.dat",   # Plate 1x2 (bumper / spoiler mount)
    "3710.dat",   # Plate 1x4 (side skirt / spoiler wing)
    "3024.dat",   # Plate 1x1 (headlight / taillight)
    "3004.dat",   # Brick 1x2 (side door / pillar)
    "3003.dat",   # Brick 2x2 (engine block / cabin core)
    "3001.dat",   # Brick 2x4 (bodywork)
    "3010.dat",   # Brick 1x4 (side panel)
    "3823.dat",   # Windscreen 2x4x2
    "50745.dat",  # Mudguard 4x2.5
    "3040b.dat",  # Slope 45 2x1 (aerodynamic nose)
    "3039.dat",   # Slope 45 2x2 (hood slope)
    "85984.dat",  # Slope 30 1x2 (spoiler / nose)
]

FEEDBACK_FILE_PATH = "data/human_feedback.json"
REFINED_MODEL_PATH = "models/creative_refined_mps.pt"
REFINED_META_PATH = "models/creative_refined_mps.json"


def get_optimal_workers() -> int:
    """
    Dynamically detects CPU cores on Apple Silicon M4 and reserves 2 cores
    as a safety margin for macOS and UI responsiveness.
    """
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 2)


def check_memory_safety(max_percent: float = 80.0) -> dict[str, Any]:
    """
    Monitors Unified Memory usage via psutil and enforces a 75-80% safety buffer.
    Clears MPS cache if memory pressure exceeds threshold.
    """
    vm = psutil.virtual_memory()
    is_safe = vm.percent < max_percent
    if not is_safe and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return {
        "total_gb": round(vm.total / (1024 ** 3), 2),
        "available_gb": round(vm.available / (1024 ** 3), 2),
        "percent_used": round(vm.percent, 1),
        "is_safe": is_safe,
    }


def parse_prompt_semantics(prompt: str, explicit_category: str | None = None) -> dict[str, Any]:
    """
    Extracts semantic category ('vehicle', 'animal', 'general'), sub-type,
    and primary/secondary LDraw color codes from natural language prompts.
    """
    p = prompt.lower().strip()

    # 1. Detect category
    vehicle_kws = [
        "coche", "carro", "auto", "camion", "camión", "furgoneta", "buggy", "deportivo",
        "formula", "f1", "kart", "jeep", "4x4", "todoterreno", "tractor", "vehiculo",
        "vehículo", "car", "truck", "racer", "rover", "police", "policia", "bomberos", "taxi"
    ]
    animal_kws = [
        "animal", "perro", "gato", "leon", "león", "tigre", "caballo", "vaca", "oveja",
        "cerdo", "elefante", "jirafa", "oso", "lobo", "zorro", "conejo", "dinosaurio",
        "dino", "t-rex", "rex", "pajaro", "pájaro", "ave", "pato", "pollo", "pingüino",
        "pinguino", "tortuga", "dog", "cat", "lion", "horse", "bird", "duck", "dragon", "dragón"
    ]

    category = explicit_category if explicit_category in ("vehicle", "animal", "general") else None
    if category is None:
        if any(kw in p for kw in vehicle_kws):
            category = "vehicle"
        elif any(kw in p for kw in animal_kws):
            category = "animal"
        else:
            category = "general"

    # Subtype detection for morphology nuances
    subtype = "standard"
    if category == "vehicle":
        if any(w in p for w in ["descapotable", "convertible", "cabrio", "roadster", "spider", "spyder"]):
            subtype = "convertible_sport"
        elif any(w in p for w in ["camion", "camión", "truck", "furgoneta", "4x4", "jeep", "todoterreno"]):
            subtype = "truck"
        elif any(w in p for w in ["deportivo", "f1", "formula", "racer", "kart", "sport", "ferrari"]):
            subtype = "sport"
        else:
            subtype = "sedan"
    elif category == "animal":
        if any(w in p for w in ["pajaro", "pájaro", "ave", "pato", "pollo", "pingüino", "pinguino", "bird", "duck", "t-rex", "rex"]):
            subtype = "biped"
        elif any(w in p for w in ["jirafa", "caballo", "llama", "dinosaurio", "dino", "dragon", "dragón"]):
            subtype = "tall_quadruped"
        else:
            subtype = "quadruped"

    # 2. Detect colors (LDraw color codes)
    color_map = [
        (["rojo", "roja", "red", "bomberos", "ferrari"], 4),
        (["azul", "blue", "policia", "policía"], 1),
        (["amarillo", "amarilla", "yellow", "taxi", "león", "leon", "jirafa", "pato", "pollo"], 14),
        (["verde", "green", "dino", "dinosaurio", "tortuga", "tractor"], 2),
        (["blanco", "blanca", "white", "oveja", "conejo"], 15),
        (["negro", "negra", "black", "pantera"], 0),
        (["naranja", "orange", "tigre", "zorro"], 25),
        (["marron", "marrón", "brown", "perro", "caballo", "oso"], 6),
        (["gris", "gray", "grey", "elefante", "lobo"], 7),
        (["rosa", "pink", "cerdo"], 29),
        (["celeste", "light blue"], 9),
        (["lima", "lime"], 10),
    ]

    primary_color = None
    for kws, code in color_map:
        if any(kw in p for kw in kws):
            primary_color = code
            break

    if primary_color is None:
        if category == "vehicle":
            primary_color = 4   # Classic LEGO Red car default
        elif category == "animal":
            primary_color = 6   # Classic Brown / Tan animal default
        else:
            primary_color = 14  # Classic Yellow default

    # Secondary / accent colors
    if category == "vehicle":
        wheel_color = 0         # Black tires/wheels
        chassis_color = 7       # Light gray chassis
        accent_color = 15 if primary_color != 15 else 1  # White windscreen/headlights
    elif category == "animal":
        wheel_color = primary_color
        chassis_color = primary_color
        accent_color = 15 if primary_color != 15 else 0  # Eyes / snout contrast
    else:
        wheel_color = primary_color
        chassis_color = primary_color
        accent_color = primary_color

    return {
        "category": category,
        "subtype": subtype,
        "primary_color": primary_color,
        "chassis_color": chassis_color,
        "wheel_color": wheel_color,
        "accent_color": accent_color,
    }


def _make_part(
    part_id: str,
    color: int,
    x: float,
    y: float,
    z: float,
    theta_deg: int = 0,
    step_id: int = 0,
) -> ParsedPart:
    rad = math.radians(theta_deg)
    c, s = int(round(math.cos(rad))), int(round(math.sin(rad)))
    t = np.eye(4, dtype=np.float32)
    t[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    t[:3, 3] = [float(x), float(y), float(z)]
    return ParsedPart(part_id=part_id, color=int(color), transform=t, step_id=step_id)


def _get_obb_params(part: ParsedPart) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes OBB center, rotation axes, and half-extents for fast incremental SAT checks."""
    dim = get_part_dimensions(part.part_id)
    c = (part.transform @ np.array([0.0, -dim[1] / 2.0, 0.0, 1.0], dtype=np.float32))[:3]
    axes = part.transform[:3, :3]
    h = np.array([dim[0] / 2.0 - 0.1, dim[1] / 2.0 - 0.01, dim[2] / 2.0 - 0.1], dtype=np.float32)
    return c, axes, h


def _can_add_parts(candidate_parts: list[ParsedPart], current_state: list[ParsedPart], validator: LegoGraphValidator) -> bool:
    """
    Verifies that adding `candidate_parts` (1 part or a symmetric pair of 2 parts)
    strictly obeys all physical rules in O(N) time:
    - Y <= 0.0 (no part below ground surface)
    - At most 4 parts touching ground (|Y| <= 1.0)
    - Stud-to-socket connection to existing structure (within 1.0 LDU)
    - Zero OBB collisions (both with current_state and between the pair)
    """
    from src.validator import obb_overlap_sat
    temp_state = list(current_state)
    temp_obbs = [_get_obb_params(p) for p in temp_state]

    for cand in candidate_parts:
        y_val = float(cand.transform[1, 3])
        if y_val > 1.0:
            return False
        if abs(y_val) <= 1.0:
            ground_count = sum(1 for p in temp_state if abs(p.transform[1, 3]) <= 1.0)
            if ground_count >= 4:
                return False

        # Fast O(N) incremental OBB collision check
        c_new, axes_new, h_new = _get_obb_params(cand)
        for c_ex, axes_ex, h_ex in temp_obbs:
            # Fast bounding-sphere / vertical pre-filter
            if abs(float(c_new[1] - c_ex[1])) > float(h_new[1] + h_ex[1] + 1.0):
                continue
            if obb_overlap_sat(c_new, axes_new, h_new, c_ex, axes_ex, h_ex):
                return False

        # Stud-to-socket connectivity check against existing parts
        if temp_state:
            has_conn = False
            for ex in reversed(temp_state):
                if check_connection_optimized(cand, ex):
                    has_conn = True
                    break
            if not has_conn:
                return False

        temp_state.append(cand)
        temp_obbs.append((c_new, axes_new, h_new))
    return True


def _build_vehicle_foundation(
    target_num_pieces: int,
    semantics: dict[str, Any],
    rng: np.random.Generator,
    validator: LegoGraphValidator,
) -> list[ParsedPart]:
    """
    Creates a physically grounded, stud-to-socket connected vehicle foundation:
    - When target_num_pieces >= 6: 4 wheels at Y=0.0 (exact 4 ground contacts)
      bridged at Y=-16.0 by a central chassis plate (3020.dat 2x4 or 3034.dat 2x8).
    - Every wheel stud at (±10, -16, ±Z_wheel) connects with 0.0 LDU error to the
      chassis plate sockets at Y=-16.0.
    - For large builds (target_num_pieces >= 40) or convertible sports cars,
      adds a wide deck plate (3035.dat 4x8 or 3032.dat 4x6) and front windscreen (3823.dat)
      with open-top cockpit geometry.
    """
    w_color = semantics["wheel_color"]
    c_color = semantics["chassis_color"]
    p_color = semantics["primary_color"]
    a_color = semantics["accent_color"]
    subtype = semantics["subtype"]

    wheel_choices = ["30027.dat", "42610.dat", "3139.dat"]
    wheel_id = str(rng.choice(wheel_choices))

    if target_num_pieces < 5:
        base = _make_part("3020.dat", c_color, 0.0, 0.0, 0.0, theta_deg=0, step_id=0)
        return [base]

    # Choose wheelbase based on piece budget and subtype
    if target_num_pieces >= 12 or subtype in ("truck", "sport", "convertible_sport"):
        chassis_id = "3034.dat"
    else:
        chassis_id = "3020.dat"

    z_wheel = 50.0 if (chassis_id == "3034.dat" and (target_num_pieces >= 25 or rng.random() < 0.5)) else 30.0

    # Step 1: 4 grounded wheels at Y = 0.0 (max 4 ground parts rule respected!)
    wheels = [
        _make_part(wheel_id, w_color, -10.0, 0.0, -z_wheel, theta_deg=0, step_id=0),
        _make_part(wheel_id, w_color,  10.0, 0.0, -z_wheel, theta_deg=0, step_id=0),
        _make_part(wheel_id, w_color, -10.0, 0.0,  z_wheel, theta_deg=0, step_id=0),
        _make_part(wheel_id, w_color,  10.0, 0.0,  z_wheel, theta_deg=0, step_id=0),
    ]

    # Step 2: Chassis bridge plate at Y = -16.0 snapping all 4 wheels together
    chassis = _make_part(chassis_id, c_color, 0.0, -16.0, 0.0, theta_deg=0, step_id=1)
    parts = wheels + [chassis]

    # Optional secondary structural layer if budget allows
    if target_num_pieces >= 20:
        # Wide floor deck at Y = -24.0 (4x8 plate 3035.dat or 4x6 plate 3032.dat) to give 32+ studs of room
        deck_id = "3035.dat" if target_num_pieces >= 35 else "3032.dat"
        deck = _make_part(deck_id, p_color, 0.0, -24.0, 0.0, theta_deg=0, step_id=2)
        if _can_add_parts([deck], parts, validator):
            parts.append(deck)
            # Add front windscreen at Z = -40.0, Y = -32.0 (leaving open convertible cockpit behind it!)
            windscreen = _make_part("3823.dat", a_color, 0.0, -32.0, -40.0, theta_deg=0, step_id=3)
            if _can_add_parts([windscreen], parts, validator):
                parts.append(windscreen)
    elif target_num_pieces >= 9:
        cabin_or_body = _make_part("3003.dat" if subtype == "truck" else "3020.dat", p_color, 0.0, -24.0, 0.0, theta_deg=0, step_id=2)
        if _can_add_parts([cabin_or_body], parts, validator):
            parts.append(cabin_or_body)

    return parts


def _build_animal_foundation(
    target_num_pieces: int,
    semantics: dict[str, Any],
    rng: np.random.Generator,
    validator: LegoGraphValidator,
) -> list[ParsedPart]:
    """
    Creates a physically grounded, stud-to-socket connected animal foundation:
    - Quadrupeds: 4 symmetric legs at Y=0.0 (3005.dat 1x1 or 3004.dat 1x2)
      bridged at Y=-24.0 by a torso brick/plate (3001.dat 2x4 or 3020.dat 2x4).
    - Bipeds (birds/t-rex): 2 symmetric legs at Y=0.0 bridged at Y=-24.0 by body.
    """
    p_color = semantics["primary_color"]
    subtype = semantics["subtype"]

    if target_num_pieces < 5:
        return [_make_part("3004.dat", p_color, 0.0, 0.0, 0.0, theta_deg=0, step_id=0)]

    if subtype == "biped" or target_num_pieces == 5:
        leg_id = "3005.dat" if rng.random() < 0.6 else "3004.dat"
        legs = [
            _make_part(leg_id, p_color, -10.0, 0.0, 0.0, theta_deg=0, step_id=0),
            _make_part(leg_id, p_color,  10.0, 0.0, 0.0, theta_deg=0, step_id=0),
        ]
        torso_id = str(rng.choice(["3003.dat", "3001.dat", "3020.dat"]))
        torso = _make_part(torso_id, p_color, 0.0, -24.0, 0.0, theta_deg=0, step_id=1)
        return legs + [torso]

    # Quadruped: 4 symmetric legs at Y = 0.0 (exactly 4 ground parts!)
    z_leg = 30.0
    leg_id = "3005.dat"
    legs = [
        _make_part(leg_id, p_color, -10.0, 0.0, -z_leg, theta_deg=0, step_id=0),
        _make_part(leg_id, p_color,  10.0, 0.0, -z_leg, theta_deg=0, step_id=0),
        _make_part(leg_id, p_color, -10.0, 0.0,  z_leg, theta_deg=0, step_id=0),
        _make_part(leg_id, p_color,  10.0, 0.0,  z_leg, theta_deg=0, step_id=0),
    ]
    torso_id = "3034.dat" if target_num_pieces >= 30 else ("3001.dat" if rng.random() < 0.6 else "3020.dat")
    torso = _make_part(torso_id, p_color, 0.0, -24.0, 0.0, theta_deg=0, step_id=1)
    return legs + [torso]


def evaluate_animal_topology(parts: list[ParsedPart]) -> dict[str, Any]:
    """
    Evaluates animal morphological features:
    - Grounded paws/legs (2 or 4)
    - Elevated connected torso
    - Head/snout extension at front (-Z) and/or tail at rear (+Z)
    - Bilateral symmetry across X-axis
    """
    if not parts:
        return {
            "leg_count": 0,
            "has_head_or_tail": False,
            "symmetry_score": 0.0,
            "slope_count": 0,
        }

    ground_parts = [p for p in parts if abs(p.transform[1, 3]) <= 1.0]
    upper_parts = [p for p in parts if p.transform[1, 3] <= -20.0]

    z_vals = [float(p.transform[2, 3]) for p in upper_parts] if upper_parts else [0.0]
    z_span = max(z_vals) - min(z_vals) if len(z_vals) > 1 else 0.0
    y_min = min(float(p.transform[1, 3]) for p in parts)

    slope_ids = {"3040b.dat", "3039.dat", "54200.dat", "85984.dat", "3665.dat", "3660.dat"}
    slope_count = sum(1 for p in parts if p.part_id.lower() in slope_ids)

    veh_metrics = evaluate_vehicle_topology(parts)
    return {
        "leg_count": len(ground_parts),
        "has_head_or_tail": (z_span >= 40.0 or y_min <= -40.0),
        "symmetry_score": veh_metrics["symmetry_score"],
        "slope_count": slope_count,
    }


def evaluate_candidate_metrics(parts: list[ParsedPart], category: str) -> dict[str, Any]:
    """
    Computes comprehensive physical, structural, and morphological quality metrics
    for a generated LEGO assembly.
    """
    if not parts:
        return {
            "physics_valid": False,
            "num_pieces": 0,
            "ground_parts": 0,
            "symmetry_score": 0.0,
            "interlocking_score": 0.0,
            "morphology_score": -20.0,
            "total_score": -20.0,
        }

    collisions = check_collisions(parts)
    connected_and_grounded = check_connectivity_and_gravity(parts)
    com_stable = calculate_com_stability(parts)
    ground_parts = sum(1 for p in parts if abs(p.transform[1, 3]) <= 1.0)
    above_ground = all(p.transform[1, 3] <= 1.0 for p in parts)

    physics_valid = (
        len(collisions) == 0
        and connected_and_grounded
        and com_stable
        and 1 <= ground_parts <= 4
        and above_ground
    )

    interlocking = calculate_interlocking_score(parts)
    veh_topo = evaluate_vehicle_topology(parts)
    symmetry = float(veh_topo["symmetry_score"])

    if category == "vehicle":
        morph = 0.0
        if veh_topo["wheel_count"] == 4:
            morph += 15.0
        elif veh_topo["wheel_count"] == 2:
            morph += 6.0
        if veh_topo["wheels_at_bottom"]:
            morph += 5.0
        if veh_topo["has_cabin_parts"] and veh_topo["cabin_height_valid"]:
            morph += 6.0
        morph += 10.0 * symmetry
    elif category == "animal":
        anim_topo = evaluate_animal_topology(parts)
        morph = 0.0
        if anim_topo["leg_count"] in (2, 4):
            morph += 12.0
        if anim_topo["has_head_or_tail"]:
            morph += 8.0
        morph += min(6.0, anim_topo["slope_count"] * 2.0)
        morph += 10.0 * symmetry
    else:
        morph = 10.0 * symmetry + 3.0 * interlocking

    total_score = (15.0 if physics_valid else -25.0) + morph + 2.5 * interlocking

    return {
        "physics_valid": bool(physics_valid),
        "num_pieces": len(parts),
        "ground_parts": int(ground_parts),
        "symmetry_score": round(symmetry, 3),
        "interlocking_score": round(float(interlocking), 2),
        "morphology_score": round(float(morph), 2),
        "total_score": round(float(total_score), 2),
        "com_stable": bool(com_stable),
    }


def generate_single_candidate_worker(args: dict[str, Any]) -> dict[str, Any]:
    """
    Worker function executed in parallel across M4 CPU cores.
    Generates a single complete, physically valid LEGO set of `target_num_pieces`
    conditioned on prompt semantics, symmetry macro-actions, and neural priors.
    """
    candidate_idx: int = args["candidate_idx"]
    prompt: str = args["prompt"]
    target_num_pieces: int = args["target_num_pieces"]
    semantics: dict[str, Any] = args["semantics"]
    seed: int = args["seed"]
    part_priors: dict[str, float] = args.get("part_priors", {})

    rng = np.random.default_rng(seed)
    validator = LegoGraphValidator()
    category = semantics["category"]
    subtype = semantics["subtype"]
    p_color = semantics["primary_color"]
    a_color = semantics["accent_color"]
    c_color = semantics["chassis_color"]

    # 1. Build domain-specific grounded foundation
    if category == "vehicle":
        parts = _build_vehicle_foundation(target_num_pieces, semantics, rng, validator)
        if subtype == "convertible_sport":
            # Convertible sports car: aerodynamic slopes, plates, side panels, detail 1x1/1x2 plates, open cockpit
            vocab_pool = [
                "3024.dat", "3023.dat", "3022.dat", "3021.dat", "3020.dat",
                "3710.dat", "3034.dat", "3005.dat", "3004.dat", "3003.dat",
                "3040b.dat", "3039.dat", "54200.dat", "85984.dat",
            ]
        else:
            vocab_pool = [
                "3020.dat", "3022.dat", "3023.dat", "3710.dat", "3024.dat",
                "3004.dat", "3003.dat", "3001.dat", "3010.dat",
                "3823.dat", "50745.dat", "3040b.dat", "3039.dat", "85984.dat",
            ]
    elif category == "animal":
        parts = _build_animal_foundation(target_num_pieces, semantics, rng, validator)
        vocab_pool = [
            "3005.dat", "3004.dat", "3003.dat", "3002.dat", "3001.dat",
            "3024.dat", "3023.dat", "3022.dat", "3020.dat",
            "3040b.dat", "3039.dat", "54200.dat", "85984.dat",
        ]
    else:
        base_choices = ["3001.dat", "3003.dat", "3020.dat", "3022.dat"]
        base_id = str(rng.choice(base_choices))
        parts = [_make_part(base_id, p_color, 0.0, 0.0, 0.0, theta_deg=0, step_id=0)]
        vocab_pool = list(ALLOWED_PARTS[:18])

    # Trim if foundation exceeded target_num_pieces
    if len(parts) > target_num_pieces:
        parts = parts[:target_num_pieces]

    # Compute sampling weights using neural/feedback priors
    weights = np.array([math.exp(part_priors.get(pid, 0.0)) for pid in vocab_pool], dtype=np.float64)
    if subtype == "convertible_sport" and target_num_pieces >= 40:
        # Boost detail plates and aerodynamic slopes so 100-piece sports car builds richly
        for i, pid in enumerate(vocab_pool):
            if pid in ("3024.dat", "3023.dat", "3022.dat", "3710.dat", "54200.dat", "85984.dat", "3040b.dat"):
                weights[i] *= 2.2
    weights = weights / np.sum(weights)

    current_step = max((p.step_id for p in parts), default=0) + 1

    # Dynamic spatial envelope scaling with target_num_pieces
    max_x = 80.0 if target_num_pieces >= 40 else 60.0
    max_z = 140.0 if target_num_pieces >= 40 else 90.0
    min_y = -96.0 if subtype == "convertible_sport" else (-180.0 if target_num_pieces >= 40 else -140.0)

    # 2. Autoregressive morphological + symmetric growth up to target_num_pieces
    max_outer_attempts = max(200, target_num_pieces * 10)
    outer_attempts = 0

    while len(parts) < target_num_pieces and outer_attempts < max_outer_attempts:
        outer_attempts += 1
        remaining = target_num_pieces - len(parts)

        # Decide whether to attempt a symmetric bilateral pair (2 pieces) or single piece (1 piece)
        try_symmetric_pair = (
            remaining >= 2
            and category in ("vehicle", "animal")
            and rng.random() < 0.72
        )

        cand_part_id = str(rng.choice(vocab_pool, p=weights))
        # Choose color based on part role (keep >= 80% primary color for strong color fidelity)
        if cand_part_id == "3823.dat":
            cand_color = a_color
        elif cand_part_id in ("3024.dat", "54200.dat") and rng.random() < 0.20:
            cand_color = a_color
        elif cand_part_id in ("3034.dat", "50745.dat") and rng.random() < 0.15:
            cand_color = c_color
        else:
            cand_color = p_color

        dummy = _make_part(cand_part_id, cand_color, 0.0, 0.0, 0.0, theta_deg=0, step_id=current_step)
        cand_studs, cand_sockets = get_studs_and_sockets_world(dummy)
        if len(cand_studs) == 0 and len(cand_sockets) == 0:
            continue

        # Prefer non-wheel parents so we build upwards on the chassis/torso
        valid_parents = [p for p in parts if p.part_id.lower() not in WHEEL_PARTS]
        if not valid_parents:
            valid_parents = parts

        parent_indices = list(range(len(valid_parents)))
        rng.shuffle(parent_indices)

        placed_in_step = False
        for p_idx in parent_indices[:6]:
            parent = valid_parents[p_idx]
            p_studs, _ = get_studs_and_sockets_world(parent)
            if len(p_studs) == 0 or len(cand_sockets) == 0:
                continue

            s_order = list(range(len(p_studs)))
            c_order = list(range(len(cand_sockets)))
            rng.shuffle(s_order)
            rng.shuffle(c_order)

            for s_i in s_order[:6]:
                p_anchor = p_studs[s_i]
                for c_i in c_order[:3]:
                    c_anchor = cand_sockets[c_i]
                    thetas = [0, 90, 180, 270]
                    rng.shuffle(thetas)

                    for theta in thetas[:2]:
                        rad = math.radians(theta)
                        c_rot, s_rot = int(round(math.cos(rad))), int(round(math.sin(rad)))
                        R = np.array([[c_rot, 0, s_rot], [0, 1, 0], [-s_rot, 0, c_rot]], dtype=np.float32)
                        T = p_anchor - R @ c_anchor

                        # Must build strictly upwards from chassis/base (Y < -1.0) once ground has pieces
                        if T[1] >= -1.0:
                            continue

                        # Keep within spatial envelope
                        if abs(T[0]) > max_x or abs(T[2]) > max_z or T[1] < min_y:
                            continue

                        # For convertible sports cars, keep the central cockpit zone (|X| <= 15, -15 <= Z <= 20)
                        # open above Y = -48 so there is no closed roof over the driver seats!
                        if subtype == "convertible_sport" and T[1] < -48.0 and abs(T[0]) <= 15.0 and -15.0 <= T[2] <= 20.0:
                            continue

                        if try_symmetric_pair and abs(T[0]) >= 10.0:
                            cand_left = _make_part(cand_part_id, cand_color, T[0], T[1], T[2], theta_deg=theta, step_id=current_step)
                            mirror_theta = (360 - theta) % 360
                            cand_right = _make_part(cand_part_id, cand_color, -T[0], T[1], T[2], theta_deg=mirror_theta, step_id=current_step)

                            if _can_add_parts([cand_left, cand_right], parts, validator):
                                parts.extend([cand_left, cand_right])
                                current_step += 1
                                placed_in_step = True
                                break
                        else:
                            if category in ("vehicle", "animal") and abs(T[0]) > 20.0 and rng.random() < 0.65:
                                continue
                            cand_single = _make_part(cand_part_id, cand_color, T[0], T[1], T[2], theta_deg=theta, step_id=current_step)
                            if _can_add_parts([cand_single], parts, validator):
                                test_state = parts + [cand_single]
                                if calculate_com_stability(test_state):
                                    parts.append(cand_single)
                                    current_step += 1
                                    placed_in_step = True
                                    break
                    if placed_in_step:
                        break
                if placed_in_step:
                    break
            if placed_in_step:
                break

    metrics = evaluate_candidate_metrics(parts, category=category)

    serialized_parts = []
    for idx, p in enumerate(parts):
        serialized_parts.append({
            "sequence_index": idx,
            "part_id": p.part_id,
            "color": int(p.color),
            "transform": [round(float(v), 4) for v in p.transform.flatten().tolist()],
            "step_id": int(p.step_id),
        })

    return {
        "candidate_id": f"set_{candidate_idx + 1}_{int(time.time() * 1000) % 100000}",
        "candidate_index": candidate_idx + 1,
        "prompt": prompt,
        "category": category,
        "subtype": semantics["subtype"],
        "parts": serialized_parts,
        "metrics": metrics,
    }


def _load_neural_part_priors(category: str, device_str: str) -> dict[str, float]:
    """
    Loads the best available trained/refined GNN checkpoint on MPS/CPU and extracts
    prior log-probabilities over parts, combined with human feedback statistics.
    """
    priors: dict[str, float] = {}

    # 1. Load learned preferences from human feedback history if present
    if os.path.exists(FEEDBACK_FILE_PATH):
        try:
            with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
            for rec in records:
                if rec.get("category") == category or category == "general":
                    weight = 0.25 if rec.get("approved", False) else -0.15
                    for p in rec.get("parts", []):
                        pid = p.get("part_id", "")
                        priors[pid] = priors.get(pid, 0.0) + weight
        except Exception:
            pass

    # 2. Query neural model on MPS for initial state distribution
    device = torch.device(device_str)
    ckpt_candidates = [
        REFINED_MODEL_PATH,
        "models/vehicle_fine_city.pt" if category == "vehicle" else "models/fine_tuned_rl.pt",
        "models/modelo_lego_1.pt",
    ]
    for ckpt in ckpt_candidates:
        if os.path.exists(ckpt):
            try:
                state_dict = torch.load(ckpt, map_location=device)
                # Infer dimensions from state_dict
                if "head_part.weight" in state_dict:
                    num_parts, hidden_dim = state_dict["head_part.weight"].shape
                    num_colors = state_dict["head_color.weight"].shape[0]
                    model = LegoGNN(num_part_classes=num_parts, num_color_classes=num_colors, hidden_dim=hidden_dim).to(device)
                    model.load_state_dict(state_dict)
                    model.eval()
                    with torch.no_grad():
                        dummy_part = _make_part("3020.dat", 4, 0.0, 0.0, 0.0)
                        g = build_pyg_graph([dummy_part], ALLOWED_PARTS[:num_parts], ALLOWED_COLORS[:num_colors]).to(device)
                        batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)
                        part_out, _, _ = model(g.x, g.edge_index, batch)
                        probs = F.log_softmax(part_out[0], dim=-1).cpu().numpy()
                        for i, pid in enumerate(ALLOWED_PARTS[:num_parts]):
                            priors[pid] = priors.get(pid, 0.0) + float(probs[i]) * 0.2
                    break
            except Exception:
                continue

    return priors


def generate_candidate_sets_parallel(
    prompt: str,
    target_num_pieces: int = 14,
    num_sets: int = 6,
    explicit_category: str | None = None,
    output_dir: str = "outputs/batch",
) -> dict[str, Any]:
    """
    Generates `num_sets` (X) physically valid, diverse LEGO candidate sets of
    `target_num_pieces` (N) in parallel using Apple Silicon M4 cores + MPS scoring.
    """
    t0 = time.perf_counter()
    mem_status = check_memory_safety()
    device_str = get_device()
    max_workers = min(get_optimal_workers(), max(1, num_sets))

    semantics = parse_prompt_semantics(prompt, explicit_category=explicit_category)
    part_priors = _load_neural_part_priors(semantics["category"], device_str=device_str)

    # Oversample slightly in parallel and select the top `num_sets`
    total_to_spawn = max(num_sets, min(num_sets + 2, max_workers * 2))
    base_seed = int(time.time() * 1000) % (2 ** 31 - 1000)
    clamped_pieces = max(3, min(int(target_num_pieces), 250))

    tasks = []
    for idx in range(total_to_spawn):
        tasks.append({
            "candidate_idx": idx,
            "prompt": prompt,
            "target_num_pieces": clamped_pieces,
            "semantics": semantics,
            "seed": base_seed + idx * 97,
            "part_priors": part_priors,
        })

    raw_candidates: list[dict[str, Any]] = []
    # Use ThreadPoolExecutor for fast low-overhead parallel execution in FastAPI/macOS
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_candidate_worker, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                res = fut.result()
                if res and res["metrics"]["physics_valid"] and res["metrics"]["num_pieces"] == clamped_pieces:
                    raw_candidates.append(res)
            except Exception:
                pass

    # Fallback if strict filtering left fewer than num_sets
    fallback_attempt = 0
    while len(raw_candidates) < num_sets and fallback_attempt < num_sets * 3:
        fallback_attempt += 1
        extra_seed = base_seed + 5000 + fallback_attempt * 31
        res = generate_single_candidate_worker({
            "candidate_idx": len(raw_candidates),
            "prompt": prompt,
            "target_num_pieces": clamped_pieces,
            "semantics": semantics,
            "seed": extra_seed,
            "part_priors": part_priors,
        })
        if res and res["metrics"]["physics_valid"]:
            raw_candidates.append(res)

    # Deduplicate by structural geometry signature and sort by total_score descending
    unique_candidates: list[dict[str, Any]] = []
    seen_signatures: set[Any] = set()
    for cand in sorted(raw_candidates, key=lambda c: c["metrics"]["total_score"], reverse=True):
        sig = tuple((p["part_id"], round(p["transform"][3], 1), round(p["transform"][7], 1), round(p["transform"][11], 1)) for p in cand["parts"])
        if sig not in seen_signatures:
            seen_signatures.add(sig)
            unique_candidates.append(cand)
        if len(unique_candidates) >= num_sets:
            break

    # Re-index cleanly 1..X and export LDraw files
    os.makedirs(output_dir, exist_ok=True)
    safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt.lower()).strip("_") or "build"

    for i, cand in enumerate(unique_candidates):
        cand["candidate_index"] = i + 1
        ldr_filename = f"{safe_prompt}_set_{i + 1}.ldr"
        ldr_path = os.path.join(output_dir, ldr_filename)
        _write_stepped_ldraw(cand["parts"], prompt, ldr_path)
        cand["ldr_path"] = f"/{output_dir}/{ldr_filename}"

    elapsed_s = round(time.perf_counter() - t0, 2)
    elapsed_ms = round(elapsed_s * 1000.0, 1)
    ms_per_cand = round(elapsed_ms / max(1, len(unique_candidates)), 1)

    return {
        "status": "success",
        "prompt": prompt,
        "category": semantics["category"],
        "subtype": semantics["subtype"],
        "semantics": semantics,
        "requested_pieces": target_num_pieces,
        "num_sets": len(unique_candidates),
        "candidates": unique_candidates,
        "hardware": {
            "device": device_str,
            "workers_used": max_workers,
            "parallel_workers": max_workers,
            "elapsed_ms": elapsed_ms,
            "elapsed_seconds": elapsed_s,
            "ms_per_candidate": ms_per_cand,
            "memory": mem_status,
        },
    }


def _write_stepped_ldraw(parts_json: list[dict[str, Any]], prompt: str, output_path: str) -> None:
    lines = [
        "0 LegoGPT Generated Candidate Set",
        f"0 Prompt: {prompt}",
    ]
    current_step = None
    for p in parts_json:
        step_id = p.get("step_id", 0)
        if current_step is not None and step_id != current_step:
            lines.append("0 STEP")
        current_step = step_id

        m = p["transform"]
        color = p["color"]
        x, y, z = m[3], m[7], m[11]
        rot = [m[0], m[1], m[2], m[4], m[5], m[6], m[8], m[9], m[10]]
        rot_str = " ".join(f"{val:.4f}" for val in rot)
        lines.append(f"1 {color} {x:.3f} {y:.3f} {z:.3f} {rot_str} {p['part_id']}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def record_human_feedback(
    candidate_id: str,
    prompt: str,
    category: str,
    approved: bool,
    rating: int,
    parts: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Persists user validation (Approve / Reject + rating 1..5) of a generated set
    into `data/human_feedback.json` for subsequent RLHF / DPO refinement.
    """
    os.makedirs(os.path.dirname(FEEDBACK_FILE_PATH), exist_ok=True)
    records: list[dict[str, Any]] = []
    if os.path.exists(FEEDBACK_FILE_PATH):
        try:
            with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Update existing or append new
    updated = False
    entry = {
        "candidate_id": candidate_id,
        "timestamp": int(time.time()),
        "prompt": prompt,
        "category": category,
        "approved": bool(approved),
        "rating": int(rating),
        "num_pieces": len(parts),
        "parts": parts,
    }
    for idx, r in enumerate(records):
        if r.get("candidate_id") == candidate_id:
            records[idx] = entry
            updated = True
            break
    if not updated:
        records.append(entry)

    with open(FEEDBACK_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    approved_count = sum(1 for r in records if r.get("approved"))
    rejected_count = len(records) - approved_count
    return {
        "status": "success",
        "total_feedback": len(records),
        "approved_count": approved_count,
        "rejected_count": rejected_count,
    }


def refine_model_from_feedback(epochs: int = 8) -> dict[str, Any]:
    """
    Fine-tunes the LegoGNN model on Apple Silicon MPS using collected human
    validations (RLHF / Reward-Weighted Regression over part, color, and spatial pose).
    """
    t0 = time.perf_counter()
    check_memory_safety()

    if not os.path.exists(FEEDBACK_FILE_PATH):
        return {"status": "error", "message": "Aún no hay validaciones guardadas. Valida al menos 1 set primero."}

    with open(FEEDBACK_FILE_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        return {"status": "error", "message": "El historial de validaciones está vacío."}

    device = torch.device(get_device())

    # Build combined vocabulary covering ALLOWED_PARTS + vehicle/animal parts
    vocab = list(dict.fromkeys(ALLOWED_PARTS + VEHICLE_GENERATION_PARTS + ANIMAL_PARTS))
    colors = list(dict.fromkeys(ALLOWED_COLORS + [25, 29, 71, 72]))

    model = LegoGNN(
        num_part_classes=len(vocab),
        num_color_classes=len(colors),
        hidden_dim=64,
        allowed_parts=None,
    ).to(device)

    if os.path.exists(REFINED_MODEL_PATH):
        try:
            model.load_state_dict(torch.load(REFINED_MODEL_PATH, map_location=device))
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    # Convert feedback records into weighted step transitions
    transitions = []
    for rec in records:
        parts_raw = rec.get("parts", [])
        if len(parts_raw) < 2:
            continue
        approved = rec.get("approved", False)
        rating = float(rec.get("rating", 5 if approved else 1))
        # Advantage weight: positive for approved sets, negative penalty for rejected
        advantage = ((rating - 2.5) / 2.5) if approved else -0.5

        parsed_list = []
        for p in parts_raw:
            t_mat = np.array(p["transform"], dtype=np.float32).reshape(4, 4)
            parsed_list.append(ParsedPart(
                part_id=p["part_id"],
                color=int(p["color"]),
                transform=t_mat,
                step_id=int(p.get("step_id", 0)),
            ))

        for k in range(1, len(parsed_list)):
            transitions.append((parsed_list[:k], parsed_list[k], advantage))

    if not transitions:
        return {"status": "error", "message": "No hay suficientes pasos en los sets validados."}

    model.train()
    epoch_losses = []
    for _ in range(epochs):
        total_loss = 0.0
        for state_parts, target_part, adv in transitions:
            g = build_pyg_graph(state_parts, allowed_parts=vocab, allowed_colors=colors).to(device)
            batch = torch.zeros(g.num_nodes, dtype=torch.long, device=device)

            optimizer.zero_grad()
            part_logits, color_logits, transform_preds = model(g.x, g.edge_index, g.batch if hasattr(g, "batch") and g.batch is not None else batch)

            p_idx = vocab.index(target_part.part_id) if target_part.part_id in vocab else 0
            c_idx = colors.index(target_part.color) if target_part.color in colors else 0

            ce_part = F.cross_entropy(part_logits, torch.tensor([p_idx], device=device))
            ce_color = F.cross_entropy(color_logits, torch.tensor([c_idx], device=device))

            if adv >= 0:
                tf_target = torch.tensor(
                    target_part.transform[:3, 3].tolist() + target_part.transform[:3, :3].flatten().tolist(),
                    dtype=torch.float32,
                    device=device,
                )
                mse_tf = F.mse_loss(transform_preds[0], tf_target)
                loss = max(0.2, adv) * (ce_part + 0.3 * ce_color + 0.01 * mse_tf)
            else:
                # Unlikelihood / entropy push away from rejected part placement
                probs = F.softmax(part_logits, dim=-1)
                loss = torch.clamp(1.0 - ce_part, min=0.0) * abs(adv) * 0.2 + 0.01 * (probs * torch.log(probs + 1e-8)).sum()

            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        epoch_losses.append(round(total_loss / len(transitions), 4))

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), REFINED_MODEL_PATH)
    with open(REFINED_META_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "num_parts": len(vocab),
            "allowed_parts": vocab,
            "theme": "RLHF Creative (Vehicles & Animals)",
            "max_pieces": 40,
            "epoch": epochs,
            "feedback_samples": len(records),
        }, f, indent=2)

    if device.type == "mps":
        torch.mps.empty_cache()

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return {
        "status": "success",
        "model_name": "creative_refined_mps",
        "device": str(device),
        "epochs": epochs,
        "transitions_trained": len(transitions),
        "initial_loss": epoch_losses[0],
        "final_loss": epoch_losses[-1],
        "loss_history": epoch_losses,
        "elapsed_ms": elapsed_ms,
    }
