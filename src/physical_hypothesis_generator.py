import os
import math
import concurrent.futures
import numpy as np
from src.apple_silicon import workers_for
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from scipy.spatial import cKDTree
from src.ldraw_geometry import LDrawGeometryResolver
from src.solid_collision_validator import SolidCollisionValidator


@dataclass
class ConnectionHypothesis:
    """Represents a physically valid placement hypothesis for a LEGO part."""
    child_part_id: str
    child_element_id: str
    rotation_matrix: np.ndarray             # 3x3 rotation matrix
    translation: np.ndarray                 # 3D translation [X, Y, Z] in LDU
    absolute_transform: np.ndarray          # 4x4 affine matrix
    parent_part_index: int
    matched_studs: List[Tuple[float, float, float]] = field(default_factory=list)
    stud_count: int = 0
    world_bbox: List[float] = field(default_factory=list) # [xmin, ymin, zmin, xmax, ymax, zmax]
    
    # Connector Ontology & Port-to-Port attributes
    connector_family: str = "STUD"
    primary_parent_port: Optional[int] = None
    primary_child_port: Optional[int] = None
    matched_port_pairs: List[Tuple[int, int]] = field(default_factory=list)
    
    # 3D Symmetry & Equivalence attributes
    yaw_deg: float = 0.0
    is_symmetric: bool = False
    symmetry_group: str = "C1"
    symmetry_order: int = 1
    equivalent_rotations_deg: List[float] = field(default_factory=list)
    
    # Visual & Topological scoring attributes
    color_id: int = 16
    color_score: float = 0.0
    iou_score: float = 0.0
    edge_score: float = 0.0
    stud_occlusion_score: float = 0.0
    orientation_score: float = 0.0
    neighbor_score: float = 0.0
    visibility_ratio: float = 1.0
    bridged_parents: int = 1
    is_symmetric_twin: bool = False
    total_score: float = 0.0

    # Infallible Vector Validation attributes
    is_infallible_match: bool = False
    vector_confidence: float = 0.0
    vector_edge_score: float = 0.0
    vector_stud_score: float = 0.0


class PhysicalHypothesisGenerator:
    """
    Generates physically valid, collision-free candidate placements (hypotheses)
    for new LEGO parts being attached to the existing assembly graph.
    Enforces:
      1. Stud-to-Socket alignment on standard LDU grid (20 LDU stud pitch, 8 LDU plate height).
      2. Zero volumetric collision (SAT 3D OBB intersection test).
      3. Stud exclusivity (cannot cover studs already locked by prior parts).
      4. Ground support constraint (Y <= 0.0 in standard coordinates).
    """
    def __init__(self, geom_resolver: LDrawGeometryResolver):
        self.geom = geom_resolver
        self.solid_validator = SolidCollisionValidator(self.geom)

    @staticmethod
    def get_canonical_rotations_y() -> List[Tuple[float, np.ndarray]]:
        """Returns 4 canonical rotations around Y axis: 0°, 90°, 180°, 270°."""
        rotations = []
        for deg in [0.0, 90.0, 180.0, 270.0]:
            rad = math.radians(deg)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            R = np.array([
                [cos_a, 0.0, sin_a],
                [0.0, 1.0, 0.0],
                [-sin_a, 0.0, cos_a]
            ], dtype=np.float32)
            rotations.append((deg, R))
        return rotations

    @staticmethod
    def get_canonical_rotations_3d() -> List[Tuple[float, np.ndarray]]:
        """Returns 24 canonical orthogonal 3D rotations in SO(3) formed by 90-degree increments."""
        dirs = [
            np.array([1, 0, 0], dtype=np.float32),
            np.array([-1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, -1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
            np.array([0, 0, -1], dtype=np.float32)
        ]
        rotations = []
        seen = set()
        for x_dir in dirs:
            for y_dir in dirs:
                if abs(float(np.dot(x_dir, y_dir))) > 0.1:
                    continue
                z_dir = np.cross(x_dir, y_dir)
                R = np.column_stack([x_dir, y_dir, z_dir])
                if float(np.linalg.det(R)) > 0.9:
                    key = tuple(np.round(R.flatten(), 2))
                    if key not in seen:
                        seen.add(key)
                        yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
                        rotations.append((yaw, R))
        return rotations

    @staticmethod
    def get_hinge_rotations_3d() -> List[Tuple[float, np.ndarray]]:
        """
        Returns continuous angle rotations (at 15° increments) for hinge, click-hinge,
        and turntable assemblies to resolve non-orthogonal articulation angles.
        """
        base_3d = PhysicalHypothesisGenerator.get_canonical_rotations_3d()
        hinge_rotations = list(base_3d)
        seen = {tuple(np.round(R.flatten(), 2)) for _, R in base_3d}

        # For each canonical 3D rotation, generate continuous hinge sweeps around local X and Y axes
        for deg_orig, R_base in base_3d:
            for sweep_deg in [-75.0, -60.0, -45.0, -30.0, -15.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
                rad = math.radians(sweep_deg)
                cos_a, sin_a = math.cos(rad), math.sin(rad)
                # Local X rotation (hinge flap flexion)
                R_x = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, cos_a, -sin_a],
                    [0.0, sin_a, cos_a]
                ], dtype=np.float32)
                R_comb = R_base @ R_x
                key = tuple(np.round(R_comb.flatten(), 2))
                if key not in seen:
                    seen.add(key)
                    yaw = float(np.degrees(np.arctan2(R_comb[0, 2], R_comb[2, 2])))
                    hinge_rotations.append((yaw, R_comb))

        return hinge_rotations

    def compute_world_bbox(self, local_bbox: List[float], R: np.ndarray, t: np.ndarray) -> List[float]:
        """Transforms 3D axis-aligned bounding box to world coordinates."""
        x0, y0, z0, x1, y1, z1 = local_bbox
        corners = np.array([
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
        ], dtype=np.float32)
        world_corners = (corners @ R.T) + t
        w_min = np.min(world_corners, axis=0)
        w_max = np.max(world_corners, axis=0)
        return [float(w_min[0]), float(w_min[1]), float(w_min[2]),
                float(w_max[0]), float(w_max[1]), float(w_max[2])]

    @staticmethod
    def check_bbox_overlap(b1: List[float], b2: List[float], margin: float = 1.0) -> bool:
        """Returns True if two axis-aligned bounding boxes overlap by more than margin LDU."""
        if b1[3] - margin <= b2[0] or b2[3] - margin <= b1[0]:
            return False
        if b1[4] - margin <= b2[1] or b2[4] - margin <= b1[1]:
            return False
        if b1[5] - margin <= b2[2] or b2[5] - margin <= b1[2]:
            return False
        return True

    def generate_candidate_hypotheses(
        self,
        child_part_id: str,
        child_element_id: str,
        existing_nodes: List[Dict[str, Any]],
        target_parent_indices: Optional[List[int]] = None,
        is_flip: bool = False,
        color_id: int = 16,
        connector_2d_filter: Optional[Dict[str, Any]] = None,
        lazy_collision_check: bool = False,
        force_3d_rotations: bool = False,
    ) -> List[ConnectionHypothesis]:
        """
        Generates all non-colliding, mechanically aligned placement hypotheses
        for child_part on existing_nodes.
        Supports:
          - Phase C 2D connector pruning (`connector_2d_filter`)
          - Multi-parent structural gap bridging (`bridged_parents`)
          - Lazy 3D SAT collision validation (`lazy_collision_check`)
        """
        child_meta = self.geom.get_part_meta(child_part_id)
        child_sockets = list(child_meta.get("inferred_sockets", child_meta.get("sockets", [])))
        child_holes = child_meta.get("holes", [])
        child_bbox = child_meta["bbox"]

        # Also include bottom/lateral sockets from connectivity_db for inverted slopes, brackets, and SNOT parts
        try:
            from src.connectivity_db import get_part_ports, PortType
            for cp in get_part_ports(child_part_id):
                if cp.port_type in (PortType.ANTISTUD, PortType.PIN_HOLE, PortType.AXLE_HOLE, PortType.CLIP):
                    if not any(np.linalg.norm(cp.local_pos - (cs["pos"] if isinstance(cs, dict) else np.asarray(cs))) < 2.0 for cs in child_sockets):
                        child_sockets.append(cp.local_pos.copy())
        except Exception:
            pass

        # Normalize child_sockets to list of 3D numpy arrays
        norm_child_sockets = []
        for cs in child_sockets:
            pos_arr = cs["pos"].copy() if isinstance(cs, dict) else np.array(cs, dtype=np.float32)
            if not any(np.linalg.norm(pos_arr - ex) < 2.0 for ex in norm_child_sockets):
                norm_child_sockets.append(pos_arr)
        child_sockets = norm_child_sockets

        # For Technic perforated parts (beams, liftarms) or parts with pinholes,
        # incorporate hole centers as valid docking sockets
        if child_holes:
            dedup_child_holes = []
            for h in child_holes:
                hp = h["pos"].copy()
                hp[1] = 0.0  # beam midplane
                if not any(np.linalg.norm(hp - dh) < 2.0 for dh in dedup_child_holes):
                    dedup_child_holes.append(hp)
            if not child_sockets or len(dedup_child_holes) >= len(child_sockets):
                for dh in dedup_child_holes:
                    if not any(np.linalg.norm(dh - cs) < 2.0 for cs in child_sockets):
                        child_sockets.append(dh)

        if not child_sockets:
            # Check if elongated part (axle, bar, pin) with no explicit socket definitions
            dx = child_bbox[3] - child_bbox[0]
            dy = child_bbox[4] - child_bbox[1]
            dz = child_bbox[5] - child_bbox[2]
            max_d = max(dx, dy, dz)
            if max_d >= 20.0:
                child_sockets = []
                axis_idx = 0 if max_d == dx else (1 if max_d == dy else 2)
                num_steps = max(1, int(round(max_d / 20.0)))
                half_steps = num_steps // 2
                for s_i in range(-half_steps, half_steps + 1):
                    pt = np.zeros(3, dtype=np.float32)
                    pt[axis_idx] = float(s_i * 20.0)
                    child_sockets.append(pt)
            else:
                child_sockets = [np.array([0.0, child_bbox[4], 0.0], dtype=np.float32)]

        # Determine target parent nodes (Recency-First ordering)
        candidate_parents = []
        for n in reversed(existing_nodes):
            p_idx = n["part_index"]
            if target_parent_indices is not None and p_idx not in target_parent_indices:
                continue
            candidate_parents.append(n)

        if not candidate_parents:
            return []

        # Collect occupied studs from prior connections
        occupied_stud_positions = []
        for n in existing_nodes:
            for conn in n.get("connections", []):
                for cp in conn.get("contact_points", []):
                    occupied_stud_positions.append(np.array(cp, dtype=np.float32))

        occ_tree = cKDTree(np.array(occupied_stud_positions, dtype=np.float32)) if occupied_stud_positions else None

        # Precompute existing world AABBs for fast core overlap rejection when lazy_collision_check=True
        existing_world_bboxes = []
        all_free_world_studs: List[Tuple[int, np.ndarray]] = []
        for n in existing_nodes:
            n_meta = self.geom.get_part_meta(n["part_id"])
            n_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
            n_R, n_t = n_tf[:3, :3], n_tf[:3, 3]
            existing_world_bboxes.append(self.compute_world_bbox(n_meta["bbox"], n_R, n_t))
            n_studs = n_meta.get("stud_details") or n_meta.get("studs", [])
            for s in n_studs:
                sp = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
                sw = (sp @ n_R.T) + n_t
                if occ_tree is None or len(occ_tree.query_ball_point(sw, r=5.0)) == 0:
                    all_free_world_studs.append((n["part_index"], sw))

        free_studs_tree = cKDTree(np.array([s[1] for s in all_free_world_studs], dtype=np.float32)) if all_free_world_studs else None

        # Phase C: Optional 2D connector pruning helper
        child_span_ldu = float(max(
            child_bbox[3] - child_bbox[0],
            child_bbox[4] - child_bbox[1],
            child_bbox[5] - child_bbox[2],
            40.0
        ))

        def _passes_2d_connector_filter(pt_3d: np.ndarray) -> bool:
            if not connector_2d_filter:
                return True
            roi_bbox = connector_2d_filter.get("roi_bbox_2d")
            if roi_bbox is None:
                return True
            scale = float(connector_2d_filter.get("scale", 2.65))
            ox, oy = connector_2d_filter.get("offset", (200.0, 200.0))
            az = math.radians(float(connector_2d_filter.get("az_deg", 45.0)))
            el = math.radians(float(connector_2d_filter.get("el_deg", 30.0)))
            c3d = connector_2d_filter.get("center_3d")
            p = pt_3d - c3d if c3d is not None else pt_3d
            X = p[0] * math.cos(az) + p[2] * math.sin(az)
            Z = -p[0] * math.sin(az) + p[2] * math.cos(az)
            Y_proj = p[1] * math.cos(el) - Z * math.sin(el)
            u = scale * X + ox
            v = scale * Y_proj + oy
            margin_px = max(45.0, (child_span_ldu * 0.85) * scale)
            x0, y0, x1, y1 = roi_bbox
            return (x0 - margin_px <= u <= x1 + margin_px) and (y0 - margin_px <= v <= y1 + margin_px)

        hypotheses = []
        seen_transforms = set()
        
        hinge_keywords = ["hinge", "2429", "2430", "4315", "3937", "3938", "3830", "3831", "44301", "44302", "3680", "3679", "2452", "2451"]
        is_technic = any("axle" in child_part_id or "pin" in child_part_id or "beam" in child_part_id for _ in [0])
        child_is_hinge = any(k in child_part_id for k in hinge_keywords)
        child_has_lateral = child_meta.get("has_lateral_ports", False)
        is_snot_brick_upright = ("87087" in child_part_id or "4070" in child_part_id) and not any(
            self.geom.get_part_meta(n["part_id"]).get("has_lateral_ports", False) for n in candidate_parents[:4]
        )

        rot_y_4 = self.get_canonical_rotations_y()
        rot_3d_24 = self.get_canonical_rotations_3d()
        has_snot = is_technic or child_is_hinge or child_has_lateral

        def _process_single_parent(parent_node: Dict[str, Any]) -> List[ConnectionHypothesis]:
            p_idx = parent_node["part_index"]
            p_meta = self.geom.get_part_meta(parent_node["part_id"])
            p_tf = np.array(parent_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
            p_R = p_tf[:3, :3]
            p_t = p_tf[:3, 3]

            parent_is_hinge = any(k in parent_node["part_id"] for k in hinge_keywords)
            parent_is_tilted = abs(abs(float(p_R[1, 1])) - 1.0) > 0.1
            parent_has_snot = p_meta.get("has_lateral_ports", False) or parent_is_tilted or parent_is_hinge
            pair_has_snot = (is_technic or child_is_hinge or child_has_lateral or parent_has_snot or force_3d_rotations) and not is_snot_brick_upright

            # Parent available connection ports (studs, lateral studs, or technic holes/sockets)
            parent_studs = []
            stud_source = p_meta.get("stud_details") or p_meta.get("studs", [])
            for s in stud_source:
                if isinstance(s, dict):
                    s_pos = s["pos"]
                    s_norm = s.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32))
                else:
                    s_pos = np.array(s, dtype=np.float32)
                    s_norm = np.array([0.0, -1.0, 0.0], dtype=np.float32)

                s_world_pos = (s_pos @ p_R.T) + p_t
                s_world_norm = p_R @ s_norm

                # Check if stud is facing child insertion direction (or lateral if SNOT)
                target_norm_y = 1.0 if is_flip else -1.0
                is_valid_dir = abs(s_world_norm[1] - target_norm_y) < 0.2
                if pair_has_snot and not is_valid_dir:
                    if abs(s_world_norm[0]) > 0.6 or abs(s_world_norm[2]) > 0.6 or abs(s_world_norm[1]) > 0.6:
                        is_valid_dir = True

                if is_valid_dir:
                    is_occ = False
                    for occ in occupied_stud_positions:
                        if np.linalg.norm(s_world_pos - occ) < 5.0:
                            is_occ = True
                            break
                    if not is_occ and _passes_2d_connector_filter(s_world_pos):
                        parent_studs.append(s_world_pos)

            # If parent has pins (e.g. Technic pins 6558, 2780, 32054), add both origin and pin engagement centers
            p_pins = p_meta.get("pins", [])
            is_pin_parent = any(k in parent_node["part_id"] for k in ["pin", "6558", "2780", "43093", "3673", "32054"])
            if is_pin_parent:
                if _passes_2d_connector_filter(p_t):
                    parent_studs.append(p_t)
                for pin in p_pins:
                    pin_w = (pin["pos"] @ p_R.T) + p_t
                    if _passes_2d_connector_filter(pin_w):
                        parent_studs.append(pin_w)
            elif p_pins:
                for pin in p_pins:
                    pin_w = (pin["pos"] @ p_R.T) + p_t
                    if _passes_2d_connector_filter(pin_w):
                        parent_studs.append(pin_w)

            # Augment with socket/hole ports when SNOT/Technic/inverted or when parent_studs is empty
            if (not parent_studs or pair_has_snot) and (p_meta.get("sockets") or p_meta.get("holes")):
                hole_sources = p_meta.get("holes") or p_meta.get("sockets", [])
                for s in hole_sources:
                    s_pos = s["pos"].copy() if isinstance(s, dict) else np.array(s, dtype=np.float32)
                    if isinstance(s, dict) and "axis" in s:
                        s_pos[1] = 0.0
                    s_world_pos = (s_pos @ p_R.T) + p_t
                    if _passes_2d_connector_filter(s_world_pos):
                        if not any(np.linalg.norm(s_world_pos - ps) < 2.0 for ps in parent_studs):
                            parent_studs.append(s_world_pos)

            # Always augment with lateral/special ports from connectivity_db when SNOT or empty
            if not parent_studs or pair_has_snot:
                try:
                    from src.connectivity_db import get_part_ports
                    for cp in get_part_ports(parent_node["part_id"]):
                        cp_w = (cp.local_pos @ p_R.T) + p_t
                        if _passes_2d_connector_filter(cp_w):
                            if not any(np.linalg.norm(cp_w - ps) < 2.0 for ps in parent_studs):
                                parent_studs.append(cp_w)
                except Exception:
                    pass

            # Fractional sub-grid offsets for brackets (4 LDU) and half-stud jumpers (10 LDU) ONLY when explicit
            is_bracket = any(k in child_part_id or k in parent_node["part_id"] for k in [
                "bracket", "44728", "99780", "99781", "36840", "36841", "10247", "73825", "2436", "3956", "21712", "32607", "18671"
            ])
            is_jumper = any(k in child_part_id or k in parent_node["part_id"] for k in [
                "jumper", "15573", "3794", "34103", "88646", "15571"
            ])

            frac_deltas = [np.zeros(3, dtype=np.float32)]
            if is_bracket:
                for sgn in [-1.0, 1.0]:
                    frac_deltas.extend([
                        np.array([sgn * 4.0, 0.0, 0.0], dtype=np.float32),
                        np.array([0.0, sgn * 4.0, 0.0], dtype=np.float32),
                        np.array([0.0, 0.0, sgn * 4.0], dtype=np.float32),
                    ])
            elif is_jumper:
                for sgn in [-1.0, 1.0]:
                    frac_deltas.extend([
                        np.array([sgn * 10.0, 0.0, 0.0], dtype=np.float32),
                        np.array([0.0, 0.0, sgn * 10.0], dtype=np.float32),
                    ])

            active_rotations = rot_3d_24 if pair_has_snot else rot_y_4

            local_hyps = []
            local_seen = set()

            # Try combinations of canonical rotations and socket-stud alignments
            if parent_studs:
                for deg, R in active_rotations:
                    # Pre-rotate child sockets for this rotation matrix to eliminate inner-loop matrix ops
                    rot_child_socks = [R @ cs for cs in child_sockets]
                    for c_sock_rot in rot_child_socks:
                        for p_stud in parent_studs:
                            t_base = p_stud - c_sock_rot
                            for delta in frac_deltas:
                                t = t_base + delta

                                rot_key = tuple(int(round(x * 10)) for x in R.flatten())
                                key = (round(t[0], 1), round(t[1], 1), round(t[2], 1), rot_key)
                                if key in local_seen:
                                    continue
                                local_seen.add(key)

                                # Test stud connections using pre-rotated vectors
                                matched_studs = []
                                bridged_parent_ids = {p_idx}
                                for s_rot in rot_child_socks:
                                    s_other_world = s_rot + t
                                    matched_here = False
                                    for ps in parent_studs:
                                        d_stud = float(np.linalg.norm(s_other_world - ps))
                                        if d_stud < 3.0 or (pair_has_snot and (abs(d_stud - 4.0) < 1.2 or abs(d_stud - 10.0) < 1.2)):
                                            matched_studs.append([float(ps[0]), float(ps[1]), float(ps[2])])
                                            matched_here = True
                                            break
                                    if not matched_here and free_studs_tree is not None:
                                        nearby_idxs = free_studs_tree.query_ball_point(s_other_world, r=3.0)
                                        for n_idx in nearby_idxs:
                                            other_p_idx, other_sw = all_free_world_studs[n_idx]
                                            if other_p_idx != p_idx:
                                                matched_studs.append([float(other_sw[0]), float(other_sw[1]), float(other_sw[2])])
                                                bridged_parent_ids.add(other_p_idx)
                                                break

                                if len(matched_studs) == 0:
                                    continue

                                child_w_bbox = self.compute_world_bbox(child_bbox, R, t)

                                # Fast 3D core AABB non-penetration check (margin=4.2 LDU allows 4 LDU stud mating)
                                core_collision = False
                                for eb in existing_world_bboxes:
                                    if self.check_bbox_overlap(child_w_bbox, eb, margin=4.2):
                                        core_collision = True
                                        break
                                if core_collision and lazy_collision_check:
                                    continue

                                tf = np.eye(4, dtype=np.float32)
                                tf[:3, :3] = R
                                tf[:3, 3] = t

                                if not lazy_collision_check:
                                    is_valid_solid, _ = self.solid_validator.validate_new_part_placement(
                                        child_part_id, tf, existing_nodes, penetration_threshold=1.5
                                    )
                                    if not is_valid_solid:
                                        continue

                                local_hyps.append(ConnectionHypothesis(
                                    child_part_id=child_part_id,
                                    child_element_id=child_element_id,
                                    rotation_matrix=R,
                                    translation=t,
                                    absolute_transform=tf,
                                    parent_part_index=p_idx,
                                    matched_studs=matched_studs,
                                    stud_count=len(matched_studs),
                                    world_bbox=child_w_bbox,
                                    yaw_deg=float(deg),
                                    color_id=int(color_id),
                                    bridged_parents=len(bridged_parent_ids),
                                ))

            # Underside Stud-to-Socket Mating: child's top studs docking into parent's bottom sockets
            child_top_studs = [
                (s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32))
                for s in (child_meta.get("stud_details") or child_meta.get("studs", []))
            ]
            parent_bot_sockets = []
            for s in p_meta.get("inferred_sockets", p_meta.get("sockets", [])):
                s_local = s["pos"] if isinstance(s, dict) else np.array(s, dtype=np.float32)
                sw = (s_local @ p_R.T) + p_t
                if _passes_2d_connector_filter(sw):
                    parent_bot_sockets.append(sw)

            if child_top_studs and parent_bot_sockets:
                under_rotations = active_rotations if pair_has_snot else active_rotations[:4]
                for deg, R in under_rotations:
                    rot_top_studs = [R @ cst for cst in child_top_studs]
                    for c_st_rot in rot_top_studs:
                        for p_sk in parent_bot_sockets:
                            t_base = p_sk - c_st_rot
                            for delta in frac_deltas[:5]:
                                t = t_base + delta
                                rot_key = tuple(int(round(x * 10)) for x in R.flatten())
                                key = (round(t[0], 1), round(t[1], 1), round(t[2], 1), rot_key)
                                if key in local_seen:
                                    continue
                                local_seen.add(key)

                                matched_under = []
                                for cst_rot in rot_top_studs:
                                    cst_w = cst_rot + t
                                    for psk_o in parent_bot_sockets:
                                        if float(np.linalg.norm(cst_w - psk_o)) < 3.0:
                                            matched_under.append([float(psk_o[0]), float(psk_o[1]), float(psk_o[2])])
                                            break
                                if not matched_under:
                                    continue

                                child_w_bbox = self.compute_world_bbox(child_bbox, R, t)
                                tf = np.eye(4, dtype=np.float32)
                                tf[:3, :3] = R
                                tf[:3, 3] = t
                                if not lazy_collision_check:
                                    is_valid_solid, _ = self.solid_validator.validate_new_part_placement(
                                        child_part_id, tf, existing_nodes, penetration_threshold=1.5
                                    )
                                    if not is_valid_solid:
                                        continue

                                local_hyps.append(ConnectionHypothesis(
                                    child_part_id=child_part_id,
                                    child_element_id=child_element_id,
                                    rotation_matrix=R,
                                    translation=t,
                                    absolute_transform=tf,
                                    parent_part_index=p_idx,
                                    matched_studs=matched_under,
                                    stud_count=len(matched_under),
                                    world_bbox=child_w_bbox,
                                    connector_family="UNDERSIDE_STUD_SOCKET",
                                    yaw_deg=float(deg),
                                    color_id=int(color_id),
                                    bridged_parents=1,
                                ))

            return local_hyps

        # Sequential parent iteration using precomputed cKDTree (avoids GIL contention in nested threads)
        for parent_node in candidate_parents:
            p_hyps = _process_single_parent(parent_node)
            for h in p_hyps:
                rot_key = tuple(int(round(x * 10)) for x in h.rotation_matrix.flatten())
                key = (round(h.translation[0], 1), round(h.translation[1], 1), round(h.translation[2], 1), rot_key)
                if key not in seen_transforms:
                    seen_transforms.add(key)
                    hypotheses.append(h)

        # 3D Symmetry Grouping & Canonicalization:
        sym_order = child_meta.get("symmetry_order", 1)
        mod_deg = 360.0 if sym_order == 1 else (180.0 if sym_order == 2 else 90.0)

        from collections import defaultdict
        grouped = defaultdict(list)
        for h in hypotheses:
            p_studs_key = tuple(sorted((round(s[0], 1), round(s[1], 1), round(s[2], 1)) for s in h.matched_studs))
            bbox_key = tuple(round(v, 1) for v in h.world_bbox)
            rot_key = tuple(int(round(x * 10)) for x in h.rotation_matrix.flatten()) if has_snot else round(h.yaw_deg % mod_deg, 1)
            grouped[(h.parent_part_index, p_studs_key, bbox_key, rot_key)].append(h)

        canonical_hypotheses = []
        for eq_key, group in grouped.items():
            canon = min(group, key=lambda cand: cand.yaw_deg)
            equiv_angles = sorted(set(cand.yaw_deg for cand in group))

            canon.is_symmetric = len(equiv_angles) > 1 or child_meta.get("is_symmetric", False)
            canon.symmetry_order = len(equiv_angles)
            canon.symmetry_group = f"C{len(equiv_angles)}"
            canon.equivalent_rotations_deg = equiv_angles

            canonical_hypotheses.append(canon)

        # Sort hypotheses by bridged_parents, number of locked studs descending, and parent recency
        canonical_hypotheses.sort(key=lambda h: (h.bridged_parents, h.stud_count, h.parent_part_index), reverse=True)
        return canonical_hypotheses

    def filter_valid_solid_hypotheses(
        self,
        hypotheses: List[ConnectionHypothesis],
        existing_nodes: List[Dict[str, Any]],
        max_keep: int = 80,
        penetration_threshold: float = 1.5,
        inside_pool: bool = False,          # ← NEW: evitar oversubscription
    ) -> List[ConnectionHypothesis]:
        """Level 1 Hard Physical Validation (Lazy 3D SAT Collision).

        Args:
            inside_pool: Pasar True cuando este método se invoca desde dentro de un
                         ThreadPoolExecutor/ProcessPoolExecutor externo (beam search branches,
                         scoring paralelo…), para evitar oversubscription en M4 Pro.
        """
        if not hypotheses or not existing_nodes:
            return hypotheses[:max_keep]

        # Mejora #2: respetar presupuesto de P-cores; si ya estamos en un pool externo
        # degradamos a secuencial (inside_pool=True → workers=1).
        max_workers = workers_for(len(hypotheses), inside_pool=inside_pool)

        if len(hypotheses) <= 2 or max_workers <= 1:
            valid_list = []
            for h in hypotheses:
                is_valid, _ = self.solid_validator.validate_new_part_placement(
                    h.child_part_id, h.absolute_transform, existing_nodes, penetration_threshold=penetration_threshold
                )
                if is_valid:
                    valid_list.append(h)
                    if len(valid_list) >= max_keep:
                        break
            return valid_list

        def _check(h):
            is_valid, _ = self.solid_validator.validate_new_part_placement(
                h.child_part_id, h.absolute_transform, existing_nodes, penetration_threshold=penetration_threshold
            )
            return h if is_valid else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(_check, hypotheses)
            valid_list = [h for h in results if h is not None]
        return valid_list[:max_keep]

    def generate_symmetric_twin_hypothesis(
        self,
        first_hyp: ConnectionHypothesis,
        child_part_id: str,
        child_element_id: str,
        existing_nodes: List[Dict[str, Any]],
        color_id: int = 16,
    ) -> List[ConnectionHypothesis]:
        """
        Special Case 2: Hidden / Paired Bilateral Symmetry Reflection (2x identical or L/R mirror parts).
        Given a high-confidence placement `first_hyp` on the visible side of the assembly,
        computes the dominant symmetry planes (X = X_sym and Z = Z_sym) of G_{t-1} and generates
        the mirror-reflected candidate on the opposite side, validating physical non-collision.
        """
        if not existing_nodes:
            return []
        positions = np.array(
            [np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))[:3, 3] for n in existing_nodes],
            dtype=np.float32,
        )
        sym_x = float(np.median(positions[:, 0]))
        sym_z = float(np.median(positions[:, 2]))
        if abs(sym_x) < 12.0:
            sym_x = 0.0
        if abs(sym_z) < 12.0:
            sym_z = 0.0

        child_meta = self.geom.get_part_meta(child_part_id)
        child_bbox = child_meta["bbox"]
        t1 = first_hyp.translation
        R1 = first_hyp.rotation_matrix

        twins: List[ConnectionHypothesis] = []
        # Candidate reflection across X = sym_x and Z = sym_z
        reflections = [
            ("X", np.array([2.0 * sym_x - t1[0], t1[1], t1[2]], dtype=np.float32)),
            ("Z", np.array([t1[0], t1[1], 2.0 * sym_z - t1[2]], dtype=np.float32)),
        ]
        for axis_name, t_mirror in reflections:
            if float(np.linalg.norm(t_mirror - t1)) < 8.0:
                continue
            for yaw_add in (0.0, 180.0, 90.0, 270.0):
                rad = math.radians(yaw_add)
                R_yaw = np.array([
                    [math.cos(rad), 0.0, math.sin(rad)],
                    [0.0, 1.0, 0.0],
                    [-math.sin(rad), 0.0, math.cos(rad)],
                ], dtype=np.float32)
                R_cand = R_yaw @ R1
                tf = np.eye(4, dtype=np.float32)
                tf[:3, :3] = R_cand
                tf[:3, 3] = t_mirror
                is_val, _ = self.solid_validator.validate_new_part_placement(
                    child_part_id, tf, existing_nodes, penetration_threshold=1.5
                )
                if not is_val:
                    continue
                w_bbox = self.compute_world_bbox(child_bbox, R_cand, t_mirror)
                twin_h = ConnectionHypothesis(
                    child_part_id=child_part_id,
                    child_element_id=child_element_id,
                    rotation_matrix=R_cand,
                    translation=t_mirror,
                    absolute_transform=tf,
                    parent_part_index=first_hyp.parent_part_index,
                    matched_studs=list(first_hyp.matched_studs),
                    stud_count=max(1, first_hyp.stud_count),
                    world_bbox=w_bbox,
                    yaw_deg=(first_hyp.yaw_deg + yaw_add) % 360.0,
                    color_id=int(color_id),
                    bridged_parents=first_hyp.bridged_parents,
                    is_symmetric_twin=True,
                )
                twins.append(twin_h)
                break
        return twins

    def generate_virtual_ground_hypotheses(
        self,
        child_part_id: str,
        child_element_id: str,
        existing_nodes: Optional[List[Dict[str, Any]]] = None
    ) -> List[ConnectionHypothesis]:
        """
        Generates candidate hypotheses for floating sub-assemblies or pieces starting on virtual ground
        when no physical stud/pin connection exists to the active assembly DAG.
        Enforces:
          1. Ground level alignment (bottom face at Y = 0.0 or assembly ground level).
          2. Canonical SO(3) rotations.
          3. Volumetric non-collision with existing assembly nodes.
        """
        child_meta = self.geom.get_part_meta(child_part_id)
        child_bbox = child_meta["bbox"]  # [x0, y0, z0, x1, y1, z1]

        ground_y = 0.0
        if existing_nodes:
            max_y_vals = []
            for n in existing_nodes:
                n_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
                p_meta = self.geom.get_part_meta(n["part_id"])
                p_bb = p_meta["bbox"]
                corners = np.array([
                    [p_bb[0], p_bb[1], p_bb[2]], [p_bb[3], p_bb[1], p_bb[2]],
                    [p_bb[0], p_bb[4], p_bb[2]], [p_bb[3], p_bb[4], p_bb[2]],
                    [p_bb[0], p_bb[1], p_bb[5]], [p_bb[3], p_bb[1], p_bb[5]],
                    [p_bb[0], p_bb[4], p_bb[5]], [p_bb[3], p_bb[4], p_bb[5]],
                ], dtype=np.float32)
                corners_w = (corners @ n_tf[:3, :3].T) + n_tf[:3, 3]
                max_y_vals.append(float(np.max(corners_w[:, 1])))
            if max_y_vals:
                ground_y = max(max_y_vals)

        canonical_rotations = self.get_canonical_rotations_y()
        hypotheses = []

        grid_offsets = [
            (0.0, 0.0), (-20.0, 0.0), (20.0, 0.0), (0.0, -20.0), (0.0, 20.0),
            (-40.0, 0.0), (40.0, 0.0), (0.0, -40.0), (0.0, 40.0),
            (-20.0, -20.0), (20.0, -20.0), (-20.0, 20.0), (20.0, 20.0)
        ]

        anchor_points = [(0.0, 0.0)]
        if existing_nodes:
            for n in existing_nodes[-4:]:
                n_pos = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))[:3, 3]
                anchor_points.append((float(n_pos[0]), float(n_pos[2])))

        seen_keys = set()
        for deg, R in canonical_rotations:
            x0, y0, z0, x1, y1, z1 = child_bbox
            corners = np.array([
                [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
            ], dtype=np.float32)
            rot_corners = corners @ R.T
            lowest_y = float(np.max(rot_corners[:, 1]))
            ty = ground_y - lowest_y

            for ax, az in anchor_points:
                for gx, gz in grid_offsets:
                    t = np.array([ax + gx, ty, az + gz], dtype=np.float32)
                    tf = np.eye(4, dtype=np.float32)
                    tf[:3, :3] = R
                    tf[:3, 3] = t

                if existing_nodes:
                    is_valid, _ = self.solid_validator.validate_new_part_placement(
                        child_part_id, tf, existing_nodes, penetration_threshold=1.5
                    )
                    if not is_valid:
                        continue

                child_w_bbox = self.compute_world_bbox(child_bbox, R, t)
                key = (round(t[0], 1), round(t[1], 1), round(t[2], 1), round(deg, 1))
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                hypotheses.append(ConnectionHypothesis(
                    child_part_id=child_part_id,
                    child_element_id=child_element_id,
                    rotation_matrix=R,
                    translation=t,
                    absolute_transform=tf,
                    parent_part_index=-1,
                    matched_studs=[],
                    stud_count=0,
                    world_bbox=child_w_bbox,
                    connector_family="VIRTUAL_GROUND",
                    yaw_deg=float(deg)
                ))

        return hypotheses

    def generate_candidates_from_port_matching(
        self,
        child_part_id: str,
        child_element_id: str,
        occupancy_map,  # ConnectorOccupancyMap
        existing_nodes: List[Dict[str, Any]],
        is_flip: bool = False,
        color_id: int = 16,
        lazy_collision_check: bool = True,
        max_candidates: int = 128,
        connector_2d_filter: Optional[Dict[str, Any]] = None,
    ) -> List[ConnectionHypothesis]:
        """
        Connector-Compatibility Gating + Geographic 2D ROI + Multi-Port Engagement (Paso 3).
        ====================================================================================
        Generates physically valid placement hypotheses by matching typed ports
        of *child_part_id* against FREE typed ports of the current assembly graph
        (from *occupancy_map*), using the COMPATIBLE_PORT_PAIRS ontology and optional
        2D geographic ROI filtering (`connector_2d_filter`), and counts all child ports
        that simultaneously engage free compatible ports (`matched_studs`, `stud_count`,
        `bridged_parents`).
        """
        from src.connectivity_db import PortType, COMPATIBLE_PORT_PAIRS

        _STUD_TYPES = {PortType.STUD, PortType.STUD_HOLLOW}
        _ANTISTUD_TYPES = {PortType.ANTISTUD}

        # --- 1. Get child typed ports ------------------------------------
        child_ports = self.geom.get_part_ports(child_part_id)
        if not child_ports:
            return []

        child_meta = self.geom.get_part_meta(child_part_id)
        child_bbox = child_meta["bbox"]
        part_span_ldu = max(
            abs(child_bbox[3] - child_bbox[0]),
            abs(child_bbox[5] - child_bbox[2]),
            40.0,
        )

        # --- 2. Get compatible FREE graph ports (with optional 2D ROI filter) ---
        free_graph_ports = occupancy_map.get_free_ports(
            filter_compatible_with=child_ports,
            connector_2d_filter=connector_2d_filter,
            part_max_span_ldu=part_span_ldu,
        )
        all_free_graph_ports = occupancy_map.get_free_ports(
            filter_compatible_with=child_ports
        )
        if not free_graph_ports:
            free_graph_ports = all_free_graph_ports
        if not free_graph_ports:
            return []

        # Build fast spatial index over all free graph ports for multi-port simultaneous counting
        all_free_positions = np.array([gp.pos_world for gp in all_free_graph_ports], dtype=np.float32)
        free_kdtree = cKDTree(all_free_positions) if len(all_free_positions) > 0 else None

        # Pre-compute existing world AABBs for fast collision rejection
        existing_world_bboxes: List[List[float]] = []
        for n in existing_nodes:
            n_meta = self.geom.get_part_meta(n["part_id"])
            n_tf = np.array(n["absolute_transform"], dtype=np.float32).reshape(4, 4)
            existing_world_bboxes.append(
                self.compute_world_bbox(n_meta["bbox"], n_tf[:3, :3], n_tf[:3, 3])
            )

        sym_order = child_meta.get("symmetry_order", 1)
        sym_group = child_meta.get("symmetry_group", "C1")
        is_sym = bool(child_meta.get("is_symmetric", sym_order > 1))

        def _sweep_ports(port_list: List[Any]) -> List[ConnectionHypothesis]:
            hyps: List[ConnectionHypothesis] = []
            seen_keys: set = set()

            for graph_crec in port_list:
                g_pos_w = graph_crec.pos_world
                g_norm_w = graph_crec.normal_world
                g_pt = graph_crec.port_type
                parent_idx = graph_crec.owner_part_index

                for child_cp in child_ports:
                    c_pt = child_cp.port_type
                    if (g_pt, c_pt) not in COMPATIBLE_PORT_PAIRS:
                        continue

                    c_norm_local = child_cp.local_normal
                    c_norm_len = float(np.linalg.norm(c_norm_local))
                    if c_norm_len < 1e-5:
                        continue
                    c_norm_local = c_norm_local / c_norm_len

                    target_child_norm_w = -g_norm_w
                    R_base = self._align_vectors(c_norm_local, target_child_norm_w)

                    # Symmetry canonicalization:
                    # C4 symmetric piece needs only [0.0] (all 4 rotations are identical in 3D);
                    # C2 symmetric piece needs [0.0, 90.0];
                    # C1 asymmetric piece needs [0.0, 90.0, 180.0, 270.0].
                    if g_pt in _STUD_TYPES or c_pt in _STUD_TYPES or g_pt in _ANTISTUD_TYPES or c_pt in _ANTISTUD_TYPES:
                        yaw_steps = {
                            4: [0.0],
                            2: [0.0, 90.0],
                            1: [0.0, 90.0, 180.0, 270.0],
                        }.get(sym_order, [0.0, 90.0, 180.0, 270.0])
                    else:
                        yaw_steps = [0.0, 90.0, 180.0, 270.0] if sym_order == 1 else [0.0, 180.0]

                    for yaw_deg in yaw_steps:
                        if abs(yaw_deg) > 0.1:
                            rad = math.radians(yaw_deg)
                            ca, sa = math.cos(rad), math.sin(rad)
                            axis = g_norm_w / (float(np.linalg.norm(g_norm_w)) + 1e-9)
                            ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
                            R_axis = np.array([
                                [ca + ux*ux*(1-ca),    ux*uy*(1-ca)-uz*sa, ux*uz*(1-ca)+uy*sa],
                                [uy*ux*(1-ca)+uz*sa,   ca + uy*uy*(1-ca),  uy*uz*(1-ca)-ux*sa],
                                [uz*ux*(1-ca)-uy*sa,   uz*uy*(1-ca)+ux*sa, ca + uz*uz*(1-ca)],
                            ], dtype=np.float32)
                            R = R_axis @ R_base
                        else:
                            R = R_base.copy()

                        c_pos_w = R @ child_cp.local_pos
                        t = g_pos_w - c_pos_w

                        if g_pt in _STUD_TYPES or c_pt in _ANTISTUD_TYPES or g_pt in _ANTISTUD_TYPES or c_pt in _STUD_TYPES:
                            if abs(float(g_norm_w[1])) > 0.8:
                                t[0] = round(float(t[0]) / 10.0) * 10.0
                                t[1] = round(float(t[1]) / 4.0) * 4.0
                                t[2] = round(float(t[2]) / 10.0) * 10.0
                            else:
                                t = np.round(t / 2.0) * 2.0

                        rot_key = tuple(int(round(x * 10)) for x in R.flatten())
                        key = (round(float(t[0]), 1), round(float(t[1]), 1),
                               round(float(t[2]), 1), rot_key)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        child_w_bbox = self.compute_world_bbox(child_bbox, R, t)
                        core_collision = any(
                            self.check_bbox_overlap(child_w_bbox, eb, margin=4.2)
                            for eb in existing_world_bboxes
                        )
                        if core_collision and lazy_collision_check:
                            continue

                        tf = np.eye(4, dtype=np.float32)
                        tf[:3, :3] = R
                        tf[:3, 3] = t

                        if not lazy_collision_check:
                            is_valid, _ = self.solid_validator.validate_new_part_placement(
                                child_part_id, tf, existing_nodes, penetration_threshold=1.5
                            )
                            if not is_valid:
                                continue

                        # Multi-port simultaneous engagement counting across all compatible child ports
                        matched_pts = []
                        engaged_parents = {parent_idx}
                        if free_kdtree is not None:
                            same_family_child_ports = [
                                cp_other for cp_other in child_ports
                                if cp_other.port_type == c_pt
                                and float(np.dot(cp_other.local_normal, child_cp.local_normal)) > 0.85
                            ]
                            if same_family_child_ports:
                                local_pts = np.array([cp_o.local_pos for cp_o in same_family_child_ports], dtype=np.float32)
                                world_pts = (local_pts @ R.T) + t
                                dists, idxs = free_kdtree.query(world_pts, distance_upper_bound=3.5)
                                for d_val, g_idx in zip(dists, idxs):
                                    if d_val <= 3.5 and 0 <= g_idx < len(all_free_graph_ports):
                                        gp_match = all_free_graph_ports[int(g_idx)]
                                        if (gp_match.port_type, c_pt) in COMPATIBLE_PORT_PAIRS:
                                            if float(np.dot(gp_match.normal_world, g_norm_w)) > 0.85:
                                                mp = [float(gp_match.pos_world[0]), float(gp_match.pos_world[1]), float(gp_match.pos_world[2])]
                                                if mp not in matched_pts:
                                                    matched_pts.append(mp)
                                                engaged_parents.add(gp_match.owner_part_index)

                        if not matched_pts:
                            matched_pts = [[float(g_pos_w[0]), float(g_pos_w[1]), float(g_pos_w[2])]]

                        conn_family = self._port_pair_to_family(g_pt, c_pt)
                        equiv_degs = (
                            [0.0, 90.0, 180.0, 270.0] if sym_order == 4
                            else ([yaw_deg, (yaw_deg + 180.0) % 360.0] if sym_order == 2 else [yaw_deg])
                        )

                        hyps.append(ConnectionHypothesis(
                            child_part_id=child_part_id,
                            child_element_id=child_element_id,
                            rotation_matrix=R,
                            translation=t,
                            absolute_transform=tf,
                            parent_part_index=parent_idx,
                            matched_studs=matched_pts,
                            stud_count=len(matched_pts),
                            world_bbox=child_w_bbox,
                            connector_family=conn_family,
                            primary_parent_port=None,
                            primary_child_port=None,
                            yaw_deg=yaw_deg,
                            is_symmetric=is_sym,
                            symmetry_group=sym_group,
                            symmetry_order=sym_order,
                            equivalent_rotations_deg=equiv_degs,
                            color_id=int(color_id),
                            bridged_parents=len(engaged_parents),
                        ))
            return hyps

        hypotheses = _sweep_ports(free_graph_ports)
        if not hypotheses and len(free_graph_ports) < len(all_free_graph_ports):
            hypotheses = _sweep_ports(all_free_graph_ports)

        # Sort: multi-parent bridging + most port contacts first, then most recent parent
        hypotheses.sort(
            key=lambda h: (h.bridged_parents, h.stud_count, h.parent_part_index), reverse=True
        )
        return hypotheses[:max_candidates]

    # -----------------------------------------------------------------------
    # Helpers for generate_candidates_from_port_matching
    # -----------------------------------------------------------------------

    @staticmethod
    def _align_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
        """
        Returns a 3x3 rotation matrix R such that R @ v_from ≈ v_to.
        Uses Rodrigues' rotation formula; falls back to identity when vectors
        are parallel or anti-parallel.
        """
        v_from = v_from / (float(np.linalg.norm(v_from)) + 1e-9)
        v_to   = v_to   / (float(np.linalg.norm(v_to))   + 1e-9)
        dot = float(np.dot(v_from, v_to))

        if dot >= 0.9999:
            return np.eye(3, dtype=np.float32)
        if dot <= -0.9999:
            perp = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if abs(float(np.dot(v_from, perp))) > 0.9:
                perp = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            axis = np.cross(v_from, perp)
            axis = axis / (float(np.linalg.norm(axis)) + 1e-9)
            R = 2.0 * np.outer(axis, axis).astype(np.float32) - np.eye(3, dtype=np.float32)
            return R

        axis = np.cross(v_from, v_to).astype(np.float32)
        axis_len = float(np.linalg.norm(axis))
        axis = axis / (axis_len + 1e-9)
        angle = math.acos(max(-1.0, min(1.0, dot)))
        c, s = math.cos(angle), math.sin(angle)
        ux, uy, uz = float(axis[0]), float(axis[1]), float(axis[2])
        R = np.array([
            [c + ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s,   c + uy*uy*(1-c),  uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s,   uz*uy*(1-c)+ux*s, c + uz*uz*(1-c)],
        ], dtype=np.float32)
        return R

    @staticmethod
    def _port_pair_to_family(g_pt, c_pt) -> str:
        """Maps a (graph_port_type, child_port_type) pair to a connector family string."""
        from src.connectivity_db import PortType
        if g_pt in (PortType.STUD, PortType.STUD_HOLLOW) and c_pt == PortType.ANTISTUD:
            return "STUD"
        if g_pt == PortType.ANTISTUD and c_pt in (PortType.STUD, PortType.STUD_HOLLOW):
            return "UNDERSIDE_STUD_SOCKET"
        if g_pt == PortType.PIN_HOLE and c_pt == PortType.PIN:
            return "TECHNIC_PIN"
        if g_pt == PortType.AXLE_HOLE and c_pt == PortType.AXLE:
            return "TECHNIC_AXLE"
        if g_pt == PortType.CLIP and c_pt == PortType.BAR:
            return "CLIP_BAR"
        if g_pt == PortType.BALL_SOCKET and c_pt == PortType.BALL:
            return "BALL_JOINT"
        return f"{g_pt.value}_to_{c_pt.value}"
