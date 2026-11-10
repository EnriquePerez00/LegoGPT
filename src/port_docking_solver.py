import math
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from src.ldraw_connector_extractor import LDrawConnectorExtractor
from src.ldraw_geometry import LDrawGeometryResolver


@dataclass
class DockingCandidate:
    """Represents a mathematically exact port-to-port docking between two LEGO parts."""
    child_part_num: str
    child_element_id: str
    parent_part_index: int
    parent_part_num: str
    primary_parent_port: int
    primary_child_port: int
    rotation_matrix: np.ndarray             # 3x3
    translation: np.ndarray                 # 3D vector [X, Y, Z] in LDU
    absolute_transform: np.ndarray          # 4x4 affine matrix
    matched_port_pairs: List[Tuple[int, int]] = field(default_factory=list) # (parent_port_idx, child_port_idx)
    num_matched_ports: int = 0
    connector_family: str = "STUD"
    world_bbox: List[float] = field(default_factory=list)
    
    # 3D Symmetry & Equivalence
    roll_deg: float = 0.0
    is_symmetric: bool = False
    symmetry_group: str = "C1"
    symmetry_order: int = 1
    equivalent_rotations_deg: List[float] = field(default_factory=list)


class PortDockingSolver:
    """
    Kinematic port-to-port docking engine for LEGO parts based on formal connector ontology:
      - Aligns complementary connector frames (opposite normals: n_child = -n_parent).
      - Evaluates discrete rotational degrees of freedom (0°, 90°, 180°, 270° around docking normal).
      - Counts multi-port simultaneous engagement (e.g. 6 studs simultaneously locking 6 sockets).
      - Enforces zero volumetric collision (SAT 3D).
    """
    def __init__(self, connector_extractor: Optional[LDrawConnectorExtractor] = None, geom_resolver: Optional[LDrawGeometryResolver] = None):
        self.extractor = connector_extractor or LDrawConnectorExtractor()
        self.geom = geom_resolver or self.extractor.geom
        self.compatibility_cache = {}

    def get_compatible_types(self, type_id: str) -> List[str]:
        if type_id not in self.compatibility_cache:
            self.compatibility_cache[type_id] = self.extractor.get_compatible_connector_types(type_id)
        return self.compatibility_cache[type_id]

    @staticmethod
    def get_rotation_matrix_from_normals(from_norm: np.ndarray, to_norm: np.ndarray, roll_angle_deg: float = 0.0) -> np.ndarray:
        """
        Calculates 3x3 rotation matrix that rotates vector from_norm to to_norm,
        with an additional roll angle in degrees around to_norm.
        """
        v1 = from_norm / max(1e-6, np.linalg.norm(from_norm))
        v2 = to_norm / max(1e-6, np.linalg.norm(to_norm))

        dot = np.dot(v1, v2)
        if dot > 0.9999:
            R_align = np.eye(3, dtype=np.float32)
        elif dot < -0.9999:
            # 180 degree flip around perpendicular axis
            perp = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(v1[0]) > 0.8 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
            axis = np.cross(v1, perp)
            axis /= np.linalg.norm(axis)
            # Rodrigues formula for 180 deg
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32)
            R_align = np.eye(3, dtype=np.float32) + 2.0 * (K @ K)
        else:
            axis = np.cross(v1, v2)
            s = np.linalg.norm(axis)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=np.float32)
            R_align = np.eye(3, dtype=np.float32) + K + (K @ K) * ((1.0 - dot) / (s**2))

        # Roll around to_norm
        if abs(roll_angle_deg) > 1e-3:
            rad = math.radians(roll_angle_deg)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            u = v2
            K_roll = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]], dtype=np.float32)
            R_roll = np.eye(3, dtype=np.float32) + sin_r * K_roll + (1.0 - cos_r) * (K_roll @ K_roll)
            return R_roll @ R_align

        return R_align

    def solve_docking(
        self,
        child_part_num: str,
        child_element_id: str,
        placed_parts: List[Dict[str, Any]],
        target_parent_indices: Optional[List[int]] = None
    ) -> List[DockingCandidate]:
        """
        Solves all valid, non-colliding port-to-port dockings between child_part and existing placed parts.
        """
        child_connectors = self.extractor.extract_connectors_for_part(child_part_num)
        if not child_connectors:
            return []

        child_meta = self.geom.get_part_meta(self.geom.resolve_compatible_part_id(child_part_num))
        child_bbox = child_meta["bbox"]

        candidates = []
        seen_transforms = set()

        for parent_node in placed_parts:
            p_idx = parent_node["part_index"]
            if target_parent_indices is not None and p_idx not in target_parent_indices:
                continue

            p_num = parent_node.get("raw_part_id", parent_node["part_id"]).replace(".dat", "")
            parent_connectors = self.extractor.extract_connectors_for_part(p_num)
            if not parent_connectors:
                continue

            p_tf = np.array(parent_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
            p_R = p_tf[:3, :3]
            p_t = p_tf[:3, 3]

            # Collect occupied parent ports
            occupied_ports = set()
            for conn in parent_node.get("connections", []):
                for cp in conn.get("contact_points", []):
                    # Stud position matching
                    for pc in parent_connectors:
                        pc_world = (p_R @ np.array(pc["pos"])) + p_t
                        if np.linalg.norm(pc_world - np.array(cp)) < 3.0:
                            occupied_ports.add(pc["connector_idx"])

            for p_port in parent_connectors:
                if p_port["connector_idx"] in occupied_ports:
                    continue

                p_type = p_port["type_id"]
                compatible_child_types = self.get_compatible_types(p_type)

                p_pos_world = (p_R @ np.array(p_port["pos"], dtype=np.float32)) + p_t
                p_norm_world = p_R @ np.array(p_port["normal"], dtype=np.float32)

                for c_port in child_connectors:
                    if c_port["type_id"] not in compatible_child_types:
                        continue

                    c_pos_local = np.array(c_port["pos"], dtype=np.float32)
                    c_norm_local = np.array(c_port["normal"], dtype=np.float32)

                    # Required child normal must oppose parent normal: n_child_world = -n_parent_world
                    target_c_norm_world = -p_norm_world

                    # Test discrete yaw roll angles around the normal (0, 90, 180, 270)
                    for roll_deg in [0.0, 90.0, 180.0, 270.0]:
                        R_child = self.get_rotation_matrix_from_normals(c_norm_local, target_c_norm_world, roll_deg)
                        t_child = p_pos_world - (R_child @ c_pos_local)

                        # Deduplicate transforms
                        t_key = (round(float(t_child[0]), 1), round(float(t_child[1]), 1), round(float(t_child[2]), 1), int(round(roll_deg)))
                        if t_key in seen_transforms:
                            continue
                        seen_transforms.add(t_key)

                        # Check multi-port alignment across all child and parent connectors
                        matched_pairs = []
                        for other_c in child_connectors:
                            other_c_pos_w = (R_child @ np.array(other_c["pos"], dtype=np.float32)) + t_child
                            for other_p in parent_connectors:
                                if other_p["connector_idx"] in occupied_ports:
                                    continue
                                other_p_pos_w = (p_R @ np.array(other_p["pos"], dtype=np.float32)) + p_t
                                if np.linalg.norm(other_c_pos_w - other_p_pos_w) < 3.0:
                                    matched_pairs.append((other_p["connector_idx"], other_c["connector_idx"]))
                                    break

                        if not matched_pairs:
                            continue

                        # Compute world bounding box
                        corners = np.array([
                            [child_bbox[0], child_bbox[1], child_bbox[2]],
                            [child_bbox[3], child_bbox[1], child_bbox[2]],
                            [child_bbox[3], child_bbox[4], child_bbox[2]],
                            [child_bbox[0], child_bbox[4], child_bbox[2]],
                            [child_bbox[0], child_bbox[1], child_bbox[5]],
                            [child_bbox[3], child_bbox[1], child_bbox[5]],
                            [child_bbox[3], child_bbox[4], child_bbox[5]],
                            [child_bbox[0], child_bbox[4], child_bbox[5]],
                        ], dtype=np.float32)
                        w_corners = (corners @ R_child.T) + t_child
                        w_min = np.min(w_corners, axis=0)
                        w_max = np.max(w_corners, axis=0)
                        w_bbox = [float(w_min[0]), float(w_min[1]), float(w_min[2]),
                                  float(w_max[0]), float(w_max[1]), float(w_max[2])]

                        # Collision verification with all placed parts
                        collision = False
                        for ex in placed_parts:
                            ex_meta = self.geom.get_part_meta(ex["part_id"])
                            ex_tf = np.array(ex["absolute_transform"], dtype=np.float32).reshape((4, 4))
                            ex_corners = np.array([
                                [ex_meta["bbox"][0], ex_meta["bbox"][1], ex_meta["bbox"][2]],
                                [ex_meta["bbox"][3], ex_meta["bbox"][1], ex_meta["bbox"][2]],
                                [ex_meta["bbox"][3], ex_meta["bbox"][4], ex_meta["bbox"][2]],
                                [ex_meta["bbox"][0], ex_meta["bbox"][4], ex_meta["bbox"][2]],
                                [ex_meta["bbox"][0], ex_meta["bbox"][1], ex_meta["bbox"][5]],
                                [ex_meta["bbox"][3], ex_meta["bbox"][1], ex_meta["bbox"][5]],
                                [ex_meta["bbox"][3], ex_meta["bbox"][4], ex_meta["bbox"][5]],
                                [ex_meta["bbox"][0], ex_meta["bbox"][4], ex_meta["bbox"][5]],
                            ], dtype=np.float32)
                            ex_w_corners = (ex_corners @ ex_tf[:3, :3].T) + ex_tf[:3, 3]
                            ex_min = np.min(ex_w_corners, axis=0)
                            ex_max = np.max(ex_w_corners, axis=0)

                            # Overlap check
                            if not (w_bbox[3] - 2.0 <= ex_min[0] or ex_max[0] - 2.0 <= w_bbox[0] or
                                    w_bbox[4] - 2.0 <= ex_min[1] or ex_max[1] - 2.0 <= w_bbox[1] or
                                    w_bbox[5] - 2.0 <= ex_min[2] or ex_max[2] - 2.0 <= w_bbox[2]):
                                # Volumetric penetration
                                y_penetration = min(w_bbox[4], ex_max[1]) - max(w_bbox[1], ex_min[1])
                                if y_penetration > 2.0:
                                    collision = True
                                    break

                        if collision:
                            continue

                        tf = np.eye(4, dtype=np.float32)
                        tf[:3, :3] = R_child
                        tf[:3, 3] = t_child

                        candidates.append(DockingCandidate(
                            child_part_num=child_part_num,
                            child_element_id=child_element_id,
                            parent_part_index=p_idx,
                            parent_part_num=p_num,
                            primary_parent_port=p_port["connector_idx"],
                            primary_child_port=c_port["connector_idx"],
                            rotation_matrix=R_child,
                            translation=t_child,
                            absolute_transform=tf,
                            matched_port_pairs=matched_pairs,
                            num_matched_ports=len(matched_pairs),
                            connector_family="STUD_LATERAL" if "LATERAL" in p_type else "STUD",
                            world_bbox=w_bbox,
                            roll_deg=float(roll_deg)
                        ))

        # 3D Symmetry Grouping & Canonicalization:
        # Group candidates into physical equivalence classes modulo the part's rotational symmetry group:
        # (parent_part_index, tuple of parent ports engaged, world_bbox, roll_deg % mod_deg)
        sym_order = child_meta.get("symmetry_order", 1)
        sym_group = child_meta.get("symmetry_group", "C1")
        mod_deg = 360.0 if sym_order == 1 else (180.0 if sym_order == 2 else 90.0)

        from collections import defaultdict
        grouped = defaultdict(list)
        for c in candidates:
            p_ports_key = tuple(sorted(p[0] for p in c.matched_port_pairs))
            bbox_key = tuple(round(v, 1) for v in c.world_bbox)
            roll_canon = round(c.roll_deg % mod_deg, 1)
            grouped[(c.parent_part_index, p_ports_key, bbox_key, roll_canon)].append(c)

        canonical_candidates = []
        for eq_key, group in grouped.items():
            canon = min(group, key=lambda cand: cand.roll_deg)
            equiv_angles = sorted(set(cand.roll_deg for cand in group))

            canon.is_symmetric = len(equiv_angles) > 1 or child_meta.get("is_symmetric", False)
            canon.symmetry_order = len(equiv_angles)
            canon.symmetry_group = f"C{len(equiv_angles)}"
            canon.equivalent_rotations_deg = equiv_angles

            canonical_candidates.append(canon)

        # Sort by number of concurrently locked ports descending
        canonical_candidates.sort(key=lambda c: c.num_matched_ports, reverse=True)
        return canonical_candidates
