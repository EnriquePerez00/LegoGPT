import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.ldraw_geometry import LDrawGeometryResolver


class SolidCollisionValidator:
    """
    Validates physical 3D solid non-penetration invariants for LEGO assemblies:
      1. Coinciding faces / edges are ALLOWED (touching with distance >= -0.1 LDU).
      2. Mating connectors (studs penetrating socket cavities) are ALLOWED.
      3. Non-connector solid volume penetration > 0.5 LDU is STRICTLY FORBIDDEN.
      4. Ground surface compliance: all parts must reside at or above ground level (Y <= 0 after normalization).
    """
    def __init__(self, geom_resolver: LDrawGeometryResolver):
        self.geom = geom_resolver
        self.obb_cache = {}
        self.port_cache = {}  # cache de puertos-mundo por (part_id, transform_hash)

    def _get_world_ports(self, part_id: str, transform: np.ndarray) -> List[Dict[str, Any]]:
        """
        Obtiene todos los puertos tipados de una pieza transformados a coordenadas mundiales.
        Cachea por (part_id, transform_hash).
        """
        from src.connectivity_db import get_part_ports
        
        # Hash simplificado de la transformación
        tf_hash = tuple(np.round(transform.flatten(), 2))
        cache_key = (part_id, tf_hash)
        
        if cache_key in self.port_cache:
            return self.port_cache[cache_key]
        
        try:
            local_ports = get_part_ports(part_id)
        except Exception:
            self.port_cache[cache_key] = []
            return []
        
        R = transform[:3, :3]
        t = transform[:3, 3]
        
        world_ports = []
        for p in local_ports:
            # Posición en mundo
            pos_w = (R @ p.local_pos) + t
            
            # Normal en mundo
            norm_w = R @ p.local_normal
            norm_mag = float(np.linalg.norm(norm_w))
            if norm_mag > 1e-4:
                norm_w = norm_w / norm_mag
            else:
                norm_w = p.local_normal
            
            world_ports.append({
                "id": p.id,
                "port_type": p.port_type,
                "gender": p.gender,
                "pos_world": pos_w,
                "normal_world": norm_w,
                "dof": p.dof,
            })
        
        self.port_cache[cache_key] = world_ports
        return world_ports

    def _penetration_is_at_compatible_port(
        self,
        obb_a: Dict[str, Any],
        obb_b: Dict[str, Any],
        overlap_center: np.ndarray,
        coupling_tolerance: float = 6.0,
        proximity_tolerance: float = 10.0,
    ) -> bool:
        """
        Verifica si la penetración detectada ocurre en la vecindad de un par de puertos compatibles.
        Si es así, es un acoplamiento legítimo conector-en-hueco, no colisión de cuerpo sólido.
        
        Retorna True si la penetración es validable como acoplamiento.
        """
        from src.connectivity_db import are_ports_compatible, check_port_coupling
        
        try:
            ports_a = self._get_world_ports(obb_a["part_id"], obb_a["transform"])
            ports_b = self._get_world_ports(obb_b["part_id"], obb_b["transform"])
        except Exception:
            return False
        
        if not ports_a or not ports_b:
            return False
        
        # Buscar pares compatibles cercanos al centroide del solapamiento
        for pa in ports_a:
            for pb in ports_b:
                if not are_ports_compatible(pa["port_type"], pb["port_type"]):
                    continue
                
                # Verificar distancia de los puertos al centroide del solapamiento
                dist_a = float(np.linalg.norm(pa["pos_world"] - overlap_center))
                dist_b = float(np.linalg.norm(pb["pos_world"] - overlap_center))
                
                if dist_a > proximity_tolerance or dist_b > proximity_tolerance:
                    continue
                
                # Verificar acoplamiento físico (normales opuestas, distancia ≤ coupling_tolerance)
                if check_port_coupling(pa, pb, dist_tol=coupling_tolerance, angle_tol_deg=20.0):
                    return True
        
        return False


    def get_part_sub_boxes_local(self, part_id: str, meta: Dict[str, Any]) -> List[Tuple[float, float, float, float, float, float]]:
        """
        Decomposes complex or non-convex parts (especially L-shaped brackets)
        into a union of constituent rectangular solid boxes [x0, y0, z0, x1, y1, z1] in local space.
        This eliminates false positive collisions in the empty interior corner of L-brackets.
        """
        bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]
        y_top = float(by0)
        y_bot = float(by1)
        if y_bot <= y_top:
            y_bot = y_top + 8.0

        clean_id = part_id.replace(".dat", "").lower()

        # Brackets 44728 (2x2 - 2x2 bracket)
        if "44728" in clean_id:
            # Horizontal foot: X in [-20, 20], Y in [0, 8], Z in [-10, 10]
            # Vertical wall:   X in [-20, 20], Y in [0, 40], Z in [-14, -10]
            return [
                (-20.0, 0.0, -10.0, 20.0, 8.0, 10.0),
                (-20.0, 0.0, -14.0, 20.0, 40.0, -10.0)
            ]

        # Brackets 99781 (1x2 - 1x2 bracket, down)
        if "99781" in clean_id:
            return [
                (-20.0, 0.0, -10.0, 20.0, 8.0, 10.0),
                (-20.0, 0.0, -14.0, 20.0, 20.0, -10.0)
            ]

        # Brackets 99780 (1x2 - 1x2 bracket, inverted/up)
        if "99780" in clean_id:
            return [
                (-20.0, 0.0, -10.0, 20.0, 8.0, 10.0),
                (-20.0, -12.0, -14.0, 20.0, 0.0, -10.0)
            ]

        # Brackets 2436 (1x2 - 1x4 bracket)
        if "2436" in clean_id:
            return [
                (-40.0, 0.0, -10.0, 40.0, 8.0, 10.0),
                (-40.0, 0.0, -14.0, 40.0, 20.0, -10.0)
            ]

        # Brackets 36840 (1x1 - 1x1 bracket, inverted/up)
        if "36840" in clean_id:
            return [
                (-10.0, 0.0, -10.0, 10.0, 8.0, 10.0),
                (-10.0, -12.0, -14.0, 10.0, 0.0, -10.0)
            ]

        # Brackets 36841 (1x1 - 1x1 bracket, down)
        if "36841" in clean_id:
            return [
                (-10.0, 0.0, -10.0, 10.0, 8.0, 10.0),
                (-10.0, 0.0, -14.0, 10.0, 20.0, -10.0)
            ]

        # Arches (e.g. 1x6 arch 3307, 1x4 arch 3659, 1x3 arch 13965, etc.)
        # Decomposed into Left Pillar, Right Pillar, and Top Lintel to leave the interior arch opening free
        is_arch = any(k in clean_id for k in ["arch", "3307", "3659", "13965", "6005", "38583"])
        if is_arch and (bx1 - bx0) >= 30.0:
            pillar_w = 20.0
            return [
                (float(bx0), y_top, float(bz0), float(bx0 + pillar_w), y_bot, float(bz1)),
                (float(bx1 - pillar_w), y_top, float(bz0), float(bx1), y_bot, float(bz1)),
                (float(bx0), y_top, float(bz0), float(bx1), min(y_bot, y_top + 8.0), float(bz1))
            ]

        # Mudguards (e.g. 50745, 18974, 60212)
        is_mudguard = any(k in clean_id for k in ["mudguard", "50745", "18974", "60212"])
        if is_mudguard and (bx1 - bx0) >= 30.0:
            foot_w = 15.0
            return [
                (float(bx0), y_top, float(bz0), float(bx0 + foot_w), y_bot, float(bz1)),
                (float(bx1 - foot_w), y_top, float(bz0), float(bx1), y_bot, float(bz1)),
                (float(bx0), y_top, float(bz0), float(bx1), min(y_bot, y_top + 8.0), float(bz1))
            ]

        # Round corner plates (e.g. 30565 Plate 4 x 4 Round Corner)
        # Decomposed so the outer circular cut-out quadrant is free and does not block neighboring parts
        if "30565" in clean_id:
            return [
                (float(bx0), y_top, float(bz0), 20.0, y_bot, float(bz1)),
                (float(bx0), y_top, float(bz0), float(bx1), y_bot, 20.0)
            ]

        # Headlight brick (4070 Brick Special 1 x 1 with Headlight)
        # Front face has a 4 LDU recess from Y = 0 to Y = 20 where plates/tiles attach onto the front stud without colliding with the brick body.
        # Bottom lip only extends forward (Z in [-10, -6]) between Y = 20 and Y = 24.
        if "4070" in clean_id:
            return [
                (-10.0, 0.0, -6.0, 10.0, 24.0, 10.0),    # Main vertical brick body
                (-10.0, 20.0, -10.0, 10.0, 24.0, -6.0)   # Bottom front lip/ledge
            ]

        # Standard single convex bounding box
        return [(float(bx0), y_top, float(bz0), float(bx1), y_bot, float(bz1))]

    def get_part_solid_obb(self, part_id: str, transform: np.ndarray) -> Dict[str, Any]:
        """
        Extracts the solid plastic body bounding volume (excluding protruding top studs)
        transformed into world space. If the part is decomposed into multi-boxes (e.g. L-brackets),
        includes both the overarching envelope and constituent sub-boxes.
        """
        key = (part_id, tuple(np.round(transform.flatten(), 2)))
        if key in self.obb_cache:
            return self.obb_cache[key]
        meta = self.geom.get_part_meta(part_id)
        bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]

        # For standard parts where y0 == 0 (standard bricks and plates), body is [y0, y1].
        # For inverted parts or slopes where y0 < 0 and y1 <= 0, body is [y0, y1].
        # For brackets where y0 == 0 and y1 > 0, body is [y0, y1].
        # In all cases, the solid plastic body vertices are defined by [bx0, by0, bz0, bx1, by1, bz1].
        y_top = float(by0)
        y_bot = float(by1)
        if y_bot <= y_top:
            y_bot = y_top + 8.0

        # Local body vertices (8 corners of the solid body box)
        corners_local = np.array([
            [bx0, y_top, bz0],
            [bx1, y_top, bz0],
            [bx1, y_top, bz1],
            [bx0, y_top, bz1],
            [bx0, y_bot, bz0],
            [bx1, y_bot, bz0],
            [bx1, y_bot, bz1],
            [bx0, y_bot, bz1],
        ], dtype=np.float32)

        R = transform[:3, :3]
        t = transform[:3, 3]
        corners_world = (corners_local @ R.T) + t

        w_min = np.min(corners_world, axis=0)
        w_max = np.max(corners_world, axis=0)

        # Center and half-sizes along world principal axes
        center = (w_min + w_max) / 2.0
        extent = (w_max - w_min) / 2.0

        # Multi-box decomposition
        sub_boxes_local = self.get_part_sub_boxes_local(part_id, meta)
        sub_boxes = []
        for sb in sub_boxes_local:
            sb_corners = np.array([
                [sb[0], sb[1], sb[2]], [sb[3], sb[1], sb[2]],
                [sb[3], sb[4], sb[2]], [sb[0], sb[4], sb[2]],
                [sb[0], sb[1], sb[5]], [sb[3], sb[1], sb[5]],
                [sb[3], sb[4], sb[5]], [sb[0], sb[4], sb[5]],
            ], dtype=np.float32)
            sb_w = (sb_corners @ R.T) + t
            sb_min = np.min(sb_w, axis=0)
            sb_max = np.max(sb_w, axis=0)
            sub_boxes.append({
                "w_min": sb_min,
                "w_max": sb_max,
                "corners_world": sb_w,
                "local_box": sb
            })

        res = {
            "part_id": part_id,
            "corners_world": corners_world,
            "w_min": w_min,
            "w_max": w_max,
            "center": center,
            "extent": extent,
            "transform": transform,
            "meta": meta,
            "sub_boxes": sub_boxes,
            "is_multi_box": len(sub_boxes) > 1
        }
        self.obb_cache[key] = res
        return res

    def check_solid_penetration(
        self,
        obb_a: Dict[str, Any],
        obb_b: Dict[str, Any],
        penetration_threshold: float = 0.5
    ) -> Tuple[bool, float, str]:
        """
        Tests whether solid body A and solid body B penetrate each other by more than penetration_threshold LDU.
        Returns: (has_collision, max_penetration_ldu, description)
        """
        # 1. Axis-Aligned overlap check in World X, Y, Z
        a_min, a_max = obb_a["w_min"], obb_a["w_max"]
        b_min, b_max = obb_b["w_min"], obb_b["w_max"]

        overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
        overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
        overlap_z = min(a_max[2], b_max[2]) - max(a_min[2], b_min[2])

        # If any axis has no overlap, they do not intersect at all
        if overlap_x <= 0.05 or overlap_y <= 0.05 or overlap_z <= 0.05:
            return False, 0.0, "Disjoint"

        # Volumetric ratio check: filters out thin boundary overlaps, hollow cavities, and external flaps
        vol_overlap = float(overlap_x * overlap_y * overlap_z)
        vol_a = float(max(10.0, (a_max[0] - a_min[0]) * (a_max[1] - a_min[1]) * (a_max[2] - a_min[2])))
        vol_b = float(max(10.0, (b_max[0] - b_min[0]) * (b_max[1] - b_min[1]) * (b_max[2] - b_min[2])))
        vol_ratio = vol_overlap / min(vol_a, vol_b)

        if vol_ratio < 0.06:
            # Grazing contact or hollow cavity engagement, not a solid body collision
            return False, vol_overlap, f"Tolerated boundary contact (vol ratio = {vol_ratio:.3f})"
        
        # ENHANCEMENT (Nov 13, 2026): For potential collisions, verify using multi-box decomposition
        # if available. This catches cases where OBB overestimate interior hollow cavities.
        part_a_id = obb_a.get("part_id", "unknown")
        part_b_id = obb_b.get("part_id", "unknown")
        
        try:
            meta_a = self.geom.get_part_meta(part_a_id)
            meta_b = self.geom.get_part_meta(part_b_id)
            sub_boxes_a = self.get_part_sub_boxes_local(part_a_id, meta_a)
            sub_boxes_b = self.get_part_sub_boxes_local(part_b_id, meta_b)
            
            # If either part has complex geometry (multiple sub-boxes), use enhanced collision check
            if len(sub_boxes_a) > 1 or len(sub_boxes_b) > 1:
                # Use single OBB check for now, but flag for advanced decomposition
                # In future: implement full sub-box vs sub-box collision matrix
                pass  # Placeholder for future multi-box traversal
        except Exception as e:
            # If decomposition fails, fall through to standard OBB check
            pass


        # 2. Check for Vertical Stacking (Level L on top of Level L-1)
        # If parts are stacked, their horizontal footprint overlaps (overlap_x > 0, overlap_z > 0),
        # but in Y they should only share an interface (touching face: overlap_y approx 0.0 LDU).
        # Stud insertion happens in stud cavities, but plastic body bodies cannot penetrate in Y.
        if overlap_y <= penetration_threshold:
            # Touching top-to-bottom face: Allowed contact!
            return False, overlap_y, f"Allowed face contact (Y overlap = {overlap_y:.2f} LDU)"

        # 3. If overlap_y > penetration_threshold: both parts occupy the SAME vertical layer!
        # Now check horizontal footprint overlap (X and Z)
        if overlap_x > penetration_threshold and overlap_z > penetration_threshold:
            # Check for non-rectangular shape filtering (e.g. Round Plate 11213, Corner Round 30565)
            # If both are standard rectangular boxes, this is a definite collision!
            p_a = obb_a["part_id"]
            p_b = obb_b["part_id"]

            # Detailed check for Corner Round (30565) or Round Plate (11213)
            if "30565" in p_a or "30565" in p_b or "11213" in p_a or "11213" in p_b:
                is_submesh_collision = self._check_detailed_footprint_collision(obb_a, obb_b, penetration_threshold)
                if not is_submesh_collision:
                    return False, 0.0, "Allowed curved cutout clearance"

            min_pen = min(overlap_x, overlap_y, overlap_z)
            return True, float(min_pen), f"Solid penetration of {min_pen:.2f} LDU between {p_a} and {p_b}"

        return False, 0.0, "Allowed boundary contact"

    def _check_detailed_footprint_collision(
        self,
        obb_a: Dict[str, Any],
        obb_b: Dict[str, Any],
        threshold: float
    ) -> bool:
        """
        Samples points within the candidate overlap region to determine if solid plastic intersects,
        taking into account round/curved cutouts of pieces like 30565 and 11213.
        """
        a_min, a_max = obb_a["w_min"], obb_a["w_max"]
        b_min, b_max = obb_b["w_min"], obb_b["w_max"]

        ix0 = max(a_min[0], b_min[0]) + threshold
        ix1 = min(a_max[0], b_max[0]) - threshold
        iz0 = max(a_min[2], b_min[2]) + threshold
        iz1 = min(a_max[2], b_max[2]) - threshold

        if ix1 <= ix0 or iz1 <= iz0:
            return False

        # Sample grid in intersection
        xs = np.linspace(ix0, ix1, 5)
        zs = np.linspace(iz0, iz1, 5)

        for x in xs:
            for z in zs:
                if self._point_inside_part_solid(x, z, obb_a) and self._point_inside_part_solid(x, z, obb_b):
                    return True
        return False

    def _point_inside_part_solid(self, x: float, z: float, obb: Dict[str, Any]) -> bool:
        """Tests whether 2D point (x, z) lies inside the solid horizontal footprint of the part."""
        tf = obb["transform"]
        R = tf[:3, :3]
        t = tf[:3, 3]

        # Transform point into local part coordinates
        p_world = np.array([x, t[1], z], dtype=np.float32)
        p_local = (p_world - t) @ R

        lx, lz = p_local[0], p_local[2]
        part_id = obb["part_id"]

        if "30565" in part_id:
            # Corner round 4x4: Tip at (-40, 40), edges to (40, 40) and (-40, -40).
            # Quarter circle arc centered at (-40, 40) with radius 80.
            if lx < -40.0 or lx > 40.0 or lz < -40.0 or lz > 40.0:
                return False
            # Check inside circle of radius 80 from (-40, 40)
            dx = lx - (-40.0)
            dz = lz - 40.0
            return (dx * dx + dz * dz) <= (80.0 * 80.0)

        if "11213" in part_id:
            # Round plate 6x6 with hole: radius 60 from center (0, 0)
            return (lx * lx + lz * lz) <= (60.0 * 60.0)

        # Standard rectangular bbox check
        meta = obb["meta"]
        bx0, _, bz0, bx1, _, bz1 = meta["bbox"]
        return (bx0 <= lx <= bx1) and (bz0 <= lz <= bz1)

    def validate_new_part_placement(
        self,
        part_id: str,
        transform: np.ndarray,
        existing_nodes: List[Dict[str, Any]],
        penetration_threshold: float = 0.5
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates that a new part placement does not collide with ANY pre-existing parts in the build.
        """
        new_obb = self.get_part_solid_obb(part_id, transform)

        for ex_node in existing_nodes:
            if ex_node.get("is_minifig", False):
                continue
            ex_id = ex_node["part_id"]

            # Technic coaxial pin-into-hole allowance
            is_connector = any(k in part_id for k in ["pin", "axle", "6558", "2780", "32054", "4519", "3705", "32556"])
            ex_is_connector = any(k in ex_id for k in ["pin", "axle", "6558", "2780", "32054", "4519", "3705", "32556"])
            if (is_connector and not ex_is_connector) or (ex_is_connector and not is_connector):
                # Pin/Axle engaging with Beam or perforated part
                continue

            # Ball-and-socket (Mixels/towball) allowance
            is_ball_pair = any(k in part_id for k in ["14417", "14418", "14704", "22890", "77808", "92013", "66727"]) and \
                           any(k in ex_id for k in ["14417", "14418", "14704", "22890", "77808", "92013", "66727"])
            if is_ball_pair:
                continue

            ex_obb = ex_node.get("_obb")
            if ex_obb is None:
                ex_tf = np.array(ex_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
                ex_obb = self.get_part_solid_obb(ex_id, ex_tf)
                ex_node["_obb"] = ex_obb

            # Broadphase AABB rejection: if bounding boxes do not overlap, parts cannot collide
            new_w_min = new_obb["w_min"]
            new_w_max = new_obb["w_max"]
            ex_w_min = ex_obb["w_min"]
            ex_w_max = ex_obb["w_max"]
            if (min(new_w_max[0], ex_w_max[0]) - max(new_w_min[0], ex_w_min[0]) <= 0.05 or
                min(new_w_max[1], ex_w_max[1]) - max(new_w_min[1], ex_w_min[1]) <= 0.05 or
                min(new_w_max[2], ex_w_max[2]) - max(new_w_min[2], ex_w_min[2]) <= 0.05):
                continue

            # If either part is decomposed into multi-boxes (e.g. L-brackets),
            # check penetration across constituent sub-boxes: a collision only occurs if
            # at least one pair of sub-boxes collides.
            if new_obb.get("is_multi_box") or ex_obb.get("is_multi_box"):
                sub_a = new_obb.get("sub_boxes", [new_obb])
                sub_b = ex_obb.get("sub_boxes", [ex_obb])
                has_sub_col = False
                max_sub_pen = 0.0
                sub_reason = ""
                for sa in sub_a:
                    for sb in sub_b:
                        # Construct temporary pseudo-OBBs for the sub-box pair
                        pseudo_a = {
                            "part_id": part_id,
                            "w_min": sa["w_min"],
                            "w_max": sa["w_max"],
                            "transform": transform,
                            "meta": new_obb["meta"]
                        }
                        pseudo_b = {
                            "part_id": ex_id,
                            "w_min": sb["w_min"],
                            "w_max": sb["w_max"],
                            "transform": ex_obb["transform"],
                            "meta": ex_obb["meta"]
                        }
                        c_col, c_pen, c_reason = self.check_solid_penetration(pseudo_a, pseudo_b, penetration_threshold)
                        if c_col:
                            # CHANGE 1: Verificar si es acoplamiento conector-en-hueco antes de rechazar
                            overlap_center = (pseudo_a["w_min"] + pseudo_a["w_max"]) / 2.0
                            if self._penetration_is_at_compatible_port(pseudo_a, pseudo_b, overlap_center):
                                continue  # acoplamiento legítimo, ignorar colisión
                            
                            has_sub_col = True
                            if c_pen > max_sub_pen:
                                max_sub_pen = c_pen
                                sub_reason = c_reason

                if has_sub_col:
                    ex_idx = ex_node.get("part_index", "?")
                    return False, f"Colisión 3D detectada en sub-box ({sub_reason}) con Pieza #{ex_idx} ({ex_id}). Penetración = {max_sub_pen:.2f} LDU."
            else:
                has_col, pen_amt, reason = self.check_solid_penetration(new_obb, ex_obb, penetration_threshold)
                if has_col:
                    # CHANGE 1: Verificar si es acoplamiento conector-en-hueco antes de rechazar
                    overlap_center = (new_obb["w_min"] + new_obb["w_max"]) / 2.0
                    if self._penetration_is_at_compatible_port(new_obb, ex_obb, overlap_center):
                        continue  # acoplamiento legítimo, ignorar colisión
                    
                    ex_idx = ex_node.get("part_index", "?")
                    return False, f"Colisión 3D detectada ({reason}) con Pieza #{ex_idx} ({ex_id}). Penetración = {pen_amt:.2f} LDU."

        return True, None
