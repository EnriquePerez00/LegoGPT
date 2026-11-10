import os
import math
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from src.ldraw_geometry import LDrawGeometryResolver


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Converts #RRGGBB hex string to OpenCV BGR tuple."""
    if not hex_str:
        return (160, 165, 175)
    hex_str = hex_str.lstrip("#")
    if len(hex_str) != 6:
        return (160, 165, 175)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (b, g, r)


class StepAxonometricRenderer:
    """
    Renders 2.5D/3D axonometric isometric view of LEGO assembly state step-by-step:
      - Pre-existing parts (context/ghost): rendered in muted translucent slate gray.
      - Current step parts (active/new): highlighted in their official LEGO colors with vibrant border.
      - Axonometric projection: 45° azimuth, 30° elevation (or -25° for inverted/flip steps).
      - Studs and physical connection alignment visualization.
    """
    def __init__(self, geom_resolver: LDrawGeometryResolver):
        self.geom = geom_resolver
        self.part_mesh_cache = {}

    def project_points(
        self,
        pts_3d: np.ndarray,
        az_deg: float,
        el_deg: float,
        scale: float,
        offset: Tuple[float, float],
        center_3d: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Projects 3D points to 2D screen coordinates relative to assembly center."""
        if len(pts_3d) == 0:
            return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)

        if center_3d is not None:
            pts_3d = pts_3d - center_3d

        az = math.radians(az_deg)
        el = math.radians(el_deg)

        # Yaw around Y
        X = pts_3d[:, 0] * math.cos(az) + pts_3d[:, 2] * math.sin(az)
        Z = -pts_3d[:, 0] * math.sin(az) + pts_3d[:, 2] * math.cos(az)
        Y = pts_3d[:, 1]

        # Pitch around X
        Y_proj = Y * math.cos(el) - Z * math.sin(el)

        u = scale * X + offset[0]
        v = scale * Y_proj + offset[1]

        # Camera depth (for Painter's algorithm depth sorting)
        depth = Z * math.cos(el) + Y * math.sin(el)
        return np.column_stack([u, v]), depth

    def get_part_faces_and_studs(self, part_id: str) -> Tuple[List[Dict], List[np.ndarray]]:
        """Extracts 3D polygonal faces and stud positions for the part."""
        if part_id in self.part_mesh_cache:
            return self.part_mesh_cache[part_id]

        meta = self.geom.get_part_meta(part_id)
        bx0, by0, bz0, bx1, by1, bz1 = meta["bbox"]

        # 1. Round 6x6 plate (11213)
        if "11213" in part_id or ("round" in part_id and bx1 > 40):
            r_out = float(max(abs(bx0), abs(bx1), abs(bz0), abs(bz1)))
            r_in = 20.0 if "hole" in part_id or "11213" in part_id else 0.0
            n_seg = 24
            angles = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
            top_out = np.column_stack([[r_out * math.cos(a) for a in angles], [by0]*n_seg, [r_out * math.sin(a) for a in angles]]).astype(np.float32)
            bot_out = np.column_stack([[r_out * math.cos(a) for a in angles], [by1]*n_seg, [r_out * math.sin(a) for a in angles]]).astype(np.float32)
            top_in = np.column_stack([[r_in * math.cos(a) for a in angles], [by0]*n_seg, [r_in * math.sin(a) for a in angles]]).astype(np.float32)
            bot_in = np.column_stack([[r_in * math.cos(a) for a in angles], [by1]*n_seg, [r_in * math.sin(a) for a in angles]]).astype(np.float32)

            faces = []
            for i in range(n_seg):
                j = (i + 1) % n_seg
                # Top ring quad (segmented so each sector depth-sorts accurately)
                faces.append({"pts": np.array([top_out[i], top_out[j], top_in[j], top_in[i]], dtype=np.float32), "normal": np.array([0, -1, 0], dtype=np.float32)})
                # Bottom ring quad
                faces.append({"pts": np.array([bot_out[j], bot_out[i], bot_in[i], bot_in[j]], dtype=np.float32), "normal": np.array([0, 1, 0], dtype=np.float32)})
                # Outer cylindrical side quad
                mid_ang = (angles[i] + angles[j]) / 2.0
                norm = np.array([math.cos(mid_ang), 0, math.sin(mid_ang)], dtype=np.float32)
                faces.append({"pts": np.array([top_out[i], top_out[j], bot_out[j], bot_out[i]], dtype=np.float32), "normal": norm})
                # Inner hole side quad if present
                if r_in > 0:
                    norm_in = np.array([-math.cos(mid_ang), 0, -math.sin(mid_ang)], dtype=np.float32)
                    faces.append({"pts": np.array([top_in[j], top_in[i], bot_in[i], bot_in[j]], dtype=np.float32), "normal": norm_in})

            res = (faces, meta["studs"])
            self.part_mesh_cache[part_id] = res
            return res

        # 2. Quarter-round 4x4 plate (30565)
        if "30565" in part_id:
            n_arc = 16
            angles = np.linspace(0, -np.pi / 2.0, n_arc)
            arc_x = -40.0 + 80.0 * np.cos(angles)
            arc_z = 40.0 + 80.0 * np.sin(angles)
            top_arc = np.column_stack([arc_x, np.full(n_arc, by0), arc_z]).astype(np.float32)
            corner_top = np.array([-40.0, by0, 40.0], dtype=np.float32)

            bot_arc = np.column_stack([arc_x, np.full(n_arc, by1), arc_z]).astype(np.float32)
            corner_bot = np.array([-40.0, by1, 40.0], dtype=np.float32)

            faces = []
            # Triangulated top pie slice fan
            for i in range(n_arc - 1):
                tri = np.array([corner_top, top_arc[i], top_arc[i+1]], dtype=np.float32)
                faces.append({"pts": tri, "normal": np.array([0, -1, 0], dtype=np.float32)})

            # Triangulated bottom pie slice fan
            for i in range(n_arc - 1):
                tri = np.array([corner_bot, bot_arc[i+1], bot_arc[i]], dtype=np.float32)
                faces.append({"pts": tri, "normal": np.array([0, 1, 0], dtype=np.float32)})

            # Straight wall 1: from corner to arc start
            faces.append({"pts": np.array([corner_top, top_arc[0], bot_arc[0], corner_bot], dtype=np.float32), "normal": np.array([0, 0, 1], dtype=np.float32)})
            # Straight wall 2: from arc end to corner
            faces.append({"pts": np.array([top_arc[-1], corner_top, corner_bot, bot_arc[-1]], dtype=np.float32), "normal": np.array([-1, 0, 0], dtype=np.float32)})

            # Curved arc quads
            for i in range(n_arc - 1):
                quad = np.array([top_arc[i], top_arc[i+1], bot_arc[i+1], bot_arc[i]], dtype=np.float32)
                mid_a = (angles[i] + angles[i+1]) / 2.0
                norm = np.array([math.cos(mid_a), 0, math.sin(mid_a)], dtype=np.float32)
                faces.append({"pts": quad, "normal": norm})

            res = (faces, meta["studs"])
            self.part_mesh_cache[part_id] = res
            return res

        # 3. Minifigure special elements (Legs, Torso, Head, Hair)
        if "970" in part_id or "legs" in part_id or "973" in part_id or "torso" in part_id or "3626" in part_id:
            # Enhanced proportions for Minifigures
            c = [
                np.array([bx0, by0, bz0], dtype=np.float32),
                np.array([bx1, by0, bz0], dtype=np.float32),
                np.array([bx1, by0, bz1], dtype=np.float32),
                np.array([bx0, by0, bz1], dtype=np.float32),
                np.array([bx0, by1, bz0], dtype=np.float32),
                np.array([bx1, by1, bz0], dtype=np.float32),
                np.array([bx1, by1, bz1], dtype=np.float32),
                np.array([bx0, by1, bz1], dtype=np.float32),
            ]
            faces = [
                {"pts": np.array([c[0], c[1], c[2], c[3]]), "normal": np.array([0, -1, 0], dtype=np.float32)},
                {"pts": np.array([c[4], c[7], c[6], c[5]]), "normal": np.array([0, 1, 0], dtype=np.float32)},
                {"pts": np.array([c[3], c[2], c[6], c[7]]), "normal": np.array([0, 0, 1], dtype=np.float32)},
                {"pts": np.array([c[1], c[0], c[4], c[5]]), "normal": np.array([0, 0, -1], dtype=np.float32)},
                {"pts": np.array([c[2], c[1], c[5], c[6]]), "normal": np.array([1, 0, 0], dtype=np.float32)},
                {"pts": np.array([c[0], c[3], c[7], c[4]]), "normal": np.array([-1, 0, 0], dtype=np.float32)},
            ]
            res = (faces, meta["studs"])
            self.part_mesh_cache[part_id] = res
            return res

        # 4. Extract true polygonal solid geometry directly from LDraw definition (wedges, slopes, wings, modified plates)
        ldraw_faces = self.geom.get_part_faces(part_id)
        if ldraw_faces and len(ldraw_faces) >= 4:
            res = (ldraw_faces, meta["studs"])
            self.part_mesh_cache[part_id] = res
            return res

        # 5. Standard rectangular box fallback (Bricks, Plates, Tiles without mesh)
        c = [
            np.array([bx0, by0, bz0], dtype=np.float32),
            np.array([bx1, by0, bz0], dtype=np.float32),
            np.array([bx1, by0, bz1], dtype=np.float32),
            np.array([bx0, by0, bz1], dtype=np.float32),
            np.array([bx0, by1, bz0], dtype=np.float32),
            np.array([bx1, by1, bz0], dtype=np.float32),
            np.array([bx1, by1, bz1], dtype=np.float32),
            np.array([bx0, by1, bz1], dtype=np.float32),
        ]
        faces = [
            {"pts": np.array([c[0], c[1], c[2], c[3]]), "normal": np.array([0, -1, 0], dtype=np.float32)},
            {"pts": np.array([c[4], c[7], c[6], c[5]]), "normal": np.array([0, 1, 0], dtype=np.float32)},
            {"pts": np.array([c[3], c[2], c[6], c[7]]), "normal": np.array([0, 0, 1], dtype=np.float32)},
            {"pts": np.array([c[1], c[0], c[4], c[5]]), "normal": np.array([0, 0, -1], dtype=np.float32)},
            {"pts": np.array([c[2], c[1], c[5], c[6]]), "normal": np.array([1, 0, 0], dtype=np.float32)},
            {"pts": np.array([c[0], c[3], c[7], c[4]]), "normal": np.array([-1, 0, 0], dtype=np.float32)},
        ]
        res = (faces, meta["studs"])
        self.part_mesh_cache[part_id] = res
        return res

    def render_step_assembly(
        self,
        current_page_num: int,
        step_nodes: List[Dict],
        all_nodes: List[Dict],
        tmpl_by_element: Dict[str, Dict],
        is_flip: bool = False,
        width: int = 760,
        height: int = 520
    ) -> np.ndarray:
        """
        Renders the axonometric digital twin simulation for current_page_num:
          - Automatically isolates Minifigure subassemblies vs Main ship model.
          - Contextual ghosting of previously placed parts.
          - Saturated lighting and crisp highlighted borders for active new parts.
        """
        # Create dark modern studio background matching the report theme (#090d16)
        img = np.full((height, width, 3), (18, 22, 30), dtype=np.uint8)

        # Subtle dark technical grid
        grid_color = (26, 32, 44)
        for gx in range(0, width, 35):
            cv2.line(img, (gx, 0), (gx, height), grid_color, 1)
        for gy in range(0, height, 35):
            cv2.line(img, (0, gy), (width, gy), grid_color, 1)

        is_mf_step = any(n.get("is_minifig", False) for n in step_nodes)

        # Filter nodes: keep only same subassembly (minifig vs main model)
        relevant_nodes = [
            n for n in all_nodes
            if n["page_num"] <= current_page_num and bool(n.get("is_minifig", False)) == is_mf_step
        ]

        if not relevant_nodes:
            # Fallback if no nodes found
            cv2.putText(img, f"No model nodes for Page {current_page_num}", (20, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 150, 160), 1, cv2.LINE_AA)
            return img

        # Determine active parts in this step
        if current_page_num == 5:
            # On first building step (page 5), base plate 11213 is the ground support;
            # active pieces are the new pieces attached to it (e.g. 30565)
            active_indices = {n["part_index"] for n in step_nodes if n.get("connections")}
            if not active_indices:
                active_indices = {n["part_index"] for n in step_nodes}
        else:
            active_indices = {n["part_index"] for n in step_nodes}

        # Axonometric camera parameters (Looking from above down at Az=45, El=35)
        az_deg = 45.0
        el_deg = -25.0 if is_flip else 35.0

        az_rad, el_rad = math.radians(az_deg), math.radians(el_deg)
        # View direction vector (from camera to scene):
        # In LDraw coordinates, +Y is DOWN. So camera above (Y < 0) looking down has positive Y view component.
        view_dir = np.array([-math.sin(az_rad)*math.cos(el_rad), math.sin(el_rad), math.cos(az_rad)*math.cos(el_rad)], dtype=np.float32)
        view_dir /= np.linalg.norm(view_dir)

        # Overhead key light shining down from the sky (+Y in LDraw is downwards):
        light_dir = np.array([-0.35, 0.85, -0.4], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)

        render_items = []
        all_3d_points = []

        # Connection coupling lines for active parts
        coupling_lines_3d = []

        for n in relevant_nodes:
            p_idx = n["part_index"]
            is_active = p_idx in active_indices
            part_id = n["part_id"]
            tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
            R = tf[:3, :3]
            t = tf[:3, 3]

            el_id = n.get("element_id")
            t_info = tmpl_by_element.get(el_id, {})
            hex_col = t_info.get("color_rgb", "#94a3b8")
            base_bgr = hex_to_bgr(hex_col)

            faces, studs = self.get_part_faces_and_studs(part_id)
            meta = self.geom.get_part_meta(part_id)

            for f in faces:
                pts_local = f["pts"]
                pts_world = (pts_local @ R.T) + t
                norm_world = R @ f["normal"]

                # Back-face culling: only render front-facing polygons towards camera
                if np.dot(norm_world, view_dir) >= -0.02:
                    continue

                all_3d_points.append(pts_world)

                diffuse = max(0.25, float(np.dot(norm_world, light_dir)))
                # Render every part in its official LEGO material color
                col = tuple(int(np.clip(c * (0.60 + 0.40 * diffuse), 0, 255)) for c in base_bgr)
                if is_active:
                    border_col = (16, 215, 140) if is_flip else (35, 225, 160)  # Bright emerald boundary for active pieces
                    border_thick = 2
                else:
                    # Previous parts: solid official color with subtle boundary line
                    border_col = tuple(int(np.clip(c * 0.45, 0, 255)) for c in base_bgr)
                    border_thick = 1

                render_items.append({
                    "type": "face",
                    "pts_world": pts_world,
                    "color": col,
                    "border_color": border_col,
                    "border_thick": border_thick,
                    "is_active": is_active,
                })

            # Studs: top/front/lateral studs pointing towards camera are visible
            stud_details = meta.get("stud_details") or [{"pos": s, "normal": np.array([0.0, -1.0, 0.0], dtype=np.float32)} for s in studs]
            for s_info in stud_details:
                s_pos = s_info["pos"] if isinstance(s_info, dict) else s_info
                s_norm_local = s_info.get("normal", np.array([0.0, -1.0, 0.0], dtype=np.float32)) if isinstance(s_info, dict) else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                s_world = (s_pos @ R.T) + t
                s_norm = R @ s_norm_local
                if np.dot(s_norm, view_dir) < -0.05:
                    all_3d_points.append(s_world.reshape(1, 3))
                    stud_col = base_bgr
                    render_items.append({
                        "type": "stud",
                        "pos_world": s_world,
                        "is_active": is_active,
                        "color": stud_col
                    })

            # Check if active part has connections to draw coupling vector
            if is_active and n.get("connections"):
                for conn in n["connections"]:
                    parent_idx = conn.get("parent_part_index")
                    p_node = next((pn for pn in all_nodes if pn["part_index"] == parent_idx), None)
                    if p_node:
                        p_tf = np.array(p_node["absolute_transform"], dtype=np.float32).reshape((4, 4))
                        p_pos = p_tf[:3, 3]
                        c_pos = t
                        coupling_lines_3d.append((p_pos, c_pos))

        if not all_3d_points:
            return img

        all_pts = np.vstack(all_3d_points)
        proj_test, _ = self.project_points(all_pts, az_deg, el_deg, scale=1.0, offset=(0, 0))
        u_min, v_min = np.min(proj_test, axis=0)
        u_max, v_max = np.max(proj_test, axis=0)
        span_w = max(10.0, u_max - u_min)
        span_h = max(10.0, v_max - v_min)

        # Scale with margin
        scale = min((width * 0.74) / span_w, (height * 0.74) / span_h)
        offset_u = (width - (u_max + u_min) * scale) / 2.0
        offset_v = (height - (v_max + v_min) * scale) / 2.0
        offset = (offset_u, offset_v)

        # Depth calculation for Painter's algorithm
        for it in render_items:
            if it["type"] == "face":
                _, depths = self.project_points(it["pts_world"], az_deg, el_deg, scale, offset)
                it["depth"] = float(np.mean(depths))
            elif it["type"] == "stud":
                _, d = self.project_points(it["pos_world"].reshape(1, 3), az_deg, el_deg, scale, offset)
                # Stud depth bias ensures studs appear in front of their hosting face (closer to camera)
                it["depth"] = float(d[0]) - 2.5

        # Sort back-to-front (descending: furthest back with largest depth drawn first, nearest foreground with smallest depth drawn last)
        render_items.sort(key=lambda x: x["depth"], reverse=True)

        # Draw 3D faces and studs
        for it in render_items:
            if it["type"] == "face":
                pts_2d, _ = self.project_points(it["pts_world"], az_deg, el_deg, scale, offset)
                poly = pts_2d.astype(np.int32)
                cv2.fillPoly(img, [poly], it["color"])
                cv2.polylines(img, [poly], True, it["border_color"], it["border_thick"], cv2.LINE_AA)
            elif it["type"] == "stud":
                pt_2d, _ = self.project_points(it["pos_world"].reshape(1, 3), az_deg, el_deg, scale, offset)
                u_s, v_s = int(round(pt_2d[0, 0])), int(round(pt_2d[0, 1]))
                r_stud = max(2, int(round(scale * 4.2)))
                if 0 <= u_s < width and 0 <= v_s < height:
                    # Base stud ellipse
                    cv2.ellipse(img, (u_s, v_s), (r_stud, max(1, int(r_stud * 0.58))), 0, 0, 360, it["color"], -1, cv2.LINE_AA)
                    outline_col = (15, 23, 42) if it["is_active"] else (40, 50, 65)
                    cv2.ellipse(img, (u_s, v_s), (r_stud, max(1, int(r_stud * 0.58))), 0, 0, 360, outline_col, 1, cv2.LINE_AA)
                    if it["is_active"]:
                        # Highlight dot in center of active stud
                        cv2.circle(img, (u_s, v_s), max(1, r_stud // 2), (255, 255, 255), -1, cv2.LINE_AA)

        # Draw coupling vectors (dashed/accent line from active piece to parent piece)
        for p_pos, c_pos in coupling_lines_3d:
            pts_line = np.array([p_pos, c_pos], dtype=np.float32)
            pts_2d, _ = self.project_points(pts_line, az_deg, el_deg, scale, offset)
            p1 = (int(round(pts_2d[0, 0])), int(round(pts_2d[0, 1])))
            p2 = (int(round(pts_2d[1, 0])), int(round(pts_2d[1, 1])))
            cv2.line(img, p1, p2, (20, 184, 166), 1, cv2.LINE_AA)

        # UI Overlay Header
        badge_flip = " [FLIP / BASE INFERIOR]" if is_flip else ""
        header_text = f"Simulacion 3D LDraw - Digital Twin (Pag. {current_page_num}){badge_flip}"
        cv2.putText(img, header_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (241, 245, 249), 2, cv2.LINE_AA)
        cv2.putText(img, f"Perspectiva Isometrica (Az=45, El={int(el_deg)})", (20, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 116, 139), 1, cv2.LINE_AA)

        # UI Legend Footer
        cv2.circle(img, (24, height - 22), 6, (115, 128, 142), -1, cv2.LINE_AA)
        cv2.putText(img, "Piezas Base Previas (Ghost Context)", (36, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

        cv2.circle(img, (285, height - 22), 6, (30, 210, 150), -1, cv2.LINE_AA)
        cv2.putText(img, "Nuevas Piezas Activas (Paso Actual)", (297, height - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

        return img

    def render_critic_quad_view(
        self,
        all_nodes: List[Dict],
        tmpl_by_element: Optional[Dict[str, Dict]] = None,
        panel_w: int = 420,
        panel_h: int = 320,
    ) -> np.ndarray:
        """
        Renders a 2x2 Multi-View Critic Grid (Front, Side, Top, Isometric) with a 20-LDU stud
        coordinate ruler so a VLM Critic can inspect proportions, symmetry, and overhangs
        without 3D perspective ambiguity.
        """
        tmpl_by_element = tmpl_by_element or {}
        views = [
            ("FRONT VIEW (X-Y)", 0.0, 0.0),
            ("SIDE VIEW (Z-Y)", 90.0, 0.0),
            ("TOP VIEW (X-Z)", 0.0, 89.0),
            ("ISOMETRIC 3D (Az=45, El=30)", 45.0, 30.0),
        ]
        panels = []
        for label, az, el in views:
            canvas = np.full((panel_h, panel_w, 3), (20, 24, 34), dtype=np.uint8)
            # Draw 20-LDU ruler grid lines
            for gx in range(0, panel_w, 40):
                cv2.line(canvas, (gx, 0), (gx, panel_h), (34, 42, 56), 1)
            for gy in range(0, panel_h, 40):
                cv2.line(canvas, (0, gy), (panel_w, gy), (34, 42, 56), 1)

            if all_nodes:
                pts_list = []
                faces_to_draw = []
                for n in all_nodes:
                    tf = np.array(n["absolute_transform"], dtype=np.float32).reshape((4, 4))
                    R, t = tf[:3, :3], tf[:3, 3]
                    faces, _ = self.get_part_faces_and_studs(n["part_id"])
                    for f in faces:
                        pw = (f["pts"] @ R.T) + t
                        pts_list.append(pw)
                        faces_to_draw.append(pw)
                if pts_list:
                    all_pts = np.vstack(pts_list)
                    proj_0, _ = self.project_points(all_pts, az, el, scale=1.0, offset=(0.0, 0.0))
                    u_min, v_min = np.min(proj_0, axis=0)
                    u_max, v_max = np.max(proj_0, axis=0)
                    scale = min((panel_w * 0.72) / max(10.0, u_max - u_min), (panel_h * 0.72) / max(10.0, v_max - v_min))
                    off = ((panel_w - (u_max + u_min) * scale) / 2.0, (panel_h - (v_max + v_min) * scale) / 2.0)
                    for pw in faces_to_draw:
                        p2d, _ = self.project_points(pw, az, el, scale=scale, offset=off)
                        poly = p2d.astype(np.int32)
                        cv2.fillPoly(canvas, [poly], (165, 175, 188))
                        cv2.polylines(canvas, [poly], True, (30, 41, 59), 1, cv2.LINE_AA)

            cv2.putText(canvas, label, (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)
            panels.append(canvas)

        top_row = np.hstack([panels[0], panels[1]])
        bot_row = np.hstack([panels[2], panels[3]])
        return np.vstack([top_row, bot_row])

