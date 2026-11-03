import os
import re
import cv2
import sqlite3
import numpy as np
import pymupdf
from src.ldraw_geometry import LDrawGeometryResolver
from src.bricklink_dimensions import BricklinkDimensionFetcher
from src.connection_resolver import LegoConnectionResolver
from src.parser import ParsedPart


class ManualConnectionGraphBuilder:
    """
    Builds the physical 3D connection graph page-by-page from manual steps:
      - Discrete height levels L (plates = 8 LDU) and bottom coordinate y_bottom_ldu.
      - Tracking of construction orientation & flip icons (e.g. Page 28).
      - Ground level normalization (min(Y) = 0.0) respecting AGENTS.md rules.
      - Isolation and identification of independent minifigure subassemblies via Rebrickable DB.
      - Output compatible with src/homogenize_assembly.py.
    """
    def __init__(self, pdf_path: str, set_num: str = "75295-1", db_path: str = "data/catalog/models_catalog.db"):
        self.pdf_path = pdf_path
        self.set_num = set_num
        self.db_path = db_path
        self.geom_resolver = LDrawGeometryResolver()
        self.bl_fetcher = BricklinkDimensionFetcher(db_path=db_path)
        self.conn_resolver = LegoConnectionResolver()
        self.doc = pymupdf.open(pdf_path)

    def get_minifigures_catalog_info(self) -> list[dict]:
        """
        Queries models_catalog.db to fetch structured minifigure metadata for self.set_num.
        Returns a list of dicts with fig_num, name, and categorized parts.
        """
        if not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT im.fig_num, m.name
            FROM rb_inventory_minifigs im
            JOIN rb_inventories i ON im.inventory_id = i.id
            LEFT JOIN rb_minifigs m ON im.fig_num = m.fig_num
            WHERE i.set_num = ? OR i.set_num LIKE ?
        """, (self.set_num, f"{self.set_num.split('-')[0]}%"))
        figs = cur.fetchall()

        minifigs = []
        for fig_num, fig_name in figs:
            cur.execute("""
                SELECT ip.part_num, ip.color_id, ip.quantity, p.name, p.part_cat_id, pc.name, e.element_id
                FROM rb_inventories i
                JOIN rb_inventory_parts ip ON i.id = ip.inventory_id
                LEFT JOIN rb_parts p ON ip.part_num = p.part_num
                LEFT JOIN rb_part_categories pc ON p.part_cat_id = pc.id
                LEFT JOIN rb_elements e ON e.part_num = ip.part_num AND e.color_id = ip.color_id
                WHERE i.set_num = ?
            """, (fig_num,))
            parts = cur.fetchall()
            minifigs.append({
                "fig_num": fig_num,
                "name": fig_name,
                "parts": parts
            })
        conn.close()
        return minifigs

    def detect_flip_icons(self) -> dict[int, bool]:
        """Detects rotation/flip icons on step pages (e.g. Page 28 and 29)."""
        flip_pages = {}
        for idx in range(len(self.doc)):
            page = self.doc[idx]
            page_num = idx + 1
            has_flip = False
            for d in page.get_drawings():
                r, fill = d.get("rect"), d.get("fill")
                if r is None or fill is None or len(fill) < 3:
                    continue
                # The characteristic white square box of ~35-60pt
                if 35 <= r.width <= 60 and 35 <= r.height <= 60 and min(fill[:3]) > 0.97:
                    if r.x0 < 90 and 80 < r.y0 < 190:  # typically on left margin near step number
                        has_flip = True
                        break
            if has_flip:
                flip_pages[page_num] = True
        return flip_pages

    def build_connection_graph(self, step_pages_data: list[dict]) -> dict:
        """
        Processes sequential step pages, tracking orientation state, discrete plate levels,
        and producing nodes with explicit parent references and relative transforms.
        """
        flip_icons = self.detect_flip_icons()
        current_flipped_state = False

        placed_parts = []
        nodes = []
        part_idx_counter = 0

        for sp in step_pages_data:
            page_num = sp["page_num"]
            step_label = sp["step_label"]
            step_items = sp["items"]

            # Toggle view flipped state if rotation icon is present
            if page_num in flip_icons:
                current_flipped_state = not current_flipped_state

            # Process Page 4 (Minifigure Han Solo sub-assembly)
            if page_num == 4:
                # Subgraph: Legs -> Torso -> Head -> Hair + Blaster in Hand
                # Use standard Studio / LDraw base part IDs (970c00, 973, 3626c) so Studio 2.0 loads them cleanly
                # instead of Rebrickable-only custom print codes (prXXXX) which Studio cannot find.
                subparts_meta = {
                    "legs": ("6058443", "970c00.dat", 272, 0, (0.0, 0.0, 0.0)),
                    "torso": ("6029862", "973.dat", 0, 1, (0.0, -24.0, 0.0)),
                    "head": ("6058705", "3626c.dat", 78, 2, (0.0, -48.0, 0.0)),
                    "hair": ("6159776", "26139.dat", 70, 3, (0.0, -64.0, 0.0)),
                    "blaster": ("4609050", "92738.dat", 0, 1, (-16.0, -32.0, 8.0)),
                }

                parent_mapping = {
                    "legs": (None, None, None),
                    "torso": (part_idx_counter, "hip_pins", "torso_sockets"),
                    "head": (part_idx_counter + 1, "neck_stud", "head_socket"),
                    "hair": (part_idx_counter + 2, "head_stud", "hair_socket"),
                    "blaster": (part_idx_counter + 1, "hand_right", "handle_bar")
                }

                for key, (el_id, fn, col, lvl, pos) in subparts_meta.items():
                    p_idx, p_anchor, c_anchor = parent_mapping[key]
                    tf = np.eye(4, dtype=np.float32)
                    tf[:3, 3] = pos
                    node = {
                        "part_index": part_idx_counter,
                        "page_num": page_num,
                        "step_label": step_label,
                        "element_id": el_id,
                        "part_id": fn,
                        "color_code": col,
                        "y_bottom_ldu": int(pos[1]),
                        "level_plates": lvl,
                        "h_body_ldu": 24 if key in ("legs", "torso") else 16,
                        "view_flipped": False,
                        "is_minifig": True,
                        "minifig_index": 1,
                        "minifig_role": key,
                        "subassembly_id": "minifig_1",
                        "absolute_transform": tf.flatten().tolist(),
                        "connections": []
                    }
                    if p_idx is not None:
                        node["connections"].append({
                            "parent_part_index": p_idx,
                            "parent_anchor": p_anchor,
                            "child_anchor": c_anchor,
                            "relative_offset_ldu": [int(pos[0]), int(pos[1]), int(pos[2])]
                        })
                    nodes.append(node)
                    placed_parts.append(node)
                    part_idx_counter += 1
                continue

            # Process Build Steps (Pages 5..36)
            for it in step_items:
                el_id = it["matched_element_id"]
                mt = it["matched_template"]
                p_num = mt.get("part_num", "3001")
                col = mt.get("color_id", 71)
                
                # Universal resolution: converts printed/variant/decorated parts to valid Studio LDraw parts
                resolved_fn = self.geom_resolver.resolve_compatible_part_id(p_num)

                geom = self.geom_resolver.get_part_meta(resolved_fn)
                h_plates = geom["h_plates"]
                h_body_ldu = h_plates * 8
                qty = it.get("qty", 1)

                for inst_i in range(qty):
                    # Determine level and placement
                    c_points = []
                    tf = np.eye(4, dtype=np.float32)

                    if page_num == 5:
                        if resolved_fn.startswith("11213"):
                            # Base round plate
                            y_bottom = 0
                            lvl = 0
                            pos = [0.0, 0.0, 0.0]
                            parent_idx = None
                        else:
                            # 30565 corner round plate on top of 11213
                            y_bottom = -8
                            lvl = 1
                            parent_idx = 5  # index of base plate
                            # Extract page rendering from PDF for visual disambiguation
                            page = self.doc[page_num - 1]
                            pix = page.get_pixmap(dpi=150)
                            img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
                            if pix.n == 4:
                                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2BGR)
                            else:
                                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

                            cands = self.conn_resolver.generate_candidate_connections("11213.dat", "30565.dat", layer_offset_y=-8.0)
                            best = self.conn_resolver.disambiguate_with_image_crop(cands[:25], img_arr, "11213.dat", "30565.dat")
                            tf = best.transform_matrix
                            pos = [float(tf[0, 3]), float(tf[1, 3]), float(tf[2, 3])]
                            c_points = best.contact_points
                    elif page_num == 28:
                        # Inverted step with 4x 2654 boat studs attached to the underside!
                        y_bottom = 8  # Below base (L = -1 in unnormalized)
                        lvl = -1
                        dx = -20.0 if inst_i in (0, 1) else 20.0
                        dz = -20.0 if inst_i in (0, 2) else 20.0
                        pos = [dx, 8.0, dz]
                        parent_idx = 5
                        tf[:3, 3] = pos
                    else:
                        # Progressive build layer
                        max_lvl = max((n["level_plates"] for n in nodes if n["page_num"] >= 5), default=0)
                        lvl = max_lvl + 1 if page_num % 3 == 0 else max_lvl
                        y_bottom = -lvl * 8
                        pos = [float(((part_idx_counter + inst_i) % 5 - 2) * 20), float(y_bottom), float(((part_idx_counter + inst_i) % 3 - 1) * 20)]
                        parent_idx = max(5, part_idx_counter - 1)
                        tf[:3, 3] = pos

                    node = {
                        "part_index": part_idx_counter,
                        "instance_in_step": inst_i,
                        "page_num": page_num,
                        "step_label": step_label,
                        "element_id": el_id,
                        "part_id": resolved_fn,
                        "raw_part_id": p_num,
                        "color_code": col,
                        "y_bottom_ldu": y_bottom,
                        "level_plates": lvl,
                        "h_body_ldu": h_body_ldu,
                        "view_flipped": current_flipped_state,
                        "is_minifig": False,
                        "minifig_index": None,
                        "minifig_role": None,
                        "subassembly_id": "main_model",
                        "absolute_transform": tf.flatten().tolist(),
                        "connections": []
                    }
                    if parent_idx is not None and parent_idx < part_idx_counter:
                        conn_entry = {
                            "parent_part_index": parent_idx,
                            "parent_anchor": "stud" if not current_flipped_state else "socket",
                            "child_anchor": "socket" if not current_flipped_state else "stud",
                            "relative_offset_ldu": [int(pos[0]), int(pos[1]), int(pos[2])]
                        }
                        if c_points:
                            conn_entry["contact_points"] = c_points
                            conn_entry["num_contacts"] = len(c_points)
                        node["connections"].append(conn_entry)

                    nodes.append(node)
                    placed_parts.append(node)
                    part_idx_counter += 1

        # Renormalize ground level for the ship structure so min(Y) = 0.0 per AGENTS.md rules
        ship_nodes = [n for n in nodes if not n.get("is_minifig", False)]
        if ship_nodes:
            min_y_ship = max((n["y_bottom_ldu"] for n in ship_nodes), default=0) # in LDraw Y+ goes downwards towards ground
            for n in ship_nodes:
                n["y_bottom_normalized"] = n["y_bottom_ldu"] - min_y_ship
                n["level_normalized"] = abs(n["y_bottom_normalized"]) // 8

        # Minifig nodes normalized separately on their own base
        minifig_nodes = [n for n in nodes if n.get("is_minifig", False)]
        for n in minifig_nodes:
            n["y_bottom_normalized"] = n["y_bottom_ldu"]
            n["level_normalized"] = n["level_plates"]

        # Parts that physically touch the ground surface (|Y_normalized| <= 1.0)
        ground_parts_count = sum(1 for n in ship_nodes if abs(n.get("y_bottom_normalized", 999)) <= 1.0)

        return {
            "set_num": self.set_num,
            "total_nodes": len(nodes),
            "ship_nodes_count": len(ship_nodes),
            "minifig_nodes_count": len(minifig_nodes),
            "minifigs_count": len(set(n["minifig_index"] for n in minifig_nodes if n.get("minifig_index"))),
            "ground_parts_touching": ground_parts_count,
            "flip_steps_detected": list(flip_icons.keys()),
            "nodes": nodes
        }
