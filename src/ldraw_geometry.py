import os
import re
import sqlite3
import numpy as np
from scipy.spatial import cKDTree


class LDrawGeometryResolver:
    """
    Parses official LDraw parts (.dat) and extracts:
      - Exact bounding box in LDU integers [xmin, ymin, zmin, xmax, ymax, zmax].
      - Dimensions in LDU and plate heights (h_body = ymax - ymin, h_plates = round(h_body / 8)).
      - Physical connection ports: Studs (top/lateral) and Sockets/Antistuds.
      - Generalized fallback resolution for printed, decorated, or variant parts not natively in LDraw/Studio.
    """
    def __init__(self, ldraw_base_dir: str = "data/ldraw/ldraw", db_path: str = "data/catalog/models_catalog.db"):
        self.ldraw_base_dir = ldraw_base_dir
        self.db_path = db_path
        self.parts_dir = os.path.join(ldraw_base_dir, "parts")
        self.p_dir = os.path.join(ldraw_base_dir, "p")
        
        # Also include system Studio 2.0 LDraw library if present on macOS
        self.studio_ldraw_dir = "/Applications/Studio 2.0/ldraw"
        self.studio_parts_dir = os.path.join(self.studio_ldraw_dir, "parts") if os.path.exists(self.studio_ldraw_dir) else None
        self.studio_p_dir = os.path.join(self.studio_ldraw_dir, "p") if os.path.exists(self.studio_ldraw_dir) else None

        self.cache = {}
        self.faces_cache = {}
        self.part_resolution_cache = {}

        # M4 Hardware Optimization: In-Memory File Indexing (O(1) memory lookup, zero disk exists checks)
        self._file_index: dict[str, str] = {}
        self._build_in_memory_file_index()

    def _build_in_memory_file_index(self):
        """Scans LDraw libraries once at startup and caches full filepaths in memory."""
        scan_dirs = [self.ldraw_base_dir]
        if self.studio_ldraw_dir and os.path.exists(self.studio_ldraw_dir):
            scan_dirs.append(self.studio_ldraw_dir)

        for base_d in scan_dirs:
            if not os.path.exists(base_d):
                continue
            for root, _, files in os.walk(base_d):
                for f in files:
                    fl = f.lower()
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, base_d).replace("\\", "/").lower()
                    if rel_p not in self._file_index:
                        self._file_index[rel_p] = full_p
                    if fl not in self._file_index:
                        self._file_index[fl] = full_p

    def _resolve_file(self, filename: str) -> str:
        """Finds file via in-memory O(1) index, falling back to disk checks only if unindexed."""
        if not filename:
            return None
        fn_clean = filename.replace("\\", "/").strip().lower()

        # 1. Fast O(1) memory lookup
        if fn_clean in self._file_index:
            return self._file_index[fn_clean]

        base_name = os.path.basename(fn_clean)
        if base_name in self._file_index:
            return self._file_index[base_name]

        # 2. Disk fallback (rarely needed)
        candidates = [
            os.path.join(self.parts_dir, fn_clean),
            os.path.join(self.p_dir, fn_clean),
            os.path.join(self.parts_dir, "s", fn_clean),
            os.path.join(self.p_dir, "8", fn_clean),
            os.path.join(self.p_dir, "48", fn_clean),
        ]
        if self.studio_parts_dir:
            candidates.extend([
                os.path.join(self.studio_parts_dir, fn_clean),
                os.path.join(self.studio_p_dir, fn_clean),
                os.path.join(self.studio_parts_dir, "s", fn_clean),
            ])

        for c in candidates:
            if os.path.exists(c):
                self._file_index[fn_clean] = c
                return c
        return None

    def resolve_compatible_part_id(self, part_num: str) -> str:
        """
        Universally resolves any LEGO part identifier (including Rebrickable printed codes prXXXX,
        alternate molds, or decorated variants) to a valid, renderable LDraw .dat file supported by Studio.

        Resolution Cascade:
          1. Direct existence in LDraw / Studio library.
          2. Rebrickable database relationships (rb_part_relationships: Print 'P', Alternate 'A', Mold 'M').
          3. Regex pattern stripping (strips prXXXX, pbXXXX, pXXXX suffixes to reach base mold).
          4. Category-level structural fallback (e.g. Minifig Legs -> 970c00.dat, Torso -> 973.dat, Head -> 3626c.dat).
          5. Fallback to original filename.
        """
        raw = part_num.lower().strip()
        if raw.endswith(".dat"):
            raw = raw[:-4]

        if raw in self.part_resolution_cache:
            return self.part_resolution_cache[raw]

        fn = f"{raw}.dat"

        # 1. Direct match check
        if self._resolve_file(fn):
            self.part_resolution_cache[raw] = fn
            return fn

        # 2. Check Rebrickable relationships and Element ID mappings
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()

                # Check if raw is a numeric element_id (e.g. 4282789 -> part_num 54384)
                cur.execute("SELECT part_num FROM rb_elements WHERE element_id = ?", (raw,))
                el_row = cur.fetchone()
                if el_row and el_row[0]:
                    el_part = str(el_row[0]).lower().strip()
                    el_fn = f"{el_part}.dat"
                    if self._resolve_file(el_fn):
                        conn.close()
                        self.part_resolution_cache[raw] = el_fn
                        return el_fn

                cur.execute("""
                    SELECT parent_part_num, rel_type 
                    FROM rb_part_relationships 
                    WHERE child_part_num = ? 
                    ORDER BY CASE rel_type WHEN 'P' THEN 1 WHEN 'A' THEN 2 WHEN 'M' THEN 3 ELSE 4 END
                """, (raw,))
                for parent, _ in cur.fetchall():
                    parent_fn = f"{parent.lower()}.dat"
                    if self._resolve_file(parent_fn):
                        conn.close()
                        self.part_resolution_cache[raw] = parent_fn
                        return parent_fn

                # 3. Minifig category heuristics from rb_parts
                cur.execute("SELECT part_cat_id FROM rb_parts WHERE part_num = ?", (raw,))
                row = cur.fetchone()
                if row:
                    cat_id = row[0]
                    if cat_id == 61 or raw.startswith("970c"):  # Minifig Lower Body (Legs)
                        conn.close()
                        self.part_resolution_cache[raw] = "970c00.dat"
                        return "970c00.dat"
                    elif cat_id == 60 or raw.startswith("973c"):  # Minifig Upper Body (Torso with Arms and Hands)
                        conn.close()
                        self.part_resolution_cache[raw] = "973c01.dat"
                        return "973c01.dat"
                    elif cat_id == 59:  # Minifig Heads
                        conn.close()
                        self.part_resolution_cache[raw] = "3626c.dat"
                        return "3626c.dat"
                conn.close()
            except Exception:
                pass

        # 4. Regex print / pattern stripping heuristic (e.g. 3069bpr0001 -> 3069b.dat)
        stripped = re.sub(r'(pr|pb|p)\d+.*$', '', raw, flags=re.IGNORECASE)
        if stripped and stripped != raw:
            stripped_fn = f"{stripped.lower()}.dat"
            if self._resolve_file(stripped_fn):
                self.part_resolution_cache[raw] = stripped_fn
                return stripped_fn

        # Default fallback
        self.part_resolution_cache[raw] = fn
        return fn

    def get_part_meta(self, part_num: str) -> dict:
        """Returns exact LDU bounding box, body height, and ports for part_num."""
        fn = part_num.lower().strip()
        if not fn.endswith(".dat"):
            fn = f"{fn}.dat"

        # Resolve compatible or base part if fn is not directly resolvable
        if not self._resolve_file(fn):
            resolved_fn = self.resolve_compatible_part_id(part_num)
            if self._resolve_file(resolved_fn):
                fn = resolved_fn

        if fn in self.cache:
            return self.cache[fn]

        filepath = self._resolve_file(fn)
        if not filepath:
            # Fallback for unregistered or minifig elements
            meta = {
                "part_id": fn,
                "bbox": [-10, -8, -10, 10, 0, 10],
                "dim_ldu": (20, 8, 20),
                "h_plates": 1,
                "stud_dim": (1, 1, 1),
                "studs": [np.array([0.0, -4.0, 0.0])],
                "sockets": [np.array([0.0, 0.0, 0.0])],
                "has_lateral_ports": False,
                "symmetry_group": "C1",
                "symmetry_order": 1,
                "symmetry_yaw_angles": [0.0],
                "is_symmetric": False
            }
            self.cache[fn] = meta
            return meta

        vertices = []
        studs = []
        sockets = []
        holes = []
        pins = []
        edges_type2 = []
        has_lateral = False

        def parse_file(curr_path: str, transform: np.ndarray, depth: int = 0):
            nonlocal has_lateral
            if depth > 8 or not os.path.exists(curr_path):
                return

            with open(curr_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    t = line.strip().split()
                    if not t:
                        continue
                    cmd = t[0]

                    # Line type 1: Subfile / Primitive call
                    # 1 <color> x y z a b c d e f g h i <subfile>
                    if cmd == "1" and len(t) >= 15:
                        sub_fn = t[14].lower()
                        tx, ty, tz = float(t[2]), float(t[3]), float(t[4])
                        rot = np.array([float(v) for v in t[5:14]], dtype=np.float32).reshape(3, 3)

                        sub_tf = np.eye(4, dtype=np.float32)
                        sub_tf[:3, 3] = [tx, ty, tz]
                        sub_tf[:3, :3] = rot
                        combined_tf = transform @ sub_tf

                        # Check for stud primitives
                        if "stud" in sub_fn or "stug" in sub_fn:
                            if "stug" in sub_fn:
                                resolved = self._resolve_file(sub_fn)
                                if resolved:
                                    parse_file(resolved, combined_tf, depth + 1)
                            else:
                                pos = combined_tf[:3, 3]
                                norm = combined_tf[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
                                studs.append({"pos": pos, "normal": norm})
                                if abs(norm[1]) < 0.8:
                                    has_lateral = True
                        elif any(h in sub_fn for h in ["beamhole", "connhole", "peghole", "axlehol", "hole"]):
                            h_pos = combined_tf[:3, 3]
                            h_axis = combined_tf[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
                            norm_val = np.linalg.norm(h_axis)
                            if norm_val > 1e-4:
                                h_axis = h_axis / norm_val
                            holes.append({"pos": h_pos, "axis": h_axis, "type": sub_fn})
                            if abs(h_axis[1]) < 0.8:
                                has_lateral = True
                        elif any(p in sub_fn for p in ["confric", "connect", "pin"]):
                            p_pos = combined_tf[:3, 3]
                            p_axis = combined_tf[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
                            norm_val = np.linalg.norm(p_axis)
                            if norm_val > 1e-4:
                                p_axis = p_axis / norm_val
                            pins.append({"pos": p_pos, "axis": p_axis, "type": sub_fn})
                            if abs(p_axis[1]) < 0.8:
                                has_lateral = True
                        elif "sub" in sub_fn or "s\\" in sub_fn or sub_fn.endswith(".dat"):
                            resolved = self._resolve_file(sub_fn)
                            if resolved:
                                parse_file(resolved, combined_tf, depth + 1)

                    # Line type 2: Explicit wireframe line segment (2 <color> x1 y1 z1 x2 y2 z2)
                    elif cmd == "2" and len(t) >= 8:
                        p1 = (transform @ np.array([float(t[2]), float(t[3]), float(t[4]), 1.0], dtype=np.float32))[:3]
                        p2 = (transform @ np.array([float(t[5]), float(t[6]), float(t[7]), 1.0], dtype=np.float32))[:3]
                        if float(np.linalg.norm(p2 - p1)) >= 3.5:
                            edges_type2.append((p1, p2))

                    # Line types 3 (triangle), 4 (quadrilateral) contain vertices
                    elif cmd in ("3", "4") and len(t) >= 10:
                        # Extract 3D vertices
                        pts_count = 3 if cmd == "3" else 4
                        for i in range(pts_count):
                            vx, vy, vz = float(t[2 + i*3]), float(t[3 + i*3]), float(t[4 + i*3])
                            pt_w = (transform @ np.array([vx, vy, vz, 1.0], dtype=np.float32))[:3]
                            vertices.append(pt_w)

        initial_tf = np.eye(4, dtype=np.float32)
        parse_file(filepath, initial_tf, 0)

        if vertices:
            v_arr = np.array(vertices, dtype=np.float32)
            # Filter studs from bounding box height (studs typically protrude by -4 LDU in Y)
            # Standard body Y is between [ymin, ymax]
            xmin, ymin, zmin = np.min(v_arr, axis=0)
            xmax, ymax, zmax = np.max(v_arr, axis=0)

            # Round to nearest LDU
            xmin, ymin, zmin = int(round(xmin)), int(round(ymin)), int(round(zmin))
            xmax, ymax, zmax = int(round(xmax)), int(round(ymax)), int(round(zmax))
        else:
            xmin, ymin, zmin = -10, -8, -10
            xmax, ymax, zmax = 10, 0, 10

        w_ldu = max(1, xmax - xmin)
        h_ldu = max(1, ymax - ymin)
        d_ldu = max(1, zmax - zmin)

        # Height in plates (1 plate = 8 LDU)
        h_plates = max(1, int(round(h_ldu / 8.0)))
        w_studs = max(1, int(round(w_ldu / 20.0)))
        d_studs = max(1, int(round(d_ldu / 20.0)))

        # Clean/deduplicate studs and filter out underside mold/tube primitives
        unique_studs = []
        for s in studs:
            p = s["pos"]
            norm = s["normal"]
            # Exclude inner mold tubes pointing downwards into bottom cavity (+Y in LDraw)
            is_underside_tube = (norm[1] > 0.5 and p[1] > 0.5)
            if is_underside_tube:
                continue

            is_dup = False
            for us in unique_studs:
                if np.linalg.norm(us["pos"] - p) < 3.0:
                    is_dup = True
                    break
            if not is_dup:
                unique_studs.append(s)

        # Deduplicate holes
        unique_holes = []
        for h in holes:
            hp = h["pos"]
            is_dup = False
            for uh in unique_holes:
                if np.linalg.norm(uh["pos"] - hp) < 2.0:
                    is_dup = True
                    break
            if not is_dup:
                unique_holes.append(h)

        # Deduplicate pins
        unique_pins = []
        for p in pins:
            pp = p["pos"]
            is_dup = False
            for up in unique_pins:
                if np.linalg.norm(up["pos"] - pp) < 2.0:
                    is_dup = True
                    break
            if not is_dup:
                unique_pins.append(p)

        # Generalized Physical Footprint & True Socket Detection
        # Rather than assuming a naive rectangular bounding box grid, compute the true
        # 2D cross-section footprint from 3D vertices to filter out fake/ghost sockets
        # for round plates, corner round plates, wedges, slopes, and parts with central holes.
        inferred_sockets = []
        hull_2d = None
        has_center_hole = "hole" in fn or any("hole" in s.get("pos", "") if isinstance(s, dict) else False for s in [])

        if vertices:
            v_xz = np.array([[v[0], v[2]] for v in vertices], dtype=np.float32)
            try:
                import cv2
                hull_2d = cv2.convexHull(v_xz)
            except Exception:
                hull_2d = None

        for ix in range(w_studs):
            sx = float(xmin + 10 + ix * 20)
            for iz in range(d_studs):
                sz = float(zmin + 10 + iz * 20)
                
                # Boundary verification against real solid geometry
                if hull_2d is not None and len(hull_2d) >= 3:
                    import cv2
                    dist = cv2.pointPolygonTest(hull_2d, (sx, sz), measureDist=True)
                    # Sockets require physical wall thickness margin (at least 3.5 LDU inside outer perimeter)
                    if dist < 3.5:
                        continue
                    # Check internal center hole (e.g. 11213 round plate with hole)
                    if "hole" in fn:
                        r_sq = sx**2 + sz**2
                        if r_sq < 18.0**2:  # Inside central 2x2 hole void
                            continue

                inferred_sockets.append(np.array([sx, ymax, sz], dtype=np.float32))

        # Add bottom-plane footprint sockets for inverted slopes (3747b, 3660, 3665, 4287) and center-tube anti-studs
        if vertices:
            bot_verts = [v for v in vertices if abs(v[1] - ymax) <= 4.5]
            if bot_verts:
                b_arr = np.array(bot_verts, dtype=np.float32)
                b_xmin, b_zmin = int(round(np.min(b_arr[:, 0]))), int(round(np.min(b_arr[:, 2])))
                b_xmax, b_zmax = int(round(np.max(b_arr[:, 0]))), int(round(np.max(b_arr[:, 2])))
                bw_s = max(1, int(round((b_xmax - b_xmin) / 20.0)))
                bd_s = max(1, int(round((b_zmax - b_zmin) / 20.0)))
                for b_ix in range(bw_s):
                    bsx = float(b_xmin + 10 + b_ix * 20)
                    for b_iz in range(bd_s):
                        bsz = float(b_zmin + 10 + b_iz * 20)
                        cand_sock = np.array([bsx, float(ymax), bsz], dtype=np.float32)
                        if not any(np.linalg.norm(cand_sock - ex_s) < 2.0 for ex_s in inferred_sockets):
                            inferred_sockets.append(cand_sock)

        # Add secondary center-tube anti-stud sockets for multi-stud plates/bricks/slopes
        if w_studs >= 2 or d_studs >= 2:
            for ix in range(max(1, w_studs - 1)):
                cx_tube = float(xmin + 20 * (ix + 1))
                for iz in range(max(1, d_studs - 1)):
                    cz_tube = float(zmin + 20 * (iz + 1))
                    tube_sock = np.array([cx_tube, float(ymax), cz_tube], dtype=np.float32)
                    if not any(np.linalg.norm(tube_sock - ex_s) < 2.0 for ex_s in inferred_sockets):
                        inferred_sockets.append(tube_sock)
            # Also center-line sockets along the long axis for 1xN or 2xN or 3x2 parts
            for sx_c in (0.0, -10.0, 10.0):
                for sz_c in (0.0, -10.0, 10.0):
                    if xmin + 4.0 <= sx_c <= xmax - 4.0 and zmin + 4.0 <= sz_c <= zmax - 4.0:
                        c_sock = np.array([sx_c, float(ymax), sz_c], dtype=np.float32)
                        if not any(np.linalg.norm(c_sock - ex_s) < 2.0 for ex_s in inferred_sockets):
                            inferred_sockets.append(c_sock)

        # Ensure jumper plates (15573, 3794, 34103, 88646) include fractional center socket & center stud
        clean_base = fn.replace(".dat", "")
        if any(clean_base.startswith(j) for j in ("15573", "3794", "34103", "88646")):
            inferred_sockets.append(np.array([0.0, float(ymax), 0.0], dtype=np.float32))
            if not any(np.linalg.norm(s["pos"] - np.array([0.0, float(ymin), 0.0])) < 2.0 for s in unique_studs):
                unique_studs.append({
                    "pos": np.array([0.0, float(ymin), 0.0], dtype=np.float32),
                    "normal": np.array([0.0, -1.0, 0.0], dtype=np.float32)
                })

        # Ensure SNOT / modified / bracket / Technic parts flag has_lateral and provide lateral ports if missing
        snot_prefixes = (
            "10197", "10247", "11211", "11212", "11214", "11215", "11458", "11476",
            "61252", "85984", "2436", "6536", "6632", "32828", "27940", "3023b",
            "3709b", "26604", "4070", "87087", "44728", "99780", "99781", "36840",
            "36841", "30414", "32952", "22885", "4733", "47905", "99207", "73825"
        )
        if any(clean_base.startswith(sp) for sp in snot_prefixes):
            has_lateral = True
            # Only synthesize 4-sided generic lateral studs if the LDraw model does not already define explicit lateral studs
            has_parsed_lateral_studs = any(abs(s["normal"][1]) < 0.8 for s in unique_studs)
            mid_y = float((ymin + ymax) / 2.0)
            for lat_norm, lat_pos in [
                (np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, mid_y, float(zmin)], dtype=np.float32)),
                (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, mid_y, float(zmax)], dtype=np.float32)),
                (np.array([-1.0, 0.0, 0.0], dtype=np.float32), np.array([float(xmin), mid_y, 0.0], dtype=np.float32)),
                (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([float(xmax), mid_y, 0.0], dtype=np.float32)),
            ]:
                if not has_parsed_lateral_studs:
                    if not any(np.linalg.norm(s["pos"] - lat_pos) < 4.0 for s in unique_studs):
                        unique_studs.append({"pos": lat_pos, "normal": lat_norm})

        if not inferred_sockets:
            inferred_sockets.append(np.array([0.0, float(ymax), 0.0], dtype=np.float32))

        # Functional anchor points (Corners, Center, Symmetry)
        sharp_corners = []
        if vertices:
            v_all = np.array(vertices, dtype=np.float32)
            # Find extreme corners
            for ex in [xmin, xmax]:
                for ez in [zmin, zmax]:
                    dists = np.linalg.norm(v_all[:, [0, 2]] - np.array([ex, ez]), axis=1)
                    if np.min(dists) < 2.0:
                        sharp_corners.append(np.array([float(ex), float(ymin), float(ez)], dtype=np.float32))

        # 3D Rotational Symmetry Analysis around normal Y axis
        # Analyzes discrete yaw rotations (0°, 90°, 180°, 270°) around geometric center
        sym_yaw_angles = [0.0]
        if vertices:
            v_all = np.array(vertices, dtype=np.float32)
            cx = (xmin + xmax) / 2.0
            cz = (zmin + zmax) / 2.0
            v_centered = v_all.copy()
            v_centered[:, 0] -= cx
            v_centered[:, 2] -= cz

            tree = cKDTree(v_centered)
            sample_pts = v_centered[::max(1, len(v_centered) // 150)]

            cand_angles = [180.0]
            if abs(w_ldu - d_ldu) <= 2.0:
                cand_angles.extend([90.0, 270.0])

            for deg in cand_angles:
                rad = np.radians(deg)
                R = np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]], dtype=np.float32)
                rot_pts = sample_pts @ R.T
                dists, _ = tree.query(rot_pts)
                if np.mean(dists) < 1.0 and np.max(dists) < 3.5:
                    sym_yaw_angles.append(deg)

        sym_yaw_angles = sorted(sym_yaw_angles)
        sym_order = len(sym_yaw_angles)
        sym_group = f"C{sym_order}" if sym_order in (1, 2, 4) else f"Sym{sym_order}"
        is_symmetric = sym_order > 1

        meta = {
            "part_id": fn,
            "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
            "dim_ldu": (w_ldu, h_ldu, d_ldu),
            "h_plates": h_plates,
            "stud_dim": (w_studs, d_studs, h_plates),
            "studs": [s["pos"] for s in unique_studs],
            "stud_details": unique_studs,
            "sockets": inferred_sockets,
            "holes": unique_holes,
            "pins": unique_pins,
            "edges_type2": edges_type2[:128],
            "has_lateral_ports": has_lateral,
            "sharp_corners": sharp_corners,
            "has_center_hole": "hole" in fn,
            "symmetry_group": sym_group,
            "symmetry_order": sym_order,
            "symmetry_yaw_angles": sym_yaw_angles,
            "is_symmetric": is_symmetric
        }
        self.cache[fn] = meta
        return meta

    def get_part_edges_type2(self, part_num: str) -> list:
        """Returns explicit LDraw line-type 2 wireframe segments for the part."""
        meta = self.get_part_meta(part_num)
        return meta.get("edges_type2", [])

    def get_part_faces(self, part_num: str) -> list:
        """
        Extracts 3D polygonal faces and normals from official LDraw commands (types 3 & 4),
        recursively parsing subfiles (s\\*.dat, etc.) while skipping stud primitives so that
        the true solid polygonal geometry (slopes, wedges, wings, round shapes) is obtained.
        """
        fn = part_num.lower().strip()
        if not fn.endswith(".dat"):
            fn = f"{fn}.dat"

        # Resolve compatible or base part if fn is not directly resolvable
        if not self._resolve_file(fn):
            resolved_fn = self.resolve_compatible_part_id(part_num)
            if self._resolve_file(resolved_fn):
                fn = resolved_fn

        if fn in self.faces_cache:
            return self.faces_cache[fn]

        filepath = self._resolve_file(fn)
        if not filepath:
            self.faces_cache[fn] = []
            return []

        faces = []

        def parse_faces(curr_path: str, transform: np.ndarray, depth: int = 0):
            if depth > 10 or not curr_path or not os.path.exists(curr_path):
                return

            with open(curr_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    t = line.strip().split()
                    if not t:
                        continue
                    cmd = t[0]

                    # Line type 1: Subfile / Primitive call
                    if cmd == "1" and len(t) >= 15:
                        sub_fn = t[14].lower()
                        # Skip stud micro-primitives from body face mesh (studs are rendered separately)
                        if "stud" in sub_fn:
                            continue
                        tx, ty, tz = float(t[2]), float(t[3]), float(t[4])
                        rot = np.array([float(v) for v in t[5:14]], dtype=np.float32).reshape(3, 3)
                        sub_tf = np.eye(4, dtype=np.float32)
                        sub_tf[:3, 3] = [tx, ty, tz]
                        sub_tf[:3, :3] = rot
                        combined_tf = transform @ sub_tf

                        res_file = self._resolve_file(sub_fn)
                        if res_file:
                            parse_faces(res_file, combined_tf, depth + 1)

                    # Line types 3 (triangle) and 4 (quadrilateral)
                    elif cmd in ("3", "4") and len(t) >= 10:
                        pts_cnt = 3 if cmd == "3" else 4
                        pts = []
                        for i in range(pts_cnt):
                            vx, vy, vz = float(t[2 + i*3]), float(t[3 + i*3]), float(t[4 + i*3])
                            pt_w = (transform @ np.array([vx, vy, vz, 1.0], dtype=np.float32))[:3]
                            pts.append(pt_w)
                        pts = np.array(pts, dtype=np.float32)
                        v1 = pts[1] - pts[0]
                        v2 = pts[2] - pts[0]
                        normal = np.cross(v1, v2)
                        norm_len = np.linalg.norm(normal)
                        if norm_len > 1e-6:
                            normal = normal / norm_len
                        else:
                            normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        faces.append({"pts": pts, "normal": normal})

        initial_tf = np.eye(4, dtype=np.float32)
        parse_faces(filepath, initial_tf, 0)
        self.faces_cache[fn] = faces
        return faces

    # ------------------------------------------------------------------
    # Typed port extraction  (Paso 1 — connector-compatibility gating)
    # ------------------------------------------------------------------

    def get_part_ports(self, part_num: str) -> list:
        """
        Returns a ``List[ConnectionPort]`` with fully-typed physical connection
        ports for *part_num*, extracted from the LDraw geometry.

        Port types are inferred from the LDraw primitive sub-file names that
        ``get_part_meta`` already parses internally:

        +-----------------------+-----------------------------+----------+
        | LDraw primitive       | PortType assigned           | Gender   |
        +-----------------------+-----------------------------+----------+
        | stud.dat / stud2.dat  | STUD  (solid)               | male     |
        | stud2.dat             | STUD_HOLLOW (hollow stud)   | male     |
        | stug*.dat             | aggregated as STUD          | male     |
        | anti-stud / socket    | ANTISTUD                    | female   |
        | peghole / connhole    | PIN_HOLE                    | female   |
        | axlehole / axlehol    | AXLE_HOLE                   | female   |
        | beamhole              | PIN_HOLE                    | female   |
        | confric / connect/pin | PIN                         | male     |
        +-----------------------+-----------------------------+----------+

        All positions and normals are in the part's local LDraw coordinate frame
        (LDU).  Normals point *away* from the connection face (outward convention).

        The result is cached inside ``self.cache`` under the key
        ``<part_id>__ports`` so repeated calls are O(1).

        Parameters
        ----------
        part_num : str
            Part identifier (with or without ``.dat`` extension).

        Returns
        -------
        List[ConnectionPort]
            Typed, deduplicated list of all physical connection ports.
        """
        from src.connectivity_db import ConnectionPort, PortType, JointDOF

        fn = part_num.lower().strip()
        if not fn.endswith(".dat"):
            fn = f"{fn}.dat"

        cache_key = f"{fn}__ports"
        if cache_key in self.cache:
            return self.cache[cache_key]

        meta = self.get_part_meta(fn)
        ports: list = []
        port_id_counter = [0]  # mutable counter for nested helper

        def _next_id(prefix: str) -> str:
            port_id_counter[0] += 1
            return f"{fn[:-4]}_{prefix}_{port_id_counter[0]:03d}"

        # ── 1. Studs (male) ────────────────────────────────────────────────
        # stud_details carries {"pos": np.array, "normal": np.array, "type": str}
        # The "type" key is the sub-file name (stud.dat, stud2.dat, etc.)
        stud_details = meta.get("stud_details", [])
        # Fallback: if stud_details is not populated, synthesise from plain studs list
        if not stud_details:
            for s_pos in meta.get("studs", []):
                pos = np.array(s_pos, dtype=np.float32) if not isinstance(s_pos, dict) else np.array(s_pos["pos"], dtype=np.float32)
                stud_details.append({"pos": pos, "normal": np.array([0.0, -1.0, 0.0], dtype=np.float32), "type": "stud.dat"})

        for s in stud_details:
            pos = np.array(s["pos"], dtype=np.float32)
            norm = np.array(s.get("normal", [0.0, -1.0, 0.0]), dtype=np.float32)
            sub_fn = s.get("type", "stud.dat").lower()
            # stud2.dat / stud2a.dat are hollow studs (can receive a bar)
            if "stud2" in sub_fn or "stud4" in sub_fn:
                p_type = PortType.STUD_HOLLOW
            else:
                p_type = PortType.STUD
            # lateral stud: DOF revolute around its axis
            is_lateral = abs(float(norm[1])) < 0.8
            dof = JointDOF.REVOLUTE if is_lateral else JointDOF.FIXED
            ports.append(ConnectionPort(
                id=_next_id("stud"),
                port_type=p_type,
                gender="male",
                local_pos=pos,
                local_normal=norm,
                dof=dof,
                symmetry=4,
                clutch_power=2.0,
            ))

        # ── 2. Anti-stud sockets (female) ─────────────────────────────────
        bbox = meta.get("bbox", [-10, 0, -10, 10, 8, 10])
        ymax_b = float(bbox[4])
        for s_pos in meta.get("sockets", []):
            pos = np.array(s_pos, dtype=np.float32)
            # Bottom sockets lie on the bottom face (y ~= ymax) and face downward (+Y in LDraw).
            # Lateral sockets on SNOT parts lie at mid_y and face outward along X or Z.
            if abs(float(pos[1]) - ymax_b) <= 2.5:
                norm = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                if abs(float(pos[0])) >= abs(float(pos[2])):
                    norm = np.array([1.0 if float(pos[0]) >= 0.0 else -1.0, 0.0, 0.0], dtype=np.float32)
                else:
                    norm = np.array([0.0, 0.0, 1.0 if float(pos[2]) >= 0.0 else -1.0], dtype=np.float32)
            ports.append(ConnectionPort(
                id=_next_id("antistud"),
                port_type=PortType.ANTISTUD,
                gender="female",
                local_pos=pos,
                local_normal=norm,
                dof=JointDOF.FIXED,
                symmetry=4,
                clutch_power=2.0,
            ))

        # ── 3. Holes: axle-holes, pin-holes, beam-holes (female) ──────────
        for h in meta.get("holes", []):
            pos = np.array(h["pos"], dtype=np.float32)
            axis = np.array(h.get("axis", [0.0, 1.0, 0.0]), dtype=np.float32)
            sub_fn = h.get("type", "").lower()
            if "axle" in sub_fn:
                p_type = PortType.AXLE_HOLE
                dof = JointDOF.PRISMATIC
            else:
                p_type = PortType.PIN_HOLE
                dof = JointDOF.REVOLUTE
            ports.append(ConnectionPort(
                id=_next_id("hole"),
                port_type=p_type,
                gender="female",
                local_pos=pos,
                local_normal=axis,
                dof=dof,
                symmetry=1,
                clutch_power=1.5,
            ))

        # ── 4. Pins / connectors (male) ────────────────────────────────────
        for p in meta.get("pins", []):
            pos = np.array(p["pos"], dtype=np.float32)
            axis = np.array(p.get("axis", [0.0, 1.0, 0.0]), dtype=np.float32)
            sub_fn = p.get("type", "").lower()
            if "axle" in sub_fn:
                p_type = PortType.AXLE
                dof = JointDOF.PRISMATIC
            else:
                p_type = PortType.PIN
                dof = JointDOF.REVOLUTE
            ports.append(ConnectionPort(
                id=_next_id("pin"),
                port_type=p_type,
                gender="male",
                local_pos=pos,
                local_normal=axis,
                dof=dof,
                symmetry=1,
                clutch_power=1.5,
            ))

        self.cache[cache_key] = ports
        return ports

    def get_compatible_port_pairs(
        self,
        child_part_num: str,
        parent_part_num: str,
    ) -> list:
        """
        Returns all physically compatible (parent_port, child_port) ``ConnectionPort``
        pairs between *parent_part_num* and *child_part_num*, using the compatibility
        table defined in ``connectivity_db.COMPATIBLE_PORT_PAIRS``.

        This is the entry-point for connector-compatibility gating (Paso 2-3):
        only pairs returned here are valid candidate attachment points.

        Returns
        -------
        List[Tuple[ConnectionPort, ConnectionPort]]
            Each tuple is (parent_port, child_port).
        """
        from src.connectivity_db import COMPATIBLE_PORT_PAIRS

        parent_ports = self.get_part_ports(parent_part_num)
        child_ports = self.get_part_ports(child_part_num)
        pairs = []
        for pp in parent_ports:
            for cp in child_ports:
                if (pp.port_type, cp.port_type) in COMPATIBLE_PORT_PAIRS:
                    pairs.append((pp, cp))
        return pairs
