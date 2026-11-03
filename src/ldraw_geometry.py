import os
import re
import sqlite3
import numpy as np


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
        self.part_resolution_cache = {}

    def _resolve_file(self, filename: str) -> str:
        """Finds file in parts/, p/, or subdirectories (e.g. s/ or 8/) across local and Studio libraries."""
        fn_clean = filename.replace("\\", "/")
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

        # 2. Check Rebrickable relationships (Parent / Alternate / Mold)
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                cur = conn.cursor()
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
                    if cat_id == 61:  # Minifig Lower Body (Legs)
                        conn.close()
                        self.part_resolution_cache[raw] = "970c00.dat"
                        return "970c00.dat"
                    elif cat_id == 60:  # Minifig Upper Body (Torso)
                        conn.close()
                        self.part_resolution_cache[raw] = "973.dat"
                        return "973.dat"
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
        fn = part_num.lower()
        if not fn.endswith(".dat"):
            fn = f"{fn}.dat"

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
                "has_lateral_ports": False
            }
            self.cache[fn] = meta
            return meta

        vertices = []
        studs = []
        sockets = []
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
                        if "stud" in sub_fn:
                            pos = combined_tf[:3, 3]
                            norm = combined_tf[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
                            studs.append({"pos": pos, "normal": norm})
                            if abs(norm[1]) < 0.8:
                                has_lateral = True
                        elif "sub" in sub_fn or "s\\" in sub_fn or sub_fn.endswith(".dat"):
                            resolved = self._resolve_file(sub_fn)
                            if resolved:
                                parse_file(resolved, combined_tf, depth + 1)

                    # Line types 2 (line), 3 (triangle), 4 (quadrilateral) contain vertices
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

        # Clean/deduplicate studs
        unique_studs = []
        for s in studs:
            p = s["pos"]
            is_dup = False
            for us in unique_studs:
                if np.linalg.norm(us["pos"] - p) < 3.0:
                    is_dup = True
                    break
            if not is_dup:
                unique_studs.append(s)

        # Bottom sockets (in standard grid if no explicit antistuds found)
        inferred_sockets = []
        for ix in range(w_studs):
            sx = xmin + 10 + ix * 20
            for iz in range(d_studs):
                sz = zmin + 10 + iz * 20
                inferred_sockets.append(np.array([sx, ymax, sz], dtype=np.float32))

        meta = {
            "part_id": fn,
            "bbox": [xmin, ymin, zmin, xmax, ymax, zmax],
            "dim_ldu": (w_ldu, h_ldu, d_ldu),
            "h_plates": h_plates,
            "stud_dim": (w_studs, d_studs, h_plates),
            "studs": [s["pos"] for s in unique_studs],
            "sockets": inferred_sockets,
            "has_lateral_ports": has_lateral
        }
        self.cache[fn] = meta
        return meta
