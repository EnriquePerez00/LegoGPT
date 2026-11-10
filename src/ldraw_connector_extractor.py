import os
import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from src.ldraw_geometry import LDrawGeometryResolver


class LDrawConnectorExtractor:
    """
    Parses LDraw part files and automatically extracts physical connection ports
    into the formal connector ontology:
      - Studs (Solid, Hollow, SNOT lateral)
      - Bottom Sockets & Anti-studs (Tubes, Corner antistuds)
      - Technic Pins & Holes
      - Clips & Bars
      - Ball joints & Hinges
    Persists extracted connectors into models_catalog.db for instant retrieval.
    """
    def __init__(self, db_path: str = "data/catalog/models_catalog.db", geom_resolver: Optional[LDrawGeometryResolver] = None):
        self.db_path = db_path
        self.geom = geom_resolver or LDrawGeometryResolver()
        self.memory_cache = {}

    def extract_connectors_for_part(self, part_num: str) -> List[Dict[str, Any]]:
        """
        Extracts all connectors for part_num. Checks DB cache first, else parses .dat file.
        """
        clean_part_num = part_num.lower().replace(".dat", "").strip()
        if clean_part_num in self.memory_cache:
            return self.memory_cache[clean_part_num]

        # 1. Check SQLite database
        db_connectors = self._load_from_db(clean_part_num)
        if db_connectors:
            self.memory_cache[clean_part_num] = db_connectors
            return db_connectors

        # 2. Parse from LDraw .dat file
        resolved_fn = self.geom.resolve_compatible_part_id(clean_part_num)
        filepath = self.geom._resolve_file(resolved_fn)
        if not filepath or not os.path.exists(filepath):
            return []

        meta = self.geom.get_part_meta(resolved_fn)
        connectors = []
        conn_idx = 0

        # Parse subfile calls from LDraw file to identify male studs, pins, clips, bars
        raw_studs = []
        
        def parse_connectors_recursive(curr_path: str, transform: np.ndarray, depth: int = 0):
            nonlocal conn_idx
            if depth > 8 or not os.path.exists(curr_path):
                return

            with open(curr_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    t = line.strip().split()
                    if not t or t[0] != "1" or len(t) < 15:
                        continue

                    sub_fn = t[14].lower().replace("\\", "/")
                    tx, ty, tz = float(t[2]), float(t[3]), float(t[4])
                    rot = np.array([float(v) for v in t[5:14]], dtype=np.float32).reshape(3, 3)

                    sub_tf = np.eye(4, dtype=np.float32)
                    sub_tf[:3, 3] = [tx, ty, tz]
                    sub_tf[:3, :3] = rot
                    comb_tf = transform @ sub_tf

                    pos = comb_tf[:3, 3]
                    normal = comb_tf[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    up = comb_tf[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)

                    # Classify by primitive subfile name
                    if "stud" in sub_fn:
                        if abs(normal[1]) < 0.2:
                            c_type = "STUD_SNOT_LATERAL"
                        elif "10" in sub_fn or "15" in sub_fn or "20" in sub_fn or "open" in sub_fn or "hollow" in sub_fn:
                            c_type = "STUD_HOLLOW"
                        else:
                            c_type = "STUD_SOLID"

                        # Deduplicate nearby studs
                        is_dup = any(np.linalg.norm(s["pos"] - pos) < 3.0 for s in raw_studs)
                        if not is_dup:
                            raw_studs.append({"pos": pos, "norm": normal, "type": c_type})
                            connectors.append({
                                "part_num": clean_part_num,
                                "connector_idx": conn_idx,
                                "type_id": c_type,
                                "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                                "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                                "up": [float(up[0]), float(up[1]), float(up[2])],
                                "is_subpart": 0
                            })
                            conn_idx += 1

                    elif "pin" in sub_fn:
                        c_type = "PIN_MALE"
                        connectors.append({
                            "part_num": clean_part_num,
                            "connector_idx": conn_idx,
                            "type_id": c_type,
                            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                            "up": [float(up[0]), float(up[1]), float(up[2])],
                            "is_subpart": 0
                        })
                        conn_idx += 1

                    elif "clip" in sub_fn:
                        c_type = "CLIP_FEMALE"
                        connectors.append({
                            "part_num": clean_part_num,
                            "connector_idx": conn_idx,
                            "type_id": c_type,
                            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                            "up": [float(up[0]), float(up[1]), float(up[2])],
                            "is_subpart": 0
                        })
                        conn_idx += 1

                    elif "bar" in sub_fn:
                        c_type = "BAR_MALE"
                        connectors.append({
                            "part_num": clean_part_num,
                            "connector_idx": conn_idx,
                            "type_id": c_type,
                            "pos": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                            "up": [float(up[0]), float(up[1]), float(up[2])],
                            "is_subpart": 0
                        })
                        conn_idx += 1

                    elif "sub" in sub_fn or "s/" in sub_fn or sub_fn.endswith(".dat"):
                        sub_res = self.geom._resolve_file(sub_fn)
                        if sub_res:
                            parse_connectors_recursive(sub_res, comb_tf, depth + 1)

        init_tf = np.eye(4, dtype=np.float32)
        parse_connectors_recursive(filepath, init_tf, 0)

        # Inferred Bottom Sockets & Tubes (Antistuds)
        inferred_sockets = meta.get("inferred_sockets", meta.get("sockets", []))
        for sock in inferred_sockets:
            s_pos = sock if isinstance(sock, (list, np.ndarray)) else sock["pos"]
            connectors.append({
                "part_num": clean_part_num,
                "connector_idx": conn_idx,
                "type_id": "SOCKET_TUBE",
                "pos": [float(s_pos[0]), float(s_pos[1]), float(s_pos[2])],
                "normal": [0.0, 1.0, 0.0],  # Facing downwards (+Y in LDraw coordinates)
                "up": [0.0, 0.0, 1.0],
                "is_subpart": 0
            })
            conn_idx += 1

        # Save to SQLite DB for persistent fast lookups
        self._save_to_db(clean_part_num, connectors)
        self.memory_cache[clean_part_num] = connectors
        return connectors

    def _load_from_db(self, clean_part_num: str) -> List[Dict[str, Any]]:
        """Loads part connectors from database."""
        if not os.path.exists(self.db_path):
            return []
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            SELECT connector_idx, type_id, pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, up_x, up_y, up_z, is_subpart
            FROM lego_part_connectors
            WHERE part_num = ?
            ORDER BY connector_idx ASC
        """, (clean_part_num,))
        rows = cur.fetchall()
        conn.close()

        if not rows:
            return []

        res = []
        for r in rows:
            res.append({
                "part_num": clean_part_num,
                "connector_idx": r[0],
                "type_id": r[1],
                "pos": [r[2], r[3], r[4]],
                "normal": [r[5], r[6], r[7]],
                "up": [r[8], r[9], r[10]],
                "is_subpart": bool(r[11])
            })
        return res

    def _save_to_db(self, clean_part_num: str, connectors: List[Dict[str, Any]]):
        """Saves part connectors to database."""
        if not connectors or not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        for c in connectors:
            cur.execute("""
                INSERT OR REPLACE INTO lego_part_connectors
                (part_num, connector_idx, type_id, pos_x, pos_y, pos_z, norm_x, norm_y, norm_z, up_x, up_y, up_z, is_subpart)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                clean_part_num,
                c["connector_idx"],
                c["type_id"],
                c["pos"][0], c["pos"][1], c["pos"][2],
                c["normal"][0], c["normal"][1], c["normal"][2],
                c["up"][0], c["up"][1], c["up"][2],
                c["is_subpart"]
            ))
        conn.commit()
        conn.close()

    def get_compatible_connector_types(self, type_id: str) -> List[str]:
        """Queries compatible connector types for a given type_id."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT compatible_types FROM lego_connector_types WHERE type_id = ?", (type_id,))
        row = cur.fetchone()
        conn.close()
        if not row:
            return []
        import json
        return json.loads(row[0])
