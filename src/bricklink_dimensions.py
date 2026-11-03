import os
import re
import urllib.request
import ssl
import sqlite3

class BricklinkDimensionFetcher:
    """
    Fetches and caches physical stud dimensions from BrickLink catalog item pages.
    Example: P=3001 -> Stud Dim.: 2 x 4 x 1 in studs, Pack. Dim.: 1.6 x 3.2 x 1.15 cm
    """
    def __init__(self, db_path: str = "data/catalog/models_catalog.db"):
        self.db_path = db_path
        self._init_cache_table()
        self.ctx = ssl._create_unverified_context()

    def _init_cache_table(self):
        if not os.path.exists(self.db_path):
            return
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bricklink_part_dimensions (
                part_num TEXT PRIMARY KEY,
                stud_w INTEGER,
                stud_d INTEGER,
                stud_h INTEGER,
                pack_dim_cm TEXT
            )
        """)
        conn.commit()
        conn.close()

    def get_stud_dimensions(self, part_num: str) -> dict:
        """Returns cached or scraped stud dimensions for part_num."""
        clean_num = part_num.lower().replace(".dat", "")
        if os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("SELECT stud_w, stud_d, stud_h, pack_dim_cm FROM bricklink_part_dimensions WHERE part_num = ?", (clean_num,))
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "part_num": clean_num,
                    "stud_w": row[0],
                    "stud_d": row[1],
                    "stud_h": row[2],
                    "pack_dim_cm": row[3]
                }

        url = f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={clean_num}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        try:
            with urllib.request.urlopen(req, context=self.ctx, timeout=8) as r:
                html = r.read().decode("utf-8", errors="ignore")
                m_stud = re.search(r"Stud Dim\.:\s*<span[^>]*>(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\s*in studs", html)
                m_pack = re.search(r"Pack\. Dim\.:\s*<span[^>]*>([^<]+)</span>", html)

                stud_w = int(m_stud.group(1)) if m_stud else None
                stud_d = int(m_stud.group(2)) if m_stud else None
                stud_h = int(m_stud.group(3)) if m_stud else None
                pack_dim = m_pack.group(1).strip() if m_pack else ""

                if stud_w is not None and os.path.exists(self.db_path):
                    conn = sqlite3.connect(self.db_path)
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO bricklink_part_dimensions (part_num, stud_w, stud_d, stud_h, pack_dim_cm)
                        VALUES (?, ?, ?, ?, ?)
                    """, (clean_num, stud_w, stud_d, stud_h, pack_dim))
                    conn.commit()
                    conn.close()

                return {
                    "part_num": clean_num,
                    "stud_w": stud_w,
                    "stud_d": stud_d,
                    "stud_h": stud_h,
                    "pack_dim_cm": pack_dim
                }
        except Exception:
            return {"part_num": clean_num, "stud_w": None, "stud_d": None, "stud_h": None, "pack_dim_cm": ""}
