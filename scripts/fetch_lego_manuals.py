#!/usr/bin/env python3
"""
Identifies official LEGO building-instruction PDFs (permanent LEGO CDN URLs) for the
sets in `sets` whose reference looks like NNNN-1 and stores them in `set_manuals`.

Usage:
    python3 scripts/fetch_lego_manuals.py --limit 20            # pilot (always includes 75105-1)
    python3 scripts/fetch_lego_manuals.py --sets 75105-1 10294-1
    python3 scripts/fetch_lego_manuals.py                       # all pending sets (resumable)
"""

import argparse
import json
import os
import re
import sqlite3
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

DB_PATH = "data/catalog/models_catalog.db"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
CDN_PDF = "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/{}.pdf"
CDN_IMG = "https://www.lego.com/cdn/product-assets/product.bi.core.img/{}.png"
# English only: the locale-prefixed page serves the English manuals (the un-prefixed URL geo-redirects
# to the visitor's locale). en-gb is only a fallback when en-us has no instructions for the set.
PAGES = [
    ("en-us", "https://www.lego.com/en-us/service/buildinginstructions/{}"),
    ("en-gb", "https://www.lego.com/en-gb/service/buildinginstructions/{}"),
]
NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
SET_RE = re.compile(r"^\d+-1$")


def setup_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS set_manuals (
            set_id TEXT NOT NULL,
            lego_set_num TEXT NOT NULL,
            file_id TEXT NOT NULL,
            pdf_url TEXT NOT NULL,
            thumb_url TEXT,
            locale TEXT,
            booklet_idx INTEGER,
            booklet_total INTEGER,
            is_additional INTEGER DEFAULT 0,
            title TEXT,
            source TEXT DEFAULT 'lego_cdn',
            http_status INTEGER,
            size_bytes INTEGER,
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (set_id, file_id)
        );
        CREATE INDEX IF NOT EXISTS idx_set_manuals_set ON set_manuals(set_id);
        CREATE TABLE IF NOT EXISTS set_manuals_progress (
            set_id TEXT PRIMARY KEY,
            status TEXT,
            n_found INTEGER,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()


def http(url, method="GET", retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, r.headers, (r.read() if method == "GET" else b"")
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return e.code, {}, b""
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    raise last


def find_instructions(node):
    """Recursively locate the `buildingInstructions` list inside the __NEXT_DATA__ JSON."""
    if isinstance(node, dict):
        if isinstance(node.get("buildingInstructions"), list):
            return node
        for v in node.values():
            r = find_instructions(v)
            if r:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find_instructions(v)
            if r:
                return r
    return None


def parse_page(html):
    m = NEXT_RE.search(html)
    if not m:
        return []
    holder = find_instructions(json.loads(m.group(1)))
    out = []
    for bi in (holder or {}).get("buildingInstructions", []):
        pdf = bi.get("pdf") or {}
        mm = re.search(r"pdf/(\d+)\.pdf", pdf.get("pdfUrl") or "")
        if not mm:
            continue
        seq = bi.get("sequence") or {}
        out.append({
            "file_id": mm.group(1),
            "idx": int(seq["element"]) if seq.get("element") else None,
            "total": int(seq["total"]) if seq.get("total") else None,
            "additional": 1 if bi.get("isAdditionalInfoBooklet") else 0,
            "title": holder.get("name"),
        })
    return out


def process_set(set_id):
    num = set_id.rsplit("-", 1)[0]
    try:
        items, locale = [], None
        for loc, tpl in PAGES:
            status, _, body = http(tpl.format(num))
            if status == 200:
                items = parse_page(body.decode("utf8", "ignore"))
                if items:
                    locale = loc
                    break
        rows = []
        for it in items:
            fid = it["file_id"]
            st, hdr, _ = http(CDN_PDF.format(fid), method="HEAD")
            if st == 200 and "pdf" in (hdr.get("content-type", "") if hdr else ""):
                rows.append((set_id, num, fid, CDN_PDF.format(fid), CDN_IMG.format(fid), locale,
                             it["idx"], it["total"], it["additional"], it["title"], st,
                             int(hdr.get("content-length") or 0)))
        return set_id, ("ok" if rows else "not_found"), rows, None
    except Exception as e:
        return set_id, "error", [], str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sets", nargs="*")
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH, timeout=60)
    setup_db(conn)

    if args.sets:
        targets = args.sets
    else:
        # 'error' rows are always re-processed on the next run; --retry-errors is kept for clarity
        skip = "('ok','not_found')"
        cand = [r[0] for r in conn.execute(
            f"SELECT set_id FROM sets WHERE set_id GLOB '[0-9]*-1' AND set_id NOT GLOB '*[^0-9-]*' "
            f"AND set_id NOT IN (SELECT set_id FROM set_manuals_progress WHERE status IN {skip}) "
            f"ORDER BY set_id")]
        cand = [s for s in cand if SET_RE.match(s)]
        if args.limit:
            pilot = ["75105-1"] if "75105-1" in cand else []
            rest = [s for s in cand if s not in pilot]
            # spread the sample across the id range instead of taking only the first ones
            step = max(1, len(rest) // max(1, args.limit - len(pilot)))
            targets = pilot + rest[::step][: args.limit - len(pilot)]
        else:
            targets = cand
    print(f"Sets to process: {len(targets)}")

    # Polite rate: few workers against lego.com (network bound, not CPU bound)
    workers = min(6, max(1, (os.cpu_count() or 12) - 4))
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_set, s) for s in targets]
        for f in as_completed(futs):
            set_id, status, rows, err = f.result()
            conn.executemany(
                "INSERT OR REPLACE INTO set_manuals(set_id,lego_set_num,file_id,pdf_url,thumb_url,locale,booklet_idx,"
                "booklet_total,is_additional,title,http_status,size_bytes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            conn.execute("INSERT OR REPLACE INTO set_manuals_progress(set_id,status,n_found) VALUES (?,?,?)",
                         (set_id, status, len(rows)))
            conn.commit()
            done += 1
            print(f"[{done}/{len(targets)}] {set_id}: {status} ({len(rows)} manuals){' ' + err if err else ''}")
    conn.close()


if __name__ == "__main__":
    main()
