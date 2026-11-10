#!/usr/bin/env python3
"""Downloads official LEGO instruction PDFs (from `set_manuals`) into data/manual/official/.

Usage: python scripts/download_official_manuals.py --theme "Creator 3-in-1"
Files are named {set_id}_{booklet_idx}_{file_id}.pdf; resumable (skips existing).
"""
import argparse
import os
import shutil
import sqlite3
import ssl
import urllib.request

import certifi
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.storage_guard import assert_disk_space_or_exit

DB = "data/catalog/models_catalog.db"
DEST = "data/manual/official"
CTX = ssl.create_default_context(cafile=certifi.where())
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"


def fetch(job):
    url, path = job
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return path, "skip"
    tmp = path + ".part"
    for _ in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60, context=CTX) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            os.replace(tmp, path)
            return path, "ok"
        except Exception as e:
            err = str(e)
    return path, f"error: {err}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", default="Creator 3-in-1")
    a = ap.parse_args()
    assert_disk_space_or_exit()
    os.makedirs(DEST, exist_ok=True)
    c = sqlite3.connect(DB).cursor()
    c.execute("""SELECT m.set_id, m.booklet_idx, m.file_id, m.pdf_url FROM set_manuals m
                 JOIN sets s ON s.set_id = m.set_id WHERE s.theme = ?""", (a.theme,))
    jobs = [(u, f"{DEST}/{sid}_b{bi}_{fid}.pdf") for sid, bi, fid, u in c.fetchall()]
    workers = min(16, max(4, (os.cpu_count() or 12) - 2))
    print(f"{len(jobs)} PDFs, {workers} threads")
    done = errs = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(fetch, j) for j in jobs]):
            p, st = fut.result()
            done += 1
            if st.startswith("error"):
                errs += 1
                print(p, st)
            if done % 50 == 0:
                print(f"{done}/{len(jobs)} (errors {errs})", flush=True)
    print(f"Done: {done - errs} ok, {errs} errors")


if __name__ == "__main__":
    main()
