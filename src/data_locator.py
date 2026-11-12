"""Data locator for cross-repo access.
Exposes stable paths to .io files, .pdf manuals, and catalog BD from LegoGPT.
LdrawNova and other consumers should import this to resolve LegoGPT data paths.
"""
from pathlib import Path
import os, json

def get_legogpt_root() -> Path:
    """Return the LegoGPT repository root."""
    return Path(__file__).parent.parent

def get_data_root() -> Path:
    """Return LegoGPT/data/ directory."""
    return get_legogpt_root() / "data"

def get_normalized_io_dir() -> Path:
    """Return LegoGPT/data/normalized_io/ (927 .io files from sets and MOCs)."""
    return get_data_root() / "normalized_io"

def get_models_io_dir() -> Path:
    """Return LegoGPT/data/models_io/ (future: raw MOC .io files)."""
    return get_data_root() / "models_io"

def get_manuals_dir() -> Path:
    """Return LegoGPT/data/manuals/ (PDF instruction manuals)."""
    return get_data_root() / "manuals"

def get_omr_raw_dir() -> Path:
    """Return LegoGPT/data/omr_raw/ (733 .mpd official set files)."""
    return get_data_root() / "omr_raw"

def get_catalog_db() -> Path:
    """Return LegoGPT/data/catalog/models_catalog.db (929 pairs, 927 with .io)."""
    return get_data_root() / "catalog" / "models_catalog.db"

def get_eval_dir() -> Path:
    """Return LegoGPT/data/eval/ (evaluation sets)."""
    return get_data_root() / "eval"

def validate_data_sources() -> dict:
    """Validate critical data sources. Returns dict with 'ok' (bool) and 'missing' (list)."""
    required = {
        "normalized_io": get_normalized_io_dir(),
        "catalog_db": get_catalog_db(),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    return {
        "ok": len(missing) == 0,
        "missing": missing,
        "inventory": {
            "normalized_io_count": len(list(get_normalized_io_dir().glob("*.io"))) if get_normalized_io_dir().exists() else 0,
            "catalog_db_exists": get_catalog_db().exists(),
            "omr_raw_count": len(list(get_omr_raw_dir().glob("*.mpd"))) if get_omr_raw_dir().exists() else 0,
        }
    }

def get_io_by_pair_id(pair_id: str) -> Path | None:
    """Resolve path to a .io file by pair_id (e.g. 'OMR-1234-1').
    Uses models_catalog.db to look up normalized_io_path.
    Returns None if not found or path doesn't exist.
    """
    import sqlite3
    db_path = get_catalog_db()
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT normalized_io_path FROM ground_truth_pairs WHERE pair_id = ?", (pair_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        io_path = get_data_root() / row[0]
        return io_path if io_path.exists() else None
    except Exception:
        return None

if __name__ == "__main__":
    status = validate_data_sources()
    print(json.dumps(status, indent=2, default=str))
