"""
src/reference_loader.py

Contract-first loader for vehicle reference models.
Supports two sources:
  - 'fixtures' -> reads from tests/fixtures/vehicles/ (synthetic TDD data)
  - 'db'       -> reads from data/catalog/models_catalog.db + processed files

This module defines the canonical RefModel dataclass and the unified loading interface.
When reference data arrives in the DB, only the source flag changes, zero code refactor.
"""

import os
import json
import logging
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from src.parser import ParsedPart, build_pyg_graph
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS

logger = logging.getLogger(__name__)

# Phase 0 constant: LDraw color 16 = "Main Colour" (inherited/neutral).
COLOR_NEUTRAL: int = 16


@dataclass
class RefModel:
    """
    Canonical data contract for a single reference LEGO vehicle model.

    Attributes:
        set_id:      BrickLink/OMR set identifier (e.g. '60050-1').
        parts:       Assembly sequence as ParsedPart list, ordered by step_id.
                     Colors collapsed to COLOR_NEUTRAL in Phase 0.
        graph:       PyTorch Geometric Data (nodes=parts, edges=connections).
        inventory:   Color-collapsed {part_id: quantity} mapping.
        parts_count: Total number of physical parts.
        source:      Data origin: 'fixtures' or 'db'.
        metadata:    Optional extra info (theme, year, etc.).
    """
    set_id: str
    parts: list
    graph: Data
    inventory: dict
    parts_count: int
    source: str = "fixtures"
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parts_count == 0:
            self.parts_count = len(self.parts)


def collapse_color(parts: list) -> list:
    """
    Returns a new list of ParsedParts with all colors replaced by COLOR_NEUTRAL (16).
    Phase 0 transformation: removes color signal, preserves full structural info.
    """
    return [
        ParsedPart(
            part_id=p.part_id,
            color=COLOR_NEUTRAL,
            transform=p.transform.copy(),
            step_id=p.step_id,
        )
        for p in parts
    ]


def build_inventory_from_parts(parts: list) -> dict:
    """
    Computes a color-collapsed inventory {part_id: count} from a parts list.
    Color intentionally ignored (Phase 0 contract).
    """
    inventory: dict = {}
    for p in parts:
        inventory[p.part_id] = inventory.get(p.part_id, 0) + 1
    return inventory


FIXTURES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "vehicles"
)


def load_from_fixtures(fixtures_dir: Optional[str] = None) -> list:
    """
    Loads synthetic vehicle reference models from JSON fixture files.

    Fixture JSON format:
    {
      "set_id": "fixture_car_01",
      "metadata": {"theme": "City"},
      "parts": [
        {"part_id": "3020.dat", "color": 4,
         "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], "step_id": 0},
        ...
      ]
    }

    Returns:
        List[RefModel] sorted by parts_count ASC (fail-fast rule).
    """
    base_dir = fixtures_dir or FIXTURES_DIR
    if not os.path.isdir(base_dir):
        logger.warning("Fixtures directory not found: %s", base_dir)
        return []

    refs = []
    json_files = sorted(f for f in os.listdir(base_dir) if f.endswith(".json"))

    for fname in json_files:
        fpath = os.path.join(base_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            set_id = data.get("set_id", os.path.splitext(fname)[0])
            raw_parts = data.get("parts", [])

            parts = []
            for entry in raw_parts:
                flat = entry.get("transform", list(np.eye(4).flatten()))
                transform = np.array(flat, dtype=np.float32).reshape(4, 4)
                parts.append(
                    ParsedPart(
                        part_id=entry["part_id"],
                        color=entry.get("color", COLOR_NEUTRAL),
                        transform=transform,
                        step_id=entry.get("step_id", 0),
                    )
                )

            parts.sort(key=lambda p: p.step_id)
            parts_neutral = collapse_color(parts)
            inventory = build_inventory_from_parts(parts_neutral)
            graph = build_pyg_graph(parts_neutral, allowed_parts=VEHICLE_ALLOWED_PARTS)

            refs.append(
                RefModel(
                    set_id=set_id,
                    parts=parts_neutral,
                    graph=graph,
                    inventory=inventory,
                    parts_count=len(parts_neutral),
                    source="fixtures",
                    metadata=data.get("metadata", {}),
                )
            )
            logger.info("Loaded fixture '%s' (%d parts)", set_id, len(parts_neutral))

        except Exception as exc:
            logger.error("Failed to load fixture '%s': %s", fname, exc)

    refs.sort(key=lambda r: r.parts_count)
    return refs


DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "catalog", "models_catalog.db"
)
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_from_db(
    db_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    min_parts: int = 100,
    max_parts: int = 200,
    theme_filter: Optional[str] = None,
) -> list:
    """
    Loads vehicle reference models from the SQLite catalog + processed files.

    Requires dataset_generator.py to have been run, producing:
        data/processed/{set_id}_assembly.json
        data/processed/{set_id}_graph.pt

    Args:
        db_path:       Path to models_catalog.db.
        processed_dir: Directory with _assembly.json / _graph.pt files.
        min_parts:     Minimum part count filter (default 100).
        max_parts:     Maximum part count filter (default 200).
        theme_filter:  Optional SQL LIKE pattern for theme.

    Returns:
        List[RefModel] sorted by parts_count ASC.
    """
    db = db_path or DB_PATH
    proc_dir = processed_dir or PROCESSED_DIR

    if not os.path.isfile(db):
        logger.warning("DB not found at '%s' -- returning empty list", db)
        return []

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    query = (
        "SELECT s.set_id, s.parts_count FROM sets s "
        "WHERE s.parts_count BETWEEN ? AND ?"
    )
    params: list = [min_parts, max_parts]

    if theme_filter:
        query += " AND s.theme LIKE ?"
        params.append(theme_filter)

    query += " ORDER BY s.parts_count ASC"
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    logger.info(
        "DB query returned %d candidate sets (parts %d-%d)",
        len(rows), min_parts, max_parts
    )

    refs = []
    for set_id, db_parts_count in rows:
        assembly_path = os.path.join(proc_dir, f"{set_id}_assembly.json")
        graph_path = os.path.join(proc_dir, f"{set_id}_graph.pt")

        if not os.path.isfile(assembly_path):
            logger.debug("Missing assembly JSON for '%s' -- skipping", set_id)
            continue
        if not os.path.isfile(graph_path):
            logger.debug("Missing graph .pt for '%s' -- skipping", set_id)
            continue

        try:
            with open(assembly_path, "r", encoding="utf-8") as fh:
                assembly_data = json.load(fh)

            parts = []
            for entry in assembly_data.get("parts", []):
                transform = np.array(entry["transform"], dtype=np.float32).reshape(4, 4)
                parts.append(
                    ParsedPart(
                        part_id=entry["part_id"],
                        color=entry.get("color", COLOR_NEUTRAL),
                        transform=transform,
                        step_id=entry.get("step_id", 0),
                    )
                )

            parts.sort(key=lambda p: p.step_id)
            parts_neutral = collapse_color(parts)
            inventory = build_inventory_from_parts(parts_neutral)
            graph = torch.load(graph_path, weights_only=False)

            refs.append(
                RefModel(
                    set_id=set_id,
                    parts=parts_neutral,
                    graph=graph,
                    inventory=inventory,
                    parts_count=len(parts_neutral),
                    source="db",
                    metadata={"db_parts_count": db_parts_count},
                )
            )
            logger.info("Loaded DB ref '%s' (%d parts)", set_id, len(parts_neutral))

        except Exception as exc:
            logger.error("Failed to load DB set '%s': %s", set_id, exc)

    refs.sort(key=lambda r: r.parts_count)
    return refs


def load_reference_models(
    source: str = "fixtures",
    fixtures_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    min_parts: int = 100,
    max_parts: int = 200,
    theme_filter: Optional[str] = None,
) -> list:
    """
    Unified loader for reference models. Identical signature for both sources.

    Args:
        source:        'fixtures' (synthetic TDD) or 'db' (real catalog).
        fixtures_dir:  Override fixtures directory (used when source='fixtures').
        db_path:       Override DB path (used when source='db').
        processed_dir: Override processed data directory (used when source='db').
        min_parts:     Min part count filter (used when source='db').
        max_parts:     Max part count filter (used when source='db').
        theme_filter:  SQL LIKE theme filter (used when source='db').

    Returns:
        List[RefModel] sorted by parts_count ASC (fail-fast).

    Usage:
        # During development (no real data needed):
        refs = load_reference_models(source='fixtures')

        # When real vehicle refs are ready:
        refs = load_reference_models(source='db', min_parts=100, max_parts=200)
    """
    if source == "fixtures":
        return load_from_fixtures(fixtures_dir=fixtures_dir)
    elif source == "db":
        return load_from_db(
            db_path=db_path,
            processed_dir=processed_dir,
            min_parts=min_parts,
            max_parts=max_parts,
            theme_filter=theme_filter,
        )
    else:
        raise ValueError(
            f"Unknown source '{source}'. Valid options: 'fixtures', 'db'."
        )
