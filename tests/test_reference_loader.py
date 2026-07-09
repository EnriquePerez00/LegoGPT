"""
tests/test_reference_loader.py

TDD tests for src/reference_loader.py
Tests the RefModel contract, collapse_color, build_inventory_from_parts,
and the unified load_reference_models interface against fixture data.
"""

import os
import json
import tempfile
import numpy as np
import pytest

from src.parser import ParsedPart
from src.reference_loader import (
    RefModel,
    COLOR_NEUTRAL,
    collapse_color,
    build_inventory_from_parts,
    load_from_fixtures,
    load_reference_models,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "vehicles")


# ---------------------------------------------------------------------------
# Tests: collapse_color
# ---------------------------------------------------------------------------

def test_collapse_color_replaces_all_colors():
    parts = [
        ParsedPart("3020.dat", color=4, transform=np.eye(4, dtype=np.float32), step_id=0),
        ParsedPart("3001.dat", color=14, transform=np.eye(4, dtype=np.float32), step_id=1),
        ParsedPart("6015.dat", color=0, transform=np.eye(4, dtype=np.float32), step_id=2),
    ]
    neutral = collapse_color(parts)
    assert len(neutral) == 3
    for p in neutral:
        assert p.color == COLOR_NEUTRAL, f"Expected {COLOR_NEUTRAL}, got {p.color}"


def test_collapse_color_preserves_structure():
    t = np.eye(4, dtype=np.float32)
    t[0, 3] = 20.0
    parts = [ParsedPart("3020.dat", color=4, transform=t, step_id=5)]
    neutral = collapse_color(parts)
    assert neutral[0].part_id == "3020.dat"
    assert neutral[0].step_id == 5
    np.testing.assert_array_almost_equal(neutral[0].transform, t)


def test_collapse_color_does_not_mutate_original():
    parts = [ParsedPart("3001.dat", color=4, transform=np.eye(4, dtype=np.float32), step_id=0)]
    _ = collapse_color(parts)
    assert parts[0].color == 4  # original unchanged


# ---------------------------------------------------------------------------
# Tests: build_inventory_from_parts
# ---------------------------------------------------------------------------

def test_build_inventory_counts_correctly():
    parts = [
        ParsedPart("3020.dat", COLOR_NEUTRAL, np.eye(4, dtype=np.float32), 0),
        ParsedPart("3020.dat", COLOR_NEUTRAL, np.eye(4, dtype=np.float32), 0),
        ParsedPart("6015.dat", COLOR_NEUTRAL, np.eye(4, dtype=np.float32), 1),
    ]
    inv = build_inventory_from_parts(parts)
    assert inv["3020.dat"] == 2
    assert inv["6015.dat"] == 1
    assert len(inv) == 2


def test_build_inventory_ignores_color():
    parts = [
        ParsedPart("3001.dat", color=4, transform=np.eye(4, dtype=np.float32), step_id=0),
        ParsedPart("3001.dat", color=14, transform=np.eye(4, dtype=np.float32), step_id=1),
    ]
    inv = build_inventory_from_parts(parts)
    # Both should count as same part (color ignored)
    assert inv["3001.dat"] == 2
    assert len(inv) == 1


# ---------------------------------------------------------------------------
# Tests: load_from_fixtures
# ---------------------------------------------------------------------------

def test_load_fixtures_returns_list():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    assert isinstance(refs, list)
    assert len(refs) >= 1, "Expected at least 1 fixture vehicle"


def test_load_fixtures_sorted_asc():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    counts = [r.parts_count for r in refs]
    assert counts == sorted(counts), "RefModels must be sorted by parts_count ASC"


def test_load_fixtures_refmodel_contract():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    for ref in refs:
        assert isinstance(ref, RefModel)
        assert isinstance(ref.set_id, str) and len(ref.set_id) > 0
        assert isinstance(ref.parts, list) and len(ref.parts) > 0
        assert ref.graph is not None
        assert isinstance(ref.inventory, dict) and len(ref.inventory) > 0
        assert ref.parts_count == len(ref.parts)
        assert ref.source == "fixtures"


def test_load_fixtures_colors_collapsed():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    for ref in refs:
        for p in ref.parts:
            assert p.color == COLOR_NEUTRAL, (
                f"Set '{ref.set_id}': part '{p.part_id}' has color {p.color}, "
                f"expected COLOR_NEUTRAL={COLOR_NEUTRAL}"
            )


def test_load_fixtures_parts_sorted_by_step():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    for ref in refs:
        steps = [p.step_id for p in ref.parts]
        assert steps == sorted(steps), f"Parts in '{ref.set_id}' not sorted by step_id"


def test_load_fixtures_inventory_matches_parts():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    for ref in refs:
        reconstructed_inv = build_inventory_from_parts(ref.parts)
        assert reconstructed_inv == ref.inventory, (
            f"Inventory mismatch for '{ref.set_id}'"
        )


def test_load_fixtures_car_has_4_wheels():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    # At least one fixture should have exactly 4 wheel parts
    wheel_parts = {
        "3139.dat", "42610.dat", "56902.dat", "30027.dat", "18976.dat",
        "18977.dat", "55981.dat", "55982.dat", "30285.dat", "6014.dat",
        "6015.dat", "56890.dat", "56891.dat", "30285b.dat", "6014b.dat",
    }
    for ref in refs:
        wheel_count = sum(1 for p in ref.parts if p.part_id in wheel_parts)
        assert wheel_count >= 4, (
            f"Vehicle '{ref.set_id}' has only {wheel_count} wheels, expected >= 4"
        )


# ---------------------------------------------------------------------------
# Tests: empty / missing fixtures directory
# ---------------------------------------------------------------------------

def test_load_fixtures_missing_dir_returns_empty():
    refs = load_from_fixtures(fixtures_dir="/nonexistent/path/xyz")
    assert refs == []


def test_load_from_fixtures_with_temp_malformed_json(tmp_path):
    # Malformed JSON should be skipped (not raise)
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ this is not valid json }")
    refs = load_from_fixtures(fixtures_dir=str(tmp_path))
    assert refs == []


def test_load_from_fixtures_with_valid_minimal_json(tmp_path):
    minimal = {
        "set_id": "test_mini",
        "parts": [
            {"part_id": "3020.dat", "color": 4,
             "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1], "step_id": 0},
        ]
    }
    f = tmp_path / "mini.json"
    f.write_text(json.dumps(minimal))
    refs = load_from_fixtures(fixtures_dir=str(tmp_path))
    assert len(refs) == 1
    assert refs[0].set_id == "test_mini"
    assert refs[0].parts_count == 1
    assert refs[0].parts[0].color == COLOR_NEUTRAL  # color collapsed


# ---------------------------------------------------------------------------
# Tests: unified load_reference_models interface
# ---------------------------------------------------------------------------

def test_load_reference_models_fixtures_source():
    refs = load_reference_models(source="fixtures", fixtures_dir=FIXTURES_DIR)
    assert len(refs) >= 1


def test_load_reference_models_db_missing_returns_empty():
    refs = load_reference_models(
        source="db",
        db_path="/nonexistent/models_catalog.db",
    )
    assert refs == []


def test_load_reference_models_invalid_source_raises():
    with pytest.raises(ValueError, match="Unknown source"):
        load_reference_models(source="invalid_source")
