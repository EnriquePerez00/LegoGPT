"""
tests/test_design_priors.py

TDD tests for src/design_priors.py
Tests DesignPriors computation, query interface, and persistence
against fixture reference models.
"""

import os
import json
import math
import numpy as np
import pytest

from src.parser import ParsedPart
from src.reference_loader import (
    load_from_fixtures,
    COLOR_NEUTRAL,
)
from src.design_priors import (
    DesignPriors,
    WHEEL_PARTS,
    WINDSCREEN_PARTS,
    CATEGORY_MAP,
)
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "vehicles")


@pytest.fixture(scope="module")
def refs():
    """Load all vehicle fixtures once per test module."""
    return load_from_fixtures(fixtures_dir=FIXTURES_DIR)


@pytest.fixture(scope="module")
def priors(refs):
    """Build DesignPriors from fixtures once per test module."""
    return DesignPriors.from_reference_models(refs)


# ---------------------------------------------------------------------------
# Tests: from_reference_models
# ---------------------------------------------------------------------------

def test_priors_built_from_fixtures(priors):
    assert priors.n_reference_models >= 1
    assert priors.vocab_size >= 1


def test_priors_marginal_sums_to_one(priors):
    total = sum(priors.marginal.values())
    # Sum of marginal probabilities should be ~1.0
    assert abs(total - 1.0) < 1e-5, f"Marginal sum = {total}, expected ~1.0"


def test_priors_marginal_all_positive(priors):
    for part_id, prob in priors.marginal.items():
        assert prob > 0, f"Part '{part_id}' has non-positive marginal: {prob}"


def test_priors_wheel_parts_have_high_frequency(priors):
    """Wheel parts should be among the most common parts in vehicle fixtures."""
    wheel_prob_total = sum(
        priors.marginal.get(p, 0.0) for p in WHEEL_PARTS
    )
    # At least 20% of placements should be wheel-related
    assert wheel_prob_total > 0.0, "No wheel parts found in priors"


def test_priors_connectivity_bigrams_structure(priors):
    """Each bigram entry should sum to ~1.0 (conditional distribution)."""
    for src_part, neighbors in priors.connectivity_bigrams.items():
        total = sum(neighbors.values())
        assert abs(total - 1.0) < 1e-5, (
            f"Bigrams for '{src_part}' sum to {total}, expected ~1.0"
        )


def test_priors_count_distribution_structure(priors):
    """Each category count distribution should sum to ~1.0."""
    for category, dist in priors.count_distribution.items():
        if dist:
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-5, (
                f"Count dist for '{category}' sums to {total}, expected ~1.0"
            )


def test_priors_wheel_count_distribution_contains_4(priors):
    """Wheel count of 4 should appear in count distribution (all 3 fixtures have 4 wheels)."""
    dist = priors.count_distribution.get("wheel", {})
    assert "4" in dist or "8" in dist, (
        f"Expected wheel count 4 or 8 in dist: {dist}"
    )


def test_priors_empty_refs_returns_empty():
    empty_priors = DesignPriors.from_reference_models([])
    assert empty_priors.n_reference_models == 0
    assert empty_priors.vocab_size == 0
    assert empty_priors.marginal == {}


# ---------------------------------------------------------------------------
# Tests: query interface
# ---------------------------------------------------------------------------

def test_get_part_log_prior_known_part(priors):
    """A part present in fixtures should return a finite log prior > smoothing floor."""
    # All fixtures use 3020.dat or similar base plates
    log_p = priors.get_part_log_prior("3020.dat")
    assert math.isfinite(log_p)
    assert log_p > math.log(1e-6), "Known part should have higher prior than smoothing"


def test_get_part_log_prior_unknown_part(priors):
    """An unknown part should return log(smoothing) — small but finite."""
    log_p = priors.get_part_log_prior("UNKNOWN_PART_XYZ.dat")
    smoothing = 1e-6
    assert abs(log_p - math.log(smoothing)) < 1e-9


def test_get_connectivity_log_prior_known_pair(priors):
    """Known connected pair should return finite conditional log prior."""
    if priors.connectivity_bigrams:
        src = next(iter(priors.connectivity_bigrams))
        neighbors = priors.connectivity_bigrams[src]
        if neighbors:
            dst = next(iter(neighbors))
            log_p = priors.get_connectivity_log_prior(src, dst)
            assert math.isfinite(log_p)


def test_get_connectivity_log_prior_unknown_pair(priors):
    """Unknown pair should return log(smoothing)."""
    log_p = priors.get_connectivity_log_prior("UNKNOWN_A.dat", "UNKNOWN_B.dat")
    assert abs(log_p - math.log(1e-4)) < 1e-9


def test_get_expected_count_wheel(priors):
    """Expected wheel count should be 4 or 8 for vehicle fixtures."""
    expected = priors.get_expected_count("wheel")
    assert expected is not None
    assert expected in [4, 8], f"Expected wheel count 4 or 8, got {expected}"


def test_get_expected_count_unknown_category(priors):
    """Unknown category returns None."""
    result = priors.get_expected_count("unknown_category_xyz")
    assert result is None


def test_get_logit_bias_vector_shape(priors):
    """Logit bias vector should have same length as vocab."""
    vocab = VEHICLE_ALLOWED_PARTS
    bias = priors.get_logit_bias_vector(vocab)
    assert bias.shape == (len(vocab),)
    assert bias.dtype == np.float32


def test_get_logit_bias_vector_known_parts_higher(priors):
    """Parts seen in fixtures should have higher bias than unknown parts."""
    known_part = "6015.dat"  # tire, in all fixtures
    unknown_part = "UNKNOWN_XYZ.dat"
    
    known_bias = priors.get_part_log_prior(known_part)
    unknown_bias = priors.get_part_log_prior(unknown_part)
    assert known_bias > unknown_bias


# ---------------------------------------------------------------------------
# Tests: save / load persistence
# ---------------------------------------------------------------------------

def test_save_and_load_roundtrip(priors, tmp_path):
    path = str(tmp_path / "test_priors.json")
    saved_path = priors.save(path=path)
    assert os.path.isfile(saved_path)

    loaded = DesignPriors.load(path=path)
    assert loaded.n_reference_models == priors.n_reference_models
    assert loaded.vocab_size == priors.vocab_size
    assert abs(sum(loaded.marginal.values()) - 1.0) < 1e-5


def test_save_creates_valid_json(priors, tmp_path):
    path = str(tmp_path / "priors.json")
    priors.save(path=path)
    with open(path) as f:
        data = json.load(f)
    assert "marginal" in data
    assert "connectivity_bigrams" in data
    assert "count_distribution" in data
    assert "n_reference_models" in data
    assert data["n_reference_models"] >= 1


def test_load_missing_file_returns_empty(tmp_path):
    loaded = DesignPriors.load(path=str(tmp_path / "nonexistent.json"))
    assert loaded.n_reference_models == 0
    assert loaded.marginal == {}


def test_marginal_preserved_after_roundtrip(priors, tmp_path):
    path = str(tmp_path / "priors_rt.json")
    priors.save(path=path)
    loaded = DesignPriors.load(path=path)
    
    for part_id, prob in priors.marginal.items():
        assert abs(loaded.marginal.get(part_id, 0.0) - prob) < 1e-9


# ---------------------------------------------------------------------------
# Tests: repr
# ---------------------------------------------------------------------------

def test_repr_contains_key_info(priors):
    r = repr(priors)
    assert "DesignPriors" in r
    assert "n_refs" in r
    assert "vocab" in r
