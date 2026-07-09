"""
tests/test_vehicle_reward_with_refs.py - TDD for FASE 3 enriched RL reward
"""
import os, numpy as np, pytest
from src.parser import ParsedPart
from src.vehicle_rules import (
    get_vehicle_rl_reward, get_vehicle_rl_reward_with_refs,
    _histogram_similarity, _bigram_compliance_score,
)
from src.reference_loader import load_from_fixtures
from src.design_priors import DesignPriors

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "vehicles")

def T(x, y, z):
    t = np.eye(4, dtype=np.float32)
    t[0,3], t[1,3], t[2,3] = x, y, z
    return t

@pytest.fixture(scope="module")
def refs():
    return load_from_fixtures(fixtures_dir=FIXTURES_DIR)

@pytest.fixture(scope="module")
def priors(refs):
    return DesignPriors.from_reference_models(refs)

def _car_parts():
    return [
        ParsedPart("3020.dat", 16, T(0,0,0), 0),
        ParsedPart("3020.dat", 16, T(0,0,40), 0),
        ParsedPart("6015.dat", 16, T(-30,8,0), 1),
        ParsedPart("6015.dat", 16, T(30,8,0), 1),
        ParsedPart("6015.dat", 16, T(-30,8,40), 1),
        ParsedPart("6015.dat", 16, T(30,8,40), 1),
        ParsedPart("3001.dat", 16, T(0,-32,0), 2),
        ParsedPart("3001.dat", 16, T(0,-32,40), 2),
    ]

# --- _histogram_similarity ---

def test_histogram_similarity_self_is_one(refs):
    if not refs: pytest.skip("No refs")
    ref = refs[0]
    sim = _histogram_similarity(ref.parts, [ref])
    assert abs(sim - 1.0) < 0.01

def test_histogram_similarity_empty_parts(refs):
    sim = _histogram_similarity([], refs)
    assert sim == 0.0

def test_histogram_similarity_empty_refs():
    parts = _car_parts()
    sim = _histogram_similarity(parts, [])
    assert sim == 0.0

def test_histogram_similarity_returns_float_in_range(refs):
    parts = _car_parts()
    sim = _histogram_similarity(parts, refs)
    assert 0.0 <= sim <= 1.0

def test_histogram_similarity_different_parts_less_than_one(refs):
    parts = [ParsedPart("3005.dat", 16, np.eye(4,dtype=np.float32), 0)] * 5
    sim = _histogram_similarity(parts, refs)
    assert 0.0 <= sim < 1.0

# --- _bigram_compliance_score ---

def test_bigram_compliance_no_priors():
    parts = _car_parts()
    score = _bigram_compliance_score(parts, None)
    assert score == 0.5  # neutral

def test_bigram_compliance_empty_priors():
    parts = _car_parts()
    empty = DesignPriors._empty()
    score = _bigram_compliance_score(parts, empty)
    assert score == 0.5

def test_bigram_compliance_single_part(priors):
    parts = [ParsedPart("3020.dat", 16, np.eye(4,dtype=np.float32), 0)]
    score = _bigram_compliance_score(parts, priors)
    assert score == 0.5  # single part, no pairs

def test_bigram_compliance_returns_float_in_range(priors):
    parts = _car_parts()
    score = _bigram_compliance_score(parts, priors)
    assert 0.0 <= score <= 1.0

# --- get_vehicle_rl_reward_with_refs ---

def test_enriched_reward_no_refs_equals_base():
    parts = _car_parts()
    base = get_vehicle_rl_reward(parts)
    enriched = get_vehicle_rl_reward_with_refs(parts, refs=None, design_priors=None)
    assert abs(enriched - base) < 1e-5

def test_enriched_reward_with_refs_ge_base(refs, priors):
    parts = _car_parts()
    base = get_vehicle_rl_reward(parts)
    enriched = get_vehicle_rl_reward_with_refs(parts, refs=refs, design_priors=priors)
    # With valid refs, bonus should be >= 0 (not necessarily > base due to early exit)
    if base > -10.0:
        assert enriched >= base - 0.001  # sim and bigram bonuses always >= 0

def test_enriched_reward_empty_parts():
    r = get_vehicle_rl_reward_with_refs([], refs=None, design_priors=None)
    assert r <= -10.0  # catastrophic penalty

def test_enriched_reward_ref_similarity_adds_bonus(refs, priors):
    parts = _car_parts()
    base = get_vehicle_rl_reward(parts)
    if base <= -10.0:
        pytest.skip("Base reward too low for enrichment")
    with_refs = get_vehicle_rl_reward_with_refs(
        parts, refs=refs, design_priors=priors,
        ref_similarity_weight=5.0, bigram_compliance_weight=3.0
    )
    without_refs = get_vehicle_rl_reward_with_refs(
        parts, refs=None, design_priors=None
    )
    # With refs, reward should be higher (bonus > 0 when sim > 0)
    assert with_refs >= without_refs

def test_enriched_reward_weights_scale_bonus(refs, priors):
    parts = _car_parts()
    base = get_vehicle_rl_reward(parts)
    if base <= -10.0:
        pytest.skip("Base reward too low")
    r_low  = get_vehicle_rl_reward_with_refs(parts, refs=refs, design_priors=priors,
                                              ref_similarity_weight=1.0, bigram_compliance_weight=1.0)
    r_high = get_vehicle_rl_reward_with_refs(parts, refs=refs, design_priors=priors,
                                              ref_similarity_weight=10.0, bigram_compliance_weight=5.0)
    assert r_high >= r_low

def test_enriched_reward_ref_identical_to_ref_max_similarity(refs):
    if not refs: pytest.skip("No refs")
    ref = refs[0]
    base = get_vehicle_rl_reward(ref.parts)
    if base <= -10.0:
        pytest.skip("Base reward too low")
    r = get_vehicle_rl_reward_with_refs(ref.parts, refs=refs, design_priors=None)
    # Self-similarity = 1.0, so bonus should be close to ref_similarity_weight
    assert r >= base + 4.9  # 5.0 weight * ~1.0 similarity - epsilon
