"""
tests/test_generator_with_priors.py

TDD tests for the DesignPriors integration in LegoGenerator (Fase 1).
Tests:
  - Generator works without priors (legacy backward-compat)
  - Generator works with priors (data-driven mode)
  - Marginal bias vector injected correctly
  - Color collapsed to neutral when priors active
  - _compute_cond_bias: data-driven vs legacy behaviour
  - Beam search with priors + target_inventory produces valid assembly
"""

import os
import numpy as np
import torch
import pytest

from src.parser import ParsedPart, ALLOWED_PARTS, ALLOWED_COLORS
from src.generator import LegoGenerator, COLOR_NEUTRAL, _INVENTORY_BONUS, _INVENTORY_OVERFLOW, _INVENTORY_UNKNOWN
from src.design_priors import DesignPriors
from src.reference_loader import load_from_fixtures
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "vehicles")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockModel(torch.nn.Module):
    """Uniform logits mock — all parts equally likely."""
    def __init__(self, num_parts, num_colors):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors

    def forward(self, x, edge_index, batch):
        part_logits = torch.zeros((1, self.num_parts), dtype=torch.float32)
        color_logits = torch.zeros((1, self.num_colors), dtype=torch.float32)
        transform_preds = torch.zeros((1, 12), dtype=torch.float32)
        return part_logits, color_logits, transform_preds


@pytest.fixture(scope="module")
def vehicle_priors():
    refs = load_from_fixtures(fixtures_dir=FIXTURES_DIR)
    return DesignPriors.from_reference_models(refs)


@pytest.fixture(scope="module")
def empty_priors():
    return DesignPriors._empty()


# ---------------------------------------------------------------------------
# Tests: generator initialisation
# ---------------------------------------------------------------------------

def test_generator_no_priors_marginal_bias_is_none():
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    assert gen.design_priors is None
    assert gen._marginal_bias is None


def test_generator_empty_priors_marginal_bias_is_none(empty_priors):
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS, design_priors=empty_priors)
    # Empty priors (n_reference_models=0) should NOT set marginal bias
    assert gen._marginal_bias is None


def test_generator_with_priors_marginal_bias_set(vehicle_priors):
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    assert gen._marginal_bias is not None
    assert gen._marginal_bias.shape == (len(VEHICLE_ALLOWED_PARTS),)
    assert gen._marginal_bias.dtype == np.float32


# ---------------------------------------------------------------------------
# Tests: _compute_cond_bias — data-driven mode
# ---------------------------------------------------------------------------

def test_cond_bias_no_inventory_no_priors():
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    bias = gen._compute_cond_bias("3001.dat", [], target_inventory=None)
    assert bias == 0.0


def test_cond_bias_inventory_legacy_mode_under_target():
    """Without priors: inventory under target → +2.0."""
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)  # no priors
    bias = gen._compute_cond_bias(
        "3001.dat", [], target_inventory={"3001.dat": 2}
    )
    assert bias == 2.0


def test_cond_bias_inventory_legacy_mode_overflow():
    """Without priors: inventory overflow → -10.0."""
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    t = np.eye(4, dtype=np.float32)
    parts = [ParsedPart("3001.dat", 0, t, 0), ParsedPart("3001.dat", 0, t, 1)]
    bias = gen._compute_cond_bias(
        "3001.dat", parts, target_inventory={"3001.dat": 2}
    )
    assert bias == -10.0


def test_cond_bias_inventory_legacy_mode_unknown_part():
    """Without priors: part not in inventory → -20.0."""
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    bias = gen._compute_cond_bias(
        "UNKNOWN.dat", [], target_inventory={"3001.dat": 2}
    )
    assert bias == -20.0


def test_cond_bias_data_driven_under_target(vehicle_priors):
    """With priors: needed part → bonus > 0."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    # 0 of 4 wheels placed → full bonus
    bias = gen._compute_cond_bias(
        "6015.dat", [], target_inventory={"6015.dat": 4}
    )
    # Should include inventory bonus + bigram (empty parts_list = no bigram)
    assert bias > 0.0


def test_cond_bias_data_driven_overflow(vehicle_priors):
    """With priors: overflow part → strong penalty."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    t = np.eye(4, dtype=np.float32)
    # Already have 4 wheels, target is 4
    parts = [ParsedPart("6015.dat", COLOR_NEUTRAL, t, i) for i in range(4)]
    bias = gen._compute_cond_bias(
        "6015.dat", parts, target_inventory={"6015.dat": 4}
    )
    assert bias < 0.0


def test_cond_bias_data_driven_unknown_part(vehicle_priors):
    """With priors: part not in inventory → large negative bias."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    bias = gen._compute_cond_bias(
        "3001.dat", [], target_inventory={"6015.dat": 4}
    )
    assert bias <= _INVENTORY_UNKNOWN


# ---------------------------------------------------------------------------
# Tests: color behaviour
# ---------------------------------------------------------------------------

def test_generator_no_priors_uses_model_colors():
    """Without priors, color comes from model logits (not forced neutral)."""
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    # use_neutral_color should be False
    use_neutral = (
        gen.design_priors is not None
        and gen.design_priors.n_reference_models > 0
    )
    assert use_neutral is False


def test_generator_with_priors_uses_neutral_color(vehicle_priors):
    """With priors active, generator should operate in color-neutral mode."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    use_neutral = (
        gen.design_priors is not None
        and gen.design_priors.n_reference_models > 0
    )
    assert use_neutral is True


# ---------------------------------------------------------------------------
# Tests: beam search integration (no priors — backward compat)
# ---------------------------------------------------------------------------

def test_beam_search_no_priors_produces_assembly():
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    result = gen.generate_beam_search(target_num_pieces=3, beam_width=2, max_candidates=2)
    assert isinstance(result, list)
    # May be empty if no valid connections found with mock model, that's ok
    # Key: must not raise


def test_beam_search_no_priors_with_inventory():
    """Legacy inventory conditioning still works (backward compat)."""
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    gen = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS)
    target_inv = {"3001.dat": 2, "3003.dat": 1}
    result = gen.generate_beam_search(
        target_num_pieces=3,
        beam_width=2,
        max_candidates=2,
        target_inventory=target_inv,
    )
    assert isinstance(result, list)
    assert len(result) == 3
    assert sum(1 for p in result if p.part_id == "3001.dat") == 2
    assert sum(1 for p in result if p.part_id == "3003.dat") == 1


# ---------------------------------------------------------------------------
# Tests: beam search with priors (Fase 1 integration)
# ---------------------------------------------------------------------------

def test_beam_search_with_priors_produces_assembly(vehicle_priors):
    """Generator with priors produces a valid assembly."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    result = gen.generate_beam_search(
        target_num_pieces=4, beam_width=2, max_candidates=3
    )
    assert isinstance(result, list)


def test_beam_search_with_priors_colors_are_neutral(vehicle_priors):
    """All parts in priors-mode output should have COLOR_NEUTRAL."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    result = gen.generate_beam_search(
        target_num_pieces=4, beam_width=2, max_candidates=3
    )
    for p in result:
        assert p.color == COLOR_NEUTRAL, (
            f"Expected COLOR_NEUTRAL={COLOR_NEUTRAL}, got {p.color} for {p.part_id}"
        )


def test_beam_search_with_priors_and_inventory(vehicle_priors):
    """Priors + inventory: data-driven biases cooperate correctly."""
    model = MockModel(len(VEHICLE_ALLOWED_PARTS), len(VEHICLE_ALLOWED_COLORS))
    gen = LegoGenerator(
        model, VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS,
        design_priors=vehicle_priors
    )
    target_inv = {"3020.dat": 1, "6015.dat": 2}
    result = gen.generate_beam_search(
        target_num_pieces=3,
        beam_width=2,
        max_candidates=3,
        target_inventory=target_inv,
    )
    assert isinstance(result, list)
    # All parts should still be color-neutral
    for p in result:
        assert p.color == COLOR_NEUTRAL
