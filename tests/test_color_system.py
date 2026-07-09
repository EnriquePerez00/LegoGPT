"""tests/test_color_system.py - TDD for Phase 4 color system"""
import os, math, json, tempfile, numpy as np, torch, pytest
from src.parser import ParsedPart, build_pyg_graph
from src.color_system import (
    ColorPriors, ColorRegionHead, VehicleLegoGNNWithColor,
    VEHICLE_COLOR_VOCAB, COLOR_NEUTRAL_TOKEN, _get_region,
)
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS

N_COLORS = len(VEHICLE_COLOR_VOCAB)

def _part(pid, color=16): return ParsedPart(pid, color, np.eye(4, dtype=np.float32), 0)
def _graph(parts=None):
    if parts is None: parts = [_part("3020.dat")]
    return build_pyg_graph(parts, allowed_parts=VEHICLE_ALLOWED_PARTS, allowed_colors=VEHICLE_COLOR_VOCAB)

class FakeRef:
    def __init__(self, parts): self.parts = parts; self.inventory = {}

# --- _get_region ---
def test_region_wheel(): assert _get_region("6015.dat") == "wheel"
def test_region_cabin(): assert _get_region("3823.dat") == "cabin"
def test_region_body():  assert _get_region("3020.dat") == "body"
def test_region_brick():  assert _get_region("3001.dat") == "body"
def test_region_unknown(): assert _get_region("UNKNOWN.dat") == "body"

# --- VEHICLE_COLOR_VOCAB ---
def test_color_vocab_nonempty(): assert len(VEHICLE_COLOR_VOCAB) > 0
def test_color_vocab_no_neutral(): assert COLOR_NEUTRAL_TOKEN not in VEHICLE_COLOR_VOCAB
def test_color_vocab_has_black(): assert 0 in VEHICLE_COLOR_VOCAB
def test_color_vocab_has_common_body(): assert 4 in VEHICLE_COLOR_VOCAB and 14 in VEHICLE_COLOR_VOCAB

# --- ColorPriors ---
def test_color_priors_empty():
    cp = ColorPriors._empty()
    assert "body" in cp.region_probs
    assert "wheel" in cp.region_probs
    assert "cabin" in cp.region_probs

def test_color_priors_empty_sums_to_one():
    cp = ColorPriors._empty()
    for region, dist in cp.region_probs.items():
        assert abs(sum(dist.values()) - 1.0) < 1e-5, f"{region} probs dont sum to 1"

def test_color_priors_from_refs_with_colors():
    raw_parts = [
        ParsedPart("3020.dat", 4, np.eye(4,dtype=np.float32), 0),   # body: Red
        ParsedPart("6015.dat", 0, np.eye(4,dtype=np.float32), 1),   # wheel: Black
        ParsedPart("3823.dat", 15, np.eye(4,dtype=np.float32), 2),  # cabin: White
    ]
    cp = ColorPriors.from_reference_models([FakeRef(raw_parts)])
    assert 4 in cp.region_probs["body"]
    assert 0 in cp.region_probs["wheel"]
    assert 15 in cp.region_probs["cabin"]

def test_color_priors_from_refs_collapsed_uses_defaults():
    # Phase 0 collapsed refs (all color=16) -> should use defaults
    raw_parts = [ParsedPart("3020.dat", COLOR_NEUTRAL_TOKEN, np.eye(4,dtype=np.float32), 0)]
    cp = ColorPriors.from_reference_models([FakeRef(raw_parts)])
    # Should fall back to defaults
    assert len(cp.region_probs["body"]) > 0

def test_color_priors_get_log_prior_finite():
    cp = ColorPriors._empty()
    lp = cp.get_color_log_prior("body", 14)
    assert math.isfinite(lp) and lp < 0

def test_color_priors_unknown_color_returns_smoothing():
    cp = ColorPriors._empty()
    lp = cp.get_color_log_prior("body", 9999)
    assert abs(lp - math.log(1e-4)) < 1e-6

def test_color_priors_logit_bias_vector_shape():
    cp = ColorPriors._empty()
    v = cp.get_color_logit_bias_vector("wheel", VEHICLE_COLOR_VOCAB)
    assert v.shape == (N_COLORS,)
    assert v.dtype == np.float32

def test_color_priors_save_load_roundtrip(tmp_path):
    cp = ColorPriors._empty()
    path = str(tmp_path / "cp.json")
    cp.save(path=path)
    cp2 = ColorPriors.load(path=path)
    assert cp2.n_reference_models == cp.n_reference_models
    assert abs(cp2.get_color_log_prior("body", 4) - cp.get_color_log_prior("body", 4)) < 1e-6

def test_color_priors_load_missing_returns_empty(tmp_path):
    cp = ColorPriors.load(path=str(tmp_path / "nonexistent.json"))
    assert "body" in cp.region_probs

def test_color_priors_repr():
    r = repr(ColorPriors._empty())
    assert "ColorPriors" in r and "body" in r

# --- ColorRegionHead ---
def test_crh_output_shape_body():
    crh = ColorRegionHead(64)
    out = crh(torch.randn(2, 64), region="body")
    assert out.shape == (2, N_COLORS)

def test_crh_output_shape_wheel():
    crh = ColorRegionHead(64)
    out = crh(torch.randn(2, 64), region="wheel")
    assert out.shape == (2, N_COLORS)

def test_crh_output_shape_cabin():
    crh = ColorRegionHead(64)
    out = crh(torch.randn(2, 64), region="cabin")
    assert out.shape == (2, N_COLORS)

def test_crh_unknown_region_defaults_to_body():
    crh = ColorRegionHead(64)
    out = crh(torch.randn(1, 64), region="unknown")
    assert out.shape == (1, N_COLORS)

def test_crh_predict_color_in_vocab():
    crh = ColorRegionHead(64)
    cp = ColorPriors._empty()
    color = crh.predict_color(torch.randn(1, 64), "3020.dat", color_priors=cp)
    assert color in VEHICLE_COLOR_VOCAB

def test_crh_predict_wheel_in_vocab():
    crh = ColorRegionHead(64)
    cp = ColorPriors._empty()
    color = crh.predict_color(torch.randn(1, 64), "6015.dat", color_priors=cp)
    assert color in VEHICLE_COLOR_VOCAB

def test_crh_predict_no_priors_in_vocab():
    crh = ColorRegionHead(64)
    color = crh.predict_color(torch.randn(1, 64), "3020.dat", color_priors=None)
    assert color in VEHICLE_COLOR_VOCAB

# --- VehicleLegoGNNWithColor ---
def test_vlgc_forward_shapes():
    model = VehicleLegoGNNWithColor(hidden_dim=64, use_fingerprint=True)
    g = _graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, c, t = model(g.x, g.edge_index, batch)
    assert p.shape == (1, len(VEHICLE_ALLOWED_PARTS))
    assert c.shape == (1, N_COLORS)
    assert t.shape == (1, 12)

def test_vlgc_freeze_reduces_trainable():
    model = VehicleLegoGNNWithColor()
    total = sum(p.numel() for p in model.parameters())
    model.freeze_structural()
    assert model.trainable_params() < total

def test_vlgc_unfreeze_restores_all():
    model = VehicleLegoGNNWithColor()
    total = sum(p.numel() for p in model.parameters())
    model.freeze_structural()
    model.unfreeze_all()
    assert model.trainable_params() == total

def test_vlgc_color_head_only_after_freeze():
    model = VehicleLegoGNNWithColor()
    model.freeze_structural()
    color_head_params = sum(p.numel() for p in model.color_head.parameters())
    assert model.trainable_params() == color_head_params

def test_vlgc_no_nan_forward():
    model = VehicleLegoGNNWithColor()
    g = _graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, c, t = model(g.x, g.edge_index, batch)
    assert not torch.isnan(p).any()
    assert not torch.isnan(c).any()

def test_vlgc_part_probs_sum_to_one():
    model = VehicleLegoGNNWithColor()
    g = _graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, _, _ = model(g.x, g.edge_index, batch)
    assert abs(p.exp().sum(dim=1)[0].item() - 1.0) < 0.01
