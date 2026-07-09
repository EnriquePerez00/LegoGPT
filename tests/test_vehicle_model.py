"""
tests/test_vehicle_model.py - TDD for VehicleHierarchicalSoftmax, GraphFingerprintConditioner, VehicleLegoGNN
"""
import os, numpy as np, torch, pytest
from src.vehicle_model import (
    VehicleHierarchicalSoftmax, GraphFingerprintConditioner, VehicleLegoGNN,
    FINGERPRINT_DIM, CATEGORY_CHASSIS, CATEGORY_BRICK_BODY, CATEGORY_WHEEL,
    CATEGORY_CABIN, CATEGORY_SLOPE, CATEGORY_SPECIAL,
)
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS
from src.parser import ParsedPart, build_pyg_graph

N_PARTS = len(VEHICLE_ALLOWED_PARTS)

# --- VehicleHierarchicalSoftmax ---

def test_vhs_output_shape():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    out = m(torch.randn(3, 64))
    assert out.shape == (3, N_PARTS)

def test_vhs_probs_sum_to_one():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    probs = m(torch.randn(4, 64)).exp()
    for i in range(4):
        assert abs(probs[i].sum().item() - 1.0) < 0.01

def test_vhs_no_nan():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    out = m(torch.randn(2, 64))
    assert not torch.isnan(out).any()

def test_vhs_compute_loss_positive():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    h = torch.randn(4, 64)
    tgts = torch.tensor([0, 5, 20, 50])
    loss = m.compute_loss(h, tgts)
    assert loss.item() > 0
    assert not torch.isnan(loss)

def test_vhs_category_coverage():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    total = (len(m.cat_chassis) + len(m.cat_brick) + len(m.cat_wheel) +
             len(m.cat_cabin) + len(m.cat_slope) + len(m.cat_special))
    assert total == N_PARTS, f"Category coverage: {total} != {N_PARTS}"

def test_vhs_wheel_category_nonempty():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    assert len(m.cat_wheel) > 0

def test_vhs_chassis_category_nonempty():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    assert len(m.cat_chassis) > 0

def test_vhs_l1_labels_valid():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    assert m.l1_labels.min() >= 0
    assert m.l1_labels.max() <= 1

def test_vhs_l2_labels_valid():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    assert m.l2_labels.min() >= 0
    assert m.l2_labels.max() <= 3

def test_vhs_all_targets_compute_loss():
    m = VehicleHierarchicalSoftmax(64, VEHICLE_ALLOWED_PARTS)
    h = torch.randn(N_PARTS, 64)
    tgts = torch.arange(N_PARTS)
    loss = m.compute_loss(h, tgts)
    assert not torch.isnan(loss)
    assert loss.item() > 0

# --- GraphFingerprintConditioner ---

def test_gfc_output_shape():
    gfc = GraphFingerprintConditioner(FINGERPRINT_DIM, 32)
    fp = torch.randn(2, FINGERPRINT_DIM)
    out = gfc(fp)
    assert out.shape == (2, 32)

def test_gfc_build_fingerprint_shape():
    inv = {"3020.dat": 2, "6015.dat": 4}
    parts = [ParsedPart("3020.dat", 16, np.eye(4, dtype=np.float32), 0)]
    fp = GraphFingerprintConditioner.build_fingerprint(inv, parts, target_parts_count=50)
    assert fp.shape == (1, FINGERPRINT_DIM)

def test_gfc_fingerprint_range():
    inv = {"3020.dat": 2, "6015.dat": 4, "3823.dat": 1}
    parts = [ParsedPart("3020.dat", 16, np.eye(4, dtype=np.float32), 0)]
    fp = GraphFingerprintConditioner.build_fingerprint(inv, parts)
    assert (fp >= 0).all() and (fp <= 1.001).all()

def test_gfc_empty_parts():
    fp = GraphFingerprintConditioner.build_fingerprint({}, [], target_parts_count=50)
    assert fp.shape == (1, FINGERPRINT_DIM)
    assert not torch.isnan(fp).any()

def test_gfc_empty_inventory():
    parts = [ParsedPart("3020.dat", 16, np.eye(4, dtype=np.float32), 0)]
    fp = GraphFingerprintConditioner.build_fingerprint({}, parts)
    assert fp.shape == (1, FINGERPRINT_DIM)

def test_gfc_inventory_reflects_in_fp():
    inv = {"3020.dat": 10}
    fp = GraphFingerprintConditioner.build_fingerprint(inv, [])
    idx = VEHICLE_ALLOWED_PARTS.index("3020.dat")
    assert fp[0, idx].item() > 0.0

# --- VehicleLegoGNN ---

def _dummy_graph():
    p = ParsedPart("3020.dat", 16, np.eye(4, dtype=np.float32), 0)
    g = build_pyg_graph([p], allowed_parts=VEHICLE_ALLOWED_PARTS, allowed_colors=list(range(8)))
    return g

def test_vehicle_gnn_output_shapes():
    model = VehicleLegoGNN(num_color_classes=8, hidden_dim=64, cond_dim=32, use_fingerprint=True)
    g = _dummy_graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    fp = torch.zeros(1, FINGERPRINT_DIM)
    p, c, t = model(g.x, g.edge_index, batch, fingerprint=fp)
    assert p.shape == (1, N_PARTS)
    assert c.shape == (1, 8)
    assert t.shape == (1, 12)

def test_vehicle_gnn_no_fingerprint():
    model = VehicleLegoGNN(use_fingerprint=False)
    g = _dummy_graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, c, t = model(g.x, g.edge_index, batch, fingerprint=None)
    assert p.shape[1] == N_PARTS

def test_vehicle_gnn_fingerprint_none_uses_zeros():
    model = VehicleLegoGNN(use_fingerprint=True)
    g = _dummy_graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p1, _, _ = model(g.x, g.edge_index, batch, fingerprint=None)
    fp0 = torch.zeros(1, FINGERPRINT_DIM)
    p2, _, _ = model(g.x, g.edge_index, batch, fingerprint=fp0)
    assert p1.shape == p2.shape

def test_vehicle_gnn_no_nan():
    model = VehicleLegoGNN(use_fingerprint=True)
    g = _dummy_graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, c, t = model(g.x, g.edge_index, batch)
    assert not torch.isnan(p).any()
    assert not torch.isnan(c).any()

def test_vehicle_gnn_part_probs_sum_to_one():
    model = VehicleLegoGNN(use_fingerprint=True)
    g = _dummy_graph()
    batch = torch.zeros(g.num_nodes, dtype=torch.long)
    p, _, _ = model(g.x, g.edge_index, batch)
    prob_sum = p.exp().sum(dim=1)
    assert abs(prob_sum[0].item() - 1.0) < 0.01

def test_vehicle_gnn_fingerprint_dim_constant():
    assert FINGERPRINT_DIM == N_PARTS + 6, f"Expected {N_PARTS+6}, got {FINGERPRINT_DIM}"
