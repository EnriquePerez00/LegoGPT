import numpy as np
import pytest
from src.parser import ParsedPart
from src.vehicle_rules import evaluate_vehicle_topology, get_vehicle_rl_reward

def create_mock_part(part_id: str, color: int, x: float, y: float, z: float) -> ParsedPart:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=transform, step_id=0)

def test_evaluate_vehicle_topology():
    # 1. Construct a valid simple 4-wheeled vehicle frame
    # 4 wheels at bottom layer (Y = 0)
    w1 = create_mock_part("42610.dat", 0, -20.0, 0.0, -40.0) # front-left
    w2 = create_mock_part("42610.dat", 0, 20.0, 0.0, -40.0)  # front-right
    w3 = create_mock_part("42610.dat", 0, -20.0, 0.0, 40.0)  # back-left
    w4 = create_mock_part("42610.dat", 0, 20.0, 0.0, 40.0)   # back-right
    
    # Flat chassis plate in the middle, sitting directly on top of the wheels (Y = -16)
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    # Steering column at higher layer (Y = -24) and centered (X = 0)
    cabin = create_mock_part("3829.dat", 14, 0.0, -24.0, 0.0)
    
    parts = [w1, w2, w3, w4, p1, cabin]
    
    metrics = evaluate_vehicle_topology(parts)
    assert metrics["wheel_count"] == 4
    assert metrics["wheels_at_bottom"] is True
    assert metrics["has_cabin_parts"] is True
    assert metrics["cabin_height_valid"] is True
    assert metrics["cabin_centered"] is True
    assert metrics["symmetry_score"] >= 0.8  # Left-right mirrored

def test_invalid_vehicle_topology():
    # Construct a vehicle missing wheels
    w1 = create_mock_part("42610.dat", 0, -20.0, 0.0, -40.0)
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    parts = [w1, p1]
    metrics = evaluate_vehicle_topology(parts)
    assert metrics["wheel_count"] == 1
    
    # Bilateral asymmetry
    w2 = create_mock_part("42610.dat", 0, 10.0, 0.0, -40.0)  # No matching counterpart at -10
    parts_asym = [w1, w2, p1]
    metrics_asym = evaluate_vehicle_topology(parts_asym)
    assert metrics_asym["symmetry_score"] < 1.0

def test_vehicle_rl_reward():
    # A valid structure gets a higher reward
    w1 = create_mock_part("42610.dat", 0, -10.0, 0.0, -30.0)
    w2 = create_mock_part("42610.dat", 0, 10.0, 0.0, -30.0)
    w3 = create_mock_part("42610.dat", 0, -10.0, 0.0, 30.0)
    w4 = create_mock_part("42610.dat", 0, 10.0, 0.0, 30.0)
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    valid_parts = [w1, w2, w3, w4, p1]
    
    reward_valid = get_vehicle_rl_reward(valid_parts)
    
    # An invalid structure (e.g. 1 wheel) gets a lower/negative reward
    invalid_parts = [w1, p1]
    reward_invalid = get_vehicle_rl_reward(invalid_parts)
    
    assert reward_valid > reward_invalid

def test_flexible_wheel_rewards():
    # 4 wheels setup (valid stable base)
    w1 = create_mock_part("42610.dat", 0, -10.0, 0.0, -30.0)
    w2 = create_mock_part("42610.dat", 0, 10.0, 0.0, -30.0)
    w3 = create_mock_part("42610.dat", 0, -10.0, 0.0, 30.0)
    w4 = create_mock_part("42610.dat", 0, 10.0, 0.0, 30.0)
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    parts_4_wheels = [w1, w2, w3, w4, p1]
    reward_4 = get_vehicle_rl_reward(parts_4_wheels)
    
    # 2 wheels setup (motorcycle/trailer style - touches ground at 2 points <= 4)
    parts_2_wheels = [w1, w3, p1]
    reward_2 = get_vehicle_rl_reward(parts_2_wheels)
    
    # 1 wheel setup
    parts_1_wheel = [w1, p1]
    reward_1 = get_vehicle_rl_reward(parts_1_wheel)
    
    # 4 wheels is preferred, but 2 wheels (soft constraint) should still be significantly better than 1 wheel
    assert reward_4 > reward_2
    assert reward_2 > reward_1

def test_blocked_aesthetic_penalty():
    # 4 wheels base
    w1 = create_mock_part("42610.dat", 0, -10.0, 0.0, -30.0)
    w2 = create_mock_part("42610.dat", 0, 10.0, 0.0, -30.0)
    w3 = create_mock_part("42610.dat", 0, -10.0, 0.0, 30.0)
    w4 = create_mock_part("42610.dat", 0, 10.0, 0.0, 30.0)
    
    # Flat plate
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    # CASE A: Aesthetic Tile 1x2 at the top (unblocked, Y = -24)
    tile_unblocked = create_mock_part("3069b.dat", 14, 0.0, -24.0, 0.0)
    parts_unblocked = [w1, w2, w3, w4, p1, tile_unblocked]
    
    # CASE B: Aesthetic Tile 1x2 blocked (Plate stacked on top of it: Tile at Y = -16, Plate at Y = -24)
    tile_blocked = create_mock_part("3069b.dat", 14, 0.0, -16.0, 0.0)
    plate_on_top = create_mock_part("3023.dat", 1, 0.0, -24.0, 0.0) # Stacked on top of tile
    parts_blocked = [w1, w2, w3, w4, tile_blocked, plate_on_top]
    
    metrics_unblocked = evaluate_vehicle_topology(parts_unblocked)
    metrics_blocked = evaluate_vehicle_topology(parts_blocked)
    
    assert metrics_unblocked["blocked_aesthetic_count"] == 0
    assert metrics_blocked["blocked_aesthetic_count"] == 1
    
    reward_unblocked = get_vehicle_rl_reward(parts_unblocked)
    reward_blocked = get_vehicle_rl_reward(parts_blocked)
    
    # Blocked tile receives a penalty, thus lower reward
    assert reward_unblocked > reward_blocked

