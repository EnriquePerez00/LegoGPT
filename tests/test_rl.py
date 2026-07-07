import numpy as np
import torch
import pytest
from src.parser import ParsedPart, ALLOWED_PARTS, ALLOWED_COLORS
from src.model import LegoGNN
from src.rl_train import calculate_interlocking_score, get_rl_reward, train_rl_step

IDENTITY_ROT = np.eye(3, dtype=np.float32)

def create_part(part_id: str, color: int, x: float, y: float, z: float) -> ParsedPart:
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = IDENTITY_ROT
    transform[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=transform, step_id=0)

def test_interlocking_score():
    # Construct a classic interlocking bridge with aligned stud pitch (multiple of 20 LDU):
    # Layer 0 (ground): Two bricks separated in space
    # Brick A: 2x4 at [0, 0, 0]
    # Brick B: 2x4 at [0, 0, 40] (separated by 40 LDU, they can be spanned by a third brick)
    part_a = create_part("3001.dat", 1, 0.0, 0.0, 0.0)
    part_b = create_part("3001.dat", 1, 0.0, 0.0, 40.0)
    
    # Layer 1 (above): A 2x4 brick spanning both, placed at [0, -24, 20]
    # It bridges across Brick A and Brick B
    part_bridge = create_part("3001.dat", 2, 0.0, -24.0, 20.0)
    
    parts = [part_a, part_b, part_bridge]
    
    score = calculate_interlocking_score(parts)
    # The bridge connects to both part_a and part_b below it, so interlocking score should be >= 1.0!
    assert score >= 1.0
    
    # Check that reward is positive and stable
    reward = get_rl_reward(parts)
    assert reward >= 15.0  # 10 base + 5 interlock

def test_rl_training_step():
    # Instantiate model and optimizer
    model = LegoGNN(num_part_classes=len(ALLOWED_PARTS), num_color_classes=len(ALLOWED_COLORS), hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    loss_val = train_rl_step(model, optimizer, device="cpu")
    # Verify that loss is returned as a float (and runs fine)
    assert isinstance(loss_val, float)
