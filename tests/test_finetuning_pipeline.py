import pytest
import torch
import torch.optim as optim
from src.model import LegoGNN
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS
from scripts.train_vehicle_pipeline import train_supervised_epoch, prepare_dataset

def test_two_stage_training_flow(tmp_path):
    device = "cpu"
    
    # 1. Instantiate the GNN model
    model = LegoGNN(
        num_part_classes=len(VEHICLE_ALLOWED_PARTS),
        num_color_classes=len(VEHICLE_ALLOWED_COLORS),
        hidden_dim=8,
        allowed_parts=None
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 2. Mock sequence data
    # We construct a mock dataset structure matching our format
    from tests.test_vehicle_reward import create_mock_part
    w1 = create_mock_part("42610.dat", 0, -20.0, 0.0, -20.0)
    w2 = create_mock_part("42610.dat", 0, 20.0, 0.0, -20.0)
    
    mock_sample = {
        "input_state": [w1],
        "target_part": w2,
        "set_name": "mock_car"
    }
    mock_dataset = [mock_sample]
    
    # 3. Test Stage 1 Supervised epoch (color is masked)
    loss_stage1 = train_supervised_epoch(model, optimizer, mock_dataset, device, stage=1)
    assert isinstance(loss_stage1, float)
    assert loss_stage1 > 0.0
    
    # Save checkpoint
    chk_path = tmp_path / "vehicle_base_chassis_test.pt"
    torch.save(model.state_dict(), chk_path)
    assert chk_path.exists()
    
    # 4. Test Stage 2 Fine-Tuning loading and training
    new_model = LegoGNN(
        num_part_classes=len(VEHICLE_ALLOWED_PARTS),
        num_color_classes=len(VEHICLE_ALLOWED_COLORS),
        hidden_dim=8,
        allowed_parts=None
    ).to(device)
    new_model.load_state_dict(torch.load(chk_path))
    
    new_optimizer = optim.Adam(new_model.parameters(), lr=0.01)
    loss_stage2 = train_supervised_epoch(new_model, new_optimizer, mock_dataset, device, stage=2)
    assert isinstance(loss_stage2, float)
    assert loss_stage2 > 0.0
