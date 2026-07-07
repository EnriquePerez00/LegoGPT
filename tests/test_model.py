import pytest
import torch
from torch_geometric.data import Data, Batch
from src.model import LegoGNN, get_device

def test_device_selection():
    """Verify device selection picks mps if available, otherwise cpu."""
    device = get_device()
    assert device in ["mps", "cpu"]
    if torch.backends.mps.is_available():
        assert device == "mps"

def test_model_forward():
    """Verify that LegoGNN performs forward pass and yields correct shape outputs."""
    device = torch.device(get_device())
    model = LegoGNN(num_part_classes=32, num_color_classes=16, hidden_dim=32).to(device)
    model.eval()
    
    # Create two graphs, pack them as a batch
    # Graph 1: 3 nodes, 2 edges
    x1 = torch.randn(3, 60, device=device)
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
    data1 = Data(x=x1, edge_index=edge_index1)
    
    # Graph 2: 2 nodes, 1 edge
    x2 = torch.randn(2, 60, device=device)
    edge_index2 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    data2 = Data(x=x2, edge_index=edge_index2)
    
    batch_data = Batch.from_data_list([data1, data2]).to(device)
    
    with torch.no_grad():
        part_logits, color_logits, transform_preds = model(
            batch_data.x, batch_data.edge_index, batch_data.batch
        )
        
    # Batch size is 2 (2 graphs)
    assert part_logits.shape == (2, 32)
    assert color_logits.shape == (2, 16)
    assert transform_preds.shape == (2, 12)

def test_model_training_step():
    """Test a single optimization step of LegoGNN on the active device to check VRAM loading."""
    device = torch.device(get_device())
    model = LegoGNN(num_part_classes=32, num_color_classes=16, hidden_dim=32).to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate mock training batch of size 4
    graphs = []
    for _ in range(4):
        num_nodes = torch.randint(2, 6, (1,)).item()
        x = torch.randn(num_nodes, 60, device=device)
        # Fully connected-like graph for testing
        edge_index = torch.tensor([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long, device=device).t()
        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        graphs.append(Data(x=x, edge_index=edge_index))
        
    batch_data = Batch.from_data_list(graphs).to(device)
    
    # Forward
    part_logits, color_logits, transform_preds = model(
        batch_data.x, batch_data.edge_index, batch_data.batch
    )
    
    # Mock targets
    target_part = torch.randint(0, 32, (4,), device=device)
    target_color = torch.randint(0, 16, (4,), device=device)
    target_transform = torch.randn(4, 12, device=device)
    
    # Compute multi-task loss
    loss_part = torch.nn.functional.cross_entropy(part_logits, target_part)
    loss_color = torch.nn.functional.cross_entropy(color_logits, target_color)
    loss_trans = torch.nn.functional.mse_loss(transform_preds, target_transform)
    
    loss = loss_part + loss_color + loss_trans
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0.0
    
    # Clean up memory
    del loss, loss_part, loss_color, loss_trans, part_logits, color_logits, transform_preds

    if device.type == "mps":
        torch.mps.empty_cache()
