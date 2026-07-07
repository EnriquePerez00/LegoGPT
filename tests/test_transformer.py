import torch
import pytest
from src.model import LegoGraphTransformer, HierarchicalSoftmax
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS
from torch_geometric.data import Data, Batch

def test_graph_transformer_forward():
    # Instantiate LegoGraphTransformer
    model = LegoGraphTransformer(
        num_part_classes=len(ALLOWED_PARTS),
        num_color_classes=len(ALLOWED_COLORS),
        hidden_dim=16,
        allowed_parts=ALLOWED_PARTS
    )
    
    # Check that model structure contains TransformerEncoder
    assert hasattr(model, "transformer")
    assert isinstance(model.head_part, HierarchicalSoftmax)
    
    # Create mock PyG graph data
    # 2 nodes, 34-dimensional feature vectors (one-hot part, one-hot color, translation, rotation)
    x = torch.randn((2, len(ALLOWED_PARTS) + len(ALLOWED_COLORS) + 12), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    batch = torch.tensor([0, 0], dtype=torch.long)  # both nodes in batch 0
    
    model.eval()
    with torch.no_grad():
        part_out, color_logits, transform_preds = model(x, edge_index, batch)
        
    # Check shapes
    assert part_out.shape == (1, len(ALLOWED_PARTS))
    assert color_logits.shape == (1, len(ALLOWED_COLORS))
    assert transform_preds.shape == (1, 12)
