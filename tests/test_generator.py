import numpy as np
import torch
import pytest
from src.parser import ParsedPart
from src.generator import LegoGenerator
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS

class MockModel(torch.nn.Module):
    def __init__(self, num_parts, num_colors):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors
        
    def forward(self, x, edge_index, batch):
        # Return mock uniform logits
        part_logits = torch.zeros((1, self.num_parts), dtype=torch.float32)
        color_logits = torch.zeros((1, self.num_colors), dtype=torch.float32)
        transform_preds = torch.zeros((1, 12), dtype=torch.float32)
        return part_logits, color_logits, transform_preds

def test_lego_generator_beam_search():
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    generator = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS, device="cpu")
    
    # Generate an assembly of 3 bricks using Beam Search
    assembly = generator.generate_beam_search(target_num_pieces=3, beam_width=2, max_candidates=2)
    
    assert len(assembly) == 3
    assert all(isinstance(p, ParsedPart) for p in assembly)
    
    # Verify physical stability and collision-free placement using graph validator
    from src.graph_validator import LegoGraphValidator
    validator = LegoGraphValidator()
    
    for i in range(1, len(assembly)):
        state = assembly[:i]
        new_part = assembly[i]
        assert validator.can_place_brick(new_part, current_state=state) is True
