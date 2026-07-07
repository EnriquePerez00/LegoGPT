import numpy as np
import torch
import pytest
from src.generator import LegoGenerator
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS

class MockModel(torch.nn.Module):
    def __init__(self, num_parts, num_colors):
        super().__init__()
        self.num_parts = num_parts
        self.num_colors = num_colors
        
    def forward(self, x, edge_index, batch):
        # Uniform logits
        part_logits = torch.zeros((1, self.num_parts), dtype=torch.float32)
        color_logits = torch.zeros((1, self.num_colors), dtype=torch.float32)
        transform_preds = torch.zeros((1, 12), dtype=torch.float32)
        return part_logits, color_logits, transform_preds

def test_instruction_conditioning():
    model = MockModel(len(ALLOWED_PARTS), len(ALLOWED_COLORS))
    generator = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS, device="cpu")
    
    # We want a target inventory containing exactly:
    # 2 of "3001.dat" (2x4 brick)
    # 1 of "3003.dat" (2x2 brick)
    target_inv = {
        "3001.dat": 2,
        "3003.dat": 1
    }
    
    # Generate assembly of size 3 using Beam Search with target inventory conditioning
    assembly = generator.generate_beam_search(
        target_num_pieces=3,
        beam_width=2,
        max_candidates=2,
        target_inventory=target_inv
    )
    
    assert len(assembly) == 3
    
    # Verify that the generated parts match the target inventory!
    part_ids = [p.part_id for p in assembly]
    assert part_ids.count("3001.dat") == 2
    assert part_ids.count("3003.dat") == 1
