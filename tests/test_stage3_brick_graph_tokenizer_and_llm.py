import os
import tempfile
import numpy as np
import torch
import pytest

from src.parser import ParsedPart, build_pyg_graph, ALLOWED_PARTS, ALLOWED_COLORS
from src.validator import check_collisions, check_connectivity_and_gravity
from src.io_parser import StudioIOParser
from generative.llm_pipeline.brick_graph_tokenizer import BrickGraphTokenizer
from generative.llm_pipeline.train_llm import (
    HybridGraphBrickLLM,
    train_llm_step,
    generate_constrained_brick_graph,
)


def _make_part(part_id: str, color: int, x: float, y: float, z: float, step_id: int = 0) -> ParsedPart:
    tf = np.eye(4, dtype=np.float32)
    tf[:3, 3] = [x, y, z]
    return ParsedPart(part_id=part_id, color=color, transform=tf, step_id=step_id)


def test_brick_graph_tokenizer_roundtrip_and_io_compilation():
    """
    Verifies that a 3D assembly with port couplings and bilateral X-mirror symmetry:
    1. Serializes into symbolic BrickNet DSL (CONNECT ... TO ... and SNAP_MIRROR_X).
    2. Encodes to token IDs and decodes back to identical DSL statements.
    3. Compiles deterministically back to 3D ParsedParts and a valid .io file with
       zero collisions and full ground connectivity.
    """
    tokenizer = BrickGraphTokenizer()

    # Base 2x4 plate (3020.dat) at origin (Y=0, top studs at Y=-8)
    # Left & Right symmetric 1x2 bricks (3004.dat) at X = -10 and X = +10, Y = -8
    parts = [
        _make_part("3020.dat", 7, 0.0, 0.0, 0.0, step_id=0),
        _make_part("3004.dat", 4, -10.0, -8.0, 0.0, step_id=1),
        _make_part("3004.dat", 4,  10.0, -8.0, 0.0, step_id=1),
    ]

    dsl = tokenizer.parts_to_symbolic_dsl(parts, schedule_if_flat=False)
    assert "AT origin" in dsl
    assert "CONNECT node_1." in dsl
    assert "SNAP_MIRROR_X OF node_1" in dsl

    # Encode -> Decode round-trip
    token_ids = tokenizer.encode(dsl)
    decoded_dsl = tokenizer.decode(token_ids)
    assert decoded_dsl.strip() == dsl.strip()

    # Compile DSL back to 3D parts and .io archive
    with tempfile.TemporaryDirectory() as tmpdir:
        io_out = os.path.join(tmpdir, "compiled_symbolic.io")
        io_path, compiled_parts = tokenizer.compile_dsl_to_io(decoded_dsl, io_out)

        assert os.path.exists(io_path)
        assert StudioIOParser(io_path).is_valid_io() is True
        assert len(compiled_parts) == 3

        # Verify physical validity of compiled assembly
        assert len(check_collisions(compiled_parts)) == 0
        assert check_connectivity_and_gravity(compiled_parts) is True

        # Verify bilateral mirror coordinates were reconstructed accurately
        x1 = float(compiled_parts[1].transform[0, 3])
        x2 = float(compiled_parts[2].transform[0, 3])
        assert abs(x1 + x2) < 1e-4


def test_hybrid_graph_brick_llm_training_and_constrained_generation_mps():
    """
    Verifies that HybridGraphBrickLLM trains on MPS/CPU and generates 100% physically
    connected, collision-free assemblies compiled to .io via generate_constrained_brick_graph.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = BrickGraphTokenizer()

    sample_dsl = (
        "[STEP_0] ADD part_3001.dat color_4 AS node_0 AT origin ROT 0\n"
        "[STEP_1] ADD part_3003.dat color_14 AS node_1 CONNECT node_1.bottom_socket_0_0 TO node_0.top_stud_0_0 ROT 0\n"
        "[STEP_2] ADD part_3005.dat color_1 AS node_2 CONNECT node_2.bottom_socket_0_0 TO node_1.top_stud_1_1 ROT 0"
    )
    tok_ids = tokenizer.encode(sample_dsl)

    model = HybridGraphBrickLLM(vocab_size=len(tokenizer.vocab) + 16, embed_dim=32, num_heads=4, hidden_dim=64, num_layers=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss1 = train_llm_step(model, optimizer, tok_ids, device=device)
    loss2 = train_llm_step(model, optimizer, tok_ids, device=device)
    assert loss1 > 0.0
    assert loss2 > 0.0

    # Constrained physical generation
    gen_dsl, gen_parts = generate_constrained_brick_graph(
        model=model,
        tokenizer=tokenizer,
        target_num_pieces=5,
        allowed_parts=["3020.dat", "3004.dat", "3003.dat", "3005.dat"],
        allowed_colors=[4, 14, 7],
        device=device,
        symmetric_pairs=True,
    )

    assert len(gen_parts) == 5
    assert "ADD part_" in gen_dsl
    assert len(check_collisions(gen_parts)) == 0
    assert check_connectivity_and_gravity(gen_parts) is True
    # Ground rule check: all Y <= 0.0 and <= 4 ground parts
    assert all(float(p.transform[1, 3]) <= 0.01 for p in gen_parts)
    assert sum(1 for p in gen_parts if abs(float(p.transform[1, 3])) <= 1.0) <= 4
