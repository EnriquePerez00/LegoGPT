import pytest
import torch
from generative.llm_pipeline.tokenizer_ldr import LdrTokenizer
from generative.llm_pipeline.train_llm import BrickLLM, train_llm_step

def test_ldr_tokenizer():
    tokenizer = LdrTokenizer()
    
    # Sample LDraw text
    sample_ldr = """
    0 Author: LegoGPT Agent
    1 14 10.3 24.1 30.8 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    0 STEP
    1 4 -10.0 -48.0 20.0 0.0 0.0 1.0 0.0 1.0 0.0 -1.0 0.0 0.0 3003.dat
    """
    
    # 1. Clean and normalize
    normalized = tokenizer.clean_and_normalize_ldr_text(sample_ldr)
    
    # Check that coordinates are rounded
    assert "1 14 10 24 31" in normalized
    assert "0 STEP" in normalized
    # Comment should be stripped
    assert "Author" not in normalized
    
    # 2. Tokenize (encode/decode)
    token_ids = tokenizer.encode(sample_ldr)
    assert len(token_ids) > 0
    
    decoded = tokenizer.decode(token_ids)
    assert "3001.dat" in decoded
    assert "3003.dat" in decoded

def test_brick_llm_model():
    tokenizer = LdrTokenizer()
    vocab_size = len(tokenizer.vocab)
    
    model = BrickLLM(vocab_size=vocab_size, embed_dim=16, num_heads=2, hidden_dim=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Sample token IDs
    token_ids = [2, 10, 20, 30, 40, 3]
    
    loss_val = train_llm_step(model, optimizer, token_ids, device="cpu")
    assert isinstance(loss_val, float)
    assert loss_val > 0.0
