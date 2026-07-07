import pytest
import torch
from generative.llm_pipeline.tokenizer_ldr import LdrTokenizer
from generative.llm_pipeline.train_llm import BrickLLM
from generative.llm_pipeline.rl_loop import reward_function, train_rl_ppo_step

def test_reward_function():
    # 1. Stable stacked bricks (Y increases downwards, so Y=0 is ground, Y=-24 is on top)
    stable_ldr = """
    1 14 0 0 0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    1 14 0 -24 0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    """
    assert reward_function(stable_ldr) == 1.0
    
    # 2. Unstable/tipping brick (Offset too far horizontally: X=200 LDU, dimensions are 40x24x80)
    # The center of mass of the top brick will fall outside the base brick footprint (which only goes X: -20 to 20)
    unstable_ldr = """
    1 14 0 0 0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    1 14 200 -24 0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    """
    assert reward_function(unstable_ldr) == -1.0
    
    # 3. Floating brick (disconnected)
    floating_ldr = """
    1 14 0 0 0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    1 14 0 -100 200 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 3001.dat
    """
    assert reward_function(floating_ldr) == -1.0

def test_rl_ppo_training_step():
    tokenizer = LdrTokenizer()
    vocab_size = len(tokenizer.vocab)
    model = BrickLLM(vocab_size=vocab_size, embed_dim=16, num_heads=2, hidden_dim=32, num_layers=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Run one step of PPO training
    loss_val = train_rl_ppo_step(model, optimizer, tokenizer, prompt_text="1 14 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat", device="cpu")
    assert isinstance(loss_val, float)
