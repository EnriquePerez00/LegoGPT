import os
import tempfile
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from generative.llm_pipeline.tokenizer_ldr import LdrTokenizer
from generative.llm_pipeline.train_llm import BrickLLM
from src.parser import parse_ldraw_file
from src.validator import check_connectivity_and_gravity, get_part_dimensions

def calculate_com_stability(parts: list) -> bool:
    """
    Checks if the center of mass (COM) of the structure projected on the X-Z plane
    falls within the bounding box of the base parts (those touching the ground).
    """
    if not parts:
        return False
        
    # Get positions and dimensions
    positions = []
    masses = []
    base_corners = []
    
    # Ground Y-level in LDraw (lowest layer is the maximum Y value)
    y_vals = [p.transform[1, 3] for p in parts]
    max_y = max(y_vals)
    
    for p in parts:
        pos = p.transform[:3, 3]
        dim = get_part_dimensions(p.part_id)
        # Approximate mass by volume
        vol = dim[0] * dim[1] * dim[2]
        
        positions.append(pos)
        masses.append(vol)
        
        # If the part is on the ground (within 12.0 LDU of max_y, which is half standard brick height)
        if abs(pos[1] - max_y) <= 12.0:
            # Add its footprint corners in X-Z plane
            w, d = dim[0], dim[2]
            base_corners.append([pos[0] - w/2.0, pos[2] - d/2.0])
            base_corners.append([pos[0] + w/2.0, pos[2] + d/2.0])
            
    if not base_corners:
        return False
        
    # Compute Center of Mass
    positions = np.array(positions)
    masses = np.array(masses)[:, None]
    com = np.sum(positions * masses, axis=0) / np.sum(masses)
    
    # Bounding box of base footprint
    base_corners = np.array(base_corners)
    min_x, min_z = np.min(base_corners, axis=0)
    max_x, max_z = np.max(base_corners, axis=0)
    
    # Check if COM X-Z is within the base bounding box (with a small margin of 2 LDU)
    is_stable_x = (min_x - 2.0) <= com[0] <= (max_x + 2.0)
    is_stable_z = (min_z - 2.0) <= com[2] <= (max_z + 2.0)
    
    return is_stable_x and is_stable_z

def reward_function(generated_ldr_text: str, mode: str = "general") -> float:
    """
    Calculates physical reward:
    - In 'vehicle' mode, delegates to get_vehicle_rl_reward.
    - Otherwise returns +1.0 if stable and connected, and -1.0 if not.
    """
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ldr", delete=False) as f:
        f.write(generated_ldr_text)
        temp_path = f.name
        
    try:
        parts = parse_ldraw_file(temp_path)
        if not parts:
            return -1.0
            
        if mode == "vehicle":
            from src.vehicle_rules import get_vehicle_rl_reward
            # Normalize reward from [-20, 30] range to [-1.0, 1.0] range for LLM PPO stability
            raw_reward = get_vehicle_rl_reward(parts)
            normalized_reward = np.clip(raw_reward / 15.0, -1.0, 1.0)
            return float(normalized_reward)
            
        # 1. Structural connectivity and gravity check
        stable = check_connectivity_and_gravity(parts)
        if not stable:
            return -1.0
            
        # 2. Center of mass stability check
        com_stable = calculate_com_stability(parts)
        if not com_stable:
            return -1.0
            
        return 1.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def train_rl_ppo_step(model, optimizer, tokenizer, prompt_text: str, device="cpu", mode="general") -> float:
    """
    Simplified Policy Gradient (REINFORCE/PPO) step for BrickLLM.
    Generates a sequence from the prompt, evaluates it via reward_function,
    and updates weights.
    """
    device = torch.device(device)
    model.train()
    
    # Encode prompt
    prompt_ids = tokenizer.encode(prompt_text)
    input_ids = list(prompt_ids)
    
    # Generate autoregressively (max 50 tokens)
    log_probs = []
    for _ in range(50):
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
            
        # Sample next token from last logits
        probs = F.softmax(logits[0, -1], dim=-1)
        dist = torch.distributions.Categorical(probs)
        next_token = dist.sample()
        
        input_ids.append(next_token.item())
        
        # Track log probability (enable grad for backward)
        x_grad = torch.tensor([input_ids[:-1]], dtype=torch.long, device=device)
        logits_grad = model(x_grad)
        prob_grad = F.log_softmax(logits_grad[0, -1], dim=-1)
        log_probs.append(prob_grad[next_token])
        
        if next_token.item() == tokenizer.w2i["[EOS]"]:
            break
            
    # Decode to LDraw text
    generated_text = tokenizer.decode(input_ids)
    
    # Calculate reward
    reward = reward_function(generated_text, mode=mode)
    
    # Policy gradient loss: -Reward * sum(log_prob)
    optimizer.zero_grad()
    loss = -reward * torch.stack(log_probs).sum()
    loss.backward()
    optimizer.step()
    
    return loss.item()
