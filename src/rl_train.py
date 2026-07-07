import torch
import torch.nn.functional as F
import numpy as np
from src.parser import build_pyg_graph, ALLOWED_PARTS, ALLOWED_COLORS
from src.validator import check_connectivity_and_gravity, get_studs_and_sockets_world
from src.generator import LegoGenerator
from src.graph_validator import LegoGraphValidator

def calculate_interlocking_score(parts: list) -> float:
    """
    Calculates a reward for interlocking/cross-bracing joints.
    An interlocking joint occurs when a brick connects to 2 or more separate 
    bricks on the layer directly below or above it.
    """
    if len(parts) < 3:
        return 0.0
        
    interlock_count = 0
    # Group parts by their vertical level (Y-coordinate rounded to nearest brick height: 24 LDU)
    # LDraw Y is inverted (negative is up, positive is down)
    levels = {}
    for idx, p in enumerate(parts):
        y_val = int(round(p.transform[1, 3] / 24.0)) * 24
        if y_val not in levels:
            levels[y_val] = []
        levels[y_val].append(idx)
        
    validator = LegoGraphValidator()
    
    # Check connections between adjacent layers
    for y_level, indices in levels.items():
        # Look at the layer below (which is y_level + 24 in LDraw coords)
        below_level = y_level + 24
        if below_level not in levels:
            continue
            
        below_indices = levels[below_level]
        for idx in indices:
            connections_to_below = 0
            part = parts[idx]
            for b_idx in below_indices:
                b_part = parts[b_idx]
                # Check connection using validator studs matching sockets
                studs_p, sockets_p = get_studs_and_sockets_world(part)
                studs_b, sockets_b = get_studs_and_sockets_world(b_part)
                
                # Check socket of upper part (part) -> stud of lower part (b_part)
                connected = False
                if len(sockets_p) > 0 and len(studs_b) > 0:
                    dists = np.linalg.norm(sockets_p[:, None, :] - studs_b[None, :, :], axis=2)
                    if np.any(dists < 5.0):
                        connected = True
                
                if connected:
                    connections_to_below += 1
            
            # If a part is supported by multiple distinct bricks below it, it's interlocking!
            if connections_to_below >= 2:
                interlock_count += 1
                
    return float(interlock_count)

def get_rl_reward(parts: list, mode: str = "general") -> float:
    """
    Computes a reward for a generated Lego structure.
    In 'vehicle' mode, delegates to get_vehicle_rl_reward.
    """
    if mode == "vehicle":
        from src.vehicle_rules import get_vehicle_rl_reward
        return get_vehicle_rl_reward(parts)
        
    if not parts:
        return -10.0
        
    # Check stability
    stable = check_connectivity_and_gravity(parts)
    
    if not stable:
        return -5.0
        
    reward = 10.0
    reward += 5.0 * calculate_interlocking_score(parts)
    return reward

def train_rl_step(model, optimizer, device="cpu", mode="general") -> float:
    """
    Performs a single step of REINFORCE policy gradient optimization.
    Returns:
        float: The calculated policy loss.
    """
    device = torch.device(device)
    model.train()
    
    # 1. Determine vocabulary and colors based on mode
    if mode == "vehicle":
        from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS
        vocab = VEHICLE_ALLOWED_PARTS
        colors = VEHICLE_ALLOWED_COLORS
    else:
        vocab = ALLOWED_PARTS
        colors = ALLOWED_COLORS
        
    # 2. Initialize generator with the current policy
    generator = LegoGenerator(model, vocab, colors, device=device)
    
    # 3. Rollout a generated structure using Beam Search (which uses our policy logits)
    parts = generator.generate_beam_search(target_num_pieces=4, beam_width=2, max_candidates=2)
    
    # 4. Calculate reward
    reward = get_rl_reward(parts, mode=mode)
    
    if len(parts) < 2:
        return 0.0
        
    # 5. Compute log-probabilities for all steps taken to compute policy gradients
    optimizer.zero_grad()
    loss = torch.tensor(0.0, dtype=torch.float32, device=device, requires_grad=True)
    
    # Re-evaluate the decisions taken during rollout to calculate gradients
    for i in range(1, len(parts)):
        state = parts[:i]
        target_part = parts[i]
        
        # Build PyG graph of the state
        graph_data = build_pyg_graph(state, vocab).to(device)
        batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=device)
        
        # Forward pass
        part_out, color_logits, _ = model(graph_data.x, graph_data.edge_index, batch)
        
        # Target indices
        try:
            part_target_idx = vocab.index(target_part.part_id)
        except ValueError:
            part_target_idx = 0
            
        try:
            color_target_idx = colors.index(target_part.color)
        except ValueError:
            color_target_idx = 0
        
        # Log probability of selections
        part_log_probs = F.log_softmax(part_out, dim=-1)
        color_log_probs = F.log_softmax(color_logits, dim=-1)
        
        log_prob = part_log_probs[0, part_target_idx] + color_log_probs[0, color_target_idx]
        
        # Policy gradient loss: -Reward * log_prob
        loss = loss - torch.tensor(reward, dtype=torch.float32, device=device) * log_prob
        
    # Optimize
    if loss.item() != 0.0:
        loss.backward()
        optimizer.step()
        
    return loss.item()
