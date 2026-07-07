import torch
import torch.nn.functional as F
import numpy as np
from src.parser import ParsedPart, build_pyg_graph
from src.validator import get_studs_and_sockets_world
from src.graph_validator import LegoGraphValidator
from src.model import HierarchicalSoftmax

class LegoGenerator:
    def __init__(self, model, allowed_parts, allowed_colors, device="cpu"):
        """
        Initializes the generator with a trained model and allowed parts/colors vocabularies.
        """
        self.model = model
        self.allowed_parts = allowed_parts
        self.allowed_colors = allowed_colors
        self.device = torch.device(device)
        self.validator = LegoGraphValidator()

    def generate_beam_search(self, target_num_pieces=8, beam_width=3, max_candidates=5, target_inventory=None) -> list[ParsedPart]:
        """
        Generates a valid LEGO structure using constrained Beam Search decoding, 
        optionally biased toward a target parts inventory.
        
        Args:
            target_num_pieces: The total number of bricks to generate.
            beam_width: The number of active beams to maintain at each step.
            max_candidates: The number of top predicted parts/colors to consider.
            target_inventory: Optional dict of {part_id: count} specifying target part requirements.
            
        Returns:
            list[ParsedPart]: The generated stable, collision-free LEGO assembly.
        """
        # 1. Initialize beams. We start by placing a base brick at the origin.
        # We seed with parts from the target inventory if provided to avoid immediate penalties.
        beams = []
        for _ in range(beam_width):
            if target_inventory is not None and len(target_inventory) > 0:
                base_part_id = np.random.choice(list(target_inventory.keys()))
            else:
                base_part_id = np.random.choice(self.allowed_parts)
            base_color = int(np.random.choice(self.allowed_colors))
            base_transform = np.eye(4, dtype=np.float32)
            base_part = ParsedPart(part_id=base_part_id, color=base_color, transform=base_transform, step_id=0)
            beams.append(([base_part], 0.0))

        # 2. Autoregressive placement loop
        for step in range(1, target_num_pieces):
            new_candidate_beams = []
            
            for parts_list, score in beams:
                # Build PyG graph for the current state in the beam
                graph_data = build_pyg_graph(parts_list, self.allowed_parts).to(self.device)
                batch = torch.zeros(graph_data.num_nodes, dtype=torch.long, device=self.device)
                
                with torch.no_grad():
                    part_out, color_logits, _ = self.model(graph_data.x, graph_data.edge_index, batch)
                
                # Get log probabilities
                if hasattr(self.model, "head_part") and isinstance(self.model.head_part, HierarchicalSoftmax):
                    part_log_probs = part_out[0].cpu().numpy()
                else:
                    part_log_probs = F.log_softmax(part_out[0], dim=-1).cpu().numpy()
                    
                color_log_probs = F.log_softmax(color_logits[0], dim=-1).cpu().numpy()
                
                # Retrieve the top predicted parts and colors
                top_part_indices = list(np.argsort(part_log_probs)[::-1][:max_candidates])
                
                # Always include target inventory parts in candidates
                if target_inventory is not None:
                    for tgt_part in target_inventory.keys():
                        try:
                            tgt_idx = self.allowed_parts.index(tgt_part)
                            if tgt_idx not in top_part_indices:
                                top_part_indices.append(tgt_idx)
                        except ValueError:
                            pass
                            
                top_color_indices = np.argsort(color_log_probs)[::-1][:max_candidates]
                
                # Evaluate placing the top candidates
                for part_idx in top_part_indices:
                    cand_part_id = self.allowed_parts[part_idx]
                    part_score = part_log_probs[part_idx]
                    
                    for color_idx in top_color_indices:
                        cand_color = int(color_idx)
                        color_score = color_log_probs[color_idx]
                        
                        # Instantiate a dummy part to extract its local studs and sockets
                        dummy_part = ParsedPart(part_id=cand_part_id, color=cand_color, transform=np.eye(4, dtype=np.float32), step_id=step)
                        cand_studs, cand_sockets = get_studs_and_sockets_world(dummy_part)
                        if len(cand_studs) == 0 and len(cand_sockets) == 0:
                            continue
                            
                        # Try to find a valid placement by anchoring to existing parts in the beam
                        for parent in parts_list:
                            parent_studs, parent_sockets = get_studs_and_sockets_world(parent)
                            if len(parent_studs) == 0 and len(parent_sockets) == 0:
                                continue
                                
                            # Connect new part's socket to parent's stud (0) or new part's stud to parent's socket (1)
                            for conn_mode in [0, 1]:
                                if conn_mode == 0 and len(parent_studs) > 0 and len(cand_sockets) > 0:
                                    p_anchors = parent_studs
                                    c_anchors = cand_sockets
                                elif conn_mode == 1 and len(parent_sockets) > 0 and len(cand_studs) > 0:
                                    p_anchors = parent_sockets
                                    c_anchors = cand_studs
                                else:
                                    continue
                                    
                                # Test connection alignments
                                for p_anchor in p_anchors[:2]:  # Test top 2 anchors to maintain speed
                                    for c_anchor in c_anchors[:2]:
                                        for theta in [0, 90, 180, 270]:
                                            rad = np.radians(theta)
                                            c, s = int(round(np.cos(rad))), int(round(np.sin(rad)))
                                            R = np.array([
                                                [c, 0, s],
                                                [0, 1, 0],
                                                [-s, 0, c]
                                            ], dtype=np.float32)
                                            
                                            T = p_anchor - R @ c_anchor
                                            
                                            t_matrix = np.eye(4, dtype=np.float32)
                                            t_matrix[:3, :3] = R
                                            t_matrix[:3, 3] = T
                                            
                                            candidate = ParsedPart(part_id=cand_part_id, color=cand_color, transform=t_matrix, step_id=step)
                                            
                                            # Validate using LegoGraphValidator (Phase 1)
                                            if self.validator.can_place_brick(candidate, current_state=parts_list):
                                                # Calculate instruction conditioning bias (Phase 5)
                                                cond_bias = 0.0
                                                if target_inventory is not None:
                                                    current_count = sum(1 for p in parts_list if p.part_id == cand_part_id)
                                                    tgt_count = target_inventory.get(cand_part_id, 0)
                                                    if tgt_count > 0:
                                                        if current_count < tgt_count:
                                                            cond_bias = 2.0  # Encourage desired parts
                                                        else:
                                                            cond_bias = -10.0  # Penalize overflow
                                                    else:
                                                        cond_bias = -20.0  # Penalize undesired parts
                                                        
                                                new_score = score + part_score + color_score + cond_bias
                                                new_parts = parts_list + [candidate]
                                                new_candidate_beams.append((new_parts, new_score))
                                                
            # Prune and keep top beam_width beams
            if new_candidate_beams:
                unique_beams = []
                seen_states = set()
                for b_parts, b_score in sorted(new_candidate_beams, key=lambda x: x[1], reverse=True):
                    signature = tuple((p.part_id, p.color, tuple(p.transform.flatten().round(2))) for p in b_parts)
                    if signature not in seen_states:
                        seen_states.add(signature)
                        unique_beams.append((b_parts, b_score))
                beams = unique_beams[:beam_width]
            else:
                break
                
        # Return the best overall assembly
        if not beams:
            return []
        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0][0]
