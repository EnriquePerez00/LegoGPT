import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from generative.llm_pipeline.tokenizer_ldr import LdrTokenizer
from generative.llm_pipeline.brick_graph_tokenizer import BrickGraphTokenizer, _yaw_matrix_from_deg
from src.parser import ParsedPart, ALLOWED_PARTS, ALLOWED_COLORS, build_pyg_graph
from src.model import SAGEConv, global_mean_pool, get_device
from src.connectivity_db import get_part_ports, get_part_ports_world
from src.graph_validator import LegoGraphValidator

class BrickLLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, num_heads: int = 4, hidden_dim: int = 128, num_layers: int = 2):
        """
        A lightweight autoregressive Transformer Decoder (BrickLLM) 
        trained to generate LDraw or symbolic BrickGraph syntax.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encodings
        self.pos_emb = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor, graph_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [batch, seq_len]
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_emb[:, :seq_len, :]
        if graph_context is not None:
            # Fuse 3D graph context vector [batch, embed_dim] into sequence embeddings
            emb = emb + graph_context.unsqueeze(1)
        
        # Causal/autoregressive mask to prevent looking ahead
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Transformer forward pass (in PyTorch, TransformerEncoder can use is_causal mask)
        h = self.transformer(emb, mask=mask, is_causal=True)
        logits = self.head(h)
        return logits


class HybridGraphBrickLLM(nn.Module):
    """
    Hybrid 3D Graph-Encoder + Symbolic Sequence Transformer-Decoder (MPS-accelerated).
    Encodes the current 3D assembly state G_{t-1} via GraphSAGE message passing over
    physical port connections, and conditions the symbolic BrickNet token generator
    on the structural 3D embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        num_part_classes: int = len(ALLOWED_PARTS),
        num_color_classes: int = len(ALLOWED_COLORS),
        embed_dim: int = 64,
        num_heads: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        in_channels = num_part_classes + num_color_classes + 12
        self.gnn1 = SAGEConv(in_channels, embed_dim)
        self.gnn2 = SAGEConv(embed_dim, embed_dim)
        self.graph_proj = nn.Linear(embed_dim, embed_dim)
        self.llm = BrickLLM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def encode_graph(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x is None or x.numel() == 0:
            dev = next(self.parameters()).device
            return torch.zeros((1, self.graph_proj.out_features), dtype=torch.float32, device=dev)
        h = F.relu(self.gnn1(x, edge_index))
        h = F.relu(self.gnn2(h, edge_index))
        g = global_mean_pool(h, batch)
        return self.graph_proj(g)

    def forward(
        self,
        token_ids: torch.Tensor,
        graph_x: Optional[torch.Tensor] = None,
        graph_edge_index: Optional[torch.Tensor] = None,
        graph_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        g_ctx = self.encode_graph(graph_x, graph_edge_index, graph_batch) if graph_x is not None else None
        return self.llm(token_ids, graph_context=g_ctx)


def generate_constrained_brick_graph(
    model: nn.Module,
    tokenizer: BrickGraphTokenizer,
    target_num_pieces: int = 6,
    allowed_parts: Optional[List[str]] = None,
    allowed_colors: Optional[List[int]] = None,
    device: Optional[str] = None,
    symmetric_pairs: bool = False,
) -> Tuple[str, List[ParsedPart]]:
    """
    Generates a 100% physically valid LEGO assembly using BrickGraphTokenizer symbolic DSL
    and Constrained Physical Masking:
      - Queries `model` on MPS/CPU for next part/color/rotation logits conditioned on current graph.
      - Restricts port couplings to unoccupied parent studs and child bottom sockets that pass:
        1. Stud-to-socket alignment (`get_part_ports_world`)
        2. Zero OBB collision (`LegoGraphValidator.can_place_brick`)
        3. Ground level rules (Y <= 0.0, at most 4 ground parts |Y| <= 1.0)
      - Returns (symbolic_dsl_text, compiled_parts_list).
    """
    dev_str = device or get_device()
    dev = torch.device(dev_str)
    model = model.to(dev)
    model.eval()

    parts_vocab = allowed_parts or ALLOWED_PARTS
    colors_vocab = allowed_colors or [4, 14, 1, 15, 7, 0]
    validator = LegoGraphValidator()

    dsl_lines: List[str] = []
    placed_parts: List[ParsedPart] = []

    # Step 0: Place grounded foundation brick at origin
    base_pid = parts_vocab[0]
    base_col = int(colors_vocab[0])
    base_tf = np.eye(4, dtype=np.float32)
    base_part = ParsedPart(part_id=base_pid, color=base_col, transform=base_tf, step_id=0)
    placed_parts.append(base_part)
    dsl_lines.append(f"[STEP_0] ADD part_{base_pid} color_{base_col} AS node_0 AT origin ROT 0")

    occupied_parent_ports: set[Tuple[int, str]] = set()
    step_id = 1

    while len(placed_parts) < target_num_pieces:
        node_idx = len(placed_parts)
        current_dsl = "\n".join(dsl_lines)
        input_ids = tokenizer.encode(current_dsl)[:-1]  # strip [EOS]
        x_tok = torch.tensor([input_ids], dtype=torch.long, device=dev)

        with torch.no_grad():
            if isinstance(model, HybridGraphBrickLLM):
                g_data = build_pyg_graph(placed_parts, ALLOWED_PARTS, ALLOWED_COLORS).to(dev)
                g_batch = torch.zeros(g_data.num_nodes, dtype=torch.long, device=dev)
                logits = model(x_tok, g_data.x, g_data.edge_index, g_batch)
            else:
                logits = model(x_tok)
            last_logits = logits[0, -1].cpu().numpy()

        # Rank candidate parts and colors using model token logits
        scored_parts = []
        for pid in parts_vocab:
            tok_name = f"part_{pid.lower()}"
            tid = tokenizer.w2i.get(tok_name, 0)
            scored_parts.append((float(last_logits[tid]) if tid < len(last_logits) else 0.0, pid))
        scored_parts.sort(key=lambda x: x[0], reverse=True)

        scored_colors = []
        for cid in colors_vocab:
            tok_name = f"color_{cid}"
            tid = tokenizer.w2i.get(tok_name, 0)
            scored_colors.append((float(last_logits[tid]) if tid < len(last_logits) else 0.0, int(cid)))
        scored_colors.sort(key=lambda x: x[0], reverse=True)

        chosen_color = scored_colors[0][1]
        placed_this_iter = False

        # Constrained search over open parent studs and child sockets
        for _, cand_pid in scored_parts[:6]:
            child_sockets = [p for p in get_part_ports(cand_pid) if p.gender == "female"]
            if not child_sockets:
                continue

            for p_idx in range(len(placed_parts) - 1, -1, -1):
                parent = placed_parts[p_idx]
                parent_studs = [
                    p for p in get_part_ports_world(parent)
                    if p["gender"] == "male" and (p_idx, p["id"]) not in occupied_parent_ports
                ]
                for p_stud in parent_studs:
                    for c_sock in child_sockets:
                        for rot_deg in (0, 90, 180, 270):
                            R_c = _yaw_matrix_from_deg(rot_deg)
                            t_c = p_stud["world_pos"] - (R_c @ c_sock.local_pos)

                            # Enforce AGENTS.md ground & height rules
                            if float(t_c[1]) > 1.0:
                                continue
                            if abs(float(t_c[1])) <= 1.0:
                                ground_cnt = sum(1 for p in placed_parts if abs(float(p.transform[1, 3])) <= 1.0)
                                if ground_cnt >= 4:
                                    continue

                            tf_c = np.eye(4, dtype=np.float32)
                            tf_c[:3, :3] = R_c
                            tf_c[:3, 3] = t_c
                            cand_part = ParsedPart(part_id=cand_pid, color=chosen_color, transform=tf_c, step_id=step_id)

                            if validator.can_place_brick(cand_part, current_state=placed_parts):
                                placed_parts.append(cand_part)
                                occupied_parent_ports.add((p_idx, p_stud["id"]))
                                dsl_lines.append(
                                    f"[STEP_{step_id}] ADD part_{cand_pid} color_{chosen_color} AS node_{node_idx} "
                                    f"CONNECT node_{node_idx}.{c_sock.id} TO node_{p_idx}.{p_stud['id']} ROT {rot_deg}"
                                )
                                placed_this_iter = True

                                # Optional bilateral mirror placement if requested and space allows
                                if (
                                    symmetric_pairs
                                    and len(placed_parts) < target_num_pieces
                                    and abs(float(t_c[0])) >= 10.0
                                ):
                                    mirror_yaw = (360 - rot_deg) % 360
                                    tf_m = np.eye(4, dtype=np.float32)
                                    tf_m[:3, :3] = _yaw_matrix_from_deg(mirror_yaw)
                                    tf_m[:3, 3] = [-float(t_c[0]), float(t_c[1]), float(t_c[2])]
                                    mirror_part = ParsedPart(
                                        part_id=cand_pid, color=chosen_color, transform=tf_m, step_id=step_id
                                    )
                                    if validator.can_place_brick(mirror_part, current_state=placed_parts):
                                        m_idx = len(placed_parts)
                                        placed_parts.append(mirror_part)
                                        dsl_lines.append(
                                            f"[STEP_{step_id}] ADD part_{cand_pid} color_{chosen_color} "
                                            f"AS node_{m_idx} SNAP_MIRROR_X OF node_{node_idx}"
                                        )
                                step_id += 1
                                break
                        if placed_this_iter:
                            break
                    if placed_this_iter:
                        break
                if placed_this_iter:
                    break
            if placed_this_iter:
                break

        if not placed_this_iter:
            break

    return "\n".join(dsl_lines), placed_parts

def train_llm_step(model, optimizer, token_ids: list[int], device="cpu") -> float:
    """
    Performs a single autoregressive training step over a sequence of token IDs.
    """
    device = torch.device(device)
    model.train()
    
    if len(token_ids) < 2:
        return 0.0
        
    x = torch.tensor([token_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([token_ids[1:]], dtype=torch.long, device=device)
    
    optimizer.zero_grad()
    logits = model(x) # [batch, seq_len, vocab_size]
    
    # Calculate cross entropy loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    loss.backward()
    optimizer.step()
    
    return loss.item()
