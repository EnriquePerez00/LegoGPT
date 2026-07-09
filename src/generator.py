"""
src/generator.py

LegoGenerator: constrained Beam Search decoding over the GNN policy.

Phase 0/1 changes:
  - Added `design_priors` parameter (optional DesignPriors instance).
  - Replaced hardcoded +2.0/-10.0/-20.0 cond_bias with data-driven log-prior bias:
      * Marginal prior bias: log P(part) from reference corpus
      * Connectivity context bias: log P(part | last_placed_part) from bigrams
      * Inventory compliance bias: scaled by target_inventory if provided
  - Color collapsed to COLOR_NEUTRAL (16) when priors are active (Phase 0).
  - Graceful degradation: if no priors provided, falls back to original behaviour.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional

from src.parser import ParsedPart, build_pyg_graph
from src.validator import get_studs_and_sockets_world
from src.graph_validator import LegoGraphValidator
from src.model import HierarchicalSoftmax

# Phase 0: neutral color constant (same as reference_loader.COLOR_NEUTRAL)
COLOR_NEUTRAL: int = 16

# Bias scale factors — tuned for log-probability space
_MARGINAL_SCALE: float = 1.0      # weight for marginal log P(part)
_BIGRAM_SCALE: float = 0.5        # weight for connectivity bigram log P(part_j|part_i)
_INVENTORY_BONUS: float = 3.0     # bonus for parts still needed (target_inventory)
_INVENTORY_OVERFLOW: float = -8.0 # penalty for exceeding target count
_INVENTORY_UNKNOWN: float = -15.0 # penalty for parts not in target_inventory


class LegoGenerator:
    def __init__(
        self,
        model,
        allowed_parts: list,
        allowed_colors: list,
        device: str = "cpu",
        design_priors=None,
    ):
        """
        Initializes the generator with a trained model and vocabulary.

        Args:
            model:          Trained GNN model (LegoGNN or LegoGraphTransformer).
            allowed_parts:  List of allowed part_id strings.
            allowed_colors: List of allowed color codes (ints).
            device:         Torch device string ('cpu', 'mps').
            design_priors:  Optional DesignPriors instance from src.design_priors.
                            If provided, replaces hardcoded cond_bias with data-driven
                            log-prior signals. If None, legacy behaviour is preserved.
        """
        self.model = model
        self.allowed_parts = allowed_parts
        self.allowed_colors = allowed_colors
        self.device = torch.device(device)
        self.validator = LegoGraphValidator()
        self.design_priors = design_priors

        # Pre-compute marginal bias vector for the full vocab (once, at init time)
        # Shape: [len(allowed_parts)] — additive offset to GNN logits
        if self.design_priors is not None and self.design_priors.n_reference_models > 0:
            self._marginal_bias = self.design_priors.get_logit_bias_vector(
                self.allowed_parts
            )
        else:
            self._marginal_bias = None

    # ------------------------------------------------------------------
    # Internal: compute data-driven conditioning bias for one candidate
    # ------------------------------------------------------------------

    def _compute_cond_bias(
        self,
        cand_part_id: str,
        parts_list: list,
        target_inventory: Optional[dict],
    ) -> float:
        """
        Computes the total conditioning bias for placing cand_part_id given the
        current beam state and optional target inventory.

        Returns:
            float: additive log-space bias (replaces old +2.0/-10.0/-20.0).
        """
        bias = 0.0

        # --- Connectivity bigram bias (context = last placed part) ---
        if (
            self.design_priors is not None
            and self.design_priors.n_reference_models > 0
            and parts_list
        ):
            last_part_id = parts_list[-1].part_id
            bias += _BIGRAM_SCALE * self.design_priors.get_connectivity_log_prior(
                last_part_id, cand_part_id
            )

        # --- Inventory compliance bias ---
        if target_inventory is not None:
            current_count = sum(1 for p in parts_list if p.part_id == cand_part_id)
            tgt_count = target_inventory.get(cand_part_id, 0)

            if self.design_priors is not None and self.design_priors.n_reference_models > 0:
                # Data-driven mode: scale bonus/penalty relative to log prior
                if tgt_count > 0:
                    if current_count < tgt_count:
                        # Still need this part — bonus proportional to gap
                        remaining_fraction = (tgt_count - current_count) / tgt_count
                        bias += _INVENTORY_BONUS * remaining_fraction
                    else:
                        bias += _INVENTORY_OVERFLOW
                else:
                    bias += _INVENTORY_UNKNOWN
            else:
                # Legacy hardcoded mode (no priors available)
                if tgt_count > 0:
                    if current_count < tgt_count:
                        bias += 2.0
                    else:
                        bias += -10.0
                else:
                    bias += -20.0

        return bias

    # ------------------------------------------------------------------
    # Main beam search
    # ------------------------------------------------------------------

    def generate_beam_search(
        self,
        target_num_pieces: int = 8,
        beam_width: int = 3,
        max_candidates: int = 5,
        target_inventory: Optional[dict] = None,
    ) -> list:
        """
        Generates a valid LEGO structure using constrained Beam Search decoding,
        biased by data-driven design priors and optional target inventory.

        Args:
            target_num_pieces: Total number of bricks to generate.
            beam_width:        Number of active beams at each step.
            max_candidates:    Number of top predicted parts/colors to consider.
            target_inventory:  Optional {part_id: count} target. Parts are encouraged
                               or penalised based on DesignPriors + inventory compliance.

        Returns:
            list[ParsedPart]: Stable, collision-free LEGO assembly.
        """
        # --- Determine active color for Phase 0 ---
        # If priors are loaded (structural training mode), collapse color to neutral.
        use_neutral_color = (
            self.design_priors is not None
            and self.design_priors.n_reference_models > 0
        )

        # 1. Initialise beams with a seeded base part
        beams = []
        for _ in range(beam_width):
            if target_inventory is not None and len(target_inventory) > 0:
                base_part_id = np.random.choice(list(target_inventory.keys()))
            else:
                base_part_id = np.random.choice(self.allowed_parts)

            base_color = COLOR_NEUTRAL if use_neutral_color else int(np.random.choice(self.allowed_colors))
            base_transform = np.eye(4, dtype=np.float32)
            base_part = ParsedPart(
                part_id=base_part_id,
                color=base_color,
                transform=base_transform,
                step_id=0,
            )
            beams.append(([base_part], 0.0))

        # 2. Autoregressive placement loop
        for step in range(1, target_num_pieces):
            new_candidate_beams = []

            for parts_list, score in beams:
                # Build PyG graph of current beam state
                graph_data = build_pyg_graph(
                    parts_list, self.allowed_parts, allowed_colors=self.allowed_colors
                ).to(self.device)
                batch = torch.zeros(
                    graph_data.num_nodes, dtype=torch.long, device=self.device
                )

                with torch.no_grad():
                    part_out, color_logits, _ = self.model(
                        graph_data.x, graph_data.edge_index, batch
                    )

                # Get GNN log-probabilities
                if hasattr(self.model, "head_part") and isinstance(
                    self.model.head_part, HierarchicalSoftmax
                ):
                    part_log_probs = part_out[0].cpu().numpy()
                else:
                    part_log_probs = F.log_softmax(part_out[0], dim=-1).cpu().numpy()

                color_log_probs = F.log_softmax(color_logits[0], dim=-1).cpu().numpy()

                # Inject marginal prior bias into GNN logits (additive in log space)
                if self._marginal_bias is not None:
                    part_log_probs = part_log_probs + _MARGINAL_SCALE * self._marginal_bias

                # Select top candidate parts (post-prior-injection ranking)
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

                # Phase 0: use only neutral color; otherwise top-k from model
                if use_neutral_color:
                    top_color_indices = [
                        self.allowed_colors.index(COLOR_NEUTRAL)
                        if COLOR_NEUTRAL in self.allowed_colors
                        else 0
                    ]
                else:
                    top_color_indices = list(
                        np.argsort(color_log_probs)[::-1][:max_candidates]
                    )

                # Evaluate placing top candidates
                for part_idx in top_part_indices:
                    cand_part_id = self.allowed_parts[part_idx]
                    part_score = part_log_probs[part_idx]

                    for color_idx in top_color_indices:
                        cand_color = (
                            COLOR_NEUTRAL
                            if use_neutral_color
                            else int(self.allowed_colors[color_idx])
                        )
                        color_score = (
                            0.0
                            if use_neutral_color
                            else color_log_probs[color_idx]
                        )

                        # Check part has studs/sockets (can connect)
                        dummy_part = ParsedPart(
                            part_id=cand_part_id,
                            color=cand_color,
                            transform=np.eye(4, dtype=np.float32),
                            step_id=step,
                        )
                        cand_studs, cand_sockets = get_studs_and_sockets_world(dummy_part)
                        if len(cand_studs) == 0 and len(cand_sockets) == 0:
                            continue

                        # Try anchoring to existing parts
                        for parent in parts_list:
                            parent_studs, parent_sockets = get_studs_and_sockets_world(parent)
                            if len(parent_studs) == 0 and len(parent_sockets) == 0:
                                continue

                            for conn_mode in [0, 1]:
                                if conn_mode == 0 and len(parent_studs) > 0 and len(cand_sockets) > 0:
                                    p_anchors = parent_studs
                                    c_anchors = cand_sockets
                                elif conn_mode == 1 and len(parent_sockets) > 0 and len(cand_studs) > 0:
                                    p_anchors = parent_sockets
                                    c_anchors = cand_studs
                                else:
                                    continue

                                for p_anchor in p_anchors[:2]:
                                    for c_anchor in c_anchors[:2]:
                                        for theta in [0, 90, 180, 270]:
                                            rad = np.radians(theta)
                                            c_val = int(round(np.cos(rad)))
                                            s_val = int(round(np.sin(rad)))
                                            R = np.array(
                                                [
                                                    [c_val, 0, s_val],
                                                    [0, 1, 0],
                                                    [-s_val, 0, c_val],
                                                ],
                                                dtype=np.float32,
                                            )

                                            T = p_anchor - R @ c_anchor

                                            t_matrix = np.eye(4, dtype=np.float32)
                                            t_matrix[:3, :3] = R
                                            t_matrix[:3, 3] = T

                                            candidate = ParsedPart(
                                                part_id=cand_part_id,
                                                color=cand_color,
                                                transform=t_matrix,
                                                step_id=step,
                                            )

                                            if self.validator.can_place_brick(
                                                candidate, current_state=parts_list
                                            ):
                                                # Data-driven conditioning bias (Fase 1)
                                                cond_bias = self._compute_cond_bias(
                                                    cand_part_id, parts_list, target_inventory
                                                )

                                                new_score = score + part_score + color_score + cond_bias
                                                new_parts = parts_list + [candidate]
                                                new_candidate_beams.append((new_parts, new_score))

            # Prune: keep top beam_width unique beams
            if new_candidate_beams:
                unique_beams = []
                seen_states = set()
                for b_parts, b_score in sorted(
                    new_candidate_beams, key=lambda x: x[1], reverse=True
                ):
                    signature = tuple(
                        (p.part_id, p.color, tuple(p.transform.flatten().round(2)))
                        for p in b_parts
                    )
                    if signature not in seen_states:
                        seen_states.add(signature)
                        unique_beams.append((b_parts, b_score))
                beams = unique_beams[:beam_width]
            else:
                break

        # Return best assembly
        if not beams:
            return []
        beams.sort(key=lambda x: x[1], reverse=True)
        return beams[0][0]
