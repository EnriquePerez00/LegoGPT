"""src/color_system.py - Phase 4: Vehicle color system with region-aware prediction."""
import json, math, logging, os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src.vehicle_model import VehicleLegoGNN, FINGERPRINT_DIM, CATEGORY_WHEEL, CATEGORY_CABIN
from src.vehicle_vocab import VEHICLE_ALLOWED_PARTS, VEHICLE_ALLOWED_COLORS

logger = logging.getLogger(__name__)

COLOR_BODY         = [1, 2, 4, 14, 19, 3, 5]
COLOR_NEUTRAL_TONES= [7, 8, 15, 71, 72, 0]
VEHICLE_COLOR_VOCAB= list(dict.fromkeys(COLOR_BODY + COLOR_NEUTRAL_TONES))
COLOR_NEUTRAL_TOKEN= 16
DEFAULT_COLOR_PRIORS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "vehicle_color_priors.json")

def _get_region(part_id: str) -> str:
    if part_id in CATEGORY_WHEEL: return "wheel"
    if part_id in CATEGORY_CABIN: return "cabin"
    return "body"


@dataclass
class ColorPriors:
    region_probs: dict = field(default_factory=dict)
    n_reference_models: int = 0

    @classmethod
    def from_reference_models(cls, refs: list) -> "ColorPriors":
        region_counts = {"body": defaultdict(int), "wheel": defaultdict(int), "cabin": defaultdict(int)}
        for ref in refs:
            for p in ref.parts:
                if p.color == COLOR_NEUTRAL_TOKEN:
                    continue
                region = _get_region(p.part_id)
                region_counts[region][p.color] += 1
        region_probs = {}
        for region, counts in region_counts.items():
            total = sum(counts.values())
            if total > 0:
                region_probs[region] = {k: v / total for k, v in counts.items()}
            else:
                if region == "wheel":
                    defaults = [0, 71, 72]
                elif region == "cabin":
                    defaults = [15, 40]
                else:
                    defaults = [1, 4, 14, 0, 72]
                region_probs[region] = {c: 1.0/len(defaults) for c in defaults}
        return cls(region_probs=region_probs, n_reference_models=len(refs))

    @classmethod
    def _empty(cls) -> "ColorPriors":
        return cls(
            region_probs={
                "body":  {1: 0.2, 4: 0.2, 14: 0.2, 0: 0.2, 72: 0.1, 15: 0.1},
                "wheel": {0: 0.5, 71: 0.35, 72: 0.15},
                "cabin": {15: 0.8, 40: 0.15, 41: 0.05},
            },
            n_reference_models=0,
        )

    def get_color_log_prior(self, region: str, color: int, smoothing: float = 1e-4) -> float:
        prob = self.region_probs.get(region, {}).get(color, smoothing)
        return math.log(max(prob, smoothing))

    def get_color_logit_bias_vector(self, region: str, color_vocab: list, smoothing: float = 1e-4) -> np.ndarray:
        return np.array([self.get_color_log_prior(region, c, smoothing) for c in color_vocab], dtype=np.float32)

    def save(self, path: Optional[str] = None) -> str:
        out = path or DEFAULT_COLOR_PRIORS_PATH
        os.makedirs(os.path.dirname(out), exist_ok=True)
        payload = {
            "n_reference_models": self.n_reference_models,
            "region_probs": {r: {str(k): v for k,v in d.items()} for r,d in self.region_probs.items()},
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("ColorPriors saved to %s", out)
        return out

    @classmethod
    def load(cls, path: Optional[str] = None) -> "ColorPriors":
        in_path = path or DEFAULT_COLOR_PRIORS_PATH
        if not os.path.isfile(in_path):
            logger.warning("ColorPriors not found at %s — using defaults", in_path)
            return cls._empty()
        with open(in_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            region_probs={r: {int(k): v for k,v in d.items()} for r,d in data.get("region_probs", {}).items()},
            n_reference_models=data.get("n_reference_models", 0),
        )

    def __repr__(self) -> str:
        return f"ColorPriors(n_refs={self.n_reference_models}, regions={list(self.region_probs.keys())})"


class ColorRegionHead(nn.Module):
    def __init__(self, hidden_dim=64, num_colors=None):
        super().__init__()
        if num_colors is None:
            num_colors = len(VEHICLE_COLOR_VOCAB)
        self.num_colors = num_colors
        self.color_vocab = VEHICLE_COLOR_VOCAB
        self.trunk      = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU())
        self.head_body  = nn.Linear(32, num_colors)
        self.head_wheel = nn.Linear(32, num_colors)
        self.head_cabin = nn.Linear(32, num_colors)

    def forward(self, h: torch.Tensor, region: str = "body") -> torch.Tensor:
        t = self.trunk(h)
        if region == "wheel":  return self.head_wheel(t)
        if region == "cabin":  return self.head_cabin(t)
        return self.head_body(t)

    def predict_color(self, h: torch.Tensor, part_id: str, color_priors=None) -> int:
        region = _get_region(part_id)
        logits = self.forward(h, region=region)[0].detach().cpu().numpy()
        if color_priors is not None:
            logits = logits + color_priors.get_color_logit_bias_vector(region, self.color_vocab)
        return self.color_vocab[int(np.argmax(logits))]

    def compute_color_loss(self, h: torch.Tensor, color_targets: torch.Tensor, part_ids: list) -> torch.Tensor:
        loss = torch.tensor(0.0, device=h.device)
        for region in ["body", "wheel", "cabin"]:
            mask = torch.tensor([_get_region(pid) == region for pid in part_ids])
            if not mask.any():
                continue
            logits = self.forward(h[mask], region=region)
            tgts = color_targets[mask]
            # Map LDraw color codes to vocab indices
            tgt_indices = torch.tensor(
                [self.color_vocab.index(c.item()) if c.item() in self.color_vocab else 0
                 for c in tgts],
                dtype=torch.long, device=h.device
            )
            loss = loss + F.cross_entropy(logits, tgt_indices)
        return loss


class VehicleLegoGNNWithColor(nn.Module):
    def __init__(self, hidden_dim=64, cond_dim=32, use_fingerprint=True):
        super().__init__()
        num_colors = len(VEHICLE_COLOR_VOCAB)
        self.backbone   = VehicleLegoGNN(num_color_classes=num_colors, hidden_dim=hidden_dim,
                                          cond_dim=cond_dim, use_fingerprint=use_fingerprint)
        self.color_head = ColorRegionHead(hidden_dim=hidden_dim, num_colors=num_colors)
        self.hidden_dim = hidden_dim

    def freeze_structural(self) -> None:
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.color_head.parameters(): p.requires_grad = True
        logger.info("Backbone frozen. ColorRegionHead is trainable.")

    def unfreeze_all(self) -> None:
        for p in self.parameters(): p.requires_grad = True
        logger.info("All parameters unfrozen.")

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_embedding(self, x, edge_index, batch, fingerprint=None) -> torch.Tensor:
        from torch_geometric.nn import global_mean_pool
        h = torch.relu(self.backbone.conv1(x, edge_index))
        h = torch.relu(self.backbone.conv2(h, edge_index))
        h = torch.relu(self.backbone.conv3(h, edge_index))
        g = global_mean_pool(h, batch)
        cond_dim = self.backbone.cond_dim
        if self.backbone.use_fingerprint:
            if fingerprint is not None:
                if fingerprint.size(0) != g.size(0):
                    fingerprint = fingerprint.expand(g.size(0), -1)
                cond = self.backbone.conditioner(fingerprint.to(g.device))
                g = torch.cat([g, cond], dim=-1)
            else:
                pad = torch.zeros(g.size(0), cond_dim, device=g.device)
                g = torch.cat([g, pad], dim=-1)
        return self.backbone.fc(g)

    def forward(self, x, edge_index, batch, fingerprint=None, part_ids=None):
        part_lp, _, transform = self.backbone(x, edge_index, batch, fingerprint=fingerprint)
        emb = self.get_embedding(x, edge_index, batch, fingerprint=fingerprint)
        region = "body"
        if part_ids and len(part_ids) == 1:
            region = _get_region(part_ids[0])
        color_logits = self.color_head(emb, region=region)
        return part_lp, color_logits, transform
