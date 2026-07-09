"""
src/design_priors.py

Statistical priors extracted from reference vehicle models.
Computes and serializes:
  - Marginal part frequencies P(part)
  - Connectivity bigrams P(part_j | part_i connected to part_i)
  - Count distributions (wheels, windscreens, axles, etc.)

These priors replace the hardcoded +2.0/-10/-20 biases in generator.py
with data-driven signals derived from real reference models.

Usage:
    from src.design_priors import DesignPriors
    priors = DesignPriors.from_reference_models(refs)
    priors.save("data/vehicle_priors.json")

    # Later, in beam search:
    priors = DesignPriors.load("data/vehicle_priors.json")
    bias = priors.get_part_log_prior("3020.dat")
"""

import json
import logging
import math
import os
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default path for persisted priors
DEFAULT_PRIORS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "vehicle_priors.json"
)

# Part categories for count distribution tracking
WHEEL_PARTS = {
    "3139.dat", "42610.dat", "56902.dat", "30027.dat", "18976.dat",
    "18977.dat", "55981.dat", "55982.dat", "30285.dat", "6014.dat",
    "6015.dat", "56890.dat", "56891.dat", "30285b.dat", "6014b.dat",
}
WINDSCREEN_PARTS = {"3823.dat", "3829.dat", "3829c01.dat", "4864.dat", "3822.dat", "98835.dat"}
AXLE_PARTS = {"2926.dat", "4150.dat", "4274.dat", "3641.dat", "6157.dat", "122c01.dat", "4032.dat"}
MUDGUARD_PARTS = {"50745.dat", "98282.dat", "30029.dat", "3788.dat"}
SLOPE_PARTS = {
    "3040.dat", "3039.dat", "3038.dat", "3298.dat", "3665.dat", "3660.dat",
    "4286.dat", "4287.dat", "50950.dat", "61678.dat", "85984.dat", "11477.dat",
    "6091.dat", "54200.dat", "85970.dat",
}

CATEGORY_MAP = {
    "wheel": WHEEL_PARTS,
    "windscreen": WINDSCREEN_PARTS,
    "axle": AXLE_PARTS,
    "mudguard": MUDGUARD_PARTS,
    "slope": SLOPE_PARTS,
}


class DesignPriors:
    """
    Statistical priors extracted from a corpus of reference vehicle models.

    Attributes:
        marginal:            {part_id: probability} — P(part) across all placements.
        connectivity_bigrams: {part_i: {part_j: probability}} — P(part_j | part_i adjacent).
        count_distribution:  {category: {count_str: probability}} — e.g. wheels: {"4": 0.85}.
        n_reference_models:  Number of reference models used to compute priors.
        vocab_size:          Number of unique parts seen in references.
    """

    def __init__(
        self,
        marginal: dict,
        connectivity_bigrams: dict,
        count_distribution: dict,
        n_reference_models: int = 0,
        vocab_size: int = 0,
    ) -> None:
        self.marginal = marginal
        self.connectivity_bigrams = connectivity_bigrams
        self.count_distribution = count_distribution
        self.n_reference_models = n_reference_models
        self.vocab_size = vocab_size

    # ------------------------------------------------------------------
    # Construction from reference models
    # ------------------------------------------------------------------

    @classmethod
    def from_reference_models(cls, refs: list) -> "DesignPriors":
        """
        Computes all priors from a list of RefModel instances.

        Args:
            refs: List of RefModel objects (from reference_loader.py).

        Returns:
            DesignPriors instance ready to use or persist.
        """
        if not refs:
            logger.warning("No reference models provided — returning empty priors.")
            return cls._empty()

        # --- Marginal frequencies ---
        part_counts: dict = defaultdict(int)
        total_placements = 0

        for ref in refs:
            for p in ref.parts:
                part_counts[p.part_id] += 1
                total_placements += 1

        marginal = {
            part_id: count / total_placements
            for part_id, count in part_counts.items()
        }

        # --- Connectivity bigrams ---
        # For each edge (i, j) in the reference graph, record co-occurrence
        bigram_counts: dict = defaultdict(lambda: defaultdict(int))
        bigram_totals: dict = defaultdict(int)

        for ref in refs:
            graph = ref.graph
            parts = ref.parts

            if graph is None or not hasattr(graph, "edge_index"):
                continue
            if graph.edge_index.numel() == 0:
                continue

            edges = graph.edge_index.t().tolist()  # list of [src, dst]
            for src_idx, dst_idx in edges:
                if src_idx >= len(parts) or dst_idx >= len(parts):
                    continue
                src_part = parts[src_idx].part_id
                dst_part = parts[dst_idx].part_id
                bigram_counts[src_part][dst_part] += 1
                bigram_totals[src_part] += 1

        connectivity_bigrams = {}
        for src_part, neighbors in bigram_counts.items():
            total = bigram_totals[src_part]
            connectivity_bigrams[src_part] = {
                dst: cnt / total for dst, cnt in neighbors.items()
            }

        # --- Count distributions per category ---
        count_distribution: dict = {}
        for category, category_parts in CATEGORY_MAP.items():
            cat_counts: dict = defaultdict(int)
            for ref in refs:
                n = sum(1 for p in ref.parts if p.part_id in category_parts)
                cat_counts[str(n)] += 1

            total_models = len(refs)
            count_distribution[category] = {
                k: v / total_models for k, v in cat_counts.items()
            }

        logger.info(
            "Priors computed from %d models: %d unique parts, %d bigrams",
            len(refs),
            len(marginal),
            sum(len(v) for v in connectivity_bigrams.values()),
        )

        return cls(
            marginal=marginal,
            connectivity_bigrams=connectivity_bigrams,
            count_distribution=count_distribution,
            n_reference_models=len(refs),
            vocab_size=len(marginal),
        )

    @classmethod
    def _empty(cls) -> "DesignPriors":
        """Returns an empty priors object (safe fallback when no refs available)."""
        return cls(
            marginal={},
            connectivity_bigrams={},
            count_distribution={cat: {} for cat in CATEGORY_MAP},
            n_reference_models=0,
            vocab_size=0,
        )

    # ------------------------------------------------------------------
    # Query interface (used by generator.py beam search)
    # ------------------------------------------------------------------

    def get_part_log_prior(self, part_id: str, smoothing: float = 1e-6) -> float:
        """
        Returns log P(part) for use as an additive bias in beam search logits.
        Replaces the hardcoded +2.0 / -10.0 / -20.0 in generator.py.

        A part with high marginal frequency gets a positive log prior (> log(smoothing)).
        An unknown part gets a small negative log prior (log(smoothing)).

        Args:
            part_id:   LDraw part identifier (e.g. '3020.dat').
            smoothing: Floor probability to avoid log(0).

        Returns:
            float: log prior (additive logit bias).
        """
        prob = self.marginal.get(part_id, smoothing)
        return math.log(max(prob, smoothing))

    def get_connectivity_log_prior(
        self, context_part_id: str, candidate_part_id: str, smoothing: float = 1e-4
    ) -> float:
        """
        Returns log P(candidate | context) — connectivity bigram prior.
        Useful for biasing which part to place on top of an existing part.

        Args:
            context_part_id:   The already-placed part providing context.
            candidate_part_id: The candidate part being evaluated.
            smoothing:         Floor probability.

        Returns:
            float: log conditional prior (additive logit bias).
        """
        neighbors = self.connectivity_bigrams.get(context_part_id, {})
        prob = neighbors.get(candidate_part_id, smoothing)
        return math.log(max(prob, smoothing))

    def get_expected_count(self, category: str) -> Optional[int]:
        """
        Returns the mode (most likely count) for a given part category.
        Useful for reward shaping and inventory targeting.

        Args:
            category: One of 'wheel', 'windscreen', 'axle', 'mudguard', 'slope'.

        Returns:
            int or None: The most frequently observed count, or None if unknown.
        """
        dist = self.count_distribution.get(category, {})
        if not dist:
            return None
        return int(max(dist, key=lambda k: dist[k]))

    def get_logit_bias_vector(self, vocab: list, smoothing: float = 1e-6) -> np.ndarray:
        """
        Returns a numpy vector of log-prior biases for every part in the vocab.
        Can be added directly to GNN logits in beam search.

        Args:
            vocab:     List of part_id strings (VEHICLE_ALLOWED_PARTS order).
            smoothing: Floor probability for unknown parts.

        Returns:
            np.ndarray of shape [len(vocab)] with log prior values.
        """
        return np.array(
            [self.get_part_log_prior(p, smoothing=smoothing) for p in vocab],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Serializes priors to JSON.

        Args:
            path: Output file path. Defaults to DEFAULT_PRIORS_PATH.

        Returns:
            str: The path where the file was saved.
        """
        out_path = path or DEFAULT_PRIORS_PATH
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        payload = {
            "n_reference_models": self.n_reference_models,
            "vocab_size": self.vocab_size,
            "marginal": self.marginal,
            "connectivity_bigrams": {
                k: dict(v) for k, v in self.connectivity_bigrams.items()
            },
            "count_distribution": self.count_distribution,
        }

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

        logger.info("Priors saved to '%s' (%d parts)", out_path, self.vocab_size)
        return out_path

    @classmethod
    def load(cls, path: Optional[str] = None) -> "DesignPriors":
        """
        Loads priors from a JSON file.

        Args:
            path: JSON file path. Defaults to DEFAULT_PRIORS_PATH.

        Returns:
            DesignPriors instance. Returns empty priors if file not found.
        """
        in_path = path or DEFAULT_PRIORS_PATH

        if not os.path.isfile(in_path):
            logger.warning("Priors file not found at '%s' — using empty priors.", in_path)
            return cls._empty()

        with open(in_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        instance = cls(
            marginal=data.get("marginal", {}),
            connectivity_bigrams=data.get("connectivity_bigrams", {}),
            count_distribution=data.get("count_distribution", {}),
            n_reference_models=data.get("n_reference_models", 0),
            vocab_size=data.get("vocab_size", 0),
        )
        logger.info(
            "Priors loaded from '%s': %d models, %d parts",
            in_path,
            instance.n_reference_models,
            instance.vocab_size,
        )
        return instance

    def __repr__(self) -> str:
        return (
            f"DesignPriors(n_refs={self.n_reference_models}, "
            f"vocab={self.vocab_size}, "
            f"bigrams={sum(len(v) for v in self.connectivity_bigrams.values())})"
        )
