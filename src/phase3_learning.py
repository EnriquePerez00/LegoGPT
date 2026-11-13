"""
FASE 3: Advanced Learning - Temporal beam, parameter search, LLM shadow eval.
"""

from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import numpy as np


class TemporalBeamSearchScheduler:
    """FASE 3: Implements cross-step beam search for backtracking on cascading errors."""
    
    def __init__(self, beam_width: int = 5, lookback_window: int = 3):
        self.beam_width = beam_width
        self.lookback_window = lookback_window
        self.step_states: List[Tuple[List[Any], float]] = []  # (nodes, probability)
    
    def track_step_state(self, nodes: List[Any], cumulative_probability: float):
        """Track state after each step for potential backtrack."""
        self.step_states.append((nodes, cumulative_probability))
    
    def can_backtrack(self, current_step_idx: int) -> bool:
        """Check if we have history to backtrack."""
        return len(self.step_states) > self.lookback_window
    
    def get_backtrack_candidates(self, current_step_idx: int) -> List[Tuple[int, float]]:
        """Returns (step_index, probability) pairs for backtracking."""
        if not self.can_backtrack(current_step_idx):
            return []
        
        start_idx = max(0, current_step_idx - self.lookback_window)
        candidates = []
        for i in range(start_idx, current_step_idx):
            _, prob = self.step_states[i]
            candidates.append((i, prob))
        
        # Sort by probability, take beam_width best
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.beam_width]


class ParameterSearchAgent:
    """FASE 3: Explores parameter space (w_edge_score, roi_margin, etc.)."""
    
    def __init__(self, param_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        self.param_ranges = param_ranges or {
            "w_edge_score": (0.4, 0.8),
            "w_stud_score": (0.2, 0.6),
            "roi_margin_ratio": (0.2, 0.5),
            "penetration_threshold_ldu": (1.0, 2.5),
        }
        self.trial_history: List[Dict[str, Any]] = []
    
    def propose_perturbation(self, current_params: Dict[str, float], iteration: int) -> Dict[str, float]:
        """FASE 3: Proposes parameter changes (random + simulated annealing)."""
        proposed = dict(current_params)
        
        # Temperature decreases over time (more aggressive early, conservative later)
        temperature = max(0.01, 1.0 / (1.0 + iteration * 0.2))
        
        for param, (min_val, max_val) in self.param_ranges.items():
            if param in current_params:
                current = current_params[param]
                # Gaussian perturbation scaled by temperature
                delta = np.random.normal(0, temperature * (max_val - min_val) / 4)
                proposed[param] = np.clip(current + delta, min_val, max_val)
        
        return proposed
    
    def record_trial(self, params: Dict[str, float], reward: float):
        """Log trial for later analysis."""
        self.trial_history.append({"params": params, "reward": reward})
    
    def get_best_params(self) -> Optional[Dict[str, float]]:
        """Returns best parameters found so far."""
        if not self.trial_history:
            return None
        best_trial = max(self.trial_history, key=lambda x: x["reward"])
        return best_trial["params"]


class LLMShadowEvaluator:
    """FASE 3: Validates LLM-proposed fixes before promotion."""
    
    @staticmethod
    def shadow_evaluate_llm_patch(
        proposed_patch: Dict[str, Any],
        prior_kpis: Dict[str, float],
        shadow_test_results: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        FASE 3: Test-drives LLM patch on shadow corpus before applying.
        Returns (is_valid, reason).
        """
        # If no shadow results available, accept with caution
        if shadow_test_results is None:
            return True, "No shadow corpus; accepting with TTL=2 (short validity)"
        
        # Check if patch made things worse in shadow
        if shadow_test_results.get("mean_survival_ratio", 0) < prior_kpis.get("mean_survival_ratio", 0):
            delta = prior_kpis.get("mean_survival_ratio", 0) - shadow_test_results.get("mean_survival_ratio", 0)
            if delta > 0.05:  # >5% regression
                return False, f"Shadow eval regressed survival by {delta:.2%}"
        
        return True, "Shadow eval passed"
