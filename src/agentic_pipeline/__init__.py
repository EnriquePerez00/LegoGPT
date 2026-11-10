"""
LegoGPT — Independent Multi-Agent Inference Pipeline (PDF -> .io Graph)
=======================================================================
Coordinates 6 specialized AI agents with cost-optimal LLM routing:
  1. Context & Camera Agent     (flash_lite)
  2. Identifier 3D Agent        (flash)
  3. Physical Guardian Agent    (flash_lite)
  4. Placer Visual ROI Agent    (flash + MPS GPU)
  5. Temporal Supervisor Agent  (flash)
  6. GT Auditor & Fixer Agent   (pro)

Preserves the legacy pipeline (`src/manual_connection_graph.py`) untouched.
"""

from src.agentic_pipeline.schemas import (
    AgentModelTier,
    MODEL_ROUTING_CONFIG,
    StepContext,
    CandidatePlacement,
    VisualROIEvaluation,
    GTAuditDiagnosis,
    ProposedFix,
    StepInferenceResult,
    AgenticPipelineReport,
)
from src.agentic_pipeline.orchestrator import AgenticGraphPipelineOrchestrator
from src.agentic_pipeline.io_exporter import export_nodes_to_studio_io, load_gt_nodes_from_io

__all__ = [
    "AgentModelTier",
    "MODEL_ROUTING_CONFIG",
    "StepContext",
    "CandidatePlacement",
    "VisualROIEvaluation",
    "GTAuditDiagnosis",
    "ProposedFix",
    "StepInferenceResult",
    "AgenticPipelineReport",
    "AgenticGraphPipelineOrchestrator",
    "export_nodes_to_studio_io",
    "load_gt_nodes_from_io",
]
