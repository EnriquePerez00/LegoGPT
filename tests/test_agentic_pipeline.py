"""
Unit and Integration Tests for the Independent Multi-Agent PDF -> .io Pipeline
(`src/agentic_pipeline/`).
"""

from __future__ import annotations

import os
import tempfile
import zipfile

import numpy as np
import pytest

from src.agentic_pipeline import (
    MODEL_ROUTING_CONFIG,
    AgentModelTier,
    AgenticGraphPipelineOrchestrator,
    export_nodes_to_studio_io,
    load_gt_nodes_from_io,
)
from scripts.run_agentic_pipeline import create_synthetic_gt_io_for_selftest


def test_cost_optimal_model_routing_table():
    """Verifies that each specialized agent is assigned its cost-optimal LLM model tier."""
    assert MODEL_ROUTING_CONFIG["lego-context-agent"]["model"] == AgentModelTier.FLASH_LITE.value
    assert MODEL_ROUTING_CONFIG["lego-guardian-agent"]["model"] == AgentModelTier.FLASH_LITE.value
    assert MODEL_ROUTING_CONFIG["lego-identifier-agent"]["model"] == AgentModelTier.FLASH.value
    assert MODEL_ROUTING_CONFIG["lego-placer-agent"]["model"] == AgentModelTier.FLASH.value
    assert MODEL_ROUTING_CONFIG["lego-supervisor-agent"]["model"] == AgentModelTier.FLASH.value
    assert MODEL_ROUTING_CONFIG["lego-gt-auditor-agent"]["model"] == AgentModelTier.PRO.value


def test_studio_io_export_and_gt_loader():
    """Verifies round-trip export to BrickLink Studio .io archive and GT loading."""
    with tempfile.TemporaryDirectory() as tmpdir:
        io_path = os.path.join(tmpdir, "test_model.io")
        create_synthetic_gt_io_for_selftest(io_path)

        assert os.path.exists(io_path)
        assert zipfile.is_zipfile(io_path)

        flat_nodes, gt_steps = load_gt_nodes_from_io(io_path)
        assert len(flat_nodes) == 5
        assert len(gt_steps) == 4
        # Base plate must be normalized at ground Y = 0.0
        assert abs(flat_nodes[0]["y_bottom_normalized"]) <= 1e-3


def test_end_to_end_6_agent_orchestrator_with_gt_audit():
    """
    Runs the full 6-agent pipeline (Context -> Identifier -> Guardian -> Placer ROI -> Supervisor -> GT Auditor)
    on a multi-step synthetic GT .io and verifies:
      1. Physical ground constraints (Y <= 0.0, <= 4 ground-touching parts).
      2. Valid .io output generation.
      3. GT Auditor root-cause diagnosis and auto-fix persistence.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        gt_io_path = os.path.join(tmpdir, "gt_reference.io")
        rules_path = os.path.join(tmpdir, "learned_rules.json")
        create_synthetic_gt_io_for_selftest(gt_io_path)

        orchestrator = AgenticGraphPipelineOrchestrator(rules_path=rules_path)
        report = orchestrator.run_pipeline(
            set_num="TEST-AGENTIC-01",
            pdf_path="",
            gt_io_path=gt_io_path,
            output_dir=tmpdir,
            teacher_forcing_on_audit=True,
        )

        assert report.total_parts_placed == 5
        assert report.total_steps == 4
        assert 1 <= report.ground_parts_touching <= 4
        assert os.path.exists(report.output_io_path)
        assert os.path.exists(report.output_json_path)
        assert report.gt_evaluated is True
        assert report.pct_parts_correct_after_fixes >= report.pct_parts_correct_before_fixes
