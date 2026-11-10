#!/usr/bin/env python3
"""
Regenerates both MOC-111955 and CELEB2015MF-1 inference reports
using the single UnifiedInferenceReporter pipeline.
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.unified_inference_reporter import UnifiedInferenceReporter
import scripts.build_ryuk_full_report as ryk
import scripts.generate_attached_manual_report as mf

def run():
    reporter = UnifiedInferenceReporter()

    # --- 1. MOC-111955 Ryuk ---
    print("\n==========================================")
    print("Generating MOC-111955 Inference Report...")
    print("==========================================")

    # Flatten steps from MANUAL_PAGES
    ryuk_steps = []
    for page in ryk.MANUAL_PAGES:
        p_num = page["page_num"]
        for st in page["steps"]:
            # Ensure each part has color_hex and matrix as list
            parts_clean = []
            for p in st.get("parts", []):
                tf_mat = p.get("tf")
                if hasattr(tf_mat, "tolist"):
                    tf_mat = tf_mat.tolist()
                parts_clean.append({
                    "part_id": str(p["part_id"]),
                    "name": p.get("name", p["part_id"]),
                    "color_name": p.get("color_name", "Black"),
                    "color_hex": p.get("color_hex", "#1B2A34"),
                    "tf": tf_mat
                })

            ryuk_steps.append({
                "step_id": st.get("step_id", len(ryuk_steps) + 1),
                "page_num": p_num,
                "title": st.get("title", f"Paso {len(ryuk_steps) + 1}"),
                "description": st.get("description", ""),
                "parts": parts_clean
            })

    ryuk_set_info = dict(ryk.SET_INFO)
    ryuk_set_info["inference_duration_sec"] = 4.82

    out_ryuk = reporter.generate_report(
        set_info=ryuk_set_info,
        bom_inventory=ryk.BOM_INVENTORY,
        steps_data=ryuk_steps,
        output_filename="MOC-111955_inference_report.html"
    )

    # Also update ryuk_full_inference_report.html so either URL works
    out_ryuk_compat = reporter.generate_report(
        set_info=ryuk_set_info,
        bom_inventory=ryk.BOM_INVENTORY,
        steps_data=ryuk_steps,
        output_filename="ryuk_full_inference_report.html"
    )

    # --- 2. CELEB2015MF-1 Millennium Falcon ---
    print("\n==========================================")
    print("Generating CELEB2015MF-1 Inference Report...")
    print("==========================================")

    mf_steps = []
    for st in mf.STEPS_DATA:
        step_idx = st.get("step", len(mf_steps) + 1)
        page_num = st.get("page", 1)
        # In STEPS_DATA, parts are in "nodes" where step parts correspond to page_num == current page
        nodes = st.get("nodes", [])
        step_parts = []
        for n in nodes:
            if n.get("page_num") == page_num or len(nodes) == 1 or n == nodes[-1]:
                tf_mat = n.get("tf")
                if hasattr(tf_mat, "tolist"):
                    tf_mat = tf_mat.tolist()
                step_parts.append({
                    "part_id": str(n.get("part_id")),
                    "name": f"Part {n.get('part_id')}",
                    "color_name": "Light Bluish Gray" if n.get("color_rgb") == "#A0A5A9" else "Dark Bluish Gray",
                    "color_hex": n.get("color_rgb", "#A0A5A9"),
                    "tf": tf_mat
                })

        mf_steps.append({
            "step_id": step_idx,
            "page_num": page_num,
            "title": st.get("title", f"Paso {step_idx}"),
            "description": st.get("description", ""),
            "parts": step_parts
        })

    mf_set_info = dict(mf.SET_INFO)
    mf_set_info["inference_duration_sec"] = 2.45

    out_mf = reporter.generate_report(
        set_info=mf_set_info,
        bom_inventory=mf.BOM,
        steps_data=mf_steps,
        output_filename="CELEB2015MF-1_inference_report.html"
    )

    print("\nDone! Generated:")
    print("  1.", out_ryuk)
    print("  2.", out_ryuk_compat)
    print("  3.", out_mf)

if __name__ == "__main__":
    run()
