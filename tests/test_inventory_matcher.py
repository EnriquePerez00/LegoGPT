import os
import pytest
from src.inventory_matcher import LegoInventoryMatcher

def test_inventory_detection():
    pdf_path = "data/75400-1.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"PDF file {pdf_path} not found.")
        
    matcher = LegoInventoryMatcher(pdf_path)
    pages = matcher.detect_inventory_pages()
    
    assert len(pages) > 0
    print(f"Detected inventory pages: {pages}")

def test_template_extraction():
    pdf_path = "data/75400-1.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"PDF file {pdf_path} not found.")
        
    matcher = LegoInventoryMatcher(pdf_path)
    templates = matcher.extract_templates()
    
    assert len(templates) > 0
    print(f"Extracted {len(templates)} templates.")
    
    # Check that template crops exist on disk
    for t in templates:
        assert os.path.exists(t["image_path"])
        assert t["qty"] > 0
        assert len(t["element_id"]) >= 6

def test_vlm_pipeline_integration():
    pdf_path = "data/75400-1.pdf"
    mbx_path = "data/75400-1.mbx"
    if not os.path.exists(pdf_path) or not os.path.exists(mbx_path):
        pytest.skip("Required test data files not found.")
        
    from src.vlm_parser_pipeline import VLMManualParserPipeline
    
    pipeline = VLMManualParserPipeline(pdf_path)
    pipeline.parse_pdf_layout()
    
    # Generate build sequence with VLM and inventory matching
    sequence = pipeline.generate_build_sequence_vlm(mbx_path)
    assert len(sequence) > 0
    print(f"Generated build sequence of length: {len(sequence)}")


def test_75295_manual_inventory_and_page_36_matching():
    pdf_path = "data/manuals/75295-1_Millennium_Falcon_Microfighter.pdf"
    if not os.path.exists(pdf_path):
        pytest.skip(f"PDF file {pdf_path} not found.")

    matcher = LegoInventoryMatcher(pdf_path, temp_dir="scratch/test_inventory_75295")
    inv_pages = matcher.detect_inventory_pages()
    # Pages 42 and 43 correspond to 0-based indices 41 and 42
    assert inv_pages == [41, 42]

    result = matcher.extract_and_match_all_steps(
        db_path="data/catalog/models_catalog.db",
        set_num="75295-1",
        steps_crop_dir="scratch/test_step_crops_75295"
    )
    templates = result["templates"]
    step_pages = result["step_pages"]

    # 53 unique element references summing to 101 total parts
    assert len(templates) == 53
    assert sum(t["qty"] for t in templates) == 101

    by_code = {t["element_id"]: t for t in templates}
    assert "300221" in by_code
    assert by_code["300221"]["qty"] == 1
    assert by_code["300221"].get("part_num") == "3002"

    assert "6208450" in by_code
    assert by_code["6208450"]["qty"] == 4
    assert by_code["6208450"].get("part_num") == "6141"

    # Verify Page 36 (Step 29) detects 4x and maps to 6208450
    page_36 = next((sp for sp in step_pages if sp["page_num"] == 36), None)
    assert page_36 is not None
    assert len(page_36["items"]) == 1
    assert page_36["items"][0]["qty"] == 4
    assert page_36["items"][0]["matched_element_id"] == "6208450"

    # Verify total step pieces across all pages equals 101 with 0 unassigned stock
    total_step_pieces = sum(it["qty"] for sp in step_pages for it in sp["items"])
    assert total_step_pieces == 101
    assert sum(result["remaining_unassigned_stock"].values()) == 0


