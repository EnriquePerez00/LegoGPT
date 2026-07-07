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

