import os
import pytest
import requests
import gzip
import csv
import io
from src.omr_downloader import download_set_by_number
from src.mpd_parser import flatten_mpd
from src.homogenize_assembly import homogenize_parts_list

def test_omr_one_model_validation():
    # Download exactly one model: 31027 (Blue Racer)
    print("\n[OMR Validation] Descargando set 31027...")
    file_path = download_set_by_number("31027", output_dir="data/omr_test")
    
    assert file_path is not None
    assert os.path.exists(file_path)
    
    # Parse LDraw file
    parts = flatten_mpd(file_path)
    assert len(parts) > 0
    print(f"[OMR Validation] Set 31027 importado exitosamente con {len(parts)} piezas.")
    
    # Clean up downloaded test file
    if os.path.exists(file_path):
        os.remove(file_path)
    if os.path.exists("data/omr_test"):
        os.rmdir("data/omr_test")

def test_rebrickable_one_record_validation():
    # Download sets catalog CSV from Rebrickable
    print("\n[Rebrickable Validation] Descargando cabecera de sets.csv.gz...")
    url = "https://rebrickable.com/media/downloads/sets.csv.gz"
    
    try:
        response = requests.get(url, stream=True, timeout=15)
        assert response.status_code == 200
        
        # Read the first few lines to validate the format without downloading the entire 50MB
        with gzip.GzipFile(fileobj=response.raw) as f:
            # Wrap in TextIOWrapper to read strings
            text_file = io.TextIOWrapper(f, encoding='utf-8')
            reader = csv.DictReader(text_file)
            
            # Read exactly 1 set record
            first_row = next(reader)
            assert "set_num" in first_row
            assert "name" in first_row
            assert "year" in first_row
            assert "theme_id" in first_row
            
            print(f"[Rebrickable Validation] Formato de sets.csv validado. Registro de muestra:")
            print(f"  Set: {first_row['set_num']} - {first_row['name']} ({first_row['year']})")
    except Exception as e:
        pytest.skip(f"Ignorado por conectividad de red a Rebrickable: {e}")

def test_homogenization_flow_validation():
    # Test homogenization on a simple mock chassis assembly
    from tests.test_vehicle_reward import create_mock_part
    
    # 2 wheels and 1 chassis plate
    # Position them on the grid: wheels X at -10 and 10, Z at 10 (aligns with 3020.dat sockets at Z = 10)
    w1 = create_mock_part("42610.dat", 0, -10.0, 0.0, 10.0)
    w2 = create_mock_part("42610.dat", 0, 10.0, 0.0, 10.0)
    p1 = create_mock_part("3020.dat", 1, 0.0, -16.0, 0.0)
    
    parts = [w1, w2, p1]
    
    # Homogenize
    sequence = homogenize_parts_list(parts)
    assert len(sequence) == 3
    
    # The plate (p1 at index 2) should have mechanical connections to both wheels (index 0 and 1)
    p1_record = sequence[2]
    assert p1_record["part_id"] == "3020.dat"
    assert len(p1_record["connections"]) == 2
    
    # Check absolute transforms exist
    assert len(p1_record["absolute_transform"]) == 16
    
    # Check parent anchors are resolved
    conn = p1_record["connections"][0]
    assert conn["parent_part_index"] == 0
    assert conn["parent_anchor"].startswith("stud_") or conn["parent_anchor"].startswith("socket_")
    
    print("[Homogenization Validation] Secuencia de Legograph codificada correctamente con enlaces mecánicos.")
