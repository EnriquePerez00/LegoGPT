import numpy as np
from src.parser import ParsedPart
from scripts.extract_vehicle_dataset import is_vehicle_model

def test_is_vehicle_model():
    # 1. Mock vehicle (contains wheel and tire parts)
    p1 = ParsedPart(part_id="3139.dat", color=0, transform=np.eye(4), step_id=0) # tire
    p2 = ParsedPart(part_id="3139.dat", color=0, transform=np.eye(4), step_id=0)
    p3 = ParsedPart(part_id="42610.dat", color=0, transform=np.eye(4), step_id=0) # wheel rim
    p4 = ParsedPart(part_id="42610.dat", color=0, transform=np.eye(4), step_id=0)
    p5 = ParsedPart(part_id="3001.dat", color=1, transform=np.eye(4), step_id=0)  # standard brick
    
    parts_car = [p1, p2, p3, p4, p5]
    assert is_vehicle_model(parts_car, "31040_Dune-Buggy.mpd") is True
    
    # 2. Mock non-vehicle (standard bricks only, no keywords)
    p_non = [p5, p5, p5]
    assert is_vehicle_model(p_non, "house_build.mpd") is False
    
    # 3. Model with keyword "car" and at least 2 wheels
    parts_kart = [p1, p2, p5]
    assert is_vehicle_model(parts_kart, "speedy_car.mpd") is True
