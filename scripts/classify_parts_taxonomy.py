"""
LegoGPT: Part Semantic Taxonomy Classifier
Classifies all 65,000+ parts in models_catalog.db into a 4-dimensional semantic taxonomy:
1. functional_role (STRUCTURAL_CORE, SKIN_AESTHETIC, KINEMATIC_MECHANISM, etc.)
2. morphology (PRISMATIC_BLOCK, PLATE_FLAT, TILE_SMOOTH, SLOPE_ANGLED, etc.)
3. interface_type (STUD_ANTISTUD, TECHNIC_PIN_AXLE, CLIP_BAR, BALL_SOCKET, etc.)
4. assembly_nature (BUILDING_BLOCK, MINIFIG_ITEM, MECHANICAL_SYSTEM, NATURE_ORGANIC)

Optimized for Apple Silicon M4 with dynamic multi-core worker allocation and SQLite batching.
"""

import os
import re
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Dict, Any

DB_PATH = "data/catalog/models_catalog.db"

# Category mappings from Rebrickable part_cat_id
CAT_ROLE_MAP = {
    1: "STRUCTURAL_CORE",        # Baseplates
    3: "SKIN_AESTHETIC",         # Bricks Sloped
    4: "NON_SYSTEM",             # Duplo, Quatro and Primo
    5: "CONNECTOR_HARDWARE",     # Bricks Special
    6: "SKIN_AESTHETIC",         # Bricks Wedged
    7: "CONTAINER_STORAGE",      # Containers
    8: "STRUCTURAL_CORE",        # Technic Bricks
    9: "CONNECTOR_HARDWARE",     # Plates Special
    11: "STRUCTURAL_CORE",       # Bricks
    12: "CONNECTOR_HARDWARE",    # Technic Connectors
    13: "MINIFIG_ANATOMY",       # Minifigs
    14: "STRUCTURAL_CORE",       # Plates
    15: "SKIN_AESTHETIC",        # Tiles Special
    16: "HABITATION_ARCHITECTURE",# Windows and Doors
    17: "KINEMATIC_MECHANISM",   # Gear Parts
    18: "CONNECTOR_HARDWARE",    # Hinges, Arms and Turntables
    19: "SKIN_AESTHETIC",        # Tiles
    20: "STRUCTURAL_CORE",       # Bricks Round and Cones
    21: "SKIN_AESTHETIC",        # Plates Round Curved and Dishes
    22: "KINEMATIC_MECHANISM",   # Pneumatics
    23: "SKIN_AESTHETIC",        # Panels
    24: "SPECIAL_FUNCTION",      # Other
    25: "KINEMATIC_MECHANISM",   # Technic Steering, Suspension and Engine
    26: "KINEMATIC_MECHANISM",   # Technic Special
    27: "MINIFIG_ACCESSORY",     # Minifig Accessories
    28: "ORGANIC_NATURE",        # Animals / Creatures
    29: "KINEMATIC_MECHANISM",   # Wheels and Tyres
    30: "KINEMATIC_MECHANISM",   # Tubes and Hoses
    31: "KINEMATIC_MECHANISM",   # String, Bands and Reels
    32: "HABITATION_ARCHITECTURE",# Bars, Ladders and Fences
    33: "ORGANIC_NATURE",        # Rock
    34: "STRUCTURAL_CORE",       # Supports, Girders and Cranes
    35: "AERODYNAMICS_COCKPIT",  # Transportation - Sea and Air
    36: "KINEMATIC_MECHANISM",   # Transportation - Land
    37: "SKIN_AESTHETIC",        # Bricks Curved
    38: "GRAPHICS_DECORATION",   # Flags, Banners and Signs
    39: "CONNECTOR_HARDWARE",    # Magnets and Holders
    40: "SKIN_AESTHETIC",        # Technic Panels
    41: "SPECIAL_FUNCTION",      # Large Buildable Figures
    42: "NON_SYSTEM",            # Belville, Scala and Fabuland
    43: "NON_SYSTEM",            # Znap
    44: "KINEMATIC_MECHANISM",   # Mechanical
    45: "ELECTRONICS_LIGHTING",  # Electronics
    46: "KINEMATIC_MECHANISM",   # Technic Axles
    47: "AERODYNAMICS_COCKPIT",  # Windscreens and Fuselage
    48: "NON_SYSTEM",            # Clikits
    49: "SKIN_AESTHETIC",        # Plates Wedged
    50: "NON_SYSTEM",            # HO Scale
    51: "STRUCTURAL_CORE",       # Technic Beams
    52: "KINEMATIC_MECHANISM",   # Technic Gears
    53: "CONNECTOR_HARDWARE",    # Technic Pins
    54: "CONNECTOR_HARDWARE",    # Technic Bushes
    55: "STRUCTURAL_CORE",       # Technic Beams Special
    56: "MINIFIG_ACCESSORY",     # Tools
    57: "NON_SYSTEM",            # Non-Buildable Figures
    58: "GRAPHICS_DECORATION",   # Stickers
    59: "MINIFIG_ANATOMY",       # Minifig Heads
    60: "MINIFIG_ANATOMY",       # Minifig Upper Body
    61: "MINIFIG_ANATOMY",       # Minifig Lower Body
    62: "MINIFIG_ANATOMY",       # Minidoll Heads
    63: "MINIFIG_ANATOMY",       # Minidoll Upper Body
    64: "MINIFIG_ANATOMY",       # Minidoll Lower Body
    65: "MINIFIG_ANATOMY",       # Minifig Headwear
    66: "NON_SYSTEM",            # Modulex
    67: "SKIN_AESTHETIC",        # Tiles Round and Curved
    68: "MINIFIG_ACCESSORY",     # Projectiles / Launchers
    69: "SPECIAL_FUNCTION",      # Energy Effects
    70: "MINIFIG_ANATOMY",       # Minifig Hipwear
    71: "MINIFIG_ANATOMY",       # Minifig Neckwear
    72: "MINIFIG_ACCESSORY",     # Minifig Headwear Accessories
    73: "MINIFIG_ACCESSORY",     # Minifig Shields, Weapons, & Tools
    74: "ORGANIC_NATURE",        # Animal / Creature Accessories
    75: "ORGANIC_NATURE",        # Animal / Creature Body Parts
    76: "ORGANIC_NATURE",        # Plants & Trees
    77: "NON_SYSTEM",            # Non-System Parts
    78: "SPECIAL_FUNCTION"       # Pen & Watch
}


def classify_single_part(part_tuple: Tuple[str, str, int, str]) -> Tuple:
    """
    Classifies a part given (part_num, name, part_cat_id, part_material)
    Returns:
    (part_num, part_name, part_cat_id, functional_role, morphology,
     interface_type, assembly_nature, description_es, confidence, classified_by)
    """
    part_num, name, cat_id, material = part_tuple
    name_clean = (name or "").strip()
    name_lower = name_clean.lower()
    
    # 1. Base from Category
    role = CAT_ROLE_MAP.get(cat_id, "SPECIAL_FUNCTION")
    morphology = "SPECIAL_SCULPTED"
    interface_type = "STUD_ANTISTUD"
    assembly_nature = "BUILDING_BLOCK"
    description_es = "Pieza de construcción general"
    confidence = 0.85
    
    # 2. Refined Pattern Matching on Name & Material
    # Material checks
    if material and "rubber" in material.lower():
        role = "KINEMATIC_MECHANISM"
        assembly_nature = "MECHANICAL_SYSTEM"
        
    # --- Wheels & Tires ---
    if any(k in name_lower for k in ["tire", "tyre"]):
        role = "KINEMATIC_MECHANISM"
        morphology = "CYLINDRICAL_ROUND"
        interface_type = "FRICTION_FIT"
        assembly_nature = "MECHANICAL_SYSTEM"
        description_es = "Neumático de caucho o plástico para vehículo"
        confidence = 0.98
    elif any(k in name_lower for k in ["wheel rim", "wheel 8", "wheel 11", "wheel 18", "wheel 30", "wheel 43", "wheel 56"]):
        role = "KINEMATIC_MECHANISM"
        morphology = "GEAR_WHEEL"
        interface_type = "TECHNIC_PIN_AXLE"
        assembly_nature = "MECHANICAL_SYSTEM"
        description_es = "Llanta o rueda de plástico para vehículo"
        confidence = 0.98
        
    # --- Windows, Doors & Glass ---
    elif any(k in name_lower for k in ["windscreen", "canopy", "cockpit"]):
        role = "AERODYNAMICS_COCKPIT"
        morphology = "FRAME_HOLLOW" if "frame" in name_lower else "SLOPE_CURVED"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Parabrisas, cabina o cúpula transparente de vehículo"
        confidence = 0.95
    elif any(k in name_lower for k in ["window", "door", "glass for window", "pane for window", "shutter"]):
        role = "HABITATION_ARCHITECTURE"
        morphology = "FRAME_HOLLOW"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Marco, cristal o persiana de puerta/ventana arquitectónica"
        confidence = 0.96
    elif any(k in name_lower for k in ["fence", "ladder", "stair", "gate", "railing"]):
        role = "HABITATION_ARCHITECTURE"
        morphology = "FRAME_HOLLOW"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Elemento arquitectónico: valla, reja o escalera"
        confidence = 0.95
        
    # --- Minifig Weapons, Accessories & Gear ---
    elif any(k in name_lower for k in ["blaster", "gun", "pistol", "rifle", "weapon", "sword", "lightsaber",
                                     "dagger", "knife", "shield", "spear", "bow", "arrow", "wand", "staff"]):
        role = "MINIFIG_ACCESSORY"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "CLIP_BAR"
        assembly_nature = "MINIFIG_ITEM"
        description_es = "Arma o accesorio de combate para minifigura"
        confidence = 0.99
    elif any(k in name_lower for k in ["helmet", "hair", "hat", "cap", "crown", "wig", "hood"]):
        role = "MINIFIG_ANATOMY"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "MINIFIG_ITEM"
        description_es = "Casco, pelo o tocado de cabeza para minifigura"
        confidence = 0.99
    elif any(k in name_lower for k in ["torso", "legs", "head", "arm", "hand"]):
        if cat_id in [13, 59, 60, 61, 62, 63, 64, 65, 70, 71]:
            role = "MINIFIG_ANATOMY"
            morphology = "SPECIAL_SCULPTED"
            interface_type = "STUD_ANTISTUD"
            assembly_nature = "MINIFIG_ITEM"
            description_es = "Parte anatómica corporal de minifigura"
            confidence = 0.99
    elif any(k in name_lower for k in ["cup", "mug", "chalice", "broom", "shovel", "hammer", "wrench", "binoculars", "torch"]):
        role = "MINIFIG_ACCESSORY"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "CLIP_BAR"
        assembly_nature = "MINIFIG_ITEM"
        description_es = "Herramienta u objeto cotidiano para minifigura"
        confidence = 0.96
        
    # --- Tiles & Plates (Finishing vs Structural) ---
    elif "tile" in name_lower:
        role = "SKIN_AESTHETIC"
        morphology = "TILE_SMOOTH"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Placa de acabado liso sin tacos superiores (Tile)"
        confidence = 0.95
    elif "baseplate" in name_lower or cat_id == 1:
        role = "STRUCTURAL_CORE"
        morphology = "PLATE_FLAT"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Placa base de soporte estructural o suelo"
        confidence = 0.98
    elif "plate" in name_lower:
        role = "STRUCTURAL_CORE" if any(w in name_lower for w in ["2 x", "4 x", "6 x", "8 x", "16 x"]) else "SKIN_AESTHETIC"
        morphology = "PLATE_FLAT"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Placa delgada (espesor 1/3 de ladrillo)"
        confidence = 0.95
    elif "slope" in name_lower or "wedge" in name_lower:
        role = "SKIN_AESTHETIC"
        morphology = "SLOPE_CURVED" if "curved" in name_lower else "SLOPE_ANGLED"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Pendiente o cuña inclinada/curva para cubiertas y aerodinámica"
        confidence = 0.95
    elif "brick" in name_lower:
        if "technic" in name_lower or cat_id == 8:
            role = "STRUCTURAL_CORE"
            morphology = "PRISMATIC_BLOCK"
            interface_type = "TECHNIC_PIN_AXLE"
            assembly_nature = "BUILDING_BLOCK"
            description_es = "Ladrillo Technic estructural con orificios para pines/ejes"
        else:
            role = "STRUCTURAL_CORE"
            morphology = "CYLINDRICAL_ROUND" if "round" in name_lower or "cone" in name_lower else "PRISMATIC_BLOCK"
            interface_type = "STUD_ANTISTUD"
            assembly_nature = "BUILDING_BLOCK"
            description_es = "Ladrillo de construcción estructural estándar (Brick)"
        confidence = 0.95
        
    # --- Technic Kinematic & Mechanical ---
    elif any(k in name_lower for k in ["gear", "cog"]):
        role = "KINEMATIC_MECHANISM"
        morphology = "GEAR_WHEEL"
        interface_type = "TECHNIC_PIN_AXLE"
        assembly_nature = "MECHANICAL_SYSTEM"
        description_es = "Engranaje o rueda dentada de transmisión de movimiento"
        confidence = 0.98
    elif any(k in name_lower for k in ["axle", "shaft"]):
        role = "KINEMATIC_MECHANISM"
        morphology = "BAR_ROD"
        interface_type = "TECHNIC_PIN_AXLE"
        assembly_nature = "MECHANICAL_SYSTEM"
        description_es = "Eje de transmisión o sujeción Technic en cruz"
        confidence = 0.98
    elif any(k in name_lower for k in ["pin ", "pin,", "pin with", "peg"]):
        role = "CONNECTOR_HARDWARE"
        morphology = "CYLINDRICAL_ROUND"
        interface_type = "TECHNIC_PIN_AXLE"
        assembly_nature = "MECHANICAL_SYSTEM"
        description_es = "Pin o perno de fijación y conexión rápida Technic"
        confidence = 0.98
    elif any(k in name_lower for k in ["hinge", "turntable", "swivel"]):
        role = "CONNECTOR_HARDWARE"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "HINGE"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Bisagra, rótula o base giratoria para articulaciones"
        confidence = 0.96
    elif any(k in name_lower for k in ["ball joint", "towball", "socket"]):
        role = "CONNECTOR_HARDWARE"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "BALL_SOCKET"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Rótula esférica articulada (Ball and Socket)"
        confidence = 0.97
    elif any(k in name_lower for k in ["clip", "bar"]):
        role = "CONNECTOR_HARDWARE"
        morphology = "BAR_ROD" if "bar" in name_lower else "SPECIAL_SCULPTED"
        interface_type = "CLIP_BAR"
        assembly_nature = "BUILDING_BLOCK"
        description_es = "Conector de pinza o barra cilíndrica de 3.18mm"
        confidence = 0.95
        
    # --- Plants, Nature & Animals ---
    elif any(k in name_lower for k in ["plant", "tree", "flower", "leaf", "bush", "grass", "vine", "seaweed"]):
        role = "ORGANIC_NATURE"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "NATURE_ORGANIC"
        description_es = "Elemento de vegetación, botánica o naturaleza orgánica"
        confidence = 0.97
    elif any(k in name_lower for k in ["animal", "horse", "dog", "cat", "bird", "fish", "shark", "dinosaur", "dragon"]):
        role = "ORGANIC_NATURE"
        morphology = "SPECIAL_SCULPTED"
        interface_type = "STUD_ANTISTUD"
        assembly_nature = "NATURE_ORGANIC"
        description_es = "Figura o cuerpo de animal o criatura viva"
        confidence = 0.97

    classified_by = "rule_based_taxonomy_v1"
    
    return (
        part_num,
        name_clean,
        cat_id,
        role,
        morphology,
        interface_type,
        assembly_nature,
        description_es,
        confidence,
        classified_by
    )


def init_taxonomy_table(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS part_semantic_taxonomy (
        part_num TEXT PRIMARY KEY,
        part_name TEXT,
        part_cat_id INTEGER,
        functional_role TEXT,
        morphology TEXT,
        interface_type TEXT,
        assembly_nature TEXT,
        description_es TEXT,
        confidence REAL,
        classified_by TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_part_tax_role ON part_semantic_taxonomy(functional_role);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_part_tax_morph ON part_semantic_taxonomy(morphology);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_part_tax_nature ON part_semantic_taxonomy(assembly_nature);")
    conn.commit()


def main():
    start_time = time.time()
    print("🚀 Iniciando Clasificación Semántica Masiva de Piezas LEGO...")
    
    conn = sqlite3.connect(DB_PATH)
    init_taxonomy_table(conn)
    
    cursor = conn.cursor()
    cursor.execute("SELECT part_num, name, part_cat_id, part_material FROM rb_parts;")
    rows = cursor.fetchall()
    total_parts = len(rows)
    print(f"📦 Total de piezas encontradas en 'rb_parts': {total_parts:,}")
    
    # M4 Dynamic Worker Calculation: 12 cores, reserve 2 -> 10 workers
    max_workers = max(1, (os.cpu_count() or 12) - 2)
    print(f"⚡ Ejecutando en paralelo sobre Apple Silicon M4 ({max_workers} procesos worker)...")
    
    # Process in chunks using ProcessPoolExecutor
    chunk_size = 5000
    classified_records = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, total_parts, chunk_size):
            chunk = rows[i:i + chunk_size]
            results = list(executor.map(classify_single_part, chunk))
            classified_records.extend(results)
            print(f"   ✓ Procesadas {len(classified_records):,} / {total_parts:,} piezas...")
            
    print(f"💾 Guardando {len(classified_records):,} registros en tabla 'part_semantic_taxonomy'...")
    cursor.execute("BEGIN TRANSACTION;")
    cursor.executemany("""
        INSERT OR REPLACE INTO part_semantic_taxonomy 
        (part_num, part_name, part_cat_id, functional_role, morphology, interface_type, 
         assembly_nature, description_es, confidence, classified_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, classified_records)
    conn.commit()
    
    # Summary stats
    cursor.execute("""
        SELECT functional_role, count(*) 
        FROM part_semantic_taxonomy 
        GROUP BY functional_role 
        ORDER BY count(*) DESC;
    """)
    role_stats = cursor.fetchall()
    
    cursor.execute("""
        SELECT assembly_nature, count(*) 
        FROM part_semantic_taxonomy 
        GROUP BY assembly_nature 
        ORDER BY count(*) DESC;
    """)
    nature_stats = cursor.fetchall()
    
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n✨ Clasificación completada en {elapsed:.2f} segundos!")
    print("\n--- DISTRIBUCIÓN POR ROL FUNCIONAL ---")
    for r, c in role_stats:
        print(f"  • {r:<25}: {c:,} ({c*100/total_parts:.1f}%)")
        
    print("\n--- DISTRIBUCIÓN POR NATURALEZA DE ENSAMBLE ---")
    for n, c in nature_stats:
        print(f"  • {n:<25}: {c:,} ({c*100/total_parts:.1f}%)")


if __name__ == "__main__":
    main()
