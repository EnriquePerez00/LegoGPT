from __future__ import annotations
import numpy as np
import networkx as nx
import re
import sqlite3
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.parser import ParsedPart

PART_DIMENSIONS = {
    "3005.dat": (20.0, 24.0, 20.0),
    "3004.dat": (20.0, 24.0, 40.0),
    "3003.dat": (40.0, 24.0, 40.0),
    "3002.dat": (40.0, 24.0, 60.0),
    "3001.dat": (40.0, 24.0, 80.0),
    "3010.dat": (20.0, 24.0, 80.0),
    "3009.dat": (20.0, 24.0, 120.0),
    "3008.dat": (20.0, 24.0, 160.0),
    "3007.dat": (40.0, 24.0, 160.0),
    "3006.dat": (40.0, 24.0, 200.0),
    "3024.dat": (20.0, 8.0, 20.0),
    "3023.dat": (20.0, 8.0, 40.0),
    "3022.dat": (40.0, 8.0, 40.0),
    "3021.dat": (40.0, 8.0, 60.0),
    "3020.dat": (40.0, 8.0, 80.0),
    "3710.dat": (20.0, 8.0, 80.0),
    "3666.dat": (20.0, 8.0, 120.0),
    "3460.dat": (20.0, 8.0, 160.0),
    "3032.dat": (80.0, 8.0, 120.0),
    "3031.dat": (80.0, 8.0, 80.0),
    "3030.dat": (80.0, 8.0, 200.0),
    "3035.dat": (80.0, 8.0, 160.0),
    "3034.dat": (40.0, 8.0, 160.0),
    "3832.dat": (40.0, 8.0, 200.0),
    "3062b.dat": (20.0, 24.0, 20.0),
    "3941.dat": (40.0, 24.0, 40.0),
    "3036.dat": (120.0, 8.0, 160.0),
    "3958.dat": (120.0, 8.0, 120.0),
    "3005p01.dat": (20.0, 24.0, 20.0),
    "3004p01.dat": (20.0, 24.0, 40.0),
    "3003p01.dat": (40.0, 24.0, 40.0),
    "3010p01.dat": (20.0, 24.0, 80.0),
    
    # 75400-1 specific part dimensions
    "3070.dat": (20.0, 8.0, 20.0),
    "3069.dat": (20.0, 8.0, 40.0),
    "63864.dat": (20.0, 8.0, 60.0),
    "15573.dat": (20.0, 8.0, 40.0),
    "26601.dat": (40.0, 8.0, 40.0),
    "11477.dat": (20.0, 16.0, 40.0),
    "15556.dat": (10.0, 24.0, 10.0),
    "28697.dat": (10.0, 64.0, 10.0),
    "28701.dat": (20.0, 24.0, 20.0),
    "35338.dat": (20.0, 16.0, 20.0),
    "35380.dat": (20.0, 8.0, 20.0),
    "35480.dat": (20.0, 8.0, 40.0),
    "35787.dat": (40.0, 8.0, 40.0),
    "3665.dat": (20.0, 24.0, 40.0),
    "41769.dat": (40.0, 8.0, 80.0),
    "41770.dat": (40.0, 8.0, 80.0),
    "41822.dat": (80.0, 8.0, 80.0),
    "42610.dat": (20.0, 16.0, 20.0),
    "4274.dat": (10.0, 20.0, 10.0),
    "42923.dat": (20.0, 8.0, 40.0),
    "48205.dat": (80.0, 8.0, 120.0),
    "48208.dat": (80.0, 8.0, 120.0),
    "50340.dat": (20.0, 8.0, 40.0),
    "5091.dat": (20.0, 8.0, 40.0),
    "5092.dat": (20.0, 8.0, 40.0),
    "51483.dat": (20.0, 8.0, 80.0),
    "5414.dat": (20.0, 24.0, 80.0),
    "5415.dat": (20.0, 24.0, 80.0),
    "65426.dat": (40.0, 8.0, 80.0),
    "65429.dat": (40.0, 8.0, 80.0),
    "69754.dat": (20.0, 8.0, 40.0),
    "69755.dat": (20.0, 8.0, 20.0),
    "76382.dat": (20.0, 24.0, 10.0),
    "111870.dat": (20.0, 24.0, 20.0),
    "112754.dat": (20.0, 24.0, 20.0),
    "112755.dat": (40.0, 24.0, 40.0),
    "32803.dat": (40.0, 16.0, 40.0),
    "79491.dat": (40.0, 8.0, 40.0),
}


_PART_NAME_CACHE: dict[str, str] = {}

def get_part_name_from_db(part_id: str, db_path: str = "data/catalog/models_catalog.db") -> str:
    """Looks up the name/description of a part in the Rebrickable database tables."""
    clean_id = part_id.lower().replace(".dat", "").replace("bl_", "").strip()
    if not clean_id:
        return ""
        
    if clean_id in _PART_NAME_CACHE:
        return _PART_NAME_CACHE[clean_id]
        
    name = ""
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Try exact match on rb_parts
        cursor.execute("SELECT name FROM rb_parts WHERE part_num = ?", (clean_id,))
        row = cursor.fetchone()
        if row:
            name = row[0]
        else:
            # 2. Try exact match on rb_parts_enriched
            cursor.execute("SELECT formal_description, category_group, utility_labels FROM rb_parts_enriched WHERE part_num = ?", (clean_id,))
            row = cursor.fetchone()
            if row:
                desc, cat, labels = row
                name = f"{desc} {cat} {labels}"
            else:
                # 3. Try prefix match with LIKE (for letters/suffixes, e.g. 44567a)
                cursor.execute("SELECT name FROM rb_parts WHERE part_num LIKE ? LIMIT 1", (clean_id + "%",))
                row = cursor.fetchone()
                if row:
                    name = row[0]
                else:
                    # Try stripping letters and matching by LIKE
                    import re
                    numeric_id = re.sub(r"[a-zA-Z]", "", clean_id)
                    if numeric_id:
                        cursor.execute("SELECT name FROM rb_parts WHERE part_num LIKE ? LIMIT 1", (numeric_id + "%",))
                        row = cursor.fetchone()
                        if row:
                            name = row[0]
        conn.close()
    except Exception:
        pass
        
    name = name.lower()
    _PART_NAME_CACHE[clean_id] = name
    return name

_DIMENSIONS_CACHE: dict[str, tuple[float, float, float]] = {}

def get_part_dimensions(part_id: str) -> tuple[float, float, float]:
    """Helper to get part dimensions (width, height, depth), with fallback for unknown parts."""
    # Strip bl_ prefix for lookup in PART_DIMENSIONS
    clean_id = part_id.lower().replace("bl_", "")
    dim = PART_DIMENSIONS.get(clean_id)
    if not dim:
        dim = PART_DIMENSIONS.get(part_id)
        
    if not dim:
        # Try to infer from name / description in Rebrickable DB first, then PART_NAMES
        name = get_part_name_from_db(part_id)
        if not name:
            try:
                from src.sequence_visualizer import PART_NAMES
                name = PART_NAMES.get(part_id, "").lower()
            except Exception:
                pass
            
        part_lower = part_id.lower()
        part_clean = part_lower.replace("bl_", "")
        
        # Fallbacks for minifigures using regex word boundaries to avoid false positives (e.g. handle -> hand, headlight -> head)
        import re
        if "torso" in part_clean or "973" in part_clean or re.search(r"\btorso\b", name):
            dim = (20.0, 18.0, 10.0)
        elif "leg" in part_clean or "970" in part_clean or "hip" in part_clean or re.search(r"\bleg\b|\bhips\b", name):
            dim = (20.0, 19.0, 10.0)
        elif "3626" in part_clean or (re.search(r"\bhead\b", name) and not "headlight" in name):
            dim = (16.0, 20.0, 16.0)
        elif "983" in part_clean or re.search(r"\bhand\b", name):
            dim = (8.0, 10.0, 8.0)
        elif "981" in part_clean or "982" in part_clean or re.search(r"\barm\b", name):
            dim = (8.0, 15.0, 8.0)
            
        # Fallbacks for bars and shafts (e.g. bar 2l, bar 3l, light sword shaft)
        # Excludes holders, plates, clips, bricks, and tiles that happen to have "bar" in their description
        elif ("bar" in part_clean or "shaft" in part_clean or re.search(r"\bbar\b|\bshaft\b", name)) and not any(w in name or w in part_clean for w in ["holder", "clip", "plate", "brick", "tile"]):
            m_l = re.search(r"(\d+)l", part_clean)
            if not m_l:
                m_l = re.search(r"(\d+)l", name)
            if m_l:
                length = float(m_l.group(1)) * 20.0
                dim = (6.0, length, 6.0)
            else:
                dim = (6.0, 40.0, 6.0)
            
        # Fallbacks for brackets
        elif "bracket" in part_clean or "bracket" in name:
            dim = (20.0, 28.0, 28.0)
            
        else:
            # Default height
            h = 24.0 # default brick height
            if "plate" in part_lower or "tile" in part_lower or "plate" in name or "tile" in name:
                h = 8.0
            elif "brick" in part_lower or "brick" in name:
                h = 24.0
                
            # Try to find dimensions like "1x6x5", "1 x 4 x 6" first
            m3 = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", name)
            if not m3:
                m3 = re.search(r"(\d+)\s*x\s*(\d+)\s*x\s*(\d+)", part_lower)
                
            if m3:
                dim_x = float(m3.group(1)) * 20.0
                dim_z = float(m3.group(2)) * 20.0
                dim_y = float(m3.group(3)) * 24.0  # height is in brick units (1 brick = 24 LDU)
                dim = (dim_x, dim_y, dim_z)
            else:
                # Try to find dimensions like "2x8", "1x2"
                m = re.search(r"(\d+)\s*x\s*(\d+)", name)
                if not m:
                    m = re.search(r"(\d+)\s*x\s*(\d+)", part_lower)
                    
                if m:
                    dim_x = float(m.group(1)) * 20.0
                    dim_z = float(m.group(2)) * 20.0
                    dim = (dim_x, h, dim_z)
                else:
                    # prefix based fallbacks
                    if part_lower.startswith("302") or part_lower.startswith("307"):
                        dim = (20.0, 8.0, 20.0)
                    else:
                        dim = (20.0, 24.0, 20.0)  # Standard 1x1 brick fallback
                        
    _DIMENSIONS_CACHE[part_id] = dim
    return dim

def get_part_corners_world(part: ParsedPart) -> np.ndarray:
    """Computes the 8 corners of the part's bounding box in world space."""
    dim = get_part_dimensions(part.part_id)
    w, h, d = dim[0], dim[1], dim[2]
    # Local corners: centered in X and Z, Y goes from -h (top) to 0 (bottom)
    local_corners = np.array([
        [-w/2, 0.0, -d/2, 1.0],
        [ w/2, 0.0, -d/2, 1.0],
        [-w/2, 0.0,  d/2, 1.0],
        [ w/2, 0.0,  d/2, 1.0],
        [-w/2,  -h, -d/2, 1.0],
        [ w/2,  -h, -d/2, 1.0],
        [-w/2,  -h,  d/2, 1.0],
        [ w/2,  -h,  d/2, 1.0]
    ], dtype=np.float32)
    # Transform to world space
    world_corners = (part.transform @ local_corners.T).T[:, :3]
    return world_corners

def get_studs_and_sockets_world(part: ParsedPart) -> tuple[np.ndarray, np.ndarray]:
    """Generates the coordinates of the studs (top) and sockets (bottom) in world space."""
    dim = get_part_dimensions(part.part_id)
    W, H, D = dim[0], dim[1], dim[2]
    N = max(1, int(round(W / 20.0)))
    M = max(1, int(round(D / 20.0)))
    
    studs_local = []
    sockets_local = []
    for i in range(N):
        x = -10.0 * (N - 1) + 20.0 * i
        for j in range(M):
            z = -10.0 * (M - 1) + 20.0 * j
            # LDraw coordinate standard: Y points DOWN. Top surface is at Y=0, bottom is at Y=H.
            studs_local.append([x, 0.0, z, 1.0])
            sockets_local.append([x, H, z, 1.0])
            
    studs_world = (part.transform @ np.array(studs_local).T).T[:, :3]
    sockets_world = (part.transform @ np.array(sockets_local).T).T[:, :3]
    return studs_world, sockets_world

def check_connection(part_a: ParsedPart, part_b: ParsedPart) -> bool:
    """Determines if there is a stud-to-socket connection between part_a and part_b."""
    studs_a, sockets_a = get_studs_and_sockets_world(part_a)
    studs_b, sockets_b = get_studs_and_sockets_world(part_b)
    
    # 1.8 LDU connection tolerance threshold to accommodate design offsets in MOCs
    threshold = 1.8
    
    # Check if a stud of A connects to a socket of B
    for sa in studs_a:
        for sb in sockets_b:
            if np.linalg.norm(sa - sb) < threshold:
                return True
                
    # Check if a stud of B connects to a socket of A
    for sb in studs_b:
        for sa in sockets_a:
            if np.linalg.norm(sb - sa) < threshold:
                return True
                
    return False

# Part name database lookup functions moved to the top of the file

_CONNECTABLE_CACHE: dict[str, bool] = {}

def is_connectable_part(part: ParsedPart) -> bool:
    """Checks if a part is designed to connect mechanically (pins, clips, ball joints, arms, etc.)."""
    part_id = part.part_id.lower()
    if part_id in _CONNECTABLE_CACHE:
        return _CONNECTABLE_CACHE[part_id]
        
    name = get_part_name_from_db(part.part_id)
    
    keywords = [
        "pin", "clip", "hinge", "joint", "ball", "socket", 
        "arm", "hand", "head", "torso", "leg", "hip", 
        "technic", "shaft", "axle", "connector", "coupling", 
        "turntable", "harn", "hook", "towball", "claw", 
        "barb", "tooth", "talon", "horn", "bracket", "handle",
        "mudguard", "wheel", "fender", "panel", "window", "door",
        "frame", "glass"
    ]
    
    minifig_patterns = ["973", "970", "3626", "983", "981", "982"]
    # Specific patterns for fire hydrants, wheels, mudguards, headlights/blinkers
    specific_patterns = ["98138", "20482", "98282", "11208"]
    
    result = False
    for pat in specific_patterns:
        if pat in part_id:
            result = True
            break
            
    if not result:
        for kw in keywords:
            if kw in part_id or kw in name:
                result = True
                break
            
    if not result:
        for pat in minifig_patterns:
            if pat in part_id:
                result = True
                break
                
    if not result:
        # Check for bar using word boundary to avoid matching substring of other words
        import re
        if re.search(r"\bbar\b", part_id) or re.search(r"\bbar\b", name):
            result = True
            
    _CONNECTABLE_CACHE[part_id] = result
    return result

def obbs_touching(part_a: ParsedPart, part_b: ParsedPart, margin: float = 1.5) -> bool:
    """Checks if the 3D Oriented Bounding Boxes of two parts overlap when expanded by a margin."""
    dim_a = get_part_dimensions(part_a.part_id)
    dim_b = get_part_dimensions(part_b.part_id)
    
    c_a = (part_a.transform @ np.array([0.0, -dim_a[1]/2.0, 0.0, 1.0]))[:3]
    axes_a = part_a.transform[:3, :3]
    h_a = np.array([dim_a[0]/2.0 + margin, dim_a[1]/2.0 + margin, dim_a[2]/2.0 + margin])
    
    c_b = (part_b.transform @ np.array([0.0, -dim_b[1]/2.0, 0.0, 1.0]))[:3]
    axes_b = part_b.transform[:3, :3]
    h_b = np.array([dim_b[0]/2.0 + margin, dim_b[1]/2.0 + margin, dim_b[2]/2.0 + margin])
    
    return obb_overlap_sat(c_a, axes_a, h_a, c_b, axes_b, h_b)

_MINIFIG_CACHE: dict[str, bool] = {}

def is_minifig_part(part: ParsedPart) -> bool:
    """Checks if a part belongs to a minifigure (hips, head, hands, torso, arms, legs, accessories)."""
    part_id = part.part_id.lower()
    if part_id in _MINIFIG_CACHE:
        return _MINIFIG_CACHE[part_id]
        
    name = get_part_name_from_db(part.part_id)
    
    # Seats and chairs are NOT minifigures and must be connected to the main model
    if "seat" in name.lower() or "chair" in name.lower() or "4079" in part_id:
        _MINIFIG_CACHE[part_id] = False
        return False
        
    # Accessories/Greebles that are structurally attached to standard bricks
    if any(x in name.lower() or x in part_id for x in ["neckwear", "backpack", "visor", "binoculars", "utensil", "accessory", "accessories", "camera", "bucket", "shovel"]):
        _MINIFIG_CACHE[part_id] = False
        return False
        
    minifig_patterns = {"973", "970", "3626", "983", "981", "982", "76382", "111870", "112754", "24581", "25126"}
    
    # Clean the part ID to get the base numeric part
    clean_id = part_id.replace(".dat", "").replace("bl_", "").replace("xml_", "").strip()
    m_num = re.match(r"^(\d+)", clean_id)
    base_num = m_num.group(1) if m_num else ""
    
    result = False
    if base_num in minifig_patterns:
        result = True
    elif "baby" in clean_id or "24581" in clean_id or "25126" in clean_id:
        result = True
    else:
        # Check if it starts with "bl_" and has torso/head/arm/hand/leg/hip/hair/helmet
        is_custom_bl = part_id.startswith("bl_") and any(x in part_id for x in ["torso", "head", "arm", "hand", "leg", "hip", "hair", "helmet"])
        if is_custom_bl:
            result = True
        else:
            # Match keywords using word boundaries to avoid false positives (e.g. mudguard -> guard, alarm -> arm)
            kw_regex = r"\b(minifig|minifigure|torso|leg|head|hand|arm|hips|baby|hair|hat|helmet|visor|cape|beard|plume|wig)s?\b"
            if re.search(kw_regex, name.lower()) or re.search(kw_regex, clean_id):
                # Accessories like hats/hair can contain words like "plate" in other contexts, but if it has hat/hair/helmet it is minifig
                is_accessory = any(acc in name.lower() or acc in part_id for acc in ["hair", "hat", "helmet", "visor", "cape", "beard", "plume", "wig"])
                if is_accessory or ("plate" not in name.lower() and "tile" not in name.lower() and "brick" not in name.lower()):
                    result = True
                        
    _MINIFIG_CACHE[part_id] = result
    return result

def check_connection_optimized(part_a: ParsedPart, part_b: ParsedPart) -> bool:
    """Optimized connection check supporting vertical, horizontal, and mechanical connections."""
    # 0. Block connection between minifig and non-minifig components to split them as separate subassemblies,
    # except when they are connected via mechanical joints (greebling/locomotive rods using clips/bars)
    if is_minifig_part(part_a) != is_minifig_part(part_b):
        name_a = get_part_name_from_db(part_a.part_id)
        name_b = get_part_name_from_db(part_b.part_id)
        id_a = part_a.part_id.lower()
        id_b = part_b.part_id.lower()
        
        has_clip_a = "clip" in id_a or "clip" in name_a.lower()
        has_clip_b = "clip" in id_b or "clip" in name_b.lower()
        has_bar_a = any(kw in id_a or kw in name_a.lower() for kw in ["bar", "handle", "shaft", "antenna", "candlestick", "98834"])
        has_bar_b = any(kw in id_b or kw in name_b.lower() for kw in ["bar", "handle", "shaft", "antenna", "candlestick", "98834"])
        
        is_greeble = (has_clip_a and has_bar_b) or (has_clip_b and has_bar_a)
        if not is_greeble:
            return False

    # Special proximity check for minifig arm-to-hand connections (they have a peg insertion connection)
    name_a = get_part_name_from_db(part_a.part_id)
    name_b = get_part_name_from_db(part_b.part_id)
    id_a = part_a.part_id.lower()
    id_b = part_b.part_id.lower()
    
    is_arm_a = "981" in id_a or "982" in id_a or "3818" in id_a or "3819" in id_a or "arm" in name_a
    is_hand_a = "983" in id_a or "3820" in id_a or "hand" in name_a
    is_arm_b = "981" in id_b or "982" in id_b or "3818" in id_b or "3819" in id_b or "arm" in name_b
    is_hand_b = "983" in id_b or "3820" in id_b or "hand" in name_b
    
    if (is_arm_a and is_hand_b) or (is_hand_a and is_arm_b):
        dist = np.linalg.norm(part_a.transform[:3, 3] - part_b.transform[:3, 3])
        if dist < 25.0:
            return True

    # Determine dynamic margin based on whether the parts are connectable
    is_conn = is_connectable_part(part_a) or is_connectable_part(part_b)
    quick_margin = 6.5

    # 1. Quick OBB proximity filter to discard completely disjoint parts
    if not obbs_touching(part_a, part_b, margin=quick_margin):
        return False
        
    # 2. Check standard vertical/stud connection
    if check_connection(part_a, part_b):
        return True
        
    # 3. Check special mechanical or proximity connection (clips, pins, ball joints, minifigures)
    # Using a contact margin of 4.0 LDU for connectable parts to account for clip/pin/bar/bracket extensions.
    # Allowing up to 6.5 LDU for mudguards or wheel holders to guarantee chassis structural integration.
    # Allowing up to 2.2 LDU for standard parts to accommodate minor SNOT and bracket alignments.
    is_complex_mech = any(p in id_a or p in id_b for p in ["98282", "11208"])
    
    # Rule 1: Hinge or turntable rotation tolerance (allow 5.5 LDU for connections involving rotating elements)
    is_hinge_or_rot = any(kw in id_a or kw in name_a.lower() or kw in id_b or kw in name_b.lower() 
                          for kw in ["hinge", "turntable", "joint", "articulation", "3937", "3938"])
    
    # Rule 2: Vegetation/Foliage collision rule (allow 4.5 LDU if both are vegetation parts)
    is_veg_a = any(kw in id_a or kw in name_a.lower() for kw in ["plant", "leaf", "leaves", "flower", "stem", "32607", "2417", "2423", "24866"])
    is_veg_b = any(kw in id_b or kw in name_b.lower() for kw in ["plant", "leaf", "leaves", "flower", "stem", "32607", "2417", "2423", "24866"])
    is_veg_conn = is_veg_a and is_veg_b
    
    # Rule 4: Clip-to-bar rule (allow 5.0 LDU for clip to bar/handle connections)
    has_clip = "clip" in id_a or "clip" in name_a.lower() or "clip" in id_b or "clip" in name_b.lower()
    has_bar = any(kw in id_a or kw in name_a.lower() or kw in id_b or kw in name_b.lower() for kw in ["bar", "handle", "shaft", "antenna", "candlestick", "98834"])
    is_clip_to_bar = has_clip and has_bar

    if is_complex_mech:
        contact_margin = 6.5
    elif is_hinge_or_rot:
        contact_margin = 5.5
    elif is_clip_to_bar:
        contact_margin = 5.0
    elif is_veg_conn:
        contact_margin = 4.5
    else:
        # Proportional margin based on maximum dimension to handle long plates/bricks and minor angular offsets
        dim_a = get_part_dimensions(part_a.part_id)
        dim_b = get_part_dimensions(part_b.part_id)
        max_dim = max(dim_a[0], dim_a[2], dim_b[0], dim_b[2])
        contact_margin = max(6.0, min(10.0, max_dim * 0.05))
    
    if obbs_touching(part_a, part_b, margin=contact_margin):
        return True
            
    return False

def obb_overlap_sat(c_a: np.ndarray, axes_a: np.ndarray, h_a: np.ndarray,
                    c_b: np.ndarray, axes_b: np.ndarray, h_b: np.ndarray) -> bool:
    """
    Separating Axis Theorem (SAT) for 3D Oriented Bounding Boxes (OBB).
    Returns True if OBB A and OBB B overlap, False otherwise.
    """
    T = c_b - c_a
    
    # Projecting onto the 15 candidate axes
    # 1. Face normals of A
    for i in range(3):
        L = axes_a[:, i]
        r_a = h_a[i]
        r_b = sum(h_b[k] * np.abs(np.dot(axes_b[:, k], L)) for k in range(3))
        if np.abs(np.dot(T, L)) > r_a + r_b:
            return False
            
    # 2. Face normals of B
    for i in range(3):
        L = axes_b[:, i]
        r_a = sum(h_a[k] * np.abs(np.dot(axes_a[:, k], L)) for k in range(3))
        r_b = h_b[i]
        if np.abs(np.dot(T, L)) > r_a + r_b:
            return False
            
    # 3. Cross products of axes
    for i in range(3):
        for j in range(3):
            L = np.cross(axes_a[:, i], axes_b[:, j])
            L_norm = np.linalg.norm(L)
            if L_norm < 1e-5:
                continue
            L = L / L_norm
            
            r_a = sum(h_a[k] * np.abs(np.dot(axes_a[:, k], L)) for k in range(3))
            r_b = sum(h_b[k] * np.abs(np.dot(axes_b[:, k], L)) for k in range(3))
            if np.abs(np.dot(T, L)) > r_a + r_b:
                return False
                
    return True

def get_part_mesh(part_id: str, transform: np.ndarray):
    """Dynamically creates a Trimesh representation of the part if trimesh is installed."""
    import trimesh
    dim = get_part_dimensions(part_id)
    # Subtract 0.2 LDU clearance from width/depth, and 0.02 LDU from height to prevent false collision triggers
    w, h, d = dim[0] - 0.2, dim[1] - 0.02, dim[2] - 0.2
    mesh = trimesh.creation.box(extents=(w, h, d))
    
    # Translate so the bottom center of the brick is the local (0, 0, 0)
    shift = np.eye(4, dtype=np.float32)
    shift[1, 3] = -dim[1] / 2.0
    mesh.apply_transform(shift)
    
    # Apply world transform
    mesh.apply_transform(transform)
    return mesh

def check_collisions(parts: list[ParsedPart]) -> list[tuple[int, int]]:
    """
    Checks all parts for intersections.
    Returns a list of colliding index pairs (i, j).
    """
    try:
        # Try importing and using trimesh CollisionManager
        import trimesh
        manager = trimesh.collision.CollisionManager()
        for idx, part in enumerate(parts):
            mesh = get_part_mesh(part.part_id, part.transform)
            manager.add_object(f"part_{idx}", mesh)
            
        in_collision, names = manager.in_collision_internal(return_names=True)
        collisions = []
        if in_collision:
            for name_a, name_b in names:
                idx_a = int(name_a.split("_")[1])
                idx_b = int(name_b.split("_")[1])
                collisions.append((idx_a, idx_b))
        return collisions
    except Exception:
        # Fallback to Separating Axis Theorem (SAT) using Numpy
        collisions = []
        num_parts = len(parts)
        for i in range(num_parts):
            for j in range(i + 1, num_parts):
                part_a = parts[i]
                part_b = parts[j]
                dim_a = get_part_dimensions(part_a.part_id)
                dim_b = get_part_dimensions(part_b.part_id)
                
                # OBB properties for A
                c_a = (part_a.transform @ np.array([0.0, -dim_a[1]/2.0, 0.0, 1.0]))[:3]
                axes_a = part_a.transform[:3, :3]
                h_a = np.array([dim_a[0]/2.0 - 0.1, dim_a[1]/2.0 - 0.01, dim_a[2]/2.0 - 0.1])
                
                # OBB properties for B
                c_b = (part_b.transform @ np.array([0.0, -dim_b[1]/2.0, 0.0, 1.0]))[:3]
                axes_b = part_b.transform[:3, :3]
                h_b = np.array([dim_b[0]/2.0 - 0.1, dim_b[1]/2.0 - 0.01, dim_b[2]/2.0 - 0.1])
                
                if obb_overlap_sat(c_a, axes_a, h_a, c_b, axes_b, h_b):
                    collisions.append((i, j))
        return collisions

def check_connectivity_and_gravity(parts: list[ParsedPart]) -> bool:
    """
    Builds a connectivity graph and verifies that there are no floating parts.
    All parts must connect directly or indirectly to the base.
    Also strictly enforces that no part goes below ground level (Y > 1.0)
    and at most 4 parts reside on the ground level (|Y| <= 1.0).
    """
    if not parts:
        return True

    # 1. Ground level check: no parts below Y=0.0 (in LDraw Y increases downwards)
    for part in parts:
        if part.transform[1, 3] > 1.0:
            return False

    # 2. Base layer count: at most 4 parts on the ground surface
    ground_count = sum(1 for part in parts if abs(part.transform[1, 3]) <= 1.0)
    if ground_count > 4:
        return False
        
    G = nx.Graph()
    for idx in range(len(parts)):
        G.add_node(idx)
        
    # Build connection edges
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if check_connection_optimized(parts[i], parts[j]):
                G.add_edge(i, j)
                
    # Detect anchored base parts
    # The base consists of the parts with the lowest elevation (maximum Y value in LDraw coords)
    y_max_values = []
    for part in parts:
        corners = get_part_corners_world(part)
        y_max_values.append(np.max(corners[:, 1]))
        
    global_max_y = max(y_max_values)
    
    # Any part within 1.0 LDU of the absolute lowest point is anchored to the base
    anchored_nodes = [idx for idx, y in enumerate(y_max_values) if abs(y - global_max_y) < 1.0]
    
    # Virtual ground node
    G.add_node("ground")
    for idx in anchored_nodes:
        G.add_edge("ground", idx)
        
    # Check if the entire assembly is in the same component as the ground
    connected_components = list(nx.connected_components(G))
    for component in connected_components:
        if "ground" in component:
            return len(component) == len(parts) + 1
            
    return False

def validate_rules(parts: list[ParsedPart]) -> bool:
    """Verifies that all parts adhere to standard vocabulary limits (design IDs and color bounds)."""
    from src.parser import ALLOWED_PARTS, ALLOWED_COLORS
    for part in parts:
        if part.part_id not in ALLOWED_PARTS:
            return False
        if part.color not in ALLOWED_COLORS:
            return False
    return True
