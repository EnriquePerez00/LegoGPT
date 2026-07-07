import os
import numpy as np
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS, ParsedPart
from src.validator import get_part_dimensions, get_studs_and_sockets_world, obb_overlap_sat, check_connectivity_and_gravity
from src.writer import write_ldraw_file

def generate_single_synthetic_model(num_pieces: int) -> list[ParsedPart]:
    """
    Generates a single stable, collision-free LEGO assembly using only
    the 8 allowed parts.
    """
    # 1. Place base piece at the origin
    base_part_id = np.random.choice(ALLOWED_PARTS)
    base_color = int(np.random.choice(ALLOWED_COLORS))
    base_transform = np.eye(4, dtype=np.float32)
    
    parts = [ParsedPart(part_id=base_part_id, color=base_color, transform=base_transform, step_id=0)]
    
    def overlaps(candidate: ParsedPart, existing_list: list[ParsedPart]) -> bool:
        for p in existing_list:
            dim_a = get_part_dimensions(candidate.part_id)
            dim_b = get_part_dimensions(p.part_id)
            
            c_a = (candidate.transform @ np.array([0.0, -dim_a[1]/2.0, 0.0, 1.0]))[:3]
            axes_a = candidate.transform[:3, :3]
            h_a = np.array([dim_a[0]/2.0 - 0.1, dim_a[1]/2.0 - 0.01, dim_a[2]/2.0 - 0.1])
            
            c_b = (p.transform @ np.array([0.0, -dim_b[1]/2.0, 0.0, 1.0]))[:3]
            axes_b = p.transform[:3, :3]
            h_b = np.array([dim_b[0]/2.0 - 0.1, dim_b[1]/2.0 - 0.01, dim_b[2]/2.0 - 0.1])
            
            if obb_overlap_sat(c_a, axes_a, h_a, c_b, axes_b, h_b):
                return True
        return False

    # 2. Add remaining pieces sequentially
    for step in range(1, num_pieces):
        placed = False
        # Try different part/color combinations
        for attempt in range(50):
            cand_part_id = np.random.choice(ALLOWED_PARTS)
            cand_color = int(np.random.choice(ALLOWED_COLORS))
            
            dummy = ParsedPart(part_id=cand_part_id, color=cand_color, transform=np.eye(4, dtype=np.float32), step_id=step)
            cand_studs, cand_sockets = get_studs_and_sockets_world(dummy)
            if len(cand_studs) == 0 and len(cand_sockets) == 0:
                continue
                
            # Pick a random existing part to connect to
            parent = np.random.choice(parts)
            parent_studs, parent_sockets = get_studs_and_sockets_world(parent)
            if len(parent_studs) == 0 and len(parent_sockets) == 0:
                continue
                
            # Choose connection direction: socket of candidate -> stud of parent (0)
            # or stud of candidate -> socket of parent (1)
            conn_mode = np.random.choice([0, 1])
            if conn_mode == 0 and len(parent_studs) > 0 and len(cand_sockets) > 0:
                p_anchor = parent_studs[np.random.choice(len(parent_studs))]
                c_anchor = cand_sockets[np.random.choice(len(cand_sockets))]
            elif conn_mode == 1 and len(parent_sockets) > 0 and len(cand_studs) > 0:
                p_anchor = parent_sockets[np.random.choice(len(parent_sockets))]
                c_anchor = cand_studs[np.random.choice(len(cand_studs))]
            else:
                continue
                
            # Rotate randomly around Y-axis
            theta = np.random.choice([0, 90, 180, 270])
            rad = np.radians(theta)
            c, s = int(round(np.cos(rad))), int(round(np.sin(rad)))
            R = np.array([
                [c, 0, s],
                [0, 1, 0],
                [-s, 0, c]
            ], dtype=np.float32)
            
            T = p_anchor - R @ c_anchor
            
            t_matrix = np.eye(4, dtype=np.float32)
            t_matrix[:3, :3] = R
            t_matrix[:3, 3] = T
            
            candidate = ParsedPart(part_id=cand_part_id, color=cand_color, transform=t_matrix, step_id=step)
            
            # Check collisions & connectivity
            if not overlaps(candidate, parts):
                test_list = parts + [candidate]
                if check_connectivity_and_gravity(test_list):
                    parts.append(candidate)
                    placed = True
                    break
                    
        if not placed:
            # If we couldn't place any piece, stop generating and return what we have
            break
            
    return parts

def generate_and_save_synthetic_dataset(num_models: int = 500, min_size: int = 2, max_size: int = 6, output_dir: str = "data/omr_raw"):

    """
    Generates a collection of synthetic LDraw models and saves them to the raw folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generando {num_models} modelos sintéticos en {output_dir}...")
    
    generated_count = 0
    while generated_count < num_models:
        size = 8
        parts = generate_single_synthetic_model(size)
        if len(parts) == size:
            file_name = f"9999-{generated_count:03d}_Synthetic.ldr"
            file_path = os.path.join(output_dir, file_name)
            write_ldraw_file(parts, file_path)
            generated_count += 1
            if generated_count % 100 == 0:
                print(f"  Generados {generated_count}/{num_models}...")
                
    print(f"Dataset sintético completado. {generated_count} modelos guardados con éxito.")

if __name__ == "__main__":
    generate_and_save_synthetic_dataset()
