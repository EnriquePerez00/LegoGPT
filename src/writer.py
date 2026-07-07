from src.parser import ParsedPart

def write_ldraw_file(parts: list[ParsedPart], output_path: str) -> None:
    """
    Writes a list of ParsedParts to an LDraw-compliant file (.ldr/.mpd).
    Organizes output sequentially by step_id using the '0 STEP' command.
    
    Args:
        parts: List of ParsedPart objects to export.
        output_path: Target file path to write to.
    """
    # Sort parts by step_id to ensure order is preserved
    sorted_parts = sorted(parts, key=lambda p: p.step_id)
    
    with open(output_path, "w", encoding="utf-8") as file_out:
        file_out.write("0 LegoGPT Generated Model\n")
        
        current_step = 0
        for part in sorted_parts:
            # If the part belongs to a later step, insert STEP boundaries
            while part.step_id > current_step:
                file_out.write("0 STEP\n")
                current_step += 1
                
            x = part.transform[0, 3]
            y = part.transform[1, 3]
            z = part.transform[2, 3]
            
            a = part.transform[0, 0]
            b = part.transform[0, 1]
            c = part.transform[0, 2]
            d = part.transform[1, 0]
            e = part.transform[1, 1]
            f = part.transform[1, 2]
            g = part.transform[2, 0]
            h = part.transform[2, 1]
            i = part.transform[2, 2]
            
            # Format: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part_name>
            line = f"1 {part.color} {x} {y} {z} {a} {b} {c} {d} {e} {f} {g} {h} {i} {part.part_id}\n"
            file_out.write(line)
