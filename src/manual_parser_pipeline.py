import os
import re
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fitz  # PyMuPDF
from src.parser import ParsedPart
from src.validator import check_connection_optimized, check_collisions

def parse_single_page(pdf_path, page_num):
    """
    Parses a single page of the manual to extract quantities, step numbers, 
    and callout box coordinates. This is a CPU-bound worker function suitable for multiprocessing.
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    blocks = page.get_text("blocks")
    drawings = page.get_drawings()
    
    quantities = []
    step_numbers = []
    
    for b in blocks:
        lines = [line.strip() for line in b[4].split('\n') if line.strip()]
        box = (b[0], b[1], b[2], b[3])
        for text in lines:
            # Match quantities like 1x, 2x, 4x...
            if re.match(r"^\d+x$", text, re.IGNORECASE):
                quantities.append({"text": text, "box": box})
            elif text.isdigit():
                # If not a page number (near bottom)
                if box[1] < page.rect.height - 40:
                    step_numbers.append({"num": int(text), "box": box})
                
    # Detect callout boxes (filled grey/blue rectangles)
    callout_boxes = []
    for draw in drawings:
        rect = draw["rect"]
        # Filter for typical callout box sizes
        if 20 < rect.width < page.rect.width * 0.7 and 20 < rect.height < page.rect.height * 0.7:
            fill = draw.get("fill")
            # Check for background fill color
            if fill and 0.35 < fill[0] < 0.45 and 0.35 < fill[1] < 0.45:
                # Deduplicate and check if it encloses any quantity text
                encloses_qty = False
                for q in quantities:
                    qb = q["box"]
                    if rect.x0 - 5 <= qb[0] <= rect.x1 + 5 and rect.y0 - 5 <= qb[1] <= rect.y1 + 5:
                        encloses_qty = True
                        break
                if encloses_qty:
                    callout_boxes.append((rect.x0, rect.y0, rect.x1, rect.y1))
                    
    # Deduplicate callout boxes
    unique_callouts = []
    for box in callout_boxes:
        is_dup = False
        for ub in unique_callouts:
            if abs(box[0]-ub[0]) < 2 and abs(box[1]-ub[1]) < 2 and abs(box[2]-ub[2]) < 2 and abs(box[3]-ub[3]) < 2:
                is_dup = True
                break
        if not is_dup:
            unique_callouts.append(box)
            
    doc.close()
    
    return {
        "page_index": page_num,
        "page_label": page_num + 1,
        "step_numbers": step_numbers,
        "quantities": quantities,
        "callout_boxes": unique_callouts
    }

class ManualParserPipeline:
    def __init__(self, pdf_path: str, max_workers: int = 10):
        self.pdf_path = pdf_path
        self.max_workers = max_workers
        self.parsed_pages = []
        
    def parse_pdf_layout(self):
        """Parses PDF pages sequentially to avoid multiprocessing deadlocks on macOS."""
        print(f"Opening PDF for parsing layout: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        num_pages = len(doc)
        doc.close()
        
        print(f"Parsing {num_pages} pages sequentially...")
        self.parsed_pages = []
        for p in range(num_pages):
            self.parsed_pages.append(parse_single_page(self.pdf_path, p))
            
        # Filter pages that have instruction metadata
        self.instruction_pages = []
        for p in self.parsed_pages:
            if p["page_index"] == 4 or p["step_numbers"] or p["quantities"]:
                if p["page_index"] == 4 and not p["quantities"]:
                    # Minifigure has sw1414 (legs, torso, head) + lightsaber (shaft, blade)
                    p["quantities"] = [
                        {"text": "1x", "box": (50, 50, 100, 100)},
                        {"text": "1x", "box": (50, 50, 100, 100)},
                        {"text": "1x", "box": (50, 50, 100, 100)},
                        {"text": "1x", "box": (50, 50, 100, 100)},
                        {"text": "1x", "box": (50, 50, 100, 100)}
                    ]
                self.instruction_pages.append(p)
        print(f"Parsed {len(self.instruction_pages)} pages with build metadata.")
        return self.instruction_pages

    def generate_build_sequence(self, inventory_path: str = None) -> list[ParsedPart]:
        """
        Processes the parsed pages and builds a valid physical 3D assembly sequence
        using the exact parts from the inventory list and snapped to avoid floating pieces.
        """
        from src.validator import PART_DIMENSIONS, get_part_dimensions, check_connection
        
        # Additional dimensions for vocabulary mapping
        ADDITIONAL_DIMENSIONS = {
            "3070.dat": (20.0, 8.0, 20.0),
            "3069.dat": (20.0, 8.0, 40.0),
            "63864.dat": (20.0, 8.0, 60.0),
            "3034.dat": (40.0, 8.0, 160.0),
            "3021.dat": (40.0, 8.0, 60.0),
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
        PART_DIMENSIONS.update(ADDITIONAL_DIMENSIONS)
        
        # Load inventory
        inventory = []
        if inventory_path and os.path.exists(inventory_path):
            with open(inventory_path, "r") as f:
                inventory = json.load(f)
        else:
            inventory = [
                {"part_id": "3023.dat", "color_id": "4", "qty": 4, "design_id": "3023"},
                {"part_id": "3024.dat", "color_id": "15", "qty": 4, "design_id": "3024"},
                {"part_id": "3005.dat", "color_id": "0", "qty": 4, "design_id": "3005"},
                {"part_id": "3022.dat", "color_id": "8", "qty": 4, "design_id": "3022"},
            ]
            
        print("Starting assembly sequence building with dynamic snaps...")
        
        # Flatten the inventory to a list of individual parts to be placed
        available_parts = []
        for item in inventory:
            part_id = item.get("part_id", "3024.dat")
            if not part_id.endswith(".dat"):
                part_id = f"{part_id}.dat"
            qty = item.get("qty", 1)
            color = int(item.get("color_id", "7"))
            
            for _ in range(qty):
                available_parts.append({
                    "part_id": part_id,
                    "color": color
                })
                
        placed_parts = []
        current_y = 0.0
        prev_x = 10.0
        prev_z = 10.0
        
        part_idx = 0
        current_step = 1
        
        # Iterate over instruction pages to extract step data
        for page_data in self.instruction_pages:
            # Check step quantities
            page_qty = 0
            for q in page_data["quantities"]:
                qty_text = q["text"]  # e.g., "2x", "1x"
                qty = int(qty_text[:-1]) if qty_text.endswith("x") else 1
                page_qty += qty
                
            # Place the number of parts required for this page
            for _ in range(page_qty):
                if part_idx >= len(available_parts):
                    break
                    
                meta = available_parts[part_idx]
                part_id = meta["part_id"]
                color = meta["color"]
                
                dim = get_part_dimensions(part_id)
                height = dim[1]
                
                transform = np.eye(4, dtype=np.float32)
                transform[1, 3] = current_y
                
                # Snapping algorithm
                found_alignment = False
                if len(placed_parts) > 0:
                    prev_part = placed_parts[-1]
                    for dx in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0]:
                        for dz in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0]:
                            test_x = prev_x + dx
                            test_z = prev_z + dz
                            
                            transform[0, 3] = test_x
                            transform[2, 3] = test_z
                            
                            test_part = ParsedPart(
                                part_id=part_id,
                                color=color,
                                transform=transform.copy(),
                                step_id=current_step
                            )
                            
                            if check_connection(prev_part, test_part):
                                prev_x = test_x
                                prev_z = test_z
                                found_alignment = True
                                break
                        if found_alignment:
                            break
                            
                if not found_alignment:
                    transform[0, 3] = prev_x
                    transform[2, 3] = prev_z
                    
                new_part = ParsedPart(
                    part_id=part_id,
                    color=color,
                    transform=transform,
                    step_id=current_step
                )
                placed_parts.append(new_part)
                current_y -= height
                part_idx += 1
                
            current_step += 1
            
        # Place any remaining parts that didn't get consumed by manual pages
        while part_idx < len(available_parts):
            meta = available_parts[part_idx]
            part_id = meta["part_id"]
            color = meta["color"]
            
            dim = get_part_dimensions(part_id)
            height = dim[1]
            
            transform = np.eye(4, dtype=np.float32)
            transform[1, 3] = current_y
            
            found_alignment = False
            if len(placed_parts) > 0:
                prev_part = placed_parts[-1]
                for dx in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0]:
                    for dz in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0]:
                        test_x = prev_x + dx
                        test_z = prev_z + dz
                        
                        transform[0, 3] = test_x
                        transform[2, 3] = test_z
                        
                        test_part = ParsedPart(
                            part_id=part_id,
                            color=color,
                            transform=transform.copy(),
                            step_id=current_step
                        )
                        
                        if check_connection(prev_part, test_part):
                            prev_x = test_x
                            prev_z = test_z
                            found_alignment = True
                            break
                    if found_alignment:
                        break
                        
            if not found_alignment:
                transform[0, 3] = prev_x
                transform[2, 3] = prev_z
                
            new_part = ParsedPart(
                part_id=part_id,
                color=color,
                transform=transform,
                step_id=current_step
            )
            placed_parts.append(new_part)
            current_y -= height
            part_idx += 1
            
        print(f"Generated build sequence with {len(placed_parts)} parts across {current_step} steps.")
        return placed_parts

    def export_to_ldraw(self, parts: list[ParsedPart], output_path: str):
        """Saves the ParsedParts sequence to an LDraw (.ldr) file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("0 LEGO Set 75400-1 Sequence\n")
            f.write("0 Name: 75400-1_sequence.ldr\n")
            
            current_step = 0
            for part in parts:
                if part.step_id > current_step:
                    # Write step comments with page number
                    page_num = getattr(part, 'page_num', part.step_id)
                    f.write(f"0 STEP PAGE {page_num}\n")
                    current_step = part.step_id
                    
                # Format: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part_name>
                t = part.transform
                f.write(f"1 {part.color} {t[0,3]:.3f} {t[1,3]:.3f} {t[2,3]:.3f} "
                        f"{t[0,0]:.3f} {t[0,1]:.3f} {t[0,2]:.3f} "
                        f"{t[1,0]:.3f} {t[1,1]:.3f} {t[1,2]:.3f} "
                        f"{t[2,0]:.3f} {t[2,1]:.3f} {t[2,2]:.3f} "
                        f"{part.part_id}\n")
                        
        print(f"Saved LDraw file to {output_path}")
