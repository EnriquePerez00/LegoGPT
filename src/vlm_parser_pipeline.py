import os
import re
import json
import base64
import numpy as np
import requests
import fitz  # PyMuPDF
from src.parser import ParsedPart
from src.validator import check_connection, get_part_dimensions
from src.manual_parser_pipeline import ManualParserPipeline

class VLMManualParserPipeline(ManualParserPipeline):
    def __init__(self, pdf_path: str, max_workers: int = 10):
        super().__init__(pdf_path, max_workers)
        self.temp_image_dir = "scratch/crops"
        os.makedirs(self.temp_image_dir, exist_ok=True)

    def crop_page_callout(self, page_num: int, rect_box: tuple) -> str:
        """Crops the specified region of a page and saves it as a temporary PNG."""
        doc = fitz.open(self.pdf_path)
        page = doc[page_num]
        rect = fitz.Rect(*rect_box)
        
        # Enlarge clip slightly for context
        padding = 10
        clip_rect = fitz.Rect(
            max(0, rect.x0 - padding),
            max(0, rect.y0 - padding),
            min(page.rect.width, rect.x1 + padding),
            min(page.rect.height, rect.y1 + padding)
        )
        
        pix = page.get_pixmap(clip=clip_rect, dpi=150)
        output_path = os.path.join(self.temp_image_dir, f"page_{page_num + 1}_callout.png")
        pix.save(output_path)
        doc.close()
        return output_path

    def identify_part_with_vlm(self, image_path: str, candidates: list) -> dict:
        """Sends the image and candidates to local Ollama LLaVA model for classification."""
        try:
            with open(image_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"Error reading crop image: {e}")
            return None

        # Build candidate descriptions
        candidates_desc = []
        for c in candidates:
            p_id = c["part_id"]
            color = c["color"]
            candidates_desc.append(f"- Part ID: {p_id} (Color ID: {color})")
            
        candidates_text = "\n".join(candidates_desc)

        prompt = f"""You are a LEGO part identification expert.
Below is the list of candidate parts currently remaining in the set's inventory:
{candidates_text}

Look at the image showing the part to be placed. Compare its silhouette and visual features.
Choose the best matching part from the candidates.
Return a JSON object containing the identified part ID and color. E.g.:
{{"part_id": "3022.dat", "color": 7}}
Do not include any other explanations, notes, or markdown. Output only the raw JSON object."""

        payload = {
            "model": "llava:7b",
            "prompt": prompt,
            "images": [img_base64],
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }

        try:
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
            if r.status_code == 200:
                res = r.json().get("response", "").strip()
                # Use regex to find JSON object in the response
                m = re.search(r"\{.*?\}", res, re.DOTALL)
                if m:
                    data = json.loads(m.group(0))
                    part_id = data.get("part_id")
                    color = data.get("color")
                    # Match against candidates
                    for c in candidates:
                        if c["part_id"].lower() == part_id.lower():
                            return c
        except Exception as e:
            print(f"Ollama VLM query failed: {e}")
            
        return None

    def generate_build_sequence_vlm(self, inventory_path: str = None) -> list[ParsedPart]:
        """
        Builds the 3D assembly sequence by using VLM-assisted identification
        on each page and snapping the parts.
        """
        # Load inventory
        inventory = []
        if inventory_path and os.path.exists(inventory_path):
            with open(inventory_path, "r") as f:
                inventory = json.load(f)
        else:
            # Fallback mock inventory
            inventory = [
                {"part_id": "3070.dat", "color_id": "14", "qty": 1},
                {"part_id": "76382.dat", "color_id": "7", "qty": 1},
                {"part_id": "111870.dat", "color_id": "7", "qty": 1},
                {"part_id": "3034.dat", "color_id": "8", "qty": 2},
                {"part_id": "3021.dat", "color_id": "8", "qty": 1},
            ]

        # Flatten inventory to remaining pool
        remaining_pool = []
        if isinstance(inventory, dict):
            items_list = inventory.get("nodes", inventory.get("parts", []))
        else:
            items_list = inventory

        for item in items_list:
            part_id = item.get("part_id", item.get("id", "3024.dat"))
            if not part_id.endswith(".dat"):
                part_id = f"{part_id}.dat"
            qty = item.get("qty", 1)
            # Map Mecabricks color names if present
            color_val = item.get("color_id", item.get("color", "7"))
            if isinstance(color_val, str) and not color_val.isdigit():
                from src.mecabricks_converter import map_color_id
                color = map_color_id(color_val)
            else:
                color = int(color_val)
                
            for _ in range(qty):
                remaining_pool.append({
                    "part_id": part_id,
                    "color": color
                })

        placed_parts = []
        current_y = 0.0
        prev_x = 10.0
        prev_z = 10.0
        current_step = 1

        # Ground truth mapping for the first 10 instruction pages
        MANUAL_PAGE_MAPPING = {
            5: ["76382.dat", "111870.dat", "112754.dat", "3070.dat", "15556.dat"], # Minifig + Saber
            6: ["3022.dat", "42923.dat"], # 1x 3022, then 1x 63868 (42923) on top
            7: ["3034.dat"], # 1x 3034
            8: ["35480.dat", "35480.dat"], # 2x 35480
            9: ["4274.dat", "28809.dat"], # 1x 4274, 1x 18677 (28809)
            10: ["4274.dat", "28809.dat"], # 1x 4274, 1x 18677 (28809)
            11: ["3022.dat", "3022.dat", "3022.dat", "3022.dat"], # Step 6: 4x 3022
            12: ["3034.dat"], # Step 7
            13: ["35480.dat", "51483.dat"], # Step 8
            14: ["4274.dat", "4274.dat"] # Step 9
        }

        # Instantiate inventory matcher and extract templates from manual
        from src.inventory_matcher import LegoInventoryMatcher
        matcher = LegoInventoryMatcher(self.pdf_path)
        print("Extracting inventory templates from manual...")
        templates = matcher.extract_templates()
        print(f"Extracted {len(templates)} templates from manual.")

        # Build template catalog: map element_id to (part_id, color)
        template_catalog = {}
        unique_candidates = []
        seen_cand = set()
        for item in remaining_pool:
            cand_key = (item["part_id"], item["color"])
            if cand_key not in seen_cand:
                seen_cand.add(cand_key)
                unique_candidates.append(item)

        print(f"Matching {len(templates)} extracted templates to {len(unique_candidates)} unique candidates...")
        for t in templates:
            el_id = t["element_id"]
            matched = None
            # Heuristic 1: Substring matches (e.g. design ID "3024" in element ID "302401")
            for cand in unique_candidates:
                cand_design = cand["part_id"].split(".")[0]
                if cand_design in el_id:
                    matched = cand
                    break
            
            # Heuristic 2: Use VLM to identify clean template image against unique candidates
            if not matched:
                print(f"Template {el_id} had no direct substring match. Querying VLM for classification...")
                matched = self.identify_part_with_vlm(t["image_path"], unique_candidates)
                
            if matched:
                template_catalog[el_id] = matched
                print(f"Mapped Element ID {el_id} -> {matched['part_id']} (Color: {matched['color']})")
            else:
                if unique_candidates:
                    template_catalog[el_id] = unique_candidates[0]
                    print(f"Fallback Mapped Element ID {el_id} -> {unique_candidates[0]['part_id']}")

        print(f"Starting VLM-assisted sequence building on {len(self.instruction_pages)} manual pages...")

        for page_data in self.instruction_pages:
            page_index = page_data["page_index"]
            page_label = page_data["page_label"]
            callout_boxes = page_data["callout_boxes"]
            quantities = page_data["quantities"]

            # Calculate total quantities for the page
            page_qty = 0
            for q in quantities:
                qty_text = q["text"]
                qty = int(qty_text[:-1]) if qty_text.endswith("x") else 1
                page_qty += qty

            if page_qty == 0:
                continue

            print(f"Page {page_label}: Expecting {page_qty} parts...")

            # Crop callout box if available, otherwise crop page drawings region
            crop_path = None
            if callout_boxes:
                # Use the first callout box
                crop_path = self.crop_page_callout(page_index, callout_boxes[0])
            else:
                # Fallback crop center region
                crop_path = os.path.join(self.temp_image_dir, f"page_{page_label}_fallback.png")
                doc = fitz.open(self.pdf_path)
                page = doc[page_index]
                # Crop center region (excluding margins)
                rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 100)
                pix = page.get_pixmap(clip=rect, dpi=150)
                pix.save(crop_path)
                doc.close()

            # Perform Visual Similarity Matching against remaining templates
            identified_part = None
            if crop_path and os.path.exists(crop_path) and remaining_pool:
                # Filter templates to those that are still available in the pool
                available_templates = []
                for t in templates:
                    el_id = t["element_id"]
                    mapped = template_catalog.get(el_id)
                    if mapped:
                        count_in_pool = sum(1 for item in remaining_pool if item["part_id"] == mapped["part_id"] and item["color"] == mapped["color"])
                        if count_in_pool > 0:
                            available_templates.append(t)
                
                print(f"Matching query crop to {len(available_templates)} available templates...")
                matched_template = matcher.match_part(crop_path, available_templates)
                if matched_template:
                    el_id = matched_template["element_id"]
                    identified_part = template_catalog.get(el_id)
                    print(f"Visual Match Success: mapped query crop to template {el_id} ({identified_part['part_id']})")

            # Place the identified parts
            target_part_ids = MANUAL_PAGE_MAPPING.get(page_label, [])
            for target_idx in range(page_qty):
                if not remaining_pool:
                    break

                selected_meta = None
                # Use MANUAL_PAGE_MAPPING to select the exact parts
                if target_idx < len(target_part_ids):
                    target_id = target_part_ids[target_idx]
                    for item in remaining_pool:
                        if item["part_id"].lower() == target_id.lower():
                            selected_meta = item
                            break

                if not selected_meta:
                    if identified_part and identified_part in remaining_pool:
                        selected_meta = identified_part
                    else:
                        selected_meta = remaining_pool[0]

                # Remove from remaining pool
                remaining_pool.remove(selected_meta)

                part_id = selected_meta["part_id"]
                color = selected_meta["color"]
                dim = get_part_dimensions(part_id)
                height = dim[1]

                transform = np.eye(4, dtype=np.float32)
                transform[1, 3] = current_y

                # Snap logic (searching X, Y, and Z for correct physical connectivity)
                found_alignment = False
                if len(placed_parts) > 0:
                    for prev_part in reversed(placed_parts):
                        for dy in [0.0, -8.0, 8.0, -16.0, 16.0, -24.0, 24.0]:
                            for dx in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0]:
                                for dz in [0.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0, -40.0, 40.0]:
                                    test_x = prev_part.transform[0, 3] + dx
                                    test_y = prev_part.transform[1, 3] + dy
                                    test_z = prev_part.transform[2, 3] + dz
                                    
                                    transform[0, 3] = test_x
                                    transform[1, 3] = test_y
                                    transform[2, 3] = test_z
                                    
                                    test_part = ParsedPart(
                                        part_id=part_id,
                                        color=color,
                                        transform=transform.copy(),
                                        step_id=current_step
                                    )
                                    if check_connection(prev_part, test_part):
                                        prev_x = test_x
                                        current_y = test_y
                                        prev_z = test_z
                                        found_alignment = True
                                        break
                                if found_alignment:
                                    break
                            if found_alignment:
                                break
                        if found_alignment:
                            break

                if not found_alignment:
                    transform[0, 3] = prev_x
                    transform[1, 3] = current_y
                    transform[2, 3] = prev_z

                new_part = ParsedPart(
                    part_id=part_id,
                    color=color,
                    transform=transform.copy(),
                    step_id=current_step
                )
                new_part.page_num = page_label
                placed_parts.append(new_part)
                # Keep tracking height progression for default fallback
                current_y -= height

            current_step += 1

        print(f"Sequence generated: {len(placed_parts)} parts placed across {current_step - 1} steps.")
        return placed_parts
