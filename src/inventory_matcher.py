import os
import re
import fitz  # PyMuPDF
import numpy as np
import torch
import torchvision.models as models
from PIL import Image

class LegoInventoryMatcher:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.temp_dir = "scratch/inventory_templates"
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Load lightweight ResNet18 for embeddings
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        # Strip final FC classification layer to get feature embeddings
        self.feature_extractor = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.feature_extractor.eval()
        self.preprocess = self.weights.transforms()
        
        self.templates = []
        self.inventory_pages = []

    def detect_inventory_pages(self) -> list[int]:
        """Detects pages containing the final parts list based on code density."""
        doc = fitz.open(self.pdf_path)
        num_pages = len(doc)
        detected = []
        # Check last 10 pages for high density of 6-7 digit codes
        start_check = max(0, num_pages - 10)
        for idx in range(start_check, num_pages):
            page = doc[idx]
            text = page.get_text()
            # Match 6 or 7 digit numbers
            codes = re.findall(r"\b\d{6,7}\b", text)
            if len(codes) >= 5:
                detected.append(idx)
        doc.close()
        self.inventory_pages = detected
        return detected

    def extract_templates(self) -> list[dict]:
        """Parses inventory pages, crops templates, and computes visual embeddings."""
        if not self.inventory_pages:
            self.detect_inventory_pages()
            
        doc = fitz.open(self.pdf_path)
        self.templates = []
        
        for page_num in self.inventory_pages:
            page = doc[page_num]
            words = page.get_text("words")
            
            # Words: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            # Find all element IDs (6-7 digit codes)
            element_words = []
            for w in words:
                text = w[4].strip()
                if re.match(r"^\d{6,7}$", text):
                    element_words.append(w)
            
            for ew in element_words:
                ex0, ey0, ex1, ey1, element_id, _, _, _ = ew
                
                # Search for quantity (e.g., "3x", "1x") right above
                qty = 1
                qty_word = None
                for w in words:
                    w_text = w[4].strip()
                    if re.match(r"^\d+x$", w_text, re.IGNORECASE):
                        wx0, wy0, wx1, wy1 = w[0], w[1], w[2], w[3]
                        # Check if it's within a vertical distance of 30pt and roughly horizontally aligned
                        if abs(wx0 - ex0) < 40 and 0 < (ey0 - wy1) < 30:
                            try:
                                qty = int(w_text[:-1])
                            except ValueError:
                                qty = 1
                            qty_word = w
                            break
                
                # Determine image crop area directly above
                ref_y = qty_word[1] if qty_word else ey0
                ref_x0 = min(ex0, qty_word[0]) if qty_word else ex0
                ref_x1 = max(ex1, qty_word[2]) if qty_word else ex1
                
                # Image bounding box heuristic (square-ish region above the text)
                crop_w = 70
                crop_h = 70
                crop_rect = fitz.Rect(
                    (ref_x0 + ref_x1)/2 - crop_w/2,
                    ref_y - crop_h - 5,
                    (ref_x0 + ref_x1)/2 + crop_w/2,
                    ref_y - 2
                )
                
                # Clip to page boundary
                crop_rect.x0 = max(0, crop_rect.x0)
                crop_rect.y0 = max(0, crop_rect.y0)
                crop_rect.x1 = min(page.rect.width, crop_rect.x1)
                crop_rect.y1 = min(page.rect.height, crop_rect.y1)
                
                # Save crop to file
                pix = page.get_pixmap(clip=crop_rect, dpi=150)
                img_path = os.path.join(self.temp_dir, f"{element_id}.png")
                pix.save(img_path)
                
                # Compute visual embedding
                try:
                    embedding = self._compute_embedding(img_path)
                    
                    self.templates.append({
                        "element_id": element_id,
                        "qty": qty,
                        "image_path": img_path,
                        "embedding": embedding
                    })
                except Exception as e:
                    print(f"Error processing template {element_id}: {e}")
                    
        doc.close()
        return self.templates

    def _compute_embedding(self, image_path: str) -> np.ndarray:
        """Helper to load image and extract ResNet-18 features."""
        image = Image.open(image_path).convert("RGB")
        img_t = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_extractor(img_t)
        return features.squeeze().numpy()

    def match_part(self, query_image_path: str, remaining_inventory: list[dict]) -> dict:
        """
        Finds the closest matching part in remaining_inventory based on Cosine Similarity.
        remaining_inventory should be a list of template dicts.
        """
        if not os.path.exists(query_image_path):
            return None
            
        try:
            query_emb = self._compute_embedding(query_image_path)
        except Exception as e:
            print(f"Error computing query embedding for {query_image_path}: {e}")
            return None
            
        best_score = -1.0
        best_match = None
        
        for t in remaining_inventory:
            t_emb = t["embedding"]
            # Cosine similarity
            dot_prod = np.dot(query_emb, t_emb)
            norm_q = np.linalg.norm(query_emb)
            norm_t = np.linalg.norm(t_emb)
            if norm_q > 0 and norm_t > 0:
                score = dot_prod / (norm_q * norm_t)
            else:
                score = 0.0
                
            if score > best_score:
                best_score = score
                best_match = t
                
        return best_match
