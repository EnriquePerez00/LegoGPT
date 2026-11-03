import os
import zipfile
import re
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

class StudioIOParser:
    """
    Parser for BrickLink Studio (.io) files.
    A .io file is a zip archive containing model.ldr and optional metadata/thumbnails.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
    def is_valid_io(self) -> bool:
        """Checks if the file is a valid zip archive containing model.ldr."""
        if not zipfile.is_zipfile(self.file_path):
            return False
        try:
            with zipfile.ZipFile(self.file_path, "r") as zf:
                return "model.ldr" in zf.namelist()
        except Exception:
            return False

    def extract_ldr_content(self) -> str:
        """Extracts the raw UTF-8 string content of model.ldr from the .io archive."""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            if "model.ldr" not in zf.namelist():
                raise ValueError("model.ldr not found inside .io archive")
            content_bytes = zf.read("model.ldr")
            return content_bytes.decode("utf-8", errors="replace")

    def extract_thumbnail(self, output_path: str) -> Optional[str]:
        """Extracts thumbnail image if present in the archive."""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            candidates = ["thumbnail.png", "thumbnail_large.png", "thumbnail/thumbnail.png"]
            for c in candidates:
                if c in zf.namelist():
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    with open(output_path, "wb") as out_f:
                        out_f.write(zf.read(c))
                    return output_path
        return None

    def analyze_steps(self) -> Dict[str, Any]:
        """
        Analyzes the step structure of model.ldr inside the .io archive.
        Returns metrics on parts, steps, and granularity.
        """
        ldr_text = self.extract_ldr_content()
        lines = ldr_text.splitlines()
        
        parts_count = 0
        step_count = 1
        current_step_parts = 0
        parts_per_step = []
        submodels = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            tokens = line_str.split()
            if not tokens:
                continue
                
            cmd = tokens[0]
            if cmd == "0":
                if len(tokens) >= 2:
                    sub_cmd = tokens[1].upper()
                    if sub_cmd == "STEP":
                        parts_per_step.append(current_step_parts)
                        step_count += 1
                        current_step_parts = 0
                    elif sub_cmd == "FILE" and len(tokens) >= 3:
                        submodels.append(" ".join(tokens[2:]))
            elif cmd == "1":
                # Part definition: 1 <color> <x> <y> <z> <a> <b> <c> <d> <e> <f> <g> <h> <i> <part_name>
                if len(tokens) >= 15:
                    parts_count += 1
                    current_step_parts += 1
                    
        parts_per_step.append(current_step_parts)
        
        has_discrete_steps = step_count > 1 and max(parts_per_step) < parts_count
        avg_parts_per_step = parts_count / max(1, step_count)
        
        return {
            "file_path": self.file_path,
            "parts_count": parts_count,
            "step_count": step_count,
            "has_discrete_steps": has_discrete_steps,
            "avg_parts_per_step": round(avg_parts_per_step, 2),
            "parts_per_step": parts_per_step,
            "submodels_count": len(submodels),
            "submodels": submodels
        }

    def save_ldr_to_disk(self, output_ldr_path: str) -> str:
        """Extracts model.ldr directly to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(output_ldr_path)), exist_ok=True)
        ldr_text = self.extract_ldr_content()
        with open(output_ldr_path, "w", encoding="utf-8") as f:
            f.write(ldr_text)
        return output_ldr_path
