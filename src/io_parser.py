import os
import json
import zipfile
import re
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from src.parser import ParsedPart

STUDIO_ZIP_PASSWORD = b"soho0909"


def export_parts_to_io(
    parts: List[ParsedPart],
    output_io_path: str,
    model_title: str = "LegoGPT Model",
) -> str:
    """
    Exports a sequence of ParsedPart objects into a BrickLink Studio 2.0 (.io)
    archive containing model.ldr, model2.ldr, .info, and errorPartList.err,
    preserving '0 STEP' markers.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_io_path)), exist_ok=True)
    filename = os.path.basename(output_io_path)

    sorted_parts = sorted(parts, key=lambda p: p.step_id)
    ldr_lines = [
        f"0 FILE {filename}",
        f"0 LEGO {model_title}",
        f"0 Name:  {filename}",
        "0 Author: LegoGPT Generator",
        "0 CustomBrick",
        "0 FlexibleBrickControlPointUnitLength -1",
        "0 FlexibleBrickLockedControlPoint",
        f"0 TotalBricks: {len(sorted_parts)}",
    ]

    current_step: Optional[int] = None
    for p in sorted_parts:
        if current_step is not None and p.step_id != current_step:
            ldr_lines.append("0 STEP")
        current_step = p.step_id

        t = p.transform
        x, y, z = float(t[0, 3]), float(t[1, 3]), float(t[2, 3])
        rot = t[:3, :3].flatten()
        rot_str = " ".join(f"{float(v):.6f}" for v in rot)
        pid = p.part_id if p.part_id.lower().endswith(".dat") else f"{p.part_id}.dat"
        ldr_lines.append(f"1 {int(p.color)} {x:.4f} {y:.4f} {z:.4f} {rot_str} {pid}")

    ldr_lines.append("0 NOFILE")
    model_ldr_content = "\n".join(ldr_lines) + "\n"

    info_json = json.dumps({
        "version": "2.25.11_3",
        "total_parts": len(sorted_parts),
        "parts_db_version": 200,
    })

    with zipfile.ZipFile(output_io_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.ldr", model_ldr_content.encode("utf-8"))
        zf.writestr("model2.ldr", model_ldr_content.encode("utf-8"))
        zf.writestr(".info", info_json.encode("utf-8"))
        zf.writestr("errorPartList.err", b"[]")

    return output_io_path


class StudioIOParser:
    """
    Parser for BrickLink Studio (.io) files.
    A .io file is a zip archive (optionally encrypted with static password 'soho0909')
    containing model.ldr and optional metadata/thumbnails.
    """
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    @staticmethod
    def _read_zip_member(zf: zipfile.ZipFile, member: str) -> bytes:
        try:
            zf.setpassword(STUDIO_ZIP_PASSWORD)
            return zf.read(member)
        except Exception:
            return zf.read(member)
            
    def is_valid_io(self) -> bool:
        """Checks if the file is a valid zip archive containing model.ldr or model2.ldr."""
        if not zipfile.is_zipfile(self.file_path):
            return False
        try:
            with zipfile.ZipFile(self.file_path, "r") as zf:
                names = zf.namelist()
                return "model.ldr" in names or "model2.ldr" in names or any(n.lower().endswith(".ldr") for n in names)
        except Exception:
            return False

    def extract_ldr_content(self) -> str:
        """Extracts the raw UTF-8 string content of model.ldr from the .io archive."""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            names = zf.namelist()
            target = None
            for cand in ("model.ldr", "model2.ldr"):
                if cand in names:
                    target = cand
                    break
            if target is None:
                for n in names:
                    if n.lower().endswith(".ldr"):
                        target = n
                        break
            if target is None:
                raise ValueError("model.ldr not found inside .io archive")
            content_bytes = self._read_zip_member(zf, target)
            return content_bytes.decode("utf-8", errors="replace")

    def extract_thumbnail(self, output_path: str) -> Optional[str]:
        """Extracts thumbnail image if present in the archive."""
        with zipfile.ZipFile(self.file_path, "r") as zf:
            candidates = ["thumbnail.png", "thumbnail_large.png", "thumbnail/thumbnail.png"]
            for c in candidates:
                if c in zf.namelist():
                    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
                    with open(output_path, "wb") as out_f:
                        out_f.write(self._read_zip_member(zf, c))
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
