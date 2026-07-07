import os
import sys
import argparse
import subprocess
import numpy as np
import torch
from src.parser import ALLOWED_PARTS, ALLOWED_COLORS
from src.model import LegoGNN
from src.generator import LegoGenerator
from src.writer import write_ldraw_file

def main():
    parser = argparse.ArgumentParser(description="LegoGPT generate_build CLI")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt describing structure (e.g. 'Silla roja')")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--num_pieces", type=int, default=None, help="Force number of pieces")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    prompt = args.prompt.lower()
    
    # 1. Semantic prompt extraction
    target_color = 14  # Default yellow
    if "roja" in prompt or "rojo" in prompt or "red" in prompt:
        target_color = 4  # Red LDraw code
    elif "azul" in prompt or "blue" in prompt:
        target_color = 1  # Blue
    elif "verde" in prompt or "green" in prompt:
        target_color = 2  # Green
        
    # Map keywords to target inventory
    target_inventory = None
    target_num_pieces = args.num_pieces if args.num_pieces is not None else 6
    
    if "silla" in prompt or "chair" in prompt:
        # A simple chair: 2 of 2x4 (3001.dat) for base/legs, 2 of 1x2 (3004.dat) for seat, 2 of 1x1 (3005.dat) for backrest
        target_inventory = {
            "3001.dat": 2,
            "3004.dat": 2,
            "3005.dat": 2
        }
        target_num_pieces = 6
    elif "torre" in prompt or "tower" in prompt:
        # A tower: 3 of 2x4 (3001.dat), 2 of 1x2 (3004.dat)
        target_inventory = {
            "3001.dat": 3,
            "3004.dat": 2
        }
        target_num_pieces = 5
        
    # 2. Instantiate LegoGPT model (use LegoGNN model)
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        
    model = LegoGNN(num_part_classes=len(ALLOWED_PARTS), num_color_classes=len(ALLOWED_COLORS), hidden_dim=16).to(device)
    
    # Check if a trained checkpoint exists
    checkpoint_path = "models/fine_tuned_rl.pt"
    if not os.path.exists(checkpoint_path):
        checkpoint_path = "models/modelo_lego_1.pt"
        
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except Exception:
            pass
            
    model.eval()
    
    # 3. Generate using LegoGenerator with Beam Search & target inventory
    generator = LegoGenerator(model, ALLOWED_PARTS, ALLOWED_COLORS, device=device)
    parts = generator.generate_beam_search(
        target_num_pieces=target_num_pieces,
        beam_width=3,
        max_candidates=3,
        target_inventory=target_inventory
    )
    
    if not parts:
        print("Error: Could not generate a valid physical structure.")
        sys.exit(1)
        
    # Override color if specified in the prompt
    for p in parts:
        p.color = target_color
        
    # 4. Group by layer (Y coordinate) and insert 0 STEP LDraw command
    # In LDraw, Y increases downwards. We sort from ground-up (highest Y to lowest Y).
    parts_sorted = sorted(parts, key=lambda p: p.transform[1, 3], reverse=True)
    
    output_ldr_name = prompt.replace(" ", "_") + ".ldr"
    output_ldr_path = os.path.join(args.output_dir, output_ldr_name)
    
    lines = ["0 LegoGPT Generated Build", f"0 Prompt: {args.prompt}"]
    
    current_y = None
    for p in parts_sorted:
        y_val = p.transform[1, 3]
        if current_y is not None and abs(y_val - current_y) > 5.0:
            lines.append("0 STEP")
        
        current_y = y_val
        
        # Serialize part
        color = p.color
        x, y, z = p.transform[:3, 3]
        rot = p.transform[:3, :3].flatten()
        rot_str = " ".join(f"{val:.6f}" for val in rot)
        lines.append(f"1 {color} {x:.4f} {y:.4f} {z:.4f} {rot_str} {p.part_id}")
        
    # Write file
    with open(output_ldr_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"Generated LDraw file saved to: {output_ldr_path}")
    
    # 5. Invoke Blender headless rendering
    blender_path = os.getenv("BLENDER_PATH", "/Users/I764690/Applications/Blender.app/Contents/MacOS/Blender")
    output_png_name = prompt.replace(" ", "_") + ".png"
    output_png_path = os.path.join(args.output_dir, output_png_name)
    
    if os.path.exists(blender_path):
        print("Invoking Blender to render the structure...")
        cmd = [
            blender_path,
            "-b",
            "-P", "scratch/render_helper.py",
            "--",
            "--filepath", output_ldr_path,
            "--output", output_png_path
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Render saved to: {output_png_path}")
        except Exception as e:
            print(f"Blender render failed: {e}")
    else:
        print(f"Blender not found at {blender_path}. Skipping render.")

if __name__ == "__main__":
    main()
