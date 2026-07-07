import os
import re

class LdrTokenizer:
    def __init__(self):
        self.vocab = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "0", "STEP"]
        # Add basic integers and floats to vocab
        for i in range(-500, 500):
            self.vocab.append(str(i))
        # Common floats for rotation matrices
        for f in ["1.0", "0.0", "-1.0", "1.00", "0.00", "-1.00"]:
            if f not in self.vocab:
                self.vocab.append(f)
        # Add allowed parts
        from src.parser import ALLOWED_PARTS
        for p in ALLOWED_PARTS:
            self.vocab.append(p)
            
        self.w2i = {w: i for i, w in enumerate(self.vocab)}
        self.i2w = {i: w for i, w in enumerate(self.vocab)}

    def clean_and_normalize_ldr_text(self, ldr_text: str) -> str:
        """
        Cleans LDraw comments (except STEP) and rounds coordinates to integers (LDU).
        """
        lines = []
        for line in ldr_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            cmd = tokens[0]
            if cmd == "0":
                if len(tokens) >= 2 and tokens[1].upper() == "STEP":
                    lines.append("0 STEP")
            elif cmd == "1" and len(tokens) >= 15:
                color = tokens[1]
                # Position
                x, y, z = float(tokens[2]), float(tokens[3]), float(tokens[4])
                # Rotation matrix (we round elements to 2 decimal places to keep vocabulary small)
                rot = [float(val) for val in tokens[5:14]]
                part_name = tokens[14].lower()
                
                # Round coordinates
                rx, ry, rz = int(round(x)), int(round(y)), int(round(z))
                rot_str = " ".join(f"{val:.2f}" for val in rot)
                
                normalized_line = f"1 {color} {rx} {ry} {rz} {rot_str} {part_name}"
                lines.append(normalized_line)
        return "\n".join(lines)

    def encode(self, text: str) -> list[int]:
        """Converts normalized LDraw text into token IDs."""
        normalized = self.clean_and_normalize_ldr_text(text)
        tokens = ["[BOS]"]
        for line in normalized.split("\n"):
            tokens.extend(line.split())
            tokens.append("\n")
        tokens.append("[EOS]")
        
        token_ids = []
        for t in tokens:
            if t in self.w2i:
                token_ids.append(self.w2i[t])
            else:
                token_ids.append(self.w2i["[UNK]"])
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        """Converts token IDs back into LDraw text."""
        words = []
        for tid in token_ids:
            w = self.i2w.get(tid, "[UNK]")
            if w in ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]:
                continue
            words.append(w)
            
        # Reconstruct lines
        reconstructed = []
        curr_line = []
        for w in words:
            if w == "\n":
                if curr_line:
                    reconstructed.append(" ".join(curr_line))
                    curr_line = []
            else:
                curr_line.append(w)
        if curr_line:
            reconstructed.append(" ".join(curr_line))
        return "\n".join(reconstructed)
