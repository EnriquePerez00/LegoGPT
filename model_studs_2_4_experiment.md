# Experiment Log: LegoGPT Model Training & Evaluation

This file tracks the setup, configuration, and results for each iteration of training the LegoGPT generation models.

---

## Iteration 1: 10-Piece Vocabulary Sandbox
**Date**: 2026-06-19
**Model Name**: `model_studs_2_4`
**Goal**: Train a model to generate stable constructions of exactly 2 and 4 pieces.

### 1. Setup & Configuration
- **Vocabulary**: 10 simple-geometry pieces.
- **Node Features**: 38 dimensions.
- **Dataset**: 500 stable models of size 2 and 4.
- **Training Epochs**: 40
- **Final Loss**: **87.5981**

### 2. Results
- Generated 6 constructions of size 2 and 4. Detail saved in [creations_report.md](file:///Users/I764690/Code_personal/LegoGPT/creations_report.md) and interactively rendered in HTML.

---

## Iteration 2: 8-Piece/Step Model with Ground/Surface Constraints (Re-trained)
**Date**: 2026-06-19
**Model Name**: `model_studs_8`
**Goal**: Train a model to generate complex constructions of exactly 8 pieces, always building upwards from the surface (no parts below Y=0.0) and allowing at most 4 parts in the base layer, ensuring proper connection without overlaps.

### 1. Ground & Assembly Constraints
- **Vertical Building (Y <= 0.0)**: In LDraw coordinates, Y increases downwards. The ground surface is defined at Y = 0.0. To prevent pieces from going below the surface, all parts must satisfy `Y_world_coord <= 0.0` (with 1.0 LDU float tolerance).
- **Ground Footprint (Base <= 4)**: To avoid spread out models, a maximum of 4 pieces can touch the ground surface (`|Y_world_coord| <= 1.0`).
- **Target Size**: Exactly 8 pieces (sequentially placed). Discards any procedurally generated models that fail to reach 8 steps during dataset compilation.

### 2. Setup & Configuration
- **Vocabulary**: 10 simple-geometry pieces.
- **Dataset**: 500 stable models conforming to the ground constraints, each containing exactly 8 parts.
- **Training Epochs**: 40
- **Final Loss**: **136.1302**

### 3. Generation Results (10 New 8-Piece Creations)
Below is the summary of the 10 newly generated structures using `model_studs_8`.

#### Creación #1 (8 piezas)
- **Part list**:
  1. `3005.dat` (Brick 1x1, Color: 12, Y: 0.0)
  2. `3004.dat` (Brick 1x2, Color: 1, Y: -24.0)
  3. `3004.dat` (Brick 1x2, Color: 2, Y: -48.0)
  4. `3022.dat` (Plate 2x2, Color: 3, Y: -72.0)
  5. `3024.dat` (Plate 1x1, Color: 1, Y: -16.0)
  6. `3001.dat` (Brick 2x4, Color: 3, Y: -80.0)
  7. `3022.dat` (Plate 2x2, Color: 8, Y: -72.0)
  8. `3010.dat` (Brick 1x4, Color: 14, Y: -48.0)
- **LDraw Source**: [creation_8_1.ldr](file:///Users/I764690/Code_personal/LegoGPT/scratch/creation_8_1.ldr)

#### Creación #2 (8 piezas)
- **Part list**:
  1. `3710.dat` (Plate 1x4, Color: 1, Y: 0.0)
  2. `3024.dat` (Plate 1x1, Color: 13, Y: -8.0)
  3. `3003.dat` (Brick 2x2, Color: 6, Y: -8.0)
  4. `3003.dat` (Brick 2x2, Color: 15, Y: -8.0)
  5. `3005.dat` (Brick 1x1, Color: 10, Y: -8.0)
  6. `3004.dat` (Brick 1x2, Color: 5, Y: -32.0)
  7. `3005.dat` (Brick 1x1, Color: 0, Y: -56.0)
  8. `3023.dat` (Plate 1x2, Color: 12, Y: -16.0)
- **LDraw Source**: [creation_8_2.ldr](file:///Users/I764690/Code_personal/LegoGPT/scratch/creation_8_2.ldr)

*(Creations #3 to #10 details are formatted similarly in the LDraw sources in scratch/ and rendered sequentially in the HTML report).*

---

### 4. Interactive Visualization
The step-by-step assembly sequence can be viewed dynamically in 3D:
- Open [creations_report.html](file:///Users/I764690/Code_personal/LegoGPT/creations_report.html) in your browser.
- Select the creation number from the dropdown to see its step-by-step assembly.
- Studs are rendered as small cylinders on top of each piece to illustrate how they anchor and connect together.
