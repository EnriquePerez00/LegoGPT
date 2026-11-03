# LegoGPT Agent Rules

## Physical Connection Constraints
- **Stud-to-Socket Rule**: Lego parts can only connect if there is a matching physical alignment between a stud (conos de encaje) and a socket (hueco). Connections without matching studs and sockets are strictly forbidden.
- **Surface and Height Level Rules**: 
  - All parts must reside at or above the ground surface level in the normalized coordinate system (Y <= 0.0 in standard LDraw coordinates, where min(Y_bottom) is normalized to ground level Y = 0.0).
  - During intermediate build steps or when the assembly is rotated/flipped (e.g. inverted steps), parts may temporarily be placed under existing layers; upon step completion, the whole assembly must be renormalized so that min(Y) aligns with the construction ground (Y = 0.0).
  - At most 4 parts are allowed to touch the ground/surface directly (|Y_world| <= 1.0 after normalization).
- **Physical Overlaps**: Overlaps between oriented bounding boxes are strictly prohibited in both physical database generation and 3D web visualizations.
