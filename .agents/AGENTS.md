# LegoGPT Agent Rules

## Physical Connection Constraints
- **Stud-to-Socket Rule**: Lego parts can only connect if there is a matching physical alignment between a stud (conos de encaje) and a socket (hueco). Connections without matching studs and sockets are strictly forbidden.
- **Surface and Height Level Rules**: 
  - All parts must reside at or above the ground surface level (Y <= 0.0 in LDraw coordinates).
  - No part can be placed below the surface level.
  - At most 4 parts are allowed to touch the ground/surface directly (|Y_world| <= 1.0).
- **Physical Overlaps**: Overlaps between oriented bounding boxes are strictly prohibited in both physical database generation and 3D web visualizations.
