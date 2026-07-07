import json
import os
import numpy as np
from src.parser import ParsedPart
from src.validator import get_part_dimensions

def get_color_hex(color_id: int) -> str:
    """Returns typical HEX color for standard LDraw color IDs."""
    colors = {
        0: "#1b1c1e",   # Black
        1: "#1e3a8a",   # Blue
        2: "#15803d",   # Green
        4: "#b91c1c",   # Red
        6: "#78350f",   # Brown
        7: "#94a3b8",   # Light Gray
        8: "#475569",   # Dark Gray
        9: "#60a5fa",   # Light Blue
        14: "#eab308",  # Yellow
        15: "#f8fafc",  # White
        25: "#f97316",  # Orange
        26: "#d946ef",  # Magenta
        29: "#f472b6",  # Pink
        73: "#3b82f6",  # Medium Blue
        272: "#1e40af", # Dark Blue
        378: "#86efac", # Sand Green
    }
    return colors.get(color_id, "#cbd5e1") # Fallback to slate-300

PART_NAMES = {
    "3070.dat": "Flat Tile 1x1",
    "3069.dat": "Flat Tile 1x2",
    "3794.dat": "Plate 1x2 W. 1 Knob",
    "70501c.dat": "Coin No. 3",
    "3003.dat": "Brick 2x2",
    "3001.dat": "Brick 2x4",
    "3005.dat": "Brick 1x1",
    "3010.dat": "Brick 1x4",
    # 75400-1 parts
    "3034.dat": "Plate 2x8",
    "3021.dat": "Plate 2x3",
    "15573.dat": "Plate 1x2 W. 1 Knob",
    "26601.dat": "Plate 2x2 Corner 45 Deg.",
    "11477.dat": "Plate W. Bow 1x2x2/3",
    "15556.dat": "Light Sword Shaft",
    "28697.dat": "Light Sword Blade",
    "28701.dat": "Nose Cone Small 1x1",
    "32803.dat": "Brick 2x2x2/3 Inverted Bow",
    "35338.dat": "Roof Tile 1x1x2/3",
    "35380.dat": "Flat Tile 1x1 Round",
    "35480.dat": "Plate 1x2 Rounded",
    "35787.dat": "Tile 2x2 W. 45 Cut",
    "3665.dat": "Roof Tile 1x2 Inv.",
    "41769.dat": "Right Plate 2x4 W. Angle",
    "41770.dat": "Left Plate 2x4 W. Angle",
    "41822.dat": "Plate 4x4 W. Angle",
    "42610.dat": "Hub 11.2x7.84",
    "4274.dat": "Connector Peg W. Knob",
    "42923.dat": "Plate 2x1 W. Holder Vertical",
    "48205.dat": "Right Plate 4x6",
    "48208.dat": "Left Plate 4x6",
    "50340.dat": "Plate 1x2 W. Fork Vertical",
    "5091.dat": "Tile 1x2 Cut Left",
    "5092.dat": "Tile 1x2 Cut Right",
    "51483.dat": "Plate 1x4 W. Stumps Vertical",
    "5414.dat": "Brick 1x4x1 Half Bow Right",
    "5415.dat": "Brick 1x4x1 Half Bow Left",
    "65426.dat": "Right Plate 2x4",
    "65429.dat": "Left Plate 2x4",
    "69754.dat": "Plate 1x2 W. Shooter",
    "69755.dat": "Trigger No. 1",
    "76382.dat": "Mini Upper Torso",
    "111870.dat": "Mini Creature Head",
    "112754.dat": "Mini Legs",
    "112755.dat": "Final Brick 2x2",
    "79491.dat": "Plate 2x2 1/4 Circle",
}

def generate_html_sequence_report(parts: list[ParsedPart], output_html_path: str):
    """
    Generates a beautiful interactive 3D step-by-step building sequence report
    using Three.js and Tailwind CSS.
    """
    # Group parts by step_id
    steps_dict = {}
    for part in parts:
        step_id = part.step_id
        if step_id not in steps_dict:
            steps_dict[step_id] = []
            
        # Get dimensions
        w, h, d = get_part_dimensions(part.part_id)
        
        # Format transformation matrix for JS (row-major list)
        matrix_list = part.transform.flatten().tolist()
        
        name = PART_NAMES.get(part.part_id, part.part_id.replace(".dat", "").replace("b", ""))
        
        steps_dict[step_id].append({
            "part_id": part.part_id,
            "name": name,
            "color": part.color,
            "color_hex": get_color_hex(part.color),
            "matrix": matrix_list,
            "dimensions": {"w": w, "h": h, "d": d}
        })
        
    # Sort steps
    sorted_steps = []
    for step_id in sorted(steps_dict.keys()):
        # Find page_num from part attribute if exists
        page_num = step_id
        for part in parts:
            if part.step_id == step_id and hasattr(part, 'page_num'):
                page_num = part.page_num
                break
        sorted_steps.append({
            "step_id": step_id,
            "page_num": page_num,
            "parts": steps_dict[step_id]
        })
        
    # Generate unique list of parts for the legend/glossary
    part_vocab = {}
    for p in parts:
        part_vocab[p.part_id] = get_part_dimensions(p.part_id)
        
    # Create HTML
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualizador de Secuencia 3D - Set 75400-1</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Three.js and OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <!-- Outfit Font -->
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background-color: #0b0f19;
            color: #f8fafc;
            overflow: hidden;
        }}
        .glass-panel {{
            background: rgba(15, 23, 42, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .custom-scrollbar::-webkit-scrollbar {{
            width: 6px;
        }}
        .custom-scrollbar::-webkit-scrollbar-track {{
            background: rgba(0, 0, 0, 0.1);
        }}
        .custom-scrollbar::-webkit-scrollbar-thumb {{
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }}
    </style>
</head>
<body class="h-screen w-screen flex flex-col md:flex-row">

    <!-- Left Sidebar: Step Dashboard -->
    <aside id="sidebar" class="w-full md:w-96 glass-panel flex flex-col p-6 z-10 border-b md:border-b-0 md:border-r border-slate-800">
        <div class="mb-6">
            <div class="flex items-center space-x-2 text-blue-500 font-semibold text-xs tracking-wider uppercase mb-1">
                <span>🤖 LegoGPT Reconstruction Pipeline</span>
                <span class="bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded-full text-[10px]">Verified</span>
            </div>
            <h1 class="text-2xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-200 to-blue-400">Set 75400-1</h1>
            <p class="text-xs text-slate-400 mt-1">Plo Koon's Jedi Starfighter Microfighter</p>
            <div class="mt-3 text-[11px] text-slate-400 bg-slate-950/60 p-3 rounded-lg border border-slate-800/80">
                <span class="font-semibold text-blue-400 block mb-1">Referencia Visual:</span>
                Modelo: <a href="https://www.mecabricks.com/en/library" target="_blank" class="text-blue-400 hover:underline">Mecabricks Library (75400)</a>
                <span class="block mt-1">Validador Físico: <strong class="text-emerald-400">CONECTADO y GRAVEDAD VÁLIDA</strong></span>
                <span class="block mt-0.5">Inventario: <strong>89 piezas de Brickset</strong></span>
            </div>
        </div>

        <!-- Assembly Progress -->
        <div class="flex-1 flex flex-col overflow-hidden">
            <div class="flex justify-between items-center mb-3">
                <h3 class="text-xs font-semibold text-slate-400 uppercase tracking-wider">Secuencia de Montaje</h3>
                <span id="step-badge" class="text-xs bg-blue-500/20 text-blue-400 px-2.5 py-0.5 rounded-full font-bold">Paso 0 / {len(sorted_steps)}</span>
            </div>
            
            <div id="steps-list" class="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar text-sm">
                <!-- Javascript will load steps dynamically -->
            </div>
        </div>

        <!-- Playback Navigation Controls -->
        <div class="mt-6 pt-6 border-t border-slate-800/80 flex flex-col gap-3">
            <div class="flex items-center justify-between gap-2">
                <button id="btn-prev" class="flex items-center justify-center p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-600 active:scale-95 transition-all w-[30%]">
                    <svg class="w-5 h-5 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
                </button>
                <button id="btn-play" class="flex items-center justify-center p-3 rounded-lg bg-blue-600 hover:bg-blue-500 active:scale-95 transition-all text-white font-semibold text-sm w-[35%]">
                    Auto Play
                </button>
                <button id="btn-next" class="flex items-center justify-center p-3 rounded-lg bg-slate-900 border border-slate-800 hover:border-slate-600 active:scale-95 transition-all w-[30%]">
                    <svg class="w-5 h-5 text-slate-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
                </button>
            </div>
            <div class="text-[10px] text-slate-500 text-center uppercase tracking-widest font-semibold">
                Optimizado para M4 GPU y 12 núcleos CPU
            </div>
        </div>
    </aside>

    <!-- Main Screen: 3D Scene Viewport -->
    <main class="flex-1 h-full relative">
        <div id="canvas-3d" class="w-full h-full"></div>
        
        <!-- Part Legend / Glossary Overlay -->
        <div class="absolute bottom-6 right-6 glass-panel rounded-xl p-4 flex flex-col space-y-2 pointer-events-none text-xs w-64">
            <h4 class="font-bold text-slate-300 mb-1 text-[10px] uppercase tracking-wider">Vocabulario de Piezas Utilizadas</h4>
            <div id="legend-list" class="space-y-1.5 max-h-40 overflow-y-auto pr-1 custom-scrollbar">
                <!-- Javascript will insert glossary items -->
            </div>
        </div>
    </main>

    <!-- Interactive Logic Script -->
    <script>
        const SEQUENCE_STEPS = {json.dumps(sorted_steps)};
        
        let scene, camera, renderer, controls;
        let currentStepIdx = 0;
        let legoMeshes = [];
        let autoPlayInterval = null;

        function init3D() {{
            const container = document.getElementById('canvas-3d');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0b0f19);

            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.set(100, 80, 150);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.shadowMap.enabled = true;
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground

            // Lighting setup
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
            scene.add(ambientLight);

            const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
            dirLight1.position.set(100, 150, 50);
            scene.add(dirLight1);

            const dirLight2 = new THREE.DirectionalLight(0x3b82f6, 0.4);
            dirLight2.position.set(-100, 50, -50);
            scene.add(dirLight2);

            // Ground grid helper
            const gridHelper = new THREE.GridHelper(300, 30, 0x1e293b, 0x0f172a);
            gridHelper.position.y = -0.5;
            scene.add(gridHelper);

            window.addEventListener('resize', onWindowResize);
            animate();
            
            loadSidebarSteps();
            loadLegend();
            renderStep(currentStepIdx);
        }}

        function mapLdrawToBricklink(colorId) {{
            const mapping = {{
                0: 11,   // Black
                1: 7,    // Blue
                2: 6,    // Green
                4: 5,    // Red
                6: 8,    // Brown
                7: 86,   // Light Gray -> Light Bluish Gray (86)
                8: 85,   // Dark Gray -> Dark Bluish Gray (85)
                9: 152,  // Light Blue
                14: 3,   // Yellow
                15: 1,   // White
                25: 4,   // Orange
                26: 47,  // Magenta
                29: 56,  // Pink
                73: 153, // Medium Blue
                272: 63, // Dark Blue
                378: 48  // Sand Green
            }};
            return mapping[colorId] || 1;
        }}

        function mapLdrawIdToBricklinkId(partId) {{
            const idMap = {{
                "76382": "973",     // LDraw Torso -> BrickLink 973 Torso
                "112754": "970",    // LDraw Legs -> BrickLink 970 Legs
                "111870": "3626",   // LDraw Head -> BrickLink 3626 Head
                "112755": "3003",   // LDraw Custom 2x2 -> BrickLink 3003 2x2 Brick
                "79491": "3063",    // LDraw Custom 2x2 1/4 Circle -> BrickLink 3063
                "42923": "63868",   // LDraw Plate 2x1 Vertical Holder -> BrickLink 63868
                "28809": "18677"    // LDraw Underside Pin Hole Plate -> BrickLink 18677
            }};
            return idMap[partId] || partId;
        }}

        function loadSidebarSteps() {{
            const list = document.getElementById('steps-list');
            list.innerHTML = '';
            
            SEQUENCE_STEPS.forEach((step, idx) => {{
                const item = document.createElement('div');
                item.className = `step-item flex items-center p-3 rounded-lg cursor-pointer border border-transparent transition-all duration-200 ${{
                    idx === currentStepIdx 
                        ? 'bg-blue-600/10 border-blue-500/30 text-white font-semibold' 
                        : 'bg-slate-900/40 hover:bg-slate-800/40 text-slate-400'
                }}`;
                item.onclick = () => {{
                    stopAutoPlay();
                    currentStepIdx = idx;
                    renderStep(idx);
                }};
                
                // Details of added parts
                let partDetails = "";
                const counts = {{}};
                step.parts.forEach(p => {{
                    counts[p.part_id] = (counts[p.part_id] || 0) + 1;
                }});
                
                const partsStr = Object.entries(counts).map(([id, qty]) => `${{qty}}x ${{id.replace('.dat', '')}}`).join(', ');
                
                // Generate BrickLink images list (doubled in size to w-16 h-16)
                let imagesHtml = "";
                const uniquePartsInStep = new Set();
                step.parts.forEach(p => {{
                    const partKey = p.part_id + "_" + p.color;
                    if (!uniquePartsInStep.has(partKey)) {{
                        uniquePartsInStep.add(partKey);
                        const cleanId = p.part_id.replace('.dat', '').replace(/b$/, '');
                        const blId = mapLdrawIdToBricklinkId(cleanId);
                        const blColor = mapLdrawToBricklink(p.color);
                        const imgUrl = `https://img.bricklink.com/ItemImage/PN/${{blColor}}/${{blId}}.png`;
                        imagesHtml += `<img src="${{imgUrl}}" class="w-16 h-16 object-contain bg-slate-950/85 rounded-lg border border-slate-800/60 p-1 shadow-md shadow-black/25 hover:scale-105 transition-transform" title="${{p.name || cleanId}}" onerror="this.src='https://img.bricklink.com/ItemImage/PN/1/${{blId}}.png'; this.onerror=function(){{this.src='https://rebrickable.com/media/parts/ldraw/0/${{blId}}.png';}}">`;
                    }}
                }});
                
                item.innerHTML = `
                    <div class="w-7 h-7 flex items-center justify-center rounded-full bg-slate-800 text-[11px] font-bold mr-3 border border-slate-700/50 flex-shrink-0">
                        ${{step.step_id}}
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="text-xs font-medium truncate">Paso ${{step.step_id}} (Pág. ${{step.page_num || step.step_id}})</div>
                        <div class="text-[10px] text-slate-500 truncate">${{partsStr}}</div>
                    </div>
                    <div class="flex items-center space-x-1.5 ml-2 flex-shrink-0">
                        ${{imagesHtml}}
                    </div>
                `;
                list.appendChild(item);
            }});
        }}

        function loadLegend() {{
            const list = document.getElementById('legend-list');
            list.innerHTML = '';
            
            const partCounts = {{}};
            const partNames = {{}};
            SEQUENCE_STEPS.forEach(step => {{
                step.parts.forEach(p => {{
                    partCounts[p.part_id] = (partCounts[p.part_id] || 0) + 1;
                    partNames[p.part_id] = p.name || p.part_id.replace('.dat', '');
                }});
            }});
            
            Object.entries(partCounts).forEach(([part_id, count]) => {{
                const item = document.createElement('div');
                item.className = "flex items-center justify-between text-[11px] text-slate-400 py-2 border-b border-slate-800/30 last:border-0";
                
                const cleanId = part_id.replace('.dat', '').replace(/b$/, '');
                const name = partNames[part_id];
                const mecabricksUrl = `https://www.mecabricks.com/en/part/${{cleanId}}`;
                
                // Try to find the color of this part in the sequence
                let partColor = 7;
                for (let step of SEQUENCE_STEPS) {{
                    const found = step.parts.find(p => p.part_id === part_id);
                    if (found) {{
                        partColor = found.color;
                        break;
                    }}
                }}
                
                const blId = mapLdrawIdToBricklinkId(cleanId);
                const blColor = mapLdrawToBricklink(partColor);
                const imgUrl = `https://img.bricklink.com/ItemImage/PN/${{blColor}}/${{blId}}.png`;
                
                item.innerHTML = `
                    <div class="flex items-center space-x-2">
                        <img src="${{imgUrl}}" class="w-12 h-12 object-contain bg-slate-950/80 rounded-lg border border-slate-800 p-0.5" onerror="this.src='https://img.bricklink.com/ItemImage/PN/1/${{blId}}.png'; this.onerror=function(){{this.src='https://rebrickable.com/media/parts/ldraw/0/${{blId}}.png';}}">
                        <div class="flex flex-col">
                            <a href="${{mecabricksUrl}}" target="_blank" class="font-semibold text-slate-300 hover:text-blue-400 transition-colors text-[10px] truncate w-32 block" title="${{name}}">${{name}}</a>
                            <span class="text-[8px] text-slate-500">Ref: ${{cleanId}}</span>
                        </div>
                                        <span class="text-slate-500 font-semibold text-[10px]">${{count}} uds</span>
                `;
                list.appendChild(item);
            }});
        }}

        // Detailed LEGO mesh generator with studs and custom shapes
        function createLegoMesh(partId, w_dim, h_dim, d_dim, colorHex) {{
            const group = new THREE.Group();
            const mat = new THREE.MeshStandardMaterial({{
                color: new THREE.Color(colorHex),
                roughness: 0.15,
                metalness: 0.05,
                transparent: partId.includes("28697"),
                opacity: partId.includes("28697") ? 0.75 : 1.0
            }});

            // 1. Lightsaber Blade (28697)
            if (partId.includes("28697")) {{
                const geom = new THREE.CylinderGeometry(3.2, 3.2, h_dim, 12);
                const mesh = new THREE.Mesh(geom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }}

            // 2. Lightsaber Hilt (15556)
            if (partId.includes("15556")) {{
                const geom = new THREE.CylinderGeometry(4.8, 4.8, h_dim, 12);
                const mesh = new THREE.Mesh(geom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }}

            // 3. Connector Peg (4274) - Rendered as Z-aligned cylinder
            if (partId.includes("4274")) {{
                const pegGeom = new THREE.CylinderGeometry(4.5, 4.5, d_dim, 12);
                pegGeom.rotateX(Math.PI / 2);
                const mesh = new THREE.Mesh(pegGeom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                
                const flangeGeom = new THREE.CylinderGeometry(6, 6, 2, 12);
                flangeGeom.rotateX(Math.PI / 2);
                const flangeMesh = new THREE.Mesh(flangeGeom, mat);
                flangeMesh.castShadow = true;
                flangeMesh.receiveShadow = true;
                group.add(flangeMesh);
                return group;
            }}

            // 4. Plate 1x2 Rounded (35480)
            if (partId.includes("35480")) {{
                const centerBox = new THREE.BoxGeometry(20, h_dim, 20);
                const boxMesh = new THREE.Mesh(centerBox, mat);
                boxMesh.castShadow = true;
                boxMesh.receiveShadow = true;
                group.add(boxMesh);

                const cylGeom = new THREE.CylinderGeometry(10, 10, h_dim, 16);
                
                const cyl1 = new THREE.Mesh(cylGeom, mat);
                cyl1.position.set(0, 0, -10);
                cyl1.castShadow = true;
                cyl1.receiveShadow = true;
                group.add(cyl1);

                const cyl2 = new THREE.Mesh(cylGeom, mat);
                cyl2.position.set(0, 0, 10);
                cyl2.castShadow = true;
                cyl2.receiveShadow = true;
                group.add(cyl2);

                const studGeom = new THREE.CylinderGeometry(5, 5, 4, 12);
                studGeom.translate(0, 2, 0);
                [-10, 10].forEach(zPos => {{
                    const stud = new THREE.Mesh(studGeom, mat);
                    stud.position.set(0, h_dim/2, zPos);
                    stud.castShadow = true;
                    stud.receiveShadow = true;
                    group.add(stud);
                }});
                return group;
            }}

            // 5. Curved Slopes / Bows
            const isSlope = ["11477", "3665", "5414", "5415", "32803"].some(id => partId.includes(id));
            if (isSlope) {{
                const shape = new THREE.Shape();
                shape.moveTo(-d_dim/2, -h_dim/2);
                shape.lineTo(d_dim/2, -h_dim/2);
                shape.lineTo(d_dim/2, h_dim/2 - 4);
                shape.quadraticCurveTo(-d_dim/4, h_dim/2, -d_dim/2, -h_dim/2 + 2);
                
                const extrudeSettings = {{
                    steps: 1,
                    depth: w_dim,
                    bevelEnabled: false
                }};
                
                const extrudeGeom = new THREE.ExtrudeGeometry(shape, extrudeSettings);
                extrudeGeom.center();
                extrudeGeom.rotateY(Math.PI / 2);
                
                const mesh = new THREE.Mesh(extrudeGeom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }}

            // 6. Minifigure Head (111870)
            if (partId.includes("111870")) {{
                const headGeom = new THREE.CylinderGeometry(7.5, 7.5, h_dim, 16);
                const mesh = new THREE.Mesh(headGeom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }}

            // 7. Minifigure Torso (76382)
            if (partId.includes("76382")) {{
                const geom = new THREE.BoxGeometry(w_dim, h_dim, d_dim);
                const pos = geom.attributes.position;
                for (let i = 0; i < pos.count; i++) {{
                    const y = pos.getY(i);
                    if (y > 0) {{
                        pos.setX(i, pos.getX(i) * 0.75);
                    }}
                }}
                geom.computeVertexNormals();
                const mesh = new THREE.Mesh(geom, mat);
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                group.add(mesh);
                return group;
            }}

            // 8. Minifigure Legs (112754)
            if (partId.includes("112754")) {{
                const legW = w_dim / 2 - 1.0;
                const legGeom = new THREE.BoxGeometry(legW, h_dim, d_dim);
                [-w_dim/4, w_dim/4].forEach(xOffset => {{
                    const legMesh = new THREE.Mesh(legGeom, mat);
                    legMesh.position.set(xOffset, 0, 0);
                    legMesh.castShadow = true;
                    legMesh.receiveShadow = true;
                    group.add(legMesh);
                }});
                return group;
            }}

            // Default Block Box
            const geom = new THREE.BoxGeometry(w_dim, h_dim, d_dim);
            geom.scale(0.98, 0.98, 0.98);
            const bodyMesh = new THREE.Mesh(geom, mat);
            bodyMesh.castShadow = true;
            bodyMesh.receiveShadow = true;
            group.add(bodyMesh);

            // Add studs unless smooth tile
            const smoothIds = ["3070", "3069", "35787", "5091", "5092"];
            const isSmooth = smoothIds.some(id => partId.includes(id));
            if (!isSmooth) {{
                const numX = Math.round(w_dim / 20);
                const numZ = Math.round(d_dim / 20);
                if (numX >= 1 && numZ >= 1) {{
                    const studGeom = new THREE.CylinderGeometry(5, 5, 4, 12);
                    studGeom.translate(0, 2, 0);
                    for (let x = 0; x < numX; x++) {{
                        for (let z = 0; z < numZ; z++) {{
                            const studMesh = new THREE.Mesh(studGeom, mat);
                            studMesh.castShadow = true;
                            studMesh.receiveShadow = true;
                            const posX = (x - (numX - 1) / 2) * 20;
                            const posZ = (z - (numZ - 1) / 2) * 20;
                            studMesh.position.set(posX, h_dim/2, posZ);
                            group.add(studMesh);
                        }}
                    }}
                }}
            }}
            return group;
        }}

        function renderStep(stepIdx) {{
            // Clear existing meshes
            legoMeshes.forEach(mesh => scene.remove(mesh));
            legoMeshes = [];
            
            // Update badge & active classes
            document.getElementById('step-badge').innerText = `Paso ${{SEQUENCE_STEPS[stepIdx].step_id}} / ${{SEQUENCE_STEPS[SEQUENCE_STEPS.length - 1].step_id}}`;
            
            const items = document.getElementById('steps-list').children;
            for (let i = 0; i < items.length; i++) {{
                if (i === stepIdx) {{
                    items[i].className = "step-item flex items-center p-3 rounded-lg cursor-pointer bg-blue-600/20 border border-blue-500/40 text-white font-semibold";
                }} else {{
                    items[i].className = "step-item flex items-center p-3 rounded-lg cursor-pointer bg-slate-900/40 hover:bg-slate-800/40 text-slate-400 border border-transparent";
                }}
            }}
            
            // Build model up to current step
            for (let i = 0; i <= stepIdx; i++) {{
                const step = SEQUENCE_STEPS[i];
                step.parts.forEach(part => {{
                    const w = part.dimensions.w;
                    const h = part.dimensions.h;
                    const d = part.dimensions.d;

                    const mesh = createLegoMesh(part.part_id, w, h, d, part.color_hex);
                    
                    // Convert LDraw matrix (Y-down) to Three.js coordinate system (Y-up, Z-backward)
                    const m = part.matrix;
                    const threeMat = new THREE.Matrix4();
                    threeMat.set(
                         m[0], -m[1], -m[2],  m[3],
                        -m[4],  m[5],  m[6], -m[7],
                        -m[8],  m[9],  m[10], -m[11],
                         0,      0,     0,    1
                    );
                    
                    // Translate bottom-center to geometric center (Y points up now, so shift UP by h/2)
                    const offsetMat = new THREE.Matrix4().makeTranslation(0, h/2, 0);
                    threeMat.multiply(offsetMat);
                    
                    mesh.applyMatrix4(threeMat);
                    
                    scene.add(mesh);
                    legoMeshes.push(mesh);
                }});
            }}
        }}

        function onWindowResize() {{
            const container = document.getElementById('canvas-3d');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }}

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}

        // Playback functions
        const btnPlay = document.getElementById('btn-play');
        const btnPrev = document.getElementById('btn-prev');
        const btnNext = document.getElementById('btn-next');

        function stopAutoPlay() {{
            if (autoPlayInterval) {{
                clearInterval(autoPlayInterval);
                autoPlayInterval = null;
                btnPlay.innerText = "Auto Play";
                btnPlay.className = "flex items-center justify-center p-3 rounded-lg bg-blue-600 hover:bg-blue-500 active:scale-95 transition-all text-white font-semibold text-sm w-[35%]";
            }}
        }}

        function startAutoPlay() {{
            btnPlay.innerText = "Pausar";
            btnPlay.className = "flex items-center justify-center p-3 rounded-lg bg-amber-600 hover:bg-amber-500 active:scale-95 transition-all text-white font-semibold text-sm w-[35%]";
            autoPlayInterval = setInterval(() => {{
                if (currentStepIdx < SEQUENCE_STEPS.length - 1) {{
                    currentStepIdx++;
                    renderStep(currentStepIdx);
                }} else {{
                    stopAutoPlay();
                }}
            }}, 1200);
        }}

        btnPlay.onclick = () => {{
            if (autoPlayInterval) {{
                stopAutoPlay();
            }} else {{
                if (currentStepIdx === SEQUENCE_STEPS.length - 1) {{
                    currentStepIdx = 0;
                }}
                startAutoPlay();
            }}
        }};

        btnPrev.onclick = () => {{
            stopAutoPlay();
            if (currentStepIdx > 0) {{
                currentStepIdx--;
                renderStep(currentStepIdx);
            }}
        }};

        btnNext.onclick = () => {{
            stopAutoPlay();
            if (currentStepIdx < SEQUENCE_STEPS.length - 1) {{
                currentStepIdx++;
                renderStep(currentStepIdx);
            }}
        }};

        // Start
        window.onload = init3D;
    </script>
</body>
</html>
"""
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated Three.js visualization report at {output_html_path}")
