#!/usr/bin/env python3
"""
Reconstruye el reporte HTML para 75295-1 usando el grafo ya generado,
con los renders axonómétricos corregidos y el viewer Three.js interactivo
con drag-to-correct (selección de piezas + desplazamiento snapped a grid LDU).
"""
import os, sys, json, base64, sqlite3, re, math
sys.path.insert(0, '/Users/I764690/Code_personal/LegoGPT')
os.chdir('/Users/I764690/Code_personal/LegoGPT')

import numpy as np
import cv2

from src.ldraw_geometry import LDrawGeometryResolver
from src.step_axonometric_renderer import StepAxonometricRenderer

SET_NUM = '75295-1'
GRAPH_JSON = 'outputs/75295-1_construction_graph.json'
SCRATCH = 'scratch/step_crops_75295-1'
DB_PATH = 'data/catalog/models_catalog.db'
OUTPUT_HTML = 'pdf_extraction_report_75295.html'

def img_to_data_uri(path):
    if not path or not os.path.exists(path): return ''
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

print('=== Loading graph and geometry ===')
geom = LDrawGeometryResolver()
renderer = StepAxonometricRenderer(geom)

with open(GRAPH_JSON) as f:
    graph = json.load(f)
nodes = graph.get('nodes', [])
print(f'Graph nodes: {len(nodes)}')

# --- Load templates from DB ---
conn_db = sqlite3.connect(DB_PATH)
cur = conn_db.cursor()
cur.execute("SELECT inv.set_num, s.name, s.year, t.name, s.num_parts FROM rb_sets s LEFT JOIN rb_themes t ON s.theme_id=t.id JOIN rb_inventories inv ON s.set_num=inv.set_num WHERE s.set_num=? LIMIT 1", ('75295-1',))
set_row = cur.fetchone() or ('75295-1','Millennium Falcon Microfighter',2021,'Star Wars',101)

# Load inventory templates from scratch dir
templates = []
for fn in os.listdir(f'scratch/inventory_templates_{SET_NUM}'):
    if fn.endswith('.png') and not fn.startswith('annotated'):
        templates.append({'element_id': fn.replace('.png',''), 'image_path': f'scratch/inventory_templates_{SET_NUM}/{fn}',
                          'part_name': fn.replace('.png',''), 'color_name': '', 'color_rgb': '#94a3b8', 'qty': 1, 'page_num': 42})

tmpl_by_element = {t['element_id']: t for t in templates}

# Enrich templates from DB
cur.execute("SELECT e.element_id, p.name, c.name, c.rgb FROM rb_elements e JOIN rb_parts p ON e.part_num=p.part_num JOIN rb_colors c ON e.color_id=c.id WHERE e.element_id IN ({}) ".format(','.join('?'*len(templates))), [t['element_id'] for t in templates])
for row in cur.fetchall():
    el, pname, cname, crgb = row
    if el in tmpl_by_element:
        tmpl_by_element[el]['part_name'] = pname or el
        tmpl_by_element[el]['color_name'] = cname or ''
        tmpl_by_element[el]['color_rgb'] = f'#{crgb}' if crgb else '#94a3b8'

# --- LEGO color map ---
LEGO_COLORS = {}
cur.execute('SELECT id, name, rgb FROM rb_colors')
for cid, cname, crgb in cur.fetchall():
    LEGO_COLORS[int(cid)] = {'name': cname or '', 'hex': f'#{crgb}' if crgb else '#94a3b8'}
conn_db.close()

# LDraw color map for rendering
LDRAW_COLOR_MAP = {
    0:'#1b2a4a', 1:'#1e40af', 2:'#166534', 3:'#dc2626', 4:'#b91c1c', 5:'#be185d',
    6:'#92400e', 7:'#94a3b8', 8:'#475569', 9:'#93c5fd', 10:'#4ade80', 11:'#22d3ee',
    14:'#eab308', 15:'#f8fafc', 16:'#94a3b8', 19:'#fef3c7', 26:'#a855f7',
    70:'#d97706', 71:'#94a3b8', 72:'#64748b', 78:'#fde68a', 85:'#6b7280',
    86:'#9ca3af', 272:'#374151'
}

def get_color_hex(color_code):
    c = LEGO_COLORS.get(int(color_code), {})
    if c.get('hex') and c['hex'] != '#None': return c['hex']
    return LDRAW_COLOR_MAP.get(int(color_code), '#94a3b8')

# --- Render axonometric images per step ---
print('=== Rendering step images ===')
nodes_by_page = {}
for n in nodes:
    nodes_by_page.setdefault(n['page_num'], []).append(n)

all_pages = sorted(nodes_by_page.keys())
sim_uris = {}  # page_num -> data URI
for page_num in all_pages:
    page_nodes = nodes_by_page[page_num]
    sim_path = f'{SCRATCH}/page_{page_num:02d}_ldraw_sim.png'
    if os.path.exists(sim_path):
        sim_uris[page_num] = img_to_data_uri(sim_path)
        print(f'  Page {page_num}: loaded existing sim image ({os.path.getsize(sim_path)//1024}KB)')
    else:
        try:
            cumulative = [n for n in nodes if n['page_num'] <= page_num and not n.get('is_minifig', False)]
            sim_img = renderer.render_step_assembly(
                current_page_num=page_num,
                step_nodes=page_nodes,
                all_nodes=cumulative,
                tmpl_by_element=tmpl_by_element,
                is_flip=False
            )
            cv2.imwrite(sim_path, sim_img)
            sim_uris[page_num] = img_to_data_uri(sim_path)
            print(f'  Page {page_num}: rendered ({sim_img.shape})')
        except Exception as e:
            print(f'  Page {page_num}: render failed ({e})')
            sim_uris[page_num] = ''

# Assembly crops
assembly_uris = {}
pdf_path = 'data/manual/official/75295-1_Millennium_Falcon_Microfighter.pdf'
for page_num in all_pages:
    crop_path = f'{SCRATCH}/page_{page_num:02d}_assembly.png'
    assembly_uris[page_num] = img_to_data_uri(crop_path)

print(f'=== Building Three.js data (101 nodes) ===')

# Build viewer_nodes: all parts with full metadata for Three.js
viewer_nodes = []
for n in nodes:
    tf = np.array(n['absolute_transform'], dtype=np.float32).reshape(4,4)
    part_id = n['part_id']
    color_code = n.get('color_code', 16)
    meta = geom.get_part_meta(part_id)
    bbox = meta.get('bbox', [-10, 0, -10, 10, 8, 10])
    el_id = n.get('element_id', '')
    t_info = tmpl_by_element.get(el_id, {})
    connections = n.get('connections', [])
    conn_info = []
    for c in connections:
        conn_info.append({
            'parent_idx': c.get('parent_part_index'),
            'family': c.get('connector_family', 'STUD'),
            'offset': c.get('relative_offset_ldu', [0,0,0]),
            'contacts': c.get('num_contacts', 1)
        })
    viewer_nodes.append({
        'part_index': n['part_index'],
        'part_id': part_id,
        'element_id': el_id,
        'part_name': t_info.get('part_name') or part_id.replace('.dat',''),
        'color_code': int(color_code),
        'color_hex': get_color_hex(color_code),
        'color_name': t_info.get('color_name') or LEGO_COLORS.get(int(color_code),{}).get('name',''),
        'page_num': n['page_num'],
        'level_plates': n.get('level_plates', 0),
        'is_minifig': n.get('is_minifig', False),
        'subassembly_id': n.get('subassembly_id', 'main_model'),
        'transform': tf.tolist(),
        'position': [float(tf[0,3]), float(tf[1,3]), float(tf[2,3])],
        'bbox': [float(x) for x in bbox],
        'connections': conn_info,
        'collision_passed': n.get('collision_check', {}).get('passed', True),
    })

# Build geometries data: mesh vertices per unique part_id
print('=== Extracting LDraw mesh vertices for WebGL ===')
viewer_geometries = {}
unique_pids = list(set(n['part_id'] for n in nodes))
for pid in unique_pids:
    try:
        faces = geom.get_part_faces(pid)
        verts = []
        for face in faces[:800]:  # cap per part
            for pt in face['pts']:
                verts.extend([float(pt[0]), float(pt[1]), float(pt[2])])
        if len(verts) >= 9:
            viewer_geometries[pid] = verts
    except Exception:
        pass
print(f'  Geometries extracted: {len(viewer_geometries)} unique parts')

# --- Build step cards HTML ---
print('=== Building HTML ===')
step_cards_html = ''
for page_num in all_pages:
    page_nodes_list = nodes_by_page[page_num]
    is_mf_page = all(n.get('is_minifig', False) for n in page_nodes_list)  # TRUE only if ALL are minifig
    sim_uri = sim_uris.get(page_num, '')
    asm_uri = assembly_uris.get(page_num, '')

    # Connection cards per node
    conn_cards = ''
    for n in page_nodes_list:
        part_id = n['part_id']
        color_code = n.get('color_code', 16)
        color_hex = get_color_hex(color_code)
        el_id = n.get('element_id', '')
        t_info = tmpl_by_element.get(el_id, {})
        part_name = t_info.get('part_name') or part_id.replace('.dat','')
        color_name = t_info.get('color_name') or ''
        t_uri = img_to_data_uri(t_info.get('image_path',''))
        lvl = n.get('level_plates', 0)
        conns = n.get('connections', [])
        is_mf = n.get('is_minifig', False)
        mf_role = n.get('minifig_role', '')
        col_check = n.get('collision_check', {})
        col_ok = col_check.get('passed', True)
        col_badge = '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">&#x1F6E1; Col: 0 LDU</span>' if col_ok else '<span class="px-1.5 py-0.5 rounded text-[9px] font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40">&#x26A0; Conflicto</span>'

        if not conns:
            conn_cards += f'''
            <div class="p-3 rounded-xl bg-slate-950/70 border border-emerald-500/30 flex items-center gap-3">
              <div class="w-10 h-10 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0 border border-slate-700"><img src="{t_uri}" class="max-w-full max-h-full object-contain"/></div>
              <div><div class="flex items-center gap-2 flex-wrap">
                <span class="text-xs font-mono font-bold text-white">#{n['part_index']}</span>
                <span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">{"&#x1F9B5; Minifig " + mf_role if is_mf else "&#x1F7E2; Pieza Raiz L="+str(lvl)}</span>
                {col_badge}
              </div>
              <span class="text-xs text-slate-300">{el_id} &middot; {color_name} &middot; {part_name}</span></div>
            </div>'''
        else:
            for c in conns:
                parent_idx = c.get('parent_part_index')
                family = c.get('connector_family','STUD')
                offset = c.get('relative_offset_ldu',[0,0,0])
                n_contacts = c.get('num_contacts', 1)
                family_color = {'STUD':'emerald','TECHNIC_PIN':'sky','TECHNIC_AXLE':'violet','UNDERSIDE_STUD_SOCKET':'amber'}.get(family,'slate')
                conn_cards += f'''
                <div class="p-3 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-{family_color}-500/40 transition-all flex items-start gap-3">
                  <div class="w-10 h-10 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0 border border-slate-700"><img src="{t_uri}" class="max-w-full max-h-full object-contain"/></div>
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2 flex-wrap mb-1">
                      <span class="text-xs font-mono font-bold text-white">#{n['part_index']}</span>
                      <span class="px-1.5 py-0.5 rounded text-[9px] bg-{family_color}-500/20 text-{family_color}-300 border border-{family_color}-500/30 font-mono">{family}</span>
                      {col_badge}
                    </div>
                    <span class="text-xs text-slate-300 block">{el_id} &middot; {color_name} &middot; {part_name}</span>
                    <span class="text-[10px] text-slate-500">Parent #{parent_idx} &middot; offset [{offset[0]},{offset[1]},{offset[2]}] LDU &middot; {n_contacts} contactos</span>
                  </div>
                </div>'''

    # Sim image block
    sim_block = ''
    if sim_uri and not is_mf_page:
        sim_block = f'''
        <div class="glass-panel rounded-xl p-3 border border-indigo-500/30">
          <div class="text-[10px] text-indigo-400 font-bold uppercase tracking-wide mb-2">&#x1F916; Digital Twin LDraw &middot; Pág. {page_num} (acumulado)</div>
          <img src="{sim_uri}" alt="Sim Pag {page_num}" class="w-full max-h-52 object-contain rounded" />
        </div>'''

    # Jump to 3D viewer button
    jump_btn = f'<button onclick="jumpTo3DStep({page_num})" class="px-3 py-1.5 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 text-xs font-semibold">&#x1F3AE; Ver en Viewer 3D</button>'

    mf_badge = '<span class="px-2 py-0.5 rounded-full text-[10px] bg-purple-600/30 text-purple-300 border border-purple-500/40 font-bold">&#x1F464; Subgrafo Minifigura</span>' if is_mf_page else ''

    step_cards_html += f'''
    <div id="page-{page_num}" class="glass-panel rounded-2xl p-5 border border-slate-800/80 scroll-mt-24">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <span class="text-2xl font-extrabold text-white">{n["step_label"] if page_nodes_list else "Paso"}</span>
          <span class="px-2.5 py-1 rounded-lg bg-slate-800 text-slate-300 text-xs font-mono">Pág. {page_num}</span>
          {mf_badge}
          {jump_btn}
        </div>
        <span class="text-xs text-slate-500">{len(page_nodes_list)} piezas</span>
      </div>
      <div class="grid grid-cols-1 {'md:grid-cols-2' if asm_uri else ''} gap-4 mb-4">
        {'<div class="glass-panel rounded-xl p-3 border border-slate-800"><div class="text-[10px] text-slate-400 font-bold uppercase mb-2">&#x1F4D6; Manual PDF &middot; Pág. ' + str(page_num) + '</div><img src="' + asm_uri + '" class="w-full max-h-52 object-contain rounded"/></div>' if asm_uri else ''}
        {sim_block}
      </div>
      <div class="space-y-2">{conn_cards}</div>
    </div>'''

print('=== Building final HTML with drag-to-correct Three.js viewer ===')
viewer_nodes_json = json.dumps(viewer_nodes, indent=None, ensure_ascii=False)
viewer_geometries_json = json.dumps(viewer_geometries, indent=None, ensure_ascii=False)

min_page = min(all_pages)
max_page_num = max(all_pages)

col_ok_count = sum(1 for n in nodes if n.get('collision_check', {}).get('passed', True))
HTML = f'''<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LegoGPT &mdash; Set 75295-1 Millennium Falcon Microfighter &mdash; Reporte 3D</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: \'Outfit\', sans-serif; background:#070b14; color:#f8fafc; }}
    .glass-panel {{ background:rgba(15,23,42,0.72); backdrop-filter:blur(16px); border:1px solid rgba(255,255,255,0.07); }}
    .gradient-text {{ background:linear-gradient(135deg,#fff 0%,#93c5fd 55%,#34d399 100%); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
    ::-webkit-scrollbar {{ width:6px; }} ::-webkit-scrollbar-track {{ background:#0f172a; }} ::-webkit-scrollbar-thumb {{ background:#334155; border-radius:3px; }}
    .part-selected {{ outline: 2px solid #f59e0b !important; outline-offset: 2px; }}
    #correction-panel {{ transition: all 0.3s; }}
    .snap-indicator {{ position:absolute; pointer-events:none; border:2px dashed #f59e0b; border-radius:4px; }}
    @keyframes pulse-border {{ 0%,100% {{ border-color:#f59e0b; }} 50% {{ border-color:#fbbf24; }} }}
  </style>
</head>
<body class="min-h-screen">

<!-- HEADER -->
<header class="max-w-7xl mx-auto px-4 pt-8 pb-4">
  <div class="glass-panel rounded-3xl p-8">
    <div class="flex items-center gap-2 text-emerald-400 font-bold text-xs tracking-widest uppercase mb-2">
      <span>&#x1F916; LegoGPT Manual Vision Pipeline &mdash; {SET_NUM}</span>
      <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
    </div>
    <h1 class="text-3xl sm:text-4xl font-extrabold gradient-text mb-2">{set_row[1]}</h1>
    <p class="text-sm text-slate-400">Extracción automática del manual &middot; <strong>{len(nodes)}</strong> piezas &middot; <strong>{len(all_pages)}</strong> pasos &middot; Grafo de encaje 3D + Viewer WebGL interactivo con corrección manual</p>
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
      <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl text-center"><span class="text-[10px] text-slate-400 uppercase font-bold block">Total Piezas</span><span class="text-2xl font-extrabold text-amber-400">{len(nodes)}</span></div>
      <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl text-center"><span class="text-[10px] text-slate-400 uppercase font-bold block">Pasos</span><span class="text-2xl font-extrabold text-emerald-400">{len(all_pages)}</span></div>
      <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl text-center"><span class="text-[10px] text-slate-400 uppercase font-bold block">Col. OK</span><span class="text-2xl font-extrabold text-sky-400">{col_ok_count}</span></div>
      <div class="bg-slate-950/80 border border-slate-800 px-4 py-3 rounded-2xl text-center"><span class="text-[10px] text-slate-400 uppercase font-bold block">Geometrías 3D</span><span class="text-2xl font-extrabold text-purple-400">{len(viewer_geometries)}</span></div>
    </div>
    <nav class="flex flex-wrap gap-3 mt-4 text-xs">
      <a href="#sec-3d" class="px-3 py-1.5 rounded-lg bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-600/40 font-semibold">&#x1F3AE; Viewer 3D Interactivo</a>
      <a href="#sec-steps" class="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 hover:bg-slate-700 font-semibold">&#x1F527; Pasos de Montaje</a>
      <a href="#sec-corrections" class="px-3 py-1.5 rounded-lg bg-amber-600/20 text-amber-300 border border-amber-500/30 hover:bg-amber-600/40 font-semibold">&#x270F;&#xFE0F; Correcciones Exportadas</a>
    </nav>
  </div>
</header>

<main class="max-w-7xl mx-auto px-4 py-6 space-y-10">

  <!-- SECTION: INTERACTIVE 3D VIEWER -->
  <section id="sec-3d" class="glass-panel rounded-3xl p-6 border border-indigo-500/40 scroll-mt-20">
    <div class="flex flex-col lg:flex-row lg:items-start justify-between gap-4 mb-5">
      <div>
        <div class="flex items-center gap-2 text-indigo-400 font-bold text-xs uppercase tracking-wider mb-1">
          <span>&#x1F3AE; Digital Twin 3D &middot; WebGL Three.js &middot; {len(nodes)} nodos</span>
        </div>
        <h2 class="text-2xl font-extrabold text-white">Viewer Interactivo Paso a Paso</h2>
        <p class="text-xs text-slate-400 mt-1">Arrastra la vista &middot; Rueda = zoom &middot; <strong class="text-amber-300">Clic en una pieza</strong> para seleccionarla &middot; Con pieza seleccionada: <kbd class="bg-slate-800 px-1 rounded">WASD</kbd> / <kbd class="bg-slate-800 px-1 rounded">QE</kbd> = mover en grid de 10 LDU &middot; <kbd class="bg-slate-800 px-1 rounded">R</kbd> = rotar 90° &middot; <kbd class="bg-slate-800 px-1 rounded">Enter</kbd> = confirmar corrección</p>
      </div>
      <!-- HUD -->
      <div class="flex items-center gap-3 flex-wrap shrink-0">
        <div class="px-4 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center min-w-[90px]"><span class="text-[10px] text-slate-400 uppercase font-bold block">Paso</span><span id="hud-step" class="text-base font-extrabold text-amber-400 font-mono">P.{min_page}</span></div>
        <div class="px-4 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center min-w-[90px]"><span class="text-[10px] text-slate-400 uppercase font-bold block">Piezas</span><span id="hud-parts" class="text-base font-extrabold text-emerald-400 font-mono">0</span></div>
        <div id="hud-selected" class="px-4 py-2 rounded-xl bg-amber-950/80 border border-amber-500/30 text-center min-w-[120px] hidden"><span class="text-[10px] text-amber-400 uppercase font-bold block">Seleccionada</span><span id="hud-selected-name" class="text-xs font-bold text-amber-300 font-mono"></span></div>
      </div>
    </div>

    <!-- Controls -->
    <div class="flex flex-wrap items-center justify-between gap-3 mb-4 bg-slate-950/80 p-3 rounded-2xl border border-slate-800">
      <div class="flex items-center gap-3 flex-1 min-w-[300px]">
        <button id="btn-play" onclick="togglePlay()" class="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center font-bold text-sm shrink-0">&#x25B6;</button>
        <button onclick="stepViewer(-1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold shrink-0">&#x25C0;</button>
        <input id="timeline" type="range" min="{min_page}" max="{max_page_num}" value="{min_page}" oninput="onSlider(this.value)" class="w-full accent-indigo-500 cursor-pointer h-2 bg-slate-800 rounded-lg"/>
        <button onclick="stepViewer(1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold shrink-0">&#x25B6;</button>
      </div>
      <div class="flex items-center gap-2">
        <button onclick="setCam(\'iso\')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Axon.</button>
        <button onclick="setCam(\'top\')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Top</button>
        <button onclick="setCam(\'front\')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Frontal</button>
        <button onclick="setCam(\'side\')" class="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Lateral</button>
        <button onclick="resetCam()" class="px-2.5 py-1 text-xs rounded-lg bg-indigo-800/50 hover:bg-indigo-700/50 text-indigo-300">Centrar</button>
      </div>
    </div>

    <!-- Canvas + Part Info Overlay -->
    <div class="relative w-full rounded-2xl overflow-hidden border border-slate-800" style="height:560px; background:#06080e;">
      <div id="viewer-container" class="w-full h-full"></div>

      <!-- Keyboard help overlay -->
      <div class="absolute top-3 left-3 pointer-events-none bg-slate-950/80 backdrop-blur-md px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] text-slate-400">
        &#x1F5B1;&#xFE0F; <b>Arrastrar</b>: rotar &middot; <b>Rueda</b>: zoom &middot; <b>Clic</b>: seleccionar pieza
      </div>

      <!-- Selected part correction mode hint -->
      <div id="correction-hint" class="absolute top-3 right-3 pointer-events-none hidden bg-amber-950/90 backdrop-blur-md px-3 py-2 rounded-xl border border-amber-500/40 text-xs text-amber-300">
        &#x270F;&#xFE0F; <b>Modo Corrección</b><br>
        <span class="text-amber-400/80">WASD = mover XZ &middot; QE = mover Y</span><br>
        <span class="text-amber-400/80">R = rotar 90° &middot; Enter = confirmar</span><br>
        <span class="text-amber-400/80">Esc = cancelar</span>
      </div>

      <!-- Part info overlay -->
      <div id="part-overlay" class="absolute bottom-3 left-3 right-3 bg-slate-950/90 backdrop-blur-md p-3 rounded-xl border border-slate-800 flex items-center justify-between opacity-0 transition-opacity pointer-events-none">
        <div class="flex items-center gap-3">
          <div id="overlay-swatch" class="w-4 h-4 rounded-full border border-slate-600 shrink-0 bg-slate-400"></div>
          <div>
            <span id="overlay-title" class="text-sm font-bold text-white block"></span>
            <span id="overlay-sub" class="text-xs text-slate-400"></span>
          </div>
        </div>
        <div class="flex items-center gap-2">
          <span id="overlay-conn" class="text-[10px] text-slate-500"></span>
          <button id="overlay-correct-btn" onclick="enterCorrectionMode()" class="hidden px-3 py-1 rounded-lg bg-amber-600/30 hover:bg-amber-600/50 text-amber-300 border border-amber-500/30 text-xs font-semibold">&#x270F; Corregir posición</button>
        </div>
      </div>
    </div>
  </section>

  <!-- SECTION: CORRECTION LOG -->
  <section id="sec-corrections" class="glass-panel rounded-3xl p-6 border border-amber-500/30 scroll-mt-20">
    <div class="flex items-center justify-between mb-4">
      <div>
        <h2 class="text-xl font-extrabold text-white">&#x270F;&#xFE0F; Registro de Correcciones Manuales</h2>
        <p class="text-xs text-slate-400 mt-1">Cada corracción (mover una pieza) genera un delta JSON que puede retroalimentar el entrenamiento del re-ranker GNN.</p>
      </div>
      <div class="flex items-center gap-2">
        <span id="correction-count" class="px-3 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-bold">0 correcciones</span>
        <button onclick="exportCorrections()" class="px-4 py-2 rounded-xl bg-amber-600/30 hover:bg-amber-600/50 text-amber-300 border border-amber-500/30 text-sm font-semibold">&#x2B07; Exportar JSON</button>
        <button onclick="clearCorrections()" class="px-4 py-2 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 text-sm font-semibold">&#x1F5D1; Limpiar</button>
      </div>
    </div>
    <div id="corrections-list" class="space-y-2 text-xs text-slate-400"><em>Sin correcciones aún. Selecciona una pieza en el viewer 3D y presiona &ldquo;Corregir posición&rdquo; para empezar.</em></div>
  </section>

  <!-- SECTION: STEP-BY-STEP GRAPH -->
  <section id="sec-steps" class="space-y-6 scroll-mt-20">
    <div>
      <h2 class="text-2xl font-bold text-white">&#x1F527; Grafo de Montaje Paso a Paso</h2>
      <p class="text-sm text-slate-400 mt-1">Para cada página: imagen del manual (izq.) + render digital twin LDraw acumulado (der.) + tarjetas de conexión física stud-to-socket.</p>
    </div>
    {step_cards_html}
  </section>

</main>

<!-- DATA INJECTION -->
<script id="assembly-data" type="application/json">
{viewer_nodes_json}
</script>
<script id="geometries-data" type="application/json">
{viewer_geometries_json}
</script>

<script>
(function(){{
  // ---- Data loading ----
  let partsData = [], geomData = {{}};
  try {{ partsData = JSON.parse(document.getElementById(\'assembly-data\').textContent); }} catch(e) {{}}
  try {{ geomData = JSON.parse(document.getElementById(\'geometries-data\').textContent); }} catch(e) {{}}

  const allPages = [...new Set(partsData.map(p=>p.page_num))].sort((a,b)=>a-b);
  const minPage = allPages[0] || 1;

  // ---- Correction state ----
  const corrections = [];
  let selectedMesh = null;
  let correctionMode = false;
  let correctionOrigPos = null;

  function updateCorrectionCount() {{
    document.getElementById(\'correction-count\').textContent = corrections.length + \' correcciones\';
    const list = document.getElementById(\'corrections-list\');
    if (corrections.length === 0) {{
      list.innerHTML = \'<em>Sin correcciones aún.</em>\';
      return;
    }}
    list.innerHTML = corrections.map((c,i) => `
      <div class="p-3 rounded-xl bg-slate-950/70 border border-amber-500/20 font-mono">
        <div class="flex items-center justify-between">
          <span class="text-amber-300 font-bold">#${{i+1}} Pieza #${{c.part_index}} (${{c.part_id}})</span>
          <button onclick="removeCorrection(${{i}})" class="text-rose-400 hover:text-rose-300 text-xs">&#x2715;</button>
        </div>
        <div class="text-slate-400 text-[10px] mt-1">
          Antes: [${{c.old_pos.map(v=>v.toFixed(1)).join(', ')}}] &rarr; Después: [${{c.new_pos.map(v=>v.toFixed(1)).join(', ')}}
          | &Delta;R: ${{c.rot_delta_deg}}° | Pág. ${{c.page_num}}
        </div>
      </div>
    `).join(\'\');
  }}

  window.removeCorrection = (i) => {{ corrections.splice(i, 1); updateCorrectionCount(); }};

  window.exportCorrections = () => {{
    if (!corrections.length) {{ alert(\'Sin correcciones para exportar.\'); return; }}
    const blob = new Blob([JSON.stringify({{ set_num: \'{SET_NUM}\', corrections }}, null, 2)], {{type:\'application/json\'}})
    const a = document.createElement(\'a\'); a.href = URL.createObjectURL(blob);
    a.download = \'corrections_{SET_NUM}_\' + Date.now() + \'.json\';
    a.click();
  }};

  window.clearCorrections = () => {{ corrections.length = 0; updateCorrectionCount(); }};

  // ---- Three.js setup ----
  const container = document.getElementById(\'viewer-container\');
  const W = container.clientWidth || 900, H = container.clientHeight || 560;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06080e);
  scene.fog = new THREE.FogExp2(0x06080e, 0.0018);

  const camera = new THREE.PerspectiveCamera(38, W/H, 1, 4000);
  camera.position.set(240, 200, 300);

  const renderer_3d = new THREE.WebGLRenderer({{antialias:true, alpha:true, powerPreference:\'high-performance\'}})
  renderer_3d.setSize(W, H);
  renderer_3d.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer_3d.shadowMap.enabled = true;
  renderer_3d.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer_3d.domElement);

  const controls = new THREE.OrbitControls(camera, renderer_3d.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06;
  controls.maxDistance = 2000; controls.minDistance = 20;

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dl = new THREE.DirectionalLight(0xffffff, 1.0);
  dl.position.set(200, 400, 200); dl.castShadow = true; scene.add(dl);
  const dl2 = new THREE.DirectionalLight(0x93c5fd, 0.45);
  dl2.position.set(-200, 150, -200); scene.add(dl2);
  const grid = new THREE.GridHelper(600, 30, 0x1e293b, 0x0f172a);
  grid.position.y = 0.5; scene.add(grid);

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();
  const partMeshes = [];
  let currentPage = minPage;
  let playInterval = null;

  // ---- Material factory ----
  const matCache = {{}};
  function getMat(hex, selected=false) {{
    const k = hex + (selected?\'_sel\':\'\');
    if (!matCache[k]) matCache[k] = new THREE.MeshLambertMaterial({{
      color: new THREE.Color(hex || \'#94a3b8\'),
      transparent: selected, opacity: selected ? 0.85 : 1.0,
      emissive: selected ? new THREE.Color(0xf59e0b) : new THREE.Color(0x000000),
      emissiveIntensity: selected ? 0.35 : 0.0
    }});
    return matCache[k];
  }}

  // ---- Geometry factory ----
  const geoCache = {{}};
  function buildGeo(part) {{
    const pid = part.part_id;
    if (geoCache[pid]) return geoCache[pid];
    const verts = geomData[pid];
    if (verts && verts.length >= 9) {{
      const g = new THREE.BufferGeometry();
      g.setAttribute(\'position\', new THREE.Float32BufferAttribute(verts, 3));
      g.computeVertexNormals();
      geoCache[pid] = g; return g;
    }}
    const [bx0,by0,bz0,bx1,by1,bz1] = part.bbox || [-10,0,-10,10,8,10];
    const dx=Math.max(1,bx1-bx0), dy=Math.max(1,by1-by0), dz=Math.max(1,bz1-bz0);
    const g = new THREE.BoxGeometry(dx,dy,dz);
    geoCache[pid] = g; return g;
  }}

  const studGeoTemplate = new THREE.CylinderGeometry(5.5, 5.5, 3.5, 12);

  // ---- Build all part meshes ----
  partsData.forEach(part => {{
    const group = new THREE.Group();
    const geo = buildGeo(part);
    const mat = getMat(part.color_hex);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = mesh.receiveShadow = true;

    // LDraw transform: matrix is row-major 4x4
    const tf = part.transform;
    if (tf && tf.length === 16) {{
      mesh.matrix.set(...tf.flat ? tf.flat() : tf.reduce((a,b)=>a.concat(b),[]));
      mesh.matrixAutoUpdate = false;
    }} else {{
      // Fallback: use position directly
      mesh.position.set(part.position[0]/20, -part.position[1]/20, part.position[2]/20);
    }}

    group.add(mesh);
    group.userData = part;
    group.visible = false;
    scene.add(group);
    partMeshes.push({{ group, mesh, part, origColor: part.color_hex }});
  }});

  // ---- Step view ----
  function updateStepView(page) {{
    currentPage = page;
    let visCount = 0;
    partMeshes.forEach(pm => {{
      const show = pm.part.page_num <= page;
      pm.group.visible = show;
      if (show) {{
        // Highlight newly added parts
        const isNew = pm.part.page_num === page;
        pm.mesh.material = getMat(pm.origColor, false);
        if (isNew) {{
          pm.mesh.material = new THREE.MeshLambertMaterial({{
            color: new THREE.Color(pm.origColor),
            emissive: new THREE.Color(0x4ade80), emissiveIntensity: 0.45
          }});
        }}
        visCount++;
      }}
    }});
    document.getElementById(\'hud-step\').textContent = \'P.\' + page;
    document.getElementById(\'hud-parts\').textContent = visCount + \' piezas\';
    document.getElementById(\'timeline\').value = page;
    // Sync scroll to step card
    const card = document.getElementById(\'page-\' + page);
    if (card) card.scrollIntoView({{behavior:\'smooth\', block:\'nearest\'}});
  }}

  window.jumpTo3DStep = (p) => {{
    const sec = document.getElementById(\'sec-3d\');
    if (sec) sec.scrollIntoView({{behavior:\'smooth\'}});
    setTimeout(() => updateStepView(p), 400);
  }};

  window.onSlider = (v) => updateStepView(parseInt(v));

  window.stepViewer = (delta) => {{
    const idx = allPages.indexOf(currentPage);
    const next = allPages[Math.max(0, Math.min(allPages.length-1, idx+delta))];
    updateStepView(next);
  }};

  window.togglePlay = () => {{
    const btn = document.getElementById(\'btn-play\');
    if (playInterval) {{ clearInterval(playInterval); playInterval=null; btn.innerHTML=\'&#x25B6;\'; }}
    else {{
      btn.innerHTML = \'&#x23F8;\';
      playInterval = setInterval(() => {{
        const idx = allPages.indexOf(currentPage);
        if (idx >= allPages.length-1) {{ clearInterval(playInterval); playInterval=null; btn.innerHTML=\'&#x25B6;\'; }}
        else updateStepView(allPages[idx+1]);
      }}, 900);
    }}
  }};

  window.setCam = (v) => {{
    controls.target.set(0,0,0);
    if (v===\'iso\') {{ camera.position.set(240,200,300); }}
    else if (v===\'top\') {{ camera.position.set(0,600,0); }}
    else if (v===\'front\') {{ camera.position.set(0,100,500); }}
    else if (v===\'side\') {{ camera.position.set(500,100,0); }}
    controls.update();
  }};

  window.resetCam = () => {{
    const vis = partMeshes.filter(pm=>pm.group.visible);
    if (!vis.length) return;
    const box = new THREE.Box3();
    vis.forEach(pm => box.expandByObject(pm.group));
    const center = new THREE.Vector3(); box.getCenter(center);
    const size = new THREE.Vector3(); box.getSize(size);
    controls.target.copy(center);
    camera.position.set(center.x + size.length()*0.8, center.y + size.length()*0.5, center.z + size.length()*0.8);
    controls.update();
  }};

  // ---- Click selection ----
  renderer_3d.domElement.addEventListener(\'click\', (e) => {{
    if (correctionMode) return;
    const rect = container.getBoundingClientRect();
    mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const visGroups = partMeshes.filter(pm=>pm.group.visible).map(pm=>pm.mesh);
    const hits = raycaster.intersectObjects(visGroups, false);
    if (hits.length) {{
      const hitMesh = hits[0].object;
      const pm = partMeshes.find(pm=>pm.mesh === hitMesh);
      if (pm) selectPart(pm);
    }} else {{
      deselectPart();
    }}
  }});

  function selectPart(pm) {{
    if (selectedMesh) deselectPart();
    selectedMesh = pm;
    pm.mesh.material = getMat(pm.origColor, true);
    const p = pm.part;
    document.getElementById(\'overlay-title\').textContent = `#${{p.part_index}} · ${{p.part_id}} (${{p.part_name}})`;
    document.getElementById(\'overlay-sub\').textContent = `Elem ${{p.element_id}} · ${{p.color_name}} · Pág. ${{p.page_num}} · Nivel L=${{p.level_plates}}`;
    document.getElementById(\'overlay-swatch\').style.backgroundColor = p.color_hex;
    document.getElementById(\'overlay-conn\').textContent = (p.connections||[]).length + \' conexiones\';
    document.getElementById(\'part-overlay\').style.opacity = \'1\';
    document.getElementById(\'overlay-correct-btn\').classList.remove(\'hidden\');
    document.getElementById(\'hud-selected\').classList.remove(\'hidden\');
    document.getElementById(\'hud-selected-name\').textContent = `#${{p.part_index}} ${{p.part_id}}`;
  }}

  function deselectPart() {{
    if (!selectedMesh) return;
    selectedMesh.mesh.material = getMat(selectedMesh.origColor, false);
    selectedMesh = null;
    document.getElementById(\'part-overlay\').style.opacity = \'0\';
    document.getElementById(\'overlay-correct-btn\').classList.add(\'hidden\');
    document.getElementById(\'hud-selected\').classList.add(\'hidden\');
    document.getElementById(\'correction-hint\').classList.add(\'hidden\');
    correctionMode = false; correctionOrigPos = null;
  }}

  // ---- Correction mode: snap-to-grid keyboard move ----
  const SNAP = 10; // 10 LDU = half stud
  window.enterCorrectionMode = () => {{
    if (!selectedMesh) return;
    correctionMode = true;
    correctionOrigPos = {{...selectedMesh.group.position}};
    controls.enabled = false;
    document.getElementById(\'correction-hint\').classList.remove(\'hidden\');
  }};

  window.addEventListener(\'keydown\', (e) => {{
    if (!selectedMesh) return;
    if (!correctionMode) {{
      if (e.key === \'c\' || e.key === \'C\') window.enterCorrectionMode();
      return;
    }}
    const step = e.shiftKey ? SNAP*2 : SNAP;
    const g = selectedMesh.group;
    let moved = true;
    switch(e.key) {{
      case \'w\': case \'W\': g.position.z -= step; break;
      case \'s\': case \'S\': g.position.z += step; break;
      case \'a\': case \'A\': g.position.x -= step; break;
      case \'d\': case \'D\': g.position.x += step; break;
      case \'q\': case \'Q\': g.position.y += step; break;
      case \'e\': case \'E\': g.position.y -= step; break;
      case \'r\': case \'R\': g.rotation.y += Math.PI/2; break;
      case \'Enter\':
        confirmCorrection(); moved=false; break;
      case \'Escape\':
        g.position.set(correctionOrigPos.x, correctionOrigPos.y, correctionOrigPos.z);
        correctionMode = false; controls.enabled = true;
        document.getElementById(\'correction-hint\').classList.add(\'hidden\');
        moved = false; break;
    }}
    if (moved) {{ e.preventDefault(); }}
  }});

  function confirmCorrection() {{
    if (!selectedMesh || !correctionOrigPos) return;
    const p = selectedMesh.part;
    const newPos = [selectedMesh.group.position.x, selectedMesh.group.position.y, selectedMesh.group.position.z];
    const rotDeg = Math.round(THREE.MathUtils.radToDeg(selectedMesh.group.rotation.y) / 90) * 90;
    corrections.push({{
      part_index: p.part_index,
      part_id: p.part_id,
      element_id: p.element_id,
      page_num: p.page_num,
      old_pos: [correctionOrigPos.x, correctionOrigPos.y, correctionOrigPos.z],
      new_pos: newPos,
      rot_delta_deg: rotDeg,
      timestamp: new Date().toISOString(),
      inferred_connections: p.connections || []
    }});
    updateCorrectionCount();
    correctionMode = false; correctionOrigPos = null;
    controls.enabled = true;
    document.getElementById(\'correction-hint\').classList.add(\'hidden\');
    // Flash confirm
    const mat = selectedMesh.mesh.material;
    mat.emissive.setHex(0x10b981); mat.emissiveIntensity = 0.8;
    setTimeout(() => {{ mat.emissive.setHex(0xf59e0b); mat.emissiveIntensity = 0.35; }}, 600);
  }}

  // ---- Resize ----
  window.addEventListener(\'resize\', () => {{
    const w=container.clientWidth, h=container.clientHeight;
    camera.aspect=w/h; camera.updateProjectionMatrix(); renderer_3d.setSize(w,h);
  }});

  // ---- Render loop ----
  function animate() {{ requestAnimationFrame(animate); controls.update(); renderer_3d.render(scene,camera); }}
  animate();

  updateStepView(minPage);
}})();
</script>

</body></html>
'''

print(f'=== Writing {OUTPUT_HTML} ===')
with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
    f.write(HTML)

size_mb = os.path.getsize(OUTPUT_HTML) / 1024 / 1024
print(f'\n=== DONE ===')
print(f'Output: {os.path.abspath(OUTPUT_HTML)}')
print(f'Size: {size_mb:.1f} MB')
print(f'Nodes: {len(viewer_nodes)}')
print(f'Pages: {len(all_pages)}')
print(f'Geometries: {len(viewer_geometries)}')
print(f'Sim images: {len([u for u in sim_uris.values() if u])}')
