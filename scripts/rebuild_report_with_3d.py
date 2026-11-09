#!/usr/bin/env python3
"""
rebuild_report_with_3d.py  --  LegoGPT 75295-1 Full 3D Report

Generates a self-contained HTML report with:
  - Solid LDraw geometry: quads fan-triangulated, studs as cylinders,
    MeshStandardMaterial (plastic finish).
  - Subgraph tabs: main_model / minifig_1 / All-together.
  - In "All" mode: two independent timelines side-by-side.
  - Correction mode: WASD/QE translate (snap 10 LDU), T/G/R/F/Y/H = rotate
    +/-90 deg on X/Y/Z axes, Shift = 15 deg fine steps.
    Enter = confirm + export delta JSON with full 3x3 rot_matrix.
  - Step cards with axonometric sim renders per page.
"""
import os, sys, json, base64, sqlite3, re
sys.path.insert(0, '/Users/I764690/Code_personal/LegoGPT')
os.chdir('/Users/I764690/Code_personal/LegoGPT')
import numpy as np
import cv2
from src.ldraw_geometry import LDrawGeometryResolver
from src.step_axonometric_renderer import StepAxonometricRenderer

SET_NUM  = '75295-1'
GRAPH    = 'outputs/75295-1_construction_graph.json'
SCRATCH  = 'scratch/step_crops_75295-1'
DB_PATH  = 'data/catalog/models_catalog.db'
OUT_HTML = 'pdf_extraction_report_75295.html'

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def b64img(path):
    if not path or not os.path.exists(path): return ''
    with open(path,'rb') as f: return 'data:image/png;base64,'+base64.b64encode(f.read()).decode()

# -----------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------
print('=== Loading graph ===')
geom     = LDrawGeometryResolver()
renderer = StepAxonometricRenderer(geom)
with open(GRAPH) as f: graph = json.load(f)
nodes = graph['nodes']
print(f'  {len(nodes)} nodes')

# DB
db  = sqlite3.connect(DB_PATH)
cur = db.cursor()
cur.execute("SELECT s.set_num,s.name,s.year,t.name FROM rb_sets s LEFT JOIN rb_themes t ON s.theme_id=t.id JOIN rb_inventories i ON s.set_num=i.set_num WHERE s.set_num=? LIMIT 1",(SET_NUM,))
set_row = cur.fetchone() or (SET_NUM,'Millennium Falcon Microfighter',2021,'Star Wars')

LEGO_COLORS={}
cur.execute('SELECT id,name,rgb FROM rb_colors')
for cid,cname,crgb in cur.fetchall():
    LEGO_COLORS[int(cid)]={'name':cname or '','hex':f'#{crgb}' if crgb else '#94a3b8'}

# Template metadata from scratch dir
tmpl_by_el={}
tmpl_dir = f'scratch/inventory_templates_{SET_NUM}'
if os.path.isdir(tmpl_dir):
    for fn in os.listdir(tmpl_dir):
        if fn.endswith('.png') and not fn.startswith('annotated'):
            el = fn.replace('.png','')
            tmpl_by_el[el]={'element_id':el,'image_path':os.path.join(tmpl_dir,fn),'part_name':el,'color_name':'','color_rgb':'#94a3b8'}
el_ids = [t['element_id'] for t in tmpl_by_el.values() if t['element_id']]
if el_ids:
    ph = ','.join('?'*len(el_ids))
    cur.execute(f"SELECT e.element_id,p.name,c.name,c.rgb FROM rb_elements e JOIN rb_parts p ON e.part_num=p.part_num JOIN rb_colors c ON e.color_id=c.id WHERE e.element_id IN ({ph})",el_ids)
    for el,pn,cn,cr in cur.fetchall():
        if el in tmpl_by_el:
            tmpl_by_el[el].update({'part_name':pn or el,'color_name':cn or '','color_rgb':f'#{cr}' if cr else '#94a3b8'})
db.close()

LDRAW_CLR={0:'#1b2a4a',1:'#1e40af',2:'#166534',3:'#dc2626',4:'#b91c1c',5:'#be185d',6:'#92400e',7:'#94a3b8',8:'#475569',9:'#93c5fd',10:'#4ade80',11:'#22d3ee',14:'#eab308',15:'#f8fafc',16:'#94a3b8',19:'#fef3c7',26:'#a855f7',70:'#d97706',71:'#94a3b8',72:'#64748b',78:'#fde68a',85:'#6b7280',86:'#9ca3af',272:'#374151'}
def chex(code):
    c=LEGO_COLORS.get(int(code),{})
    h=c.get('hex','')
    return h if h and h!='#None' else LDRAW_CLR.get(int(code),'#94a3b8')

# -----------------------------------------------------------------------
# Geometry extraction  (quad fan-triangulation)
# -----------------------------------------------------------------------
print('=== Extracting geometries ===')
viewer_geos  = {}   # part_id -> flat float32 triangle vertices
viewer_studs = {}   # part_id -> [[x,y,z], ...]  local stud positions (top-facing only)
unique_pids  = list(set(n['part_id'] for n in nodes))
for pid in unique_pids:
    try:
        faces = geom.get_part_faces(pid)
        verts = []
        for face in faces[:3000]:
            pts = face['pts']   # shape (3,3) or (4,3)
            if len(pts) == 3:
                for pt in pts:
                    verts += [float(pt[0]), float(pt[1]), float(pt[2])]
            elif len(pts) == 4:
                A,B,C,D = pts[0],pts[1],pts[2],pts[3]
                for pt in (A,B,C,A,C,D):  # fan split
                    verts += [float(pt[0]), float(pt[1]), float(pt[2])]
        if len(verts) >= 9:
            viewer_geos[pid] = verts
        # top-facing studs for cylinder overlay
        meta = geom.get_part_meta(pid)
        sl = []
        for s in (meta.get('stud_details') or meta.get('studs',[])):
            pos  = s['pos']    if isinstance(s,dict) else s
            norm = s.get('normal',[0,-1,0]) if isinstance(s,dict) else [0,-1,0]
            if hasattr(norm,'__len__'): ny=float(norm[1])
            else: ny=-1.0
            if ny < -0.5:
                sl.append([float(pos[0]),float(pos[1]),float(pos[2])])
        if sl: viewer_studs[pid] = sl
    except Exception:
        pass
print(f'  {len(viewer_geos)} geometries, {len(viewer_studs)} stud maps')

# -----------------------------------------------------------------------
# Build viewer node list
# -----------------------------------------------------------------------
viewer_nodes = []
for n in nodes:
    tf  = np.array(n['absolute_transform'],dtype=np.float32).reshape(4,4)
    pid = n['part_id']
    col = n.get('color_code',16)
    meta = geom.get_part_meta(pid)
    ei  = n.get('element_id','')
    ti  = tmpl_by_el.get(ei,{})
    conns=[{'parent_idx':c.get('parent_part_index'),'family':c.get('connector_family','STUD'),'offset':c.get('relative_offset_ldu',[0,0,0]),'contacts':c.get('num_contacts',1)} for c in n.get('connections',[])]
    viewer_nodes.append({
        'part_index': n['part_index'],
        'part_id':    pid,
        'element_id': ei,
        'part_name':  ti.get('part_name') or pid.replace('.dat',''),
        'color_code': int(col),
        'color_hex':  chex(col),
        'color_name': ti.get('color_name') or LEGO_COLORS.get(int(col),{}).get('name',''),
        'page_num':   n['page_num'],
        'level_plates': n.get('level_plates',0),
        'is_minifig': n.get('is_minifig',False),
        'subassembly_id': n.get('subassembly_id','main_model'),
        'transform':  tf.tolist(),
        'position':   [float(tf[0,3]),float(tf[1,3]),float(tf[2,3])],
        'bbox':       [float(x) for x in meta.get('bbox',[-10,0,-10,10,8,10])],
        'connections': conns,
        'collision_passed': n.get('collision_check',{}).get('passed',True),
    })

# -----------------------------------------------------------------------
# Subgraph metadata
# -----------------------------------------------------------------------
sub_meta = {}
for n in viewer_nodes:
    s = n['subassembly_id']
    if s not in sub_meta:
        sub_meta[s] = {'id':s,'count':0,'pages':set(),'is_minifig':n['is_minifig']}
    sub_meta[s]['count']  += 1
    sub_meta[s]['pages'].add(n['page_num'])
for s in sub_meta:
    sub_meta[s]['pages'] = sorted(sub_meta[s]['pages'])
    sub_meta[s]['min_page'] = sub_meta[s]['pages'][0]
    sub_meta[s]['max_page'] = sub_meta[s]['pages'][-1]
    sub_meta[s]['icon'] = '\U0001f464' if sub_meta[s]['is_minifig'] else '\U0001f680'
    sub_meta[s]['label'] = 'Han Solo (Minifig)' if sub_meta[s]['is_minifig'] else f"Modelo Principal ({SET_NUM})"

print('Subgraphs:')
for s,m in sub_meta.items():
    print(f'  {s}: {m["count"]} nodes, pages {m["pages"]}')

# -----------------------------------------------------------------------
# Sim images + assembly crops
# -----------------------------------------------------------------------
print('=== Loading sim images ===')
nodes_by_page = {}
for n in nodes: nodes_by_page.setdefault(n['page_num'],[]).append(n)
all_pages = sorted(nodes_by_page.keys())
sim_uris, asm_uris = {}, {}
for p in all_pages:
    sp = f'{SCRATCH}/page_{p:02d}_ldraw_sim.png'
    ap = f'{SCRATCH}/page_{p:02d}_assembly.png'
    if os.path.exists(sp):
        sim_uris[p] = b64img(sp)
        print(f'  P{p}: {os.path.getsize(sp)//1024}KB')
    asm_uris[p] = b64img(ap)

# -----------------------------------------------------------------------
# Step cards HTML
# -----------------------------------------------------------------------
print('=== Building step cards ===')
cards_html = ''
for p in all_pages:
    pnodes = nodes_by_page[p]
    all_mf = all(n.get('is_minifig',False) for n in pnodes)
    sim_u  = sim_uris.get(p,'')
    asm_u  = asm_uris.get(p,'')
    step_lbl = pnodes[0].get('step_label','') if pnodes else ''

    conn_cards = ''
    for n in pnodes:
        pid  = n['part_id']
        col  = n.get('color_code',16)
        chx  = chex(col)
        ei   = n.get('element_id','')
        ti   = tmpl_by_el.get(ei,{})
        pn   = ti.get('part_name') or pid.replace('.dat','')
        cn   = ti.get('color_name') or ''
        turi = b64img(ti.get('image_path',''))
        lvl  = n.get('level_plates',0)
        conns= n.get('connections',[])
        is_mf= n.get('is_minifig',False)
        mfr  = n.get('minifig_role','')
        col_ok = n.get('collision_check',{}).get('passed',True)
        col_badge = '<span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">\U0001f6e1 0 LDU</span>' if col_ok else '<span class="px-1.5 py-0.5 rounded text-[9px] bg-rose-500/20 text-rose-300 border border-rose-500/40">\u26a0 Colision</span>'
        sub_badge = '' if n.get('subassembly_id','main_model')=='main_model' else '<span class="px-1.5 py-0.5 rounded text-[9px] bg-purple-500/20 text-purple-300 border border-purple-500/30">' + n.get('subassembly_id','') + '</span>'

        if not conns:
            lbl = ('\U0001f9b5 Minifig ' + mfr) if is_mf else ('\U0001f7e2 Raiz L=' + str(lvl))
            conn_cards += f'<div class="p-3 rounded-xl bg-slate-950/70 border border-emerald-500/30 flex items-center gap-3"><div class="w-10 h-10 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0"><img src="{turi}" class="max-w-full max-h-full object-contain"/></div><div><div class="flex items-center gap-2 flex-wrap"><span class="text-xs font-mono font-bold text-white">#{n["part_index"]}</span><span class="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">{lbl}</span>{col_badge}{sub_badge}</div><span class="text-xs text-slate-300">{ei} &middot; {cn} &middot; {pn}</span></div></div>'
        else:
            for c in conns:
                par_idx = c.get('parent_part_index')
                fam = c.get('connector_family','STUD')
                off = c.get('relative_offset_ldu',[0,0,0])
                nc  = c.get('num_contacts',1)
                fc  = {'STUD':'emerald','TECHNIC_PIN':'sky','TECHNIC_AXLE':'violet','UNDERSIDE_STUD_SOCKET':'amber'}.get(fam,'slate')
                conn_cards += f'<div class="p-3 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-{fc}-500/40 transition-all flex items-start gap-3"><div class="w-10 h-10 rounded-lg bg-slate-200/95 p-1 flex items-center justify-center shrink-0"><img src="{turi}" class="max-w-full max-h-full object-contain"/></div><div class="flex-1 min-w-0"><div class="flex items-center gap-2 flex-wrap mb-1"><span class="text-xs font-mono font-bold text-white">#{n["part_index"]}</span><span class="px-1.5 py-0.5 rounded text-[9px] bg-{fc}-500/20 text-{fc}-300 border border-{fc}-500/30 font-mono">{fam}</span>{col_badge}{sub_badge}</div><span class="text-xs text-slate-300 block">{ei} &middot; {cn} &middot; {pn}</span><span class="text-[10px] text-slate-500">Parent #{par_idx} &middot; offset [{off[0]},{off[1]},{off[2]}] LDU &middot; {nc} contactos</span></div></div>'

    sim_blk = ''
    if sim_u and not all_mf:
        sim_blk = f'<div class="glass-panel rounded-xl p-3 border border-indigo-500/30"><div class="text-[10px] text-indigo-400 font-bold uppercase mb-2">\U0001f916 Digital Twin LDraw P.{p}</div><img src="{sim_u}" class="w-full max-h-52 object-contain rounded"/></div>'

    mf_badge = '<span class="px-2 py-0.5 rounded-full text-[10px] bg-purple-600/30 text-purple-300 border border-purple-500/40">\U0001f464 Minifig</span>' if all_mf else ''
    cols = 'md:grid-cols-2' if asm_u and sim_blk else ''
    cards_html += f'''
<div id="page-{p}" class="glass-panel rounded-2xl p-5 border border-slate-800/80 scroll-mt-24">
  <div class="flex items-center justify-between mb-4">
    <div class="flex items-center gap-3">
      <span class="text-xl font-extrabold text-white">{step_lbl}</span>
      <span class="px-2 py-0.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-mono">Pág.{p}</span>
      {mf_badge}
      <button onclick="jumpTo3D({p})" class="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 text-xs font-semibold">\U0001f3ae Ver 3D</button>
    </div>
    <span class="text-xs text-slate-500">{len(pnodes)} piezas</span>
  </div>
  <div class="grid grid-cols-1 {cols} gap-4 mb-4">
    {'<div class="glass-panel rounded-xl p-3 border border-slate-800"><div class="text-[10px] text-slate-400 font-bold uppercase mb-2">\U0001f4d6 Manual P.' + str(p) + '</div><img src="' + asm_u + '" class="w-full max-h-52 object-contain rounded"/></div>' if asm_u else ''}
    {sim_blk}
  </div>
  <div class="space-y-2">{conn_cards}</div>
</div>'''

# -----------------------------------------------------------------------
# Serialise JS data
# -----------------------------------------------------------------------
nodes_js  = json.dumps(viewer_nodes, ensure_ascii=False)
geos_js   = json.dumps(viewer_geos,  ensure_ascii=False)
studs_js  = json.dumps(viewer_studs, ensure_ascii=False)
sub_meta_for_js = [
    {'id':m['id'],'label':m['label'],'icon':m['icon'],'count':m['count'],'pages':m['pages'],'min_page':m['min_page'],'max_page':m['max_page'],'is_minifig':m['is_minifig']}
    for m in sub_meta.values()
]
submeta_js = json.dumps(sub_meta_for_js, ensure_ascii=False)

col_ok_count = sum(1 for n in nodes if n.get('collision_check',{}).get('passed',True))
min_page_main = min((n['page_num'] for n in nodes if n.get('subassembly_id','main_model')=='main_model'), default=5)
min_page_mf   = min((n['page_num'] for n in nodes if n.get('is_minifig',False)), default=4)
max_page_main = max((n['page_num'] for n in nodes if n.get('subassembly_id','main_model')=='main_model'), default=36)
max_page_mf   = max((n['page_num'] for n in nodes if n.get('is_minifig',False)), default=4)

print('=== Writing HTML ===')


# -----------------------------------------------------------------------
# HTML build (string concatenation throughout to avoid quoting conflicts)
# -----------------------------------------------------------------------

HEAD = (
    '<!DOCTYPE html>
<html lang="es">
<head>
'
    '<meta charset="UTF-8">'
    '<title>LegoGPT — ' + set_row[1] + ' — 3D</title>'
    '<script src="https://cdn.tailwindcss.com"></script>'
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>'
    '<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>'
    '<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">'
    '<style>'
    'body{font-family:'Outfit',sans-serif;background:#070b14;color:#f8fafc;}'
    '.glass{background:rgba(15,23,42,.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.07);}'
    '.grad{background:linear-gradient(135deg,#fff 0%,#93c5fd 55%,#34d399 100%);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;}'
    '::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}'
    'kbd{background:#1e293b;border:1px solid #334155;border-radius:3px;padding:1px 4px;font-size:10px;}'
    '</style>'
    '</head>'
    '<body class="min-h-screen">'
)
