
#!/usr/bin/env python3
"""generate_html.py -- Full 3D HTML report for 75295-1."""
import os, sys, json, base64, sqlite3
sys.path.insert(0, '/Users/I764690/Code_personal/LegoGPT')
os.chdir('/Users/I764690/Code_personal/LegoGPT')
import numpy as np
from src.ldraw_geometry import LDrawGeometryResolver

SET_NUM  = '75295-1'
GRAPH    = 'outputs/75295-1_construction_graph.json'
SCRATCH  = 'scratch/step_crops_75295-1'
DB_PATH  = 'data/catalog/models_catalog.db'
OUT_HTML = 'pdf_extraction_report_75295.html'

def b64img(path):
    if not path or not os.path.exists(path): return ''
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()

print('=== Loading ===')
geom  = LDrawGeometryResolver()
with open(GRAPH) as f: graph = json.load(f)
nodes = graph['nodes']
print(f'  {len(nodes)} nodes')

db  = sqlite3.connect(DB_PATH)
cur = db.cursor()
cur.execute(
    "SELECT s.set_num,s.name,s.year,t.name FROM rb_sets s "
    "LEFT JOIN rb_themes t ON s.theme_id=t.id "
    "JOIN rb_inventories i ON s.set_num=i.set_num WHERE s.set_num=? LIMIT 1",
    (SET_NUM,))
set_row = cur.fetchone() or (SET_NUM,'Millennium Falcon Microfighter',2021,'Star Wars')

LEGO_COLORS = {}
cur.execute('SELECT id,name,rgb FROM rb_colors')
for cid,cname,crgb in cur.fetchall():
    LEGO_COLORS[int(cid)] = {'name':cname or '','hex':f'#{crgb}' if crgb else '#94a3b8'}

tmpl_by_el = {}
tmpl_dir = f'scratch/inventory_templates_{SET_NUM}'
if os.path.isdir(tmpl_dir):
    for fn in os.listdir(tmpl_dir):
        if fn.endswith('.png') and not fn.startswith('annotated'):
            el = fn.replace('.png','')
            tmpl_by_el[el] = {'element_id':el,'image_path':os.path.join(tmpl_dir,fn),
                              'part_name':el,'color_name':'','color_rgb':'#94a3b8'}
el_ids = list(tmpl_by_el.keys())
if el_ids:
    ph = ','.join('?'*len(el_ids))
    cur.execute(
        f"SELECT e.element_id,p.name,c.name,c.rgb FROM rb_elements e "
        f"JOIN rb_parts p ON e.part_num=p.part_num "
        f"JOIN rb_colors c ON e.color_id=c.id WHERE e.element_id IN ({ph})",
        el_ids)
    for el,pn,cn,cr in cur.fetchall():
        if el in tmpl_by_el:
            tmpl_by_el[el].update({'part_name':pn or el,'color_name':cn or '',
                                   'color_rgb':f'#{cr}' if cr else '#94a3b8'})
db.close()

LDRAW_CLR = {0:'#1b2a4a',1:'#1e40af',2:'#166534',3:'#dc2626',4:'#b91c1c',
             5:'#be185d',6:'#92400e',7:'#94a3b8',8:'#475569',9:'#93c5fd',
             10:'#4ade80',11:'#22d3ee',14:'#eab308',15:'#f8fafc',16:'#94a3b8',
             19:'#fef3c7',26:'#a855f7',70:'#d97706',71:'#94a3b8',72:'#64748b',
             78:'#fde68a',85:'#6b7280',86:'#9ca3af',272:'#374151'}
def chex(code):
    c = LEGO_COLORS.get(int(code),{})
    h = c.get('hex','')
    return h if h and h != '#None' else LDRAW_CLR.get(int(code),'#94a3b8')

print('=== Extracting geometries ===')
viewer_geos, viewer_studs, viewer_edges = {}, {}, {}
for pid in list(set(n['part_id'] for n in nodes)):
    try:
        faces = geom.get_part_faces(pid)
        verts = []
        all_pts = []
        up_cnt = down_cnt = 0
        for face in faces[:20000]:
            pts = face['pts']
            all_pts.extend(pts)
            ny = face['normal'][1]
            if ny > 0.5: up_cnt += 1
            elif ny < -0.5: down_cnt += 1
            # pre-flip winding: Y-row negation = reflection, det<0, reverses front-face
            if len(pts) == 3:
                A,B,C = pts[0],pts[1],pts[2]
                for pt in (A,C,B): verts += [float(pt[0]),float(pt[1]),float(pt[2])]
            elif len(pts) == 4:
                A,B,C,D = pts[0],pts[1],pts[2],pts[3]
                for pt in (A,C,B,A,D,C): verts += [float(pt[0]),float(pt[1]),float(pt[2])]
        # Synthesise bottom cap when no (or very few) downward faces exist
        # Uses contour of upward faces projected to y_bottom (closes hollow LEGO undersides)
        if faces and all_pts and down_cnt < max(1, up_cnt // 4):
            ys = [float(p[1]) for p in all_pts]
            y_bot = max(ys)   # LDraw: larger Y = lower visually before inversion
            # Collect XZ boundary of up-facing faces for bottom cap polygon
            cap_xz = []
            for face in faces:
                if face['normal'][1] > 0.3:
                    for p in face['pts']: cap_xz.append((float(p[0]), float(p[2])))
            if len(cap_xz) >= 3:
                import numpy as _np
                cap_arr = _np.array(cap_xz)
                # Convex hull of XZ boundary points
                from scipy.spatial import ConvexHull
                try:
                    hull = ConvexHull(cap_arr)
                    hull_pts = cap_arr[hull.vertices]  # ordered CCW
                    cx = float(hull_pts[:,0].mean())
                    cz = float(hull_pts[:,1].mean())
                    # Fan-triangulate hull from centroid
                    # Winding: normal should point in +Y (LDraw down) BEFORE Y-inversion
                    # After Y-inversion +Y becomes visual bottom, so we emit A,B,C order
                    # (opposite of the pre-flipped top faces) so cap faces DOWN visually
                    n_h = len(hull_pts)
                    for i in range(n_h):
                        ax,az = hull_pts[i]
                        bx,bz = hull_pts[(i+1)%n_h]
                        # tri: centroid -> A -> B (pre-flip already in A,B,C order)
                        verts += [cx, y_bot, cz, float(ax), y_bot, float(az), float(bx), y_bot, float(bz)]
                except Exception: pass
        if len(verts) >= 9: viewer_geos[pid] = verts
        meta = geom.get_part_meta(pid)
        sl = []
        for s in (meta.get('stud_details') or meta.get('studs',[])):
            pos  = s['pos']            if isinstance(s,dict) else s
            norm = s.get('normal',[0,-1,0]) if isinstance(s,dict) else [0,-1,0]
            ny   = float(norm[1]) if hasattr(norm,'__len__') else -1.0
            if ny <= 0.5: sl.append([float(pos[0]),float(pos[1]),float(pos[2])])
        if sl: viewer_studs[pid] = sl
        try:
            segs = geom.get_part_edges_type2(pid)
            if segs:
                flat = []
                for p1, p2 in segs:
                    flat += [float(p1[0]),float(p1[1]),float(p1[2]),
                             float(p2[0]),float(p2[1]),float(p2[2])]
                if flat: viewer_edges[pid] = flat
        except Exception: pass
    except Exception: pass
print(f'  {len(viewer_geos)} geos, {len(viewer_studs)} stud maps')

viewer_nodes = []
for n in nodes:
    tf  = np.array(n['absolute_transform'],dtype=np.float32).reshape(4,4)
    pid = n['part_id']
    col = n.get('color_code',16)
    meta= geom.get_part_meta(pid)
    ei  = n.get('element_id','')
    ti  = tmpl_by_el.get(ei,{})
    conns=[{'parent_idx':c.get('parent_part_index'),'family':c.get('connector_family','STUD'),
            'offset':c.get('relative_offset_ldu',[0,0,0]),'contacts':c.get('num_contacts',1)}
           for c in n.get('connections',[])]
    viewer_nodes.append({
        'part_index':n['part_index'],'part_id':pid,'element_id':ei,
        'part_name':ti.get('part_name') or pid.replace('.dat',''),
        'color_code':int(col),'color_hex':chex(col),
        'color_name':ti.get('color_name') or LEGO_COLORS.get(int(col),{}).get('name',''),
        'page_num':n['page_num'],'level_plates':n.get('level_plates',0),
        'is_minifig':n.get('is_minifig',False),'subassembly_id':n.get('subassembly_id','main_model'),
        'transform':tf.tolist(),'position':[float(tf[0,3]),float(tf[1,3]),float(tf[2,3])],
        'bbox':[float(x) for x in meta.get('bbox',[-10,0,-10,10,8,10])],
        'connections':conns,
        'collision_passed':n.get('collision_check',{}).get('passed',True),
    })

sub_meta = {}
for n in viewer_nodes:
    s = n['subassembly_id']
    if s not in sub_meta:
        sub_meta[s] = {'id':s,'count':0,'pages':set(),'is_minifig':n['is_minifig']}
    sub_meta[s]['count'] += 1
    sub_meta[s]['pages'].add(n['page_num'])
for s in sub_meta:
    sub_meta[s]['pages']    = sorted(sub_meta[s]['pages'])
    sub_meta[s]['min_page'] = sub_meta[s]['pages'][0]
    sub_meta[s]['max_page'] = sub_meta[s]['pages'][-1]
    sub_meta[s]['icon']     = '👤' if sub_meta[s]['is_minifig'] else '🚀'
    sub_meta[s]['label']    = 'Han Solo (Minifig)' if sub_meta[s]['is_minifig'] else f'Modelo ({SET_NUM})'
print('Subgraphs:', {s:m['count'] for s,m in sub_meta.items()})

print('=== Loading sim images ===')
nodes_by_page = {}
for n in nodes: nodes_by_page.setdefault(n['page_num'],[]).append(n)
all_pages = sorted(nodes_by_page.keys())
sim_uris,asm_uris = {},{}
for p in all_pages:
    sp = f'{SCRATCH}/page_{p:02d}_ldraw_sim.png'
    ap = f'{SCRATCH}/page_{p:02d}_assembly.png'
    if os.path.exists(sp):
        sim_uris[p] = b64img(sp)
        print(f'  P{p}: {os.path.getsize(sp)//1024}KB')
    asm_uris[p] = b64img(ap)

print('=== Building step cards ===')
cards_html = ''
for p in all_pages:
    pnodes   = nodes_by_page[p]
    all_mf   = all(n.get('is_minifig',False) for n in pnodes)
    sim_u    = sim_uris.get(p,'')
    asm_u    = asm_uris.get(p,'')
    step_lbl = pnodes[0].get('step_label','') if pnodes else ''
    conn_cards = ''
    for n in pnodes:
        pid   = n['part_id']
        ei    = n.get('element_id','')
        ti    = tmpl_by_el.get(ei,{})
        pn    = ti.get('part_name') or pid.replace('.dat','')
        cn    = ti.get('color_name') or ''
        turi  = b64img(ti.get('image_path',''))
        lvl   = n.get('level_plates',0)
        conns = n.get('connections',[])
        is_mf = n.get('is_minifig',False)
        mfr   = n.get('minifig_role','')
        col_ok= n.get('collision_check',{}).get('passed',True)
        col_b = ('<span class="ok-badge">🛡 0 LDU</span>' if col_ok
                 else '<span class="err-badge">⚠ Colision</span>')
        img_d = (f'<div class="w-10 h-10 rounded-lg bg-slate-200/95 p-1 flex items-center '
                 f'justify-center shrink-0"><img src="{turi}" class="max-w-full max-h-full object-contain"/></div>')
        if not conns:
            lbl = ('🦵 Minifig ' + mfr) if is_mf else ('🟢 Raiz L=' + str(lvl))
            conn_cards += (
                f'<div class="conn-card border-emerald-500/30">{img_d}'
                f'<div><div class="flex items-center gap-2 flex-wrap">'
                f'<span class="text-xs font-mono font-bold text-white">#{n["part_index"]}</span>'
                f'<span class="ok-badge">{lbl}</span>{col_b}</div>'
                f'<span class="text-xs text-slate-300">{ei} &middot; {cn} &middot; {pn}</span></div></div>')
        else:
            for c in conns:
                par  = c.get('parent_idx')
                fam  = c.get('family','STUD')
                off  = c.get('offset',[0,0,0])
                nc   = c.get('contacts',1)
                fc   = {'STUD':'emerald','TECHNIC_PIN':'sky','TECHNIC_AXLE':'violet',
                        'UNDERSIDE_STUD_SOCKET':'amber'}.get(fam,'slate')
                conn_cards += (
                    f'<div class="conn-card border-slate-800 hover:border-{fc}-500/40">{img_d}'
                    f'<div class="flex-1 min-w-0"><div class="flex items-center gap-2 flex-wrap mb-1">'
                    f'<span class="text-xs font-mono font-bold text-white">#{n["part_index"]}</span>'
                    f'<span class="px-1.5 py-0.5 rounded text-[9px] bg-{fc}-500/20 text-{fc}-300 '
                    f'border border-{fc}-500/30 font-mono">{fam}</span>{col_b}</div>'
                    f'<span class="text-xs text-slate-300 block">{ei} &middot; {cn} &middot; {pn}</span>'
                    f'<span class="text-[10px] text-slate-500">Parent #{par} &middot; '
                    f'[{off[0]},{off[1]},{off[2]}] LDU &middot; {nc} contactos</span></div></div>')
    sim_blk = ''
    if sim_u and not all_mf:
        sim_blk = (f'<div class="glass rounded-xl p-3 border border-indigo-500/30">'
                   f'<div class="text-[10px] text-indigo-400 font-bold uppercase mb-2">'
                   f'🤖 Digital Twin P.{p}</div>'
                   f'<img src="{sim_u}" class="w-full max-h-52 object-contain rounded"/></div>')
    mf_b = ('<span class="px-2 py-0.5 rounded-full text-[10px] bg-purple-600/30 text-purple-300 '
            'border border-purple-500/40">👤 Minifig</span>' if all_mf else '')
    asm_blk = ''
    if asm_u:
        asm_blk = (f'<div class="glass rounded-xl p-3 border border-slate-800">'
                   f'<div class="text-[10px] text-slate-400 font-bold uppercase mb-2">'
                   f'📖 Manual P.{p}</div>'
                   f'<img src="{asm_u}" class="w-full max-h-52 object-contain rounded"/></div>')
    cols = 'md:grid-cols-2' if asm_u and sim_blk else ''
    cards_html += (
        f'<div id="page-{p}" class="glass rounded-2xl p-5 border border-slate-800/80 scroll-mt-24">'
        f'<div class="flex items-center justify-between mb-4">'
        f'<div class="flex items-center gap-3">'
        f'<span class="text-xl font-extrabold text-white">{step_lbl}</span>'
        f'<span class="px-2 py-0.5 rounded-lg bg-slate-800 text-slate-300 text-xs font-mono">Pág.{p}</span>'
        f'{mf_b}'
        f'<button onclick="jumpTo3D({p})" class="px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 '
        f'text-indigo-300 border border-indigo-500/40 text-xs font-semibold">🎮 Ver 3D</button>'
        f'</div><span class="text-xs text-slate-500">{len(pnodes)} piezas</span></div>'
        f'<div class="grid grid-cols-1 {cols} gap-4 mb-4">{asm_blk}{sim_blk}</div>'
        f'<div class="space-y-2">{conn_cards}</div></div>'
    )


print('=== Serialising JS data ===')
nodes_js    = json.dumps(viewer_nodes,  ensure_ascii=False)
geos_js     = json.dumps(viewer_geos,   ensure_ascii=False)
studs_js    = json.dumps(viewer_studs,  ensure_ascii=False)
edges_js    = json.dumps(viewer_edges,  ensure_ascii=False)
sub_meta_js = json.dumps([
    {'id':m['id'],'label':m['label'],'icon':m['icon'],'count':m['count'],
     'pages':m['pages'],'min_page':m['min_page'],'max_page':m['max_page'],'is_minifig':m['is_minifig']}
    for m in sub_meta.values()], ensure_ascii=False)

col_ok_count = sum(1 for n in nodes if n.get('collision_check',{}).get('passed',True))
pages_main   = sorted(set(n['page_num'] for n in nodes if n.get('subassembly_id','main_model')=='main_model'))
pages_mf     = sorted(set(n['page_num'] for n in nodes if n.get('is_minifig',False)))

DATA_TAGS = (
    '<script id="asm-data"  type="application/json">\n' + nodes_js    + '\n</script>\n'
    '<script id="geo-data"  type="application/json">\n' + geos_js     + '\n</script>\n'
    '<script id="stud-data" type="application/json">\n' + studs_js    + '\n</script>\n'
    '<script id="edge-data" type="application/json">\n' + edges_js    + '\n</script>\n'
    '<script id="sub-meta"  type="application/json">\n' + sub_meta_js + '\n</script>\n'
)


JS_CODE = '\n<script>\n(function(){\n  var PD=[],GD={},SD={},ED={},SM=[];\n  try{PD=JSON.parse(document.getElementById(\'asm-data\').textContent);}catch(e){}\n  try{GD=JSON.parse(document.getElementById(\'geo-data\').textContent);}catch(e){}\n  try{SD=JSON.parse(document.getElementById(\'stud-data\').textContent);}catch(e){}\n  try{ED=JSON.parse(document.getElementById(\'edge-data\').textContent);}catch(e){}\n  try{SM=JSON.parse(document.getElementById(\'sub-meta\').textContent);}catch(e){}\n\n  var allPages=[...new Set(PD.map(function(p){return p.page_num;}))].sort(function(a,b){return a-b;});\n  var pagesMain=allPages.filter(function(p){return PD.some(function(n){return n.page_num===p&&n.subassembly_id===\'main_model\';});});\n  var pagesMF  =allPages.filter(function(p){return PD.some(function(n){return n.page_num===p&&n.is_minifig;});});\n\n  var corrections=[], selPM=null, corrMode=false, origPos=null, origRot=null, origBodyMatrix=null, origCentroid=null;\n  var activeTab=\'all\', stepState={all:allPages[0]||0, main:pagesMain[0]||0, mf:pagesMF[0]||0};\n\n  var container=document.getElementById(\'viewer\');\n  var W=container.clientWidth||900, H=container.clientHeight||580;\n  var scene=new THREE.Scene();\n  scene.background=new THREE.Color(0x06080e);\n  scene.fog=new THREE.FogExp2(0x06080e,0.0015);\n  var camera=new THREE.PerspectiveCamera(38,W/H,1,4000);\n  camera.position.set(240,180,290);\n  var rend=new THREE.WebGLRenderer({antialias:true,powerPreference:\'high-performance\'});\n  rend.setSize(W,H); rend.setPixelRatio(Math.min(devicePixelRatio,2));\n  rend.outputEncoding=THREE.sRGBEncoding;\n  rend.toneMapping=THREE.LinearToneMapping; rend.toneMappingExposure=1.0;\n  rend.shadowMap.enabled=true; rend.shadowMap.type=THREE.PCFSoftShadowMap;\n  container.appendChild(rend.domElement);\n  var ctrl=new THREE.OrbitControls(camera,rend.domElement);\n  ctrl.enableDamping=true; ctrl.dampingFactor=0.06;\n  ctrl.maxDistance=2500; ctrl.minDistance=15;\n  var tc=new THREE.TransformControls(camera,rend.domElement);\n  tc.setSize(0.75);\n  tc.addEventListener(\'dragging-changed\',function(e){ctrl.enabled=!e.value;});\n  tc.addEventListener(\'objectChange\',function(){if(selPM&&!origPos){origPos={x:selPM.group.position.x,y:selPM.group.position.y,z:selPM.group.position.z};origRot={x:selPM.group.rotation.x,y:selPM.group.rotation.y,z:selPM.group.rotation.z};}updateRotHUD();});\n  scene.add(tc);\n  scene.add(new THREE.HemisphereLight(0xd0e4ff,0x1a120a,0.30));\n  var dl=new THREE.DirectionalLight(0xffffff,0.75); dl.position.set(200,380,200); dl.castShadow=true; dl.shadow.mapSize.width=2048; dl.shadow.mapSize.height=2048; scene.add(dl);\n  var dl2=new THREE.DirectionalLight(0x9bb8e8,0.30); dl2.position.set(-220,160,-200); scene.add(dl2);\n  var dl3=new THREE.DirectionalLight(0xffe8d0,0.20); dl3.position.set(0,80,350); scene.add(dl3);\n  var pmrem=new THREE.PMREMGenerator(rend); pmrem.compileEquirectangularShader();\n  var envSc=new THREE.Scene(); envSc.background=new THREE.Color(0x8ab0d8);\n  scene.environment=pmrem.fromScene(envSc).texture; pmrem.dispose();\n  var grid=new THREE.GridHelper(800,40,0x1e293b,0x0f172a); grid.position.y=0.5; scene.add(grid);\n  var ray=new THREE.Raycaster(), mouse=new THREE.Vector2();\n\n  var matCache={};\n  function getMat(hex,sel){\n    var k=hex+(sel?\'_s\':\'\');\n    if(!matCache[k]) matCache[k]=new THREE.MeshStandardMaterial({\n      color:new THREE.Color(hex||\'#94a3b8\'), roughness:0.40, metalness:0.01,\n      envMapIntensity:0.20, side:THREE.DoubleSide,\n      polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1,\n      depthWrite:true, transparent:!!sel, opacity:sel?0.88:1.0,\n      emissive:sel?new THREE.Color(0xf59e0b):new THREE.Color(0x000000),\n      emissiveIntensity:sel?0.45:0.0\n    });\n    return matCache[k];\n  }\n  var geoCache={};\n  function buildGeo(part){\n    var pid=part.part_id;\n    if(geoCache[pid]) return geoCache[pid];\n    var verts=GD[pid];\n    if(verts&&verts.length>=9){\n      var g=new THREE.BufferGeometry();\n      g.setAttribute(\'position\',new THREE.Float32BufferAttribute(new Float32Array(verts),3));\n      g.computeVertexNormals(); geoCache[pid]=g; return g;\n    }\n    var b=part.bbox||[-10,0,-10,10,8,10];\n    var g=new THREE.BoxGeometry(Math.max(1,b[3]-b[0]),Math.max(1,b[4]-b[1]),Math.max(1,b[5]-b[2]));\n    g.translate((b[0]+b[3])/2,(b[1]+b[4])/2,(b[2]+b[5])/2);\n    geoCache[pid]=g; return g;\n  }\n  var studGeo=new THREE.CylinderGeometry(5.7,5.7,3.8,20);\n\n  var PM=[];\n  PD.forEach(function(part){\n    var group=new THREE.Group();\n    var body=new THREE.Mesh(buildGeo(part),getMat(part.color_hex));\n    body.castShadow=body.receiveShadow=true;\n    var m=part.transform;\n    if(m&&m.length===4){\n      body.matrix.set(m[0][0],m[0][1],m[0][2],m[0][3],-m[1][0],-m[1][1],-m[1][2],-m[1][3],\n                      m[2][0],m[2][1],m[2][2],m[2][3],m[3][0],m[3][1],m[3][2],m[3][3]);\n      body.matrixAutoUpdate=false;\n    }\n    group.add(body);\n    var studPts=SD[part.part_id]||[];\n    var studs=[];\n    studPts.forEach(function(sp){\n      var wPos=new THREE.Vector3(sp[0],sp[1],sp[2]);\n      if(m&&m.length===4){\n        wPos.applyMatrix4(new THREE.Matrix4().set(\n          m[0][0],m[0][1],m[0][2],m[0][3],-m[1][0],-m[1][1],-m[1][2],-m[1][3],\n          m[2][0],m[2][1],m[2][2],m[2][3],m[3][0],m[3][1],m[3][2],m[3][3]));\n      }\n      var sm=new THREE.Mesh(studGeo,getMat(part.color_hex));\n      sm.position.copy(wPos); sm.position.y+=1.9;\n      sm.castShadow=true; group.add(sm); studs.push(sm);\n    });\n    \n    // Black LEGO outline edges (LDraw type-2 boundary lines)\n    var edgePts=ED[part.part_id]||[];\n    if(edgePts.length>=6){\n      var eg=new THREE.BufferGeometry();\n      eg.setAttribute(\'position\',new THREE.Float32BufferAttribute(new Float32Array(edgePts),3));\n      var em=new THREE.LineBasicMaterial({color:0x0d0d0d,linewidth:1});\n      var el=new THREE.LineSegments(eg,em);\n      el.matrixAutoUpdate=false;\n      if(m&&m.length===4){\n        el.matrix.set(m[0][0],m[0][1],m[0][2],m[0][3],-m[1][0],-m[1][1],-m[1][2],-m[1][3],\n                      m[2][0],m[2][1],m[2][2],m[2][3],m[3][0],m[3][1],m[3][2],m[3][3]);\n      }\n      group.add(el);\n    }\n    // ---- Pivot group at part centroid (so rotations spin in place) ----\n    var _bb=new THREE.Box3().setFromObject(body);\n    var _cen=new THREE.Vector3();\n    _bb.getCenter(_cen);\n    // Shift group to centroid; offset body so it stays in same world pos\n    group.position.copy(_cen);\n    body.position.sub(_cen);\n    // Studs and edges are already children – offset them too\n    studs.forEach(function(s){s.position.sub(_cen);});\n    group.children.forEach(function(c){\n      if(c!==body&&studs.indexOf(c)<0){c.position.sub(_cen);}\n    });\n    group.userData=part; group.userData._studs=studs; group.userData._centroid=_cen.clone();\n    group.visible=false; scene.add(group);\n    PM.push({group:group,mesh:body,part:part,origCol:part.color_hex});\n  });\n\n  var tabBar=document.getElementById(\'sub-tabs\');\n  SM.forEach(function(sm){\n    var btn=document.createElement(\'button\');\n    btn.id=\'tab-\'+sm.id;\n    btn.className=\'tab-btn px-4 py-2 rounded-xl text-sm font-bold border border-slate-700 bg-slate-800/50 text-slate-300\';\n    btn.innerHTML=sm.icon+\' \'+sm.label+\' (\'+sm.count+\')\';\n    btn.onclick=function(){selectTab(sm.id);};\n    tabBar.appendChild(btn);\n  });\n\n  function showEl(id,show){\n    var el=document.getElementById(id);\n    if(el) el.style.display=show?\'\':\'none\';\n  }\n\n  function selectTab(id){\n    activeTab=id;\n    document.querySelectorAll(\'.tab-btn\').forEach(function(b){\n      b.className=b.id===\'tab-\'+id\n        ?\'tab-btn px-4 py-2 rounded-xl text-sm font-bold border border-indigo-500/60 bg-indigo-600/30 text-indigo-200\'\n        :\'tab-btn px-4 py-2 rounded-xl text-sm font-bold border border-slate-700 bg-slate-800/50 text-slate-300\';\n    });\n    var isDual=(id===\'all\');\n    showEl(\'ctrl-single\',!isDual); showEl(\'ctrl-dual\',isDual);\n    if(isDual){ applyDual(); refreshDualSliders(); }\n    else { applySingle(id, stepState[id]||pagesForSub(id)[0]||0); }\n    fitAll();\n  }\n\n  function pagesForSub(id){\n    return [...new Set(PM.filter(function(pm){return id===\'all\'||pm.part.subassembly_id===id||pm.part.is_minifig&&id.indexOf(\'minifig\')>=0;}).map(function(pm){return pm.part.page_num;}))].sort(function(a,b){return a-b;});\n  }\n\n  function applyDual(){\n    var limMain=stepState.main, limMF=stepState.mf, cnt=0;\n    PM.forEach(function(pm){\n      var isMF=pm.part.is_minifig;\n      var show=isMF?(pm.part.page_num<=limMF):(pm.part.page_num<=limMain);\n      pm.group.visible=show;\n      if(show){\n        var isNew=isMF?(pm.part.page_num===limMF):(pm.part.page_num===limMain);\n        pm.mesh.material=isNew?newMat(pm.origCol):getMat(pm.origCol);\n        cnt++;\n      }\n    });\n    document.getElementById(\'hud-step\').textContent=\'M:P.\'+limMain+\' F:P.\'+limMF;\n    document.getElementById(\'hud-parts\').textContent=cnt+\' piezas\';\n  }\n\n  function applySingle(subId, page){\n    var cnt=0;\n    PM.forEach(function(pm){\n      var belongs=(subId===\'all\')||pm.part.subassembly_id===subId||(pm.part.is_minifig&&subId.indexOf(\'minifig\')>=0);\n      var show=belongs&&pm.part.page_num<=page;\n      pm.group.visible=show;\n      if(show){\n        pm.mesh.material=pm.part.page_num===page?newMat(pm.origCol):getMat(pm.origCol);\n        cnt++;\n      }\n    });\n    stepState[subId]=page;\n    document.getElementById(\'hud-step\').textContent=\'P.\'+page;\n    document.getElementById(\'hud-parts\').textContent=cnt+\' piezas\';\n    var pg=pagesForSub(subId);\n    var sl=document.getElementById(\'slider-single\');\n    if(sl){sl.max=Math.max(0,pg.length-1);sl.value=pg.indexOf(page);}\n  }\n\n  function newMat(hex){\n    return new THREE.MeshStandardMaterial({color:new THREE.Color(hex),roughness:0.55,emissive:new THREE.Color(0x4ade80),emissiveIntensity:0.55});\n  }\n\n  function refreshDualSliders(){\n    var sm=document.getElementById(\'slider-main\'),sf=document.getElementById(\'slider-mf\');\n    if(sm){sm.min=0;sm.max=Math.max(0,pagesMain.length-1);sm.value=pagesMain.indexOf(stepState.main);}\n    if(sf){sf.min=0;sf.max=Math.max(0,pagesMF.length-1);sf.value=pagesMF.indexOf(stepState.mf);}\n    var lm=document.getElementById(\'lbl-main\'),lf=document.getElementById(\'lbl-mf\');\n    if(lm) lm.textContent=\'P.\'+stepState.main;\n    if(lf) lf.textContent=\'P.\'+stepState.mf;\n  }\n\n  var playTimer=null;\n  window.togglePlay=function(){\n    var btn=document.getElementById(\'btn-play\');\n    if(playTimer){clearInterval(playTimer);playTimer=null;btn.innerHTML=\'&#x25B6;\';return;}\n    btn.innerHTML=\'&#x23F8;\';\n    playTimer=setInterval(function(){stepV(1);},900);\n  };\n\n  window.stepV=function(d){\n    if(activeTab===\'all\') return;\n    var pg=pagesForSub(activeTab);\n    var idx=pg.indexOf(stepState[activeTab]||pg[0]);\n    var next=pg[Math.max(0,Math.min(pg.length-1,idx+d))];\n    applySingle(activeTab,next);\n    if((idx+d)>=pg.length-1&&d>0){clearInterval(playTimer);playTimer=null;var btn=document.getElementById(\'btn-play\');if(btn)btn.innerHTML=\'&#x25B6;\';}\n  };\n\n  window.stepDual=function(key,d){\n    var pages=(key===\'main\'?pagesMain:pagesMF);\n    var cur=stepState[key===\'main\'?\'main\':\'mf\'];\n    var idx=pages.indexOf(cur);\n    var next=pages[Math.max(0,Math.min(pages.length-1,idx+d))];\n    stepState[key===\'main\'?\'main\':\'mf\']=next;\n    applyDual(); refreshDualSliders();\n  };\n\n  window.onSlider=function(key,v){\n    var vi=parseInt(v);\n    if(key===\'single\'){\n      var pg=pagesForSub(activeTab);\n      applySingle(activeTab,pg[vi]||pg[0]);\n    } else if(key===\'main\'){\n      stepState.main=pagesMain[vi]||pagesMain[0];\n      applyDual(); refreshDualSliders();\n    } else {\n      stepState.mf=pagesMF[vi]||pagesMF[0];\n      applyDual(); refreshDualSliders();\n    }\n  };\n\n  window.jumpTo3D=function(page){\n    document.getElementById(\'sec-3d\').scrollIntoView({behavior:\'smooth\'});\n    setTimeout(function(){\n      var isMF=PD.some(function(n){return n.page_num===page&&n.is_minifig;});\n      var subId=isMF?\'minifig_1\':\'main_model\';\n      if(activeTab===\'all\'){\n        if(isMF){stepState.mf=page;}else{stepState.main=page;}\n        applyDual(); refreshDualSliders();\n      } else { selectTab(subId); applySingle(subId,page); }\n    },400);\n  };\n\n  window.setCam=function(v){\n    ctrl.target.set(0,0,0);\n    if(v===\'iso\'){camera.position.set(240,180,290);}\n    else if(v===\'top\'){camera.position.set(0,600,1);}\n    else if(v===\'front\'){camera.position.set(0,80,500);}\n    else if(v===\'side\'){camera.position.set(500,80,0);}\n    ctrl.update();\n  };\n\n  window.fitAll=function(){\n    var vis=PM.filter(function(p){return p.group.visible;});\n    if(!vis.length) return;\n    var box=new THREE.Box3();\n    vis.forEach(function(p){box.expandByObject(p.group);});\n    var ctr=new THREE.Vector3(),sz=new THREE.Vector3();\n    box.getCenter(ctr); box.getSize(sz);\n    ctrl.target.copy(ctr);\n    camera.position.set(ctr.x+sz.length()*0.75,ctr.y+sz.length()*0.5,ctr.z+sz.length()*0.75);\n    ctrl.update();\n  };\n\n  rend.domElement.addEventListener(\'click\',function(e){\n    if(corrMode||tc.dragging) return;\n    var rc=container.getBoundingClientRect();\n    mouse.x=((e.clientX-rc.left)/rc.width)*2-1;\n    mouse.y=-((e.clientY-rc.top)/rc.height)*2+1;\n    ray.setFromCamera(mouse,camera);\n    var meshes=PM.filter(function(pm){return pm.group.visible;}).map(function(pm){return pm.mesh;});\n    var hits=ray.intersectObjects(meshes,false);\n    if(hits.length){selectPM(PM.find(function(pm){return pm.mesh===hits[0].object;}));}\n    else deselect();\n  });\n\n  function selectPM(pm){\n    if(selPM) deselect();\n    selPM=pm;\n    pm.mesh.material=getMat(pm.origCol,true);\n    var p=pm.part;\n    document.getElementById(\'ov-title\').textContent=\'#\'+p.part_index+\' - \'+p.part_id+\' (\'+p.part_name+\')\';\n    document.getElementById(\'ov-sub\').textContent=p.element_id+\' - \'+p.color_name+\' - Pag.\'+p.page_num+\' L=\'+p.level_plates;\n    document.getElementById(\'ov-swatch\').style.backgroundColor=p.color_hex;\n    document.getElementById(\'ov-conn\').textContent=(p.connections||[]).length+\' conexiones\';\n    document.getElementById(\'part-overlay\').style.opacity=\'1\';\n    showEl(\'ov-rot-btn\',true);\n    window.enterCorr(\'translate\');\n    showEl(\'hud-sel\',true); document.getElementById(\'hud-sel-name\').textContent=\'#\'+p.part_index+\' \'+p.part_id;\n    showEl(\'hud-rot\',true); updateRotHUD();\n  }\n  function deselect(){\n    if(!selPM) return;\n    selPM.mesh.material=getMat(selPM.origCol);\n    selPM=null;\n    document.getElementById(\'part-overlay\').style.opacity=\'0\';\n    showEl(\'ov-corr-btn\',false); showEl(\'ov-rot-btn\',false);\n    showEl(\'ov-ok-btn\',false); showEl(\'ov-cancel-btn\',false);\n    showEl(\'hud-sel\',false); showEl(\'hud-rot\',false);\n    showEl(\'corr-hint\',false);\n    corrMode=false; origPos=null; origRot=null; origBodyMatrix=null; origCentroid=null; ctrl.enabled=true;\n    tc.detach();\n  }\n  function updateRotHUD(){\n    if(!selPM) return;\n    var r=selPM.group.rotation;\n    var d=function(v){return Math.round(THREE.MathUtils.radToDeg(v));};\n    document.getElementById(\'hud-rot-val\').textContent=d(r.x)+\'/\'+d(r.y)+\'/\'+d(r.z);\n  }\n\n  var SNAP=20;   // XZ: 1 stud pitch = 20 LDU = 8mm\n  var SNAP_Y=8;  // Y: 1 plate = 8 LDU = 3.2mm\n  window.enterCorr=function(mode){\n    if(!selPM) return;\n    mode=mode||\'translate\';\n    corrMode=true;\n    if(!origPos){origPos={x:selPM.group.position.x,y:selPM.group.position.y,z:selPM.group.position.z};origRot={x:selPM.group.rotation.x,y:selPM.group.rotation.y,z:selPM.group.rotation.z};origBodyMatrix=selPM.mesh.matrix.clone();origCentroid=selPM.group.position.clone();}\n    tc.attach(selPM.group);\n    tc.setMode(mode);\n    ctrl.enabled=false;\n    showEl(\'ov-corr-btn\',false); showEl(\'ov-rot-btn\',false);\n    showEl(\'ov-ok-btn\',true); showEl(\'ov-cancel-btn\',true);\n    showEl(\'corr-hint\',true);\n  };\n\n  document.addEventListener(\'keydown\',function(e){\n    if(!selPM) return;\n    if(!corrMode){ if(e.key===\'c\'||e.key===\'C\') window.enterCorr(); return; }\n    var step=e.shiftKey?Math.PI/12:Math.PI/2;\n    var t=e.shiftKey?5:SNAP;    // XZ snap (1 stud)\n    var ty=e.shiftKey?1:SNAP_Y; // Y snap (1 plate = 8 LDU = 3.2mm)\n    var g=selPM.group; var moved=true;\n    switch(e.key){\n      case \'w\':case \'W\':g.position.z-=t;break;\n      case \'s\':case \'S\':g.position.z+=t;break;\n      case \'a\':case \'A\':g.position.x-=t;break;\n      case \'d\':case \'D\':g.position.x+=t;break;\n      case \'q\':case \'Q\':g.position.y+=ty;break;\n      case \'e\':case \'E\':g.position.y-=ty;break;\n      case \'t\':case \'T\':g.rotation.x+=step;break;\n      case \'g\':case \'G\':g.rotation.x-=step;break;\n      case \'r\':case \'R\':g.rotation.y+=step;break;\n      case \'f\':case \'F\':g.rotation.y-=step;break;\n      case \'y\':case \'Y\':g.rotation.z+=step;break;\n      case \'h\':case \'H\':g.rotation.z-=step;break;\n      case \'Enter\':if(tc&&tc.object===selPM.group){e.preventDefault();confirmCorr();}moved=false;break;\n      case \'Escape\': window.cancelCorr(); moved=false; break;\n    }\n    if(moved){e.preventDefault();updateRotHUD();}\n  });\n\n  function mat3FromEuler(rx,ry,rz){\n    var cx=Math.cos(rx),sx=Math.sin(rx),cy=Math.cos(ry),sy=Math.sin(ry),cz=Math.cos(rz),sz=Math.sin(rz);\n    return[[cy*cz,cy*sz,-sy],[sx*sy*cz-cx*sz,sx*sy*sz+cx*cz,sx*cy],[cx*sy*cz+sx*sz,cx*sy*sz-sx*cz,cx*cy]];\n  }\n\n  window.cancelCorr=function(){if(!selPM) return;if(origCentroid){selPM.group.position.copy(origCentroid);}else if(origPos){selPM.group.position.set(origPos.x,origPos.y,origPos.z);}if(origPos){selPM.group.rotation.set(origRot.x,origRot.y,origRot.z);}if(origBodyMatrix){selPM.mesh.matrix.copy(origBodyMatrix);selPM.mesh.matrixAutoUpdate=false;origBodyMatrix=null;}corrMode=false;origPos=null;origRot=null;tc.detach();ctrl.enabled=true;showEl(\'corr-hint\',false);showEl(\'ov-ok-btn\',false);showEl(\'ov-cancel-btn\',false);showEl(\'ov-corr-btn\',true);showEl(\'ov-rot-btn\',true);};\n  function confirmCorr(){\n    if(!selPM||!origPos) return;\n    var p=selPM.part,g=selPM.group;\n    var bm=origBodyMatrix||selPM.mesh.matrix;\n    var ox=bm.elements[12],oy=bm.elements[13],oz=bm.elements[14];\n    // Snap delta to nearest LDU grid (XZ: 20 LDU = 1 stud, Y: 8 LDU = 1 plate)\n    var _oc=origCentroid||new THREE.Vector3(ox,0,oz);\n    var _dx=g.position.x-_oc.x, _dy=g.position.y-_oc.y, _dz=g.position.z-_oc.z;\n    var _sdx=Math.round(_dx/20)*20, _sdy=Math.round(_dy/8)*8, _sdz=Math.round(_dz/20)*20;\n    // Apply snap back to group (visual correction before export)\n    g.position.set(_oc.x+_sdx, _oc.y+_sdy, _oc.z+_sdz);\n    // Dedup: update existing correction for this part_index instead of duplicating\n    var existIdx=corrections.findIndex(function(c){return c.part_index===p.part_index;});\n    var rec={\n      part_index:p.part_index,part_id:p.part_id,element_id:p.element_id,\n      subassembly_id:p.subassembly_id,page_num:p.page_num,\n      old_pos:[ox,oy,oz],\n      new_pos:[ox+g.position.x,oy+g.position.y,oz+g.position.z],\n      delta_pos:[g.position.x,g.position.y,g.position.z],\n      rot_euler_deg:{x:Math.round(THREE.MathUtils.radToDeg(g.rotation.x)),\n                     y:Math.round(THREE.MathUtils.radToDeg(g.rotation.y)),\n                     z:Math.round(THREE.MathUtils.radToDeg(g.rotation.z))},\n      rot_matrix:mat3FromEuler(g.rotation.x,g.rotation.y,g.rotation.z),\n      timestamp:new Date().toISOString(),\n      inferred_connections:p.connections||[]\n    };\n    if(existIdx>=0){corrections[existIdx]=rec;}else{corrections.push(rec);}\n    updateCorrUI();\n    corrMode=false;origPos=null;origRot=null;origBodyMatrix=null;origCentroid=null;ctrl.enabled=true;showEl(\'corr-hint\',false);\n    tc.detach();showEl(\'ov-ok-btn\',false);showEl(\'ov-cancel-btn\',false);showEl(\'ov-corr-btn\',true);showEl(\'ov-rot-btn\',true);\n    selPM.mesh.material.emissive.setHex(0x10b981);selPM.mesh.material.emissiveIntensity=0.8;document.getElementById(\'hud-sel-name\').textContent=\'✓ #\'+p.part_index+\' \'+p.part_id;\n    setTimeout(function(){if(selPM){selPM.mesh.material.emissive.setHex(0xf59e0b);selPM.mesh.material.emissiveIntensity=0.4;}},700);\n  }\n\n  function updateCorrUI(){\n    document.getElementById(\'corr-count\').textContent=corrections.length+\' correcciones\';\n    var list=document.getElementById(\'corr-list\');\n    if(!corrections.length){list.innerHTML=\'<em>Sin correcciones.</em>\';return;}\n    list.innerHTML=corrections.map(function(c,i){\n      return \'<div class="p-3 rounded-xl bg-slate-950/70 border border-amber-500/20 font-mono">\'\n        +\'<div class="flex items-center justify-between">\'\n        +\'<span class="text-amber-300 font-bold">#\'+(i+1)+\' Pieza #\'+c.part_index+\' (\'+c.part_id+\')</span>\'\n        +\'<button onclick="removeCorr(\'+i+\')" class="text-rose-400 hover:text-rose-300 text-xs">&times;</button>\'\n        +\'</div>\'\n        +\'<div class="text-slate-400 text-[10px] mt-1">\'\n        +\'Antes:[\'+c.old_pos.map(function(v){return v.toFixed(1);}).join(\',\')+\'] &rarr; \'\n        +\'Despues:[\'+c.new_pos.map(function(v){return v.toFixed(1);}).join(\',\')+\']<br>\'\n        +\'RotXYZ: \'+c.rot_euler_deg.x+\'/\'+c.rot_euler_deg.y+\'/\'+c.rot_euler_deg.z+\' &middot; Pag.\'+c.page_num\n        +\'</div></div>\';\n    }).join(\'\');\n  }\n\n  window.removeCorr=function(i){corrections.splice(i,1);updateCorrUI();};\n  window.exportCorr=function(){\n    if(!corrections.length){alert(\'Sin correcciones.\');return;}\n    var blob=new Blob([JSON.stringify({set_num:\'75295-1\',corrections:corrections},null,2)],{type:\'application/json\'});\n    var a=document.createElement(\'a\');a.href=URL.createObjectURL(blob);\n    a.download=\'corrections_75295-1_\'+Date.now()+\'.json\';a.click();\n  };\n  window.clearCorr=function(){corrections.length=0;updateCorrUI();};\n\n  window.addEventListener(\'resize\',function(){\n    var w=container.clientWidth,h=container.clientHeight;\n    camera.aspect=w/h;camera.updateProjectionMatrix();rend.setSize(w,h);\n  });\n\n  function animate(){requestAnimationFrame(animate);ctrl.update();rend.render(scene,camera);}\n  animate();\n\n  selectTab(\'all\');\n  applyDual(); refreshDualSliders();\n  setTimeout(fitAll,300);\n})();\n</script>\n'
print('=== Assembling HTML ===')

# CSS utility classes used in step cards
EXTRA_CSS = (
    '.conn-card{display:flex;align-items:start;gap:12px;padding:12px;border-radius:12px;'
    'background:rgba(2,6,23,.7);border:1px solid transparent;transition:border-color .2s;}'
    '.ok-badge{display:inline-flex;padding:1px 6px;border-radius:4px;font-size:9px;font-family:monospace;'
    'background:rgba(16,185,129,.2);color:#6ee7b7;border:1px solid rgba(16,185,129,.3);}'
    '.err-badge{display:inline-flex;padding:1px 6px;border-radius:4px;font-size:9px;font-family:monospace;'
    'background:rgba(244,63,94,.2);color:#fda4af;border:1px solid rgba(244,63,94,.4);}'
)

# Full HTML structure
HTML_PARTS = []

# Head
HTML_PARTS.append('<!DOCTYPE html>\n<html lang="es">\n<head>\n')

HTML_PARTS.append('<meta charset="UTF-8">')
HTML_PARTS.append('<meta name="viewport" content="width=device-width,initial-scale=1.0">')
HTML_PARTS.append('<title>LegoGPT — ' + set_row[1] + ' — 3D</title>')
HTML_PARTS.append('<script src="https://cdn.tailwindcss.com"></script>')
HTML_PARTS.append('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>')
HTML_PARTS.append('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>')
HTML_PARTS.append('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/TransformControls.js"></script>')
HTML_PARTS.append('<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">')
HTML_PARTS.append(
    '<style>'
    'body{font-family:"Outfit",sans-serif;background:#070b14;color:#f8fafc;}'
    '.glass{background:rgba(15,23,42,.72);backdrop-filter:blur(16px);border:1px solid rgba(255,255,255,.07);}'
    '.grad{background:linear-gradient(135deg,#fff 0%,#93c5fd 55%,#34d399 100%);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;}'
    '::-webkit-scrollbar{width:5px}::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}'
    'kbd{background:#1e293b;border:1px solid #334155;border-radius:3px;padding:1px 4px;font-size:10px;}'
    + EXTRA_CSS +
    '</style></head>'
    '<body class="min-h-screen">\n'
)


# Header
HTML_PARTS.append('<header class="max-w-7xl mx-auto px-4 pt-8 pb-4"><div class="glass rounded-3xl p-8">')
HTML_PARTS.append('<h1 class="text-4xl font-extrabold grad mb-2">' + set_row[1] + '</h1>')
HTML_PARTS.append(
    '<p class="text-sm text-slate-400">' + str(len(nodes)) + ' piezas &middot; '
    + str(len(all_pages)) + ' pasos &middot; '
    + str(len(viewer_geos)) + ' geometrias LDraw + studs</p>'
)
HTML_PARTS.append(
    '<nav class="flex gap-3 mt-3 text-xs">'
    '<a href="#sec-3d" class="px-3 py-1.5 rounded-lg bg-indigo-600/20 text-indigo-300 border border-indigo-500/30 font-semibold">&#x1F3AE; Viewer 3D</a>'
    '<a href="#sec-steps" class="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 font-semibold">&#x1F527; Pasos</a>'
    '<a href="#sec-corrections" class="px-3 py-1.5 rounded-lg bg-amber-600/20 text-amber-300 border border-amber-500/30 font-semibold">&#x270F; Correcciones</a>'
    '</nav></div></header>'
)

# 3D viewer section
HTML_PARTS.append('<main class="max-w-7xl mx-auto px-4 py-6 space-y-10">')
HTML_PARTS.append('<section id="sec-3d" class="glass rounded-3xl p-6 border border-indigo-500/40 scroll-mt-20">')
HTML_PARTS.append(
    '<div class="mb-4">'
    '<div class="text-indigo-400 font-bold text-xs uppercase tracking-wider mb-1">'
    '&#x1F3AE; Digital Twin 3D &middot; Three.js r128 &middot; MeshStandardMaterial (plastic)</div>'
    '<h2 class="text-2xl font-extrabold text-white">Viewer Interactivo Paso a Paso</h2>'
    '<p class="text-xs text-slate-400 mt-1 leading-5">'
    '<b>Orbita</b>: arrastrar &middot; <b>Zoom</b>: rueda &middot; <b>Clic</b>: seleccionar &middot; '
    '<kbd>C</kbd>=correcc. &middot; '
    '<kbd>WASD</kbd>=XZ <kbd>QE</kbd>=Y '
    '<kbd>TG</kbd>=pitchX <kbd>RF</kbd>=yawY <kbd>YH</kbd>=rollZ '
    '<kbd>Shift</kbd>=15&deg; <kbd>Enter</kbd>=OK <kbd>Esc</kbd>=cancel'
    '</p></div>'
)

# HUD
HTML_PARTS.append(
    '<div class="flex items-center gap-3 flex-wrap mb-4">'
    '<div class="px-3 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center">'
    '<span class="text-[9px] text-slate-400 uppercase font-bold block">Paso</span>'
    '<span id="hud-step" class="text-base font-extrabold text-amber-400 font-mono">-</span></div>'
    '<div class="px-3 py-2 rounded-xl bg-slate-950/80 border border-slate-800 text-center">'
    '<span class="text-[9px] text-slate-400 uppercase font-bold block">Piezas</span>'
    '<span id="hud-parts" class="text-base font-extrabold text-emerald-400 font-mono">0</span></div>'
    '<div id="hud-sel" class="px-3 py-2 rounded-xl bg-amber-950/80 border border-amber-500/30 text-center" style="display:none">'
    '<span class="text-[9px] text-amber-400 uppercase font-bold block">Seleccionada</span>'
    '<span id="hud-sel-name" class="text-xs font-bold text-amber-300 font-mono"></span></div>'
    '<div id="hud-rot" class="px-3 py-2 rounded-xl bg-slate-950/80 border border-slate-700 text-center" style="display:none">'
    '<span class="text-[9px] text-slate-400 uppercase font-bold block">Rot X/Y/Z (deg)</span>'
    '<span id="hud-rot-val" class="text-xs font-bold text-sky-300 font-mono">0/0/0</span></div>'
    '</div>'
)

# Tabs
HTML_PARTS.append('<div id="sub-tabs" class="flex items-center gap-2 mb-4 flex-wrap"><button onclick="selectTab("all")" id="tab-all" class="tab-btn px-4 py-2 rounded-xl text-sm font-bold border border-indigo-500/60 bg-indigo-600/30 text-indigo-200">&#x1F535; Todos</button></div>')

# Single timeline controls
HTML_PARTS.append(
    '<div id="ctrl-single" class="flex flex-wrap items-center gap-3 mb-3 bg-slate-950/80 p-3 rounded-2xl border border-slate-800">'
    '<button id="btn-play" onclick="togglePlay()" class="w-9 h-9 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white flex items-center justify-center font-bold">&#x25B6;</button>'
    '<button onclick="stepV(-1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold">&#x25C0;</button>'
    '<input id="slider-single" type="range" min="0" max="1" value="0" oninput="onSlider("single",this.value)" class="flex-1 accent-indigo-500 h-2 bg-slate-800 rounded-lg cursor-pointer"/>'
    '<button onclick="stepV(1)" class="w-8 h-8 rounded-lg bg-slate-800 hover:bg-slate-700 text-white text-xs font-bold">&#x25B6;</button>'
    '<button onclick="setCam("iso")"   class="px-2 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Axon.</button>'
    '<button onclick="setCam("top")"   class="px-2 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Top</button>'
    '<button onclick="setCam("front")" class="px-2 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Frontal</button>'
    '<button onclick="setCam("side")"  class="px-2 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300">Lateral</button>'
    '<button onclick="fitAll()"          class="px-2 py-1 text-xs rounded-lg bg-indigo-800/50 hover:bg-indigo-700/50 text-indigo-300">Centrar</button>'
    '</div>'
)

# Dual timelines
HTML_PARTS.append(
    '<div id="ctrl-dual" style="display:none">'
    '<div class="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-3">'
    '<div class="bg-slate-950/80 p-3 rounded-2xl border border-slate-800">'
    '<div class="text-[10px] text-slate-400 uppercase font-bold mb-2">&#x1F680; Modelo Principal</div>'
    '<div class="flex items-center gap-2">'
    '<button onclick="stepDual("main",-1)" class="w-7 h-7 rounded-lg bg-slate-800 text-white text-xs font-bold">&#x25C0;</button>'
    '<input id="slider-main" type="range" min="0" max="1" value="0" oninput="onSlider("main",this.value)" class="flex-1 accent-sky-500 h-2 bg-slate-800 rounded-lg cursor-pointer"/>'
    '<button onclick="stepDual("main",1)"  class="w-7 h-7 rounded-lg bg-slate-800 text-white text-xs font-bold">&#x25B6;</button>'
    '<span id="lbl-main" class="text-xs text-sky-400 font-mono w-12 text-right"></span>'
    '</div></div>'
    '<div class="bg-slate-950/80 p-3 rounded-2xl border border-slate-800">'
    '<div class="text-[10px] text-slate-400 uppercase font-bold mb-2">&#x1F464; Han Solo (Minifig)</div>'
    '<div class="flex items-center gap-2">'
    '<button onclick="stepDual("mf",-1)" class="w-7 h-7 rounded-lg bg-slate-800 text-white text-xs font-bold">&#x25C0;</button>'
    '<input id="slider-mf" type="range" min="0" max="1" value="0" oninput="onSlider("mf",this.value)" class="flex-1 accent-purple-500 h-2 bg-slate-800 rounded-lg cursor-pointer"/>'
    '<button onclick="stepDual("mf",1)"  class="w-7 h-7 rounded-lg bg-slate-800 text-white text-xs font-bold">&#x25B6;</button>'
    '<span id="lbl-mf" class="text-xs text-purple-400 font-mono w-12 text-right"></span>'
    '</div></div>'
    '</div></div>'
)

# Canvas
HTML_PARTS.append(
    '<div class="relative w-full rounded-2xl overflow-hidden border border-slate-800" style="height:580px;background:#06080e;">'
    '<div id="viewer" class="w-full h-full"></div>'
    '<div class="absolute top-3 left-3 pointer-events-none bg-slate-950/80 px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] text-slate-400">'
    '&#x1F5B1; Arrastrar=orbita &middot; Rueda=zoom &middot; Clic=seleccionar &middot; <kbd>C</kbd>=correcc.</div>'
    '<div id="corr-hint" class="absolute top-3 right-3 bg-amber-950/90 px-3 py-2 rounded-xl border border-amber-500/40 text-xs text-amber-300" style="display:none;pointer-events:none;">'
    '&#x270F; <b>Modo Correccion</b><br>'
    'Arrastra <b>flechas</b>=mover &middot; <b>aros</b>=rotar<br>'
    'WASD=XZ &middot; QE=Y &middot; TG/RF/YH=rot &middot; Shift=fino<br>'
    'Enter=OK &middot; Esc=Cancelar</div>'
    '<div id="part-overlay" class="absolute" '
    'style="bottom:12px;left:12px;right:12px;opacity:0;transition:opacity .3s;pointer-events:none;'
    'background:rgba(2,6,23,.9);padding:12px;border-radius:12px;border:1px solid #1e293b;'
    'display:flex;align-items:center;justify-content:space-between;">'
    '<div style="display:flex;align-items:center;gap:12px;">'
    '<div id="ov-swatch" style="width:16px;height:16px;border-radius:50%;border:1px solid #475569;background:#94a3b8;flex-shrink:0"></div>'
    '<div><span id="ov-title" class="text-sm font-bold text-white" style="display:block"></span>'
    '<span id="ov-sub" class="text-xs text-slate-400"></span></div></div>'
    '<div style="display:flex;align-items:center;gap:8px;">'
    '<span id="ov-conn" class="text-xs text-slate-500"></span>'
    '<button id="ov-corr-btn" onclick="enterCorr(\'translate\')" style="display:none;" '
    'class="px-3 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/30 text-xs font-semibold">&#x2195; Mover</button>'
    '<button id="ov-rot-btn" onclick="enterCorr(\'rotate\')" style="display:none;" '
    'class="px-3 py-1 rounded-lg bg-violet-600/30 hover:bg-violet-600/50 text-violet-300 border border-violet-500/30 text-xs font-semibold">&#x21BA; Rotar</button>'
    '<button id="ov-ok-btn" onclick="confirmCorr()" style="display:none;" '
    'class="px-3 py-1 rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 border border-emerald-500/30 text-xs font-semibold">&#x2713; OK</button>'
    '<button id="ov-cancel-btn" onclick="cancelCorr()" style="display:none;" '
    'class="px-3 py-1 rounded-lg bg-rose-600/30 hover:bg-rose-600/50 text-rose-300 border border-rose-500/30 text-xs font-semibold">&#x2715; Cancelar</button>'
    '</div></div>'
    '</div>'  # end canvas
)
HTML_PARTS.append('</section>')

# Corrections section
HTML_PARTS.append(
    '<section id="sec-corrections" class="glass rounded-3xl p-6 border border-amber-500/30 scroll-mt-20">'
    '<div class="flex items-center justify-between mb-4">'
    '<div><h2 class="text-xl font-extrabold text-white">&#x270F;&#xFE0F; Correcciones Manuales</h2>'
    '<p class="text-xs text-slate-400 mt-1">Cada correccion exporta old_pos, new_pos, rot_matrix 3x3 para el re-ranker GNN.</p></div>'
    '<div class="flex items-center gap-2">'
    '<span id="corr-count" class="px-3 py-1 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 text-xs font-bold">0 correcciones</span>'
    '<button onclick="exportCorr()" class="px-4 py-2 rounded-xl bg-amber-600/30 hover:bg-amber-600/50 text-amber-300 border border-amber-500/30 text-sm font-semibold">&#x2B07; Exportar JSON</button>'
    '<button onclick="clearCorr()" class="px-4 py-2 rounded-xl bg-rose-600/20 hover:bg-rose-600/30 text-rose-300 border border-rose-500/30 text-sm font-semibold">&#x1F5D1; Limpiar</button>'
    '</div></div>'
    '<div id="corr-list" class="space-y-2 text-xs text-slate-400"><em>Sin correcciones. Selecciona una pieza y presiona C.</em></div>'
    '</section>'
)

# Steps section
HTML_PARTS.append(
    '<section id="sec-steps" class="space-y-6 scroll-mt-20">'
    '<div><h2 class="text-2xl font-bold text-white">&#x1F527; Grafo de Montaje Paso a Paso</h2>'
    '<p class="text-sm text-slate-400 mt-1">Manual PDF (izq.) + render Digital Twin LDraw acumulado (der.) + tarjetas de conexion fisica.</p></div>\n'
)

HTML_PARTS.append(cards_html)
HTML_PARTS.append('</section>')
HTML_PARTS.append('</main>')

# Data + JS
HTML_PARTS.append(DATA_TAGS)
HTML_PARTS.append(JS_CODE)
HTML_PARTS.append('</body></html>\n')


print('=== Writing HTML ===')
html_str = ''.join(HTML_PARTS)
with open(OUT_HTML, 'w', encoding='utf-8') as f:
    f.write(html_str)
size_mb = os.path.getsize(OUT_HTML)/1024/1024
print(f'Output: {os.path.abspath(OUT_HTML)}')
print(f'Size:   {size_mb:.1f} MB')
print(f'Nodes:  {len(viewer_nodes)}')
print(f'Geos:   {len(viewer_geos)}')
print(f'Studs:  {len(viewer_studs)}')
print(f'Sim imgs: {len([u for u in sim_uris.values() if u])}/{len(all_pages)}')
print('=== DONE ===')
