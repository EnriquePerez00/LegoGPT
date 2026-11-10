"""
LegoGPT: Set Semantic Manifest & Visual Enrichment Engine
Generates rich, comprehensive semantic descriptions for all sets in models_catalog.db:
- Construction archetype and subclass (Vehículo, Edificio, Nave Espacial, etc.)
- Exact minifigures breakdown (count, names, roles)
- Quantitative color palette with RGB hex and percentages
- Functional piece composition (% structural core, % aesthetic skin, % kinematics, etc.)
- 3D shape, proportions and silhouette characterization
- Rich narrative visual description and LLM assembly prompt

Optimized for Apple Silicon M4 (dynamic worker allocation, SQLite batch transactions, memory-safe).
"""

import os
import sys
import json
import sqlite3
import time
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor
import psutil

DB_PATH = "data/catalog/models_catalog.db"

def init_tables(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS set_semantic_manifest (
        set_id TEXT PRIMARY KEY,
        set_name TEXT,
        theme TEXT,
        year INTEGER,
        archetype_category TEXT,
        archetype_subclass TEXT,
        minifigures_count INTEGER,
        minifigures_detail TEXT,
        color_palette TEXT,
        parts_composition TEXT,
        shape_and_silhouette TEXT,
        llm_visual_description TEXT,
        llm_construction_prompt TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_set_manifest_archetype ON set_semantic_manifest(archetype_category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_set_manifest_theme ON set_semantic_manifest(theme);")
    conn.commit()


def infer_archetype(name: str, theme: str, role_counts: Dict[str, int], parts_count: int) -> Tuple[str, str, str, str, str]:
    """
    Infers:
    - archetype_category: 'Vehículo', 'Edificio/Arquitectura', 'Nave Espacial', 'Criatura/Mec', 'Diorama/Escena', 'Objeto/Decorativo'
    - archetype_subclass: e.g. 'Coche/Automóvil', 'Caza Estelar', 'Edificio Modular', etc.
    - level_1_entorno: 'Terrestre', 'Aéreo', 'Acuático', 'Espacial', 'Multientorno'
    - level_2_proposito: 'Civil/Pasajeros', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia', etc.
    - level_4_escala: 'Microscale', 'Minifig-scale', 'UCS/Gran Escala'
    """
    name_lower = (name or "").lower()
    theme_lower = (theme or "").lower()
    
    wheels = role_counts.get("wheels_count", 0)
    wings_cockpit = role_counts.get("AERODYNAMICS_COCKPIT", 0)
    architecture = role_counts.get("HABITATION_ARCHITECTURE", 0)
    kinematics = role_counts.get("KINEMATIC_MECHANISM", 0)
    minifigs_anatomy = role_counts.get("MINIFIG_ANATOMY", 0)
    nature = role_counts.get("ORGANIC_NATURE", 0)
    
    # Scale determination
    if "microfighter" in name_lower or "microscale" in name_lower or (parts_count < 150 and any(k in name_lower for k in ["fighter", "cruiser", "destroyer", "falcon"])):
        scale = "Microscale"
    elif parts_count > 2500 or "ucs" in name_lower or "ultimate collector" in name_lower:
        scale = "UCS/Gran Escala"
    else:
        scale = "Minifig-scale"

    # Space & Sci-Fi
    if any(k in theme_lower for k in ["star wars", "space", "galaxy", "blacktron", "m-tron"]) or any(k in name_lower for k in ["starfighter", "cruiser", "speeder", "tie fighter", "x-wing", "falcon", "shuttle", "spaceship", "interceptor"]):
        category = "Nave Espacial"
        entorno = "Espacial"
        proposito = "Militar/Combate" if any(k in name_lower for k in ["fighter", "bomber", "destroyer", "cannon", "blaster", "attack"]) else "Ficcion/Fantasia"
        if "speeder" in name_lower:
            subclass = "Speeder / Deslizador Ligero"
            entorno = "Terrestre"
        elif any(k in name_lower for k in ["walker", "at-at", "at-st", "mech"]):
            category = "Criatura/Mec"
            subclass = "Caminante Bípedo / Cuadrúpedo de Combate"
            entorno = "Terrestre"
        else:
            subclass = "Caza Estelar o Crucero Espacial"
            
    # Land Vehicles
    elif wheels >= 2 or any(k in name_lower for k in ["car", "truck", "van", "motorcycle", "bike", "jeep", "kart", "train", "rover", "buggy", "racer", "tractor", "bus"]):
        category = "Vehículo"
        entorno = "Terrestre"
        if any(k in name_lower for k in ["racer", "f1", "gt", "speed", "ferrari", "porsche", "lamborghini"]):
            subclass = "Coche de Competición / Superdeportivo"
            proposito = "Competicion/Deportes"
        elif any(k in name_lower for k in ["police", "fire", "ambulance", "rescue"]):
            subclass = "Vehículo de Emergencias / Servicios"
            proposito = "Emergencias/Servicios"
        elif any(k in name_lower for k in ["truck", "dumper", "crane", "loader", "excavator", "tractor"]):
            subclass = "Camión / Maquinaria Pesada de Construcción"
            proposito = "Construccion/Industrial"
        elif any(k in name_lower for k in ["train", "locomotive"]):
            subclass = "Tren / Transporte Ferroviario"
            proposito = "Civil/Pasajeros"
        elif any(k in name_lower for k in ["motorcycle", "bike", "chopper"]):
            subclass = "Motocicleta / Vehículo Ligero de 2 Ruedas"
            proposito = "Civil/Pasajeros"
        else:
            subclass = "Automóvil / Transporte Terrestre Civil"
            proposito = "Civil/Pasajeros"

    # Air Vehicles
    elif any(k in name_lower for k in ["plane", "airplane", "jet", "helicopter", "chopper", "biplane", "glider"]) or (wings_cockpit > 3 and wheels < 4):
        category = "Vehículo"
        subclass = "Aeronave (Avión o Helicóptero)"
        entorno = "Aéreo"
        proposito = "Militar/Combate" if any(k in name_lower for k in ["fighter", "war", "bomber", "strike"]) else "Civil/Pasajeros"

    # Sea Vehicles
    elif any(k in name_lower for k in ["boat", "ship", "yacht", "submarine", "raft", "ferry", "vessel", "cutter"]):
        category = "Vehículo"
        subclass = "Embarcación Marítima o Submarino"
        entorno = "Acuático"
        proposito = "Civil/Pasajeros"

    # Architecture & Buildings
    elif architecture >= 3 or any(k in theme_lower for k in ["architecture", "modular buildings", "castle", "pirates", "harry potter"]) or any(k in name_lower for k in ["house", "castle", "station", "building", "tower", "shop", "temple", "palace", "mansion", "hotel", "villa"]):
        category = "Edificio/Arquitectura"
        subclass = "Estructura Arquitectónica / Inmueble Modular"
        entorno = "Terrestre"
        proposito = "Civil/Pasajeros"

    # Creatures, Mechs & Bionicle
    elif any(k in theme_lower for k in ["bionicle", "hero factory", "ninjago", "monkie kid"]) or any(k in name_lower for k in ["dragon", "dinosaur", "creature", "mech", "robot", "titan"]):
        category = "Criatura/Mec"
        subclass = "Robot Mecha Articulado o Bestia Legendaria"
        entorno = "Multientorno"
        proposito = "Ficcion/Fantasia"

    # Botanical & Art & Decorative
    elif any(k in theme_lower for k in ["botanical", "art", "brickheadz", "ideas"]) or any(k in name_lower for k in ["flower", "bouquet", "plant", "bonsai", "succulent", "portrait", "mosaic", "helmet"]):
        category = "Objeto/Decorativo"
        subclass = "Escultura de Exhibición, Botánica o Mosaico"
        entorno = "Terrestre"
        proposito = "Civil/Pasajeros"
        scale = "UCS/Gran Escala"

    else:
        category = "Diorama/Escena"
        subclass = "Escenario de Juego o Set Temático Polivalente"
        entorno = "Terrestre"
        proposito = "Civil/Pasajeros"

    return category, subclass, entorno, proposito, scale


def generate_set_manifest_record(data_pack: Dict[str, Any]) -> Tuple:
    """
    Worker function to compute rich descriptions for a set.
    """
    set_id = data_pack["set_id"]
    name = data_pack["name"]
    theme = data_pack["theme"]
    year = data_pack["year"]
    parts_count = data_pack["parts_count"]
    role_counts = data_pack["role_counts"]
    color_breakdown = data_pack["color_breakdown"]
    minifigs_list = data_pack["minifigs_list"]
    
    total_parts_in_inv = sum(color_breakdown.values()) or parts_count or 1
    
    # 1. Colors JSON
    sorted_colors = sorted(color_breakdown.items(), key=lambda x: x[1], reverse=True)
    color_palette = []
    top_color_names = []
    for c_name, qty in sorted_colors[:6]:
        pct = round((qty / total_parts_in_inv) * 100, 1)
        color_palette.append({"color": c_name, "quantity": qty, "percentage": pct})
        top_color_names.append(f"{c_name} ({pct}%)")
        
    color_palette_json = json.dumps(color_palette, ensure_ascii=False)
    
    # 2. Composition JSON
    parts_composition = {}
    for role, cnt in role_counts.items():
        if role != "wheels_count":
            pct = round((cnt / total_parts_in_inv) * 100, 1)
            parts_composition[role] = {"count": cnt, "percentage": pct}
    parts_composition_json = json.dumps(parts_composition, ensure_ascii=False)
    
    # 3. Minifigures
    minifig_count = len(minifigs_list)
    minifig_names = [m["name"] for m in minifigs_list]
    minifigs_detail_json = json.dumps(minifigs_list, ensure_ascii=False)
    
    # 4. Archetype & Taxonomy
    category, subclass, entorno, proposito, scale = infer_archetype(name, theme, role_counts, total_parts_in_inv)
    
    # 5. Shape & Silhouette description
    structural_pct = parts_composition.get("STRUCTURAL_CORE", {}).get("percentage", 0)
    aesthetic_pct = parts_composition.get("SKIN_AESTHETIC", {}).get("percentage", 0)
    kinematic_pct = parts_composition.get("KINEMATIC_MECHANISM", {}).get("percentage", 0)
    
    if category == "Vehículo":
        if "Competición" in subclass or "Automóvil" in subclass:
            silhouette = "Perfil aerodinámico bajo y alargado sobre 4 ruedas, morro inclinado mediante slopes y alerón o cabina posterior."
        elif "Aeronave" in subclass:
            silhouette = "Fuselaje esbelto con alas simétricas laterales, cabina frontal transparente y conjunto de timón de cola vertical."
        else:
            silhouette = "Estructura cúbica robusta con cabina de conducción elevada y chasis reforzado sobre ejes de ruedas."
    elif category == "Nave Espacial":
        silhouette = "Geometría angular estilizada en cuña o delta con cabina acristalada central, detalles de motores traseros y alas laterales."
    elif category == "Edificio/Arquitectura":
        silhouette = "Construcción vertical ortogonal con fachada detallada, ventanas simétricas, puerta principal y cubierta superior o tejado."
    elif category == "Criatura/Mec":
        silhouette = "Silueta figurativa articulada con extremidades móviles, torso central reforzado y cabeza o cabina de piloto."
    elif category == "Objeto/Decorativo":
        silhouette = "Volumen estático orgánico o escultórico diseñado con alta densidad de piezas curvas de acabado para exhibición."
    else:
        silhouette = "Distribución compacta en diorama horizontal con elementos escénicos y accesorios interactivos."
        
    # 6. Rich LLM Visual Description
    colors_str = ", ".join(top_color_names) if top_color_names else "colores variados de construcción"
    minifig_str = f"Incluye {minifig_count} minifigura(s): {', '.join(minifig_names)}." if minifig_count > 0 else "No contiene minifiguras (modelo centrado exclusivamente en la construcción)."
    
    llm_visual_description = (
        f"El set '{name}' (ID: {set_id}) pertenece a la línea '{theme}' ({year}) y representa un(a) {subclass}. "
        f"Al ser construido, genera una estructura de tipo {category} con escala '{scale}'. "
        f"Su silueta visual destaca por: {silhouette} "
        f"La paleta cromática dominante está compuesta por {colors_str}. "
        f"A nivel constructivo, la maqueta está formada por {total_parts_in_inv} piezas con un {structural_pct}% de ladrillos y placas estructurales de núcleo, "
        f"un {aesthetic_pct}% de piezas de acabado exterior (slopes aerodinámicas, cuñas y azulejos lisos) "
        f"y un {kinematic_pct}% de mecanismos o articulaciones funcionales. {minifig_str}"
    )
    
    # 7. LLM Assembly Prompt (Prompt condensado para que un modelo generativo 3D o LLM entienda el ensamble)
    llm_assembly_prompt = (
        f"Modelo 3D '{name}' ({theme}, {year}). Clasificación: {category} -> {subclass}. "
        f"Morfología: {silhouette} Colores dominantes: {colors_str}. "
        f"Composición: {structural_pct}% núcleo estructural, {aesthetic_pct}% carenado exterior, {kinematic_pct}% mecánica/articulación. "
        f"Minifiguras: {minifig_count}."
    )
    
    return (
        set_id,
        name,
        theme,
        year,
        category,
        subclass,
        minifig_count,
        minifigs_detail_json,
        color_palette_json,
        parts_composition_json,
        silhouette,
        llm_visual_description,
        llm_assembly_prompt,
        entorno,
        proposito,
        subclass,
        scale
    )


def process_all_sets():
    start_time = time.time()
    print("🚀 Iniciando Motor de Enriquecimiento Visual y Manifiestos de Sets...")
    
    conn = sqlite3.connect(DB_PATH)
    init_tables(conn)
    cursor = conn.cursor()
    
    # Fetch all sets from 'sets' table
    cursor.execute("""
        SELECT set_id, name, theme, year, parts_count, description
        FROM sets
        ORDER BY set_id;
    """)
    sets_data = cursor.fetchall()
    total_sets = len(sets_data)
    print(f"📦 Total de sets a procesar: {total_sets:,}")
    
    # Check memory before processing
    mem = psutil.virtual_memory()
    print(f"🧠 Memoria Disponible: {mem.available / (1024**3):.1f} GB / {mem.total / (1024**3):.1f} GB")
    
    # Fetch inventory data indexed by set_num
    print("🔍 Indexando inventarios, colores y taxonomía de piezas en memoria RAM...")
    cursor.execute("""
        SELECT 
            inv.set_num,
            c.name as color_name,
            t.functional_role,
            p.name as part_name,
            sum(ip.quantity) as qty
        FROM rb_inventories inv
        JOIN rb_inventory_parts ip ON ip.inventory_id = inv.id
        JOIN rb_colors c ON c.id = ip.color_id
        LEFT JOIN part_semantic_taxonomy t ON t.part_num = ip.part_num
        LEFT JOIN rb_parts p ON p.part_num = ip.part_num
        WHERE ip.is_spare = 0
        GROUP BY inv.set_num, c.name, t.functional_role, p.name;
    """)
    inv_rows = cursor.fetchall()
    
    # Organize in dict by set_num
    set_inventories: Dict[str, Dict[str, Any]] = {}
    for set_num, color_name, role, part_name, qty in inv_rows:
        if set_num not in set_inventories:
            set_inventories[set_num] = {
                "color_breakdown": {},
                "role_counts": {"wheels_count": 0}
            }
        item = set_inventories[set_num]
        item["color_breakdown"][color_name] = item["color_breakdown"].get(color_name, 0) + qty
        if role:
            item["role_counts"][role] = item["role_counts"].get(role, 0) + qty
        if part_name and any(w in part_name.lower() for w in ["wheel", "tire", "tyre"]):
            item["role_counts"]["wheels_count"] += qty

    # Fetch minifigures indexed by set_num
    print("👤 Indexando minifiguras asociadas...")
    cursor.execute("""
        SELECT inv.set_num, m.fig_num, m.name, im.quantity
        FROM rb_inventories inv
        JOIN rb_inventory_minifigs im ON im.inventory_id = inv.id
        JOIN rb_minifigs m ON m.fig_num = im.fig_num;
    """)
    minifig_rows = cursor.fetchall()
    set_minifigs: Dict[str, List[Dict[str, Any]]] = {}
    for set_num, fig_num, fig_name, qty in minifig_rows:
        if set_num not in set_minifigs:
            set_minifigs[set_num] = []
        set_minifigs[set_num].append({
            "fig_num": fig_num,
            "name": fig_name,
            "quantity": qty
        })

    # Prepare data packs for parallel processing
    data_packs = []
    for s_id, name, theme, year, parts_count, desc in sets_data:
        # Search direct set_id or with suffix '-1'
        inv_data = set_inventories.get(s_id) or set_inventories.get(f"{s_id}-1") or {
            "color_breakdown": {},
            "role_counts": {"wheels_count": 0}
        }
        minifigs = set_minifigs.get(s_id) or set_minifigs.get(f"{s_id}-1") or []
        
        data_packs.append({
            "set_id": s_id,
            "name": name or "",
            "theme": theme or "",
            "year": year or 0,
            "parts_count": parts_count or 0,
            "role_counts": inv_data["role_counts"],
            "color_breakdown": inv_data["color_breakdown"],
            "minifigs_list": minifigs
        })
        
    print(f"📦 Paquetes de datos listos ({len(data_packs):,} sets).")
    
    # M4 Dynamic multi-core workers: os.cpu_count() - 2 (10 workers)
    max_workers = max(1, (os.cpu_count() or 12) - 2)
    print(f"⚡ Procesando en paralelo con {max_workers} núcleos Apple Silicon M4...")
    
    chunk_size = 2000
    all_manifests = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(0, len(data_packs), chunk_size):
            chunk = data_packs[i:i + chunk_size]
            results = list(executor.map(generate_set_manifest_record, chunk))
            all_manifests.extend(results)
            print(f"   ✓ Generados {len(all_manifests):,} / {total_sets:,} manifiestos semánticos...")
            
    print(f"💾 Guardando {len(all_manifests):,} registros en tabla 'set_semantic_manifest'...")
    
    manifest_records = []
    sets_update_records = []
    for m in all_manifests:
        (s_id, name, theme, year, category, subclass, minifig_count, minifigs_detail_json,
         color_palette_json, parts_composition_json, silhouette, llm_visual_description,
         llm_assembly_prompt, entorno, proposito, subclase_txt, scale) = m
        
        manifest_records.append((
            s_id, name, theme, year, category, subclass, minifig_count, minifigs_detail_json,
            color_palette_json, parts_composition_json, silhouette, llm_visual_description,
            llm_assembly_prompt
        ))
        
        sets_update_records.append((
            llm_visual_description,
            entorno,
            proposito,
            subclase_txt,
            scale,
            s_id
        ))
        
    cursor.execute("BEGIN TRANSACTION;")
    cursor.executemany("""
        INSERT OR REPLACE INTO set_semantic_manifest 
        (set_id, set_name, theme, year, archetype_category, archetype_subclass, 
         minifigures_count, minifigures_detail, color_palette, parts_composition, 
         shape_and_silhouette, llm_visual_description, llm_construction_prompt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, manifest_records)
    
    # Update sets table: populate description and taxonomy levels where empty
    cursor.executemany("""
        UPDATE sets
        SET 
            description = COALESCE(description, ?),
            level_1_entorno = COALESCE(level_1_entorno, ?),
            level_2_proposito = COALESCE(level_2_proposito, ?),
            level_3_clase = COALESCE(level_3_clase, ?),
            level_4_escala = COALESCE(level_4_escala, ?)
        WHERE set_id = ?;
    """, sets_update_records)
    conn.commit()
    
    # Stats on categories
    cursor.execute("""
        SELECT archetype_category, count(*) 
        FROM set_semantic_manifest 
        GROUP BY archetype_category 
        ORDER BY count(*) DESC;
    """)
    arch_stats = cursor.fetchall()
    
    # Check sets table description coverage
    cursor.execute("SELECT count(*), count(description) FROM sets;")
    total_db_sets, described_db_sets = cursor.fetchone()
    
    conn.close()
    
    elapsed = time.time() - start_time
    print(f"\n🎉 Enriquecimiento completado con éxito en {elapsed:.2f} segundos!")
    print(f"📊 Cobertura de descripciones en tabla 'sets': {described_db_sets:,} / {total_db_sets:,} ({described_db_sets*100/total_db_sets:.1f}%)")
    print("\n--- DISTRIBUCIÓN DE ARQUETIPOS DE CONSTRUCCIÓN ---")
    for cat, cnt in arch_stats:
        print(f"  • {cat:<25}: {cnt:,} ({cnt*100/total_sets:.1f}%)")


if __name__ == "__main__":
    process_all_sets()
