import os
import sys
import sqlite3
import json

# Add workspace directory to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_REPORT = "classification_report.html"

def generate_report():
    print(f"Generando reporte HTML desde la base de datos: {DB_PATH}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Fetch all classified sets (where level_1_entorno is not NULL and not 'Otros')
    cursor.execute("""
    SELECT 
        s.set_id, s.name, s.theme, s.year, s.description, s.source, s.file_path, 
        COALESCE(s.parts_count, r.num_parts, 0) as parts_count,
        s.tags, s.source_url, s.image_url,
        s.level_1_entorno, s.level_2_proposito, s.level_3_clase, s.level_4_escala, s.level_4_motorizacion, s.level_4_licencia,
        s.confidence_score, s.reasoning_notes, s.needs_human_review, s.review_table_payload,
        s.subassemblies_count, s.is_fully_connected, s.classification_status
    FROM sets s
    LEFT JOIN rb_sets r ON s.set_id = r.set_num
    WHERE s.level_1_entorno IS NOT NULL AND s.level_1_entorno != 'Otros'
    ORDER BY s.confidence_score DESC, s.set_id ASC
    """)
    
    rows = cursor.fetchall()
    print(f"Encontrados {len(rows)} modelos clasificados.")
    
    if not rows:
        print("No hay modelos clasificados en la base de datos para reportar.")
        conn.close()
        return

    # Build catalog of images for each set_id
    cursor.execute("SELECT set_id, image_url FROM set_images")
    img_rows = cursor.fetchall()
    
    images_by_set = {}
    for sid, img_url in img_rows:
        if sid not in images_by_set:
            images_by_set[sid] = []
        images_by_set[sid].append(img_url)
        
    conn.close()
    
    # Build HTML Content
    html_items = []
    
    for idx, row in enumerate(rows):
        (
            set_id, name, theme, year, desc, source, file_path, parts, tags, src_url, main_img,
            l1, l2, l3, l4_escala, l4_moto, l4_lic,
            confidence, reasoning, needs_review, review_payload,
            subassemblies, fully_connected, classification_status) = row
        
        # Get all images for this set
        set_imgs = images_by_set.get(set_id, [])
        if not set_imgs and main_img:
            set_imgs = [main_img]
        if not set_imgs:
            # Fallback placeholder if no image exists
            set_imgs = ["https://placehold.co/600x400?text=No+Image+Available"]
            
        # Format tags
        tag_badges = ""
        if tags:
            tag_list = [t.strip() for t in tags.split(",") if t.strip()]
            for t in tag_list[:8]:  # Limit to 8 badges
                tag_badges += f'<span class="tag-badge">#{t}</span>'
                
        # Source badge style
        src_class = f"badge-{source.lower()}"
        
        # Review status badge style
        if classification_status == 'human_verified':
            review_class = "review-human-verified"
            review_text = "VALIDADO POR HUMANO"
        else:
            review_class = "review-required" if needs_review else "review-approved"
            review_text = "REQUIERE VALIDACIÓN" if needs_review else "AUTOCLASIFICADO VLM"
        
        # Confidence color
        confidence = confidence if confidence is not None else 0.0
        conf_color = "#10b981" if confidence > 0.8 else ("#f59e0b" if confidence > 0.5 else "#ef4444")
        conf_pct = int(confidence * 100)
        
        # Build carousel HTML
        carousel_slides = ""
        carousel_indicators = ""
        for i, url in enumerate(set_imgs):
            active_class = "active" if i == 0 else ""
            carousel_slides += f"""
            <div class="carousel-slide {active_class}">
                <img src="{url}" alt="Render {i+1} de {name}" onerror="this.src='https://placehold.co/600x400?text=Error+Loading+Image'">
                <div class="slide-number">{i+1} / {len(set_imgs)}</div>
            </div>
            """
            carousel_indicators += f"""
            <button class="carousel-dot {active_class}" onclick="setSlide('{set_id}', {i})"></button>
            """
            
        carousel_html = f"""
        <div class="carousel" id="carousel-{set_id}">
            <div class="carousel-track">
                {carousel_slides}
            </div>
            """
        if len(set_imgs) > 1:
            carousel_html += f"""
            <button class="carousel-control prev" onclick="moveSlide('{set_id}', -1)">&#10094;</button>
            <button class="carousel-control next" onclick="moveSlide('{set_id}', 1)">&#10095;</button>
            <div class="carousel-dots">
                {carousel_indicators}
            </div>
            """
        carousel_html += "</div>"
        
        # Format metadata variables safely
        subassemblies = subassemblies if subassemblies is not None else "N/A"
        fully_connected_str = "Sí" if fully_connected == 1 else ("No" if fully_connected == 0 else "N/A")
        
        item_html = f"""
        <div class="card" data-source="{source}" data-review="{needs_review}">
            <div class="card-header">
                <div class="header-left">
                    <span class="source-badge {src_class}">{source}</span>
                    <span class="review-badge {review_class}">{review_text}</span>
                    <h2 class="model-title">{name} <span class="model-id">({set_id})</span></h2>
                </div>
                <div class="confidence-gauge" style="--conf-color: {conf_color}">
                    <div class="gauge-value">{conf_pct}%</div>
                    <div class="gauge-label">Confianza</div>
                </div>
            </div>
            
            <div class="card-body">
                <div class="media-column">
                    {carousel_html}
                    <div class="tags-container">
                        {tag_badges}
                    </div>
                </div>
                
                <div class="info-column">
                    <div class="info-section">
                        <h3>Propuesta de Taxonomía</h3>
                        <div class="taxonomy-grid">
                            <div class="tax-item"><strong>Entorno (L1):</strong> <span>{l1}</span></div>
                            <div class="tax-item"><strong>Propósito (L2):</strong> <span>{l2}</span></div>
                            <div class="tax-item"><strong>Clase (L3):</strong> <span class="highlight">{l3}</span></div>
                            <div class="tax-item"><strong>Escala (L4):</strong> <span>{l4_escala}</span></div>
                        </div>
                    </div>
                    
                    <div class="info-section">
                        <h3>Metadatos Escrapeados y 3D</h3>
                        <div class="metadata-grid">
                            <div class="meta-item"><strong>Piezas:</strong> <span>{parts}</span></div>
                            <div class="meta-item"><strong>Subensamblajes:</strong> <span>{subassemblies}</span></div>
                            <div class="meta-item"><strong>Totalmente Conectado:</strong> <span>{fully_connected_str}</span></div>
                            <div class="meta-item"><strong>Temática:</strong> <span>{theme}</span></div>
                            <div class="meta-item"><strong>Año:</strong> <span>{year or 'Desconocido'}</span></div>
                            <div class="meta-item"><strong>Origen URL:</strong> <span><a href="{src_url or '#'}" target="_blank" class="link-url">Visitar Enlace</a></span></div>
                        </div>
                    </div>
                    
                    <div class="info-section">
                        <h3>Descripción</h3>
                        <p class="description-text">{desc or 'Sin descripción disponible.'}</p>
                    </div>

                    <div class="info-section">
                        <h3>Notas de Razonamiento de la IA</h3>
                        <div class="reasoning-box">
                            {reasoning}
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        html_items.append(item_html)
        
    # Write full HTML template
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Clasificación de Vehículos LEGO</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-color: #0b0f19;
            --card-bg: #151c2c;
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.15);
            --border-color: #243049;
            
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Plus Jakarta Sans', sans-serif;
            line-height: 1.5;
            padding: 40px 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 40px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 24px;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #a5b4fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        
        /* Filters Bar */
        .filters-bar {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .filter-group {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .filter-label {{
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--text-muted);
        }}
        
        select, input {{
            background-color: #1e293b;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-color);
            padding: 8px 12px;
            font-family: inherit;
            outline: none;
            cursor: pointer;
        }}
        
        /* Cards */
        .cards-list {{
            display: flex;
            flex-direction: column;
            gap: 30px;
        }}
        
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .card:hover {{
            box-shadow: 0 15px 40px rgba(99, 102, 241, 0.1);
        }}
        
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 24px;
            flex-wrap: wrap;
            gap: 20px;
        }}
        
        .header-left {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        
        .source-badge {{
            align-self: flex-start;
            font-size: 0.75rem;
            font-weight: 800;
            text-transform: uppercase;
            padding: 4px 10px;
            border-radius: 9999px;
            letter-spacing: 0.05em;
        }}
        
        .badge-official {{
            background-color: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
        }}
        
        .badge-bricklink {{
            background-color: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .badge-omr {{
            background-color: rgba(168, 85, 247, 0.15);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }}
        
        .review-badge {{
            align-self: flex-start;
            font-size: 0.75rem;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 6px;
        }}
        
        .review-required {{
            background-color: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.3);
        }}
        
        .review-approved {{
            background-color: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}
        
        .model-title {{
            font-family: 'Outfit', sans-serif;
            font-size: 1.8rem;
            font-weight: 600;
        }}
        
        .model-id {{
            color: var(--text-muted);
            font-size: 1.1rem;
            font-weight: 400;
        }}
        
        .confidence-gauge {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            border-left: 2px solid var(--border-color);
            padding-left: 24px;
        }}
        
        .gauge-value {{
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--conf-color);
            font-family: 'Outfit', sans-serif;
        }}
        
        .gauge-label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        /* Card Body Layout */
        .card-body {{
            display: grid;
            grid-template-columns: 400px 1fr;
            gap: 40px;
        }}
        
        @media (max-width: 900px) {{
            .card-body {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* Carousel styling */
        .carousel {{
            width: 100%;
            position: relative;
            background-color: #0b0f19;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            aspect-ratio: 4 / 3;
        }}
        
        .carousel-track {{
            display: flex;
            width: 100%;
            height: 100%;
            transition: transform 0.3s ease-in-out;
        }}
        
        .carousel-slide {{
            min-width: 100%;
            height: 100%;
            display: none;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        
        .carousel-slide.active {{
            display: flex;
        }}
        
        .carousel-slide img {{
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }}
        
        .slide-number {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            background-color: rgba(0, 0, 0, 0.7);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        
        .carousel-control {{
            position: absolute;
            top: 50%;
            transform: translateY(-50%);
            background-color: rgba(0, 0, 0, 0.5);
            color: white;
            border: none;
            width: 36px;
            height: 36px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.2s;
        }}
        
        .carousel-control:hover {{
            background-color: var(--primary);
        }}
        
        .carousel-control.prev {{ left: 12px; }}
        .carousel-control.next {{ right: 12px; }}
        
        .carousel-dots {{
            position: absolute;
            bottom: 12px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 6px;
        }}
        
        .carousel-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: rgba(255, 255, 255, 0.4);
            border: none;
            cursor: pointer;
            padding: 0;
        }}
        
        .carousel-dot.active {{
            background-color: white;
            transform: scale(1.2);
        }}
        
        /* Tags */
        .tags-container {{
            margin-top: 16px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }}
        
        .tag-badge {{
            font-size: 0.75rem;
            background-color: #1e293b;
            color: var(--text-muted);
            padding: 4px 8px;
            border-radius: 4px;
            border: 1px solid var(--border-color);
        }}
        
        /* Info Column */
        .info-column {{
            display: flex;
            flex-direction: column;
            gap: 24px;
        }}
        
        .info-section h3 {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 6px;
            margin-bottom: 12px;
        }}
        
        .taxonomy-grid, .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 12px;
        }}
        
        .tax-item, .meta-item {{
            background-color: #1a2235;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 0.9rem;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        
        .tax-item strong, .meta-item strong {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }}
        
        .tax-item span.highlight {{
            color: #a5b4fc;
            font-weight: 600;
        }}
        
        .description-text {{
            font-size: 0.95rem;
            color: #d1d5db;
        }}
        
        .reasoning-box {{
            background-color: #0b0f19;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 16px;
            font-size: 0.9rem;
            color: #e5e7eb;
            font-style: italic;
            border-left: 4px solid var(--primary);
        }}
        
        .link-url {{
            color: var(--primary);
            text-decoration: none;
            font-weight: 600;
        }}
        
        .link-url:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>LEGO Vehicle Classification Accuracy Report</h1>
            <div class="subtitle">Análisis del proceso de asignación de categorías para MOCs y Sets ({len(rows)} modelos cargados)</div>
        </header>
        
        <div class="filters-bar">
            <div class="filter-group">
                <span class="filter-label">Fuente:</span>
                <select id="filter-source" onchange="applyFilters()">
                    <option value="all">Todas</option>
                    <option value="Official">Sets Oficiales</option>
                    <option value="BrickLink">BrickLink Gallery</option>
                    <option value="OMR">OMR LDraw</option>
                </select>
            </div>
            <div class="filter-group">
                <span class="filter-label">Estado de Revisión:</span>
                <select id="filter-review" onchange="applyFilters()">
                    <option value="all">Todos</option>
                    <option value="1">Requiere Validación (<= 80%)</option>
                    <option value="0">Autoclasificados (> 80%)</option>
                </select>
            </div>
            <div class="filter-group" style="margin-left: auto;">
                <input type="text" id="search-input" placeholder="Buscar por nombre o ID..." oninput="applyFilters()">
            </div>
        </div>
        
        <div class="cards-list">
            {"".join(html_items)}
        </div>
    </div>

    <script>
        // Slideshow controls
        const slideStates = {{}};
        
        function setSlide(setId, slideIdx) {{
            const carousel = document.getElementById('carousel-' + setId);
            if (!carousel) return;
            
            const slides = carousel.querySelectorAll('.carousel-slide');
            const dots = carousel.querySelectorAll('.carousel-dot');
            
            slides.forEach((slide, idx) => {{
                if (idx === slideIdx) {{
                    slide.classList.add('active');
                }} else {{
                    slide.classList.remove('active');
                }}
            }});
            
            dots.forEach((dot, idx) => {{
                if (idx === slideIdx) {{
                    dot.classList.add('active');
                }} else {{
                    dot.classList.remove('active');
                }}
            }});
            
            slideStates[setId] = slideIdx;
        }}
        
        function moveSlide(setId, direction) {{
            const carousel = document.getElementById('carousel-' + setId);
            if (!carousel) return;
            
            const slides = carousel.querySelectorAll('.carousel-slide');
            const total = slides.length;
            
            let current = slideStates[setId] || 0;
            current = (current + direction + total) % total;
            
            setSlide(setId, current);
        }}
        
        // Filtering logic
        function applyFilters() {{
            const sourceFilter = document.getElementById('filter-source').value;
            const reviewFilter = document.getElementById('filter-review').value;
            const searchVal = document.getElementById('search-input').value.toLowerCase();
            
            const cards = document.querySelectorAll('.card');
            
            cards.forEach(card => {{
                const src = card.getAttribute('data-source');
                const rev = card.getAttribute('data-review');
                const titleText = card.querySelector('.model-title').innerText.toLowerCase();
                
                const matchesSource = (sourceFilter === 'all' || src === sourceFilter);
                const matchesReview = (reviewFilter === 'all' || rev === reviewFilter);
                const matchesSearch = (!searchVal || titleText.includes(searchVal));
                
                if (matchesSource && matchesReview && matchesSearch) {{
                    card.style.display = 'block';
                }} else {{
                    card.style.display = 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
"""
    # Save Report
    os.makedirs(os.path.dirname(OUTPUT_REPORT) if os.path.dirname(OUTPUT_REPORT) else ".", exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"Reporte HTML guardado exitosamente en {OUTPUT_REPORT}!")

if __name__ == "__main__":
    generate_report()
