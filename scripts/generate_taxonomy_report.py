import os
import sqlite3
import json

DB_PATH = "data/catalog/models_catalog.db"
OUTPUT_HTML = "taxonomy_vehicles_report.html"

def get_db_stats():
    if not os.path.exists(DB_PATH):
        return {}
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    stats = {}
    levels = {
        'level_1_entorno': 'Level_1_Entorno',
        'level_2_proposito': 'Level_2_Proposito',
        'level_3_clase': 'Level_3_Clase',
        'level_4_escala': 'Level_4_Escala',
        'level_4_motorizacion': 'Level_4_Motorizacion',
        'level_4_licencia': 'Level_4_Licencia'
    }
    
    for db_col, label in levels.items():
        cursor.execute(f"SELECT {db_col}, COUNT(*) FROM sets WHERE {db_col} IS NOT NULL GROUP BY {db_col} ORDER BY COUNT(*) DESC")
        stats[label] = cursor.fetchall()
        
    # Also get cross tabulations
    cursor.execute("""
        SELECT level_1_entorno, level_2_proposito, COUNT(*) 
        FROM sets 
        WHERE level_1_entorno IS NOT NULL AND level_2_proposito IS NOT NULL
        GROUP BY level_1_entorno, level_2_proposito
        ORDER BY level_1_entorno, level_2_proposito
    """)
    stats['cross_l1_l2'] = cursor.fetchall()
    
    conn.close()
    return stats

def main():
    stats = get_db_stats()
    
    # Predefined schema values (from src/classification_models.py)
    schema_l1 = ['Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno']
    schema_l2 = ['Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia']
    schema_l4_escala = ['Microscale', 'Minifig-scale', 'UCS/Gran Escala']
    schema_l4_moto = ['Estatico', 'Ruedas Libres', 'Pull-back', 'Motorizado']
    
    # Helper to generate rows for lists
    def make_values_list(schema_list, db_tuples):
        db_dict = dict(db_tuples) if db_tuples else {}
        rows_html = ""
        
        # Combine schema values and any additional values found in DB
        all_vals = list(schema_list)
        for val in db_dict.keys():
            if val not in all_vals:
                all_vals.append(val)
                
        for val in all_vals:
            count = db_dict.get(val, 0)
            status_badge = ""
            if val in schema_list:
                status_badge = '<span class="badge badge-schema">Definido en Esquema</span>'
            else:
                status_badge = '<span class="badge badge-dynamic">Dinámico / Libre</span>'
                
            rows_html += f"""
            <tr>
                <td class="font-semibold">{val}</td>
                <td>{status_badge}</td>
                <td class="text-right">{count}</td>
            </tr>
            """
        return rows_html

    # Helper for dynamic levels (Level 3 and License)
    def make_dynamic_list(db_tuples):
        if not db_tuples:
            return "<tr><td colspan='2' class='text-center text-muted'>Sin datos en la base de datos</td></tr>"
        rows_html = ""
        for val, count in db_tuples:
            rows_html += f"""
            <tr>
                <td class="font-semibold">{val}</td>
                <td class="text-right">{count}</td>
            </tr>
            """
        return rows_html

    l1_rows = make_values_list(schema_l1, stats.get('Level_1_Entorno', []))
    l2_rows = make_values_list(schema_l2, stats.get('Level_2_Proposito', []))
    l3_rows = make_dynamic_list(stats.get('Level_3_Clase', []))
    l4_escala_rows = make_values_list(schema_l4_escala, stats.get('Level_4_Escala', []))
    l4_moto_rows = make_values_list(schema_l4_moto, stats.get('Level_4_Motorizacion', []))
    l4_lic_rows = make_dynamic_list(stats.get('Level_4_Licencia', []))
    
    # Cross table
    cross_rows = ""
    for l1, l2, count in stats.get('cross_l1_l2', []):
        cross_rows += f"""
        <tr>
            <td>{l1}</td>
            <td>{l2}</td>
            <td class="text-right font-semibold">{count}</td>
        </tr>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Resumen de la Taxonomía de Vehículos - LegoGPT</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary: #1e3a8a;
            --primary-light: #eff6ff;
            --text-dark: #1e293b;
            --text-light: #64748b;
            --border-color: #cbd5e1;
            --bg-gray: #f8fafc;
        }}
        
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}
        
        body {{
            font-family: 'Inter', sans-serif;
            color: var(--text-dark);
            line-height: 1.6;
            background-color: #f1f5f9;
            padding: 40px 20px;
        }}
        
        .printable-card {{
            max-width: 850px;
            margin: 0 auto;
            background-color: #ffffff;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            border-radius: 12px;
            padding: 50px 60px;
            border: 1px solid #e2e8f0;
        }}
        
        /* Cover and Header */
        .header {{
            border-bottom: 3px solid var(--primary);
            padding-bottom: 24px;
            margin-bottom: 32px;
        }}
        
        .project-tag {{
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            color: var(--primary);
            text-transform: uppercase;
            font-size: 0.9rem;
            letter-spacing: 0.1em;
            margin-bottom: 8px;
            display: inline-block;
        }}
        
        h1 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 2.2rem;
            line-height: 1.2;
            color: #0f172a;
            margin-bottom: 12px;
        }}
        
        .subtitle {{
            color: var(--text-light);
            font-size: 1.1rem;
        }}
        
        /* Section styling */
        .section {{
            margin-bottom: 40px;
            page-break-inside: avoid;
        }}
        
        h2 {{
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 1.4rem;
            color: var(--primary);
            margin-bottom: 16px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 8px;
        }}
        
        p.intro {{
            margin-bottom: 20px;
            color: var(--text-light);
        }}
        
        /* Table Styling */
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
            font-size: 0.95rem;
        }}
        
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        th {{
            background-color: var(--bg-gray);
            font-weight: 600;
            color: var(--text-dark);
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        
        tr:hover {{
            background-color: #f8fafc;
        }}
        
        .text-right {{
            text-align: right;
        }}
        
        .font-semibold {{
            font-weight: 600;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }}
        
        .badge-schema {{
            background-color: #dbeafe;
            color: #1e40af;
        }}
        
        .badge-dynamic {{
            background-color: #fef3c7;
            color: #92400e;
        }}
        
        .badge-stat {{
            background-color: #f1f5f9;
            color: #334155;
            font-family: monospace;
        }}
        
        /* Printable Styles Override */
        @media print {{
            body {{
                background-color: #ffffff;
                padding: 0;
                color: #000000;
            }}
            
            .printable-card {{
                box-shadow: none;
                padding: 0;
                border: none;
                max-width: 100%;
            }}
            
            .section {{
                page-break-inside: avoid;
            }}
            
            h1, h2, h3, th {{
                color: #000000 !important;
            }}
            
            .badge-schema {{
                border: 1px solid #1e40af;
                background: none !important;
                color: #1e40af !important;
            }}
            
            .badge-dynamic {{
                border: 1px solid #92400e;
                background: none !important;
                color: #92400e !important;
            }}
            
            /* Add page numbers or page margins */
            @page {{
                size: A4;
                margin: 20mm;
            }}
        }}
    </style>
</head>
<body>

    <div class="printable-card">
        <!-- Header -->
        <div class="header">
            <span class="project-tag">PROYECTO LegoGPT</span>
            <h1>Catálogo de Taxonomía y Categorización de Vehículos</h1>
            <div class="subtitle">Estructura formal de clasificación multidimensional para modelos LDraw de vehículos</div>
        </div>

        <!-- Introducción -->
        <div class="section">
            <h2>1. Introducción a la Taxonomía</h2>
            <p class="intro">
                El sistema de clasificación de LegoGPT utiliza un esquema de taxonomía estructurado en 4 niveles para etiquetar e identificar con precisión las propiedades y el propósito físico de las construcciones de vehículos. Esto permite optimizar el entrenamiento de modelos generativos y automatizar la validación de MOCs (My Own Creations) y Sets oficiales.
            </p>
            <table>
                <thead>
                    <tr>
                        <th style="width: 25%;">Nivel de Taxonomía</th>
                        <th style="width: 30%;">Tipo de Dato</th>
                        <th>Descripción</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="font-semibold">Level 1: Entorno</td>
                        <td>Literal (Enum Cerrado)</td>
                        <td>El medio físico o entorno primario para el cual está diseñado el vehículo.</td>
                    </tr>
                    <tr>
                        <td class="font-semibold">Level 2: Propósito</td>
                        <td>Literal (Enum Cerrado)</td>
                        <td>La función de uso o intención operativa del vehículo.</td>
                    </tr>
                    <tr>
                        <td class="font-semibold">Level 3: Clase</td>
                        <td>Texto Dinámico (Inferencia)</td>
                        <td>La clase específica o tipología común del vehículo (ej: Coche, Helicóptero, Avión).</td>
                    </tr>
                    <tr>
                        <td class="font-semibold">Level 4: Escala</td>
                        <td>Literal (Enum Cerrado)</td>
                        <td>La escala física de construcción del modelo LEGO.</td>
                    </tr>
                    <tr>
                        <td class="font-semibold">Level 4: Motorización</td>
                        <td>Literal (Enum Cerrado)</td>
                        <td>El tipo de propulsión o funcionalidad mecánica implementada.</td>
                    </tr>
                    <tr>
                        <td class="font-semibold">Level 4: Licencia</td>
                        <td>Texto Dinámico (Inferencia)</td>
                        <td>La propiedad intelectual, marca o temática específica asociada.</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- Level 1 -->
        <div class="section">
            <h2>2. Nivel 1: Entorno (Level_1_Entorno)</h2>
            <p class="intro">Define el medio de transporte o desplazamiento del vehículo.</p>
            <table>
                <thead>
                    <tr>
                        <th>Valor de la Taxonomía</th>
                        <th>Estado de Validación</th>
                        <th class="text-right">Frecuencia en BD</th>
                    </tr>
                </thead>
                <tbody>
                    {l1_rows}
                </tbody>
            </table>
        </div>

        <!-- Level 2 -->
        <div class="section">
            <h2>3. Nivel 2: Propósito (Level_2_Proposito)</h2>
            <p class="intro">Especifica la finalidad operativa o temática de la creación.</p>
            <table>
                <thead>
                    <tr>
                        <th>Valor de la Taxonomía</th>
                        <th>Estado de Validación</th>
                        <th class="text-right">Frecuencia en BD</th>
                    </tr>
                </thead>
                <tbody>
                    {l2_rows}
                </tbody>
            </table>
        </div>

        <!-- Level 3 -->
        <div class="section">
            <h2>4. Nivel 3: Clase (Level_3_Clase)</h2>
            <p class="intro">Clasificación morfológica inferida dinámicamente por la IA para categorizar la forma física del vehículo.</p>
            <table>
                <thead>
                    <tr>
                        <th>Clase de Vehículo Detectada</th>
                        <th class="text-right">Modelos Registrados</th>
                    </tr>
                </thead>
                <tbody>
                    {l3_rows}
                </tbody>
            </table>
        </div>

        <!-- Level 4: Escala y Motorización -->
        <div class="section">
            <h2>5. Nivel 4: Atributos y Escala</h2>
            <p class="intro">Metadatos de detalle técnico y de escala del modelo LEGO.</p>
            
            <h3>Escala de Construcción</h3>
            <table>
                <thead>
                    <tr>
                        <th>Escala</th>
                        <th>Estado de Validación</th>
                        <th class="text-right">Frecuencia en BD</th>
                    </tr>
                </thead>
                <tbody>
                    {l4_escala_rows}
                </tbody>
            </table>

            <h3 style="margin-top: 24px; margin-bottom: 12px; font-family: 'Outfit'; font-weight: 600; color: var(--primary);">Motorización y Mecánica</h3>
            <table>
                <thead>
                    <tr>
                        <th>Mecanismo</th>
                        <th>Estado de Validación</th>
                        <th class="text-right">Frecuencia en BD</th>
                    </tr>
                </thead>
                <tbody>
                    {l4_moto_rows}
                </tbody>
            </table>
        </div>

        <!-- Level 4: Licencia -->
        <div class="section">
            <h2>6. Nivel 4: Licencias y Temáticas</h2>
            <p class="intro">Temáticas oficiales o licencias comerciales asociadas al modelo.</p>
            <table>
                <thead>
                    <tr>
                        <th>Licencia / Temática</th>
                        <th class="text-right">Frecuencia en BD</th>
                    </tr>
                </thead>
                <tbody>
                    {l4_lic_rows}
                </tbody>
            </table>
        </div>

        <!-- Distribución y Cruce -->
        <div class="section">
            <h2>7. Matriz de Distribución (Cruce L1 / L2)</h2>
            <p class="intro">Distribución cruzada de los modelos de vehículos clasificados según su Entorno (L1) y Propósito (L2):</p>
            <table>
                <thead>
                    <tr>
                        <th>Entorno (L1)</th>
                        <th>Propósito (L2)</th>
                        <th class="text-right">Cantidad de Modelos</th>
                    </tr>
                </thead>
                <tbody>
                    {cross_rows}
                </tbody>
            </table>
        </div>
        
    </div>

</body>
</html>
"""

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Reporte de taxonomía generado exitosamente en {OUTPUT_HTML}")

if __name__ == "__main__":
    main()
