import sqlite3
import argparse

def query_catalog(db_path="data/catalog/models_catalog.db", args=None):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if args.show:
        # Show set details and inventory
        cursor.execute("SELECT name, theme, description, parts_count FROM sets WHERE set_id = ?", (args.show,))
        set_info = cursor.fetchone()
        if not set_info:
            print(f"Set '{args.show}' no encontrado.")
            conn.close()
            return
            
        print(f"\n==================================================")
        print(f"Detalles del Set: {args.show}")
        print(f"Nombre: {set_info[0]}")
        print(f"Temática: {set_info[1]}")
        print(f"Descripción: {set_info[2]}")
        print(f"Total Piezas: {set_info[3]}")
        print(f"==================================================")
        
        # Query inventory
        cursor.execute("""
        SELECT part_id, color, quantity FROM parts_inventory 
        WHERE set_id = ? 
        ORDER BY quantity DESC
        """, (args.show,))
        inv = cursor.fetchall()
        print(f"Inventario ({len(inv)} tipos de piezas únicas):")
        for part, col, qty in inv[:25]: # limit to top 25
            print(f"  - Pieza: {part:<12} Color: {col:<4} Cantidad: {qty}")
        if len(inv) > 25:
            print(f"  ... y {len(inv) - 25} tipos de piezas más.")
        conn.close()
        return

    # Build dynamically queries
    query = """
    SELECT DISTINCT s.set_id, s.name, s.theme, s.parts_count, s.source 
    FROM sets s
    """
    
    conditions = []
    params = []
    
    if args.part:
        query += " JOIN parts_inventory i ON s.set_id = i.set_id"
        conditions.append("i.part_id = ?")
        params.append(args.part.lower())
        
    if args.theme:
        conditions.append("s.theme LIKE ?")
        params.append(f"%{args.theme}%")
        
    if args.keyword:
        conditions.append("(s.name LIKE ? OR s.description LIKE ?)")
        params.append(f"%{args.keyword}%")
        params.append(f"%{args.keyword}%")
        
    if args.min_parts is not None:
        conditions.append("s.parts_count >= ?")
        params.append(args.min_parts)
        
    if args.max_parts is not None:
        conditions.append("s.parts_count <= ?")
        params.append(args.max_parts)
        
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
        
    query += " ORDER BY s.parts_count DESC"
    
    # Run query
    cursor.execute(query, params)
    results = cursor.fetchall()
    
    print(f"\nEncontrados {len(results)} modelos coincidentes:")
    print(f"{'Set ID':<15} | {'Nombre':<30} | {'Temática':<25} | {'Piezas':<6} | {'Fuente':<6}")
    print("-" * 90)
    for row in results[:30]: # Limit to top 30
        print(f"{row[0]:<15} | {row[1][:30]:<30} | {row[2][:25]:<25} | {row[3]:<6} | {row[4]:<6}")
    if len(results) > 30:
        print(f"... y {len(results) - 30} modelos más.")
        
    conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query Lego 3D Model Catalog Database")
    parser.add_argument("--part", type=str, help="Search by LDraw Part ID")
    parser.add_argument("--theme", type=str, help="Search by Theme name")
    parser.add_argument("--keyword", type=str, help="Search by keyword in name/description")
    parser.add_argument("--min_parts", type=int, help="Minimum part count")
    parser.add_argument("--max_parts", type=int, help="Maximum part count")
    parser.add_argument("--show", type=str, help="Show full inventory of target set")
    
    args = parser.parse_args()
    query_catalog(args=args)
