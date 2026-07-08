import sqlite3
import sys

DB_PATH = "data/catalog/models_catalog.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total unique parts in City
    cursor.execute("""
        SELECT COUNT(DISTINCT ip.part_num)
        FROM rb_inventory_parts ip
        JOIN rb_inventories i ON ip.inventory_id = i.id
        JOIN rb_sets s ON i.set_num = s.set_num
        JOIN rb_themes t ON s.theme_id = t.id
        WHERE t.name LIKE '%City%'
    """)
    city_total = cursor.fetchone()[0]
    
    # Total unique parts in City that have been enriched
    cursor.execute("""
        SELECT COUNT(DISTINCT ip.part_num)
        FROM rb_inventory_parts ip
        JOIN rb_inventories i ON ip.inventory_id = i.id
        JOIN rb_sets s ON i.set_num = s.set_num
        JOIN rb_themes t ON s.theme_id = t.id
        JOIN rb_parts_enriched e ON ip.part_num = e.part_num
        WHERE t.name LIKE '%City%'
    """)
    city_enriched = cursor.fetchone()[0]
    
    conn.close()
    
    percentage = (city_enriched / city_total * 100) if city_total > 0 else 0
    print(f"STATUS: {city_enriched}/{city_total} ({percentage:.2f}%)")
    
    if city_enriched >= city_total and city_total > 0:
        sys.exit(0)  # Finished
    else:
        sys.exit(1)  # Still running

if __name__ == "__main__":
    main()
