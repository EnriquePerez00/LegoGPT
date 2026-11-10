import sqlite3
import json
import os


def setup_connector_ontology(db_path: str = "data/catalog/models_catalog.db"):
    """
    Initializes the formal LEGO physical connector ontology tables in models_catalog.db.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Table for Connector Types (Ontology definitions)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lego_connector_types (
        type_id TEXT PRIMARY KEY,
        family TEXT NOT NULL,
        gender TEXT NOT NULL,
        description TEXT NOT NULL,
        dof INTEGER DEFAULT 0,
        compatible_types TEXT NOT NULL
    )
    """)

    # 2. Table for Part-Level Physical Connectors (Instances within 3D LDraw parts)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS lego_part_connectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        part_num TEXT NOT NULL,
        connector_idx INTEGER NOT NULL,
        type_id TEXT NOT NULL,
        pos_x REAL NOT NULL,
        pos_y REAL NOT NULL,
        pos_z REAL NOT NULL,
        norm_x REAL NOT NULL,
        norm_y REAL NOT NULL,
        norm_z REAL NOT NULL,
        up_x REAL DEFAULT 0.0,
        up_y REAL DEFAULT 1.0,
        up_z REAL DEFAULT 0.0,
        is_subpart INTEGER DEFAULT 0,
        FOREIGN KEY(type_id) REFERENCES lego_connector_types(type_id)
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_part_connectors_part ON lego_part_connectors(part_num)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_part_connectors_type ON lego_part_connectors(type_id)")

    # Standard ontology dictionary of physical connector classes
    connector_types = [
        # STUD Family
        {
            "type_id": "STUD_SOLID",
            "family": "STUD",
            "gender": "MALE",
            "description": "Estándar stud superior cilíndrico sólido (radio 6 LDU, altura 4 LDU)",
            "dof": 1,
            "compatible_types": ["SOCKET_TUBE", "SOCKET_ANTISTUD", "SOCKET_CORNER", "SOCKET_LATERAL"]
        },
        {
            "type_id": "STUD_HOLLOW",
            "family": "STUD",
            "gender": "MALE",
            "description": "Stud con orificio central pasante de 3.18mm (admite barras y clips)",
            "dof": 1,
            "compatible_types": ["SOCKET_TUBE", "SOCKET_ANTISTUD", "SOCKET_CORNER", "BAR_MALE"]
        },
        {
            "type_id": "STUD_SNOT_LATERAL",
            "family": "STUD_LATERAL",
            "gender": "MALE",
            "description": "Stud lateral SNOT orientado horizontalmente",
            "dof": 1,
            "compatible_types": ["SOCKET_TUBE", "SOCKET_ANTISTUD", "SOCKET_LATERAL"]
        },
        {
            "type_id": "SOCKET_TUBE",
            "family": "STUD",
            "gender": "FEMALE",
            "description": "Tubo inferior cilíndrico estándar entre studs (antistud primario)",
            "dof": 1,
            "compatible_types": ["STUD_SOLID", "STUD_HOLLOW", "STUD_SNOT_LATERAL"]
        },
        {
            "type_id": "SOCKET_ANTISTUD",
            "family": "STUD",
            "gender": "FEMALE",
            "description": "Hueco de esquina o hueco modular 1x1 para stud",
            "dof": 1,
            "compatible_types": ["STUD_SOLID", "STUD_HOLLOW", "STUD_SNOT_LATERAL"]
        },
        {
            "type_id": "SOCKET_LATERAL",
            "family": "STUD_LATERAL",
            "gender": "FEMALE",
            "description": "Hueco antistud en cara vertical o lateral",
            "dof": 1,
            "compatible_types": ["STUD_SOLID", "STUD_HOLLOW", "STUD_SNOT_LATERAL"]
        },

        # TECHNIC Family
        {
            "type_id": "PIN_MALE",
            "family": "TECHNIC",
            "gender": "MALE",
            "description": "Pin Technic estándar con ranura de fijación por fricción o libre",
            "dof": 1,
            "compatible_types": ["PIN_HOLE"]
        },
        {
            "type_id": "PIN_HOLE",
            "family": "TECHNIC",
            "gender": "FEMALE",
            "description": "Orificio cilíndrico Technic liso de 4.8mm",
            "dof": 1,
            "compatible_types": ["PIN_MALE", "AXLE_MALE"]
        },
        {
            "type_id": "AXLE_MALE",
            "family": "TECHNIC",
            "gender": "MALE",
            "description": "Eje en cruz Technic con perfil estriado",
            "dof": 0,
            "compatible_types": ["AXLE_HOLE", "PIN_HOLE"]
        },
        {
            "type_id": "AXLE_HOLE",
            "family": "TECHNIC",
            "gender": "FEMALE",
            "description": "Orificio con forma de cruz para acoplamiento de eje Technic",
            "dof": 0,
            "compatible_types": ["AXLE_MALE"]
        },

        # CLIP & BAR Family
        {
            "type_id": "BAR_MALE",
            "family": "CLIP_BAR",
            "gender": "MALE",
            "description": "Barra cilíndrica de 3.18mm (antenas, asas, varillas)",
            "dof": 2,
            "compatible_types": ["CLIP_FEMALE", "STUD_HOLLOW"]
        },
        {
            "type_id": "CLIP_FEMALE",
            "family": "CLIP_BAR",
            "gender": "FEMALE",
            "description": "Pinza/mordaza flexible en forma de C para barra de 3.18mm",
            "dof": 2,
            "compatible_types": ["BAR_MALE"]
        },

        # BALL JOINT Family
        {
            "type_id": "BALL_MALE",
            "family": "BALL_JOINT",
            "gender": "MALE",
            "description": "Bola esférica de articulación (sistema Mixel o Technic towball)",
            "dof": 3,
            "compatible_types": ["BALL_SOCKET_FEMALE"]
        },
        {
            "type_id": "BALL_SOCKET_FEMALE",
            "family": "BALL_JOINT",
            "gender": "FEMALE",
            "description": "Copa/zócalo esférico hembra para rótula",
            "dof": 3,
            "compatible_types": ["BALL_MALE"]
        },

        # HINGE Family
        {
            "type_id": "HINGE_MALE",
            "family": "HINGE",
            "gender": "MALE",
            "description": "Articulación dentada macho (bisagra con pasador)",
            "dof": 1,
            "compatible_types": ["HINGE_FEMALE"]
        },
        {
            "type_id": "HINGE_FEMALE",
            "family": "HINGE",
            "gender": "FEMALE",
            "description": "Articulación dentada hembra receptora de bisagra",
            "dof": 1,
            "compatible_types": ["HINGE_MALE"]
        }
    ]

    for ct in connector_types:
        cur.execute("""
        INSERT OR REPLACE INTO lego_connector_types 
        (type_id, family, gender, description, dof, compatible_types)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (
            ct["type_id"],
            ct["family"],
            ct["gender"],
            ct["description"],
            ct["dof"],
            json.dumps(ct["compatible_types"])
        ))

    conn.commit()
    print(f"Ontología de conectores LEGO inicializada con éxito: {len(connector_types)} tipos registrados.")
    conn.close()


if __name__ == "__main__":
    setup_connector_ontology()
