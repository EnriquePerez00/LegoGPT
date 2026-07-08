from pydantic import BaseModel
from typing import Literal, Optional

class LegoTaxonomy(BaseModel):
    Level_1_Entorno: Literal['Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno']
    Level_2_Proposito: Literal['Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia']
    Level_3_Clase: str  # (Ej: Coche, Moto, Tren, Caza_Estelar, Mech_Caminante. El modelo debe inferirlo)
    Level_4_Escala: Literal['Microscale', 'Minifig-scale', 'UCS/Gran Escala']
    Level_4_Motorizacion: Literal['Estatico', 'Ruedas Libres', 'Pull-back', 'Motorizado']
    Level_4_Licencia: str  # (Ej: Genérico, Star Wars, City)

class ClassificationResult(BaseModel):
    set_id: str
    source: Literal['Official', 'BrickLink', 'OMR']
    taxonomy_proposal: LegoTaxonomy
    confidence_score: float  # Valor entre 0.0 y 1.0
    reasoning_notes: str  # Breve explicación de por qué se eligieron esas categorías
    needs_human_review: bool  # True si confidence_score <= 0.80
    review_table_payload: Optional[dict] = None  # Payload generado si needs_human_review es True
