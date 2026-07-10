from pydantic import BaseModel
from typing import Literal, Optional

class LegoTaxonomy(BaseModel):
    Level_0_Categoria: Optional[str] = None
    Level_1_Entorno: Literal['Terrestre', 'Acuático', 'Aéreo', 'Espacial', 'Multientorno', 'Otros']
    Level_2_Proposito: Literal['Civil/Pasajeros', 'Carga/Comercial', 'Emergencias/Servicios', 'Construccion/Industrial', 'Competicion/Deportes', 'Militar/Combate', 'Ficcion/Fantasia', 'Otros']
    Level_3_Clase: str  # (Ej: Coche, Moto, Tren, Barco, Caza_Estelar, Mech_Caminante. El modelo debe inferirlo)
    Level_4_Escala: Literal['Microscale', 'Minifig-scale', 'UCS/Gran Escala', 'Otros']

class LegoAnimalTaxonomy(BaseModel):
    Level_0_Categoria: Optional[str] = "Animal"
    Level_1_Habitat: Literal['Terrestre', 'Acuático', 'Aéreo', 'Anfibio/Multientorno', 'Extinto/Prehistórico', 'Mitológico/Fantasía']
    Level_2_Categoria: Literal['Mamífero', 'Ave', 'Reptil/Anfibio', 'Pez/Vida Marina', 'Insecto/Invertebrado', 'Dinosaurio', 'Criatura Fantástica', 'Otros']
    Level_3_Especie: str  # Ej: Gato, Perro, Oso, Tiburón, Águila, T-Rex, Dragón. El modelo debe inferirlo
    Level_4_Estilo: Literal['Escala Minifig', 'Escultura/Exhibición', 'Brick-built (Pequeña escala)', 'Otros']

class ClassificationResult(BaseModel):
    set_id: str
    source: Literal['Official', 'BrickLink', 'OMR']
    taxonomy_proposal: LegoTaxonomy
    confidence_score: float  # Valor entre 0.0 y 1.0
    reasoning_notes: str  # Breve explicación de por qué se eligieron esas categorías
    needs_human_review: bool  # True si confidence_score <= 0.80
    review_table_payload: Optional[dict] = None  # Payload generado si needs_human_review es True

class AnimalClassificationResult(BaseModel):
    set_id: str
    source: Literal['Official', 'BrickLink', 'OMR']
    taxonomy_proposal: LegoAnimalTaxonomy
    confidence_score: float  # Valor entre 0.0 y 1.0
    reasoning_notes: str
    needs_human_review: bool
    review_table_payload: Optional[dict] = None
