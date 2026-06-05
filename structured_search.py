import json
from unidecode import unidecode
 
 
def normalize(text: str) -> str:
    return unidecode(text).lower().strip()
 
 
class PlantStructuredSearch:
    def __init__(self, json_path: str = "plants.json") -> None:
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
 
        # Acepta tanto lista directa como {"plantas": [...]}
        self.plants: list[dict] = raw if isinstance(raw, list) else raw.get("plantas", [])
 
        # lookup normalizado → planta (nombre, alias, nombre_cientifico)
        self.lookup: dict[str, dict] = {}
        for plant in self.plants:
            candidates = [
                plant.get("nombre", ""),
                plant.get("nombre_cientifico", ""),
            ]
            candidates.extend(plant.get("Alias", []))
            for name in candidates:
                if name:
                    key = normalize(name)
                    if key not in self.lookup:
                        self.lookup[key] = plant
 
    # ------------------------------------------------------------------
    # Búsqueda principal
    # ------------------------------------------------------------------
    def find_plants(self, question: str) -> list[dict]:
        """
        Devuelve lista de plantas mencionadas en la pregunta.
        Detecta por nombre, alias y nombre científico.
        Elimina duplicados preservando orden.
        """
        q = normalize(question)
        seen: set[str] = set()
        matches: list[dict] = []
 
        for key, plant in self.lookup.items():
            if key in q:
                nombre = plant["nombre"]
                if nombre not in seen:
                    matches.append(plant)
                    seen.add(nombre)
 
        return matches
 
    # ------------------------------------------------------------------
    # Detección de comparación
    # ------------------------------------------------------------------
    def is_comparison(self, question: str) -> bool:
        """
        Devuelve True si la pregunta pide comparar dos o más plantas.
        """
        q = normalize(question)
        comparison_triggers = [
            "diferencia",
            "comparar",
            "comparacion",
            "vs",
            "versus",
            "entre",
            "cual es mejor",
            "cual es mas",
        ]
        plants_found = self.find_plants(question)
        has_trigger = any(t in q for t in comparison_triggers)
        return len(plants_found) >= 2 or (has_trigger and len(plants_found) >= 1)
 
    # ------------------------------------------------------------------
    # Compatibilidad con nombre antiguo
    # ------------------------------------------------------------------
    def find_plant(self, question: str) -> list[dict]:
        return self.find_plants(question)
 