import json
from unidecode import unidecode


def normalize(text: str) -> str:
    return (
        unidecode(text)
        .lower()
        .strip()
    )


class PlantStructuredSearch:
    def __init__(self, json_path: str = "plants.json"):
        with open(json_path, encoding="utf-8") as f:
            self.plants = json.load(f)

        self.lookup: dict = {}
        for plant in self.plants:
            names = [
                plant.get("nombre", ""),
                plant.get("nombre_cientifico", ""),
            ]
            names.extend(plant.get("Alias", []))

            for name in names:
                if name:
                    self.lookup[normalize(name)] = plant

    def find_plant(self, question: str) -> list:
        q = normalize(question)
        matches = []

        for key, plant in self.lookup.items():
            if key in q:
                matches.append(plant)

        if not matches:
            return []

        unique = []
        seen: set = set()
        for plant in matches:
            name = plant["nombre"]
            if name not in seen:
                unique.append(plant)
                seen.add(name)

        return unique