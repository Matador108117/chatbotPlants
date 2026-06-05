from unidecode import unidecode


def normalize(text: str) -> str:
    return unidecode(text).lower().strip()


# Intents soportados:
# toxicity | scientific | family | category | definition
# comparison | uses | warnings | general

_INTENT_MAP: list[tuple[str, list[str]]] = [
    ("comparison", [
        "diferencia", "comparar", "comparacion",
        "vs", "versus", "entre", "cual es mejor", "cual es mas",
    ]),
    ("toxicity", [
        "toxica", "toxico", "toxicidad",
        "venenosa", "venenoso", "veneno",
        "peligrosa", "peligroso",
    ]),
    ("scientific", [
        "nombre cientifico", "nombre científico",
        "cientifico", "científico", "especie", "taxonomia",
    ]),
    ("family", [
        "familia", "familia botanica", "familia botánica",
    ]),
    ("category", [
        "categoria", "categoría", "tipo", "clasificacion", "clasificación",
    ]),
    ("uses", [
        "uso", "usos", "para que sirve", "para qué sirve",
        "sirve para", "utilidad", "utilidades", "aplicacion", "aplicaciones",
    ]),
    ("warnings", [
        "advertencia", "advertencias", "precaucion", "precaución",
        "contraindicacion", "contraindicaciones", "cuidado", "riesgo",
    ]),
    ("definition", [
        "que es", "qué es", "describeme", "descríbeme",
        "descripcion", "descripción", "cuéntame", "cuentame",
    ]),
]


def detect_intent(question: str) -> str:
    q = normalize(question)
    for intent, keywords in _INTENT_MAP:
        if any(kw in q for kw in keywords):
            return intent
    return "general"
