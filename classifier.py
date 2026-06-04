from unidecode import unidecode


def normalize(text: str) -> str:
    return unidecode(text).lower().strip()


def detect_intent(question: str) -> str:

    q = normalize(question)

    comparison_words = [
        "diferencia",
        "comparar",
        "comparacion",
        "versus",
        "vs"
    ]

    toxicity_words = [
        "toxica",
        "toxico",
        "venenosa",
        "venenoso"
    ]

    scientific_words = [
        "nombre cientifico",
        "cientifico"
    ]

    if any(word in q for word in comparison_words):
        return "comparison"

    if any(word in q for word in toxicity_words):
        return "toxicity"

    if any(word in q for word in scientific_words):
        return "scientific"

    return "general"