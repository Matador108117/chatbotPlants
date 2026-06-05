import json
import requests
import chromadb

from sentence_transformers import SentenceTransformer
from structured_search import PlantStructuredSearch
from intent_detector import detect_intent

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"
CHROMA_PATH = "./chroma_db"
PLANTS_JSON = "plants.json"
COLLECTION_NAME = "plants"

# ---------------------------------------------------------------------------
# Inicialización (una sola vez al importar)
# ---------------------------------------------------------------------------
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION_NAME)
structured = PlantStructuredSearch(PLANTS_JSON)


# ---------------------------------------------------------------------------
# Recuperación de contexto híbrido JSON + Chroma
# ---------------------------------------------------------------------------
def retrieve_context(question: str) -> tuple[list[dict], str]:
    plants = structured.find_plants(question)

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True,
    ).tolist()

    contexts: list[str] = []

    if plants:
        for plant in plants:
            plant_id = plant["nombre"].lower().replace(" ", "_")
            try:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=3,
                    where={"plant": plant_id},
                )
                docs = results.get("documents", [[]])[0]
                contexts.extend(docs)
            except Exception:
                pass
    else:
        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=5,
            )
            docs = results.get("documents", [[]])[0]
            contexts.extend(docs)
        except Exception:
            pass

    context_text = "\n\n".join(contexts).strip()
    return plants, context_text


# ---------------------------------------------------------------------------
# Sistema de prompt: identidad Aloe + primera persona + anti-alucinación
# ---------------------------------------------------------------------------

# Prompt de sistema fijo — define la identidad de Aloe de forma autoritaria.
# Qwen respeta un bloque de sistema explícito y separado de la instrucción.
_SYSTEM_IDENTITY = """Eres Aloe. Eres una asistente botánica especializada en plantas medicinales, comestibles y tóxicas.

Tu forma de hablar:
- Hablas siempre en PRIMERA PERSONA. Nunca digas "el asistente" ni "según la información". Di "yo sé", "en mi base de datos tengo", "lo que conozco de esta planta es…"
- Tu tono es cálido, directo y experto. Como una botánica de confianza, no como un manual.
- Cuando sí tienes datos: comparte todo lo que sabes con detalle y naturalidad.
- Cuando NO tienes datos de una planta: lo dices honestamente en primera persona. Ejemplo: "Honestamente, esa planta no está en mi base de datos. No tengo información sobre ella y prefiero no inventar nada."
- NUNCA inventes nombres científicos, familias, usos ni propiedades que no estén en los DATOS ESTRUCTURADOS o el CONTEXTO. Si no está ahí, no lo sabes.
- No menciones que tienes un "contexto recuperado" ni "datos estructurados". Eso es interno. Habla como si el conocimiento fuera tuyo."""


def build_prompt(
    question: str,
    plants: list[dict],
    context: str,
    intent: str,
) -> str:
    structured_block = ""
    for plant in plants:
        structured_block += json.dumps(plant, ensure_ascii=False, indent=2) + "\n\n"
    structured_block = structured_block.strip()

    intent_instruction = _intent_instruction(intent, plants)

    has_data = bool(structured_block or context)
    context_block = context if context else "Sin contexto disponible."

    # Aviso explícito de vacío para que Aloe no alucine
    data_warning = (
        ""
        if has_data
        else "\n⚠️ ADVERTENCIA: No hay datos sobre esta planta en tu base de conocimiento. "
             "Debes decirle al usuario honestamente que no tienes información sobre ella. "
             "No inventes nada.\n"
    )

    return f"""{_SYSTEM_IDENTITY}

---

INSTRUCCIÓN PARA ESTA RESPUESTA:
{intent_instruction}
{data_warning}
DATOS QUE CONOCES (usa estos como fuente principal):
{structured_block if structured_block else "— ninguno para esta consulta —"}

CONTEXTO ADICIONAL (complementa si aporta algo nuevo):
{context_block}

PREGUNTA DEL USUARIO:
{question}

TU RESPUESTA (en primera persona, en español):"""


def _intent_instruction(intent: str, plants: list[dict]) -> str:
    names = [p["nombre"] for p in plants]
    listing = " y ".join(names) if names else "la planta que menciona el usuario"

    instructions: dict[str, str] = {
        "toxicity": (
            f"El usuario pregunta sobre toxicidad de {listing}. "
            f"Explica en primera persona qué tan tóxica es, qué síntomas puede causar "
            f"y qué precauciones recomiendas. Si no es tóxica, dilo claramente."
        ),
        "scientific": (
            f"El usuario quiere saber el nombre científico de {listing}. "
            f"Dalo junto con la familia botánica. Habla como si fuera tu conocimiento propio."
        ),
        "family": (
            f"El usuario pregunta por la familia botánica de {listing}. "
            f"Menciona la familia y, si puedes, otras plantas relacionadas que conozcas."
        ),
        "category": (
            f"El usuario quiere saber a qué categoría o tipo pertenece {listing}. "
            f"Explica las categorías en primera persona, con ejemplos de qué significa cada una."
        ),
        "uses": (
            f"El usuario pregunta para qué sirve {listing}. "
            f"Describe los usos que conoces de forma ordenada y en primera persona."
        ),
        "warnings": (
            f"El usuario pregunta por advertencias de {listing}. "
            f"Lista las advertencias que conoces con énfasis en seguridad, hablando como experta."
        ),
        "definition": (
            f"El usuario quiere saber qué es {listing}. "
            f"Da una descripción completa en primera persona: qué es, cómo es, para qué sirve. "
            f"Sé natural, no estructures la respuesta como una ficha técnica."
        ),
        "comparison": (
            f"El usuario quiere comparar {listing}. "
            f"Haz una comparación clara por categorías: toxicidad, usos, familia, altura y advertencias. "
            f"Habla como si estuvieras explicándole a alguien de confianza."
        ),
        "general": (
            f"El usuario hace una pregunta general sobre {listing}. "
            f"Responde con todo lo que sabes, en primera persona y de forma conversacional."
        ),
    }
    return instructions.get(intent, instructions["general"])


# ---------------------------------------------------------------------------
# Llamada a Ollama con manejo de errores
# ---------------------------------------------------------------------------
def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "No pude conectarme a Ollama. "
            "Verifica que el servicio esté corriendo en localhost:11434."
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(
            "Ollama tardó demasiado en responder (timeout 300s)."
        )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama respondió con error {response.status_code}: {response.text}"
        )

    data = response.json()
    answer = data.get("response", "").strip()

    if not answer:
        raise RuntimeError("Ollama devolvió una respuesta vacía.")

    return answer


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def run_pipeline(question: str) -> dict:
    plants, context = retrieve_context(question)
    intent = detect_intent(question)
    prompt = build_prompt(question, plants, context, intent)
    answer = ask_llm(prompt)

    return {
        "answer": answer,
        "plants_found": [p["nombre"] for p in plants],
        "intent": intent,
        "context_used": bool(context),
    }