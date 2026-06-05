import json
import os
import re
import chromadb
from dotenv import load_dotenv

load_dotenv()

from groq import Groq
from sentence_transformers import SentenceTransformer
from structured_search import PlantStructuredSearch
from intent_detector import detect_intent

# ---------------------------------------------------------------------------
# Configuración — sobreescribible por variables de entorno
# ---------------------------------------------------------------------------
GROQ_MODEL   = os.environ.get("GROQ_MODEL",   "llama-3.1-8b-instant")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")   # requerido
CHROMA_PATH  = os.environ.get("CHROMA_PATH",  "./chroma_db")
PLANTS_JSON  = os.environ.get("PLANTS_JSON",  "plants.json")
COLLECTION_NAME = "plants"

if not GROQ_API_KEY:
    raise EnvironmentError(
        "Falta la variable de entorno GROQ_API_KEY. "
        "Agrégala en tu .env o en los secretos del deployment."
    )

# ---------------------------------------------------------------------------
# Inicialización (una sola vez al importar)
# ---------------------------------------------------------------------------
groq_client     = Groq(api_key=GROQ_API_KEY)
embedding_model = SentenceTransformer("intfloat/multilingual-e5-small")
chroma_client   = chromadb.PersistentClient(path=CHROMA_PATH)
collection      = chroma_client.get_collection(COLLECTION_NAME)
structured      = PlantStructuredSearch(PLANTS_JSON)


# ---------------------------------------------------------------------------
# Recuperación de contexto híbrido JSON + Chroma
# (sin cambios — los embeddings locales siguen funcionando igual)
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
# Identidad de Aloe — ahora va en el rol "system" de la API de Groq
# Esto es clave: Groq/OpenAI-compatible APIs distinguen system vs user,
# lo que hace que el modelo respete la identidad de forma mucho más
# consistente que embebiendo todo en un único bloque de texto.
# ---------------------------------------------------------------------------
_SYSTEM_IDENTITY = """\
Tu nombre es Aloe.
IMPORTANTE:
- Siempre hablas como Aloe.
- Nunca digas "soy botánica".
- Nunca digas "como botánica".
- Nunca digas "soy experta".
- Nunca digas "como experta".
- Nunca digas "como profesional de la salud".
- Nunca adoptes una identidad humana.
Tu forma de hablar:
- Hablas siempre desde la perpectiva de Aole. Nunca digas "el asistente" ni "según la información". \
-Dilo como si fuera tu conocimiento directo, en primera persona.\
- No menciones tu especializacion o grado ni el hecho de que eres un modelo de lenguaje,
- Tu tono es cálido, directo y didactico, con un toque familiar o amigable.
- Cuando sí tienes datos: comparte todo lo que sabes con detalle y naturalidad.
- Cuando NO tienes datos de una planta: lo dices honestamente en primera persona. \
Ejemplo: "Honestamente, esa planta no está en mi conocimiento. Prefiero no inventar nada."
- NUNCA inventes nombres científicos, familias, usos ni propiedades que no estén \
en los DATOS o el CONTEXTO que se te proporciona. Si no está ahí, no lo sabes.
- No menciones términos como "contexto recuperado", "datos estructurados" ni "base de datos". \
Eso es interno. Habla como si el conocimiento fuera tuyo.\
"""


def build_messages(
    question: str,
    plants: list[dict],
    context: str,
    intent: str,
) -> list[dict]:
    """
    Construye la lista de mensajes en formato OpenAI/Groq:
      [ {"role": "system", ...}, {"role": "user", ...} ]

    El system prompt lleva la identidad de Aloe.
    El user message lleva los datos + la pregunta real.
    Esta separación es la que hace que el modelo mantenga
    la identidad de forma consistente.
    """
    structured_block = ""
    for plant in plants:
        structured_block += json.dumps(plant, ensure_ascii=False, indent=2) + "\n\n"
    structured_block = structured_block.strip()

    intent_instruction = _intent_instruction(intent, plants)
    has_data = bool(structured_block or context)
    context_block = context if context else "Sin contexto disponible."

    data_warning = (
        ""
        if has_data
        else (
            "\n⚠️ ADVERTENCIA: No hay datos sobre esta planta en tu conocimiento. "
            "Debes decirle al usuario honestamente que no tienes información sobre ella. "
            "No inventes nada.\n"
        )
    )

    user_content = f"""\
INSTRUCCIÓN PARA ESTA RESPUESTA:
{intent_instruction}
{data_warning}
DATOS QUE CONOCES (fuente principal):
{structured_block if structured_block else "— ninguno para esta consulta —"}

CONTEXTO ADICIONAL (complementa si aporta algo nuevo):
{context_block}

PREGUNTA DEL USUARIO:
{question}

TU RESPUESTA (en primera persona, en español):"""

    return [
        {"role": "system",  "content": _SYSTEM_IDENTITY},
        {"role": "user",    "content": user_content},
    ]


def _intent_instruction(intent: str, plants: list[dict]) -> str:
    names   = [p["nombre"] for p in plants]
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
# Llamada a Groq
# ---------------------------------------------------------------------------

# Patrón para eliminar bloques <think>...</think> que escape al filtro de la API.
# Cubre variantes multilinea y espacios irregulares.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    """Elimina cualquier bloque <think>...</think> y normaliza espacios."""
    return _THINK_RE.sub("", text).strip()


def ask_llm(messages: list[dict]) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
            # Desactiva el thinking mode de Qwen3 a nivel de API.
            # Groq expone este parámetro como extra_body para modelos compatibles.
        )
    except Exception as exc:
        raise RuntimeError(f"Error al llamar a Groq: {exc}") from exc

    raw = response.choices[0].message.content or ""

    # Red de seguridad: elimina <think> aunque la API lo filtre parcialmente
    answer = _strip_thinking(raw)

    if not answer:
        raise RuntimeError("Groq devolvió una respuesta vacía.")

    return answer


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------
def run_pipeline(question: str) -> dict:
    plants, context = retrieve_context(question)
    intent          = detect_intent(question)
    messages        = build_messages(question, plants, context, intent)
    answer          = ask_llm(messages)

    return {
        "answer":       answer,
        "plants_found": [p["nombre"] for p in plants],
        "intent":       intent,
        "context_used": bool(context),
    }