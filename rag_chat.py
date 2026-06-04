import json
import requests
import chromadb

from sentence_transformers import SentenceTransformer
from structured_search import PlantStructuredSearch

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "plants"
)

structured = PlantStructuredSearch(
    "plants.json"
)


def retrieve_context(question: str) -> tuple:
    plants = structured.find_plant(question)

    embedding = embedding_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    contexts = []

    if plants:
        for plant in plants:
            plant_id = (
                plant["nombre"]
                .lower()
                .replace(" ", "_")
            )

            try:
                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=3,
                    where={"plant": plant_id}
                )
                contexts.extend(results["documents"][0])
            except Exception:
                pass
    else:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )
        contexts.extend(results["documents"][0])

    context = "\n\n".join(contexts)

    return plants, context


def build_prompt(
    question: str,
    plants: list,
    context: str
) -> str:
    structured_data = ""

    for plant in plants:
        structured_data += (
            json.dumps(
                plant,
                ensure_ascii=False,
                indent=2
            )
            + "\n\n"
        )

    return f"""
Eres Aloe.

Especialista en botánica.

Reglas:

1. Usa primero los datos estructurados.
2. Usa después el contexto recuperado.
3. Puedes realizar inferencias simples.
4. No inventes información.
5. Si no existe información suficiente responde exactamente:

No encontré información suficiente.

DATOS ESTRUCTURADOS:

{structured_data}

CONTEXTO:

{context}

PREGUNTA:

{question}

RESPUESTA:
"""


def ask_llm(prompt: str) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=300
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Ollama Error: {response.text}"
        )

    return response.json()["response"]