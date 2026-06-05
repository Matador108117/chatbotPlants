from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
 
from rag_chat import run_pipeline
 
app = FastAPI(
    title="Aloe - Chatbot Botánico",
    description="RAG híbrido sobre plantas medicinales y tóxicas.",
    version="1.0.0",
)
 
 
# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Pregunta del usuario sobre plantas.",
    )
 
 
class ChatResponse(BaseModel):
    answer: str = Field(description="Respuesta en lenguaje natural de Aloe.")
    plants_found: list[str] = Field(description="Nombres de plantas detectadas en la pregunta.")
    intent: str = Field(description="Intención detectada (toxicity, scientific, comparison, etc.).")
    context_used: bool = Field(description="Indica si se encontró contexto en Chroma.")
 
 
# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Endpoint de salud para Docker healthcheck."""
    return {"status": "ok"}
 
 
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Recibe una pregunta y devuelve la respuesta de Aloe
    junto con metadatos de diagnóstico.
    """
    try:
        result = run_pipeline(request.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno del pipeline: {exc}",
        )
 
    return ChatResponse(**result)
 