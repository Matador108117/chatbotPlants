
FROM python:3.14-slim
 
WORKDIR /app
 
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Pre-descargar modelo de embeddings en la imagen
# (embeddings siguen siendo locales — solo el LLM es Groq)
RUN python -c "from sentence_transformers import SentenceTransformer; \
               SentenceTransformer('intfloat/multilingual-e5-small')"
 
COPY structured_search.py intent_detector.py rag_chat.py main.py ./
 
# plants.json y chroma_db se montan como volúmenes en producción
# COPY plants.json .
# COPY chroma_db/ ./chroma_db/
 
EXPOSE 7860
 
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1
 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
 