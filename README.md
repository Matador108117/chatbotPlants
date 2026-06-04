# Chatbot con RAG y Ollama

Este es un proyecto de chatbot que utiliza Retrieval-Augmented Generation (RAG) combinando **ChromaDB** para la búsqueda vectorial y **Ollama** con el modelo **qwen2.5:3b** para la generación de respuestas. El proyecto también expone una API con FastAPI.

## Requisitos Previos

Antes de comenzar, debes tener instalado lo siguiente:

1. **Python 3.8+**
2. **Ollama**: Necesario para ejecutar el modelo de lenguaje de forma local. Puedes descargarlo e instalarlo desde [ollama.com](https://ollama.com/).
3. **Dependencias del sistema**: Es posible que requieras herramientas básicas de compilación para algunas librerías de Python (`build-essential`).

## Configuración de Ollama y el Modelo

1. Asegúrate de que el servicio de Ollama esté ejecutándose en tu máquina (por defecto en `http://localhost:11434`).
2. Descarga el modelo requerido (**qwen2.5:3b**) ejecutando en tu terminal:
   ```bash
   ollama pull qwen2.5:3b
   ```
   *(Nota: Puedes verificar que se instaló correctamente ejecutando `ollama run qwen2.5:3b`)*.

## Instalación y Configuración del Repositorio

1. **Clonar este repositorio:**
   ```bash
   git clone <URL_DEL_REPOSITORIO>
   cd chatbot
   ```

2. **Crear y activar un entorno virtual (Recomendado):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   # venv\Scripts\activate   # En Windows
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   *Nota: El proyecto emplea FastAPI y Uvicorn para la API, así como la librería `requests`. Si notas que falta alguna de estas en el `requirements.txt`, instálalas manualmente:*
   ```bash
   pip install fastapi uvicorn requests
   ```

## Uso del Proyecto

### 1. Ingesta de Datos (Vectorización)

Primero, necesitas procesar tus documentos y guardarlos en la base de datos vectorial (ChromaDB). Los textos fragmentados se tomarán del directorio `data/chunks/`.

Ejecuta el script de ingesta:
```bash
python ingest.py
```
Esto descargará el modelo de embeddings (`intfloat/multilingual-e5-small`) si no lo tienes, y almacenará los vectores en `/chroma_db/`.

### 2. Iniciar el Servidor API

El componente principal está definido en `main.py` empleando FastAPI. Para levantar el servidor en desarrollo, ejecuta:

```bash
uvicorn main:app --reload
```

El servidor estará escuchando en `http://127.0.0.1:8000`.

### 3. Realizar Consultas al Chatbot

Puedes probar tu chatbot haciendo una petición POST al endpoint `/chat`. Por ejemplo, usando `curl`:

```bash
curl -X POST http://127.0.0.1:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "¿Para qué sirve la albahaca?"}'
```

La aplicación buscará el contexto en ChromaDB y formará un prompt enriquecido para enviarlo a **Ollama (qwen2.5:3b)**. Mismo que te devolverá una respuesta fundamentada en tus documentos locales.

## Estructura Principal del Repositorio

- `main.py`: Archivo de entrada de FastAPI que expone el endpoint `/chat`.
- `ingest.py`: Script responsable de vectorizar los textos ubicados en `data/chunks/` guardándolos en ChromaDB.
- `chat.py / rag_chat.py`: Utilidades base para la búsqueda (Search) en ChromaDB.
- `chroma_db/`: Directorio donde se almacena la base de datos de índices vectoriales (Se genera/modifica con `ingest.py`).
- `data/`: Información en texto y json procesada de las plantas para la base de conocimiento vectorial.