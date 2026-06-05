import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

CHROMA_PATH = "./chroma_db"
CHUNKS_DIR = Path("data/chunks")
COLLECTION_NAME = "plants"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


def main() -> None:
    # Validar directorio de chunks
    if not CHUNKS_DIR.exists():
        print(f"[ERROR] No existe el directorio de chunks: {CHUNKS_DIR}")
        sys.exit(1)

    files = sorted(CHUNKS_DIR.glob("*.txt"))
    if not files:
        print(f"[ERROR] No se encontraron archivos .txt en {CHUNKS_DIR}")
        sys.exit(1)

    print(f"[INFO] Chunks encontrados: {len(files)}")

    # Inicializar Chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Eliminar colección anterior si existe (re-indexado limpio)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"[INFO] Colección '{COLLECTION_NAME}' eliminada para re-indexar.")
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Cargar modelo de embeddings
    print(f"[INFO] Cargando modelo: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Indexar chunks
    errors = 0
    for file in files:
        try:
            content = file.read_text(encoding="utf-8").strip()
            if not content:
                print(f"[WARN] Archivo vacío, omitido: {file.name}")
                continue

            # Convención de nombre: nombreplanta_chunk_N.txt
            # plant_id = todo excepto el último segmento (_N)
            plant_id = "_".join(file.stem.split("_")[:-1])

            embedding = model.encode(
                content,
                normalize_embeddings=True,
            ).tolist()

            collection.add(
                ids=[file.stem],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{
                    "plant": plant_id,
                    "source": file.name,
                }],
            )
            print(f"[OK]   Indexado: {file.name}  (planta: {plant_id})")

        except Exception as exc:
            print(f"[ERROR] No se pudo indexar {file.name}: {exc}")
            errors += 1

    print(f"\n[INFO] Indexación completa. Éxitos: {len(files) - errors}  Errores: {errors}")


if __name__ == "__main__":
    main()