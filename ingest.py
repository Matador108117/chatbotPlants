from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

COLLECTION_NAME = "plants"

try:
    client.delete_collection(
        COLLECTION_NAME
    )
except:
    pass

collection = (
    client.get_or_create_collection(
        name=COLLECTION_NAME
    )
)

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

chunk_dir = Path(
    "data/chunks"
)

files = list(
    chunk_dir.glob("*.txt")
)

print(
    f"Chunks encontrados: {len(files)}"
)

for file in files:

    content = file.read_text(
        encoding="utf-8"
    )

    plant = "_".join(
        file.stem.split("_")[:-1]
    )

    embedding = model.encode(
        content,
        normalize_embeddings=True
    ).tolist()

    collection.add(
        ids=[file.stem],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{
            "plant": plant,
            "source": file.name
        }]
    )

    print(
        f"Indexado: {file.name}"
    )

print(
    "Embeddings generados"
)