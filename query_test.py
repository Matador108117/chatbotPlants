from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "plants"
)

while True:

    question = input(
        "\nPregunta: "
    )

    if question.lower() == "salir":
        break

    embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    print(
        "\nRESULTADOS:"
    )

    for idx, (
        doc,
        meta
    ) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0]
        ),
        start=1
    ):

        print("\n")
        print("=" * 80)

        print(
            f"Resultado {idx}"
        )

        print(
            f"Planta: {meta['plant']}"
        )

        print(
            f"Chunk: {meta['source']}"
        )

        print("=" * 80)

        print(
            doc[:1000]
        )