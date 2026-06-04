from sentence_transformers import SentenceTransformer
import chromadb

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    "plants"
)

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-small"
)

def search(question):

    embedding = embedding_model.encode(
        question
    ).tolist()

    result = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    return result["documents"][0]