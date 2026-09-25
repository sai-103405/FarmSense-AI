from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# CONNECT TO CHROMADB
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)


collection = client.get_collection(
    name="agriculture_knowledge"
)


# ---------------------------------------------------------
# RETRIEVAL FUNCTION
# ---------------------------------------------------------

def retrieve_knowledge(
    query,
    top_k=3
):
    """
    Retrieve the most relevant agricultural
    knowledge chunks for a user query.
    """

    query_embedding = embedding_model.encode(
        [query]
    ).tolist()


    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )


    documents = results.get(
        "documents",
        [[]]
    )[0]


    distances = results.get(
        "distances",
        [[]]
    )[0]


    retrieved_chunks = []

    for document, distance in zip(
        documents,
        distances
    ):

        retrieved_chunks.append({
            "document": document,
            "distance": distance
        })


    return retrieved_chunks


# ---------------------------------------------------------
# TEST THE RETRIEVER
# ---------------------------------------------------------

if __name__ == "__main__":

    query = input(
        "\nEnter your agriculture question: "
    ).strip()


    results = retrieve_knowledge(
        query,
        top_k=3
    )


    print(
        "\n================================"
    )

    print(
        "RETRIEVED AGRICULTURAL KNOWLEDGE"
    )

    print(
        "================================"
    )


    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- Result {index} ---"
        )

        print(
            result["document"]
        )

        print(
            f"\nDistance: "
            f"{result['distance']:.4f}"
        )
