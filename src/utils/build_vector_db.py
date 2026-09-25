from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "plant_disease_guide.txt"
)

VECTOR_DB_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_db"
)


# ---------------------------------------------------------
# LOAD KNOWLEDGE DOCUMENT
# ---------------------------------------------------------

with open(
    KNOWLEDGE_BASE,
    "r",
    encoding="utf-8"
) as file:

    text = file.read()


# ---------------------------------------------------------
# SPLIT DOCUMENT INTO CHUNKS
# ---------------------------------------------------------

chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print(
    f"Loaded {len(chunks)} knowledge chunks."
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# CREATE CHROMA DATABASE
# ---------------------------------------------------------

VECTOR_DB_PATH.mkdir(
    parents=True,
    exist_ok=True
)


client = chromadb.PersistentClient(
    path=str(VECTOR_DB_PATH)
)


collection = client.get_or_create_collection(
    name="agriculture_knowledge"
)


# ---------------------------------------------------------
# CREATE EMBEDDINGS
# ---------------------------------------------------------

embeddings = embedding_model.encode(
    chunks
).tolist()


# ---------------------------------------------------------
# STORE DOCUMENTS
# ---------------------------------------------------------

ids = [
    f"chunk_{index}"
    for index in range(len(chunks))
]


collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings
)


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------

print(
    "Vector database created successfully."
)

print(
    f"Stored {len(chunks)} chunks."
)

print(
    f"Database location: {VECTOR_DB_PATH}"
)
