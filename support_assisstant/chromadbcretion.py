from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

import warnings
warnings.filterwarnings("ignore")
BASE_DIR=Path(__file__).parent
DOCS_DIR =Path(f"{Path(__file__).parent}\\Documents")
print(DOCS_DIR)
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "zepto_policies"

def load_documents():
    documents = []
    
    for file_path in sorted(DOCS_DIR.glob("*.txt")):
        # print(file_path)
        # text = file_path.read_text(encoding="utf-8").strip()

        text = file_path.read_text(encoding="cp1252").strip()
        if text:
            documents.append(
            {
                "id": file_path.stem,
                "text": text,
            }
        )

    return documents

def build_vector_store():
    documents = load_documents()

    if len(documents) != 8:
        raise ValueError(
            f"Expected 8 documents, but found {len(documents)}."
        )

    print(f"Loaded {len(documents)} documents.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in documents]
    ids = [doc["id"] for doc in documents]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    ).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Clear existing records so ingestion is deterministic.
    existing = collection.get()

    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {"source": doc["id"]}
            for doc in documents
        ]
    )

    print(f"Stored {len(ids)} documents in ChromaDB.")
    print(f"Collection: {COLLECTION_NAME}")

build_vector_store()