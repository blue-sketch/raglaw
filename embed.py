import json
import os
import shutil

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "processed_chunks.json"

CHROMA_PATH = "./chroma_db"

COLLECTION_NAME = "constitution_of_india"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

print("Loading chunks...")

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")
print(f"Model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print("Embedding model loaded.")


# ============================================================
# PREPARE DATA
# ============================================================

ids = []
texts = []
metadatas = []

for i, chunk in enumerate(chunks):

    text = chunk.get("text", "").strip()

    if not text:
        continue

    # Metadata is nested inside chunk["metadata"]
    metadata = chunk.get("metadata", {})

    # IMPORTANT:
    # Do NOT use chunk["chunk_id"] as the Chroma ID.
    # Some chunk_ids are duplicated.
    chroma_id = f"chunk_{i}"

    ids.append(chroma_id)

    texts.append(text)

    metadatas.append({
        "document": str(
            metadata.get("document", "Constitution of India")
        ),
        "article": str(
            metadata.get("article", "")
        ),
        "article_title": str(
            metadata.get("article_title", "")
        ),
        "chunk_type": str(
            metadata.get("chunk_type", "")
        )
    })


print(f"Texts to embed: {len(texts)}")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("\nEmbeddings generated.")

print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {embeddings.shape[1]}")


# ============================================================
# DELETE OLD CHROMA DATABASE
# ============================================================

print("\nRemoving old ChromaDB...")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

print("Old ChromaDB removed.")


# ============================================================
# CREATE NEW CHROMADB
# ============================================================

print("\nCreating new ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={
        "hnsw:space": "cosine"
    }
)


# ============================================================
# STORE DATA
# ============================================================

print("\nAdding embeddings to ChromaDB...")

collection.add(
    ids=ids,
    embeddings=embeddings.tolist(),
    documents=texts,
    metadatas=metadatas
)


# ============================================================
# VERIFY STORED DATA
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING STORED DATA")
print("=" * 70)

test = collection.get(
    limit=5,
    include=[
        "documents",
        "metadatas"
    ]
)

for i in range(len(test["documents"])):

    print("\n" + "-" * 70)

    print("TEXT:")
    print(test["documents"][i])

    print("\nMETADATA:")
    print(test["metadatas"][i])


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("EMBEDDING COMPLETE")
print("=" * 70)

print(f"Chunks stored       : {collection.count()}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Database location   : {CHROMA_PATH}")
print(f"Collection          : {COLLECTION_NAME}")