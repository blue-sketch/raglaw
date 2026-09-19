import chromadb
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURATION
# ==============================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "constitution_of_india"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)
print("Embedding model loaded.")


# ==============================
# CONNECT TO CHROMADB
# ==============================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Collection loaded.")
print(f"Total documents: {collection.count()}")


# ==============================
# GET USER QUERY
# ==============================

query = input("\nEnter your question: ").strip()

if not query:
    print("No question entered.")
    exit()


# ==============================
# CREATE QUERY EMBEDDING
# ==============================

print("\nCreating query embedding...")

query_embedding = model.encode(
    query,
    normalize_embeddings=True
)

print("Query embedding created.")


# ==============================
# SEARCH CHROMADB
# ==============================

print("\nSearching vector database...")

results = collection.query(
    query_embeddings=[
        query_embedding.tolist()
    ],
    n_results=TOP_K,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)


# ==============================
# DISPLAY RESULTS
# ==============================

print("\n" + "=" * 80)
print("RETRIEVAL RESULTS")
print("=" * 80)

documents = results["documents"][0]
metadatas = results["metadatas"][0]
distances = results["distances"][0]

for i in range(len(documents)):

    print("\n" + "-" * 80)
    print(f"RESULT #{i + 1}")
    print("-" * 80)

    metadata = metadatas[i]

    print(f"Article       : {metadata.get('article', 'N/A')}")
    print(f"Article Title : {metadata.get('article_title', 'N/A')}")
    print(f"Chunk Type    : {metadata.get('chunk_type', 'N/A')}")
    print(f"Distance      : {distances[i]:.4f}")

    print("\nTEXT:")
    print(documents[i])


print("\n" + "=" * 80)
print("SEARCH COMPLETE")
print("=" * 80)