import json

INPUT_FILE = "processed_chunks.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

if isinstance(chunks, dict):
    chunks = chunks["chunks"]

# Find chunks with 20 characters or less
small_chunks = [
    chunk for chunk in chunks
    if len(chunk.get("text", "").strip()) <= 20
]

print(f"Total small chunks (<=20 chars): {len(small_chunks)}")

print("\n" + "=" * 70)
print("SMALL CHUNKS")
print("=" * 70)

for i, chunk in enumerate(small_chunks, 1):

    print(f"\n--- #{i} | {len(chunk['text'].strip())} chars ---")
    print(repr(chunk["text"]))

    print("Metadata:")
    print(f"  Article      : {chunk.get('article', '')}")
    print(f"  Article Title: {chunk.get('article_title', '')}")
    print(f"  Chunk Type   : {chunk.get('chunk_type', '')}")