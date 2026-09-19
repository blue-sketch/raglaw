## 🗄️ ChromaDB

The ChromaDB vector database is **not included in the repository**.

It is generated locally from the processed legal documents and is excluded using `.gitignore`.

To recreate the vector database after cloning the repository:

```bash
python embed.py
```

This will:

1. Load `processed_chunks.json`
2. Generate embeddings using `all-MiniLM-L6-v2`
3. Create the ChromaDB database
4. Store the embeddings, document text, and metadata

The resulting directory will be:

```text
chroma_db/
```

This directory is generated locally and should not be committed to GitHub.

### Data Pipeline

```text
Constitution PDF
       ↓
   clean.txt
       ↓
semantic_chunks.json
       ↓
processed_chunks.json
       ↓
    embed.py
       ↓
   Embeddings
       ↓
    ChromaDB
```

Because ChromaDB is generated from the processed source data, the repository contains the **pipeline required to recreate the vector database**, rather than committing the generated database itself.
