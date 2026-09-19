# 🇮🇳 Indian Legal RAG

An AI-powered **Retrieval-Augmented Generation (RAG)** system for querying Indian legal and constitutional documents using semantic search, vector databases, and Gemini.

The project currently uses the **Constitution of India** as its primary knowledge source and is being designed to expand into a multi-source legal knowledge system covering **Acts, Rules, Regulations, Notifications, Orders, Circulars, and Judicial Decisions**.

---

## 📌 Overview

Large Language Models can answer questions about law, but relying only on an LLM's internal knowledge can result in outdated, incomplete, or unsupported answers.

This project takes a different approach.

Instead of asking Gemini to answer directly, the system first searches a collection of legal documents and retrieves relevant passages. Those passages are then provided to Gemini as context.

```text
User Question
      ↓
Query Embedding
      ↓
Vector Search
      ↓
Relevant Legal Passages
      ↓
Context
      ↓
Gemini
      ↓
Grounded Answer
```

The goal is to make the system's answers **traceable to the retrieved legal sources**.

---

# 🎯 Project Goals

The project aims to build a legal RAG system that can:

* Search Indian constitutional and legal documents semantically
* Retrieve relevant Articles, Sections, Rules, and other provisions
* Provide retrieved source information alongside answers
* Use Gemini for natural-language answer generation
* Reduce unsupported answers by grounding responses in retrieved context
* Preserve legal document metadata and provenance
* Support multiple categories of legal sources
* Evaluate retrieval quality separately from LLM generation quality

---

# 🏗️ Current Architecture

The current system is based on the following pipeline:

```text
                    USER QUESTION
                         │
                         ▼
                  Streamlit UI
                         │
                         ▼
                Query Embedding
                         │
                         ▼
              all-MiniLM-L6-v2
                         │
                         ▼
                    ChromaDB
                         │
                    Top-K Search
                         │
                         ▼
               Retrieved Chunks
                         │
                         ▼
                 Context Builder
                         │
                         ▼
                    Gemini API
                         │
                         ▼
                   Final Answer
```

---

# 📚 Current Knowledge Base

The current version uses the **Constitution of India** as the primary source.

The processing pipeline is:

```text
Constitution PDF
       ↓
PDF Text Extraction
       ↓
clean.txt
       ↓
Semantic Chunking
       ↓
semantic_chunks.json
       ↓
Chunk Processing
       ↓
processed_chunks.json
       ↓
Embedding Generation
       ↓
ChromaDB
```

---

# 📊 Current RAG Statistics

The current Constitution dataset contains approximately:

| Component            |         Current Value |
| -------------------- | --------------------: |
| Source               | Constitution of India |
| Processed chunks     |                 2,363 |
| Embedding model      |    `all-MiniLM-L6-v2` |
| Embedding dimensions |                   384 |
| Vector database      |              ChromaDB |
| Retrieval method     |     Cosine similarity |
| Current Top-K        |                     5 |
| LLM                  |                Gemini |
| UI                   |             Streamlit |

---

# 🧩 Project Components

## 1. PDF Text Extraction

The Constitution PDF is converted into text using **PyMuPDF**.

```text
Constitution PDF
       ↓
     PyMuPDF
       ↓
    clean.txt
```

The extracted text preserves useful line structure for subsequent processing.

---

## 2. Semantic Chunking

The extracted text is divided into semantically meaningful chunks using LangChain's `SemanticChunker`.

```text
clean.txt
    ↓
SemanticChunker
    ↓
semantic_chunks.json
```

The project does not force every chunk to have the same number of characters.

Semantic chunking produces variable-sized chunks because boundaries are determined by changes in semantic content.

This is useful for legal documents where a provision may be naturally short or long.

---

## 3. Chunk Processing

The semantic chunks are processed before embedding.

Each chunk contains:

* Text
* Document metadata
* Article information where available
* Chunk type

Example:

```json
{
  "chunk_id": "article_1_0",
  "text": "1. Name and territory of the Union.",
  "metadata": {
    "document": "Constitution of India",
    "article": "1",
    "article_title": "Name and territory of the Union.",
    "chunk_type": "article"
  }
}
```

---

# 🧠 Embedding Model

The current embedding model is:

```text
all-MiniLM-L6-v2
```

Each legal chunk is converted into a **384-dimensional vector**.

```text
Legal Text
    ↓
all-MiniLM-L6-v2
    ↓
384-dimensional vector
```

When a user submits a query, the same model converts the query into a vector.

The query vector is then compared against the vectors stored in ChromaDB.

---

# 🗄️ Vector Database

The project uses **ChromaDB** for vector storage and similarity search.

Current collection:

```text
constitution_of_india
```

Local database:

```text
./chroma_db
```

Each stored record contains:

```text
ID
Text
Embedding
Metadata
```

---

## Why ChromaDB Is Not Included in the Repository

The `chroma_db/` directory is a generated artifact.

It is intentionally **not pushed to GitHub**.

Instead, the repository contains the data-processing and embedding pipeline required to recreate it.

```text
processed_chunks.json
        ↓
     embed.py
        ↓
    Embeddings
        ↓
     ChromaDB
```

After cloning the repository, the vector database can be regenerated locally by running:

```bash
python embed.py
```

The resulting directory:

```text
chroma_db/
```

will be created locally.

---

# 🔎 Retrieval

When the user asks a question, the query follows this process:

```text
User Question
      ↓
Query Embedding
      ↓
ChromaDB Similarity Search
      ↓
Top-K Results
```

The current system retrieves the top **5** relevant chunks.

Each result contains:

* Article
* Article title
* Chunk type
* Distance
* Original retrieved text

Example:

```text
RESULT #1

Article       : 21
Article Title : Protection of life and personal liberty.
Chunk Type    : article
Distance      : 0.3479

TEXT:
21. Protection of life and personal liberty...
```

The retrieved passages are then assembled into the context provided to Gemini.

---

# 🤖 Gemini Integration

Gemini is used for the **generation stage** of the RAG pipeline.

The system does not send the entire Constitution to Gemini.

Instead:

```text
User Question
      +
Retrieved Chunks
      ↓
Gemini
      ↓
Answer
```

Gemini is instructed to:

* Use the retrieved context
* Mention relevant Article numbers when available
* Avoid inventing constitutional provisions
* State when the retrieved context is insufficient
* Explain provisions in understandable language

---

# 🖥️ Streamlit Interface

The project includes a Streamlit interface for interacting with the RAG system.

The interface provides:

### Question Input

Users can ask questions such as:

```text
What does Article 21 protect?
```

### Retrieved Context

The application displays the chunks retrieved from ChromaDB.

### Metadata

The application displays information such as:

```text
Article
Article Title
Chunk Type
Distance
```

### Gemini Answer

The retrieved context is sent to Gemini and the generated answer is displayed.

### RAG Details

The interface also exposes information about the RAG process, including:

```text
Embedding Model
Vector Database
Top-K
Number of Retrieved Chunks
Gemini Model
Context Size
```

The exact context sent to Gemini can also be inspected.

---

# 🔐 API Key

The Gemini API key should **not** be hardcoded into the source code.

Set it as an environment variable.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Then run:

```bash
streamlit run app.py
```

For production deployments, use the deployment platform's secret-management system.

---

# 📁 Current Project Structure

```text
legal-rag/
│
├── constitution.pdf
│
├── clean.txt
│
├── semantic_chunks.json
│
├── processed_chunks.json
│
├── embed.py
│
├── query.py
│
├── app.py
│
├── README.md
│
├── .gitignore
│
├── requirements.txt
│
└── chroma_db/              # Generated locally, NOT committed
```

---

# 🚫 `.gitignore`

The repository should exclude generated databases and secrets.

Example:

```gitignore
# Python
__pycache__/
*.py[cod]

# Virtual environments
venv/
.venv/

# ChromaDB
chroma_db/

# Environment variables
.env

# Streamlit secrets
.streamlit/secrets.toml

# Logs
*.log

# IDE
.vscode/
.idea/

# OS files
.DS_Store
Thumbs.db
```

---

# 📦 Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
```

Enter the project directory:

```bash
cd legal-rag
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

The current project uses:

```text
streamlit
chromadb
sentence-transformers
google-genai
langchain
pymupdf
```

These can be stored in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1: Generate the Vector Database

After installing dependencies:

```bash
python embed.py
```

This will:

1. Load `processed_chunks.json`
2. Load `all-MiniLM-L6-v2`
3. Generate embeddings
4. Create the ChromaDB collection
5. Store the embeddings
6. Store the original text
7. Store metadata

The resulting database will be:

```text
./chroma_db
```

---

## Step 2: Test Retrieval

Run:

```bash
python query.py
```

Example:

```text
Enter your question:
What does Article 21 protect?
```

The program will display the retrieved chunks and their metadata.

---

## Step 3: Start the RAG Application

Run:

```bash
streamlit run app.py
```

The Streamlit application will open in the browser.

---

# 🔬 RAG Evaluation

An important part of this project is evaluating the **retrieval system separately from the LLM**.

A poor answer can happen for two different reasons.

### Retrieval Failure

The correct legal provision was not retrieved.

```text
Question
   ↓
ChromaDB
   ↓
❌ Wrong chunks
   ↓
Gemini
   ↓
Poor answer
```

### Generation Failure

The correct provision was retrieved, but Gemini failed to use it correctly.

```text
Question
   ↓
ChromaDB
   ↓
✅ Correct chunks
   ↓
Gemini
   ↓
❌ Unsupported answer
```

Separating these two problems is an important part of improving the RAG system.

---

# 🔭 Multi-Source Legal Knowledge Base

The long-term goal is to move beyond the Constitution.

The planned knowledge base will include multiple categories of Indian legal material.

```text
                    INDIAN LEGAL RAG
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
 Constitution             Acts              Rules
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                     Regulations
                           │
                           ▼
                    Notifications
                           │
                           ▼
                       Orders
                           │
                           ▼
                      Judgments
```

---

# 📚 Planned Sources

## Constitution

Currently implemented.

Potential metadata:

```json
{
  "source_type": "constitution",
  "document": "Constitution of India",
  "article": "21",
  "article_title": "Protection of life and personal liberty"
}
```

---

## Central Acts

Potential sources include:

* Bharatiya Nyaya Sanhita, 2023
* Bharatiya Nagarik Suraksha Sanhita, 2023
* Bharatiya Sakshya Adhiniyam, 2023
* Information Technology Act, 2000
* Consumer Protection Act, 2019
* Companies Act, 2013
* Right to Information Act, 2005
* Digital Personal Data Protection Act, 2023
* Indian Contract Act, 1872
* Copyright Act, 1957

Example metadata:

```json
{
  "source_type": "act",
  "document": "Information Technology Act, 2000",
  "section": "43",
  "year": "2000"
}
```

---

## Rules

Rules associated with Acts and government departments can be added as separate sources.

Example:

```json
{
  "source_type": "rule",
  "document": "Information Technology Rules",
  "rule": "3",
  "year": "2021",
  "parent_act": "Information Technology Act, 2000"
}
```

---

## Regulations

Regulations issued by regulatory authorities can eventually be included.

Potential categories include regulations from bodies such as:

* RBI
* SEBI
* TRAI
* IRDAI
* PFRDA

The exact source and current version of each regulation should be preserved during ingestion.

---

## Notifications, Orders and Circulars

These can provide implementation details associated with Acts and Rules.

Potential metadata:

```json
{
  "source_type": "notification",
  "issuing_authority": "...",
  "document": "...",
  "notification_number": "...",
  "date": "...",
  "parent_act": "..."
}
```

---

## Judicial Decisions

A future version can include:

* Supreme Court judgments
* High Court judgments
* Relevant court orders

Example metadata:

```json
{
  "source_type": "judgment",
  "court": "Supreme Court of India",
  "case_name": "...",
  "judgment_date": "...",
  "citation": "..."
}
```

This allows the system to distinguish between:

```text
Constitutional provision
        ↓
Statutory provision
        ↓
Rules / Regulations
        ↓
Judicial interpretation
```

---

# 🗂️ Planned Multi-Collection Architecture

As more sources are added, the vector database can use separate collections.

```text
ChromaDB
│
├── constitution
│
├── acts
│
├── rules
│
├── regulations
│
└── judgments
```

These collections can still be queried together when a question requires information from multiple legal sources.

For example:

```text
User Question
      ↓
Search Constitution
      +
Search Acts
      +
Search Judgments
      ↓
Combine Results
      ↓
Rerank
      ↓
Gemini
```

The purpose of separate collections is logical organization and source-aware retrieval, not simply creating more databases.

---

# 🏷️ Legal Metadata

Metadata will become increasingly important as more sources are added.

The system should preserve information such as:

```text
Source Type
Document
Year
Article
Section
Rule
Regulation
Chapter
Court
Case Name
Judgment Date
Citation
Parent Act
Issuing Authority
```

This allows the retrieval system to perform more targeted searches.

For example:

```text
Query:
"What does Section 43 of the IT Act say?"

          ↓

Filter / prioritize

source_type = act
document = Information Technology Act
section = 43

          ↓

Retrieve relevant evidence
```

---

# 🚀 Planned Retrieval Architecture

The current system uses basic vector similarity retrieval.

The planned system will evolve toward:

```text
                    USER QUERY
                         │
                         ▼
                  Query Analysis
                         │
                         ▼
                 Source Selection
                         │
                         ▼
                 Vector Retrieval
                         │
                         ▼
                Metadata Filtering
                         │
                         ▼
                    Reranking
                         │
                         ▼
                  Evidence Set
                         │
                         ▼
                      Gemini
                         │
                         ▼
                Answer + Sources
```

Potential improvements include:

* Query rewriting
* Multi-collection retrieval
* Metadata filtering
* Hybrid keyword + vector search
* Reranking
* Legal-structure-aware chunking
* Source provenance
* Citation generation
* Retrieval evaluation
* Answer evaluation

---

# 🧪 Future Evaluation Dataset

A dedicated evaluation set will eventually be created.

Example:

```json
{
  "question": "What does Article 21 protect?",
  "expected_sources": [
    {
      "document": "Constitution of India",
      "article": "21"
    }
  ]
}
```

This can be used to measure:

### Retrieval

* Recall@K
* Precision@K
* MRR
* Hit Rate

### Generation

* Faithfulness
* Context relevance
* Answer relevance
* Citation correctness

This makes it possible to measure whether a change to chunking, embeddings, retrieval, or reranking actually improves the system.

---

# ⚠️ Limitations

This project is an experimental **legal information retrieval and question-answering system**.

It should not be treated as a substitute for professional legal advice.

The quality of the system depends on:

* Source document quality
* Source completeness
* Document version
* Chunking quality
* Metadata quality
* Embedding quality
* Retrieval quality
* LLM generation quality

Indian laws and regulations can be amended, repealed, replaced, or supplemented over time.

Therefore, future versions should preserve:

```text
Document
Version
Effective Date
Publication Date
Source
```

where available.

---

# 📜 Source Strategy

The project intends to prioritize authoritative sources for the corresponding document types.

Examples include:

* Official government legal repositories for legislation and subordinate legislation
* Official court sources for judgments and court documents
* Official regulatory authority sources for regulations and circulars

Third-party legal websites may be useful for research and discovery, but source provenance should be preserved when documents are ingested.

---

# 🛠️ Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| User Interface       | Streamlit                 |
| PDF Processing       | PyMuPDF                   |
| Chunking             | LangChain SemanticChunker |
| Embeddings           | Sentence Transformers     |
| Embedding Model      | `all-MiniLM-L6-v2`        |
| Vector Database      | ChromaDB                  |
| LLM                  | Gemini API                |
| Retrieval            | Vector similarity search  |
| RAG                  | Custom Python pipeline    |

---

# 📈 Development Roadmap

## Phase 1: Constitution RAG

* [x] Constitution PDF extraction
* [x] Text cleaning
* [x] Semantic chunking
* [x] Chunk processing
* [x] Metadata generation
* [x] Embedding generation
* [x] ChromaDB integration
* [x] Retrieval
* [x] Gemini integration
* [x] Streamlit interface

## Phase 2: Retrieval Improvement

* [ ] Improve legal-aware chunking
* [ ] Query evaluation dataset
* [ ] Retrieval metrics
* [ ] Metadata filtering
* [ ] Reranking
* [ ] Better source display
* [ ] Citation/provenance

## Phase 3: Multi-Source Legal RAG

* [ ] Central Acts
* [ ] Rules
* [ ] Regulations
* [ ] Notifications
* [ ] Orders
* [ ] Circulars
* [ ] Supreme Court judgments
* [ ] High Court judgments

## Phase 4: Advanced RAG

* [ ] Multi-collection retrieval
* [ ] Hybrid search
* [ ] Query rewriting
* [ ] Source-aware retrieval
* [ ] Legal hierarchy awareness
* [ ] Retrieval evaluation
* [ ] Answer evaluation
* [ ] Version-aware retrieval

---

# 🌱 Long-Term Vision

The long-term goal is to build a **source-grounded Indian legal RAG system** capable of connecting different layers of legal information.

```text
                 USER QUESTION
                       │
                       ▼
                 Query Analysis
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     Constitution     Acts       Judgments
          │            │            │
          ▼            ▼            ▼
       Articles      Sections   Interpretation
          │            │
          └───────┬────┘
                  ▼
              Rules
                  │
                  ▼
            Regulations
                  │
                  ▼
          Notifications
                  │
                  ▼
              Evidence
                  │
                  ▼
               Gemini
                  │
                  ▼
          Answer + Sources
```

The central idea is simple:

> **Retrieve the relevant law first, then use the LLM to explain it.**

The system should make it possible to inspect not only **what the AI answered**, but also **which legal sources were retrieved and why they were used**.

---

## 📌 Current Status

**Project:** Indian Legal RAG

**Current source:** Constitution of India

**Vector database:** ChromaDB, generated locally

**Embedding model:** `all-MiniLM-L6-v2`

**LLM:** Gemini

**Interface:** Streamlit

**Status:** Constitution RAG functional, multi-source legal RAG under development.
