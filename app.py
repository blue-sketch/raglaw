import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "constitution_of_india"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Change this if you want to use another Gemini model available
# to your API key.
GEMINI_MODEL = "gemini-2.5-flash"

TOP_K = 5


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Constitution RAG",
    page_icon="🇮🇳",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #777;
        margin-bottom: 30px;
    }

    .rag-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-bottom: 12px;
    }

    .metric-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD CHROMADB
# ============================================================

@st.cache_resource
def load_chroma():

    client = chromadb.PersistentClient(
        path=CHROMA_PATH
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    return client, collection


# ============================================================
# LOAD GEMINI
# ============================================================

@st.cache_resource
def load_gemini():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    client = genai.Client(
        api_key=api_key
    )

    return client


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query, model, collection):

    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )

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

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for i in range(len(documents)):

        retrieved.append(
            {
                "text": documents[i],
                "metadata": metadatas[i],
                "distance": distances[i]
            }
        )

    return retrieved


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(retrieved_documents):

    context_parts = []

    for i, item in enumerate(retrieved_documents):

        metadata = item["metadata"]

        article = metadata.get(
            "article",
            "N/A"
        )

        article_title = metadata.get(
            "article_title",
            "N/A"
        )

        text = item["text"]

        context_parts.append(
            f"""
SOURCE {i + 1}

Article: {article}
Article Title: {article_title}

Text:
{text}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(query, context, gemini_client):

    system_instruction = """
You are a Constitution of India question-answering assistant.

You are part of a Retrieval-Augmented Generation (RAG) system.

IMPORTANT RULES:

1. Answer using the retrieved Constitution context provided to you.
2. Do not invent constitutional provisions.
3. If the retrieved context does not contain enough information,
   clearly say that the retrieved context is insufficient.
4. Prefer the exact constitutional text when explaining provisions.
5. Mention relevant Article numbers when available.
6. Do not pretend that retrieved information exists if it does not.
7. Give a clear explanation that is easy for a student to understand.
"""

    prompt = f"""
{system_instruction}

RETRIEVED CONTEXT
=================

{context}

USER QUESTION
=============

{query}

TASK
====

Answer the user's question using the retrieved context.

Where useful:
- Mention the Article number.
- Explain the provision in simple language.
- Distinguish between the constitutional text and your explanation.

Do not use information that contradicts the retrieved context.
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write("### Embedding Model")

    st.code(
        EMBEDDING_MODEL,
        language="text"
    )

    st.write("### Vector Database")

    st.code(
        "ChromaDB",
        language="text"
    )

    st.write("### Collection")

    st.code(
        COLLECTION_NAME,
        language="text"
    )

    st.write("### Gemini Model")

    st.code(
        GEMINI_MODEL,
        language="text"
    )

    st.write("### Retrieval")

    st.metric(
        "Top-K",
        TOP_K
    )

    st.divider()

    st.write("### RAG Pipeline")

    st.markdown(
        """
        **1. User Question**

        ↓

        **2. Query Embedding**

        ↓

        **3. ChromaDB Search**

        ↓

        **4. Top-K Chunks**

        ↓

        **5. Context Construction**

        ↓

        **6. Gemini**

        ↓

        **7. Final Answer**
        """
    )


# ============================================================
# MAIN UI
# ============================================================

st.markdown(
    '<div class="main-title">🇮🇳 Constitution RAG</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the Constitution of India using
    Retrieval-Augmented Generation.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD COMPONENTS
# ============================================================

try:

    with st.spinner("Loading RAG system..."):

        embedding_model = load_embedding_model()

        chroma_client, collection = load_chroma()

        gemini_client = load_gemini()

except Exception as e:

    st.error(
        f"Failed to load RAG system: {e}"
    )

    st.stop()


# ============================================================
# SYSTEM STATUS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Vector Chunks",
        collection.count()
    )

with col2:

    st.metric(
        "Embedding Size",
        "384"
    )

with col3:

    st.metric(
        "Top-K",
        TOP_K
    )

with col4:

    if gemini_client:

        st.metric(
            "Gemini",
            "Connected"
        )

    else:

        st.metric(
            "Gemini",
            "Not Connected"
        )


st.divider()


# ============================================================
# API KEY WARNING
# ============================================================

if gemini_client is None:

    st.warning(
        """
        Gemini API key not found.

        Set the `GEMINI_API_KEY` environment variable before
        running the application.
        """
    )


# ============================================================
# QUESTION INPUT
# ============================================================

query = st.text_input(
    "Ask a question",
    placeholder="Example: What does Article 21 protect?"
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.write("**Try an example:**")

example_cols = st.columns(4)

examples = [
    "What does Article 21 protect?",
    "What are the Fundamental Rights?",
    "What is the Right to Equality?",
    "What qualifications are required to become President?"
]

for i, example in enumerate(examples):

    with example_cols[i]:

        if st.button(
            example,
            use_container_width=True
        ):

            query = example


# ============================================================
# PROCESS QUERY
# ============================================================

if query:

    if gemini_client is None:

        st.error(
            "Gemini API key is not configured."
        )

        st.stop()


    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    with st.spinner("Searching Constitution..."):

        retrieved_documents = retrieve_documents(
            query,
            embedding_model,
            collection
        )


    # --------------------------------------------------------
    # SHOW RETRIEVAL
    # --------------------------------------------------------

    st.subheader("🔎 Retrieved Context")

    st.caption(
        f"Retrieved {len(retrieved_documents)} chunks from ChromaDB"
    )


    for i, item in enumerate(
        retrieved_documents
    ):

        metadata = item["metadata"]

        article = metadata.get(
            "article",
            "N/A"
        )

        article_title = metadata.get(
            "article_title",
            "N/A"
        )

        distance = item["distance"]

        with st.expander(
            f"Source {i + 1} | Article {article} | Distance {distance:.4f}"
        ):

            st.write(
                f"**Article:** {article}"
            )

            st.write(
                f"**Title:** {article_title}"
            )

            st.write(
                f"**Chunk Type:** "
                f"{metadata.get('chunk_type', 'N/A')}"
            )

            st.write(
                f"**Cosine Distance:** "
                f"{distance:.4f}"
            )

            st.markdown("**Retrieved Text:**")

            st.code(
                item["text"],
                language="text"
            )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_context(
        retrieved_documents
    )


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    with st.spinner("Gemini is generating the answer..."):

        try:

            answer = ask_gemini(
                query,
                context,
                gemini_client
            )

        except Exception as e:

            st.error(
                f"Gemini API error: {e}"
            )

            st.stop()


    # --------------------------------------------------------
    # FINAL ANSWER
    # --------------------------------------------------------

    st.subheader("🤖 Answer")

    st.markdown(answer)


    # --------------------------------------------------------
    # RAG DETAILS
    # --------------------------------------------------------

    st.divider()

    st.subheader("🧠 RAG Details")

    rag_col1, rag_col2 = st.columns(2)

    with rag_col1:

        st.write("### Query")

        st.code(
            query,
            language="text"
        )

        st.write("### Embedding Model")

        st.code(
            EMBEDDING_MODEL,
            language="text"
        )

        st.write("### Vector Database")

        st.code(
            "ChromaDB",
            language="text"
        )


    with rag_col2:

        st.write("### Retrieved Chunks")

        st.metric(
            "Chunks sent to Gemini",
            len(retrieved_documents)
        )

        st.write("### Gemini Model")

        st.code(
            GEMINI_MODEL,
            language="text"
        )

        st.write("### Context Size")

        st.metric(
            "Characters",
            len(context)
        )


    # --------------------------------------------------------
    # RAW CONTEXT
    # --------------------------------------------------------

    with st.expander("📄 View exact context sent to Gemini"):

        st.code(
            context,
            language="text"
        )