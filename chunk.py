import re
import json

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "clean.txt"
OUTPUT_FILE = "semantic_chunks.json"


# ============================================================
# 1. LOAD TEXT
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# ============================================================
# 2. CLEAN PAGE MARKERS
# ============================================================

text = re.sub(
    r"^\s*(?:=====)?\s*PAGE\s+\d+\s*(?:=====)?\s*$",
    "",
    text,
    flags=re.IGNORECASE | re.MULTILINE
)


# ============================================================
# 3. NORMALIZE
# ============================================================

text = text.replace("\r\n", "\n")
text = text.replace("\r", "\n")

text = re.sub(
    r"[ \t]+",
    " ",
    text
)

text = re.sub(
    r"\n{3,}",
    "\n\n",
    text
)


# ============================================================
# 4. EMBEDDING MODEL
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# 5. SEMANTIC CHUNKER
# ============================================================

semantic_chunker = SemanticChunker(
    embeddings,

    breakpoint_threshold_type="percentile",

    breakpoint_threshold_amount=90,

    min_chunk_size=150
)


# ============================================================
# 6. ARTICLE DETECTION
# ============================================================

ARTICLE_PATTERN = re.compile(
    r"(?m)^(\d+[A-Z]{0,2})\.\s+(.+)$"
)


matches = list(
    ARTICLE_PATTERN.finditer(text)
)


# ============================================================
# 7. CREATE ARTICLE SECTIONS
# ============================================================

articles = []


for i, match in enumerate(matches):

    article_number = match.group(1)

    article_title = match.group(2).strip()

    start = match.start()

    if i + 1 < len(matches):

        end = matches[i + 1].start()

    else:

        end = len(text)

    article_text = text[
        start:end
    ].strip()


    # --------------------------------------------------------
    # Safety filter
    # --------------------------------------------------------

    number_match = re.match(
        r"(\d+)",
        article_number
    )

    if not number_match:
        continue

    number = int(
        number_match.group(1)
    )

    if number > 395:
        continue


    articles.append({

        "article_number":
            article_number,

        "article_title":
            article_title,

        "text":
            article_text

    })


# ============================================================
# 8. SEMANTICALLY CHUNK EACH ARTICLE
# ============================================================

final_chunks = []


for article in articles:

    article_number = (
        article["article_number"]
    )

    article_title = (
        article["article_title"]
    )

    article_text = (
        article["text"]
    )


    # --------------------------------------------------------
    # If Article is small, keep it whole
    # --------------------------------------------------------

    if len(article_text) < 1200:

        final_chunks.append({

            "chunk_id":
                f"article_{article_number}_0",

            "text":
                article_text,

            "metadata": {

                "document":
                    "Constitution of India",

                "article":
                    article_number,

                "article_title":
                    article_title,

                "chunk_type":
                    "article"
            }

        })

        continue


    # --------------------------------------------------------
    # Semantic splitting
    # --------------------------------------------------------

    semantic_chunks = (
        semantic_chunker.split_text(
            article_text
        )
    )


    # --------------------------------------------------------
    # Store chunks
    # --------------------------------------------------------

    for index, chunk in enumerate(
        semantic_chunks
    ):

        final_chunks.append({

            "chunk_id":
                (
                    f"article_{article_number}"
                    f"_{index}"
                ),

            "text":
                chunk,

            "metadata": {

                "document":
                    "Constitution of India",

                "article":
                    article_number,

                "article_title":
                    article_title,

                "chunk_type":
                    "semantic"
            }

        })


# ============================================================
# 9. SAVE
# ============================================================

output = {

    "document":
        "Constitution of India",

    "total_chunks":
        len(final_chunks),

    "chunks":
        final_chunks

}


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        ensure_ascii=False,
        indent=2
    )


# ============================================================
# 10. STATISTICS
# ============================================================

print("=" * 60)
print("SEMANTIC CHUNKING COMPLETE")
print("=" * 60)

print(
    "Articles detected:",
    len(articles)
)

print(
    "Final chunks:",
    len(final_chunks)
)

print(
    "Output:",
    OUTPUT_FILE
)