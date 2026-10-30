"""
L7 — Evaluate Your Search System  (STARTER)
=============================================
Run with:
    python starter.py

Your goal: build an evaluation framework that measures precision and recall
for a semantic search system across multiple settings.

Required features:
    1. A ChromaDB collection with at least 12 documents across 3–4 topics
    2. An evaluation set: at least 6 test queries, each with a list of
       expected relevant document IDs
    3. An evaluate() function that computes per-query and average
       precision & recall
    4. Results at 3 different settings (vary threshold, n_results, or both)
    5. A written analysis in the output

Key concepts:
    Precision — of the results returned, how many were relevant?
        precision = len(relevant ∩ retrieved) / len(retrieved)

    Recall — of all relevant documents, how many did we retrieve?
        recall = len(relevant ∩ retrieved) / len(relevant)

    ChromaDB returns distances (lower = more similar).
    Convert to similarity for thresholding:
        similarity = 1 - distance   (for cosine distance)
        OR just filter by distance <= threshold

    Example evaluation set entry:
        {
            "query": "How does FastAPI validate request data?",
            "relevant_ids": ["doc_03", "doc_07"]
        }
"""

import chromadb

# ── Documents ─────────────────────────────────────────────────────────────────
# TODO: Define ids, documents, and metadatas lists (at least 12 items)
#       Cover 3–4 topics so that some queries are easy and some are hard.
#       Use unique IDs like "doc_01", "doc_02", ... — these are what the
#       evaluation set references.

ids = [
    "doc_01", "doc_02", "doc_03",
    "doc_04", "doc_05", "doc_06",
    "doc_07", "doc_08", "doc_09",
    "doc_10", "doc_11", "doc_12"
]

documents = [
    # FastAPI
    "FastAPI uses Pydantic models to validate incoming request data.",
    "JWT authentication protects API routes by verifying user tokens.",
    "FastAPI supports dependency injection for reusable authentication and database connections.",

    # Streamlit
    "Streamlit session_state preserves values between application reruns.",
    "Streamlit widgets include buttons, sliders, text inputs, and selectboxes.",
    "Streamlit forms group inputs and submit them together using a form submit button.",

    # Embeddings and semantic search
    "SentenceTransformer converts text into numerical embedding vectors.",
    "Cosine similarity measures how closely two embedding vectors align.",
    "Semantic search retrieves documents based on meaning rather than exact keyword matches.",

    # ChromaDB
    "ChromaDB stores document embeddings in collections for vector similarity search.",
    "ChromaDB metadata allows filtering search results by topic or source.",
    "Chunking divides long documents into smaller sections before embedding and indexing."
]

metadatas = [
    {"topic": "fastapi"},
    {"topic": "fastapi"},
    {"topic": "fastapi"},
    {"topic": "streamlit"},
    {"topic": "streamlit"},
    {"topic": "streamlit"},
    {"topic": "embeddings"},
    {"topic": "embeddings"},
    {"topic": "embeddings"},
    {"topic": "chromadb"},
    {"topic": "chromadb"},
    {"topic": "chromadb"}
]

# ── ChromaDB setup ────────────────────────────────────────────────────────────
# TODO: Create a chromadb.EphemeralClient() (in-memory, no files needed)
# TODO: Create or get a collection called "eval_collection"
# TODO: Upsert all documents

client = chromadb.EphemeralClient()

collection = client.get_or_create_collection(
    name="eval_collection"
)

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)


# ── Evaluation set ────────────────────────────────────────────────────────────
# TODO: Define eval_set — a list of dicts, each with:
#   "query":        the natural language query string
#   "relevant_ids": list of document IDs that are truly relevant

eval_set = [
    {
        "query": "How does FastAPI validate request data?",
        "relevant_ids": ["doc_01"]
    },
    {
        "query": "How can I protect API routes with authentication?",
        "relevant_ids": ["doc_02", "doc_03"]
    },
    {
        "query": "How does Streamlit remember values between reruns?",
        "relevant_ids": ["doc_04"]
    },
    {
        "query": "What controls can users interact with in Streamlit?",
        "relevant_ids": ["doc_05", "doc_06"]
    },
    {
        "query": "How do embeddings help find similar text?",
        "relevant_ids": ["doc_07", "doc_08", "doc_09"]
    },
    {
        "query": "How does ChromaDB store and organize documents?",
        "relevant_ids": ["doc_10", "doc_11", "doc_12"]
    }
]


# ── Evaluation function ────────────────────────────────────────────────────────
def evaluate(n_results: int, distance_threshold: float = None):
    """
    Run every query in eval_set, compute precision and recall for each,
    and print per-query results plus averages.

    Args:
        n_results:          how many results to retrieve per query
        distance_threshold: if set, only count results with distance <= this
    """
    print(f"\n=== Evaluation: n_results={n_results}, threshold={distance_threshold} ===")

    precisions = []
    recalls    = []

    for i, entry in enumerate(eval_set, 1):
        query = entry["query"]
        relevant_ids = set(entry["relevant_ids"])

        # TODO: Query the collection with query_texts=[query], n_results=n_results
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["distances"]
        )

        # TODO: Extract retrieved IDs from results["ids"][0]
        retrieved_ids = results["ids"][0]

        # TODO: If distance_threshold is set, filter out IDs whose distance
        #       exceeds the threshold

        if distance_threshold is not None:
            retrieved_ids = [
                doc_id
                for doc_id, distance in zip(
                    retrieved_ids, results["distances"][0]
                )
                if distance <= distance_threshold
            ]

        # TODO: Compute precision and recall
        #   overlap   = len(relevant_ids ∩ set(retrieved_ids))
        #   precision = overlap / len(retrieved_ids)  (handle zero division)
        #   recall    = overlap / len(relevant_ids)   (handle zero division)
        overlap = len(relevant_ids & set(retrieved_ids))

        precision = overlap / len(retrieved_ids) if retrieved_ids else 0
        recall = overlap / len(relevant_ids) if relevant_ids else 0

        precisions.append(precision)
        recalls.append(recall)

        print(f"  Query {i}: P={precision*100:.1f}%  R={recall*100:.1f}%  "
              f"| '{query[:50]}'")

    avg_p = sum(precisions) / len(precisions) if precisions else 0
    avg_r = sum(recalls)    / len(recalls)    if recalls    else 0
    print(f"  AVERAGE: P={avg_p*100:.1f}%  R={avg_r*100:.1f}%")
    return avg_p, avg_r


# ── Run at 3 settings ─────────────────────────────────────────────────────────
# TODO: Call evaluate() with 3 different combinations of n_results / threshold
#       e.g. evaluate(n_results=3)
#            evaluate(n_results=5)
#            evaluate(n_results=5, distance_threshold=0.5)

# Evaluation 1: Retrieve 3 documents
evaluate(n_results=3)

# Evaluation 2: Retrieve 5 documents
evaluate(n_results=5)

# Evaluation 3: Retrieve 5 with distance filtering
evaluate(n_results=5, distance_threshold=0.5)


# ── Analysis ─────────────────────────────────────────────────────────────────
print("\n=== ANALYSIS ===")
# TODO: Print your observations as a multi-line string or separate print() calls.
#       Answer:
#         - Which queries consistently performed well? Why?
#         - Which queries failed? What caused the low score?
#         - What would you change to improve the weakest queries?
#         - How did increasing n_results affect precision vs recall?
print("""
ANALYSIS:

1. The best overall setting was n_results=3 with no distance
   threshold. It achieved 61.1% average precision and
   94.4% average recall.

2. Query 6 performed best, reaching 100% precision and
   100% recall with n_results=3. Query 3 performed worst,
   with only 33.3% precision and 100% recall at that setting.

3. Increasing n_results from 3 to 5 improved average recall
   from 94.4% to 100%, but reduced precision from 61.1%
   to 40%. More documents were retrieved, including
   irrelevant results.

4. Using a distance threshold of 0.5 was too restrictive.
   Four queries returned no accepted documents, reducing
   average recall to 25%.

5. To improve search quality, I would test additional
   distance thresholds, expand the document collection,
   improve document descriptions, and evaluate different
   embedding models.

CONCLUSION:
The experiment demonstrates the tradeoff between precision
and recall. Retrieving more documents improves coverage but
can introduce irrelevant results. The best configuration
depends on the application's needs.
""")
