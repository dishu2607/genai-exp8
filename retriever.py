import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "data" / "processed" / "spit_faiss.index"

METADATA_FILE = (
    BASE_DIR / "data" / "processed" / "spit_chunks_metadata.pkl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5


# ============================================================
# LOAD MODEL, INDEX AND METADATA
# ============================================================

def load_retriever():

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Loading FAISS index...")

    index = faiss.read_index(str(INDEX_FILE))

    print("Loading chunk metadata...")

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index contains {index.ntotal} vectors, "
            f"but metadata contains {len(metadata)} chunks."
        )

    print(f"Loaded {index.ntotal} indexed chunks.")

    return model, index, metadata


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(query, model, index, metadata, top_k=TOP_K):

    # Generate the query embedding using the same model.
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # FAISS expects float32 vectors.
    query_embedding = query_embedding.astype("float32")

    # Search the vector index.
    scores, indices = index.search(
        query_embedding,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        chunk = metadata[idx]

        results.append({
            "similarity": float(score),
            "text": chunk["text"],
            "source_file": chunk["source_file"],
            "document_type": chunk["document_type"],
            "domain": chunk["domain"],
            "page_number": chunk.get("page_number"),
            "source_url": chunk.get("source_url"),
            "chunk_id": chunk["chunk_id"]
        })

    return results


# ============================================================
# INTERACTIVE TESTING
# ============================================================

def main():

    print("=" * 65)
    print("SPIT CHATBOT — SEMANTIC RETRIEVAL TEST")
    print("=" * 65)

    model, index, metadata = load_retriever()

    print("\nEnter an academic or placement question.")
    print("Type 'exit' to stop.\n")

    while True:

        query = input("Your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results = retrieve(
            query,
            model,
            index,
            metadata,
            top_k=TOP_K
        )

        print("\n" + "=" * 65)
        print("RETRIEVED CHUNKS")
        print("=" * 65)

        for rank, result in enumerate(results, start=1):

            print(f"\nResult {rank}")
            print(f"Similarity: {result['similarity']:.4f}")
            print(f"Domain: {result['domain']}")
            print(f"Source: {result['source_file']}")
            print(f"Page: {result['page_number']}")
            print(f"URL: {result['source_url']}")
            print("-" * 50)

            print(result["text"][:1200])

        print("\n")

    print("Retrieval testing finished.")


if __name__ == "__main__":
    main()