import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
INDEX_FILE = OUTPUT_DIR / "spit_faiss.index"
METADATA_FILE = OUTPUT_DIR / "spit_chunks_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}\n"
            "Run chunker.py first."
        )

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))

    # Ignore empty text records.
    chunks = [
        chunk for chunk in chunks
        if chunk.get("text", "").strip()
    ]

    if not chunks:
        raise ValueError("No usable chunks found.")

    return chunks


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks):

    print(f"\nLoading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    print(f"Embedding matrix shape: {embeddings.shape}")

    return embeddings


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(embeddings):

    dimension = embeddings.shape[1]

    # Inner product on normalized vectors equals cosine similarity.
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# SAVE INDEX AND METADATA
# ============================================================

def save_outputs(index, chunks):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    # Keep metadata in exactly the same order as indexed vectors.
    with METADATA_FILE.open("wb") as f:
        pickle.dump(chunks, f)

    print("\nSaved vector index:")
    print(INDEX_FILE)

    print("\nSaved chunk metadata:")
    print(METADATA_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("SPIT CHATBOT — EMBEDDING GENERATION")
    print("=" * 65)

    chunks = load_chunks()

    print(f"Chunks loaded: {len(chunks)}")

    embeddings = generate_embeddings(chunks)

    index = build_index(embeddings)

    save_outputs(index, chunks)

    print("\n" + "=" * 65)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 65)

    print(f"Indexed vectors: {index.ntotal}")
    print(f"Vector dimensions: {embeddings.shape[1]}")

    print("\nNext stage: semantic retrieval testing.")


if __name__ == "__main__":
    main()