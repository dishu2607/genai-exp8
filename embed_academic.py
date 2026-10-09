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

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "academic_chunks.jsonl"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
INDEX_FILE = OUTPUT_DIR / "academic_faiss.index"
METADATA_FILE = OUTPUT_DIR / "academic_chunks_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32


# ============================================================
# LOAD ACADEMIC CHUNKS
# ============================================================

def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Chunk file not found: {CHUNKS_FILE}")

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            chunk = json.loads(line)

            if chunk.get("domain") != "academic":
                raise ValueError(
                    f"Non-academic chunk found: {chunk.get('source_file')}"
                )

            if chunk.get("text", "").strip():
                chunks.append(chunk)

    if not chunks:
        raise ValueError("No usable academic chunks found.")

    return chunks


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks):
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} academic chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Unexpected embedding shape: {embeddings.shape}"
        )

    print("Embedding matrix shape:", embeddings.shape)

    return embeddings


# ============================================================
# BUILD INDEX
# ============================================================

def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# ============================================================
# SAVE INDEX AND MATCHING METADATA
# ============================================================

def save_outputs(index, chunks):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_FILE))

    with METADATA_FILE.open("wb") as f:
        pickle.dump(chunks, f)

    print("\nFAISS index saved:", INDEX_FILE)
    print("Metadata saved:", METADATA_FILE)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("SPIT CHATBOT — ACADEMIC EMBEDDINGS")
    print("=" * 65)

    chunks = load_chunks()
    print("Academic chunks loaded:", len(chunks))

    embeddings = generate_embeddings(chunks)
    index = build_index(embeddings)

    save_outputs(index, chunks)

    print("\n" + "=" * 65)
    print("ACADEMIC INDEX CREATED SUCCESSFULLY")
    print("=" * 65)
    print("Indexed vectors:", index.ntotal)
    print("Vector dimensions:", embeddings.shape[1])
    print("Placement chunks included: 0")


if __name__ == "__main__":
    main()