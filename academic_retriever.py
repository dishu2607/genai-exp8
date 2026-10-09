import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = BASE_DIR / "data" / "processed" / "academic_faiss.index"
METADATA_FILE = BASE_DIR / "data" / "processed" / "academic_chunks_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


def main():
    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    model = SentenceTransformer(MODEL_NAME)

    questions = [
        "What undergraduate programmes does SPIT offer?",
        "What are the academic regulations for undergraduate students?",
        "What subjects are included in the first-year engineering syllabus?",
        "Where is SPIT located?",
        "Where can students find the academic calendar?",
    ]

    for question in questions:
        query_vector = model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        scores, indices = index.search(query_vector, TOP_K)

        print("\n" + "=" * 90)
        print("QUESTION:", question)
        print("=" * 90)

        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0:
                continue

            chunk = metadata[idx]

            print(f"\nRank: {rank} | Similarity: {score:.4f}")
            print("Source:", chunk.get("source_file"))
            print("Page:", chunk.get("page_number"))
            print("Type:", chunk.get("document_type"))
            print("Text:", chunk.get("text", "")[:700].replace("\n", " "))


if __name__ == "__main__":
    main()
