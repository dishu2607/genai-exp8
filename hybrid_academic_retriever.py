import pickle
import re
from pathlib import Path
from collections import defaultdict

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_FILE = (
    BASE_DIR / "data" / "processed" / "academic_faiss.index"
)

METADATA_FILE = (
    BASE_DIR / "data" / "processed" / "academic_chunks_metadata.pkl"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SEMANTIC_TOP_K = 20
KEYWORD_TOP_K = 20
CANDIDATE_TOP_K = 20
FINAL_TOP_K = 5

RRF_K = 60

# How much each signal contributes to the final reranking score.
RRF_WEIGHT = 1.0
SEMANTIC_WEIGHT = 0.25
KEYWORD_WEIGHT = 0.25
COVERAGE_WEIGHT = 0.35
PHRASE_WEIGHT = 0.15

# Avoid treating common question words as important keywords.
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "what", "which", "who", "where", "when", "why",
    "how", "does", "do", "did", "can", "could",
    "would", "should", "will", "shall",
    "i", "we", "you", "they", "it", "this", "that",
    "of", "for", "to", "in", "on", "at", "by",
    "and", "or", "with", "from", "about", "find",
    "tell", "me", "please", "student", "students",
}


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text):
    """Convert text to lowercase tokens."""
    return re.findall(
        r"[a-z0-9]+(?:[-'][a-z0-9]+)*",
        text.lower()
    )


def get_query_terms(question):
    """Keep meaningful terms from the question."""
    return {
        token
        for token in tokenize(question)
        if token not in STOPWORDS and len(token) > 1
    }


def reciprocal_rank_fusion(rankings, k=RRF_K):
    """
    Combine ranked lists using Reciprocal Rank Fusion.

    Each ranking contains (chunk_index, original_score) pairs.
    """
    fused_scores = defaultdict(float)

    for ranking in rankings:
        for rank, (chunk_index, _) in enumerate(
            ranking,
            start=1
        ):
            fused_scores[chunk_index] += 1.0 / (k + rank)

    return sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )


# ============================================================
# LOAD RESOURCES
# ============================================================

def load_resources():
    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Metadata not found: {METADATA_FILE}"
        )

    index = faiss.read_index(str(INDEX_FILE))

    with METADATA_FILE.open("rb") as f:
        metadata = pickle.load(f)

    if index.ntotal != len(metadata):
        raise ValueError(
            f"Index/metadata mismatch: {index.ntotal} vectors, "
            f"{len(metadata)} metadata records."
        )

    if not metadata:
        raise ValueError("Academic metadata is empty.")

    for item in metadata:
        if item.get("domain") != "academic":
            raise ValueError(
                "Non-academic record found: "
                f"{item.get('source_file')}"
            )

    model = SentenceTransformer(MODEL_NAME)

    tokenized_corpus = [
        tokenize(item.get("text", ""))
        for item in metadata
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return index, metadata, model, bm25


# ============================================================
# QUERY-FOCUSED RERANKING
# ============================================================

def calculate_coverage(question, chunk_text):
    """
    Calculate the proportion of meaningful query terms
    appearing in the retrieved passage.
    """
    query_terms = get_query_terms(question)

    if not query_terms:
        return 0.0

    chunk_terms = set(tokenize(chunk_text))

    matched_terms = query_terms.intersection(chunk_terms)

    return len(matched_terms) / len(query_terms)


def calculate_phrase_match(question, chunk_text):
    """
    Give a boost when a meaningful phrase from the question
    occurs directly in the passage.

    This is a simple lexical feature, not a semantic judgment.
    """
    question_lower = " ".join(tokenize(question))
    chunk_lower = " ".join(tokenize(chunk_text))

    phrases = [
        "undergraduate programmes",
        "undergraduate programs",
        "academic regulations",
        "first year engineering",
        "engineering syllabus",
        "academic calendar",
        "college located",
        "spit located",
    ]

    matching_phrases = [
        phrase
        for phrase in phrases
        if phrase in question_lower
        and phrase in chunk_lower
    ]

    if matching_phrases:
        return 1.0

    # Also check the cleaned complete question when appropriate.
    if (
        len(question_lower) >= 12
        and question_lower in chunk_lower
    ):
        return 1.0

    return 0.0


def calculate_reranking_scores(
    question,
    candidate_indices,
    fused_ranking,
    semantic_scores,
    keyword_scores,
    metadata,
):
    """
    Combine retrieval rankings and question-to-passage features.

    All component scores are normalized before combination.
    """
    fused_lookup = dict(fused_ranking)

    # Normalize semantic scores across the retrieved candidates.
    semantic_values = [
        semantic_scores.get(idx, 0.0)
        for idx in candidate_indices
    ]

    semantic_min = min(semantic_values, default=0.0)
    semantic_max = max(semantic_values, default=0.0)

    # Normalize BM25 scores across the retrieved candidates.
    keyword_values = [
        keyword_scores.get(idx, 0.0)
        for idx in candidate_indices
    ]

    keyword_min = min(keyword_values, default=0.0)
    keyword_max = max(keyword_values, default=0.0)

    results = []

    for idx in candidate_indices:
        chunk_text = metadata[idx].get("text", "")

        semantic_score = semantic_scores.get(idx, 0.0)
        keyword_score = keyword_scores.get(idx, 0.0)

        if semantic_max > semantic_min:
            normalized_semantic = (
                semantic_score - semantic_min
            ) / (semantic_max - semantic_min)
        else:
            normalized_semantic = 0.0

        if keyword_max > keyword_min:
            normalized_keyword = (
                keyword_score - keyword_min
            ) / (keyword_max - keyword_min)
        else:
            normalized_keyword = 0.0

        coverage = calculate_coverage(
            question,
            chunk_text
        )

        phrase_match = calculate_phrase_match(
            question,
            chunk_text
        )

        # RRF scores are small, so normalize them for combination.
        rrf_score = fused_lookup.get(idx, 0.0)

        max_rrf = max(fused_lookup.values(), default=0.0)

        normalized_rrf = (
            rrf_score / max_rrf
            if max_rrf > 0
            else 0.0
        )

        final_score = (
            RRF_WEIGHT * normalized_rrf
            + SEMANTIC_WEIGHT * normalized_semantic
            + KEYWORD_WEIGHT * normalized_keyword
            + COVERAGE_WEIGHT * coverage
            + PHRASE_WEIGHT * phrase_match
        )

        results.append({
            "index": idx,
            "score": final_score,
            "rrf": normalized_rrf,
            "semantic": normalized_semantic,
            "keyword": normalized_keyword,
            "coverage": coverage,
            "phrase_match": phrase_match,
        })

    return sorted(
        results,
        key=lambda item: item["score"],
        reverse=True
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve(
    question,
    index,
    metadata,
    model,
    bm25,
    final_top_k=FINAL_TOP_K,
):
    # --------------------------------------------------------
    # 1. Semantic retrieval
    # --------------------------------------------------------

    query_vector = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    semantic_scores_array, semantic_indices_array = (
        index.search(
            query_vector,
            min(SEMANTIC_TOP_K, len(metadata))
        )
    )

    semantic_ranking = []
    semantic_score_lookup = {}

    for score, idx in zip(
        semantic_scores_array[0],
        semantic_indices_array[0]
    ):
        if idx < 0:
            continue

        idx = int(idx)
        score = float(score)

        semantic_ranking.append((idx, score))
        semantic_score_lookup[idx] = score

    # --------------------------------------------------------
    # 2. Keyword retrieval
    # --------------------------------------------------------

    query_tokens = tokenize(question)
    keyword_scores_array = bm25.get_scores(query_tokens)

    keyword_count = min(
        KEYWORD_TOP_K,
        len(metadata)
    )

    keyword_indices = sorted(
        range(len(keyword_scores_array)),
        key=lambda idx: float(keyword_scores_array[idx]),
        reverse=True
    )[:keyword_count]

    keyword_ranking = [
        (idx, float(keyword_scores_array[idx]))
        for idx in keyword_indices
    ]

    keyword_score_lookup = {
        idx: float(keyword_scores_array[idx])
        for idx in range(len(keyword_scores_array))
    }

    # --------------------------------------------------------
    # 3. Fuse rankings
    # --------------------------------------------------------

    fused_ranking = reciprocal_rank_fusion(
        [semantic_ranking, keyword_ranking]
    )

    # --------------------------------------------------------
    # 4. Build candidate pool
    # --------------------------------------------------------

    candidate_indices = [
        idx
        for idx, _ in fused_ranking[:CANDIDATE_TOP_K]
    ]

    # --------------------------------------------------------
    # 5. Rerank candidates
    # --------------------------------------------------------

    reranked_results = calculate_reranking_scores(
        question=question,
        candidate_indices=candidate_indices,
        fused_ranking=fused_ranking,
        semantic_scores=semantic_score_lookup,
        keyword_scores=keyword_score_lookup,
        metadata=metadata,
    )

    return reranked_results[:final_top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def main():
    print("=" * 90)
    print("SPIT ACADEMIC CHATBOT — HYBRID RETRIEVAL + RERANKING")
    print("=" * 90)

    index, metadata, model, bm25 = load_resources()

    print("Academic chunks:", len(metadata))
    print("FAISS vectors:", index.ntotal)

    questions = [
        "What undergraduate programmes does SPIT offer?",
        "What are the academic regulations for undergraduate students?",
        "What subjects are included in the first-year engineering syllabus?",
        "Where is SPIT located?",
        "Where can students find the academic calendar?",
    ]

    for question in questions:
        results = retrieve(
            question,
            index,
            metadata,
            model,
            bm25,
        )

        print("\n" + "=" * 100)
        print("QUESTION:", question)
        print("=" * 100)

        for rank, result in enumerate(results, start=1):
            chunk = metadata[result["index"]]

            print(
                f"\nRank: {rank} | "
                f"Final score: {result['score']:.4f}"
            )

            print(
                f"RRF: {result['rrf']:.3f} | "
                f"Semantic: {result['semantic']:.3f} | "
                f"BM25: {result['keyword']:.3f} | "
                f"Coverage: {result['coverage']:.3f} | "
                f"Phrase: {result['phrase_match']:.1f}"
            )

            print("Source:", chunk.get("source_file"))
            print("Page:", chunk.get("page_number"))
            print("Type:", chunk.get("document_type"))

            print(
                "Text:",
                chunk.get("text", "")[:900].replace("\n", " ")
            )


if __name__ == "__main__":
    main()