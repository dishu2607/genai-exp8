"""
Experiment D: ICL vs Hybrid RAG on SPIT First-Year Engineering syllabus.

Run from project root:
    python experiment_d_rag_vs_icl.py

Uses the existing test_single_pdf_rag.py index and functions; does not rebuild it.
Outputs CSV, JSON results, and metrics under experiments/experiment_d/.
Keyword scores are proxies; manually verify answers against the actual PDF.
"""
import csv
import json
import re
import time
from pathlib import Path
from statistics import mean

import test_single_pdf_rag as rag

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experiments" / "experiment_d"
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "rag_vs_icl_results.csv"
JSON_PATH = OUT / "rag_vs_icl_results.json"
METRICS_PATH = OUT / "rag_vs_icl_metrics.json"
TOP_K = 6

# Wide topic coverage: identity, semesters, curriculum groups, outcomes,
# teaching/assessment structure, course lists, and out-of-document questions.
# expected=None means manual scoring is required; no guessed gold label is used.
QUESTIONS = [
    {"id": 1, "topic": "Course identity", "question": "What is the course code for Engineering Calculus?", "expected": ["MA101"], "answerable": True},
    {"id": 2, "topic": "Course identity", "question": "What is the full course name associated with MA101?", "expected": ["Engineering Calculus"], "answerable": True},
    {"id": 3, "topic": "Credits", "question": "How many credits does MA101 carry?", "expected": None, "answerable": None},
    {"id": 4, "topic": "Semester structure", "question": "Which semesters are covered by the First-Year Engineering syllabus?", "expected": ["Semester I", "Semester II"], "answerable": True},
    {"id": 5, "topic": "Curriculum overview", "question": "What is the official title of the syllabus document?", "expected": ["First Year Engineering", "Semester I-II"], "answerable": True},
    {"id": 6, "topic": "Curriculum categories", "question": "What does BSESC stand for in the curriculum classification table?", "expected": ["Basic Science", "Engineering Science Courses"], "answerable": True},
    {"id": 7, "topic": "Curriculum categories", "question": "What does BSESEC stand for?", "expected": ["Basic Science", "Engineering Science Elective Courses"], "answerable": True},
    {"id": 8, "topic": "Curriculum categories", "question": "What does SEC stand for in the course-category table?", "expected": ["Skill Enhancement Course"], "answerable": True},
    {"id": 9, "topic": "Curriculum categories", "question": "What does LLC stand for in the curriculum classification table?", "expected": ["Liberal Learning Courses"], "answerable": True},
    {"id": 10, "topic": "Curriculum categories", "question": "What does HSSMC stand for?", "expected": ["Humanities", "Social Science", "Management Courses"], "answerable": True},
    {"id": 11, "topic": "Curriculum categories", "question": "What does PCC stand for in the curriculum classification table?", "expected": ["Program Core Courses"], "answerable": True},
    {"id": 12, "topic": "Curriculum categories", "question": "What does PEC stand for?", "expected": ["Program Elective Courses"], "answerable": True},
    {"id": 13, "topic": "Curriculum categories", "question": "What does OEC stand for?", "expected": ["Open Elective Courses"], "answerable": True},
    {"id": 14, "topic": "Curriculum categories", "question": "What does MDM stand for?", "expected": ["Multidisciplinary Minor"], "answerable": True},
    {"id": 15, "topic": "Course list", "question": "List the courses prescribed for Semester I.", "expected": None, "answerable": True},
    {"id": 16, "topic": "Course list", "question": "List the courses prescribed for Semester II.", "expected": None, "answerable": True},
    {"id": 17, "topic": "Learning outcomes", "question": "What learning outcomes are listed for Engineering Calculus?", "expected": None, "answerable": True},
    {"id": 18, "topic": "Course structure", "question": "What do L, T, P, C, SE, and TE mean in the syllabus abbreviations table?", "expected": None, "answerable": True},
    {"id": 19, "topic": "Assessment", "question": "What assessment components and marks are prescribed for MA101?", "expected": None, "answerable": True},
    {"id": 20, "topic": "Teaching structure", "question": "What are the lecture, tutorial, and practical hours for MA101?", "expected": None, "answerable": True},
    {"id": 21, "topic": "Document scope", "question": "What was the average CGPA of students who graduated from SPIT last year?", "expected": None, "answerable": False},
    {"id": 22, "topic": "Document scope", "question": "What was the highest placement package offered at SPIT this year?", "expected": None, "answerable": False},
    {"id": 23, "topic": "Document scope", "question": "What is the personal phone number of the institute director?", "expected": None, "answerable": False},
    {"id": 24, "topic": "Document scope", "question": "What was the attendance percentage of a specific student in Semester I?", "expected": None, "answerable": False},
    {"id": 25, "topic": "Document scope", "question": "What is tomorrow's weather forecast for Mumbai?", "expected": None, "answerable": False},
]

# Few-shot demonstrations are not test questions. The ICL condition gets no
# retrieved syllabus passages; examples teach answer style and abstention.
DEMONSTRATIONS = [
    ("What is the course code for Engineering Calculus?", "The course code is MA101."),
    ("What does SEC stand for?", "SEC stands for Skill Enhancement Course."),
    ("What does MDM stand for?", "MDM stands for Multidisciplinary Minor."),
    ("What was last year's average graduate CGPA?", "I cannot determine that from the supplied syllabus."),
]


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").lower())).strip()


def is_abstention(answer):
    text = normalize(answer)
    phrases = [
        "not available", "not specified", "not provided", "cannot be determined",
        "cannot determine", "cannot answer", "cannot establish", "unable to determine",
        "insufficient information", "does not contain", "does not include",
        "does not provide", "does not mention", "not present in the document",
        "not included in the document", "not stated in the document",
        "not mentioned in the document", "not established in the evidence",
        "based on the available evidence", "from the available evidence",
        "from the provided evidence", "the document does not", "the syllabus does not",
        "i cannot verify", "i do not have enough information",
        "no information is available", "there is no information",
        "could not find enough evidence", "couldn t find enough evidence",
    ]
    return any(p in text for p in phrases)


def score_answer(answer, item):
    expected = item.get("expected")
    abstained = is_abstention(answer)
    if expected is None:
        return None, abstained, "manual_review_required"
    text = normalize(answer)
    expected_norm = [normalize(x) for x in expected if normalize(x)]
    correct = bool(expected_norm) and all(x in text for x in expected_norm)
    return correct, abstained, "automatic_keyword_proxy"


def answer_icl(question):
    examples = "\n\n".join(
        f"Question: {q}\nAnswer: {a}" for q, a in DEMONSTRATIONS
    )
    # Existing generate_answer(query, evidence) function keeps the same
    # Groq/model configuration as the RAG condition.
    evidence = (
        "FEW-SHOT EXAMPLES (response style only; not the complete syllabus):\n"
        + examples
        + "\n\nNo syllabus passages have been retrieved for this question. "
        "Answer only if the examples support it; otherwise abstain."
    )
    return rag.generate_answer(question, evidence)


def chunk_text(chunk):
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        for key in ("text", "content", "page_content", "chunk_text"):
            value = chunk.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(chunk, ensure_ascii=False, default=str)
    if hasattr(chunk, "text"):
        return str(chunk.text)
    return str(chunk)


def retrieve_evidence(question, model, chunks, records, vector_index, bm25):
    retrieved = rag.retrieve(question, model, chunks, vector_index, bm25)
    parts, details = [], []
    for rank, item in enumerate(retrieved[:TOP_K], start=1):
        chunk = item.get("chunk", item) if isinstance(item, dict) else item
        text = chunk_text(chunk)
        if text.strip():
            parts.append(f"[PASSAGE {rank}]\n{text}")
        details.append({
            "rank": rank,
            "rrf_score": item.get("rrf_score") if isinstance(item, dict) else None,
            "prior": item.get("prior") if isinstance(item, dict) else None,
            "chunk": text,
        })
    evidence = "\n\n".join(parts)
    try:
        structured = rag.structured_lookup(question, records)
        if structured is not None:
            evidence += "\n\nSTRUCTURED LOOKUP RESULT:\n" + json.dumps(
                structured, ensure_ascii=False, indent=2, default=str
            )
    except Exception as exc:
        details.append({"structured_lookup_error": str(exc)})
    return evidence or "No relevant evidence was retrieved.", details


def calculate_metrics(results, prefix):
    scored = [r for r in results if r.get(prefix + "_correct") is not None]
    correct = sum(r[prefix + "_correct"] is True for r in scored)
    unanswerable = [r for r in results if r.get("answerable") is False]
    non_abstentions = [
        r for r in unanswerable if r.get(prefix + "_abstained") is False
    ]
    latencies = [
        r[prefix + "_latency_seconds"] for r in results
        if r.get(prefix + "_latency_seconds") is not None
    ]
    return {
        "total_questions": len(results),
        "automatically_scored_questions": len(scored),
        "correct_answers": correct,
        "keyword_accuracy_percent": round(100 * correct / len(scored), 2) if scored else None,
        "manual_review_questions": sum(r.get(prefix + "_correct") is None for r in results),
        "unanswerable_questions": len(unanswerable),
        "unanswerable_non_abstentions": len(non_abstentions),
        "unanswerable_non_abstention_rate_percent": (
            round(100 * len(non_abstentions) / len(unanswerable), 2)
            if unanswerable else None
        ),
        "mean_latency_seconds": round(mean(latencies), 3) if latencies else None,
    }


def main():
    print("=" * 84)
    print("EXPERIMENT D: ICL VS HYBRID RAG — MULTI-TOPIC SYLLABUS QA")
    print("=" * 84)
    print(f"Questions: {len(QUESTIONS)}")
    print("Loading existing saved index...")
    model, chunks, records, vector_index, bm25 = rag.load_index()
    print(f"Index loaded: {len(chunks)} chunks, {len(records)} records.")

    results = []
    for item in QUESTIONS:
        qid, question = item["id"], item["question"]
        print("\n" + "-" * 84)
        print(f"Q{qid}/{len(QUESTIONS)} | {item['topic']}: {question}")
        row = {
            "id": qid, "topic": item["topic"], "question": question,
            "expected": item.get("expected"), "answerable": item.get("answerable"),
        }

        try:
            start = time.perf_counter()
            icl_answer = answer_icl(question)
            elapsed = time.perf_counter() - start
            correct, abstained, status = score_answer(icl_answer, item)
            row.update({
                "icl_answer": icl_answer,
                "icl_latency_seconds": round(elapsed, 3),
                "icl_correct": correct, "icl_abstained": abstained,
                "icl_scoring_status": status,
            })
        except Exception as exc:
            row.update({
                "icl_answer": "ERROR: " + str(exc), "icl_latency_seconds": None,
                "icl_correct": None, "icl_abstained": None,
                "icl_scoring_status": "error",
            })

        try:
            evidence, details = retrieve_evidence(
                question, model, chunks, records, vector_index, bm25
            )
            start = time.perf_counter()
            rag_answer = rag.generate_answer(question, evidence)
            elapsed = time.perf_counter() - start
            correct, abstained, status = score_answer(rag_answer, item)
            row.update({
                "rag_answer": rag_answer,
                "rag_latency_seconds": round(elapsed, 3),
                "rag_correct": correct, "rag_abstained": abstained,
                "rag_scoring_status": status,
                "rag_evidence": evidence, "retrieval_details": details,
            })
        except Exception as exc:
            row.update({
                "rag_answer": "ERROR: " + str(exc), "rag_latency_seconds": None,
                "rag_correct": None, "rag_abstained": None,
                "rag_scoring_status": "error",
                "rag_evidence": "", "retrieval_details": [],
            })

        results.append(row)
        print("ICL:", row.get("icl_answer", ""))
        print("RAG:", row.get("rag_answer", ""))
        print("RAG evidence preview:\n", (row.get("rag_evidence") or "")[:800])
        print("Keyword score proxy: ICL =", row.get("icl_correct"),
              "| RAG =", row.get("rag_correct"))

    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    csv_rows = []
    for row in results:
        flat = dict(row)
        flat["retrieval_details"] = json.dumps(
            flat.get("retrieval_details", []), ensure_ascii=False, default=str
        )
        csv_rows.append(flat)
    fields = sorted({key for row in csv_rows for key in row})
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)

    metrics = {
        "experiment": "ICL vs Hybrid RAG",
        "question_count": len(QUESTIONS),
        "evaluation_warning": (
            "Keyword accuracy is only a proxy. expected=None items require manual "
            "review against the syllabus. Do not report these figures as definitive "
            "accuracy or hallucination rates."
        ),
        "ICL": calculate_metrics(results, "icl"),
        "RAG": calculate_metrics(results, "rag"),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 84)
    print("FINAL METRICS")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("\nSaved files:")
    print(CSV_PATH)
    print(JSON_PATH)
    print(METRICS_PATH)
    print("\nVerify gold answers against the PDF before reporting accuracy.")


if __name__ == "__main__":
    main()
