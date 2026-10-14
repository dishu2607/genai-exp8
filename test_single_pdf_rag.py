
import os
import re
import json
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss
from dotenv import load_dotenv
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

ROOT = Path(__file__).resolve().parent

SOURCE_FILE = (
    ROOT / "data" / "processed" / "single_pdf_docling.md"
)

INDEX_DIR = ROOT / "data" / "index"

CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
RECORDS_FILE = INDEX_DIR / "curriculum_records.json"
FAISS_FILE = INDEX_DIR / "faiss.index"
BM25_FILE = INDEX_DIR / "bm25.pkl"
MANIFEST_FILE = INDEX_DIR / "manifest.json"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

# Set True for one run to rebuild all indexes.
FORCE_REBUILD = False

DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 6

PROSE_CHUNK_WORDS = 220
PROSE_OVERLAP_WORDS = 40
TABLE_ROWS_PER_CHUNK = 5

SCHEMA_VERSION = 3


# ============================================================
# 2. UTILITIES
# ============================================================

def normalize_text(text):
    text = str(text).replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def tokenize(text):
    return re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )


def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(
            lambda: file.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:12]


def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as file:
        for item in items:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def read_jsonl(path):
    items = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                items.append(json.loads(line))

    return items


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# 3. MARKDOWN TABLE PARSING
# ============================================================

def is_separator_row(line):
    if not line.strip().startswith("|"):
        return False

    cells = line.strip().strip("|").split("|")

    return bool(cells) and all(
        re.fullmatch(
            r"\s*:?-{3,}:?\s*",
            cell
        )
        for cell in cells
    )


def parse_table_row(line):
    line = line.strip()

    if not line.startswith("|"):
        return None

    if is_separator_row(line):
        return None

    cells = line.strip("|").split("|")

    return [
        normalize_text(cell)
        for cell in cells
    ]


def parse_markdown(markdown):
    """
    Split the cached Markdown into prose and table blocks.

    This preserves table rows, but cannot restore relationships
    that were already lost during PDF extraction.
    """

    lines = markdown.splitlines()

    blocks = []
    prose_buffer = []
    current_heading = ""
    i = 0

    def flush_prose():
        nonlocal prose_buffer

        text = normalize_text(
            "\n".join(prose_buffer)
        )

        if text:
            blocks.append({
                "type": "prose",
                "text": text,
                "heading": current_heading
            })

        prose_buffer = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("#"):
            flush_prose()

            current_heading = (
                stripped.lstrip("#").strip()
            )

            i += 1
            continue

        is_table = (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and (
                is_separator_row(lines[i + 1])
                or lines[i + 1].strip().startswith("|")
            )
        )

        if is_table:
            flush_prose()

            table_lines = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
            ):
                table_lines.append(lines[i])
                i += 1

            rows = []

            for table_line in table_lines:
                row = parse_table_row(table_line)

                if row is not None:
                    rows.append(row)

            if rows:
                blocks.append({
                    "type": "table",
                    "rows": rows,
                    "heading": current_heading
                })

            continue

        if stripped:
            prose_buffer.append(line)

        i += 1

    flush_prose()

    return blocks


# ============================================================
# 4. TABLE CLASSIFICATION
# ============================================================

COURSE_CODE_PATTERN = re.compile(
    r"\b(?:MA|EC|PH|CH|CS|CE|ME|EE|HS|AS|"
    r"BSC|ESC|PCC|BSES)"
    r"[A-Z0-9]{0,5}\d{2,4}[A-Z]?\b",
    re.IGNORECASE
)


def get_course_codes(text):
    return sorted({
        match.group(0).upper()
        for match in COURSE_CODE_PATTERN.finditer(text)
    })


def table_to_text(rows):
    return "\n".join(
        " | ".join(row)
        for row in rows
    )


def classify_table(block):
    """
    Classify likely curriculum tables separately from
    detailed syllabus and course-outcome tables.

    This is heuristic classification, not proof of correctness.
    """

    rows = block["rows"]
    heading = block.get("heading", "").lower()

    first_row = (
        " ".join(rows[0]).lower()
        if rows else ""
    )

    sample = table_to_text(rows[:3]).lower()

    score = 0

    if "curriculum structure" in heading:
        score += 5

    if "course code" in first_row:
        score += 4

    if (
        "course name" in first_row
        or "subject name" in first_row
    ):
        score += 4

    if "semester" in first_row:
        score += 2

    if re.search(r"\bcredits?\b", first_row):
        score += 2

    if "course (category)" in first_row:
        score -= 6

    if "examination scheme" in sample:
        score -= 3

    if "pre-requisite course codes" in first_row:
        score -= 6

    if "course outcomes" in sample:
        score -= 3

    if score >= 7:
        return "curriculum_candidate"

    if (
        "course outcomes" in sample
        or "pre-requisite course codes" in first_row
    ):
        return "outcomes_or_details"

    return "other_table"


# ============================================================
# 5. EXTRACT CURRICULUM RECORD CANDIDATES
# ============================================================

def extract_curriculum_records(blocks):
    """
    Extract candidate course records from tables classified
    as curriculum candidates.

    Records retain source locations so that conflicting
    extraction results can be investigated.
    """

    records = []

    for table_number, block in enumerate(
        blocks,
        start=1
    ):
        if block["type"] != "table":
            continue

        if classify_table(block) != "curriculum_candidate":
            continue

        rows = block["rows"]

        if len(rows) < 2:
            continue

        headers = [
            cell.lower().strip()
            for cell in rows[0]
        ]

        code_col = next(
            (
                i for i, header in enumerate(headers)
                if (
                    "course code" in header
                    and "category" not in header
                )
            ),
            None
        )

        name_col = next(
            (
                i for i, header in enumerate(headers)
                if (
                    "course name" in header
                    or "subject name" in header
                )
            ),
            None
        )

        credit_col = next(
            (
                i for i, header in enumerate(headers)
                if header in {
                    "c",
                    "credits",
                    "credit",
                    "total credits",
                    "credit assigned",
                    "credits assigned"
                }
            ),
            None
        )

        semester_col = next(
            (
                i for i, header in enumerate(headers)
                if "semester" in header
            ),
            None
        )

        for row_number, row in enumerate(
            rows[1:],
            start=2
        ):
            padded = row + [
                ""
            ] * max(0, len(headers) - len(row))

            code = (
                padded[code_col]
                if code_col is not None
                else ""
            )

            name = (
                padded[name_col]
                if name_col is not None
                else ""
            )

            credits = (
                padded[credit_col]
                if credit_col is not None
                else ""
            )

            semester = (
                padded[semester_col]
                if semester_col is not None
                else ""
            )

            # Fallback code detection, without pretending
            # that the remaining columns are necessarily correct.
            if not COURSE_CODE_PATTERN.search(code):
                matches = [
                    COURSE_CODE_PATTERN.search(cell)
                    for cell in row
                ]

                matches = [
                    match for match in matches
                    if match is not None
                ]

                if matches:
                    code = matches[0].group(0)

            code_match = COURSE_CODE_PATTERN.search(code)

            if not code_match:
                continue

            code = code_match.group(0).upper()
            name = normalize_text(name)
            credits = normalize_text(credits)
            semester = normalize_text(semester)

            if not name:
                continue

            if name.lower() in {
                "total",
                "theory",
                "practical",
                "tutorial",
                "examination scheme",
                "course outcomes"
            }:
                continue

            if len(name) > 140:
                continue

            credit_match = re.fullmatch(
                r"\s*(\d+(?:\.\d+)?)\s*",
                credits
            )

            credit_value = (
                credit_match.group(1)
                if credit_match
                else None
            )

            records.append({
                "course_code": code,
                "course_name": name,
                "semester": semester or None,
                "credits": credit_value,
                "table_number": table_number,
                "row_number": row_number,
                "heading": block.get("heading", ""),
                "source_row": " | ".join(row),
                "table_kind": "curriculum_candidate"
            })

    # Do not silently collapse conflicting source rows.
    # Deduplicate only records with identical values.
    unique = []
    seen = set()

    for record in records:
        key = (
            record["course_code"],
            record["course_name"].casefold(),
            record["semester"],
            record["credits"],
            record["table_number"],
            record["row_number"]
        )

        if key not in seen:
            seen.add(key)
            unique.append(record)

    return unique


# ============================================================
# 6. BUILD PROSE CHUNKS
# ============================================================

def split_prose(text, heading=""):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0

    while start < len(words):
        end = min(
            start + PROSE_CHUNK_WORDS,
            len(words)
        )

        content = " ".join(words[start:end])

        if heading:
            content = (
                f"Section: {heading}\n"
                f"{content}"
            )

        chunks.append(content)

        if end >= len(words):
            break

        start = max(
            start + 1,
            end - PROSE_OVERLAP_WORDS
        )

    return chunks


# ============================================================
# 7. BUILD ALL CHUNKS
# ============================================================

def build_chunks(blocks, records):
    chunks = []

    table_number = 0

    for block in blocks:
        heading = block.get("heading", "")

        if block["type"] == "prose":
            prose_chunks = split_prose(
                block["text"],
                heading
            )

            for part_number, text in enumerate(
                prose_chunks,
                start=1
            ):
                chunks.append({
                    "text": text,
                    "chunk_type": "prose",
                    "heading": heading,
                    "table_kind": None,
                    "source_table": None,
                    "source_row_start": None,
                    "course_codes": get_course_codes(text)
                })

            continue

        table_number += 1

        rows = block["rows"]
        kind = classify_table(block)

        if not rows:
            continue

        header = rows[0]
        data_rows = rows[1:] if len(rows) > 1 else rows

        for start in range(
            0,
            len(data_rows),
            TABLE_ROWS_PER_CHUNK
        ):
            selected_rows = data_rows[
                start:start + TABLE_ROWS_PER_CHUNK
            ]

            content_lines = [
                " | ".join(header)
            ]

            content_lines.extend(
                " | ".join(row)
                for row in selected_rows
            )

            text = (
                f"Section: {heading or 'Unlabelled section'}\n"
                f"Table type: {kind}\n"
                f"Table number: {table_number}\n"
                + "\n".join(content_lines)
            )

            chunks.append({
                "text": text,
                "chunk_type": "table",
                "heading": heading,
                "table_kind": kind,
                "source_table": table_number,
                "source_row_start": start + 2,
                "course_codes": get_course_codes(text)
            })

    # Add individual curriculum records as their own searchable chunks.
    for record in records:
        text = (
            "Curriculum record candidate\n"
            f"Course code: {record['course_code']}\n"
            f"Course name: {record['course_name']}\n"
            f"Semester: {record.get('semester') or 'Not specified'}\n"
            f"Credits: {record.get('credits') or 'Not established'}\n"
            f"Source heading: {record.get('heading') or 'Unknown'}\n"
            f"Source table: {record['table_number']}\n"
            f"Source row: {record['row_number']}"
        )

        chunks.append({
            "text": text,
            "chunk_type": "curriculum_record_candidate",
            "heading": record.get("heading", ""),
            "table_kind": "curriculum_candidate",
            "source_table": record["table_number"],
            "source_row_start": record["row_number"],
            "course_codes": [record["course_code"]]
        })

    # CRITICAL FIX:
    # Assign unique IDs using the chunk's position AND text hash.
    # Identical text is allowed, but IDs must remain unique.
    for position, chunk in enumerate(chunks):
        digest = text_hash(chunk["text"])

        chunk["chunk_id"] = (
            f"chunk_{position:06d}_{digest}"
        )

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Unique ID assignment failed unexpectedly."
        )

    return chunks


# ============================================================
# 8. INDEX VALIDATION
# ============================================================

def index_needs_rebuild(source_hash):
    if FORCE_REBUILD:
        return True

    required_files = [
        CHUNKS_FILE,
        RECORDS_FILE,
        FAISS_FILE,
        BM25_FILE,
        MANIFEST_FILE
    ]

    if not all(path.exists() for path in required_files):
        return True

    try:
        with open(
            MANIFEST_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            manifest = json.load(file)

        return (
            manifest.get("source_sha256") != source_hash
            or manifest.get("embedding_model") != EMBEDDING_MODEL
            or manifest.get("schema_version") != SCHEMA_VERSION
        )

    except Exception:
        return True


# ============================================================
# 9. BUILD AND SAVE PERSISTENT INDEXES
# ============================================================

def build_index(markdown, source_hash, model):
    print("\n[INDEX] Parsing Markdown...")

    blocks = parse_markdown(markdown)

    table_count = sum(
        block["type"] == "table"
        for block in blocks
    )

    print(f"[INDEX] Parsed blocks: {len(blocks)}")
    print(f"[INDEX] Tables: {table_count}")

    records = extract_curriculum_records(blocks)

    chunks = build_chunks(
        blocks,
        records
    )

    if not chunks:
        raise RuntimeError(
            "No chunks created. Check the Markdown source."
        )

    print(
        "[INDEX] Curriculum record candidates:",
        len(records)
    )

    print(
        "[INDEX] Total chunks:",
        len(chunks)
    )

    print(
        "[INDEX] Duplicate text chunks:",
        len(chunks) - len({
            chunk["text"] for chunk in chunks
        })
    )

    INDEX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save chunks and structured candidates.
    write_jsonl(
        CHUNKS_FILE,
        chunks
    )

    save_json(
        RECORDS_FILE,
        records
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("\n[INDEX] Generating BGE embeddings...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    # Normalized vectors + inner product = cosine similarity.
    vector_index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    vector_index.add(embeddings)

    faiss.write_index(
        vector_index,
        str(FAISS_FILE)
    )

    print("[INDEX] Building BM25 index...")

    tokenized_corpus = [
        tokenize(text)
        for text in texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    # Persist the corpus order alongside the BM25 object.
    with open(BM25_FILE, "wb") as file:
        pickle.dump(
            {
                "bm25": bm25,
                "chunk_ids": [
                    chunk["chunk_id"]
                    for chunk in chunks
                ]
            },
            file
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_file": str(
            SOURCE_FILE.relative_to(ROOT)
        ),
        "source_sha256": source_hash,
        "embedding_model": EMBEDDING_MODEL,
        "chunk_count": len(chunks),
        "curriculum_record_candidates": len(records),
        "vector_dimension": int(embeddings.shape[1]),
        "normalized_embeddings": True
    }

    save_json(
        MANIFEST_FILE,
        manifest
    )

    print("[INDEX] Persistent indexes saved.")


# ============================================================
# 10. LOAD PERSISTENT INDEXES
# ============================================================

def load_index():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"\nMarkdown source not found:\n{SOURCE_FILE}"
        )

    source_hash = sha256_file(
        SOURCE_FILE
    )

    print(
        f"[MODEL] Loading embedding model: "
        f"{EMBEDDING_MODEL}"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    if index_needs_rebuild(source_hash):
        print(
            "\n[INDEX] Missing, outdated, or forced rebuild."
        )

        markdown = SOURCE_FILE.read_text(
            encoding="utf-8"
        )

        build_index(
            markdown,
            source_hash,
            model
        )

    else:
        print(
            "[INDEX] Loading existing persistent indexes."
        )

    chunks = read_jsonl(
        CHUNKS_FILE
    )

    with open(
        RECORDS_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        records = json.load(file)

    vector_index = faiss.read_index(
        str(FAISS_FILE)
    )

    with open(
        BM25_FILE,
        "rb"
    ) as file:
        bm25_data = pickle.load(file)

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    # Validate unique IDs.
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(
            "Duplicate chunk IDs exist in the saved chunks. "
            "Rebuild the index."
        )

    # Validate FAISS and BM25 alignment.
    if chunk_ids != bm25_data["chunk_ids"]:
        raise RuntimeError(
            "Chunk IDs do not match the saved BM25 corpus. "
            "Set FORCE_REBUILD=True and rebuild."
        )

    if vector_index.ntotal != len(chunks):
        raise RuntimeError(
            "FAISS vector count does not match chunk count. "
            "Set FORCE_REBUILD=True and rebuild."
        )

    if vector_index.d != model.get_sentence_embedding_dimension():
        raise RuntimeError(
            "Embedding dimension mismatch. Rebuild the index."
        )

    return (
        model,
        chunks,
        records,
        vector_index,
        bm25_data["bm25"]
    )


# ============================================================
# 11. STRUCTURED LOOKUP
# ============================================================

def find_course_by_name(query, records):
    query_lower = query.lower()

    exact_matches = [
        record for record in records
        if record["course_name"].lower() in query_lower
    ]

    if exact_matches:
        return exact_matches

    query_tokens = set(tokenize(query))
    matches = []

    for record in records:
        name_tokens = set(
            tokenize(record["course_name"])
        )

        if (
            len(name_tokens) >= 2
            and name_tokens.issubset(query_tokens)
        ):
            matches.append(record)

    return matches


def structured_lookup(query, records):
    query_lower = query.lower()

    code_match = COURSE_CODE_PATTERN.search(query)

    if code_match:
        code = code_match.group(0).upper()

        candidates = [
            record for record in records
            if record["course_code"] == code
        ]

        if not candidates:
            return {
                "status": "not_found",
                "course_code": code,
                "message": (
                    "No matching curriculum candidate "
                    "was extracted."
                )
            }

        names = sorted({
            record["course_name"]
            for record in candidates
        })

        credits = sorted({
            record["credits"]
            for record in candidates
            if record.get("credits") is not None
        })

        if len(names) > 1 or len(credits) > 1:
            return {
                "status": "conflict",
                "course_code": code,
                "candidate_names": names,
                "candidate_credits": credits,
                "sources": [
                    {
                        "table": record["table_number"],
                        "row": record["row_number"],
                        "name": record["course_name"],
                        "credits": record.get("credits"),
                        "heading": record.get("heading")
                    }
                    for record in candidates
                ]
            }

        return {
            "status": "candidate",
            "course_code": code,
            "course_name": names[0],
            "credits": (
                credits[0]
                if credits
                else None
            ),
            "sources": [
                {
                    "table": record["table_number"],
                    "row": record["row_number"],
                    "heading": record.get("heading")
                }
                for record in candidates
            ]
        }

    named_records = find_course_by_name(
        query,
        records
    )

    asks_for_code = (
        "course code" in query_lower
        or "code for" in query_lower
        or "what is the code" in query_lower
    )

    asks_for_credits = "credit" in query_lower

    if named_records and (
        asks_for_code or asks_for_credits
    ):
        codes = sorted({
            record["course_code"]
            for record in named_records
        })

        names = sorted({
            record["course_name"]
            for record in named_records
        })

        credits = sorted({
            record["credits"]
            for record in named_records
            if record.get("credits") is not None
        })

        if (
            len(codes) == 1
            and len(names) == 1
            and len(credits) <= 1
        ):
            return {
                "status": "candidate",
                "course_code": codes[0],
                "course_name": names[0],
                "credits": (
                    credits[0]
                    if credits
                    else None
                ),
                "sources": [
                    {
                        "table": record["table_number"],
                        "row": record["row_number"],
                        "heading": record.get("heading")
                    }
                    for record in named_records
                ]
            }

        return {
            "status": "conflict",
            "candidate_codes": codes,
            "candidate_names": names,
            "candidate_credits": credits
        }

    return None


# ============================================================
# 12. HYBRID RETRIEVAL: FAISS + BM25 + RRF
# ============================================================

def reciprocal_rank_fusion(rankings, k=60):
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, chunk_id in enumerate(
            ranking,
            start=1
        ):
            scores[chunk_id] += (
                1.0 / (k + rank)
            )

    return scores


def ranking_prior(chunk, query):
    """
    Small ranking adjustment.

    This is not a substitute for correct extraction.
    It only prefers matching course records and table types.
    """

    query_lower = query.lower()
    score = 0.0

    code_match = COURSE_CODE_PATTERN.search(query)

    if code_match:
        code = code_match.group(0).upper()

        if code in chunk.get("course_codes", []):
            score += 0.20

    if chunk["chunk_type"] == "curriculum_record_candidate":
        score += 0.06

    semester_question = (
        "semester i" in query_lower
        or "semester 1" in query_lower
        or "first-year engineering" in query_lower
    )

    if semester_question:
        if chunk.get("table_kind") == "curriculum_candidate":
            score += 0.08

        if chunk.get("table_kind") == "outcomes_or_details":
            score -= 0.06

    if (
        "pre-requisite course codes"
        in chunk["text"].lower()
    ):
        score -= 0.08

    return score


def retrieve(
    query,
    model,
    chunks,
    vector_index,
    bm25
):
    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    # Dense retrieval.
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dense_scores, dense_indices = vector_index.search(
        query_embedding,
        min(DENSE_TOP_K, len(chunks))
    )

    dense_ranking = [
        chunks[int(index)]["chunk_id"]
        for index in dense_indices[0]
        if int(index) >= 0
    ]

    # Keyword retrieval.
    lexical_scores = bm25.get_scores(
        tokenize(query)
    )

    lexical_indices = np.argsort(
        lexical_scores
    )[::-1][:BM25_TOP_K]

    bm25_ranking = [
        chunks[int(index)]["chunk_id"]
        for index in lexical_indices
    ]

    # Combine the two rankings.
    rrf_scores = reciprocal_rank_fusion(
        [dense_ranking, bm25_ranking]
    )

    ranked_ids = sorted(
        rrf_scores,
        key=lambda chunk_id: (
            rrf_scores[chunk_id]
            + ranking_prior(
                chunk_by_id[chunk_id],
                query
            )
        ),
        reverse=True
    )

    results = []

    for chunk_id in ranked_ids[:FINAL_TOP_K]:
        chunk = chunk_by_id[chunk_id]

        results.append({
            "chunk": chunk,
            "rrf_score": round(
                rrf_scores[chunk_id],
                6
            ),
            "prior": round(
                ranking_prior(chunk, query),
                4
            )
        })

    return results


# ============================================================
# 13. PREPARE EVIDENCE FOR GROQ
# ============================================================

def build_evidence(retrieved, structured_result):
    evidence = []

    if structured_result is not None:
        evidence.append(
            "[STRUCTURED LOOKUP CANDIDATE]\n"
            + json.dumps(
                structured_result,
                ensure_ascii=False,
                indent=2
            )
        )

    for rank, result in enumerate(
        retrieved,
        start=1
    ):
        chunk = result["chunk"]

        evidence.append(
            f"[RETRIEVED CHUNK {rank}]\n"
            f"ID: {chunk['chunk_id']}\n"
            f"Type: {chunk['chunk_type']}\n"
            f"Heading: {chunk.get('heading')}\n"
            f"Table type: {chunk.get('table_kind')}\n"
            f"Source table: {chunk.get('source_table')}\n"
            f"Content:\n{chunk['text']}"
        )

    return "\n\n".join(evidence)


# ============================================================
# 14. GROQ GENERATION
# ============================================================

def generate_answer(query, evidence):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from your .env file."
        )

    client = Groq(
        api_key=api_key
    )

    system_prompt = """
You are an evidence-grounded academic syllabus assistant.

Use only the supplied evidence.

Rules:
1. Never invent a course code, course name, credit value,
   semester, subject, or other fact.
2. Extracted PDF text may contain errors or malformed tables.
3. A structured lookup is a candidate, not automatically verified truth.
4. If structured evidence contains conflicting values, explain the
   conflict instead of arbitrarily selecting one.
5. Do not use unrelated course-outcome rows to answer curriculum questions.
6. For exact course-code and credit questions, answer concisely if the
   evidence is consistent.
7. For a question asking for all subjects in a semester, list subjects
   only when supported by curriculum evidence. If the retrieved evidence
   appears incomplete, state that limitation.
8. If the evidence does not support an answer, say so.
9. Do not claim to have verified the original PDF unless verification
   information is explicitly supplied.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=900,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    f"Evidence:\n{evidence}\n\n"
                    "Answer using only the evidence."
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 15. DIAGNOSTICS
# ============================================================

def print_retrieved_evidence(results):
    print("\n" + "=" * 76)
    print("RETRIEVED EVIDENCE")
    print("=" * 76)

    for rank, result in enumerate(
        results,
        start=1
    ):
        chunk = result["chunk"]

        print(
            f"\nRank {rank} | "
            f"{chunk['chunk_id']} | "
            f"type={chunk['chunk_type']} | "
            f"table={chunk.get('source_table')} | "
            f"kind={chunk.get('table_kind')} | "
            f"RRF={result['rrf_score']} | "
            f"prior={result['prior']}"
        )

        print("-" * 76)
        print(chunk["text"][:1800])


def print_record_summary(records):
    print("\n" + "=" * 76)
    print("EXTRACTED CURRICULUM RECORD CANDIDATES")
    print("=" * 76)

    if not records:
        print("No curriculum record candidates were extracted.")
        return

    for record in records:
        print(
            f"{record['course_code']} | "
            f"{record['course_name']} | "
            f"Semester={record.get('semester')} | "
            f"Credits={record.get('credits')} | "
            f"Table={record['table_number']} | "
            f"Row={record['row_number']}"
        )


# ============================================================
# 16. MAIN INTERACTIVE LOOP
# ============================================================

def main():
    (
        model,
        chunks,
        records,
        vector_index,
        bm25
    ) = load_index()

    print("\n" + "=" * 76)
    print("SPIT ACADEMIC RAG")
    print("=" * 76)

    print(f"Source: {SOURCE_FILE.name}")
    print(f"Chunks: {len(chunks)}")
    print(f"Curriculum candidates: {len(records)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Groq model: {GROQ_MODEL}")

    print("\nCommands:")
    print("  records  - display extracted curriculum candidates")
    print("  rebuild  - rebuild persistent indexes")
    print("  exit     - quit")

    while True:
        query = input("\nYou: ").strip()

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit"
        }:
            break

        if query.lower() == "records":
            print_record_summary(records)
            continue

        if query.lower() == "rebuild":
            global FORCE_REBUILD

            FORCE_REBUILD = True

            (
                model,
                chunks,
                records,
                vector_index,
                bm25
            ) = load_index()

            FORCE_REBUILD = False

            print("Rebuild completed.")
            continue

        structured_result = structured_lookup(
            query,
            records
        )

        retrieved = retrieve(
            query,
            model,
            chunks,
            vector_index,
            bm25
        )

        print("\n" + "=" * 76)
        print("STRUCTURED LOOKUP")
        print("=" * 76)

        if structured_result is None:
            print("No applicable structured lookup.")
        else:
            print(json.dumps(
                structured_result,
                ensure_ascii=False,
                indent=2
            ))

        print_retrieved_evidence(
            retrieved
        )

        evidence = build_evidence(
            retrieved,
            structured_result
        )

        print("\n" + "=" * 76)
        print("FINAL ANSWER")
        print("=" * 76)

        try:
            answer = generate_answer(
                query,
                evidence
            )

            print(answer)

        except Exception as error:
            print(
                f"Groq generation error: {error}"
            )


if __name__ == "__main__":
    main()
