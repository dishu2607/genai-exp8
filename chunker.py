import re
import json
from pathlib import Path
from tqdm import tqdm
import tiktoken


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CLEANED_PDF_DIR = (
    BASE_DIR / "data" / "processed" / "cleaned" / "pdfs"
)

CLEANED_WEBPAGE_DIR = (
    BASE_DIR / "data" / "processed" / "cleaned" / "webpages"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "chunks.jsonl"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

# Tokenizer used for measuring chunk length
TOKENIZER = tiktoken.get_encoding("cl100k_base")


# ============================================================
# TOKEN COUNTING
# ============================================================

def count_tokens(text):
    """Return the number of tokens in a text string."""
    return len(TOKENIZER.encode(text))


# ============================================================
# DOMAIN CLASSIFICATION
# ============================================================

def classify_domain(source, text):
    """
    Classify a chunk as academic or placement.

    This is a lightweight heuristic, not a trained classifier.
    """

    combined = f"{source} {text}".lower()

    placement_terms = [
        "placement",
        "placed",
        "ctc",
        "recruiter",
        "higher studies",
        "industrial internship",
        "research internship",
        "training and placement",
        "placement statistics",
        "placement report",
        "campus recruitment",
        "package offered",
        "company visited",
    ]

    academic_terms = [
        "academic",
        "curriculum",
        "syllabus",
        "semester",
        "admission",
        "programme",
        "program",
        "regulation",
        "examination",
        "course",
        "faculty",
        "eligibility",
        "attendance",
        "credits",
        "academic calendar",
    ]

    placement_score = sum(
        combined.count(term)
        for term in placement_terms
    )

    academic_score = sum(
        combined.count(term)
        for term in academic_terms
    )

    if placement_score > academic_score:
        return "placement"

    return "academic"


# ============================================================
# PDF PAGE EXTRACTION
# ============================================================

def extract_page_sections(text):
    """
    Split PDF text using existing page markers.

    Returns:
        [(page_number, page_text), ...]

    Page numbers are preserved for source citations.
    """

    pattern = r"--- PAGE (\d+) ---"
    matches = list(re.finditer(pattern, text))

    if not matches:
        return [(None, text.strip())] if text.strip() else []

    sections = []

    # Preserve any text before the first page marker.
    prefix = text[:matches[0].start()].strip()

    if prefix:
        sections.append((None, prefix))

    for index, match in enumerate(matches):

        start = match.end()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(text)

        page_number = int(match.group(1))
        page_text = text[start:end].strip()

        if page_text:
            sections.append((page_number, page_text))

    return sections


# ============================================================
# WEBPAGE METADATA
# ============================================================

def get_webpage_metadata(text, filename):
    """
    Extract title and source URL from webpage text.
    """

    title = filename
    source_url = None

    for line in text.splitlines()[:15]:

        line = line.strip()

        if line.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()

        elif line.startswith("SOURCE_URL:"):
            source_url = line.split(":", 1)[1].strip()

    return title, source_url


# ============================================================
# TEXT PREPARATION
# ============================================================

def split_into_sentences(text):
    """
    Split text at likely sentence boundaries.

    This is a lightweight rule-based splitter. It does not
    guarantee perfect sentence detection for every document.
    """

    text = text.strip()

    if not text:
        return []

    # Keep paragraph structure meaningful while normalizing
    # excessive whitespace within individual paragraphs.
    text = re.sub(r"[ \t]+", " ", text)

    # Avoid splitting decimal numbers such as 8.5 or 2025.26.
    sentence_pattern = (
        r'(?<=[.!?])\s+(?=[A-Z0-9"“(\[])'
    )

    sentences = re.split(sentence_pattern, text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_oversized_sentence(sentence, chunk_size, overlap):
    """
    Split a sentence that exceeds the token limit.

    Token windows are a fallback for unusually long sentences,
    table rows, URLs, or malformed extracted text.
    """

    tokens = TOKENIZER.encode(sentence)

    if len(tokens) <= chunk_size:
        return [sentence.strip()] if sentence.strip() else []

    pieces = []
    start = 0

    while start < len(tokens):

        end = min(start + chunk_size, len(tokens))

        piece = TOKENIZER.decode(
            tokens[start:end]
        ).strip()

        if piece:
            pieces.append(piece)

        if end == len(tokens):
            break

        start = end - overlap

    return pieces


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):
    """
    Create chunks while preserving meaningful text boundaries.

    Strategy:
    1. Split the text into paragraphs.
    2. Split oversized paragraphs into sentences.
    3. Split oversized sentences using token windows.
    4. Combine adjacent units when they fit.
    5. Apply limited overlap where complete units fit.
    6. Enforce the final token limit.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be positive."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be non-negative and smaller than chunk size."
        )

    if not text or not text.strip():
        return []

    # --------------------------------------------------------
    # Step 1: Split into paragraphs.
    # --------------------------------------------------------

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(
            r"\n\s*\n",
            text
        )
        if paragraph.strip()
    ]

    units = []

    # --------------------------------------------------------
    # Step 2: Prepare units that fit within the token limit.
    # --------------------------------------------------------

    for paragraph in paragraphs:

        if count_tokens(paragraph) <= chunk_size:
            units.append(paragraph)
            continue

        sentences = split_into_sentences(paragraph)

        for sentence in sentences:

            if count_tokens(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(
                    split_oversized_sentence(
                        sentence,
                        chunk_size,
                        overlap
                    )
                )

    # --------------------------------------------------------
    # Step 3: Combine units into chunks.
    # --------------------------------------------------------

    chunks = []
    current_parts = []

    for unit in units:

        candidate_parts = current_parts + [unit]
        candidate = "\n\n".join(candidate_parts)

        if count_tokens(candidate) <= chunk_size:
            current_parts.append(unit)
            continue

        # Save the completed chunk.
        if current_parts:
            chunks.append(
                "\n\n".join(current_parts).strip()
            )

        # Select complete trailing units for overlap.
        overlap_parts = []

        for previous in reversed(current_parts):

            candidate_overlap = "\n\n".join(
                [previous] + overlap_parts
            )

            if count_tokens(candidate_overlap) > overlap:
                break

            overlap_parts.insert(0, previous)

        current_parts = overlap_parts

        # If overlap prevents the next unit from fitting,
        # discard the overlap and start a new chunk.
        candidate = "\n\n".join(
            current_parts + [unit]
        )

        if count_tokens(candidate) <= chunk_size:
            current_parts.append(unit)

        else:
            if current_parts:
                chunks.append(
                    "\n\n".join(current_parts).strip()
                )

            current_parts = [unit]

    if current_parts:
        chunks.append(
            "\n\n".join(current_parts).strip()
        )

    # --------------------------------------------------------
    # Step 4: Validate final chunks.
    # --------------------------------------------------------

    final_chunks = []

    for chunk in chunks:

        chunk = chunk.strip()

        if not chunk:
            continue

        tokens = TOKENIZER.encode(chunk)

        if len(tokens) <= chunk_size:
            final_chunks.append(chunk)
            continue

        # Defensive fallback.
        start = 0

        while start < len(tokens):

            end = min(
                start + chunk_size,
                len(tokens)
            )

            piece = TOKENIZER.decode(
                tokens[start:end]
            ).strip()

            if piece:
                final_chunks.append(piece)

            if end == len(tokens):
                break

            start = end - overlap

    return final_chunks


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(file_path):
    """
    Process one cleaned PDF text file.

    Each chunk retains its source filename and page number.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    page_sections = extract_page_sections(text)

    records = []
    chunk_index = 0

    for page_number, page_text in page_sections:

        chunks = split_into_chunks(page_text)

        for chunk in chunks:

            records.append({
                "chunk_id": (
                    f"{file_path.stem}_chunk_{chunk_index}"
                ),
                "text": chunk,
                "source_file": file_path.name,
                "document_type": "pdf",
                "domain": classify_domain(
                    file_path.name,
                    chunk
                ),
                "page_number": page_number,
                "source_url": None,
                "token_count": count_tokens(chunk),
            })

            chunk_index += 1

    return records


# ============================================================
# WEBPAGE PROCESSING
# ============================================================

def process_webpage(file_path):
    """
    Process one cleaned webpage text file.

    Each chunk retains its title and original source URL.
    """

    text = file_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    title, source_url = get_webpage_metadata(
        text,
        file_path.name
    )

    # Remove metadata headers from the page content.
    content_lines = [
        line
        for line in text.splitlines()
        if not line.strip().startswith("TITLE:")
        and not line.strip().startswith("SOURCE_URL:")
    ]

    content = "\n".join(content_lines).strip()

    chunks = split_into_chunks(content)

    records = []

    for index, chunk in enumerate(chunks):

        records.append({
            "chunk_id": (
                f"{file_path.stem}_chunk_{index}"
            ),
            "text": chunk,
            "source_file": file_path.name,
            "document_type": "webpage",
            "domain": classify_domain(
                title,
                chunk
            ),
            "page_number": None,
            "source_url": source_url,
            "title": title,
            "token_count": count_tokens(chunk),
        })

    return records


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("SPIT ACADEMIC & PLACEMENT CHATBOT — CHUNKING")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = sorted(
        CLEANED_PDF_DIR.glob("*.txt")
    )

    webpage_files = sorted(
        CLEANED_WEBPAGE_DIR.glob("*.txt")
    )

    print(f"\nCleaned PDF files: {len(pdf_files)}")
    print(f"Cleaned webpage files: {len(webpage_files)}")

    all_records = []

    # --------------------------------------------------------
    # Process PDFs.
    # --------------------------------------------------------

    print("\nProcessing PDF text files...")

    for file_path in tqdm(pdf_files):

        try:
            records = process_pdf(file_path)
            all_records.extend(records)

        except Exception as error:
            print(
                f"\nError processing PDF {file_path.name}: {error}"
            )

    # --------------------------------------------------------
    # Process webpages.
    # --------------------------------------------------------

    print("\nProcessing webpage text files...")

    for file_path in tqdm(webpage_files):

        try:
            records = process_webpage(file_path)
            all_records.extend(records)

        except Exception as error:
            print(
                f"\nError processing webpage {file_path.name}: {error}"
            )

    # --------------------------------------------------------
    # Write JSONL output.
    # --------------------------------------------------------

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for record in all_records:
            output_file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------------
    # Report statistics.
    # --------------------------------------------------------

    token_counts = [
        record["token_count"]
        for record in all_records
    ]

    oversized_chunks = [
        record
        for record in all_records
        if record["token_count"] > CHUNK_SIZE
    ]

    missing_sources = [
        record
        for record in all_records
        if not record.get("source_file")
    ]

    print("\n" + "=" * 70)
    print("CHUNKING COMPLETE")
    print("=" * 70)

    print(f"PDF files processed: {len(pdf_files)}")
    print(f"Webpage files processed: {len(webpage_files)}")
    print(f"Total chunks: {len(all_records)}")

    if token_counts:

        print(f"Minimum tokens: {min(token_counts)}")
        print(f"Maximum tokens: {max(token_counts)}")
        print(
            f"Average tokens: "
            f"{sum(token_counts) / len(token_counts):.2f}"
        )

    print(
        f"Chunks exceeding {CHUNK_SIZE} tokens: "
        f"{len(oversized_chunks)}"
    )

    print(
        f"Chunks missing source filenames: "
        f"{len(missing_sources)}"
    )

    print(f"Output: {OUTPUT_FILE}")

    print("\nDomain distribution:")

    domain_counts = {}

    for record in all_records:

        domain = record["domain"]

        domain_counts[domain] = (
            domain_counts.get(domain, 0) + 1
        )

    for domain, count in sorted(domain_counts.items()):

        print(f"  {domain}: {count}")

    if oversized_chunks or missing_sources:

        print(
            "\nWARNING: Some validation checks failed. "
            "Inspect the output before embedding."
        )

    else:

        print(
            "\nBasic validation passed. "
            "Chunk quality still requires manual inspection."
        )


if __name__ == "__main__":
    main()