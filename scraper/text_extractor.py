from pathlib import Path
from datetime import datetime, timezone
import json

from pypdf import PdfReader
from tqdm import tqdm

from scraper.config import (
    RAW_PDFS_DIR,
    PROCESSED_PDFS_DIR,
    METADATA_DIR,
)

# Output files
EXTRACTION_METADATA_FILE = METADATA_DIR / "pdf_extraction_metadata.jsonl"
EXTRACTION_ERROR_FILE = METADATA_DIR / "pdf_extraction_errors.jsonl"


def extract_text_from_pdf(pdf_path):
    """
    Extract text page by page from a PDF.

    Returns:
        list: A list of dictionaries containing page numbers and text.
    """
    pages = []

    reader = PdfReader(str(pdf_path))

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:
            print(
                f"\nWarning: Could not extract page "
                f"{page_number} from {pdf_path.name}: {error}"
            )
            text = ""

        # Remove excessive blank lines while preserving paragraphs.
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        pages.append({
            "page_number": page_number,
            "text": text,
        })

    return pages


def save_jsonl(file_path, record):
    """Append one JSON record to a JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_pdf(pdf_path):
    """Extract one PDF and save its processed text and metadata."""
    extracted_at = datetime.now(timezone.utc).isoformat()

    try:
        pages = extract_text_from_pdf(pdf_path)

        # Keep page boundaries for later citations and debugging.
        output_text = []

        for page in pages:
            output_text.append(
                f"--- PAGE {page['page_number']} ---\n"
                f"{page['text']}"
            )

        combined_text = "\n\n".join(output_text)

        # Use the original filename so files remain identifiable.
        output_path = PROCESSED_PDFS_DIR / f"{pdf_path.stem}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(combined_text, encoding="utf-8")

        total_characters = sum(len(page["text"]) for page in pages)
        pages_with_text = sum(
            1 for page in pages if page["text"].strip()
        )

        record = {
            "source_id": pdf_path.stem,
            "source_type": "pdf",
            "source_filename": pdf_path.name,
            "local_path": str(pdf_path.resolve()),
            "processed_path": str(output_path.resolve()),
            "page_count": len(pages),
            "pages_with_text": pages_with_text,
            "text_characters": total_characters,
            "extracted_at": extracted_at,
            "status": (
                "success" if total_characters > 0
                else "no_extractable_text"
            ),
        }

        save_jsonl(EXTRACTION_METADATA_FILE, record)

        if total_characters == 0:
            save_jsonl(EXTRACTION_ERROR_FILE, {
                "source_filename": pdf_path.name,
                "error": "No extractable text; PDF may be scanned or empty.",
                "timestamp": extracted_at,
            })

        return record

    except Exception as error:
        save_jsonl(EXTRACTION_ERROR_FILE, {
            "source_filename": pdf_path.name,
            "error": str(error),
            "timestamp": extracted_at,
        })

        return {
            "source_filename": pdf_path.name,
            "status": "failed",
            "error": str(error),
        }


def extract_all_pdfs():
    """Process every PDF in the raw PDF directory."""
    PROCESSED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(
        path for path in RAW_PDFS_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )

    if not pdf_files:
        print(f"No PDFs found in: {RAW_PDFS_DIR}")
        return

    print("=" * 65)
    print("SPIT PDF TEXT EXTRACTION")
    print("=" * 65)
    print(f"PDFs found: {len(pdf_files)}")
    print(f"Output directory: {PROCESSED_PDFS_DIR}")
    print()

    results = []

    for pdf_path in tqdm(pdf_files, desc="Extracting PDFs"):
        results.append(process_pdf(pdf_path))

    successful = sum(
        result.get("status") == "success"
        for result in results
    )

    no_text = sum(
        result.get("status") == "no_extractable_text"
        for result in results
    )

    failed = sum(
        result.get("status") == "failed"
        for result in results
    )

    print("\n" + "=" * 65)
    print("EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"PDFs processed: {len(results)}")
    print(f"Successfully extracted: {successful}")
    print(f"No extractable text: {no_text}")
    print(f"Failed: {failed}")
    print(f"Processed text: {PROCESSED_PDFS_DIR}")
    print(f"Metadata: {EXTRACTION_METADATA_FILE}")
    print(f"Errors: {EXTRACTION_ERROR_FILE}")


if __name__ == "__main__":
    extract_all_pdfs()