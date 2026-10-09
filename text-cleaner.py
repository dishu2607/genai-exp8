import re
from pathlib import Path
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_TEXT_DIR = BASE_DIR / "data" / "processed" / "pdfs"
WEBPAGE_TEXT_DIR = BASE_DIR / "data" / "raw" / "webpages"

CLEANED_PDF_DIR = BASE_DIR / "data" / "processed" / "cleaned" / "pdfs"
CLEANED_WEBPAGE_DIR = BASE_DIR / "data" / "processed" / "cleaned" / "webpages"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """Clean extracted text while preserving meaningful structure."""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove null characters and common replacement characters
    text = text.replace("\x00", "")
    text = text.replace("\ufffd", "")

    # Repair words split by a hyphen at a line break.
    # Example: place-
    #          ment -> placement
    text = re.sub(
        r"(?<=[A-Za-z])-\s*\n\s*(?=[a-z])",
        "",
        text
    )

    # Normalize whitespace within individual lines
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Remove excessive consecutive blank lines
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove repeated spaces around punctuation
    text = re.sub(r" +([,.;:!?])", r"\1", text)

    return text.strip() + "\n"


# ============================================================
# FILE PROCESSING
# ============================================================

def process_directory(source_dir, output_dir, label):
    """Clean all text files in a directory without modifying originals."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)

    if not source_dir.exists():
        print(f"\n[{label}] Source directory not found: {source_dir}")
        return

    files = sorted(source_dir.glob("*.txt"))
    output_dir.mkdir(parents=True, exist_ok=True)

    successful = 0
    empty = 0
    failed = 0

    print(f"\n{'=' * 65}")
    print(f"CLEANING {label.upper()}")
    print(f"{'=' * 65}")
    print(f"Input directory:  {source_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Files found:      {len(files)}\n")

    for file_path in tqdm(files, desc=f"Cleaning {label}"):
        try:
            original = file_path.read_text(
                encoding="utf-8",
                errors="replace"
            )

            cleaned = clean_text(original)

            output_path = output_dir / file_path.name
            output_path.write_text(cleaned, encoding="utf-8")

            if cleaned.strip():
                successful += 1
            else:
                empty += 1

        except Exception as exc:
            failed += 1
            print(f"\nError processing {file_path.name}: {exc}")

    print(f"\nFiles processed: {successful}")
    print(f"Empty files:     {empty}")
    print(f"Failed files:    {failed}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("SPIT ACADEMIC & PLACEMENT CHATBOT — TEXT CLEANING")
    print("=" * 65)

    # Clean extracted PDF text
    process_directory(
        PDF_TEXT_DIR,
        CLEANED_PDF_DIR,
        "PDF Documents"
    )

    # Clean scraped webpage text
    process_directory(
        WEBPAGE_TEXT_DIR,
        CLEANED_WEBPAGE_DIR,
        "Webpages"
    )

    print("\n" + "=" * 65)
    print("TEXT CLEANING COMPLETED")
    print("=" * 65)
    print(f"Cleaned PDFs:     {CLEANED_PDF_DIR}")
    print(f"Cleaned webpages: {CLEANED_WEBPAGE_DIR}")