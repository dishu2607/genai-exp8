from pathlib import Path

# ==================================================
# 1. PROJECT PATHS
# ==================================================

# Root directory: genai-exp8/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


# ==================================================
# 2. RAW DATA DIRECTORIES
# ==================================================

RAW_WEBPAGES_DIR = RAW_DIR / "webpages"
RAW_PDFS_DIR = RAW_DIR / "pdfs"


# ==================================================
# 3. PROCESSED DATA DIRECTORIES
# ==================================================

PROCESSED_WEBPAGES_DIR = PROCESSED_DIR / "webpages"
PROCESSED_PDFS_DIR = PROCESSED_DIR / "pdfs"


# ==================================================
# 4. WEBSITE CONFIGURATION
# ==================================================

BASE_URL = "https://www.spit.ac.in/"

# Crawl only the main SPIT website for now.
ALLOWED_HOSTS = {
    "www.spit.ac.in",
    "spit.ac.in",
    "placements.spit.ac.in",
}


# ==================================================
# 5. CRAWLER SETTINGS
# ==================================================

# Maximum number of HTML pages to visit.
MAX_PAGES = 150

# Delay between requests to avoid aggressive crawling.
REQUEST_DELAY = 1.0

# Maximum time to wait for an HTTP response.
REQUEST_TIMEOUT = 15

# Identify the crawler honestly.
USER_AGENT = (
    "SPIT-RAG-EducationalBot/1.0 "
    "(educational research)"
)


# ==================================================
# 6. METADATA FILES
# ==================================================

WEBPAGE_METADATA_FILE = (
    METADATA_DIR / "webpages.jsonl"
)

PDF_METADATA_FILE = (
    METADATA_DIR / "pdfs.jsonl"
)

ERROR_LOG_FILE = (
    METADATA_DIR / "errors.jsonl"
)


# ==================================================
# 7. CREATE REQUIRED DIRECTORIES
# ==================================================

ALL_DIRECTORIES = [
    RAW_WEBPAGES_DIR,
    RAW_PDFS_DIR,
    PROCESSED_WEBPAGES_DIR,
    PROCESSED_PDFS_DIR,
    METADATA_DIR,
]

for directory in ALL_DIRECTORIES:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ==================================================
# 8. DISPLAY CONFIGURATION
# ==================================================

if __name__ == "__main__":
    print("SPIT RAG Scraper Configuration")
    print("-" * 40)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Website: {BASE_URL}")
    print(f"Maximum pages: {MAX_PAGES}")
    print(f"Request delay: {REQUEST_DELAY}s")

    print("\nConfigured directories:")

    for directory in ALL_DIRECTORIES:
        print(f"  {directory}")

    print("\nConfiguration loaded successfully!")