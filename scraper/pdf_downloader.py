import time
from pathlib import Path
from urllib.parse import urlparse, unquote
from urllib.robotparser import RobotFileParser

import requests

from scraper.config import (
    ALLOWED_HOSTS,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    USER_AGENT,
    RAW_PDFS_DIR,
    METADATA_DIR,
)

from scraper.utils import (
    normalize_url,
    is_allowed_domain,
    generate_id,
    save_pdf_metadata,
    log_error,
)


# ============================================================
# CONFIGURATION
# ============================================================

PDF_LIST_FILE = METADATA_DIR / "discovered_pdfs.txt"

MAX_PDF_SIZE_MB = 50

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": USER_AGENT,
})

ROBOTS_CACHE = {}


# ============================================================
# DOCUMENT CATEGORIES
# ============================================================

CATEGORY_KEYWORDS = {
    "placements": [
        "placement",
        "placements",
        "recruiter",
        "recruiters",
        "recruitment",
        "training-placement",
        "training_and_placement",
        "tpo",
        "career",
        "internship",
        "package",
    ],

    "academics": [
        "syllabus",
        "curriculum",
        "academic-calendar",
        "academic_calendar",
        "exam-manual",
        "examination",
        "academic-regulation",
        "time-table",
        "timetable",
    ],

    "admissions": [
        "admission",
        "fee-approved",
        "fee-notice",
        "cap-round",
    ],

    "institutional": [
        "code-of-conduct",
        "roles-and-responsibilities",
        "annual-report",
        "mandatory-disclosure",
        "disclosure",
        "nirf",
        "naac",
        "student-handbook",
        "institutional",
        "orientation",
        "induction",
    ],
}


# Exclude irrelevant procurement material.
EXCLUDED_KEYWORDS = [
    "tender",
    "architectural-consultancy",
    "project-management-consultant",
    "liaison-consultant",
    "housekeeping",
    "hk-services",
]


# ============================================================
# ROBOTS.TXT
# ============================================================

def can_fetch(url):
    """
    Respect robots.txt for the URL's host.
    """

    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin not in ROBOTS_CACHE:
        robots_url = f"{origin}/robots.txt"

        try:
            response = SESSION.get(
                robots_url,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code != 200:
                ROBOTS_CACHE[origin] = None
            else:
                parser = RobotFileParser()
                parser.set_url(robots_url)
                parser.parse(response.text.splitlines())

                ROBOTS_CACHE[origin] = parser

        except requests.RequestException:
            ROBOTS_CACHE[origin] = None

    parser = ROBOTS_CACHE[origin]

    if parser is None:
        return False

    return parser.can_fetch(USER_AGENT, url)


# ============================================================
# CLASSIFY PDF
# ============================================================

def classify_pdf(url):
    """
    Classify a PDF using its filename.

    Returns a category or None if it is not a priority document.
    """

    filename = unquote(
        Path(urlparse(url).path).name
    ).lower()

    if not filename.endswith(".pdf"):
        return None

    if any(
        keyword in filename
        for keyword in EXCLUDED_KEYWORDS
    ):
        return None

    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(
            keyword in filename
            for keyword in keywords
        ):
            return category

    return None


# ============================================================
# DOWNLOAD PDF
# ============================================================

def download_pdf(url, category):
    """
    Download one relevant PDF and save its metadata.
    """

    url = normalize_url(url)

    if not url or not is_allowed_domain(url):
        print(f"[SKIP] Unapproved URL: {url}")
        return False

    source_id = generate_id(url)

    original_filename = Path(
        unquote(urlparse(url).path)
    ).name

    filename = f"{source_id}_{original_filename}"

    destination = RAW_PDFS_DIR / filename

    # Avoid downloading the same document again.
    if destination.exists():
        print(f"[EXISTS] {original_filename}")
        return True

    if not can_fetch(url):
        print(f"[SKIP] Robots policy unavailable or disallows: {url}")
        return False

    try:
        time.sleep(REQUEST_DELAY)

        with SESSION.get(
            url,
            timeout=REQUEST_TIMEOUT,
            stream=True,
            allow_redirects=True,
        ) as response:

            response.raise_for_status()

            # Prevent redirects to unapproved domains.
            if not is_allowed_domain(response.url):
                print(f"[SKIP] Unapproved redirect: {url}")
                return False

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            # Reject responses that do not look like PDFs.
            if (
                "application/pdf" not in content_type
                and not response.url.lower().endswith(".pdf")
            ):
                print(f"[SKIP] Not identified as a PDF: {url}")
                return False

            max_bytes = MAX_PDF_SIZE_MB * 1024 * 1024

            content_length = int(
                response.headers.get("Content-Length", 0)
            )

            if content_length > max_bytes:
                print(f"[SKIP] File exceeds size limit: {url}")
                return False

            total_bytes = 0

            with destination.open("wb") as file:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if not chunk:
                        continue

                    total_bytes += len(chunk)

                    if total_bytes > max_bytes:
                        raise ValueError(
                            "PDF exceeds the configured size limit"
                        )

                    file.write(chunk)

        # Verify the downloaded file's PDF signature.
        with destination.open("rb") as file:
            signature = file.read(5)

        if signature != b"%PDF-":
            destination.unlink(missing_ok=True)

            print(f"[ERROR] Invalid PDF: {url}")
            return False

        save_pdf_metadata(
            source_url=url,
            local_path=destination,
            size_bytes=destination.stat().st_size,
            title=original_filename,
        )

        print(
            f"[DOWNLOADED] [{category}] "
            f"{original_filename}"
        )

        return True

    except (
        requests.RequestException,
        OSError,
        ValueError,
    ) as exc:

        destination.unlink(missing_ok=True)

        log_error(
            url=url,
            error=exc,
        )

        print(f"[ERROR] {url}: {exc}")

        return False


# ============================================================
# MAIN DOWNLOADER
# ============================================================

def download_all_pdfs():
    """
    Read discovered PDF URLs and download priority documents.
    """

    if not PDF_LIST_FILE.exists():
        print(
            "PDF URL list not found. "
            "Run the website crawler first."
        )
        return

    urls = {
        line.strip()
        for line in PDF_LIST_FILE.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    }

    print("=" * 65)
    print("SPIT PDF DOWNLOADER")
    print("=" * 65)

    print(f"Discovered PDF URLs: {len(urls)}")

    selected = []

    for url in sorted(urls):
        if not is_allowed_domain(url):
            continue

        category = classify_pdf(url)

        if category:
            selected.append((url, category))

    print(f"Priority PDFs selected: {len(selected)}")

    category_counts = {}

    for _, category in selected:
        category_counts[category] = (
            category_counts.get(category, 0) + 1
        )

    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    successful = 0

    for index, (url, category) in enumerate(
        selected,
        start=1,
    ):
        print(f"\n[{index}/{len(selected)}]")

        if download_pdf(url, category):
            successful += 1

    print("\n" + "=" * 65)
    print("DOWNLOAD SUMMARY")
    print("=" * 65)

    print(f"Priority PDFs selected: {len(selected)}")
    print(f"Successful downloads: {successful}")
    print(f"PDF directory: {RAW_PDFS_DIR}")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    download_all_pdfs()