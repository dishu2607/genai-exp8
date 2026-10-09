import json
import hashlib
import re

from datetime import datetime, timezone
from urllib.parse import urlparse, urldefrag, unquote

from scraper.config import (
    ALLOWED_HOSTS,
    METADATA_DIR,
    WEBPAGE_METADATA_FILE,
    PDF_METADATA_FILE,
    ERROR_LOG_FILE,
)


# ==================================================
# 1. URL NORMALIZATION
# ==================================================

def normalize_url(url):
    """
    Remove URL fragments and normalize basic URL formatting.
    Return None for unsupported or invalid URLs.
    """

    if not url:
        return None

    url = url.strip()

    # Remove fragments such as #about or #contact.
    url, _ = urldefrag(url)

    parsed = urlparse(url)

    # Allow only HTTP and HTTPS.
    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    if not parsed.hostname:
        return None

    # Reject URLs containing embedded credentials.
    if parsed.username or parsed.password:
        return None

    # Normalize the hostname and scheme.
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname.lower()

    # Preserve non-default ports.
    netloc = hostname

    if parsed.port and not (
        (scheme == "https" and parsed.port == 443)
        or (scheme == "http" and parsed.port == 80)
    ):
        netloc = f"{hostname}:{parsed.port}"

    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc,
    ).geturl()

    return normalized


# ==================================================
# 2. DOMAIN VALIDATION
# ==================================================

def is_allowed_domain(url):
    """
    Check whether the URL belongs to an approved SPIT host.
    """

    normalized = normalize_url(url)

    if not normalized:
        return False

    hostname = urlparse(normalized).hostname

    return hostname in ALLOWED_HOSTS


# ==================================================
# 3. URL IDENTIFIER
# ==================================================

def generate_id(value):
    """
    Generate a stable identifier using SHA-256.
    """

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()[:20]


# ==================================================
# 4. SAFE FILENAME GENERATION
# ==================================================

def safe_filename(value, max_length=120):
    """
    Convert a title or URL-derived name into a safe filename.
    """

    value = unquote(str(value))

    # Replace filesystem-sensitive characters.
    value = re.sub(
        r'[<>:"/\\|?*\x00-\x1f]',
        "_",
        value,
    )

    # Replace repeated whitespace.
    value = re.sub(r"\s+", "_", value)

    # Remove unsafe leading/trailing characters.
    value = value.strip("._ ")

    if not value:
        value = "document"

    return value[:max_length]


# ==================================================
# 5. JSONL METADATA WRITER
# ==================================================

def save_metadata(file_path, record):
    """
    Append one metadata record to a JSONL file.

    Each line contains one independent JSON object.
    """

    file_path = file_path if hasattr(
        file_path, "parent"
    ) else METADATA_DIR / file_path

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "a",
        encoding="utf-8",
    ) as file:

        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            ) + "\n"
        )


# ==================================================
# 6. WEBPAGE METADATA
# ==================================================

def save_webpage_metadata(
    url,
    title,
    local_path,
    text_characters=0,
):
    """
    Save metadata for a scraped webpage.
    """

    record = {
        "source_id": generate_id(url),
        "source_type": "webpage",
        "title": title,
        "source_url": url,
        "local_path": str(local_path),
        "text_characters": text_characters,
        "retrieved_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_metadata(
        WEBPAGE_METADATA_FILE,
        record,
    )


# ==================================================
# 7. PDF METADATA
# ==================================================

def save_pdf_metadata(
    source_url,
    local_path,
    size_bytes=0,
    title="",
):
    """
    Save metadata for a downloaded PDF.
    """

    record = {
        "source_id": generate_id(source_url),
        "source_type": "pdf",
        "title": title,
        "source_url": source_url,
        "local_path": str(local_path),
        "size_bytes": size_bytes,
        "downloaded_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_metadata(
        PDF_METADATA_FILE,
        record,
    )


# ==================================================
# 8. ERROR LOGGING
# ==================================================

def log_error(url, error, status_code=None):
    """
    Record a scraping or downloading error.
    """

    record = {
        "url": url,
        "error": str(error),
        "status_code": status_code,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    save_metadata(
        ERROR_LOG_FILE,
        record,
    )