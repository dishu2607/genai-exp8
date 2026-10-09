import time
import logging

from collections import deque
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from scraper.config import (
    BASE_URL,
    ALLOWED_HOSTS,
    MAX_PAGES,
    REQUEST_DELAY,
    REQUEST_TIMEOUT,
    USER_AGENT,
    RAW_WEBPAGES_DIR,
    WEBPAGE_METADATA_FILE,
    METADATA_DIR,
)

from scraper.utils import (
    normalize_url,
    is_allowed_domain,
    generate_id,
    save_webpage_metadata,
    save_metadata,
    log_error,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Begin with a small test crawl.
TEST_PAGE_LIMIT = 10

# Only collect pages relevant to the SPIT knowledge base.
RELEVANT_KEYWORDS = {
    "about",
    "academic",
    "academics",
    "admission",
    "admissions",
    "department",
    "computer",
    "engineering",
    "curriculum",
    "syllabus",
    "placement",
    "faculty",
    "research",
    "student",
    "activities",
    "infrastructure",
    "facility",
    "facilities",
    "library",
    "mandatory",
    "disclosure",
    "regulation",
    "rules",
    "contact",
    "programme",
    "program",
    "notice",
    "download",
    "naac",
    "nirf",
}

# Avoid crawling unnecessary or potentially sensitive paths.
EXCLUDED_PATH_KEYWORDS = {
    "wp-admin",
    "wp-login",
    "login",
    "logout",
    "register",
    "cart",
    "checkout",
    "feed",
    "author",
    "tag",
    "search",
}

SESSION = requests.Session()

SESSION.headers.update({
    "User-Agent": USER_AGENT,
})

ROBOTS_CACHE = {}


# ============================================================
# 2. ROBOTS.TXT CHECK
# ============================================================

def get_robots_parser(url):
    """
    Retrieve robots.txt rules for the relevant website host.

    If robots.txt cannot be checked, do not crawl that host.
    """

    parsed = urlparse(url)
    origin = f"{parsed.scheme}://{parsed.netloc}"

    if origin in ROBOTS_CACHE:
        return ROBOTS_CACHE[origin]

    robots_url = urljoin(origin, "/robots.txt")

    try:
        response = SESSION.get(
            robots_url,
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code != 200:
            ROBOTS_CACHE[origin] = None
            return None

        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())

        ROBOTS_CACHE[origin] = parser

        return parser

    except requests.RequestException:
        ROBOTS_CACHE[origin] = None
        return None


def can_fetch(url):
    """Check whether the crawler may fetch a URL."""

    parser = get_robots_parser(url)

    if parser is None:
        return False

    return parser.can_fetch(USER_AGENT, url)


# ============================================================
# 3. FETCH A WEBPAGE
# ============================================================

def fetch_page(url):
    """
    Fetch a page only if its host and crawling rules permit it.
    """

    if not is_allowed_domain(url):
        return None

    if not can_fetch(url):
        print(f"[SKIPPED] Robots policy unavailable or disallows: {url}")
        return None

    try:
        time.sleep(REQUEST_DELAY)

        response = SESSION.get(
            url,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

        response.raise_for_status()

        # Never continue onto an unapproved redirect destination.
        if not is_allowed_domain(response.url):
            print(f"[SKIPPED] Redirect outside approved domain: {url}")
            return None

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if "text/html" not in content_type:
            return None

        return response

    except requests.RequestException as exc:
        status_code = getattr(
            getattr(exc, "response", None),
            "status_code",
            None,
        )

        log_error(
            url=url,
            error=exc,
            status_code=status_code,
        )

        print(f"[ERROR] {url}: {exc}")

        return None


# ============================================================
# 4. CHECK PAGE RELEVANCE
# ============================================================

def is_relevant_url(url):
    """
    Check whether a URL looks relevant to the SPIT knowledge base.
    """

    path = urlparse(url).path.lower()

    if any(
        keyword in path
        for keyword in EXCLUDED_PATH_KEYWORDS
    ):
        return False

    # Always allow the homepage.
    if path in {"", "/"}:
        return True

    return any(
        keyword in path
        for keyword in RELEVANT_KEYWORDS
    )


# ============================================================
# 5. EXTRACT TEXT AND LINKS
# ============================================================

def extract_content(html, page_url):
    """
    Extract readable webpage text and discover linked URLs.
    """

    soup = BeautifulSoup(html, "html.parser")

    title = (
        soup.title.get_text(" ", strip=True)
        if soup.title
        else "Untitled Page"
    )

    # Remove scripts and non-content elements.
    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "iframe",
    ]):
        element.decompose()

    # Remove repeated page layout elements.
    for element in soup.find_all([
        "nav",
        "header",
        "footer",
    ]):
        element.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id="content")
        or soup.find(class_="site-content")
        or soup.body
        or soup
    )

    # Preserve headings, paragraphs and list content.
    text_blocks = []

    for element in main_content.find_all([
        "h1",
        "h2",
        "h3",
        "h4",
        "p",
        "li",
        "dt",
        "dd",
        "tr",
    ]):
        text = element.get_text(
            " ",
            strip=True,
        )

        if text:
            text_blocks.append(text)

    page_text = "\n".join(text_blocks)

    # Discover links while preserving their anchor text.
    links = []

    for anchor in soup.find_all("a", href=True):
        absolute_url = normalize_url(
            urljoin(page_url, anchor["href"])
        )

        if absolute_url:
            links.append({
                "url": absolute_url,
                "anchor_text": anchor.get_text(
                    " ",
                    strip=True,
                ),
            })

    return title, page_text, links


# ============================================================
# 6. SAVE WEBPAGE CONTENT
# ============================================================

def save_page(url, title, text):
    """Save extracted webpage text and metadata."""

    source_id = generate_id(url)

    filename = f"{source_id}.txt"

    file_path = RAW_WEBPAGES_DIR / filename

    content = (
        f"TITLE: {title}\n"
        f"SOURCE_URL: {url}\n\n"
        f"{text}"
    )

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    save_webpage_metadata(
        url=url,
        title=title,
        local_path=file_path,
        text_characters=len(text),
    )

    return file_path


# ============================================================
# 7. MAIN CRAWLER
# ============================================================

def crawl_website(
    start_url=BASE_URL,
    max_pages=TEST_PAGE_LIMIT,
):
    """
    Crawl relevant public HTML pages on the main SPIT domain.

    PDF downloading is handled separately.
    """

    start_url = normalize_url(start_url)

    if not start_url or not is_allowed_domain(start_url):
        raise ValueError(
            "The starting URL is not an approved SPIT URL."
        )

    queue = deque([start_url])

    queued_urls = {start_url}
    visited_urls = set()
    pdf_urls = set()

    print("=" * 65)
    print("SPIT WEBSITE CRAWLER")
    print("=" * 65)

    while queue and len(visited_urls) < max_pages:
        current_url = queue.popleft()

        if current_url in visited_urls:
            continue

        visited_urls.add(current_url)

        print(
            f"\n[{len(visited_urls)}/{max_pages}] "
            f"Checking: {current_url}"
        )

        response = fetch_page(current_url)

        if response is None:
            continue

        # Use the final URL after any permitted redirect.
        final_url = normalize_url(response.url)

        title, text, links = extract_content(
            response.text,
            final_url,
        )

        # Save the homepage and relevant pages.
        if text.strip() and (
            is_relevant_url(final_url)
            or final_url == start_url
        ):
            save_page(
                url=final_url,
                title=title,
                text=text,
            )

            print(f"[SAVED] {title}")

        # Inspect discovered links.
        for link in links:
            linked_url = link["url"]

            if not is_allowed_domain(linked_url):
                continue

            path = urlparse(
                linked_url
            ).path.lower()

            # Collect PDF URLs without downloading them yet.
            if path.endswith(".pdf"):
                pdf_urls.add(linked_url)
                continue

            # Only queue relevant pages.
            if not is_relevant_url(linked_url):
                continue

            if (
                linked_url not in queued_urls
                and linked_url not in visited_urls
            ):
                queue.append(linked_url)
                queued_urls.add(linked_url)

    # Save the PDF discovery list for the next module.
    pdf_list_path = METADATA_DIR / "discovered_pdfs.txt"

    pdf_list_path.write_text(
        "\n".join(sorted(pdf_urls)),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("CRAWL SUMMARY")
    print("=" * 65)

    print(f"HTML URLs visited: {len(visited_urls)}")
    print(f"HTML URLs remaining in queue: {len(queue)}")
    print(f"PDF URLs discovered: {len(pdf_urls)}")

    print(f"Webpage output: {RAW_WEBPAGES_DIR}")
    print(f"PDF link list: {pdf_list_path}")

    print("\nTest crawl completed.")


# ============================================================
# 8. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    crawl_website()