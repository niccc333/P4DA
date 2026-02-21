"""
seminar_10q.py
==============
A seminar walkthrough for parsing SEC 10-Q filings with Python.

Prerequisites (install once):
    pip install sec-parser requests rich google-generativeai

Usage:
    export GOOGLE_API_KEY="your-gemini-api-key"
    python seminar_10q.py
"""

import os
import json
import warnings
import requests
import sec_parser as sp
from sec_parser import TopSectionTitle

try:
    from rich.console import Console
    from rich.tree import Tree as RichTree
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TICKER = "MSTR"
FORM_TYPE = "10-Q"

# Reads your API key from the environment; set it before running:
#   export GOOGLE_API_KEY="sk-..."
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")

# SEC requires a descriptive User-Agent so they can contact you if needed
SEC_HEADERS = {
    "User-Agent": "Seminar Demo contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> str:
    """
    Resolve a ticker symbol to a CIK (Central Index Key) using the SEC's
    company-search API.
    """
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2000-01-01&enddt=2099-01-01&forms=10-Q"
    # Simpler: use the company-tickers.json file SEC publishes
    resp = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=SEC_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            # CIK must be zero-padded to 10 digits for EDGAR URLs
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"Ticker '{ticker}' not found in SEC company list.")


def get_latest_10q_info(cik: str) -> tuple[str, str]:
    """
    Return (accession_number, primary_document_filename) for the most recent
    10-Q filing of the given CIK.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=SEC_HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    filings = data["filings"]["recent"]
    forms = filings["form"]
    accessions = filings["accessionNumber"]
    primary_docs = filings["primaryDocument"]

    for form, accession, primary_doc in zip(forms, accessions, primary_docs):
        if form == FORM_TYPE:
            return accession, primary_doc

    raise ValueError(f"No {FORM_TYPE} found for CIK {cik}.")


def download_filing_html(cik: str, accession: str, primary_doc: str) -> str:
    """
    Download the raw HTML for the primary document of a filing.
    """
    # EDGAR URL format: accession number without dashes
    accession_clean = accession.replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession_clean}/{primary_doc}"
    )
    print(f"  Downloading: {url}")
    resp = requests.get(url, headers=SEC_HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.text


# ---------------------------------------------------------------------------
# Step 1 – Fetch the latest 10-Q
# ---------------------------------------------------------------------------

print("\n" + "=" * 65)
print(f"  STEP 1 — Fetching the latest {FORM_TYPE} for {TICKER}")
print("=" * 65)

cik = get_cik(TICKER)
print(f"  CIK: {cik}")

accession, primary_doc = get_latest_10q_info(cik)
print(f"  Most recent {FORM_TYPE}: {accession}  (document: {primary_doc})")

html = download_filing_html(cik, accession, primary_doc)
print(f"  Downloaded {len(html):,} characters of HTML.\n")


# ---------------------------------------------------------------------------
# Step 2 – Parse with sec_parser and visualise the document tree
# ---------------------------------------------------------------------------

print("=" * 65)
print("  STEP 2 — Parsing the filing and visualising the document tree")
print("=" * 65 + "\n")

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Invalid section type")
    parser = sp.Edgar10QParser()
    elements = parser.parse(html)

# Build a semantic tree from the flat element list
tree = sp.TreeBuilder().build(elements)


def _preview(text: str, width: int = 60) -> str:
    """Return a clipped single-line preview of node text."""
    snippet = " ".join((text or "").split())
    return snippet[:width] + ("…" if len(snippet) > width else "")


# sec_parser's sp.render() produces a ready-made indented tree string.
# For the seminar we print it directly; it works with or without rich.
print(sp.render(tree))

print()


# ---------------------------------------------------------------------------
# Step 3 – Extract MD&A and summarise with Gemini
# ---------------------------------------------------------------------------

print("=" * 65)
print("  STEP 3 — Extracting MD&A and summarising with Gemini")
print("=" * 65 + "\n")

# ── Find the MD&A section ────────────────────────────────────────────────────
MDNA_KEYWORDS = [
    "management",
    "discussion",
    "financial condition",
    "results of operations",
]


def _is_mdna_header(node) -> bool:
    """Return True if this node looks like the MD&A top-level section."""
    if not isinstance(node.semantic_element, TopSectionTitle):
        return False
    text = (getattr(node.semantic_element, "text", None) or "").lower()
    has_item2 = "item 2" in text
    has_mgmt = "management" in text and "discussion" in text
    return has_item2 or has_mgmt


def _collect_text(node) -> str:
    """Recursively collect all text from a tree node."""
    parts = []
    text = getattr(node.semantic_element, "text", None)
    if text:
        parts.append(text.strip())
    for child in node.children:
        parts.append(_collect_text(child))
    return "\n".join(p for p in parts if p)


# tree.nodes gives the flat list of ALL nodes in the tree; we only need the
# top-level ones (depth == 0) to find the MD&A section header.
mdna_node = None
for node in tree.nodes:
    if node.parent is None and _is_mdna_header(node):
        mdna_node = node
        break

if mdna_node is None:
    # Fallback: accept any depth
    for node in tree.nodes:
        if _is_mdna_header(node):
            mdna_node = node
            break

if mdna_node is None:
    print("  [!] Could not locate the MD&A section automatically.")
    print("      Tip: inspect the tree above and adjust MDNA_KEYWORDS.\n")
else:
    mdna_text = _collect_text(mdna_node)
    # Trim to a reasonable token budget (~8 000 words) for the LLM call
    word_limit = 8_000
    words = mdna_text.split()
    if len(words) > word_limit:
        mdna_text = " ".join(words[:word_limit]) + "\n\n[... truncated for length ...]"

    print(f"  MD&A section found ({len(mdna_text.split()):,} words). Sending to Gemini…\n")

    if not GENAI_AVAILABLE:
        print("  [!] google-genai is not installed.")
        print("      Run:  pip install google-genai\n")
    elif not GOOGLE_API_KEY:
        print("  [!] GOOGLE_API_KEY environment variable is not set.")
        print("      Run:  export GOOGLE_API_KEY='your-key-here'\n")
    else:
        client = genai.Client(api_key=GOOGLE_API_KEY)

        prompt = f"""You are a financial analyst assistant.
Below is the Management's Discussion & Analysis (MD&A) section from {TICKER}'s most recent 10-Q filing.

Please provide a concise, plain-English summary (4–6 paragraphs) covering:
1. Key business highlights and revenue drivers
2. Notable changes in profitability or expenses
3. Liquidity and capital resources
4. Any significant risks or forward-looking statements management highlighted

MD&A Text:
---
{mdna_text}
---

Summary:"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        summary = response.text

        print("─" * 65)
        print("  GEMINI SUMMARY OF MD&A")
        print("─" * 65)
        print(summary)
        print("─" * 65 + "\n")

print("\nDone! 🎉")
