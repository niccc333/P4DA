"""
seminar_10q.py
==============
A seminar walkthrough for parsing SEC 10-Q filings with Python.

Prerequisites (install once):
    pip install sec-parser requests rich google-generativeai

Usage:
    $env:GOOGLE_API_KEY="your-gemini-api-key"  (PowerShell)
    set GOOGLE_API_KEY=your-gemini-api-key     (CMD)
    python seminar_10q.py
"""

import os
import warnings
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from bs4 import BeautifulSoup
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
        print("      PowerShell: $env:GOOGLE_API_KEY='your-key-here'")
        print("      CMD:        set GOOGLE_API_KEY=your-key-here\n")
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

        print("â”€" * 65)
        print("  GEMINI SUMMARY OF MD&A")
        print("â”€" * 65)
        print(summary)
        print("â”€" * 65 + "\n")

# ---------------------------------------------------------------------------
# Step 4 “ Parse income statement and graph as time series
# ---------------------------------------------------------------------------

print("=" * 65)
print("  STEP 4  ” Graphing Income Statement (Revenue & Net Income)")
print("=" * 65 + "\n")


def _parse_number(text: str):
    """Convert a financial cell string like '(1,234)' or '12,345' to float."""
    t = text.strip().replace("$", "").replace(",", "").replace("\xa0", "")
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        val = float(t)
        return -val if negative else val
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Step 4 - Parse balance sheet and graph Debt-to-Equity ratio
# ---------------------------------------------------------------------------

import re as _re

# -- Colour palette ----------------------------------------------------------
BG       = "#0b0b0f"   # near-black background
PANEL    = "#13131a"   # slightly lighter panel
PURPLE   = "#9b59f5"   # accent purple
BLUE     = "#4e8ef7"   # accent blue
GRAY_LT  = "#9aa0b4"   # light gray text
GRAY_MID = "#3a3a4d"   # mid gray grid / spines


def _find_balance_sheet_table(soup):
    DATE_HINTS = ("september", "december", "march", "june", "31,", "30,", "20")

    def _score(text_rows):
        all_labels = " ".join(r[0].lower() if r else "" for r in text_rows)
        s  = 3 if "total liabilities" in all_labels else 0
        s += 3 if "stockholders" in all_labels or "total equity" in all_labels else 0
        hdr = " ".join(" ".join(r) for r in text_rows[:5]).lower()
        s += sum(1 for h in DATE_HINTS if h in hdr)
        return s

    best_rows, best_score = None, 0
    for table in soup.find_all("table"):
        tr_list = table.find_all("tr")
        if len(tr_list) < 5:
            continue
        text_rows = [[td.get_text(" ", strip=True) for td in r.find_all(["td", "th"])]
                     for r in tr_list]
        sc = _score(text_rows)
        if sc > best_score:
            best_score, best_rows = sc, text_rows

    if best_score < 3 or best_rows is None:
        return None, {}

    header_row = None
    for cells in best_rows[:8]:
        if _re.search(r"\b20\d{2}\b", " ".join(cells)):
            header_row = cells
            break
    if header_row is None:
        ncols = max(len(r) for r in best_rows)
        header_row = [""] + [f"Period {i}" for i in range(1, ncols)]

    result = {}
    for cells in best_rows:
        if not cells:
            continue
        label = cells[0].strip()
        if not label:
            continue
        vals = [_parse_number(c) for c in cells[1:]]
        if any(v is not None for v in vals):
            result[label.lower()] = vals

    return header_row, result


def _extract_date_ddmmyy(text):
    MONTHS = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12",
    }
    t = text.strip()
    m = _re.search(
        r"(january|february|march|april|may|june|july|august"
        r"|september|october|november|december)\s+(\d{1,2})[,\s]+(\d{4})",
        t, _re.IGNORECASE,
    )
    if m:
        return "{}/{}/{}".format(
            m.group(2).zfill(2), MONTHS[m.group(1).lower()], m.group(3)[-2:])
    m2 = _re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
    if m2:
        return "{}/{}/{}".format(
            m2.group(2).zfill(2), m2.group(1).zfill(2), m2.group(3)[-2:])
    m3 = _re.search(r"(\d{4})-(\d{2})-(\d{2})", t)
    if m3:
        return "{}/{}/{}".format(m3.group(3), m3.group(2), m3.group(1)[-2:])
    return t


print("=" * 65)
print("  STEP 4 - Debt-to-Equity Ratio (Balance Sheet)")
print("=" * 65 + "\n")

try:
    soup = BeautifulSoup(html, "html.parser")
    bs_headers, bs_rows = _find_balance_sheet_table(soup)

    if not bs_rows:
        print("  [!] Could not locate the balance sheet table.\n")
    else:
        # Find Total Liabilities (not the grand total that includes equity)
        liab_key = next(
            (k for k in bs_rows
             if "total liabilit" in k and "equity" not in k and "assets" not in k),
            None,
        )

        # Find Total Equity (handles curly vs. straight apostrophes)
        eq_key = None
        EXACT_EQ = {
            "total stockholders' equity",
            "total stockholders\u2019 equity",
            "total stockholders' equity (deficit)",
            "total stockholders\u2019 equity (deficit)",
            "total stockholders equity",
            "total stockholders equity (deficit)",
            "total equity",
        }
        for k in bs_rows:
            if k in EXACT_EQ:
                eq_key = k
                break
        if eq_key is None:
            eq_key = next(
                (k for k in bs_rows
                 if ("stockholder" in k or "total equity" in k)
                 and any(w in k for w in ("equity", "deficit"))),
                None,
            )

        if liab_key is None or eq_key is None:
            print("  [!] Could not find liabilities/equity rows.")
            print(f"      liab_key found: {liab_key!r}")
            print(f"      eq_key found  : {eq_key!r}")
            print(f"      Available rows: {list(bs_rows.keys())[:20]}\n")
        else:
            raw_labels  = [h.strip() for h in bs_headers[1:] if h.strip()]
            date_labels = [_extract_date_ddmmyy(h) for h in raw_labels]
            n = len(date_labels)

            def _pick(vals, count):
                out = [v for v in vals if v is not None][:count]
                return out + [None] * (count - len(out))

            liab_vals = _pick(bs_rows[liab_key], n)
            eq_vals   = _pick(bs_rows[eq_key],   n)

            de_ratios = []
            for l_v, e_v in zip(liab_vals, eq_vals):
                if l_v is not None and e_v is not None and e_v != 0:
                    de_ratios.append(round(l_v / e_v, 4))
                else:
                    de_ratios.append(None)

            print(f"  Liabilities row : {liab_key!r}")
            print(f"  Equity row      : {eq_key!r}")
            print(f"  Dates (DD/MM/YY): {date_labels}")
            print(f"  Liabilities     : {liab_vals}")
            print(f"  Equity          : {eq_vals}")
            print(f"  D/E Ratios      : {de_ratios}\n")

            x      = list(range(n))
            y_vals = [abs(v) if v is not None else 0 for v in de_ratios]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor(BG)
            ax.set_facecolor(PANEL)

            bar_colors = [PURPLE if i < n - 1 else BLUE for i in range(n)]
            bars = ax.bar(x, y_vals, color=bar_colors, width=0.5,
                          zorder=3, edgecolor=GRAY_MID, linewidth=0.8)

            for bar, val in zip(bars, de_ratios):
                if val is not None and max(y_vals) > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(y_vals) * 0.02,
                        f"{val:.2f}x",
                        ha="center", va="bottom",
                        color=GRAY_LT, fontsize=10, fontweight="bold",
                    )

            y_line = [abs(v) if v is not None else float("nan") for v in de_ratios]
            ax.plot(x, y_line, color=BLUE, linewidth=2, linestyle="--",
                    marker="o", markersize=7, zorder=4)

            if n >= 2 and de_ratios[-1] is not None and de_ratios[-2] is not None:
                delta = de_ratios[-1] - de_ratios[-2]
                arr   = "\u25b2" if delta > 0 else "\u25bc"
                acol  = "#e74c3c" if delta > 0 else "#2ecc71"
                ax.annotate(
                    f"{arr} {abs(delta):.2f} vs prior quarter",
                    xy=(x[-1], y_vals[-1]),
                    xytext=(max(0, x[-1] - 0.7), y_vals[-1] + max(y_vals) * 0.14),
                    color=acol, fontsize=10, fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=acol, lw=1.5),
                )

            ax.axhline(0, color=GRAY_MID, linewidth=0.8, linestyle=":")
            ax.set_xticks(x)
            ax.set_xticklabels(date_labels, color=GRAY_LT, fontsize=11)
            ax.set_ylabel("D/E Ratio  (Total Liabilities / Total Equity)",
                          color=GRAY_LT, fontsize=10)
            ax.tick_params(axis="y", colors=GRAY_LT)
            ax.tick_params(axis="x", colors=GRAY_LT)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.1f}x"))
            for spine in ax.spines.values():
                spine.set_color(GRAY_MID)
            ax.yaxis.grid(True, color=GRAY_MID, linewidth=0.5,
                          linestyle="--", alpha=0.5)
            ax.set_axisbelow(True)

            from matplotlib.patches import Patch
            ax.legend(
                handles=[
                    Patch(facecolor=PURPLE, label="Prior Quarter"),
                    Patch(facecolor=BLUE,   label="Current Quarter"),
                ],
                facecolor=PANEL, edgecolor=GRAY_MID,
                labelcolor=GRAY_LT, loc="upper left",
            )

            plt.title(
                f"{TICKER} - Debt-to-Equity Ratio  (Balance Sheet)",
                color="white", fontsize=14, pad=15,
            )
            plt.tight_layout()
            plot_path = "financial_trend.png"
            plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor())
            print(f"  [+] Chart saved to {plot_path}\n")

except Exception as e:
    print(f"  [!] Error generating graph: {e}\n")
    import traceback; traceback.print_exc()
# Hi
# -nic and carter
print("\nDone!")
