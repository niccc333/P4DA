"""Debug script to inspect balance sheet row keys and values for MSTR."""
import requests
from bs4 import BeautifulSoup

SEC_HEADERS = {"User-Agent": "Seminar Demo contact@example.com", "Accept-Encoding": "gzip, deflate"}
TICKER = "MSTR"
resp = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=15)
cik = next(str(e["cik_str"]).zfill(10) for e in resp.json().values() if e["ticker"].upper() == TICKER)

sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=SEC_HEADERS, timeout=15).json()
f = sub["filings"]["recent"]
accession, primary_doc = next(
    (a, p) for fm, a, p in zip(f["form"], f["accessionNumber"], f["primaryDocument"]) if fm == "10-Q"
)
url = "https://www.sec.gov/Archives/edgar/data/{}/{}/{}".format(
    int(cik), accession.replace("-", ""), primary_doc)
print(f"Fetching {url}")
html = requests.get(url, headers=SEC_HEADERS, timeout=60).text

soup = BeautifulSoup(html, "html.parser")

DATE_HINTS = ("september", "december", "march", "june", "31,", "30,", "20")

def _parse_number(text):
    t = text.strip().replace("$", "").replace(",", "").replace("\xa0", "")
    negative = t.startswith("(") and t.endswith(")")
    t = t.strip("()")
    try:
        val = float(t)
        return -val if negative else val
    except ValueError:
        return None

def _score(text_rows):
    all_labels = " ".join(r[0].lower() if r else "" for r in text_rows)
    s  = 3 if "total liabilities" in all_labels else 0
    s += 3 if "stockholders" in all_labels or "total equity" in all_labels else 0
    s += sum(1 for h in DATE_HINTS if h in " ".join(" ".join(r) for r in text_rows[:5]).lower())
    return s

best_rows, best_score = None, 0
for table in soup.find_all("table"):
    tr_list = table.find_all("tr")
    if len(tr_list) < 5:
        continue
    text_rows = [[td.get_text(" ", strip=True) for td in r.find_all(["td", "th"])] for r in tr_list]
    sc = _score(text_rows)
    if sc > best_score:
        best_score, best_rows = sc, text_rows

print(f"\nBest score: {best_score}\n")

# Print all rows with numbers
result = {}
for row in (best_rows or []):
    lbl = row[0].strip() if row else ""
    if not lbl:
        continue
    vals = [_parse_number(c) for c in row[1:]]
    numeric = [v for v in vals if v is not None]
    if numeric:
        result[lbl.lower()] = vals
        print(f"  {repr(lbl.lower()[:60])}")
        print(f"    -> {vals[:4]}")

print("\n\nLiabilities search:")
LIAB_KEYS = ["total liabilities", "total liabilities and stockholders", "total liabilities and mezzanine"]
for k in LIAB_KEYS:
    if k in result:
        print(f"  FOUND: {k!r} -> {result[k][:4]}")

liab_key = next((k for k in LIAB_KEYS if k in result), None)
if liab_key is None:
    liab_key = next((k for k in result if "total liabilit" in k), None)
print(f"  => liab_key = {liab_key!r}")
if liab_key:
    print(f"     values   = {result[liab_key][:4]}")

print("\nEquity search:")
eq_key = next((k for k in result if ("stockholder" in k or "total equity" in k) and any(w in k for w in ("equity", "deficit"))), None)
print(f"  => eq_key = {eq_key!r}")
if eq_key:
    print(f"     values = {result[eq_key][:4]}")
