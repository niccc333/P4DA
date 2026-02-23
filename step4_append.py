"""Appended Step 4 code for balance sheet D/E ratio chart."""

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

print("\nDone!")
