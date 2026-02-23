# P4DA — Python for Data & Analysis
### A QUARCC Seminar Series

This repo contains the code for QUARCC's **Python for Data & Analysis** seminar. The featured demo `seminar_10q.py` walks through three steps live during the session:

1. **Fetch** the most recent MSTR 10-Q filing directly from SEC EDGAR
2. **Parse** it into a semantic document tree using `sec-parser`
3. **Summarize** the MD&A section using Google Gemini

---

## Prerequisites

- Python 3.10+
- A **free** Google Gemini API key (takes ~60 seconds — no credit card required)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-org/P4DA.git
cd P4DA
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install sec-parser requests rich google-genai
```

### 4. Get your free Gemini API key

1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with a Google account
3. Click **"Create API Key"** and copy it

Then set it in your terminal:

```bash
export GOOGLE_API_KEY="paste-your-key-here"   # Mac/Linux
# set GOOGLE_API_KEY=paste-your-key-here      # Windows CMD
```

> [!TIP]
> You can add this line to your `~/.bashrc` or `~/.zshrc` so you don't have to re-enter it each session.

### 5. Run the seminar script

```bash
python seminar_10q.py
```

You should see the SEC fetch, document tree, and a Gemini-generated MD&A summary printed to your terminal.

---

## What each step does

| Step | What happens |
|------|-------------|
| **Step 1** | Fetches MSTR's CIK from SEC, finds the most recent 10-Q accession, and downloads the HTML (~6 MB) |
| **Step 2** | Parses the HTML with `sec-parser`, builds a semantic tree, and prints it using `sp.render()` |
| **Step 3** | Locates the MD&A (Item 2) section in the tree, sends it to Gemini 2.5 Flash, and prints a plain-English summary |

---

## Troubleshooting

**`GOOGLE_API_KEY` not set** — the script will print a reminder and skip the Gemini call. Steps 1 and 2 will still run fine.

**Rate limit / quota error** — free Gemini keys have generous limits but if you hit one, wait a moment and re-run.

**Slow download** — SEC EDGAR occasionally throttles. The script will still complete; it just may take a few extra seconds.
