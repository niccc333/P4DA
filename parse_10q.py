# Imports
from importlib.metadata import distribution
import os
import requests
import glob
import polars as pl
import hdbscan
import torch
import numpy as np
import sec_parser as sp
import math
import warnings
import re
import umap
from requests.adapters import HTTPAdapter
from sec_parser import TopSectionTitle, TitleElement
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from urllib3.util.retry import Retry
from transformers import AutoTokenizer, AutoModel, Cache
from alpaca.trading import AssetExchange, TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus

# Globs
alpaca_key = os.environ.get("ALPACA_API_KEY")
alpaca_secret_key = os.environ.get("ALPACA_SECRET_KEY")
massive_api_key = os.environ.get("MASSIVE_API_KEY")
common_stock_sic_map_path = r"/home/ptable/Data/Mapping/common_stock_sic_map.ipc"

retry_strategy = Retry(
    total=5,
    backoff_factor=2,
    status_forcelist=[502, 503, 504],
    allowed_methods=["GET"],
)

adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


# Init
trading_client = TradingClient(alpaca_key, alpaca_secret_key)
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModel.from_pretrained("ProsusAI/finbert").to("cuda")


### Start here with the asset selection
# We discover the tickers we want using REST since flatfiles doesn't support it
def get_common_stocks(api_key):
    if not api_key:
        raise ValueError("API key is missing")

    tickers = []
    url = "https://api.massive.com/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "CS",  # Common stock = CS
        "active": "true",
        "limit": 1000,
        "apiKey": massive_api_key,
    }

    try:
        while url:
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            print(
                f"Batch Status: {data.get('status')} | Items found: {data.get('count')}"
            )

            for d in data.get("results", []):
                tickers.append(
                    {"ticker": d["ticker"], "sic_code": d.get("sic_code", "Unknown")}
                )

            url = data.get("next_url")

            if url:
                url += f"&apiKey={api_key}"
                params = {}

            else:
                break

        return pl.DataFrame(tickers)

    except Exception as e:
        print(f"Error: {e}")
        return pl.DataFrame()


## First Filtering
# No penny stocks, and enought liquidity
# Must have a description
# Check dol vol and share vol to remove low denom stocks
def apply_coarse_filter(common_stock_df: pl.DataFrame, api_key: str):
    # Grab the snapshot_tickers
    url = f"https://api.massive.com/v2/snapshot/locale/us/markets/stocks/tickers?apiKey={api_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    snapshot_tickers = data.get("tickers", [])

    filtered_universe = []

    common_set = set(common_stock_df["ticker"])

    for s in snapshot_tickers:
        ticker = s.get("ticker")

        if ticker in common_set:
            day = s.get(
                "prevDay", {}
            )  # We're using the previous day as it contains all data
            price = day.get("c", 0)
            volume = day.get("v", 0)

            if price > 5 and volume > 500_000:
                if (price * volume) > 5_000_000:
                    filtered_universe.append(ticker)

    return filtered_universe


def fetch_description(ticker: str, api_key: str) -> str:
    # Pull the description of each ticker
    url = f"https://api.massive.com/v3/reference/tickers/{ticker}"
    params = {"apiKey": api_key}

    try:
        response = session.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        results = data.get("results", {})

        description = results.get("description")

        return description if description else "None"

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return "None"


# Unfortunately we now need to do some complex parsing for all the files we got
def get_ticker_item1_text(ticker_dir: str) -> tuple[str, str]:
    # Pattern: [ticker_dir]/10-K/[accession_number]/primary-document.html
    pattern = os.path.join(ticker_dir, "10-K", "*", "primary-document.html")
    matches = glob.glob(pattern)

    if matches:
        # If there are more than one in the accession_number we get most recent
        target_file = sorted(matches, reverse=True)[0]
        return extract_item_1(target_file)

    # If there are no matches on the first search take the largest html,
    # it's probs what we are looking for
    fallback_pattern = os.path.join(ticker_dir, "10-K", "*", "*.htm*")
    all_html_files = glob.glob(fallback_pattern)

    # Filter out index files which are small and useless
    valid_html = [f for f in all_html_files if "index.html" not in f.lower()]

    if valid_html:
        # The 10-K body is virtually always the largest file in the folder
        biggest_file = max(valid_html, key=os.path.getsize)
        return extract_item_1(biggest_file)

    return "", "failed"


def extract_item_1(html_path: str) -> tuple[str, str]:
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        parser = sp.Edgar10QParser()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Invalid section type")
            elements = parser.parse(html)

        content = _extract_with_triggers(elements, mode="strict")
        if content and len(content) >= 1000:
            return content, "strict"

        content = _extract_with_triggers(elements, mode="loose")
        if content and len(content) >= 1000:
            print(
                f"  [!] Strict failed for {html_path.split('/')[-4]}, trying loose mode..."
            )
            return content, "loose"

        print(f"  [!] Warning: {html_path.split('/')[-4]} result is empty/short.")
        return ("", "failed")

    except Exception as e:
        print(f"  [X] Error parsing {html_path}: {e}")
        return "", "failed"


def _extract_with_triggers(elements, mode="strict") -> str:
    item_1_content = []
    capturing = False
    waiting_for_business_text = False

    for node in elements:
        if not hasattr(node, "text") or not node.text:
            continue

        text = node.text.strip()
        clean_text = " ".join(text.split()).lower()

        is_short = len(clean_text) < 100

        if mode == "strict":
            starts_correctly = clean_text.startswith("item 1") or clean_text.startswith(
                "item 1."
            )
            is_start_candidate = is_short or starts_correctly
        elif mode == "loose":
            stop_triggers = [
                "item 1a",
                "item 1b",
                "item 2",
                "part ii",
                "risk factors",
                "management's discussion",
                "financial condition",
                "financial statements",
            ]

            starts_correctly = any(clean_text.startswith(t) for t in stop_triggers)
            is_start_candidate = is_short and starts_correctly

        if capturing:
            current_length = sum(len(t) for t in item_1_content)

            # Allow stopping if we have enough text OR if we hit a very strong stop signal
            if current_length > 1000 and is_short:
                stop_triggers = [
                    "item 1a",
                    "item 1b",
                    "item 2",
                    "part ii",
                    "risk factors",
                ]

                # For GE/MCD, "Risk Factors" is often the next major header
                if any(trigger in clean_text for trigger in stop_triggers):
                    capturing = False
                    break

        if not capturing and is_start_candidate:
            if "table of contents" in clean_text:
                continue

            if mode == "strict":
                snippet = clean_text[:100]

                if "item 1" in snippet and "business" in snippet:
                    capturing = True
                    waiting_for_business_text = False
                    if not is_short:
                        item_1_content.append(text)
                    continue

                if "item 1" in snippet and "business" not in snippet:
                    waiting_for_business_text = True
                    continue

                if waiting_for_business_text:
                    if "business" in clean_text:
                        capturing = True
                        waiting_for_business_text = False
                        continue
                    if len(clean_text) < 20 or clean_text.isdigit():
                        continue
                    waiting_for_business_text = False

            elif mode == "loose":
                if starts_correctly:
                    capturing = True
                    continue

        if capturing:
            item_1_content.append(text)
    return "\n".join(item_1_content)


def generate_company_embeddings(text: str, ticker, chunk_size=400, overlap=50):
    words = text.split()
    total_words = len(words)

    step = chunk_size - overlap  # To ensure semantic continuity

    chunk_vectors = []

    # Go over each Batch
    for i in range(0, total_words, step):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)

        # If the chunk is too small to be meaningful, continue
        if len(chunk_words) < 10:
            continue

        inputs = tokenizer(
            chunk_text, return_tensors="pt", truncation=True, max_length=512
        ).to("cuda")

        with torch.no_grad():
            outputs = model(**inputs)  # ** is automatically unpacking the dict
            chunk_vectors.append(outputs.last_hidden_state[0, 0, :].detach().cpu())

        if i + chunk_size >= total_words:
            break

        if not chunk_vectors:
            print(f"  [!] No valid chunks for {ticker}")
            return None

    # Mean pooling gets one average company_vector
    company_vector = torch.stack(chunk_vectors).mean(dim=0)
    return company_vector.numpy()


def cache_embeddings(
    embeddings_df: pl.DataFrame,
    ticker_col: str,
    embedding_col: str,
    extraction_state_col: str,
    cache_dir: str,
) -> None:
    full_cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(full_cache_dir, exist_ok=True)

    # Create new df with only the two cols
    cache_df = embeddings_df.select([ticker_col, embedding_col, extraction_state_col])

    save_path = os.path.join(full_cache_dir, "10K_embedding_cache_V2.parquet")

    cache_df.write_parquet(save_path)  # Add the file name to the end
    print(f"Successfully cached {len(cache_df)} embeddings to {save_path}")


# Ideally don't use this unless you have a full 10K, descriptions don't give enough info
def get_emergent_clusters(embeddings_df: pl.DataFrame, min_size=5) -> pl.DataFrame:
    # DO NOT sort the ticker list or embeddings in any other way before
    # using this function

    matrix = np.stack(embeddings_df["embedding"].to_list())

    # Standardize the matrix, squashes its dimensionality
    matrix_scaled = MinMaxScaler().fit_transform(matrix)

    reducer = umap.UMAP(
        n_neighbors=15,
        n_components=10,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )

    matrix_reduced = reducer.fit_transform(matrix_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_size,
        min_samples=1,
        metric="euclidean",
        prediction_data=True,
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(matrix_reduced)

    soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
    best_clusters = np.argmax(soft_clusters, axis=1)
    strengths = np.max(soft_clusters, axis=1)

    final_labels = []

    for i, label in enumerate(labels):
        if label == -1 and strengths[i] > 0.05:
            final_labels.append(best_clusters[i])
        else:
            final_labels.append(label)

    return embeddings_df.with_columns(
        [
            pl.Series("cluster_id", final_labels),
            pl.Series("cluster_strength", strengths),
        ]
    )


def calculate_grouped_zscore(
    df: pl.DataFrame, metric_col: str, group_col: str, suffix: str = "_zscore"
) -> pl.DataFrame:
    """
    Calculates a Z-score for a metric relative to its group.

    Args:
        df: The input DataFrame.
        metric_col: The column to normalize (e.g., 'pe_ratio').
        group_col: The column defining the peer groups (e.g., 'cluster_id').
        suffix: What to append to the new column name.
    """
    zscore = (
        pl.when(pl.len().over(group_col) > 4)
        .then(
            (pl.col(metric_col) - pl.col(metric_col).mean().over(group_col))
            / pl.col(metric_col).std().over(group_col)
        )
        .otherwise(pl.lit(None))
    )

    new_df = df.with_columns(zscore.alias(f"{metric_col}{suffix}"))

    if new_df[f"{metric_col}{suffix}"].null_count() == len(new_df):
        print(
            f"Warning: No groups in {group_col} had more than 5 members. All Z-scores are null."
        )

    return new_df


if __name__ == "__main__":
    if massive_api_key:
        # # Get the master list, no filter except common stock status
        # all_common_stock = get_common_stocks(massive_api_key)
        # print(f"Got {len(all_common_stock)} active common stocks")

        # # Apply the liquidity filter
        # liquid_universe = apply_coarse_filter(all_common_stock, massive_api_key)
        # print(f"Got {len(liquid_universe)} active liquid stocks")
        #
        # Grab all folders in the filings folder
        # root = Path("~/Data/Filings/USA/2025-Q4/sec-edgar-filings").expanduser()
        # ticker_dirs = [d for d in root.iterdir() if d.is_dir()]
        #
        # results = []
        # # Now we loop over each folder (ticker) get item1, embed it and cache it
        # for ticker_dir in tqdm(ticker_dirs, desc="Embedding 10K's", unit="tickers"):
        #     item1_text, status = get_ticker_item1_text(str(ticker_dir))
        #
        #     if item1_text:
        #         embedding = generate_company_embeddings(item1_text, ticker_dir.name)
        #         if embedding is not None:
        #             results.append(
        #                 {
        #                     "ticker": ticker_dir.name,
        #                     "embedding": embedding,
        #                     "extraction_state": status,
        #                 }
        #             )
        #
        # cleaned_df = pl.DataFrame(results)
        # print(cleaned_df)
        #
        # cache_embeddings(
        #     cleaned_df, "ticker", "embedding", "extraction_state", "~/Data/Cache"
        # )
        # print(f"Final cache size {len(cleaned_df)}")
        ### THIS IS AN EXAMPLE OF USING THE HDBSCAN, UNCOMMENT TO USE
        # Create clusters, they emerge themselves! Yipee
        tree_parser_embeddings = pl.read_parquet(
            "/home/ptable/Data/Cache/10K_embedding_cache_V2.parquet"
        )
        re_embeddings = pl.read_parquet(
            "/home/ptable/Data/Cache/10K_embedding_cache.parquet"
        )
        print(
            f"Extraction State Overview: {tree_parser_embeddings['extraction_state'].value_counts()}"
        )
        tree_parser_cluster = get_emergent_clusters(tree_parser_embeddings)
        re_parser_cluster = get_emergent_clusters(re_embeddings)
        to_inspect = [tree_parser_cluster, re_parser_cluster]

        for item in to_inspect:
            stats = item.select(
                [
                    (pl.col("cluster_strength") > 0.3).sum().alias("keep_30%"),
                    (pl.col("cluster_strength") > 0.5).sum().alias("keep_50%"),
                    (pl.col("cluster_strength") > 0.7).sum().alias("keep_70%"),
                ]
            )
            print(stats)
            unique_ids = item["cluster_id"].unique().to_list()
            print(f"Distinct Cluster IDs: {unique_ids}")
            distribution = (
                item.group_by("cluster_id")
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
            )

        print(distribution)
        noise_audit = (
            tree_parser_cluster.filter(pl.col("cluster_id") == -1)
            .group_by("extraction_state")
            .len()
        )
        print(noise_audit)

        tree_parser_cluster.write_parquet(
            "/home/ptable/Data/Cache/tree_cluster.parquet"
        )

    else:
        print("API key is None")
