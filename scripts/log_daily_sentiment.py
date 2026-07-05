"""
scripts/log_daily_sentiment.py

Runs daily via GitHub Actions. Loops over a broad set of equities
(spanning sectors) plus major forex pairs, scores today's headlines
with FinBERT, appends the aggregated result to a local
sentiment_history.csv, then pushes that updated file to a separate
Hugging Face Dataset repo (not the Space, so pushes don't trigger app
rebuilds).

Note: Yahoo Finance's `.news` endpoint is built around company
tickers. Forex pairs (e.g. "EURUSD=X") typically return sparse or
empty news -- this is expected, not a bug. Those rows will just log
0 headlines / blank sentiment on days with no coverage rather than
crashing.
"""

import os
import time
from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from transformers import pipeline
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# ~20 equities spanning sectors (tech, finance, EV, consumer, energy, healthcare)
EQUITY_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMD",       # tech
    "JPM", "BAC", "GS", "V", "MA",                 # finance
    "TSLA", "RIVN",                                 # EV / mobility
    "AMZN", "WMT", "COST", "SBUX",                  # consumer
    "XOM", "CVX",                                   # energy
    "UNH", "JNJ", "LLY",                            # healthcare
]

# Major forex pairs -- also usable for the dissertation's Forex analysis
FOREX_PAIRS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X"]

ALL_SYMBOLS = EQUITY_TICKERS + FOREX_PAIRS

HISTORY_PATH = "sentiment_history.csv"

# Fill these in to match your setup
HF_DATASET_REPO_ID = "smrzv/sentwell-sentiment-history"   # separate Dataset repo (not the Space)
HF_TOKEN_ENV_VAR = "HF_TOKEN"                              # GitHub Actions secret name


_model = None


def get_model():
    """Load FinBERT once per script run (same model Sentwell's app.py uses)."""
    global _model
    if _model is None:
        _model = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    return _model


def signed_score(label: str, score: float) -> float:
    """
    FinBERT's score is always a positive confidence value (0-1) --
    it doesn't encode direction by itself. Convert label + score into
    a single signed value so sentiment_mean actually reflects bullish
    vs bearish lean, not just how confident the model was.
    """
    if label == "positive":
        return score
    if label == "negative":
        return -score
    return 0.0


def yf_fetch_with_retry(fn, retries=3, delay=4):
    """
    Same backoff pattern as Sentwell's app.py (yf_fetch_with_retry) --
    Yahoo Finance rate-limits aggressively, and looping over ~25
    symbols in one run makes hitting that limit far more likely than
    the app's interactive single-ticker lookups.
    """
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                if attempt < retries - 1:
                    time.sleep(delay * (attempt + 1))
                else:
                    raise
            else:
                raise


def get_todays_headline_scores(symbol: str) -> List[float]:
    """
    Same headline source and FinBERT scoring as Sentwell's app.py
    (fetch_and_analyse): pull recent Yahoo Finance news for the
    symbol, score each headline's title with FinBERT, and return
    the signed sentiment scores. Forex pairs often return little or
    no news -- that's expected, not an error.
    """
    try:
        news = yf_fetch_with_retry(lambda: yf.Ticker(symbol).news[:25])
    except Exception as e:
        print(f"[{symbol}] news fetch failed: {e}")
        return []

    if not news:
        return []

    model = get_model()
    scores = []
    for article in news:
        content = article.get("content", {})
        title = content.get("title", "")
        if not title:
            continue
        result = model(title)[0]
        scores.append(signed_score(result["label"], result["score"]))

    return scores


def aggregate_today(symbol: str) -> dict:
    scores = get_todays_headline_scores(symbol)
    if not scores:
        return {
            "ticker": symbol,
            "date": date.today().isoformat(),
            "sentiment_mean": np.nan,
            "headline_count": 0,
        }
    return {
        "ticker": symbol,
        "date": date.today().isoformat(),
        "sentiment_mean": float(np.mean(scores)),
        "headline_count": len(scores),
    }


def pull_existing_history():
    """
    Download the current sentiment_history.csv from the HF Dataset repo
    (if it exists yet) so today's run appends to real history instead of
    starting over each time -- GitHub Actions runners are stateless
    between runs.
    """
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_DATASET_REPO_ID,
            repo_type="dataset",
            filename=HISTORY_PATH,
            token=token,
        )
        # Copy into the working directory under the expected name
        import shutil
        shutil.copy(downloaded_path, HISTORY_PATH)
        print(f"Pulled existing {HISTORY_PATH} from dataset repo.")
    except EntryNotFoundError:
        print(f"No {HISTORY_PATH} in dataset repo yet -- starting fresh.")
    except Exception as e:
        print(f"Could not pull existing history ({e}) -- starting fresh.")


def append_to_history(new_rows: List[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)

    if os.path.exists(HISTORY_PATH):
        history = pd.read_csv(HISTORY_PATH)
        combined = pd.concat([history, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    combined["sentiment_ma_3"] = (
        combined.groupby("ticker")["sentiment_mean"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    combined.to_csv(HISTORY_PATH, index=False)
    return combined


def push_to_hf_dataset():
    """Upload the updated CSV to the separate HF Dataset repo (not the Space)."""
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    if not token:
        raise EnvironmentError(
            f"{HF_TOKEN_ENV_VAR} not set. Add it as a GitHub Actions secret."
        )
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=HISTORY_PATH,
        path_in_repo=HISTORY_PATH,
        repo_id=HF_DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update sentiment history {date.today().isoformat()}",
    )
    print(f"Pushed {HISTORY_PATH} to dataset repo: {HF_DATASET_REPO_ID}")


def main():
    pull_existing_history()

    rows = []
    n = len(ALL_SYMBOLS)
    for i, symbol in enumerate(ALL_SYMBOLS):
        try:
            rows.append(aggregate_today(symbol))
            print(f"[{symbol}] logged today's sentiment")
        except Exception as e:
            print(f"[{symbol}] FAILED -- {e}")
        if i < n - 1:
            time.sleep(2)  # small gap between symbols, same spirit as app.py's pacing

    if rows:
        combined = append_to_history(rows)
        print(f"Master history now has {len(combined)} rows.")
        push_to_hf_dataset()
        print(f"Done. Timestamp: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
