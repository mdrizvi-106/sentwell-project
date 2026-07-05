"""
scripts/log_daily_sentiment.py

Runs daily via GitHub Actions. Loops over all five tickers, scores
today's headlines with FinBERT, appends the aggregated result to a
local sentiment_history.csv, then pushes that updated file straight
into the Sentwell Hugging Face Space repo.

IMPORTANT: `get_todays_headline_scores()` below is a placeholder.
Replace its body with the actual headline-fetch + FinBERT scoring
code copied over from your Sentwell app.py on Hugging Face -- this
script can't import from the Space directly, so the logic needs to
live here too.
"""

import os
from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf
from transformers import pipeline
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

TICKERS = ["AAPL", "MSFT", "JPM", "TSLA", "XOM"]
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


def get_todays_headline_scores(ticker: str) -> List[float]:
    """
    Same headline source and FinBERT scoring as Sentwell's app.py
    (fetch_and_analyse): pull recent Yahoo Finance news for the
    ticker, score each headline's title with FinBERT, and return
    the signed sentiment scores.
    """
    try:
        news = yf.Ticker(ticker).news[:25]
    except Exception as e:
        print(f"[{ticker}] news fetch failed: {e}")
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


def aggregate_today(ticker: str) -> dict:
    scores = get_todays_headline_scores(ticker)
    if not scores:
        return {
            "ticker": ticker,
            "date": date.today().isoformat(),
            "sentiment_mean": np.nan,
            "headline_count": 0,
        }
    return {
        "ticker": ticker,
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
    for ticker in TICKERS:
        try:
            rows.append(aggregate_today(ticker))
            print(f"[{ticker}] logged today's sentiment")
        except NotImplementedError as e:
            print(f"[{ticker}] SKIPPED -- {e}")
        except Exception as e:
            print(f"[{ticker}] FAILED -- {e}")

    if rows:
        combined = append_to_history(rows)
        print(f"Master history now has {len(combined)} rows.")
        push_to_hf_dataset()
        print(f"Done. Timestamp: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
