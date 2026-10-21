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

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import os
import time
import urllib.parse
from datetime import date, datetime
from typing import List

import feedparser
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
        _model = pipeline("sentiment-analysis", model=config.FINBERT_MODEL, revision=config.FINBERT_REVISION)
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


def _rss_titles(symbol: str, count: int):
    """Backup news source when Yahoo returns nothing: Yahoo Finance RSS, then Google News RSS."""
    query = symbol.replace("=X", "") + (" forex" if symbol.endswith("=X") else " stock")
    urls = [
        ("yahoo_rss", f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={urllib.parse.quote(symbol)}&region=US&lang=en-US"),
        ("google_news", f"https://news.google.com/rss/search?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en"),
    ]
    for source, url in urls:
        try:
            entries = feedparser.parse(url).entries[:count]
        except Exception as e:
            print(f"[{symbol}] RSS failed ({url.split('/')[2]}): {e}")
            continue
        titles = [e.get("title", "").strip() for e in entries if e.get("title")]
        if titles:
            print(f"[{symbol}] {len(titles)} headlines from RSS backup ({url.split('/')[2]})")
            return titles, source
    return [], "none"


def get_news(symbol: str, count: int = 25):
    """
    Yahoo Finance via yfinance first; if it errors or returns nothing,
    fall back to RSS. Always returns yfinance-shaped items
    ({"content": {"title": ...}}) so the scoring code stays the same,
    plus the name of the source that supplied them.
    """
    items = []
    try:
        t = yf.Ticker(symbol)
        raw = yf_fetch_with_retry(lambda: t.get_news(count=count) if hasattr(t, "get_news") else t.news)
        items = [a for a in (raw or []) if (a.get("content") or {}).get("title")][:count]
    except Exception as e:
        print(f"[{symbol}] yfinance news failed: {e}")
    if items:
        return items, "yfinance"
    titles, source = _rss_titles(symbol, count)
    return [{"content": {"title": t}} for t in titles], source


def get_todays_headline_scores(symbol: str):
    """
    Same headline source and FinBERT scoring as Sentwell's app.py
    (fetch_and_analyse): pull recent Yahoo Finance news for the
    symbol, score each headline's title with FinBERT, and return
    the signed sentiment scores. Forex pairs often return little or
    no news -- that's expected, not an error.
    """
    news, source = get_news(symbol, 25)
    if not news:
        return [], source

    model = get_model()
    scores = []
    for article in news:
        content = article.get("content", {})
        title = content.get("title", "")
        if not title:
            continue
        result = model(title)[0]
        scores.append(signed_score(result["label"], result["score"]))

    return scores, source


def aggregate_today(symbol: str, run_date: str) -> dict:
    """run_date is fixed once per run, so a run that crosses midnight UTC
    doesn't split one day's symbols across two dates (seen on 29/30 Aug 2026)."""
    scores, source = get_todays_headline_scores(symbol)
    return {
        "ticker": symbol,
        "date": run_date,
        "sentiment_mean": float(np.mean(scores)) if scores else np.nan,
        "headline_count": len(scores),
        "news_source": source if scores else "none",
        "model_revision": config.FINBERT_REVISION[:7],
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

    # Rows written before news_source existed all came from yfinance.
    src = combined["news_source"] if "news_source" in combined else pd.Series(np.nan, index=combined.index)
    backfill = np.where(combined["headline_count"] > 0, "yfinance", "none")
    combined["news_source"] = src.astype(object).where(src.notna(), backfill)

    # 3-day average over calendar days (not the last 3 rows), so it never
    # bridges a gap such as 4 Sep - 11 Oct 2026.
    ma = pd.Series(np.nan, index=combined.index)
    for _, g in combined.groupby("ticker"):
        rolled = g.set_index(pd.to_datetime(g["date"]))["sentiment_mean"].rolling("3D", min_periods=1).mean()
        ma.loc[g.index] = rolled.to_numpy()
    combined["sentiment_ma_3"] = ma

    combined.to_csv(HISTORY_PATH, index=False)
    return combined


def stale_symbols(combined: pd.DataFrame, run_date: str, days: int = 3) -> List[str]:
    """
    Symbols whose sentiment was identical on each of the last `days` weekdays
    (headline_count > 0). Weekend repeats are normal; weekday repeats mean the
    source keeps returning the same old articles (USDJPY, Jul-Sep 2026).
    """
    stale = []
    end = pd.Timestamp(run_date)
    for ticker, g in combined.groupby("ticker"):
        g = g.assign(_d=pd.to_datetime(g["date"]))
        g = g[(g["_d"] <= end) & (g["_d"].dt.dayofweek < 5) & (g["headline_count"] > 0)].tail(days)
        if len(g) == days and g["_d"].iloc[0] >= end - pd.Timedelta(days=days + 4) \
                and g["sentiment_mean"].round(6).nunique() == 1:
            stale.append(ticker)
    return stale


def push_to_hf_dataset(run_date: str):
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
        commit_message=f"Update sentiment history {run_date}",
    )
    print(f"Pushed {HISTORY_PATH} to dataset repo: {HF_DATASET_REPO_ID}")


def main():
    pull_existing_history()
    run_date = datetime.utcnow().date().isoformat()   # one date for the whole run
    print(f"Run date: {run_date}")

    rows = []
    n = len(ALL_SYMBOLS)
    for i, symbol in enumerate(ALL_SYMBOLS):
        try:
            row = aggregate_today(symbol, run_date)
            rows.append(row)
            print(f"[{symbol}] logged {row['headline_count']} headlines ({row['news_source']})")
        except Exception as e:
            print(f"[{symbol}] FAILED -- {e}")
        if i < n - 1:
            time.sleep(2)  # small gap between symbols, same spirit as app.py's pacing

    empty = [r["ticker"] for r in rows if r["headline_count"] == 0]
    print(f"Symbols with headlines: {len(rows) - len(empty)}/{n}; no headlines: {empty or 'none'}")
    if rows and len(empty) == len(rows):
        # Every symbol came back empty: the news source is down. Fail loudly
        # instead of silently writing a day of blank rows.
        raise SystemExit("No headlines for any symbol today -- news source unavailable, history not updated.")

    if rows:
        combined = append_to_history(rows)
        print(f"Master history now has {len(combined)} rows.")
        stale = stale_symbols(combined, run_date)
        print(f"Stale (same score on last 3 weekdays): {stale or 'none'}")
        push_to_hf_dataset(run_date)
        print(f"Done. Timestamp: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
