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
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

TICKERS = ["AAPL", "MSFT", "JPM", "TSLA", "XOM"]
HISTORY_PATH = "sentiment_history.csv"

# Fill these in to match your Space
HF_SPACE_REPO_ID = "smrzv/marketpulse"   # your Space's repo id
HF_TOKEN_ENV_VAR = "HF_TOKEN"            # GitHub Actions secret name


def get_todays_headline_scores(ticker: str) -> List[float]:
    """
    REPLACE THIS with your actual Sentwell logic:
      1. fetch today's headlines for `ticker` (whatever source you use)
      2. run them through FinBERT
      3. return the list of per-headline sentiment scores

    Copy the relevant functions from your HF Space's app.py here.
    Keeping it a separate function (rather than inlining) makes it a
    clean drop-in replacement.
    """
    raise NotImplementedError(
        "Copy your headline fetch + FinBERT scoring logic from app.py into this function"
    )


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
    Download the current sentiment_history.csv from the HF Space (if it
    exists yet) so today's run appends to real history instead of
    starting over each time -- GitHub Actions runners are stateless
    between runs.
    """
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_SPACE_REPO_ID,
            repo_type="space",
            filename=HISTORY_PATH,
            token=token,
        )
        # Copy into the working directory under the expected name
        import shutil
        shutil.copy(downloaded_path, HISTORY_PATH)
        print(f"Pulled existing {HISTORY_PATH} from Space.")
    except EntryNotFoundError:
        print(f"No {HISTORY_PATH} in Space yet -- starting fresh.")
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


def push_to_hf_space():
    """Upload the updated CSV straight into the HF Space repo."""
    token = os.environ.get(HF_TOKEN_ENV_VAR)
    if not token:
        raise EnvironmentError(
            f"{HF_TOKEN_ENV_VAR} not set. Add it as a GitHub Actions secret."
        )
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=HISTORY_PATH,
        path_in_repo=HISTORY_PATH,
        repo_id=HF_SPACE_REPO_ID,
        repo_type="space",
        commit_message=f"Update sentiment history {date.today().isoformat()}",
    )
    print(f"Pushed {HISTORY_PATH} to HF Space repo: {HF_SPACE_REPO_ID}")


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
        push_to_hf_space()
        print(f"Done. Timestamp: {datetime.utcnow().isoformat()}")


if __name__ == "__main__":
    main()
