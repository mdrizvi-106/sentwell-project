
"""Unit tests for pure helper functions in scripts/log_daily_sentiment.py.

These tests need no internet, no FinBERT model and no Hugging Face token,
so they run in seconds locally and in CI.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/ importable
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))
import log_daily_sentiment as lds  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ---------- signed_score: FinBERT label + confidence -> signed value ----------

def test_signed_score_positive_keeps_score():
    assert lds.signed_score("positive", 0.8) == 0.8


def test_signed_score_negative_flips_sign():
    assert lds.signed_score("negative", 0.8) == -0.8


def test_signed_score_neutral_is_zero():
    assert lds.signed_score("neutral", 0.8) == 0.0


# ---------- stale_symbols: same score on 3 weekdays = stale feed ----------

def make_history(rows):
    """rows: list of (ticker, date, sentiment_mean, headline_count)."""
    return pd.DataFrame(
        rows,
        columns=["ticker", "date", "sentiment_mean", "headline_count"],
    )


def test_identical_scores_on_three_weekdays_is_stale():
    df = make_history([
        ("USDJPY=X", "2026-10-14", 0.25, 4),
        ("USDJPY=X", "2026-10-15", 0.25, 4),
        ("USDJPY=X", "2026-10-16", 0.25, 4),
    ])
    assert lds.stale_symbols(df, "2026-10-16") == ["USDJPY=X"]


def test_changing_scores_are_not_stale():
    df = make_history([
        ("AAPL", "2026-10-14", 0.10, 12),
        ("AAPL", "2026-10-15", 0.32, 9),
        ("AAPL", "2026-10-16", -0.05, 15),
    ])
    assert lds.stale_symbols(df, "2026-10-16") == []


def test_weekend_repeats_do_not_count():
    # Fri, Sat, Sun with the same score: only ONE weekday, so not stale
    df = make_history([
        ("EURUSD=X", "2026-10-16", 0.20, 3),
        ("EURUSD=X", "2026-10-17", 0.20, 3),
        ("EURUSD=X", "2026-10-18", 0.20, 3),
    ])
    assert lds.stale_symbols(df, "2026-10-18") == []


# ---------- aggregate_today: one row per symbol per day ----------

def test_aggregate_today_averages_scores(monkeypatch):
    monkeypatch.setattr(
        lds,
        "get_todays_headline_scores",
        lambda s: ([0.5, -0.1], "yahoo"),
    )
    row = lds.aggregate_today("AAPL", "2026-10-20")

    assert row["ticker"] == "AAPL"
    assert row["date"] == "2026-10-20"
    assert row["headline_count"] == 2
    assert row["sentiment_mean"] == pytest.approx(0.2)
    assert row["news_source"] == "yahoo"
    assert row["model_revision"] == config.FINBERT_REVISION[:7]


def test_aggregate_today_no_headlines(monkeypatch):
    monkeypatch.setattr(
        lds,
        "get_todays_headline_scores",
        lambda s: ([], "yahoo"),
    )
    row = lds.aggregate_today("USDCHF=X", "2026-10-20")

    assert row["headline_count"] == 0
    assert row["news_source"] == "none"
    assert np.isnan(row["sentiment_mean"])

def test_get_model_returns_loaded_pipeline(monkeypatch):
    # Replace the real (slow) pipeline with a fake so no model is downloaded
    fake = object()
    monkeypatch.setattr(lds, "pipeline", lambda *args, **kwargs: fake)
    monkeypatch.setattr(lds, "_model", None)
    assert lds.get_model() is fake