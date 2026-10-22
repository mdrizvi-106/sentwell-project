# Model card: SentWell sentiment

## Overview
- **Task:** classify financial news headlines as positive, negative or neutral, then average the signed scores per symbol per day
- **Model:** ProsusAI/finbert (BERT fine-tuned on financial text), pinned to revision `0bc2ac1` in `config.py`
- **LLM layer:** llama-3.1-8b-instant via the Groq API writes the WHY brief and the AI summary (prompt versions `why-v1` and `summary-v1`). The WHY output is validated as JSON; if validation fails, the app shows a safe fallback instead.
- **Owner:** Mohamed Rizvi Shaik Abdulla · **Last updated:** 21 Oct 2026
- 22 Oct 2026: **switched from llama-3.1-8b-instant** (retired by Groq; all LLM calls were silently falling back) to openai/gpt-oss-20b with reasoning_effort='low' (without it, reasoning used the whole token budget and returned empty answers).

## Intended use
- Research and portfolio demonstration of news-sentiment monitoring for 21 US equities and 5 major FX pairs
- Exploring how news tone changes over time, alongside price charts

## Not intended for
- Trading decisions or investment advice
- Non-English or non-financial text
- Any use where a single headline's label is treated as reliable on its own (see evaluation below)

## Data
- **Input:** up to 25 headlines per symbol per run from yfinance; when yfinance returns nothing, Yahoo Finance RSS and then Google News RSS are used as backups
- **Daily pipeline:** GitHub Actions at 21:15 UTC scores each symbol and appends one row per symbol per day to the Hugging Face dataset `smrzv/sentwell-sentiment-history`
- **Stored columns include:** ticker, date, sentiment_mean, headline_count, news_source, model_revision (from 21 Oct 2026; earlier rows have no value)
- **Known gap:** no data from 4 Sep to 11 Oct 2026 (the scheduled workflow was disabled by GitHub; not backfilled)

## Evaluation (EVAL-1, 18 Oct 2026)
- **Test set:** 95 unique headlines for 10 tickers, collected with the app's own `get_news()`, hand-labelled with written rules before running the model
- **Results:** accuracy 43.2% (41/95) vs 51.6% for an always-positive baseline; macro-F1 0.443 vs 0.227 for that baseline
- **Error pattern:** FinBERT leans heavily neutral (57 neutral predictions vs 24 neutral labels). In the largest error group (30 human-positive headlines scored neutral), 12 were not about the ticker, 9 were labelling errors and 9 were genuine model misses
- **Reproducibility:** the same evaluation run on AWS EC2 (data in S3, IAM role scoped to one bucket) gave identical metrics
- Full method and error analysis: `eval/RESULTS.md` (EVAL-1 branch)

## Known limitations
- **Off-topic headlines:** the news feed returns articles about other companies (e.g. a Tesla headline under AAPL); 12 of the 30 audited errors came from this. Planned fix: relevance filter (REL-1)
- **Neutral bias:** FinBERT often scores clickbait or two-part headlines as neutral even when they contain good news
- **Small evaluation set:** 95 headlines, one day's news, one labeller who labelled quickly; the numbers are indicative, not a benchmark
- **FX coverage:** forex pairs get few or no headlines on many days (USDCHF had 0 headlines for weeks)
- **Daily averages hide detail:** one strongly worded headline can move a day's mean when headline_count is low

## Monitoring
- **Staleness check:** flags symbols whose score is identical on the last 3 weekdays (catches stuck feeds like USDJPY Jul–Sep 2026)
- **Traceability:** `news_source` and `model_revision` are stored on every row; `headline_count` shows how much evidence each score has
- **Tests and CI:** 9 unit tests run on every pull request and push to `main` (GitHub Actions)
- **Keepalive:** the workflow commits if the repo has been idle 30+ days, so GitHub does not disable the schedule

## Changing the model or prompts
Edit `config.py` only, bump the version, re-run the EVAL-1 evaluation, and record the old and new metrics in the pull request.