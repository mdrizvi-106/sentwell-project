# Sentwell · Financial Sentiment Intelligence

> Real-time stock news sentiment analysis powered by FinBERT and gpt-oss-20b (via Groq)

[![Live App](https://img.shields.io/badge/Live%20App-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/spaces/smrzv/marketpulse)
[![GitHub](https://img.shields.io/badge/GitHub-mdrizvi--106-black?style=flat-square&logo=github)](https://github.com/mdrizvi-106)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sam1061-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/sam1061)

---

## What is Sentwell?

Sentwell scans live financial news, runs it through **FinBERT** — a BERT model fine-tuned exclusively on financial text — and tells you whether the market mood around a stock is bullish, bearish, or neutral.

Type in up to 3 tickers. Get sentiment scores, confidence ratings, an AI-generated market brief, and trend charts.

**Live app →** [huggingface.co/spaces/smrzv/marketpulse](https://huggingface.co/spaces/smrzv/marketpulse)

---

## Features

| Feature | Description |
|---|---|
| **FinBERT NLP** | Finance-specific sentiment model, evaluated on 95 hand-labelled headlines (macro-F1 0.44 vs 0.23 baseline) |
| **Live news feed** | Pulls up to 25 headlines per ticker from Yahoo Finance in real time |
| **AI summaries** | gpt-oss-20b (via Groq) synthesises sentiment into a 3-sentence market brief |
| **Multi-ticker** | Analyse and compare up to 3 stocks simultaneously |
| **Trend analysis** | Track how sentiment has shifted day by day |
| **Watchlist** | Save your key tickers across sessions |
| **CSV export** | Download the full scored dataset for your own analysis |
| **Confidence filter** | Adjustable threshold — only surface headlines the model is sure about |

---

## How it works

```
User inputs ticker(s)
        ↓
Yahoo Finance live news feed (up to 25 headlines)
        ↓
FinBERT (ProsusAI) — classifies each headline as
positive / negative / neutral + confidence score
        ↓
gpt-oss-20b (via Groq) — generates 3-sentence AI market brief
        ↓
Plotly charts + summary dashboard
```

---


## Documentation

- [Model card](docs/MODEL_CARD.md): what the model does, how well it performs (EVAL-1), known limitations and monitoring
- [Runbook](docs/RUNBOOK.md): how to run, deploy and troubleshoot SentWell


---


## Tech stack

- **FinBERT** — `ProsusAI/finbert` via HuggingFace Transformers
- **gpt-oss-20b** — via Groq API for AI summaries
- **yFinance** — live news and price data
- **Streamlit** — UI framework
- **Plotly** — interactive charts
- **Pandas** — data handling
- **Deployed on** — Hugging Face Spaces

---

## Run locally

```bash
git clone https://github.com/mdrizvi-106/sentwell-project
cd sentwell-project
pip install -r requirements.txt
```

Set your Groq API key:

```bash
export GROQ_API_KEY=your_key_here
```

Run the app:

```bash
streamlit run app.py
```

The first run will download the FinBERT model (~440MB). Subsequent runs use the cached version.

---

## Requirements

```
streamlit
yfinance
plotly
pandas
transformers
torch
groq
```

---

## Project structure

```
sentwell/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── index.html          # Landing page (served via GitHub Pages)
└── README.md
```

---

## Versions

| Version | Status | Description |
|---|---|---|
| **v2** (current) | 🟢 Live on Hugging Face | FinBERT + AI summaries + multi-ticker + trend analysis |
| **v1** | 📁 GitHub archive | Original version, fully documented |

---

## Limitations

- Yahoo Finance caps news at ~25 headlines per ticker — less coverage for smaller-cap stocks
- Sentiment reflects current news coverage, not fundamental value
- Not financial advice

---

## Author

**Sam Rizvi** — Data Scientist / ML Engineer

- LinkedIn: [linkedin.com/in/sam1061](https://linkedin.com/in/sam1061)
- GitHub: [github.com/mdrizvi-106](https://github.com/mdrizvi-106)
- App: [huggingface.co/spaces/smrzv/marketpulse](https://huggingface.co/spaces/smrzv/marketpulse)

---

*Built with FinBERT · Hosted on Hugging Face · © 2026 Sam Rizvi*
