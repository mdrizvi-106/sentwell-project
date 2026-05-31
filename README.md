# Sentwell · Financial Sentiment Intelligence

> Real-time stock news sentiment analysis powered by FinBERT and LLaMA 3.1

[![Live App](https://img.shields.io/badge/Live%20App-Hugging%20Face-yellow?style=flat-square)](https://huggingface.co/spaces/smrzv/marketpulse)
[![GitHub](https://img.shields.io/badge/GitHub-mdrizvi--106-black?style=flat-square&logo=github)](https://github.com/mdrizvi-106)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sam1061-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/sam1061)

---

## What is Sentwell?

Sentwell scans live financial news, runs it through **FinBERT** — a BERT model fine-tuned exclusively on financial text — and tells you whether the market mood around a stock is bullish, bearish, or neutral. In seconds.

Type in up to 3 tickers. Get sentiment scores, confidence ratings, an AI-generated market brief, and trend charts. No noise. Just signal.

**Live app →** [huggingface.co/spaces/smrzv/marketpulse](https://huggingface.co/spaces/smrzv/marketpulse)

---

## Features

| Feature | Description |
|---|---|
| **FinBERT NLP** | Financial-grade sentiment classification — far more accurate on market text than general-purpose models |
| **Live news feed** | Pulls up to 25 headlines per ticker from Yahoo Finance in real time |
| **AI summaries** | LLaMA 3.1 via Groq synthesises sentiment into a 3-sentence market brief |
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
LLaMA 3.1 (Groq) — generates 3-sentence AI market brief
        ↓
Plotly charts + summary dashboard
```

---

## Tech stack

- **FinBERT** — `ProsusAI/finbert` via HuggingFace Transformers
- **LLaMA 3.1 8B** — via Groq API for AI summaries
- **yFinance** — live news and price data
- **Streamlit** — UI framework
- **Plotly** — interactive charts
- **Pandas** — data handling
- **Deployed on** — Hugging Face Spaces

---

## Run locally

```bash
git clone https://github.com/mdrizvi-106/sentwell
cd sentwell
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
