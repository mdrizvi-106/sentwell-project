import sys
import pandas as pd

sys.path.append("scripts")                 # lets Python find the pipeline script
from log_daily_sentiment import get_news

TICKERS = ["AAPL", "MSFT", "NVDA", "JPM", "GS", "XOM", "TSLA", "AMZN", "META", "BP"]

rows = []
seen = set()
for ticker in TICKERS:
    items, source = get_news(ticker, count=10)
    for item in items:
        title = item["content"]["title"]
        if title in seen:
            continue
        seen.add(title)
        rows.append((ticker,title))
    print(ticker, source, len(items))

df = pd.DataFrame(rows, columns=["ticker", "headline"])
df["human_label"] = ""
df.to_csv("eval/headlines.csv", index=False)
print("Saved", len(df), "headlines")