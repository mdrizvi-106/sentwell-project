# SentWell evaluation — EVAL-1

## Why
SentWell uses FinBERT (ProsusAI/finbert) to score news headlines. Before trusting it,
I measured how often it agrees with careful human labels on real headlines from the app's own news feed.

## Part A: FinBERT vs hand labels

### Method
- **Data:** 95 unique headlines for 10 tickers (AAPL, MSFT, NVDA, JPM, GS, XOM, TSLA, AMZN, META, BP),
  collected on 18 Oct 2026 with the same `get_news()` function the app uses (`eval/collect_headlines.py`).
- **Labels:** I labelled every headline by hand as positive / negative / neutral, following written rules
  (`eval/LABELLING.md`). Labels were committed **before** FinBERT was run, so they could not be influenced by the model.
- **Model:** `transformers.pipeline("sentiment-analysis", model="ProsusAI/finbert")`, default settings.
- **Metrics:** accuracy, macro-F1, per-class precision/recall, confusion matrix (scikit-learn).

### Label split (human)
| positive | negative | neutral | total |
|---|---|---|---|
| 49 | 22 | 24 | 95 |

### Results
| Metric | FinBERT | Majority baseline (always "positive") |
|---|---|---|
| Accuracy | **43.2%** (41/95) | 51.6% (49/95) |
| Macro-F1 | **0.443** | 0.227 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| positive | 0.62 | 0.27 | 0.37 | 49 |
| negative | 0.59 | 0.45 | 0.51 | 22 |
| neutral | 0.32 | 0.75 | 0.44 | 24 |

**Confusion matrix** (rows = human, columns = FinBERT):

| | positive | negative | neutral |
|---|---|---|---|
| **positive** | 13 | 6 | 30 |
| **negative** | 3 | 10 | 9 |
| **neutral** | 5 | 1 | 18 |

**Reading it:**
- FinBERT has lower accuracy than a model that always says "positive", because half my labels are positive.
  On macro-F1, which weights all three classes equally, FinBERT is about twice as good as that baseline.
  This is why macro-F1 is the main metric here.
- FinBERT leans heavily towards neutral. It predicted neutral 57 times; I labelled only 24 headlines neutral.
- The biggest error bucket is **human positive → FinBERT neutral: 30 of the 54 errors.**

### Error audit: the 30 positive → neutral errors
I put each of the 30 into one of three groups, using fixed rules applied in order:

| Group | Meaning | Count |
|---|---|---|
| A. Not about the ticker | The feed returned a headline mainly about another company, or a list of stocks | 12 |
| B. Labeller error | Question, opinion, list or explainer: my own rules say neutral, but I labelled it positive | 9 |
| C. Genuine model miss | A clear positive fact for the ticker that FinBERT scored neutral | 9 |

**Finding:** only 9 of the 30 (30%) are clearly the model's fault. The other 21 come from the data feed (12)
and from my own labelling (9). The genuine misses are mostly long two-part headlines: a positive fact,
then a question or "here's what it means".

### Three examples
1. **Genuine model miss (C).** GS: *"Learn Why The Bull Case For Goldman Sachs Stock Could Change Following Earnings Beat"*.
   Human: positive. FinBERT: neutral.
   Why: the positive fact ("earnings beat") comes at the end of a hedged sentence ("could change").
   FinBERT seems to react to the cautious wording rather than the fact.
2. **Feed noise (A).** NVDA: *"ASML is the Most Important Company in the AI Ecosystem, and It Just Delivered Record Results…"*.
   Human: positive. FinBERT: neutral.
   Why: the headline is about ASML, not Nvidia. FinBERT has no idea which ticker the row belongs to, so the
   problem is in the input, not the model.
3. **Positive → negative (the most serious error type).** TODO: ticker, headline, and one line on why.

### Limitations
- One labeller (me), labelled quickly in one sitting; some headlines were judgement calls or guesses. My labels lean positive (49 of 95), and the audit found 9 cases where I broke my own rules. A careful second pass would likely raise FinBERT’s agreement score.
- 95 headlines from a single day's news: a small sample, not a benchmark.
- The news feed returns headlines about other companies (group A), which affects any model's score.
- The labels were not changed after seeing the model output. The audit above is reported alongside the
  original result, not instead of it.

### Future work (not built yet)
- **Relevance filter:** only score headlines that mention the company or ticker. This targets group A.
- **First-sentence scoring:** score only the first sentence of two-part headlines. This targets group C.
- **Second labeller:** have someone else label the same headlines, to measure label agreement.
- **Label consistency check:** in a week, re-label 30 random headlines blind (without my old labels or FinBERT’s output) and measure how often I agree with myself (Cohen’s kappa). That shows how much of the 43% is label noise.

### Reproduce
```bash
python eval/collect_headlines.py   # writes eval/headlines.csv (labels added by hand)
python eval/eval_finbert.py        # prints metrics, writes eval/finbert_predictions.csv
```

## Part B: LLM brief reliability
To be added.