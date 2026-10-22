"""Prompt building and output checking for the WHY intelligence brief.

No Streamlit imports here, so the app, the eval script and the tests can all use it.
"""
import json

REQUIRED_KEYS = ["drivers", "narratives", "root_causes", "contradictions"]


def build_why_prompt(company, sym, price_str, df):
    """Return the WHY prompt text.

    df needs columns 'headline', 'sentiment' and 'score'.
    """
    pos = len(df[df['sentiment'] == 'positive'])
    neg = len(df[df['sentiment'] == 'negative'])
    neu = len(df[df['sentiment'] == 'neutral'])
    dom = df['sentiment'].mode()[0]

    # Use all headlines ranked by confidence
    headlines_pos = df[df['sentiment']=='positive'].sort_values('score', ascending=False)['headline'].head(8).tolist()
    headlines_neg = df[df['sentiment']=='negative'].sort_values('score', ascending=False)['headline'].head(8).tolist()
    headlines_neu = df[df['sentiment']=='neutral'].sort_values('score', ascending=False)['headline'].head(5).tolist()

    all_headlines = "\n".join(
        [f"[POSITIVE] {h}" for h in headlines_pos] +
        [f"[NEGATIVE] {h}" for h in headlines_neg] +
        [f"[NEUTRAL]  {h}" for h in headlines_neu]
    )

    prompt = f"""You are a senior equity analyst conducting deep sentiment analysis on {company} ({sym}).
Current price: {price_str}
Sentiment: {pos} positive, {neg} negative, {neu} neutral headlines. Dominant: {dom.upper()}.
All headlines with their sentiment labels:
{all_headlines}
Analyse WHY the sentiment is what it is. Respond ONLY with a JSON object (no markdown, no preamble) with exactly these 4 keys, each containing a list of 3-4 concise bullet strings:
{{
  "drivers": ["...", "...", "..."],
  "narratives": ["...", "...", "..."],
  "root_causes": ["...", "...", "..."],
  "contradictions": ["...", "...", "..."]
}}
Definitions:
- "drivers": The specific, concrete factors actively pushing sentiment in the dominant direction (product launches, earnings beats/misses, partnerships, macro tailwinds/headwinds, executive decisions).
- "narratives": The overarching story or theme the market is telling about this company right now — what is the dominant frame investors/media are using? Include emerging counter-narratives.
- "root_causes": The deeper, underlying structural or fundamental reasons behind the sentiment — not just what happened, but WHY it matters (competitive dynamics, industry shifts, valuation concerns, regulatory environment).
- "contradictions": Conflicting signals present in the headlines — places where positive and negative sentiment coexist, or where short-term and long-term outlooks diverge.
Be specific to the company and headlines. No generic platitudes."""

    return prompt 


def parse_why_response(raw):
    """Check a raw LLM reply.

    Returns (data, status). status is one of:
      "ok"           - valid JSON with all 4 keys, each a list of strings
      "empty"        - reply was empty or only whitespace
      "invalid_json" - could not be parsed as JSON
      "wrong_type"   - parsed, but not a dict, or a value is not a list of strings
      "missing_keys" - parsed dict is missing one of REQUIRED_KEYS
    data is the dict when status is "ok", otherwise None.
    """
    # TODO 1: if raw is None or raw.strip() == "": return None, "empty"
    # TODO 2: if the reply contains ``` fences, keep only the text between them,
    #         and drop a leading "json" (copy the cleanup logic from app.py)
    # TODO 3: try json.loads; on json.JSONDecodeError return None, "invalid_json"
    # TODO 4: if it isn't a dict, return None, "wrong_type"
    # TODO 5: if any key in REQUIRED_KEYS is missing, return None, "missing_keys"
    # TODO 6: if any value isn't a list of strings, return None, "wrong_type"
    # TODO 7: return data, "ok"
    ...