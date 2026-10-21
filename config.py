"""Single source of truth for model and prompt versions.

Change a version here, never inline in app.py or the pipeline,
so every output can be traced to the exact model and prompt that made it.
"""

# FinBERT sentiment model, pinned to an exact Hugging Face commit
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"

# LLM used for the WHY intelligence and AI summary (Groq)
LLM_MODEL = "llama-3.1-8b-instant"

# Bump these whenever the prompt text changes (v1 -> v2 ...)
WHY_PROMPT_VERSION = "why-v1"
SUMMARY_PROMPT_VERSION = "summary-v1"