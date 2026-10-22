# Runbook: SentWell

## Run the app locally
```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
set -a; source .env; set +a
streamlit run app.py
```
## Run with Docker
```bash
docker build -t sentwell .
docker run --rm -p 8501:8501 --env-file .env -v hf-cache:/root/.cache/huggingface sentwell
```

## Daily pipeline
- **Schedule:** 21:15 UTC daily (after the US market close in summer and winter time). Scheduled runs can start late, so the script fixes one run date at the start of each run.
- **Run manually:** Actions → Log daily sentiment → Run workflow
- Secrets needed: 
    - **Daily pipeline (GitHub Actions secret):** `HF_TOKEN`, used to push rows to the Hugging Face dataset
    - **App:** `GROQ_API_KEY` in `.env` locally or passed with `--env-file` in Docker; never committed

## Run the tests
```bash
python -m pytest -v
```

## When things go wrong
| Symptom | Likely cause | Fix |
|---|---|---|
| `cannot import name 'pipeline' from 'transformers'` | torch is missing or incompatible with the pinned transformers version. transformers hides `pipeline` when torch fails to load. Seen 18 Oct (stale image) and 21 Oct (unpinned torch). | Check with `docker run --rm sentwell python -c "import torch; print(torch.__version__)"`. torch must be pinned (`torch==2.14.1`, CPU wheel) in the Dockerfile. Rebuild with `docker build --no-cache -t sentwell .` |
| Docker "read-only file system" during build or run | Docker's disk is full; Docker Desktop switches to read-only. | Quit and restart Docker Desktop, then `docker system prune -a -f` and `docker builder prune -a -f`, check free space with `docker system df`, and rebuild. Prune after each Docker task. |
| A symbol shows the same score for days | The news source keeps returning the same old articles (USDJPY was stuck on the same 4 articles from 10 Jul to 3 Sep), or no new headlines. | The pipeline logs symbols flagged by `stale_symbols()` (same score on 3 weekdays). Check the `news_source` and `headline_count` columns for that symbol; the backup sources (Yahoo RSS, then Google News) should take over. If they don't, check the feeds manually. |
| Daily workflow stopped running | GitHub disables scheduled workflows in public repos after 60 days without commits (happened 4 Sep 2026). | Re-enable it under Actions → Log daily sentiment → Enable workflow. The keepalive step commits a timestamp if the repo has been idle 30+ days; check that step still exists in `daily_sentiment.yml`. Run it manually once with "Run workflow" to confirm. |
| `ModuleNotFoundError: No module named 'config'` in Docker | `config.py` not copied into the image | Dockerfile must have `COPY config.py .` next to `COPY app.py .`, then rebuild |
| WHY brief shows the fallback message | `GROQ_API_KEY` missing (locally: `.env` not loaded) or Groq returned invalid JSON | Locally: `set -a; source .env; set +a`. Docker: `--env-file .env`. Never commit `.env`. |
| Summary or WHY shows generic or empty text | Model retired (`model_not_found`), auth error, or reasoning used the whole token budget | Check the terminal for the error; confirm `config.LLM_MODEL` is in Groq's model list and `LLM_REASONING_EFFORT` is set |

## Changing a model or prompt
Edit `config.py` only and bump the version (e.g. `why-v1` → `why-v2`). Re-run the EVAL-1 evaluation and record the old and new metrics in the pull request.