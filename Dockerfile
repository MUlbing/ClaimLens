FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /usr/local/bin/uv

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    NLTK_DATA=/usr/local/share/nltk_data \
    MODEL_PATH=/app/models/tfidf_balanced_ngrams.joblib

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --locked --no-dev --python /usr/local/bin/python

RUN python -c "import nltk; nltk.download('wordnet', download_dir='/usr/local/share/nltk_data', raise_on_error=True)"

COPY api/ ./api/
COPY models/tfidf_balanced_ngrams.joblib ./models/tfidf_balanced_ngrams.joblib

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
