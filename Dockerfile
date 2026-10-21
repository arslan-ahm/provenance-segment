# Read-only Provenance Explorer, e.g. for a Hugging Face Space (Docker SDK, port 7860).
# It serves committed runs from runs/ and never calls a model, so it needs no API key.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /bin/

RUN useradd -m -u 1000 app
WORKDIR /home/app/provseg
COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev --extra app && chown -R app:app /home/app/provseg
COPY --chown=app:app runs ./runs

USER app
ENV PATH="/home/app/provseg/.venv/bin:$PATH"
EXPOSE 7860
CMD ["provseg", "serve", "--runs-dir", "runs", "--host", "0.0.0.0", "--port", "7860"]
