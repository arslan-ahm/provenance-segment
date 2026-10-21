# Provenance Explorer, e.g. for a Hugging Face Space (Docker SDK, port 7860).
# It serves committed runs from runs/ and runs model-free checks (budgeted baseline contexts,
# quote anchoring, the local NLI verifier on CPU). It never calls a model provider and needs no API key.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.9 /uv /uvx /bin/

RUN useradd -m -u 1000 app && mkdir -p /home/app/provseg && chown app:app /home/app/provseg
USER app
WORKDIR /home/app/provseg
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PATH="/home/app/provseg/.venv/bin:$PATH"

COPY --chown=app:app pyproject.toml uv.lock README.md ./
COPY --chown=app:app src ./src
RUN uv sync --frozen --no-dev --extra app --extra nli
# Bake the NLI verifier into the image so no visitor waits for a download.
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/nli-deberta-v3-small', device='cpu')"
COPY --chown=app:app runs ./runs

ENV PROVSEG_WARM_NLI=1 HF_HUB_OFFLINE=1
EXPOSE 7860
CMD ["provseg", "serve", "--runs-dir", "runs", "--host", "0.0.0.0", "--port", "7860"]
