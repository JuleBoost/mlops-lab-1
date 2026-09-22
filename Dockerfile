# Lab 3: multi-stage build for the food11 inference API.
# Stage 1 installs deps with uv (cached unless pyproject/uv.lock change).
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock ./
# --no-install-project: deps only; the project itself needs src/ which comes later.
RUN uv sync --frozen --no-dev --no-install-project

# Stage 2: slim runtime, no build tools, only venv + source.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
