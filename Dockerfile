# Containerised data-analysis agent.
# Build:  docker build -t data-analysis-agent .
# Run CLI:
#   docker run --rm --env-file .env -v "$PWD/data:/app/data" data-analysis-agent \
#     data-agent --csv data/sales.csv --query "Which region has the most revenue?"
# Run UI:
#   docker run --rm -p 8501:8501 --env-file .env data-analysis-agent \
#     streamlit run app/streamlit_app.py --server.address=0.0.0.0

FROM python:3.12-slim

# Bring in uv (fast, reproducible installs) from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (cached layer), then the project.
COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --extra ui --no-dev

# App code + sample data.
COPY app ./app
COPY tests ./tests
COPY data ./data

# uv sync creates a project venv at /app/.venv; put it on PATH so the console
# scripts (data-agent, data-agent-eval) and streamlit resolve directly.
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8501

# Default: print CLI help. Override the command to run an analysis or the UI.
CMD ["data-agent", "--help"]
