# Containerised PC-build agent.
# Build:  docker build -t pc-build-agent .
# Run CLI (one-shot):
#   docker run --rm --env-file .env pc-build-agent \
#     pc-agent --query "A gaming PC under \$1200, prefer AMD"
# Run UI:
#   docker run --rm -p 8501:8501 --env-file .env pc-build-agent \
#     streamlit run app/streamlit_app.py --server.address=0.0.0.0
#
# For Amazon Bedrock, also pass AWS credentials, e.g. mount your config:
#   -v "$HOME/.aws:/root/.aws:ro" -e AWS_PROFILE=your-profile

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

# App code, dataset, tests.
COPY app ./app
COPY tests ./tests
COPY data ./data

# uv sync creates a project venv at /app/.venv; put it on PATH so the console
# scripts (pc-agent, pc-agent-eval) and streamlit resolve directly.
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8501

# Default: print CLI help. Override the command to run a build or the UI.
CMD ["pc-agent", "--help"]
