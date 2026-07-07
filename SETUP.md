# Setup Guide

Step-by-step setup for the PC Build Agent. For what it does and how it's designed,
see [`README.md`](README.md); for real run traces and evaluation results, see
[`AGENT_RUN_REPORT.md`](AGENT_RUN_REPORT.md).

---

## 1. Prerequisites

- **Python 3.11+** (pinned to 3.12 via `.python-version`).
- **[`uv`](https://docs.astral.sh/uv/)** — package/environment manager.
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv --version
  ```
- **Credentials for one LLM provider** — an Anthropic API key **or** AWS access to
  Amazon Bedrock (step 3).

## 2. Install

```bash
uv sync --extra ui     # agent + CLI + evaluator + Streamlit UI + tests
# or: uv sync          # core only (no UI)
```

`uv sync` creates `.venv/` and installs the pinned dependency set. Prefix commands
with `uv run` to use it. Non-uv users can `pip install -r requirements.txt` (core
deps) instead.

Verify without credentials:

```bash
uv run pytest -q       # expect: 18 passed
uv run pc-agent --help
```

## 3. Configure a provider

```bash
cp .env.example .env
```

### Option A — Direct Anthropic API
```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5     # or any model your account can access
```
Key from <https://console.anthropic.com/>.

### Option B — Amazon Bedrock
```dotenv
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-6   # a model/inference-profile you can access
```
AWS credentials come from your normal chain — not from `.env`. Use a named profile
by prefixing commands:
```bash
AWS_PROFILE=your-profile uv run pc-agent --query "..."
```
List Bedrock model IDs you can use (read-only):
```bash
AWS_PROFILE=your-profile aws bedrock list-foundation-models \
  --region us-east-1 --by-provider anthropic \
  --query 'modelSummaries[].modelId' | head
```

### Optional tuning (defaults in `config.py`)

| Variable | Default | Meaning |
|---|---|---|
| `LLM_TEMPERATURE` | `0` | Sampling temperature (ignored by adaptive-thinking models). |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens per model turn. |
| `AGENT_MAX_STEPS` | `14` | Max reason→act→observe iterations per turn. |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout. |
| `LLM_MAX_RETRIES` | `3` | SDK auto-retries on 429/5xx/network. |
| `DATA_DIR` | `data` | Where the dataset CSVs live. |
| `TRACE_DIR` | `traces` | Where JSONL reasoning traces are written. |

## 4. Dataset

The 8 build-relevant CSVs are already vendored in `data/`. To refresh them from
the source repo:

```bash
# straight from GitHub
uv run python scripts/fetch_dataset.py --github
# or from a local clone
uv run python scripts/fetch_dataset.py --from /path/to/Computer_Components_Dataset
```

Source: <https://github.com/vinayak-ensemble/Computer_Components_Dataset>

## 5. Run

```bash
# One-shot
uv run pc-agent --query "A 1080p gaming PC, budget about \$1200, prefer AMD"

# Interactive chat (feedback loop): describe a PC, then type feedback to amend it
uv run pc-agent

# Evaluation scenarios
uv run pc-agent-eval --scenarios tests/scenarios.json

# Web UI (needs: uv sync --extra ui)
uv run streamlit run app/streamlit_app.py
```

Prefix with `AWS_PROFILE=...` on Bedrock. Each run prints a `traces/<run_id>.jsonl`
path with the full reasoning chain.

## 6. Docker (optional)

```bash
docker build -t pc-build-agent .
docker run --rm --env-file .env pc-build-agent \
  pc-agent --query "A compact Mini ITX Intel build under \$1500"
# UI:
docker run --rm -p 8501:8501 --env-file .env pc-build-agent \
  streamlit run app/streamlit_app.py --server.address=0.0.0.0
```
For Bedrock in Docker, also mount creds: `-v "$HOME/.aws:/root/.aws:ro" -e AWS_PROFILE=your-profile`.

## 7. Troubleshooting

**`uv sync` compiles numpy/pyarrow/Pillow from source and fails.** On older glibc
(≈2.26) the newest wheels aren't available. The project already pins compatible
versions (`numpy<2.3`, older `streamlit`/`pyarrow`/`pillow` for the UI extra). On a
modern machine you can loosen those in `pyproject.toml`.

**`Missing dataset file for 'cpu'`.** The CSVs aren't in `data/`. Run
`uv run python scripts/fetch_dataset.py --github`.

**`botocore ... ProfileNotFound`.** The `AWS_PROFILE` name doesn't match
`~/.aws/config`. Check `aws configure list-profiles`.

**Agent stops with `max_steps`.** The turn hit `AGENT_MAX_STEPS`. GPU searches in
this dataset can take several tries (many null-priced rows); raise `AGENT_MAX_STEPS`
in `.env` or narrow the request.

**Streamlit command not found.** Re-run `uv sync --extra ui`.
