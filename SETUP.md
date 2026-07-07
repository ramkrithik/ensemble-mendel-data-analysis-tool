# Setup Guide

Step-by-step setup for the Data Analysis Agent. For what it does and how it's
designed, see [`README.md`](README.md); for real run traces and evaluation
results, see [`AGENT_RUN_REPORT.md`](AGENT_RUN_REPORT.md).

---

## 1. Prerequisites

- **Python 3.11+** (the project is pinned to 3.12 via `.python-version`).
- **[`uv`](https://docs.astral.sh/uv/)** — the package/environment manager.
  ```bash
  # install uv if you don't have it (see https://docs.astral.sh/uv/ for other methods)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  uv --version
  ```
- **Credentials for one LLM provider** — either an Anthropic API key **or** AWS
  access to Amazon Bedrock (see step 3).

You do **not** need to create a virtualenv or run `pip` yourself — `uv` handles it.

---

## 2. Install

From the project root:

```bash
# Core agent + CLI + evaluator + offline tests
uv sync

# ...or include the optional Streamlit UI as well
uv sync --extra ui
```

`uv sync` creates a project virtualenv at `.venv/` and installs exactly the
pinned dependency set. Every command below is prefixed with `uv run`, which uses
that environment automatically.

Verify the install without needing any credentials:

```bash
uv run pytest -q          # expect: 15 passed
uv run data-agent --help  # prints CLI usage
```

---

## 3. Configure a provider

Copy the template and edit it:

```bash
cp .env.example .env
```

Real environment variables always override `.env`, so you can also export values
inline (e.g. `LLM_PROVIDER=bedrock uv run ...`).

### Option A — Direct Anthropic API

In `.env`:

```dotenv
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5     # or any model your account can access
```

Get a key at <https://console.anthropic.com/>.

### Option B — Amazon Bedrock

In `.env`:

```dotenv
LLM_PROVIDER=bedrock
AWS_REGION=us-east-1
BEDROCK_MODEL=us.anthropic.claude-sonnet-4-6   # a model/inference-profile you're entitled to
```

AWS credentials are **not** put in `.env` — they come from your normal AWS chain
(environment variables, `~/.aws/credentials`, or an assumed role). To use a named
profile, pass it per-command:

```bash
AWS_PROFILE=your-profile uv run data-agent --csv data/sales.csv --query "..."
```

If you're unsure which Bedrock model IDs you can use, list them (read-only):

```bash
AWS_PROFILE=your-profile aws bedrock list-foundation-models \
  --region us-east-1 --by-provider anthropic \
  --query 'modelSummaries[].modelId' | head
```

### Optional tuning (any provider)

These have sensible defaults in `config.py`; override in `.env` only if needed:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_TEMPERATURE` | `0` | Sampling temperature (ignored by adaptive-thinking models). |
| `LLM_MAX_TOKENS` | `4096` | Max output tokens per model turn. |
| `AGENT_MAX_STEPS` | `8` | Hard cap on reason→act→observe iterations. |
| `LLM_TIMEOUT_SECONDS` | `60` | Per-request timeout. |
| `LLM_MAX_RETRIES` | `3` | SDK auto-retries on 429/5xx/network. |
| `TRACE_DIR` | `traces` | Where JSONL reasoning traces are written. |

---

## 4. Run

```bash
# Ask one question (add -v to stream the reasoning trace to your terminal)
uv run data-agent \
  --csv data/sales.csv \
  --query "Which region generated the most net revenue?"

# Run the evaluation scenarios
uv run data-agent-eval --scenarios tests/scenarios.json

# Launch the web UI (requires: uv sync --extra ui)
uv run streamlit run app/streamlit_app.py
```

With Bedrock and a named profile, prefix each command with `AWS_PROFILE=...`.

Every run prints a trace path like `traces/<run_id>.jsonl` — that file holds the
full reasoning chain (thoughts, tool calls, observations, final answer).

---

## 5. Docker (optional)

```bash
docker build -t data-analysis-agent .

# CLI
docker run --rm --env-file .env -v "$PWD/data:/app/data" data-analysis-agent \
  data-agent --csv data/sales.csv --query "Top category by average price?"

# UI
docker run --rm -p 8501:8501 --env-file .env data-analysis-agent \
  streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

For Bedrock inside Docker, pass AWS credentials through `--env-file` or mount
`~/.aws` (`-v "$HOME/.aws:/root/.aws:ro"`).

---

## 6. Troubleshooting

**`uv sync` tries to compile numpy / pyarrow / Pillow from source and fails.**
On hosts with an older glibc (≈2.26), the newest wheels aren't available, so `uv`
falls back to a source build (which needs a C toolchain / jpeg headers). This
project already pins compatible versions (`numpy<2.3`, and for the UI extra an
older `streamlit`/`pyarrow`/`pillow`). If you're on a **modern** machine you can
loosen those pins in `pyproject.toml` for the latest versions.

**`botocore ... ProfileNotFound`.** The `AWS_PROFILE` name doesn't match a
profile in `~/.aws/config` / `~/.aws/credentials`. Check with
`aws configure list-profiles`.

**`Error: CSV not found`.** Pass a path relative to your current directory, e.g.
`--csv data/sales.csv` from the project root.

**The model refuses or an LLM error is returned.** The agent degrades gracefully
(it returns an error message rather than crashing). Check the printed trace path
for the recorded reason, and confirm your API key / AWS entitlement and the model
ID in `.env`.

**Streamlit command not found.** You installed the core set only — re-run
`uv sync --extra ui`.
