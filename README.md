# Data Analysis Agent

An agentic AI assistant that analyses CSV data. You point it at a CSV and ask a
question in plain English; it plans an approach, **writes and executes pandas
code**, reads the result (or the traceback), self-corrects, and returns grounded
insights — never guessed numbers.

Built on **Claude** via a provider-neutral interface with **two interchangeable
implementations**: the direct Anthropic API and **Amazon Bedrock**. The agent
loop is hand-rolled (no framework) so every reason → plan → act → observe step is
explicit and fully traced.

---

## Why this design

The **Data Analysis** domain was chosen because it *forces* the full agent loop:
generated code frequently fails on the first try (wrong column name, dtype
mismatch), so the agent must observe the traceback and self-correct — exercising
autonomous multi-step reasoning, tool use, structured output, and error handling
all at once, and reproducibly offline (just pandas + a model).

Key decisions and trade-offs:

| Decision | Rationale | Trade-off |
|---|---|---|
| **Custom loop, no LangChain/LangGraph** | Every step is explicit and easy to instrument for the required reasoning trace; reads as stronger GenAI engineering. | More code than importing a framework agent. |
| **Provider abstraction (Anthropic + Bedrock)** | One env var swaps backends; satisfies "any LLM provider" and the Bedrock requirement. | A thin interface layer to maintain. |
| **`run_python` as the core tool** | Real computation over the data → answers are grounded, not hallucinated. | Requires sandboxing (see below). |
| **`ask_clarifying_question` tool** | Lets the agent pause on genuine ambiguity instead of guessing. | One extra round-trip when triggered. |
| **`final_answer` tool (structured)** | Machine-checkable output for evaluation; clean UI rendering. | Model must be steered to call it. |
| **In-process code executor with a denylist** | Zero-dependency, fully offline, good enough to demo the loop. | **Not a hard security boundary** — see Security. |

## Architecture

```
                         ┌──────────────────────────────────────────────┐
   CSV ─► Dataset ───────►  System prompt (goal + rules + data profile)  │
          (profile)       └──────────────────────────────────────────────┘
                                            │
   question ──────────────────────────────►│
                                            ▼
                    ┌─────────────────────────────────────────┐
                    │            AGENT LOOP  (agent.py)        │
                    │                                          │
                    │   ┌─► REASON  (model thinks)             │
                    │   │      │                               │
                    │   │      ▼                               │
                    │   │   PLAN + ACT  (emit tool_use)        │
                    │   │      │                               │
                    │   │      ▼                               │
                    │   │   ┌──────────── tool router ───────┐ │
                    │   │   │ run_python  → CodeExecutor      │ │
                    │   │   │ ask_clarify → pause, return     │ │
                    │   │   │ final_answer→ done, return      │ │
                    │   │   └────────────────────────────────┘ │
                    │   │      │ observation (stdout/traceback) │
                    │   └──────┘  fed back as tool_result       │
                    │                                          │
                    │   guard-rails: step budget, input size,  │
                    │   refusal handling, LLM-error fallback   │
                    └──────────────────┬───────────────────────┘
                                       │ every step
                                       ▼
                         Tracer → traces/<run_id>.jsonl
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                         ▼
         CLI (cli.py)          Streamlit UI (app/)      Evaluator (evaluate.py)
```

**LLM abstraction** (`src/data_agent/llm/`): the loop talks only to `LLMClient`;
`build_client` returns the Anthropic-direct or Bedrock implementation from config.
Both share one request/response normalisation path.

### Advanced techniques demonstrated
- **Self-reflection / self-correction** — the loop feeds execution tracebacks back
  to the model, which fixes its own code (see the trace in the run report).
- **Chain-of-thought prompting** — the system prompt mandates an explicit
  reason → plan → act → observe cycle; a few-shot example seeds the rhythm.
- **Structured output** — tool-call schemas (`run_python`, `final_answer`) and a
  typed `AgentResult` give machine-checkable results.

## Project layout

```
src/data_agent/
  config.py            # all env-driven config (one source of truth)
  agent.py             # the reason→plan→act→observe→respond loop + AgentResult
  prompts.py           # system prompt + few-shot (separated from logic)
  tracing.py           # JSONL reasoning-trace writer
  cli.py               # `data-agent` entry point
  evaluate.py          # `data-agent-eval` harness
  llm/                 # provider-neutral client + Anthropic/Bedrock impls
  tools/
    dataset.py         # CSV load + compact profile
    code_executor.py   # sandboxed pandas execution (the "act")
    registry.py        # tool schemas (the model's contract)
app/streamlit_app.py   # optional web UI
tests/
  scenarios.json       # 5 evaluation scenarios + expectations
  test_offline.py      # unit + loop tests (no API key/network needed)
data/sales.csv         # sample dataset
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync --extra ui              # install everything (agent + UI + dev)
cp .env.example .env            # then edit .env (see below)
```

Configure `.env` for **one** provider:

- **Direct Anthropic API:** set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY`
  (and optionally `ANTHROPIC_MODEL`).
- **Amazon Bedrock:** set `LLM_PROVIDER=bedrock`, `AWS_REGION`, and `BEDROCK_MODEL`;
  AWS credentials come from your normal AWS chain (env vars, `~/.aws`, or a role).

## Usage

```bash
# Ask one question (add -v to stream the reasoning trace to your terminal)
uv run data-agent --domain data-analysis \
  --csv data/sales.csv \
  --query "Which region generated the most net revenue?"

# Run the evaluation scenarios
uv run data-agent-eval --scenarios tests/scenarios.json

# Launch the web UI
uv run streamlit run app/streamlit_app.py
```

Every run writes a full reasoning trace to `traces/<run_id>.jsonl` (path printed
at the end). That JSONL is what the UI and the run report read back.

## Evaluation

`tests/scenarios.json` defines 5 representative scenarios with cheap,
LLM-free checks (expected status, keyword presence, a numeric ground-truth
within tolerance, minimum findings), including one **deliberately ambiguous**
scenario where a good agent asks a clarifying question. `data-agent-eval` runs
each, scores it, and prints per-scenario pass/fail plus the trace path so a human
can audit the reasoning chain.

## Testing (offline — no API key)

The deterministic parts — config, profiling, the sandbox and its guard-rails, and
the **entire agent loop against a scripted fake LLM** (including a self-correction
path and an injected API outage) — are covered without any network call:

```bash
uv run pytest -q
```

## Robustness & guard-rails

- **LLM failures** — SDK-level timeouts + automatic retries (429/5xx/network);
  unrecoverable errors are caught and returned as a graceful fallback answer, not
  a crash.
- **Refusals** — `stop_reason == "refusal"` is handled explicitly before reading
  content.
- **Input validation** — empty/oversized questions are rejected up front.
- **Step budget** — `AGENT_MAX_STEPS` bounds the loop so it can't run forever.
- **Full observability** — every thought, tool call, observation, and answer is
  written to a JSONL trace.

## Security

The code executor is **defence-in-depth, not a hard sandbox**: it runs
model-authored code in-process with a restricted builtins map and a denylist
(no `os`/`sys`/`subprocess`/`open`/network/file-writes/dunder access), and
truncates output. This is appropriate for a local demo over trusted data. For
untrusted input or production, replace `CodeExecutor` with a real isolate (a
locked-down subprocess/container, or Anthropic's server-side code-execution tool)
— the interface is deliberately narrow so that swap stays local.

## Bonus features included
- **Streamlit UI** (`app/streamlit_app.py`)
- **Dockerfile** (uv-based; runs CLI or UI)
- **Conversation/session persistence** in the UI via `st.session_state`
- **Streaming/preview hook** — the agent accepts an `on_token` callback for
  real-time reasoning display.

## Docker

```bash
docker build -t data-analysis-agent .
docker run --rm --env-file .env -v "$PWD/data:/app/data" data-analysis-agent \
  data-agent --csv data/sales.csv --query "Top category by average price?"
```
