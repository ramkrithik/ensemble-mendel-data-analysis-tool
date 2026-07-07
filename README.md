# PC Build Agent

An agentic AI assistant that configures a **working, compatible PC build**. You
describe what you want ("1080p gaming PC around $1200, prefer AMD"); the agent
gathers requirements, reasons over a real components dataset, searches for parts,
**verifies compatibility with a deterministic engine**, and proposes a build. Then
you give feedback ("make it cheaper", "switch to Intel") and it amends the build.

Built on **Claude** through a provider-neutral interface with two interchangeable
backends — the direct Anthropic API and **Amazon Bedrock**. The agent loop is
hand-rolled (no framework) so every reason → plan → act → observe step is explicit
and fully traced.

Dataset: [vinayak-ensemble/Computer_Components_Dataset](https://github.com/vinayak-ensemble/Computer_Components_Dataset)
(PCPartPicker-style CSVs). The 8 build-relevant categories are vendored in `data/`.

---

## Why this design

The correctness risk in a build recommender is **incompatible parts** and
**hallucinated specs/prices**. Two decisions address that head-on:

1. **Compatibility is decided by code, not the LLM.** The model proposes a build;
   a deterministic engine (`compatibility.py`) checks socket, form-factor, memory,
   power, and budget rules and returns a typed verdict. The agent re-runs that
   check itself before delivering — the LLM cannot "reason past" a real mismatch.
2. **Every part is grounded in a dataset query.** The model can only pick parts
   returned by `search_components`, and references them by a **stable `uid`** (not
   name) — because in this dataset names are *not* unique (93 different cards are
   all "Gigabyte GAMING OC" at prices from $250 to $2800). Uids eliminate a whole
   class of mispricing bugs (see the Agent Run Report for the bug this caught).

A key data-reasoning wrinkle the dataset forces: **`cpu.csv` has no socket
column**, so CPU↔motherboard compatibility is only checkable after *deriving* each
CPU's socket from its `microarchitecture` (Zen 4 → AM5, Raptor Lake → LGA1700, …).
That derivation lives in `catalog/normalize.py`.

| Decision | Rationale | Trade-off |
|---|---|---|
| Custom loop (no LangChain/LangGraph) | Every step explicit and traceable. | More hand-written code. |
| Provider abstraction (Anthropic + Bedrock) | One env var swaps backends. | A thin interface to maintain. |
| Deterministic compatibility engine | Correctness guarantee independent of the LLM. | Rules are curated, not exhaustive. |
| `uid`-based part references | Names aren't unique → prevents mispricing. | Model must echo uids (prompted + schema-enforced). |
| Socket derived from microarchitecture | The dataset has no CPU socket column. | A curated arch→socket map (unknown → warning). |
| Pydantic models for I/O | Structured, validated tool inputs & final build. | — |

## Architecture

```
   customer message ─────────────────────────────────────────────┐
                                                                  ▼
  data/*.csv ─► Catalog.load()                        ┌──────────────────────────┐
     (normalise: derive CPU socket from microarch,    │   PCBuildAgent  (agent.py)│
      parse memory "count,size" + DDR gen, canonical  │  reason → plan → act →    │
      form factors, assign stable uids)               │  observe → respond        │
                     │                                 │                           │
                     ├───────────► system prompt ─────►│  ┌─ tool router ────────┐ │
                     │  (goal, rules, catalog summary) │  │ search_components    │ │
                     │                                 │  │   → Catalog.search   │ │
                     ▼                                 │  │ check_compatibility  │ │
        ┌────────────────────────┐                    │  │   → CompatChecker    │ │
        │ Compatibility engine    │◄───────────────────┤  │ ask_clarifying_q     │ │
        │ (deterministic rules)   │  re-checked before │  │ propose_build        │ │
        │ socket/form-factor/mem/ │  every delivery     │  └──────────────────────┘ │
        │ power/budget/essentials │                    │  guard-rails: input valid,│
        └────────────────────────┘                    │  refusal, LLM-error       │
                                                       │  fallback, step budget    │
                                                       └───────────┬───────────────┘
                                    feedback turn ◄────────────────┤ (state persists
                                    ("make it cheaper")            │  across chat() calls)
                                                                   ▼ every step
                                                    Tracer → traces/<run_id>.jsonl
                                                                   │
                              ┌────────────────────────────────────┼───────────────┐
                              ▼                                    ▼                ▼
                         CLI (chat + one-shot)          Streamlit UI          Evaluator
```

### Agent goal, tools, decision flow
- **Goal:** produce a logically consistent, compatible, budget-appropriate build
  from real dataset parts, and amend it on feedback.
- **Tools:** `search_components` (dataset query), `check_compatibility`
  (deterministic verification), `ask_clarifying_question` (requirement gathering),
  `propose_build` (structured final answer).
- **Flow:** reason about needs → plan the platform (CPU+socket first) → search for
  real parts → observe results → **check compatibility** → propose; on any error,
  the engine's verdict is fed back and the model fixes the build within the turn.

### Advanced techniques
- **Self-critique / reflection:** the agent must call `check_compatibility` and
  resolve errors before proposing; the loop *re-runs the deterministic check* and
  rejects a non-compatible proposal, feeding the failure back for repair.
- **Chain-of-thought:** the system prompt mandates the reason→plan→act→observe
  cycle; a few-shot example seeds the search→check→propose rhythm.
- **Structured output:** Pydantic models (`Requirements`, `Build`,
  `CompatibilityReport`) + tool-call schemas.

## Project layout

```
src/pc_agent/
  config.py               # all env-driven config (one source of truth)
  agent.py                # reason→plan→act→observe loop + feedback loop + TurnResult
  models.py               # Pydantic: Requirements, BuildPart, Build, CompatibilityReport
  compatibility.py        # deterministic compatibility & sanity engine
  prompts.py              # system prompt + few-shot (separated from logic)
  tracing.py              # JSONL reasoning-trace writer
  cli.py                  # `pc-agent` (interactive chat + one-shot --query)
  evaluate.py             # `pc-agent-eval` harness
  llm/                    # provider-neutral client + Anthropic/Bedrock impls
  catalog/
    catalog.py            # load/query the CSVs, derive columns, assign uids
    normalize.py          # dataset quirks: socket map, memory parsing, form factors
  tools/registry.py       # tool schemas + executors (ToolKit)
app/streamlit_app.py      # chat UI (build tables + trace viewer)
tests/
  scenarios.json          # 5 eval scenarios + expectations
  test_offline.py         # unit + loop tests (no API key/network)
scripts/fetch_dataset.py  # pull the CSVs into data/
data/*.csv                # vendored components dataset (8 categories)
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11+. Full detail in
[`SETUP.md`](SETUP.md).

```bash
uv sync --extra ui              # install agent + UI + dev deps
cp .env.example .env            # then edit .env (pick a provider — see below)
```

Configure `.env` for **one** provider:
- **Anthropic API:** `LLM_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`.
- **Amazon Bedrock:** `LLM_PROVIDER=bedrock`, `AWS_REGION`, `BEDROCK_MODEL`; AWS
  creds come from your normal chain (prefix commands with `AWS_PROFILE=...`).

The dataset is already vendored in `data/`. To refresh it:
`uv run python scripts/fetch_dataset.py --github`.

Dependencies are also exported to `requirements.txt` (core) for non-uv installs:
`pip install -r requirements.txt`.

## Usage — reproduce the example runs

```bash
# One-shot request
uv run pc-agent --query "A 1080p gaming PC, budget about \$1200, prefer AMD"

# Interactive chat — give feedback to amend the build (the feedback loop)
uv run pc-agent
#   you › a gaming PC around \$1000, AMD
#   ... (proposes a build)
#   you › make it cheaper, under \$750
#   ... (amends: keeps compatible parts, swaps the pricey ones)

# Run the evaluation scenarios
uv run pc-agent-eval --scenarios tests/scenarios.json

# Web UI
uv run streamlit run app/streamlit_app.py
```

On Bedrock, prefix with `AWS_PROFILE=your-profile`. Every run prints a
`traces/<run_id>.jsonl` path holding the full reasoning chain.

## Evaluation

`tests/scenarios.json` defines 5 scenarios with LLM-free checks: a mid-range AMD
gaming build (compatible + on-budget + right brand), a no-GPU office build, a
**Mini ITX Intel** build (form-factor + socket reasoning), an **infeasible $150**
request (must not silently ship a broken/over-budget build), and a **feedback**
scenario (the amended build must be compatible *and* cheaper). The harness
re-runs the deterministic compatibility check on each delivered build, so a
"proposed" build that isn't actually buildable fails. See the Agent Run Report
for the live 5/5 results.

## Testing (offline — no API key)

```bash
uv run pytest -q      # 18 tests: normalisation, catalog, full compat rule set,
                      # and the agent loop against a scripted LLM (incompatible-
                      # proposal recovery, feedback loop, LLM-failure fallback).
```

## Robustness & guard-rails
- **LLM failures** — SDK timeouts + automatic retries (429/5xx/network);
  unrecoverable errors return a graceful fallback, not a crash.
- **Refusals** — `stop_reason == "refusal"` handled explicitly.
- **Input validation** — empty/oversized messages rejected up front.
- **Compatibility gate** — a proposed build that fails the deterministic check is
  rejected and sent back for repair; it is never delivered.
- **Infeasible requirements** — the agent gets as close as it can and states the
  trade-off honestly rather than shipping a broken or silently over-budget build.
- **Step budget** — `AGENT_MAX_STEPS` bounds each turn.
- **Full observability** — every thought, tool call, observation, compatibility
  report, and proposal is written to a JSONL trace.

## Docker

```bash
docker build -t pc-build-agent .
docker run --rm --env-file .env pc-build-agent \
  pc-agent --query "A compact Mini ITX Intel build under \$1500"
# UI: docker run --rm -p 8501:8501 --env-file .env pc-build-agent \
#       streamlit run app/streamlit_app.py --server.address=0.0.0.0
```

## Bonus features included
- **Conversation persistence & feedback loop** — the agent keeps state across
  `chat()` calls; follow-ups amend the previous build.
- **Streamlit UI** with build tables and an inline trace viewer.
- **Dockerfile** (uv-based; runs CLI or UI).
