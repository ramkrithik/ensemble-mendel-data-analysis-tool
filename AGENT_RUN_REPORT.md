# Agent Run Report — Data Analysis Agent

This report documents the design of the agent, and shows **real traces and
evaluation results** captured from live runs against **Amazon Bedrock**
(`us.anthropic.claude-sonnet-4-6`). Every number below was produced by the agent
computing over `data/sales.csv` — none are illustrative.

---

## 1. Architecture

```
   CSV ─► Dataset.profile()  ─────────────►  System prompt (goal, rules, CoT
          (shape, dtypes,                     contract, data profile)
           sample, stats)                              │
                                                        │
   question ───────────────────────────────────────────┤
                                                        ▼
              ┌──────────────────── AGENT LOOP (agent.py) ────────────────────┐
              │  REASON → PLAN → ACT (tool_use) → OBSERVE (tool_result) → …    │
              │                                                               │
              │   tool router:                                                │
              │     run_python            → CodeExecutor (pandas sandbox)     │
              │     ask_clarifying_question → pause, return to user           │
              │     final_answer          → structured close-out, done        │
              │                                                               │
              │   guard-rails: step budget · input validation · refusal       │
              │                handling · LLM-error fallback                  │
              └──────────────────────────────┬────────────────────────────────┘
                                             │ every step
                                             ▼
                                Tracer → traces/<run_id>.jsonl
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    ▼                        ▼                         ▼
                CLI              Streamlit UI (app/)          Evaluator (evaluate.py)
```

**Goal.** Answer natural-language questions about a CSV with evidence the agent
computes itself.

**Tools.** `run_python` (execute pandas, observe stdout/traceback — the core
act/observe surface), `ask_clarifying_question` (pause on genuine ambiguity),
`final_answer` (structured, machine-checkable output).

**LLM abstraction.** The loop talks only to an `LLMClient` interface; a factory
returns either the direct Anthropic client or the Bedrock client from one env
var (`LLM_PROVIDER`). This report used the Bedrock implementation.

**Advanced techniques.** (1) *Self-reflection / self-correction* — execution
tracebacks are fed back and the model fixes its own code (demonstrated live in
§3.2). (2) *Chain-of-thought* — the system prompt mandates an explicit
reason→plan→act→observe cycle, seeded by a few-shot example. (3) *Structured
output* — tool schemas + a typed `AgentResult`.

---

## 2. Design decisions & trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Custom loop (no LangChain/LangGraph) | Every step explicit and trivially traceable; stronger demonstration of the mechanics. | More hand-written code. |
| Provider abstraction (Anthropic + Bedrock) | One env var swaps backends; meets the Bedrock requirement without touching the loop. | A thin interface to maintain. |
| `run_python` grounds every answer | Real computation → no hallucinated statistics. | Needs sandboxing. |
| In-process executor + denylist | Zero-dependency, fully offline, enough to demo the loop safely over trusted data. | **Not a hard security boundary** (documented in README → Security). |
| `ask_clarifying_question` tool | Agent pauses on real ambiguity instead of guessing (see §4, scenario 5). | Extra round-trip when it fires. |
| `final_answer` structured tool | Machine-checkable output for the evaluator; clean UI. | Model must be steered to call it. |

---

## 3. Sample agent traces (real)

### 3.1 Straight-line run — "Which region generated the most net revenue?"

Run `95b25e0fe876`, provider `bedrock:us.anthropic.claude-sonnet-4-6`, **2 steps**.

**Step 1 — ACT (`run_python`)**, intent *"Calculate net revenue per region and rank them."*
```python
df['net_revenue'] = df['units'] * df['unit_price'] * (1 - df['discount'])
region_revenue = df.groupby('region')['net_revenue'].sum().sort_values(ascending=False)
print(region_revenue)
```

**Step 1 — OBSERVE** (`ok=true`):
```
region
East     22955.0
North    19839.0
South    19665.0
West     14977.5
Name: net_revenue, dtype: float64
```

**Step 2 — RESPOND (`final_answer`)**:
> The **East region** generated the most net revenue at **$22,955.00**, calculated
> as units × unit_price × (1 − discount) across all orders.
> Findings: East $22,955.00 (highest); North $19,839.00; South $19,665.00; West
> $14,977.50 (lowest).

✅ Independently verified against the CSV: East = 22955.0. Correct.

### 3.2 Self-correction in the wild — "Is there a monthly trend in units sold?"

Run `baf241714ee8`, **4 steps** — this run demonstrates the loop's core value:
the agent wrote code that raised, read the traceback, and fixed itself.

| Step | Action | Result |
|---|---|---|
| 1 | `run_python` — units per month | ✅ ok |
| 2 | `run_python` — month-over-month change | ❌ **`NameError: name 'monthly_units' is not defined`** |
| 3 | `run_python` — same intent, corrected code | ✅ ok |
| 4 | `final_answer` | Delivered |

Verbatim traceback the agent observed at step 2 and recovered from:
```
File "<agent_code>", line 3, in <module>
NameError: name 'monthly_units' is not defined
```
The model had referenced a variable from a *previous* snippet (each `run_python`
call is an independent execution), saw the error, re-declared it, and continued —
exactly the reason→act→**observe-error**→correct behaviour the domain was chosen
to showcase.

### 3.3 Clarifying question — "Which product is the best?"

Run `1bc8e7697dea`. The agent recognised genuine ambiguity and, instead of
guessing a metric, called `ask_clarifying_question`:

> **Question:** How would you like to define "best" — by total revenue (units ×
> unit_price after discount), total units sold, number of orders, or some other
> metric?
>
> **Why:** The ranking of products can differ significantly depending on the
> metric used. A cheap product sold in high volume may top the "units sold" list,
> while an expensive product with fewer sales may lead on revenue.

---

## 4. Evaluation results (real)

Command: `uv run data-agent-eval --scenarios tests/scenarios.json`
Provider: `bedrock:us.anthropic.claude-sonnet-4-6`. **Result: 5/5 passed.**

| # | Scenario | What it tests | Status | Checks |
|---|---|---|---|---|
| 1 | `avg_price_category` | groupby + mean | answered | keyword `electronics` ✓, findings ≥1 ✓ |
| 2 | `total_revenue` | defined-formula sum vs ground truth | answered | numeric **77436.5** within 1.0 ✓ |
| 3 | `top_region_by_revenue` | derived column + argmax | answered | region keyword ✓, findings ≥1 ✓ |
| 4 | `monthly_trend` | date parsing + aggregation (self-corrected) | answered | keyword `month` ✓, findings ≥1 ✓ |
| 5 | `ambiguous_best` | should ask, not guess | **needs_clarification** | status ✓, keyword ✓ |

Scenario 2's numeric check is the strongest signal: the agent's reported total
matched the independently-computed ground truth (77,436.5) to within tolerance,
confirming the answer was *computed*, not approximated.

Every scenario writes a full JSONL trace (paths printed by the evaluator) so each
reasoning chain is auditable.

---

## 5. Robustness (verified offline, `uv run pytest` — 15/15 passing)

- **LLM failure** — an injected outage (`test_agent_handles_llm_failure_gracefully`)
  returns a graceful fallback answer, not a crash.
- **Self-correction** — `test_agent_full_loop_with_self_correction` drives a
  broken-code → traceback → fix → finalise loop against a scripted LLM and asserts
  the traceback appears in the trace.
- **Guard-rails** — dangerous code (`import os`, `open(...)`, dunder access,
  file writes) is refused (`test_executor_blocks_dangerous_code`); the step budget
  is enforced (`test_agent_enforces_step_budget`); empty questions are rejected.
- **Refusals** — `stop_reason == "refusal"` is handled before reading content.
- **Retries/timeouts** — delegated to the Anthropic SDK (429/5xx/network) with the
  configured `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES`.

---

## 6. Summary

The agent runs end-to-end on Amazon Bedrock, answers grounded in computed
evidence, self-corrects on execution errors, asks for clarification on genuine
ambiguity, and passes 5/5 evaluation scenarios plus 15/15 offline tests. The main
production gap is the code sandbox: the in-process executor is defence-in-depth
for a trusted-data demo, and the narrow `CodeExecutor` interface is designed so it
can be swapped for a container or Anthropic's server-side code-execution tool
without touching the agent loop.
