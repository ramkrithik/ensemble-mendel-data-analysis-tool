# Agent Run Report — PC Build Agent

Real traces and evaluation results captured from **live runs against Amazon
Bedrock** (`us.anthropic.claude-sonnet-4-6`). Every build, price, and verdict
below was produced by the agent reasoning over the vendored components dataset —
nothing here is illustrative.

---

## 1. Architecture

```
   customer message
        │
   Catalog.load(data/*.csv)  ── normalise ──►  derive CPU socket from microarch,
        │                                       parse memory "count,size" + DDR gen,
        │                                       canonical form factors, assign uids
        ▼
   PCBuildAgent  (reason → plan → act → observe → respond)
        │  tools:
        │    search_components   → Catalog.search   (grounded part lookup, returns uids)
        │    check_compatibility → CompatibilityChecker (deterministic verdict)
        │    ask_clarifying_question (requirement gathering)
        │    propose_build       → re-checked by the engine before delivery
        │
        ▼
   Deterministic compatibility engine: socket match, form-factor fit, memory
   DDR/capacity, PSU headroom, budget, essentials present.
        │
        ▼
   TurnResult (build + report + rationale)  ──► state persists across turns
        │                                        so the next message is FEEDBACK
        └─► Tracer → traces/<run_id>.jsonl  (full reasoning chain, every step)
```

**Goal:** a compatible, budget-appropriate build from real parts, amendable on
feedback. **Compatibility is decided by code, not the model.** **Parts are
referenced by stable `uid`** because names are not unique in this dataset.

---

## 2. Design decisions & trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Deterministic compatibility engine, re-run before delivery | Correctness can't depend on the LLM "reasoning" a mismatch away | Curated rules, not exhaustive |
| Parts referenced by `uid`, not name | Names are not unique (see §5) → prevents mispricing | Model must echo uids (prompt + schema enforce it) |
| CPU socket derived from `microarchitecture` | `cpu.csv` has **no socket column** | Curated arch→socket map; unknown → warning, not crash |
| Custom loop, no framework | Every step explicit & traceable | More code |
| Provider abstraction (Anthropic + Bedrock) | One env var swaps backends | Thin interface to maintain |
| Feedback via persistent conversation state | Natural "amend the build" UX | Must keep tool_use/tool_result history valid (see §5) |

---

## 3. Full agent trace (real) — "1080p gaming PC, ~$1200, prefer AMD"

Run `f03e33485962`, `bedrock:us.anthropic.claude-sonnet-4-6`, 8 steps → **compatible $825.68 build**.

```
user: A 1080p gaming PC, budget about $1200, prefer AMD.
 s1 THINK  Search components, starting with AMD CPUs and AM5 motherboards to define the platform.
 s1 ACT    search_components {category: cpu, socket: AM5, max_price: 250}
 s1 ACT    search_components {category: motherboard, socket: AM5, max_price: 180}
 s1 ACT    search_components {category: video-card, keyword: "RX 7600"}
 s2 THINK  No RX 7600 found; broaden the AMD GPU search and grab memory/storage/PSU/case.
 s2 ACT    search_components {memory ddr_gen:5 min_total_gb:16}, {internal-hard-drive SSD}, {power-supply}, {case}
 s3–s6     Iteratively widens the GPU search (RX 7→AMD→by price→RTX) as keyword filters return empty.
 s7 THINK  "An RTX 3060 12GB at $299.97 is a great 1080p card. Now check compatibility."
 s7 ACT    check_compatibility {cpu: cpu#3, motherboard: motherboard#98, memory: memory#241,
                                video-card: video-card#0, internal-hard-drive:..., power-supply:..., case:...}
 s7 OBSERVE COMPATIBLE — no issues found.
 s8 ACT    propose_build (same parts)  →  total $825.68, compatible=True
 END       status=proposed
```

Two things worth noting from the real run:
- The model references parts by **uid** (`cpu#3`, `video-card#0`), exactly as the
  schema requires.
- The GPU search genuinely took several tries because this dataset has many
  null-priced video-card rows (priced-only filtering hides them); the agent
  **adapted its search strategy** across steps 2–6 rather than giving up — visible
  multi-step reasoning.

---

## 4. Feedback loop (real) — "make it cheaper, under $750"

Two-turn conversation; state persists so turn 2 is treated as an amendment.

| | Total | CPU | Motherboard | GPU | PSU | Case |
|---|---|---|---|---|---|---|
| **Turn 1** ("gaming PC ~$1000, AMD") | **$789.40** | cpu#3 | motherboard#98 | video-card#67 | power-supply#28 | case#125 |
| **Turn 2** ("make it cheaper, under $750") | **$593.02** | cpu#37 | motherboard#181 | video-card#37 | power-supply#51 | case#113 |

Turn-2 first thought (verbatim):
> "Let me swap out the pricier parts — I'll try a cheaper CPU, motherboard, GPU,
> and PSU while keeping the RAM and storage that already fit well."

It **kept the parts that still fit** (memory#241, internal-hard-drive#188), swapped
only the expensive ones, re-ran the compatibility check, and delivered a
compatible build **$196 cheaper** and under the new target — the core "take
feedback and amend" requirement, verified live.

---

## 5. A real bug this design caught (and fixed)

The first live feedback run failed in an instructive way, and fixing it hardened
the system:

1. **Symptom:** turn 2 asked for "cheaper" but the total went *up* to $1088, and
   the model's own thought noted "the compatibility engine is counting … pricing
   differently."
2. **Root cause:** the model had selected a $250 GPU from search but passed the
   **name** `"Gigabyte GAMING OC"` to `propose_build`. That name is shared by **93
   different cards** in the dataset ($250–$2809); name-based lookup returned the
   first ($729), mispricing the build.
3. **Fix:** every catalog row gets a stable **`uid`** (`video-card#67`);
   `search_components` returns it, and `check_compatibility`/`propose_build`
   reference parts by uid. Unresolved references now raise a `part_not_found`
   error instead of silently mispricing.

A second bug surfaced first: a terminal/interrupting tool returned without
emitting a `tool_result` for its `tool_use`, leaving dangling `tool_use` blocks
that made the **next (feedback) turn** 400. Fixed by always closing out every
`tool_use` with a `tool_result` before returning. Both fixes are covered by
offline tests (`test_agent_rejects_incompatible_proposal_then_recovers`,
`test_agent_feedback_loop_amends`).

---

## 6. Evaluation results (real)

`uv run pc-agent-eval --scenarios tests/scenarios.json`,
`bedrock:us.anthropic.claude-sonnet-4-6`. **5/5 passed.** The harness re-runs the
deterministic compatibility check on every delivered build.

| # | Scenario | Tests | Result |
|---|---|---|---|
| 1 | `amd_gaming_midrange` | compatible + within $1200 + AMD + all categories | ✅ compatible, **$825.68** |
| 2 | `office_no_gpu` | compatible budget build, GPU optional | ✅ compatible, **$269.13** |
| 3 | `itx_intel_build` | Mini-ITX form-factor fit + Intel socket | ✅ compatible, Intel |
| 4 | `infeasible_budget` ($150 4K rig) | must not silently ship a broken/over-budget build | ✅ delivered closest build + stated budget trade-off |
| 5 | `feedback_make_cheaper` | amended build compatible **and** cheaper | ✅ **$593.02 < $789.40** |

Every scenario writes JSONL traces (paths printed by the evaluator) for audit.

---

## 7. Robustness (offline, `uv run pytest` — 18/18 passing)

- **Compatibility rules**: socket mismatch, missing essentials, over-budget
  (warning vs error), power scaling with GPU — all asserted against the real data.
- **Normalisation**: socket derivation, memory `"count,size"`/DDR parsing, case
  form-factor acceptance.
- **Agent loop**: valid proposal; **incompatible proposal → engine rejects →
  model recovers** within one turn; clarifying question; **feedback loop amends**;
  injected LLM outage → graceful fallback; empty-input rejection.

---

## 8. Summary

The agent runs end-to-end on Amazon Bedrock: it gathers requirements, grounds
every part in a dataset query, guarantees compatibility with a deterministic
engine (re-checked before delivery), handles an infeasible budget honestly, and
amends builds on feedback across a persistent conversation. 5/5 live eval
scenarios and 18/18 offline tests pass. The main scoped-down area is the rule set
and power model — deliberately curated rather than exhaustive — and the in-repo
catalog is the 8 build-relevant categories; both are structured so they can be
extended without touching the agent loop.
