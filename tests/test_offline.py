"""Offline tests — no API key or network required.

Cover the deterministic core (normalisation, catalog queries, the full
compatibility rule set) and the agent loop against a scripted fake LLM (including
a proposal that fails the compatibility gate and must be re-proposed, and an
injected API outage).

Run with:  uv run pytest -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pc_agent.agent import PCBuildAgent
from pc_agent.catalog import Catalog
from pc_agent.catalog import normalize as nz
from pc_agent.compatibility import CompatibilityChecker, estimate_power_draw
from pc_agent.config import load_config
from pc_agent.llm.base import LLMError, LLMResponse, ToolCall
from pc_agent.models import Build, BuildPart, Requirements, UseCase
from pc_agent.tools import ToolKit

DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def catalog():
    return Catalog.load(DATA)


@pytest.fixture
def config(tmp_path):
    os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-used")
    cfg = load_config(env_file=None)
    object.__setattr__(cfg, "trace_dir", tmp_path / "traces")
    object.__setattr__(cfg, "data_dir", DATA)
    return cfg


# ── Normalisation ─────────────────────────────────────────────────────────────

def test_socket_derivation():
    assert nz.socket_for_microarchitecture("Zen 4") == "AM5"
    assert nz.socket_for_microarchitecture("Zen 3") == "AM4"
    assert nz.socket_for_microarchitecture("Raptor Lake") == "LGA1700"
    assert nz.socket_for_microarchitecture("Totally Unknown Arch") is None


def test_memory_parsing():
    assert nz.parse_memory_modules("2,16") == (2, 16)
    assert nz.total_memory_gb("2,16") == 32
    assert nz.parse_memory_ddr("5,6000") == (5, 6000)
    assert nz.total_memory_gb("garbage") is None


def test_case_size_and_acceptance():
    assert nz.canon_case_size("ATX Mid Tower") == "ATX"
    assert nz.canon_case_size("MicroATX Mini Tower") == "Micro ATX"
    assert "Micro ATX" in nz.CASE_ACCEPTS["ATX"]      # big case fits small board
    assert "ATX" not in nz.CASE_ACCEPTS["Mini ITX"]   # small case rejects big board


# ── Catalog ─────────────────────────────────────────────────────────────────

def test_catalog_loads_and_derives(catalog):
    assert set(catalog.categories()) >= {"cpu", "motherboard", "memory"}
    cpu = catalog.frame("cpu")
    assert "socket" in cpu.columns  # derived column present

def test_search_respects_filters(catalog):
    res = catalog.search("cpu", socket="AM5", max_price=200, limit=5)
    assert res, "expected some AM5 CPUs under $200"
    assert all(r["socket"] == "AM5" for r in res)
    assert all(r["price"] <= 200 for r in res)

def test_search_only_returns_priced(catalog):
    res = catalog.search("video-card", limit=20)
    assert all(isinstance(r["price"], (int, float)) for r in res)


# ── Compatibility engine ──────────────────────────────────────────────────────

def _first_case_uid_of_size(catalog, size: str) -> str:
    """Pick a case whose derived canonical size matches (not a name substring)."""
    df = catalog.frame("case")
    df = df[(df["size_canon"] == size) & (df["price"].notna())]
    return str(df.sort_values("price").iloc[0]["uid"])


def _valid_am5_parts(catalog) -> dict[str, str]:
    """Return a {category: uid} map for a valid AM5 build (parts referenced by uid)."""
    cpu = catalog.search("cpu", socket="AM5", limit=1)[0]["uid"]
    mobo = catalog.search("motherboard", socket="AM5", form_factors=["ATX"], limit=1)[0]["uid"]
    mem = catalog.search("memory", ddr_gen=5, min_total_gb=16, limit=1)[0]["uid"]
    psu = catalog.search("power-supply", min_wattage=650, limit=1)[0]["uid"]
    case = _first_case_uid_of_size(catalog, "ATX")  # a case that actually fits an ATX board
    ssd = catalog.search("internal-hard-drive", min_capacity_gb=500, limit=1)[0]["uid"]
    return {"cpu": cpu, "motherboard": mobo, "memory": mem,
            "power-supply": psu, "case": case, "internal-hard-drive": ssd}


def test_valid_build_is_compatible(catalog):
    kit = ToolKit(catalog)
    chk = CompatibilityChecker(catalog)
    build = kit.build_from_parts(_valid_am5_parts(catalog))
    report = chk.check(build, Requirements())
    assert report.compatible, report.summary()


def test_socket_mismatch_is_error(catalog):
    kit = ToolKit(catalog)
    chk = CompatibilityChecker(catalog)
    parts = _valid_am5_parts(catalog)
    parts["motherboard"] = catalog.search("motherboard", socket="LGA1700", limit=1)[0]["uid"]
    report = chk.check(kit.build_from_parts(parts), Requirements())
    assert not report.compatible
    assert any(i.rule == "cpu_socket_mismatch" for i in report.errors)


def test_missing_essentials_is_error(catalog):
    chk = CompatibilityChecker(catalog)
    build = Build(parts=[BuildPart(category="cpu", name="whatever")])
    report = chk.check(build)
    assert not report.compatible
    assert any(i.rule == "missing_essential_parts" for i in report.errors)


def test_over_budget_is_warning(catalog):
    kit = ToolKit(catalog)
    chk = CompatibilityChecker(catalog)
    build = kit.build_from_parts(_valid_am5_parts(catalog))
    report = chk.check(build, Requirements(budget_usd=1.0))
    assert report.compatible  # over-budget is a warning, not a hard error
    assert any(i.rule == "over_budget" for i in report.warnings)


def test_power_estimate_scales_with_gpu(catalog):
    kit = ToolKit(catalog)
    base = kit.build_from_parts(_valid_am5_parts(catalog))
    gpu = catalog.search("video-card", keyword="RTX 4090", limit=1)
    if gpu:
        parts = _valid_am5_parts(catalog)
        parts["video-card"] = gpu[0]["uid"]
        with_gpu = kit.build_from_parts(parts)
        assert estimate_power_draw(with_gpu) > estimate_power_draw(base)


# ── Agent loop with a scripted LLM ────────────────────────────────────────────

class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.model = "scripted"
        self.calls = []

    def complete(self, *, system, messages, tools=None):
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self._responses.pop(0)


def _tool(tc, stop="tool_use"):
    return LLMResponse(text="", tool_calls=[tc], stop_reason=stop, raw_content=[])


def test_agent_proposes_valid_build(catalog, config):
    parts = _valid_am5_parts(catalog)
    scripted = ScriptedLLM([
        _tool(ToolCall("s1", "search_components", {"category": "cpu", "socket": "AM5"})),
        _tool(ToolCall("c1", "check_compatibility", {"parts": parts})),
        _tool(ToolCall("p1", "propose_build", {"parts": parts, "rationale": "Solid AM5 build."})),
    ])
    agent = PCBuildAgent(catalog, scripted, config)
    result = agent.chat("Build me an AM5 PC around $1200")
    assert result.status == "proposed"
    assert result.build is not None and result.report.compatible
    trace = Path(result.trace_path).read_text()
    assert "propose_build" in trace


def test_agent_rejects_incompatible_proposal_then_recovers(catalog, config):
    """Model first proposes a socket-mismatched build; the engine rejects it, and
    the model proposes a valid one on the next turn — all within one chat()."""
    good = _valid_am5_parts(catalog)
    bad = dict(good)
    bad["motherboard"] = catalog.search("motherboard", socket="LGA1700", limit=1)[0]["uid"]
    scripted = ScriptedLLM([
        _tool(ToolCall("p1", "propose_build", {"parts": bad, "rationale": "oops"})),
        _tool(ToolCall("p2", "propose_build", {"parts": good, "rationale": "fixed"})),
    ])
    agent = PCBuildAgent(catalog, scripted, config)
    result = agent.chat("AM5 gaming PC")
    assert result.status == "proposed"
    assert result.report.compatible
    trace = Path(result.trace_path).read_text()
    assert "reject_proposal" in trace  # the safety net fired


def test_agent_clarifying_question(catalog, config):
    scripted = ScriptedLLM([
        _tool(ToolCall("q1", "ask_clarifying_question",
                       {"question": "What's your budget?", "why": "drives every part"})),
    ])
    agent = PCBuildAgent(catalog, scripted, config)
    result = agent.chat("Build me a PC")
    assert result.status == "needs_clarification"
    assert "budget" in result.clarifying_question.lower()


def test_agent_feedback_loop_amends(catalog, config):
    """After a first proposal, feedback triggers a second, amended proposal."""
    good = _valid_am5_parts(catalog)
    scripted = ScriptedLLM([
        _tool(ToolCall("p1", "propose_build", {"parts": good, "rationale": "v1"})),
        _tool(ToolCall("p2", "propose_build", {"parts": good, "rationale": "cheaper v2"})),
    ])
    agent = PCBuildAgent(catalog, scripted, config)
    first = agent.chat("AM5 build ~$1500")
    assert first.status == "proposed"
    second = agent.chat("make it cheaper")   # feedback turn
    assert second.status == "proposed"
    assert "cheaper" in second.message.lower()


def test_agent_handles_llm_failure(catalog, config):
    class Broken:
        model = "broken"
        def complete(self, **_):
            raise LLMError("simulated outage")

    agent = PCBuildAgent(catalog, Broken(), config)
    result = agent.chat("anything")
    assert result.status == "error"
    assert "unavailable" in result.message.lower()


def test_agent_rejects_empty_input(catalog, config):
    agent = PCBuildAgent(catalog, ScriptedLLM([]), config)
    assert agent.chat("   ").status == "error"


def test_requirement_capture_heuristics(catalog, config):
    agent = PCBuildAgent(catalog, ScriptedLLM([]), config)
    agent._maybe_capture_requirements("gaming build with a budget of $1500, prefer AMD")
    assert agent._requirements.budget_usd == 1500.0
    assert agent._requirements.cpu_brand == "AMD"
    assert agent._requirements.use_case == UseCase.GAMING
