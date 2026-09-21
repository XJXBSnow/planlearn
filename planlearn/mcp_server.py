"""MCP server exposing planlearn as tools an MCP client (e.g. Claude Code) can call.

Run directly for local testing:
    python -m planlearn.mcp_server

Register with Claude Code (project-scoped, see .mcp.json) or globally:
    claude mcp add planlearn -- <path-to-venv>/bin/python -m planlearn.mcp_server

Env vars (all optional, default to paths inside this repo):
    PLANLEARN_CHUNKS   textbook chunk JSONL to plan from (default: data/islp_chunks.jsonl)
    PLANLEARN_CACHE    plan cache file (default: runs/cache.json)
    PLANLEARN_AUDIT    audit log file (default: runs/audit.jsonl)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import Level, UserProfile, build_system
from .models import NodeType
from .orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = os.environ.get("PLANLEARN_CHUNKS", str(ROOT / "data" / "islp_chunks.jsonl"))
CACHE_PATH = os.environ.get("PLANLEARN_CACHE", str(ROOT / "runs" / "cache.json"))
AUDIT_PATH = os.environ.get("PLANLEARN_AUDIT", str(ROOT / "runs" / "audit.jsonl"))

mcp = MCPServer("planlearn", instructions=(
    "Plans personalized, 20-minute-slot learning roadmaps from an indexed textbook. "
    "Call plan_learning_roadmap for a fresh plan; call refine_plan to re-plan after the "
    "learner gives feedback (topics already known, slots to drop, or a faster/slower pace)."
))

# One Orchestrator per (chunks_path, use_llm) combo: knowledge-graph parsing and TF-IDF
# fitting are the expensive part, so reuse them across calls within a server process.
_systems: dict[tuple[str, bool], Orchestrator] = {}


def _get_system(chunks_path: str, use_llm: bool) -> Orchestrator:
    key = (chunks_path, use_llm)
    if key not in _systems:
        llm = None
        if use_llm:
            from .llm import AnthropicLLM
            llm = AnthropicLLM()
        _systems[key] = build_system(chunks_path, llm=llm, cache_path=CACHE_PATH,
                                     audit_path=AUDIT_PATH)
    return _systems[key]


def _summarize(plan: dict[str, Any]) -> dict[str, Any]:
    """Trim the raw knowledge graph and bookkeeping out of the plan; pass
    full_output=True on the tool call to get everything back instead."""
    return {k: plan[k] for k in (
        "goal", "level", "slot_minutes", "total_minutes", "from_cache",
        "roadmap", "starting_points", "recommendations", "alternates",
    )}


def _filter_by_type(plan: dict[str, Any], node_types: list[str] | None) -> dict[str, Any]:
    """Keep only roadmap slots whose `type` is in node_types (case-insensitive; also
    accepts "math" as an alias for "math_proof"). Re-numbers slots, drops "after"
    references to slots that got filtered out, and recomputes total_minutes /
    starting_points so the trimmed roadmap is internally consistent."""
    if not node_types:
        return plan
    valid = {t.value for t in NodeType}
    wanted = {"math_proof" if t.strip().lower() == "math" else t.strip().lower()
             for t in node_types}
    unknown = wanted - valid
    if unknown:
        raise ValueError(f"Unknown node type(s) {sorted(unknown)}; valid: {sorted(valid)}")

    kept = [s for s in plan["roadmap"] if s["type"] in wanted]
    kept_ids = {s["id"] for s in kept}
    roadmap = []
    for i, s in enumerate(kept):
        s = {**s, "slot": i + 1, "after": [a for a in s["after"] if a in kept_ids]}
        roadmap.append(s)
    return {
        **plan,
        "roadmap": roadmap,
        "total_minutes": round(sum(s["minutes"] for s in roadmap), 1),
        "starting_points": [s["id"] for s in roadmap if not s["after"]],
    }


@mcp.tool()
def plan_learning_roadmap(
    goal: str,
    level: str = "beginner",
    known_topics: list[str] | None = None,
    slot_minutes: int = 20,
    include_exercises: bool = False,
    use_llm: bool = False,
    chunks_path: str | None = None,
    node_types: list[str] | None = None,
    full_output: bool = False,
) -> dict[str, Any]:
    """Plan a personalized roadmap of `slot_minutes`-long learning steps toward `goal`,
    drawn from the indexed textbook. `level` is one of beginner/intermediate/advanced.
    `known_topics` lets the learner skip material they already know. Set use_llm=True to
    let Claude refine prerequisite/relevance judgments (needs ANTHROPIC_API_KEY set in the
    server's environment); by default everything runs offline (TF-IDF + heuristics).
    `node_types` filters the returned roadmap to only these slot types (see
    list_node_types), e.g. ["concept"] for a coding/math-free conceptual walkthrough;
    omit it to get every slot type. full_output=True also returns the raw knowledge
    graph plus rejected/merged bookkeeping (computed before any node_types filtering).
    """
    system = _get_system(chunks_path or DEFAULT_CHUNKS, use_llm)
    profile = UserProfile(goal=goal, level=Level(level), known_topics=list(known_topics or []),
                          slot_minutes=slot_minutes, include_exercises=include_exercises)
    plan = system.run(profile)
    if full_output:
        return plan
    return _filter_by_type(_summarize(plan), node_types)


@mcp.tool()
def refine_plan(
    goal: str,
    level: str = "beginner",
    known_topics: list[str] | None = None,
    slot_minutes: int = 20,
    include_exercises: bool = False,
    use_llm: bool = False,
    chunks_path: str | None = None,
    add_known: list[str] | None = None,
    remove_ids: list[str] | None = None,
    pace_factor: float = 1.0,
    node_types: list[str] | None = None,
    full_output: bool = False,
) -> dict[str, Any]:
    """Re-plan after learner feedback on a previous plan_learning_roadmap call. Pass the
    same goal/level/known_topics/slot_minutes/include_exercises used originally, plus:
    add_known (topics the learner now knows), remove_ids (roadmap slot 'id' values to drop
    entirely), and/or pace_factor (>1 if the learner is slower than estimated, <1 if
    faster) to re-estimate slot durations. `node_types` filters the result the same way
    as in plan_learning_roadmap (see list_node_types).
    """
    system = _get_system(chunks_path or DEFAULT_CHUNKS, use_llm)
    profile = UserProfile(goal=goal, level=Level(level), known_topics=list(known_topics or []),
                          slot_minutes=slot_minutes, include_exercises=include_exercises)
    plan = system.apply_feedback(profile, {
        "known": add_known or [], "remove": remove_ids or [], "pace_factor": pace_factor,
    })
    if full_output:
        return plan
    return _filter_by_type(_summarize(plan), node_types)


@mcp.tool()
def list_levels() -> list[str]:
    """List the valid `level` values accepted by plan_learning_roadmap / refine_plan."""
    return [l.value for l in Level]


@mcp.tool()
def list_node_types() -> list[str]:
    """List the valid `node_types` values (roadmap slot types) accepted by
    plan_learning_roadmap / refine_plan: concept (explanation, no code/proof),
    coding (lab/implementation), math_proof (derivations), exercise (practice)."""
    return [t.value for t in NodeType]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
