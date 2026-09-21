"""Orchestrator: owns the workflow. Agents only *suggest* next steps; the
orchestrator accepts a suggestion only if it is an allowed transition and
within budget (bounded autonomy). Every step is written to the audit log."""
from __future__ import annotations

import time
from collections import deque
from enum import Enum
from typing import Any

from .agents import EvaluatorAgent, GraphPlannerAgent, ResearcherAgent
from .audit import AuditLog
from .cache import GraphCache
from .config import Config
from .knowledge_graph import SEP
from .models import (AgentResult, LearningGraph, LearningNode, NextAction,
                     UserProfile, Verdict)


class State(str, Enum):
    PLAN = "plan"
    EVALUATE = "evaluate"
    RESEARCH = "research"
    APPEND = "append"
    EXPAND = "expand"
    MERGE = "merge"
    REVIEW = "review"
    DONE = "done"


ALLOWED = {
    State.PLAN: {State.EVALUATE, State.DONE},          # DONE = cache hit
    State.EVALUATE: {State.RESEARCH, State.APPEND, State.MERGE},
    State.RESEARCH: {State.EVALUATE},
    State.APPEND: {State.EXPAND},
    State.EXPAND: {State.EVALUATE, State.MERGE},
    State.MERGE: {State.REVIEW},
    State.REVIEW: {State.DONE},
}
SUGGESTION_TO_STATE = {
    NextAction.CALL_RESEARCHER: State.RESEARCH,
    NextAction.CALL_GRAPH_PLANNER: State.APPEND,
    NextAction.MERGE: State.MERGE,
}


class TransitionError(RuntimeError):
    pass


class Orchestrator:
    def __init__(self, planner: GraphPlannerAgent, evaluator: EvaluatorAgent,
                 researcher: ResearcherAgent, config: Config | None = None,
                 cache: GraphCache | None = None, audit: AuditLog | None = None):
        self.planner, self.evaluator, self.researcher = planner, evaluator, researcher
        self.cfg = config or Config()
        self.cache = cache or GraphCache()
        self.audit = audit or AuditLog()
        self.state = State.PLAN
        self.step = 0

    # ---------------- plumbing ----------------
    def _go(self, to: State, why: str = "") -> None:
        if to not in ALLOWED.get(self.state, set()):
            raise TransitionError(f"{self.state.value} -> {to.value} not allowed")
        self.audit.record(event="transition", step=self.step, frm=self.state.value,
                          to=to.value, why=why)
        self.state = to

    def _call(self, fn, *args, **kwargs) -> AgentResult:
        t0 = time.perf_counter()
        res: AgentResult = fn(*args, **kwargs)
        self.audit.record(event="agent_call", step=self.step, state=self.state.value,
                          agent=res.agent, latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                          suggestion=res.suggestion.value, reason=res.reason)
        return res

    def _accept(self, res: AgentResult, default: State, budget_ok: bool = True) -> State:
        """Bounded autonomy: take the agent's suggestion only if allowed and affordable."""
        want = SUGGESTION_TO_STATE.get(res.suggestion)
        ok = (want is not None and want in ALLOWED[self.state]
              and (budget_ok or want != State.RESEARCH))
        self.audit.record(event="suggestion", step=self.step, agent=res.agent,
                          suggested=res.suggestion.value, accepted=ok,
                          chosen=(want if ok else default).value)
        return want if ok else default

    # ---------------- main entry ----------------
    def run(self, profile: UserProfile, use_cache: bool = True) -> dict[str, Any]:
        """ToT as BFS: the root step picks the best seed sections; after that each step
        expands ONE node from the frontier, evaluates only its children, and keeps the
        best `beam_width`. Oversized keeps join the frontier; siblings that were
        plausible but not kept become alternates the user can swap in at review."""
        lim, slot = self.cfg.limits, profile.slot_minutes
        self.state, self.step = State.PLAN, 0
        self.research_calls = 0
        self.audit.record(event="request", goal=profile.goal, level=profile.level.value,
                          known=profile.known_topics)

        cached = (self.cache.get(profile.goal, profile.level.value, profile.known_topics)
                  if use_cache and profile.pace_factor == 1.0 else None)
        if cached:
            self._go(State.DONE, "cache hit")
            return {**cached, "from_cache": True}

        seed = self._call(self.planner.seed, profile, lim.n_targets)
        ctx = {"relevance": seed.payload["relevance"],
               "chunk_relevance": seed.payload["chunk_relevance"]}
        graph = LearningGraph()
        rejected: list[dict] = []
        alternates: dict[str, list[dict]] = {}
        frontier: deque[tuple[LearningNode, int]] = deque()

        # root step: seed sections (goal targets + prerequisites) compete together
        self._go(State.EVALUATE, "seeded candidates")
        self.step = 1
        self._evaluate_and_append(seed.payload["candidates"], "(root)", profile, ctx, graph,
                                  frontier, 0, rejected, alternates, lim.root_beam)

        while frontier and self.step < lim.max_steps and len(graph.nodes) < lim.max_nodes:
            self.step += 1
            node, depth = frontier.popleft()               # BFS order
            intro, children = (self.planner.expand(node, profile)
                               if depth < lim.max_depth else (None, []))
            if not children:                               # nothing structural left
                self._add_parts(graph, node, profile)
                continue
            if intro:                                      # part of the parent, not a choice
                graph.add(intro)
            beam = lim.beam_width if node.role == "target" else lim.prereq_beam
            self._go(State.EVALUATE, f"expand {node.id}")
            self._evaluate_and_append(children, node.id, profile, ctx, graph, frontier,
                                      depth + 1, rejected, alternates, beam)

        self._go(State.MERGE, "search finished")   # loop only exits from EXPAND
        for leftover, _ in frontier:               # budget ran out: split what's left
            self._add_parts(graph, leftover, profile)
        self._finalize_oversized(graph, profile)
        merged = self._merge_small(graph, profile)

        self._go(State.REVIEW, "present for user-in-the-loop review")
        plan = self._roadmap(graph, profile, rejected, merged, alternates)
        if profile.pace_factor == 1.0:   # personalized pace plans aren't shared
            self.cache.put(profile.goal, profile.level.value, profile.known_topics, plan)
        return plan

    def _evaluate_and_append(self, cands: list[LearningNode], parent_id: str,
                             profile: UserProfile, ctx: dict, graph: LearningGraph,
                             frontier: deque, depth: int, rejected: list[dict],
                             alternates: dict[str, list[dict]], beam: int) -> None:
        lim = self.cfg.limits
        ev = self._call(self.evaluator.score, cands, profile, ctx["relevance"],
                        chunk_relevance=ctx["chunk_relevance"], beam_width=beam)
        nxt = self._accept(ev, default=State.APPEND,
                           budget_ok=self.research_calls < lim.max_research_calls)
        if nxt == State.RESEARCH:
            self._go(State.RESEARCH, ev.reason)
            tied = [n for n in cands if n.id in ev.payload["tied"]]
            rs = self._call(self.researcher.trends, tied)
            self.research_calls += 1
            self._go(State.EVALUATE, "re-score with trend signal")
            ev = self._call(self.evaluator.score, cands, profile, ctx["relevance"],
                            trends=rs.payload, researched=True,
                            chunk_relevance=ctx["chunk_relevance"], beam_width=beam)
            nxt = self._accept(ev, default=State.APPEND)

        ranked, verdicts = ev.payload["ranked"], ev.payload["verdicts"]
        if parent_id == "(root)":   # goal sections always survive; prereqs compete for the rest
            for n in ranked:
                if n.role == "target" and n.score_detail["relevance"] >= lim.min_relevance:
                    verdicts[n.id] = Verdict.KEEP
            n_pre = max(0, beam - sum(n.role == "target" and verdicts[n.id] == Verdict.KEEP
                                      for n in ranked))
            pre = [n for n in ranked if n.role == "prereq" and verdicts[n.id] != Verdict.REJECT]
            for i, n in enumerate(pre):
                verdicts[n.id] = Verdict.KEEP if i < n_pre else Verdict.REVISE
        keep = [n for n in ranked if verdicts[n.id] == Verdict.KEEP]
        if not keep and parent_id != "(root)" and ranked and ranked[0].score >= lim.reject_below:
            keep = [ranked[0]]   # coverage: a kept section always contributes its best child
            self.audit.record(event="coverage", step=self.step, container=parent_id,
                              added=keep[0].id, score=keep[0].score)
        for n in ranked:
            if n in keep:
                continue
            item = {"id": n.id, "minutes": n.minutes, "score": n.score, "detail": n.score_detail}
            if verdicts[n.id] == Verdict.REVISE:
                alternates.setdefault(parent_id, []).append(item)
            else:
                rejected.append(item)
        self.audit.record(event="step_result", step=self.step, parent=parent_id,
                          kept=[n.id for n in keep],
                          scores={n.id: n.score for n in ranked})

        self._go(State.APPEND, ev.reason)
        self._call(self.planner.append, graph, keep)
        self._go(State.EXPAND, "queue oversized nodes")
        for n in keep:
            if n.minutes > profile.slot_minutes:
                frontier.append((n, depth))

    # ---------------- post-processing ----------------
    def _add_parts(self, graph: LearningGraph, node: LearningNode, profile: UserProfile) -> None:
        parts = self.planner.split(node, profile)
        for p in parts:
            p.score = node.score
            graph.add(p)
        if parts:
            self.audit.record(event="split", step=self.step, node=node.id, parts=len(parts))

    def _finalize_oversized(self, graph: LearningGraph, profile: UserProfile) -> None:
        """Leaves still over the slot (search budget ran out) get chunk-split."""
        for leaf in list(graph.leaves()):
            if leaf.minutes > profile.slot_minutes:
                self._add_parts(graph, leaf, profile)
        self.planner._link_prereqs(graph)

    def _merge_small(self, graph: LearningGraph, profile: UserProfile) -> list[str]:
        """Merge too-small leaves into an adjacent sibling if the pair still fits a slot."""
        slot, frac = profile.slot_minutes, self.cfg.limits.min_slot_fraction
        merged = []
        changed = True
        while changed:
            changed = False
            leaves = sorted(graph.leaves(), key=lambda n: (n.parent or "", n.pages[0]))
            for a, b in zip(leaves, leaves[1:]):
                if a.parent != b.parent or a.parent is None or b.pages[0] - a.pages[1] > 1:
                    continue
                if min(a.minutes, b.minutes) < slot * frac and a.minutes + b.minutes <= slot * 1.1:
                    combo = LearningNode(
                        id=f"{a.id} + {b.title}", title=f"{a.title} + {b.title}",
                        node_type=a.node_type if a.minutes >= b.minutes else b.node_type,
                        minutes=round(a.minutes + b.minutes, 1),
                        chunk_ids=a.chunk_ids + b.chunk_ids,
                        pages=(min(a.pages[0], b.pages[0]), max(a.pages[1], b.pages[1])),
                        parent=a.parent, score=max(a.score, b.score))
                    for old in (a, b):
                        graph.nodes.pop(old.id)
                        graph.contains_edges.discard((old.parent, old.id))
                        graph.prereq_edges = {(combo.id if x == old.id else x,
                                               combo.id if y == old.id else y)
                                              for x, y in graph.prereq_edges}
                    graph.prereq_edges = {(x, y) for x, y in graph.prereq_edges if x != y}
                    graph.add(combo)
                    merged.append(combo.id)
                    changed = True
                    break
        return merged

    def _roadmap(self, graph: LearningGraph, profile: UserProfile,
                 rejected: list[dict], merged: list[str],
                 alternates: dict[str, list[dict]]) -> dict[str, Any]:
        leaves = graph.leaves()
        leaf_ids = {n.id for n in leaves}

        def root_of(nid: str) -> list[str]:   # map any node to the leaves under it
            if nid in leaf_ids:
                return [nid]
            return [l for l in leaf_ids if l.startswith(nid.split(" [")[0] + SEP)
                    or l.startswith(nid + " [") or l.startswith(nid + " +")]

        deps = {l: set() for l in leaf_ids}
        for a, b in graph.prereq_edges:
            for la in root_of(a):
                for lb in root_of(b):
                    if la != lb:
                        deps[lb].add(la)

        order, done = [], set()
        order_key = lambda n: min((int(c.split("-")[-1]) for c in n.chunk_ids), default=0)
        remaining = sorted(leaves, key=order_key)                 # book order as tiebreak
        while remaining:
            ready = [n for n in remaining if deps[n.id] <= done] or remaining[:1]  # break cycles
            nxt = ready[0]
            order.append(nxt)
            done.add(nxt.id)
            remaining.remove(nxt)

        starts = [n.id for n in order if not deps[n.id]]
        recs = self._call(self.researcher.recommend, order[:3]).payload
        self._go(State.DONE, "roadmap ready")
        return {
            "goal": profile.goal,
            "level": profile.level.value,
            "slot_minutes": profile.slot_minutes,
            "total_minutes": round(sum(n.minutes for n in order), 1),
            "roadmap": [{"slot": i + 1, "id": n.id, "title": n.title,
                         "type": n.node_type.value, "minutes": n.minutes,
                         "pages": list(n.pages), "after": sorted(deps[n.id]),
                         "score": n.score} for i, n in enumerate(order)],
            "starting_points": starts,
            "recommendations": recs,
            "graph": graph.to_dict(),
            "alternates": alternates,   # plausible siblings the user can swap in
            "rejected": rejected,
            "merged": merged,
            "awaiting_review": True,
            "from_cache": False,
        }

    # ---------------- user in the loop ----------------
    def apply_feedback(self, profile: UserProfile, feedback: dict[str, Any]) -> dict[str, Any]:
        """feedback: {"known": [...topics], "pace_factor": float, "remove": [...ids]}.
        Known topics and pace re-run planning; removals are also remembered as known."""
        profile.known_topics = sorted(set(profile.known_topics) | set(feedback.get("known", []))
                                      | {i.split(SEP)[-1] for i in feedback.get("remove", [])})
        if "pace_factor" in feedback:
            profile.pace_factor = float(feedback["pace_factor"])
        self.audit.record(event="feedback", feedback=feedback)
        return self.run(profile, use_cache=False)
