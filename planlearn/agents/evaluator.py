"""Pedagogical Evaluator: scores candidate learning nodes and issues Keep/Revise/Reject.

Scoring is a transparent weighted rubric (every component is logged), so ties
and surprises are explainable. When the beam cut falls on a near-tie, it
suggests calling the Researcher instead of guessing."""
from __future__ import annotations

import json
import math

from ..config import Config
from ..llm import LLM, NoLLM
from ..knowledge_graph import KnowledgeGraph
from ..models import (AgentResult, LearningNode, Level, NextAction, NodeType,
                      UserProfile, Verdict)

LEVEL_FIT = {
    Level.BEGINNER:     {NodeType.CONCEPT: 1.0, NodeType.CODING: 0.85, NodeType.MATH: 0.5, NodeType.EXERCISE: 0.6},
    Level.INTERMEDIATE: {NodeType.CONCEPT: 0.9, NodeType.CODING: 1.0, NodeType.MATH: 0.8, NodeType.EXERCISE: 0.8},
    Level.ADVANCED:     {NodeType.CONCEPT: 0.7, NodeType.CODING: 0.9, NodeType.MATH: 1.0, NodeType.EXERCISE: 0.9},
}


class EvaluatorAgent:
    name = "evaluator"

    def __init__(self, kg: KnowledgeGraph, config: Config, llm: LLM | None = None):
        self.kg, self.cfg, self.llm = kg, config, llm or NoLLM()
        self._max_deg = max(kg.in_degree.values(), default=1)

    def _llm_relevance(self, candidates: list[LearningNode], profile: UserProfile
                       ) -> dict[str, float]:
        """Optional LLM judge: how necessary is each candidate for the goal?
        Catches what lexical retrieval can't ('What Are PCs?' is foundational)."""
        if isinstance(self.llm, NoLLM) or not candidates:
            return {}
        items = [{"id": n.id, "title": n.title, "type": n.node_type.value,
                  "excerpt": " ".join(self.kg.chunks[c].text for c in n.chunk_ids[:1])[:400]}
                 for n in candidates]
        out = self.llm.complete_json(
            "You are a pedagogical evaluator for a learning planner. Rate how necessary "
            "each candidate section is for a learner to reach their goal, 0.0-1.0. "
            "Foundational material the goal builds on is necessary; adjacent topics are not.",
            json.dumps({"goal": profile.goal, "level": profile.level.value,
                        "known": profile.known_topics, "candidates": items,
                        "return": {"scores": {"<id>": "<float 0-1>"}}}))
        scores = (out or {}).get("scores", {})
        return {k: float(v) for k, v in scores.items()
                if k in {n.id for n in candidates} and isinstance(v, (int, float))}

    def relevance_of(self, node: LearningNode, relevance: dict[str, float],
                     chunk_relevance: dict[str, float] | None = None) -> float:
        """Blend the node's *own* content match with relevance inherited from the
        seeded section it came from, so siblings are told apart by what they contain."""
        rid, decay, inherited = node.id.split(" [")[0], 1.0, 0.0
        while rid:
            if rid in relevance:
                inherited = relevance[rid] * decay
                break
            parent = self.kg.nodes[rid].parent if rid in self.kg.nodes else None
            rid, decay = parent, decay * 0.8
        if chunk_relevance is None:
            return inherited
        own = max((chunk_relevance.get(c, 0.0) for c in node.chunk_ids), default=0.0)
        return 0.6 * own + 0.4 * inherited

    def score(self, candidates: list[LearningNode], profile: UserProfile,
              relevance: dict[str, float], trends: dict[str, float] | None = None,
              researched: bool = False, chunk_relevance: dict[str, float] | None = None,
              beam_width: int | None = None) -> AgentResult:
        w, lim, slot = self.cfg.weights, self.cfg.limits, profile.slot_minutes
        k = beam_width or lim.beam_width
        judged = self._llm_relevance(candidates, profile)
        for n in candidates:
            base = n.id.split(" [")[0]
            deg = self.kg.in_degree.get(base, 0)
            if n.minutes <= slot:
                size = 1.0 if n.minutes >= slot * lim.min_slot_fraction else 0.5
            else:
                size = 0.6  # still useful; will be unwound
            lexical = self.relevance_of(n, relevance, chunk_relevance)
            detail = {
                # with an LLM judge, blend its view in; retrieval stays as a grounding signal
                "relevance": (0.4 * lexical + 0.6 * judged[n.id]) if n.id in judged else lexical,
                "importance": math.log1p(deg) / math.log1p(self._max_deg),
                "size_fit": size,
                "level_fit": LEVEL_FIT[profile.level][n.node_type],
                "trend": (trends or {}).get(n.id, 0.5),
            }
            n.score_detail = {k: round(v, 3) for k, v in detail.items()}
            n.score = round(sum(getattr(w, k) * v for k, v in detail.items()), 4)

        ranked = sorted(candidates, key=lambda n: n.score, reverse=True)
        verdicts: dict[str, Verdict] = {}
        eligible = [n for n in ranked if n.score_detail["relevance"] >= lim.min_relevance]
        beam = {n.id for n in eligible[:k] if n.score >= lim.keep_min}
        for n in ranked:
            if n.id in beam:
                verdicts[n.id] = Verdict.KEEP
            elif n.score >= lim.reject_below:
                verdicts[n.id] = Verdict.REVISE
            else:
                verdicts[n.id] = Verdict.REJECT

        # near-tie exactly at the beam cut -> ask for outside evidence
        tie = (len(eligible) > k and not researched
               and abs(eligible[k - 1].score - eligible[k].score) < lim.tie_epsilon)
        if tie:
            tied = [n.id for n in eligible
                    if abs(n.score - eligible[k - 1].score) < lim.tie_epsilon]
            return AgentResult(self.name, {"ranked": ranked, "verdicts": verdicts, "tied": tied},
                               NextAction.CALL_RESEARCHER,
                               f"{len(tied)} candidates within {lim.tie_epsilon} at the beam cut")
        keeps = [n for n in ranked if verdicts[n.id] == Verdict.KEEP]
        return AgentResult(self.name, {"ranked": ranked, "verdicts": verdicts, "tied": []},
                           NextAction.CALL_GRAPH_PLANNER if keeps else NextAction.MERGE,
                           f"keep {len(keeps)}, revise "
                           f"{sum(v == Verdict.REVISE for v in verdicts.values())}, reject "
                           f"{sum(v == Verdict.REJECT for v in verdicts.values())}")
