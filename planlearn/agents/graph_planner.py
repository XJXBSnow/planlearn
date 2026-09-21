"""Graph Planner: retrieves the static KG, finds where the goal lives in the book,
identifies prerequisites, and unwinds large nodes into smaller ones."""
from __future__ import annotations

import json
from collections import defaultdict

from ..estimate import estimate_minutes
from ..knowledge_graph import SEP, KnowledgeGraph
from ..llm import LLM, NoLLM
from ..models import AgentResult, LearningGraph, LearningNode, NextAction, NodeType, UserProfile
from ..retrieval import Retriever

PREREQ_SYSTEM = (
    "You identify prerequisites between textbook sections for a learning planner. "
    "Only choose from the candidate section ids provided."
)


class GraphPlannerAgent:
    name = "graph_planner"

    def __init__(self, kg: KnowledgeGraph, retriever: Retriever, llm: LLM | None = None):
        self.kg, self.retriever, self.llm = kg, retriever, llm or NoLLM()

    # ----- search start: where does the goal live, and what does it need? -----
    def seed(self, profile: UserProfile, n_targets: int) -> AgentResult:
        hits = self.retriever.search(profile.goal, k=150)
        top_hit = max((h.score for h in hits), default=1.0)
        chunk_relevance = {h.chunk.id: h.score / top_hit for h in hits}
        by_section: dict[str, float] = defaultdict(float)
        for h in hits[:40]:
            parts = h.chunk.section.split(SEP)
            # aggregate at section level (depth 2) so one strong subsection doesn't dominate
            key = SEP.join(parts[:2]) if len(parts) >= 2 else parts[0]
            if "Exercises" in key and not profile.include_exercises:
                continue
            by_section[key] += h.score
        top = max(by_section.values(), default=1.0)
        relevance = {k: v / top for k, v in by_section.items()}
        targets = sorted(relevance, key=relevance.get, reverse=True)[:n_targets]

        prereqs: dict[str, set[str]] = {t: self._prereqs(t, profile) for t in targets}
        required = set().union(*prereqs.values()) - set(targets) if prereqs else set()
        for p in required:  # prerequisites inherit some relevance from what needs them
            relevance[p] = max(relevance.get(p, 0.0), 0.5)
            # Which *part* of the prerequisite matters? Ask the book: retrieve with the
            # sentences that reference it ("...cross-validation error... Chapter 5").
            contexts = [c for t in targets for c in self.kg.why_needed(t, p)]
            if contexts:
                phits = [h for h in self.retriever.search(" ".join(contexts), k=60)
                         if h.chunk.section.startswith(p)]
                ptop = max((h.score for h in phits), default=1.0)
                for h in phits:
                    chunk_relevance[h.chunk.id] = max(chunk_relevance.get(h.chunk.id, 0.0),
                                                      0.9 * h.score / ptop)

        nodes = [self.make_node(n, profile) for n in targets + sorted(required)]
        for n in nodes:
            n.role = "target" if n.id in targets else "prereq"
        return AgentResult(
            self.name,
            {"candidates": nodes, "relevance": relevance, "prereqs": prereqs,
             "chunk_relevance": chunk_relevance},
            NextAction.CONTINUE,
            f"{len(targets)} target sections, {len(required)} prerequisites",
        )

    def _prereqs(self, nid: str, profile: UserProfile) -> set[str]:
        found = self.kg.prerequisites_of(nid)
        refined = self._llm_prereqs(nid, found)
        if refined is not None:
            found = refined
        # collapse to section level and drop what the user already knows
        out = set()
        for p in found:
            parts = p.split(SEP)
            sec = SEP.join(parts[:2]) if len(parts) >= 2 else p
            if not self._known(sec, profile) and not nid.startswith(sec):
                out.add(sec)
        return out

    def _llm_prereqs(self, nid: str, candidates: set[str]) -> set[str] | None:
        if isinstance(self.llm, NoLLM):
            return None
        excerpt = " ".join(c.text for c in self.kg.subtree_chunks(nid))[:6000]
        pool = sorted(candidates | {p for p in self.kg.nodes if p.count(SEP) == 1
                                    and self.kg._book_order(p) < self.kg._book_order(nid)})
        out = self.llm.complete_json(PREREQ_SYSTEM, json.dumps({
            "section": nid, "excerpt": excerpt, "candidate_sections": pool[:80],
            "task": "Return {\"prerequisites\": [ids]} a learner must know first. Max 6.",
        }))
        if not out or not isinstance(out.get("prerequisites"), list):
            return None
        return {p for p in out["prerequisites"] if p in self.kg.nodes}

    @staticmethod
    def _known(nid: str, profile: UserProfile) -> bool:
        title = nid.split(SEP)[-1].lower()
        return any(k.lower() in title for k in profile.known_topics)

    # ----- node construction and unwinding -----
    def make_node(self, nid: str, profile: UserProfile) -> LearningNode:
        ntype = self.kg.classify(nid)
        chunks = self.kg.subtree_chunks(nid)
        return LearningNode(
            id=nid, title=nid.split(SEP)[-1], node_type=ntype,
            minutes=estimate_minutes(chunks, ntype, profile.level, profile.pace_factor),
            chunk_ids=[c.id for c in chunks], pages=self.kg.pages(nid),
            parent=self.kg.nodes[nid].parent,
        )

    def expand(self, node: LearningNode, profile: UserProfile
               ) -> tuple[LearningNode | None, list[LearningNode]]:
        """(intro, children). Children are a *choice* for the evaluator; the intro (text
        before the first subsection) is part of the parent and is always included.
        No children -> the caller splits the node mechanically."""
        kg_node = self.kg.nodes.get(node.id)
        children = []
        if kg_node and kg_node.children:
            for ch in kg_node.children:
                if "Exercises" in ch and not profile.include_exercises:
                    continue
                if self._known(ch, profile):
                    continue
                children.append(self.make_node(ch, profile))
        intro = None
        own = [self.kg.chunks[c] for c in kg_node.chunk_ids] if kg_node else []
        if children and own:   # text before the first subsection
            intro = self._part_node(node, own, "intro", profile)
        for c in children + ([intro] if intro else []):
            c.parent, c.role = node.id, node.role
        return intro, children

    def split(self, node: LearningNode, profile: UserProfile) -> list[LearningNode]:
        """Pack consecutive chunks into parts that fit one slot."""
        chunks = [self.kg.chunks[c] for c in node.chunk_ids]
        groups, cur = [], []
        for c in chunks:
            trial = cur + [c]
            if cur and estimate_minutes(trial, self.kg.classify_chunks(trial), profile.level,
                                        profile.pace_factor) > profile.slot_minutes:
                groups.append(cur)
                cur = [c]
            else:
                cur = trial
        if cur:
            groups.append(cur)
        if len(groups) <= 1:
            return []   # a single oversized chunk can't be split further
        return [self._part_node(node, g, f"part {i}/{len(groups)}", profile)
                for i, g in enumerate(groups, 1)]

    def _part_node(self, parent: LearningNode, chunks, label: str, profile: UserProfile) -> LearningNode:
        ntype = self.kg.classify_chunks(chunks)   # a part is typed by its own content
        return LearningNode(
            id=f"{parent.id} [{label}]", title=f"{parent.title} [{label}]",
            node_type=ntype,
            minutes=estimate_minutes(chunks, ntype, profile.level, profile.pace_factor),
            chunk_ids=[c.id for c in chunks],
            pages=(min(c.page_start for c in chunks), max(c.page_end for c in chunks)),
            parent=parent.id,
        )

    # ----- graph maintenance -----
    def append(self, graph: LearningGraph, nodes: list[LearningNode]) -> AgentResult:
        for n in nodes:
            graph.add(n)
        self._link_prereqs(graph)
        big = [n.id for n in nodes if n.minutes > 0 and n.id in graph.nodes]
        return AgentResult(self.name, {"appended": [n.id for n in nodes]},
                           NextAction.EXPAND if big else NextAction.CONTINUE)

    def _link_prereqs(self, graph: LearningGraph) -> None:
        ids = list(graph.nodes)
        for nid in ids:
            base = nid.split(" [")[0]
            if base not in self.kg.nodes:
                continue
            for p in self.kg.prerequisites_of(base):
                for other in ids:
                    ob = other.split(" [")[0]
                    if ob == base or base.startswith(ob + SEP) or ob.startswith(base + SEP):
                        continue   # never link a node to its own ancestor/descendant
                    if other != nid and (ob == p or p.startswith(ob + SEP) or ob.startswith(p + SEP)):
                        graph.prereq_edges.add((other, nid))
