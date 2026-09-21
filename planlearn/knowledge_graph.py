"""Static knowledge graph derived from the chunked textbook.

- Tree edges come for free from each chunk's "Chapter > Section > Subsection" path.
- Prerequisite edges come from explicit cross-references in the text
  ("as discussed in Section 3.1", "see Chapter 6"), a deterministic, auditable
  signal. An LLM can add more edges later (see GraphPlannerAgent).
- Importance = how often other parts of the book point at a node (in-degree).
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .models import Chunk, NodeType

SEP = " > "
_NUM = re.compile(r"^(\d+(?:\.\d+)*)\s")
_XREF = re.compile(r"\b(?:Sections?|Chapters?)\s+(\d+(?:\.\d+){0,2})")
_MATH_HINTS = re.compile(r"[βθλσμ∑∏∫≤≥≈∂]|\bproof\b|\btheorem\b|\bderive\b", re.I)


@dataclass
class KGNode:
    id: str
    title: str
    parent: str | None
    children: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)   # chunks directly under this node
    number: str | None = None                            # "6.2" etc.


class KnowledgeGraph:
    def __init__(self, chunks: list[Chunk]):
        self.chunks: dict[str, Chunk] = {c.id: c for c in chunks}
        self.nodes: dict[str, KGNode] = {}
        self.by_number: dict[str, str] = {}
        self.prereqs: dict[str, set[str]] = defaultdict(set)   # node -> nodes it depends on
        self.in_degree: dict[str, int] = defaultdict(int)
        # text around each reference: *why* src needs tgt ("cross-validation error ... Chapter 5")
        self.xref_context: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._build_tree()
        self._build_xrefs()

    # ---------- construction ----------
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "KnowledgeGraph":
        chunks = []
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                chunks.append(Chunk(r["id"], r["text"], r["section"], r["page_start"],
                                    r["page_end"], r["is_lab"], r["tokens"]))
        return cls(chunks)

    def _build_tree(self) -> None:
        for c in self.chunks.values():
            parts = c.section.split(SEP)
            for depth in range(1, len(parts) + 1):
                nid = SEP.join(parts[:depth])
                if nid not in self.nodes:
                    parent = SEP.join(parts[: depth - 1]) or None
                    title = parts[depth - 1]
                    m = _NUM.match(title)
                    self.nodes[nid] = KGNode(nid, title, parent, number=m.group(1) if m else None)
                    if parent:
                        self.nodes[parent].children.append(nid)
                    if m:
                        self.by_number.setdefault(m.group(1), nid)
            self.nodes[c.section].chunk_ids.append(c.id)

    def _build_xrefs(self) -> None:
        for c in self.chunks.values():
            src = c.section
            for m in _XREF.finditer(c.text):
                num = m.group(1)
                tgt = self.by_number.get(num)
                if tgt and not src.startswith(tgt) and not tgt.startswith(src):
                    if self._book_order(tgt) < self._book_order(src):
                        # earlier section referenced -> prerequisite
                        ctx = c.text[max(0, m.start() - 220): m.end() + 60]
                        self.xref_context[(src, tgt)].append(ctx)
                        if tgt not in self.prereqs[src]:
                            self.prereqs[src].add(tgt)
                            self.in_degree[tgt] += 1

    # ---------- queries ----------
    def why_needed(self, dependent: str, prereq: str) -> list[str]:
        """Reference contexts from anywhere inside `dependent` pointing into `prereq`."""
        return [ctx for (src, tgt), cs in self.xref_context.items()
                if src.startswith(dependent) and (tgt == prereq or tgt.startswith(prereq + " > "))
                for ctx in cs]

    def _book_order(self, nid: str) -> int:
        ids = self.subtree_chunk_ids(nid)
        return min(int(i.split("-")[-1]) for i in ids) if ids else 10**9

    def subtree_chunk_ids(self, nid: str) -> list[str]:
        n = self.nodes[nid]
        out = list(n.chunk_ids)
        for ch in n.children:
            out.extend(self.subtree_chunk_ids(ch))
        return sorted(out)

    def subtree_chunks(self, nid: str) -> list[Chunk]:
        return [self.chunks[i] for i in self.subtree_chunk_ids(nid)]

    def pages(self, nid: str) -> tuple[int, int]:
        cs = self.subtree_chunks(nid)
        return (min(c.page_start for c in cs), max(c.page_end for c in cs)) if cs else (0, 0)

    def prerequisites_of(self, nid: str) -> set[str]:
        """Prereqs of this node or anything inside it, excluding its own subtree."""
        out: set[str] = set()
        stack = [nid]
        while stack:
            cur = stack.pop()
            out |= self.prereqs.get(cur, set())
            stack.extend(self.nodes[cur].children)
        return {p for p in out if not p.startswith(nid)}

    def classify(self, nid: str) -> NodeType:
        return self.classify_chunks(self.subtree_chunks(nid), nid)

    @staticmethod
    def classify_chunks(cs: list[Chunk], nid: str = "") -> NodeType:
        """Type from content: labs -> coding, dense notation -> math, else concept."""
        if "Exercises" in nid or any("Exercises" in c.section for c in cs[:1]):
            return NodeType.EXERCISE
        if cs and sum(c.is_lab for c in cs) / len(cs) > 0.5:
            return NodeType.CODING
        text = " ".join(c.text for c in cs)[:20000]
        density = len(_MATH_HINTS.findall(text)) / max(1, len(text) / 1000)
        return NodeType.MATH if density > 6 else NodeType.CONCEPT

    def section_of_chunk(self, chunk_id: str) -> str:
        return self.chunks[chunk_id].section
