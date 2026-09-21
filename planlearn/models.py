"""Core data types shared by every agent."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    CONCEPT = "concept"
    CODING = "coding"
    MATH = "math_proof"
    EXERCISE = "exercise"


class Level(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class Verdict(str, Enum):
    KEEP = "keep"        # append to the personalized graph
    REVISE = "revise"    # plausible, stays in the candidate pool
    REJECT = "reject"    # dropped (logged for audit)


class NextAction(str, Enum):
    """What an agent may *suggest* to the orchestrator. The orchestrator decides."""
    CONTINUE = "continue"
    CALL_RESEARCHER = "call_researcher"
    CALL_GRAPH_PLANNER = "call_graph_planner"
    EXPAND = "expand"
    MERGE = "merge"
    REVIEW = "review"
    DONE = "done"


@dataclass
class Chunk:
    id: str
    text: str
    section: str          # "Chapter > Section > Subsection"
    page_start: int
    page_end: int
    is_lab: bool
    tokens: int


@dataclass
class UserProfile:
    goal: str
    level: Level = Level.BEGINNER
    known_topics: list[str] = field(default_factory=list)
    slot_minutes: int = 20
    include_exercises: bool = False
    pace_factor: float = 1.0   # learned from feedback: >1 = user is slower than estimate


@dataclass
class LearningNode:
    id: str                       # section path, or path + " [part k/n]"
    title: str
    node_type: NodeType
    minutes: float                # estimated learning time for this user
    chunk_ids: list[str] = field(default_factory=list)
    pages: tuple[int, int] = (0, 0)
    parent: str | None = None
    score: float = 0.0
    score_detail: dict[str, float] = field(default_factory=dict)
    role: str = "target"          # "target" (the goal) or "prereq" (support material)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["node_type"] = self.node_type.value
        return d


@dataclass
class LearningGraph:
    """The personalized graph the ToT search builds up, one node at a time."""
    nodes: dict[str, LearningNode] = field(default_factory=dict)
    prereq_edges: set[tuple[str, str]] = field(default_factory=set)  # (before, after)
    contains_edges: set[tuple[str, str]] = field(default_factory=set)  # (parent, child)

    def add(self, node: LearningNode) -> None:
        self.nodes[node.id] = node
        if node.parent and node.parent in self.nodes:
            self.contains_edges.add((node.parent, node.id))

    def leaves(self) -> list[LearningNode]:
        parents = {p for p, _ in self.contains_edges}
        return [n for nid, n in self.nodes.items() if nid not in parents]

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "prerequisites": sorted(self.prereq_edges),
            "contains": sorted(self.contains_edges),
        }


@dataclass
class AgentResult:
    """Every agent returns data plus an optional suggestion for the next step."""
    agent: str
    payload: Any
    suggestion: NextAction = NextAction.CONTINUE
    reason: str = ""
