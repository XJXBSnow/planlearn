"""Search limits and scoring weights, the knobs that bound ToT growth and latency."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SearchLimits:
    beam_width: int = 3            # children kept per expanded node ("pick 3 best")
    prereq_beam: int = 2           # prerequisites: "just enough", narrower beam
    root_beam: int = 4             # seed sections kept at the root step
    max_steps: int = 40            # hard stop: nodes expanded
    max_depth: int = 4             # TOC expansion depth before forcing a chunk split
    max_nodes: int = 60            # cap on graph size
    max_research_calls: int = 3    # researcher is expensive (web search)
    n_targets: int = 3             # sections matching the goal to seed the search
    tie_epsilon: float = 0.03      # scores closer than this at the beam cut = a tie
    keep_min: float = 0.35         # below this, even a top-k candidate is not kept
    min_relevance: float = 0.35    # relevance gate for KEEP: stops drift into loosely related topics
    reject_below: float = 0.25     # below this, drop from the pool
    min_slot_fraction: float = 0.3 # leaves shorter than slot*this get merged


@dataclass
class Weights:
    relevance: float = 0.55        # inclusion is mostly "is this what they asked for?"
    importance: float = 0.20
    size_fit: float = 0.05         # size is handled by expansion, not exclusion
    level_fit: float = 0.10        # difficulty shapes pacing more than inclusion
    trend: float = 0.10


@dataclass
class Config:
    limits: SearchLimits = field(default_factory=SearchLimits)
    weights: Weights = field(default_factory=Weights)
