"""Learning-time estimates from chunk content.

Rates are *priors*: study pace, not skim pace. Calibrate with real user
feedback via UserProfile.pace_factor (see Orchestrator.apply_feedback).
"""
from __future__ import annotations

from .models import Chunk, Level, NodeType

WORDS_PER_TOKEN = 0.75
WPM = {  # words per minute when studying
    NodeType.CONCEPT: 110,
    NodeType.MATH: 45,
    NodeType.CODING: 70,
    NodeType.EXERCISE: 40,
}
MIN_PER_CODE_BLOCK = 1.5          # typing/running/inspecting output
LEVEL_FACTOR = {Level.BEGINNER: 1.3, Level.INTERMEDIATE: 1.0, Level.ADVANCED: 0.75}


def estimate_minutes(chunks: list[Chunk], node_type: NodeType, level: Level,
                     pace_factor: float = 1.0) -> float:
    words = sum(c.tokens for c in chunks) * WORDS_PER_TOKEN
    minutes = words / WPM[node_type]
    if node_type == NodeType.CODING:
        blocks = sum(c.text.count("```python") for c in chunks)
        minutes += blocks * MIN_PER_CODE_BLOCK
    return round(minutes * LEVEL_FACTOR[level] * pace_factor, 1)
