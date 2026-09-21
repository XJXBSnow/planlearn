import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHUNKS = ROOT / "data" / "islp_chunks.jsonl"


@pytest.fixture(scope="session")
def kg():
    from planlearn.knowledge_graph import KnowledgeGraph
    return KnowledgeGraph.from_jsonl(CHUNKS)


@pytest.fixture
def system():
    from planlearn import build_system
    return build_system(CHUNKS)
