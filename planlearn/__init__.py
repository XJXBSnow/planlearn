"""planlearn: multi-agent planner that turns textbooks into 20-minute learning roadmaps."""
from pathlib import Path

from .agents import EvaluatorAgent, GraphPlannerAgent, ResearcherAgent
from .audit import AuditLog
from .cache import GraphCache
from .config import Config
from .knowledge_graph import KnowledgeGraph
from .llm import LLM, NoLLM
from .models import Level, UserProfile
from .orchestrator import Orchestrator
from .retrieval import TfidfRetriever


def build_system(chunks_path: str | Path, llm: LLM | None = None, retriever=None,
                 web=None, config: Config | None = None, cache_path=None,
                 audit_path=None) -> Orchestrator:
    """Wire up the full system. Defaults run fully offline (TF-IDF, no LLM, no web)."""
    kg = KnowledgeGraph.from_jsonl(chunks_path)
    retriever = retriever or TfidfRetriever(list(kg.chunks.values()))
    config = config or Config()
    return Orchestrator(
        planner=GraphPlannerAgent(kg, retriever, llm or NoLLM()),
        evaluator=EvaluatorAgent(kg, config, llm or NoLLM()),
        researcher=ResearcherAgent(web),
        config=config,
        cache=GraphCache(cache_path),
        audit=AuditLog(audit_path),
    )


__all__ = ["build_system", "Orchestrator", "UserProfile", "Level", "Config"]
