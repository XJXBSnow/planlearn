"""LLM paths tested with a fake model, so they run offline and in CI."""
from planlearn import build_system, UserProfile, Level
from conftest import CHUNKS


class FakeLLM:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def complete_json(self, system, prompt):
        self.calls += 1
        return self.fn(system, prompt)


def test_llm_judge_changes_selection():
    import json

    def judge(system, prompt):
        if "pedagogical evaluator" not in system:
            return None
        ids = [c["id"] for c in json.loads(prompt)["candidates"]]
        return {"scores": {i: (1.0 if "Ridge Regression" in i else 0.0) for i in ids}}

    llm = FakeLLM(judge)
    plan = build_system(CHUNKS, llm=llm).run(UserProfile("regularization", Level.BEGINNER),
                                             use_cache=False)
    assert llm.calls > 0
    assert any("Ridge Regression" in s["id"] for s in plan["roadmap"])


def test_unparseable_llm_output_falls_back_to_heuristics():
    llm = FakeLLM(lambda s, p: None)
    plan = build_system(CHUNKS, llm=llm).run(
        UserProfile("learn lasso and ridge regression", Level.BEGINNER), use_cache=False)
    assert any("6.2.2 The Lasso" in s["id"] for s in plan["roadmap"])
