import pytest

from planlearn import UserProfile, Level
from planlearn.models import AgentResult, NextAction
from planlearn.orchestrator import ALLOWED, State, TransitionError

GOAL = UserProfile("learn lasso and ridge regression", Level.BEGINNER)


def plan_for(system, profile=None):
    return system.run(profile or UserProfile(GOAL.goal, GOAL.level), use_cache=False)


def test_goal_topics_are_covered(system):
    titles = " | ".join(s["id"] for s in plan_for(system)["roadmap"])
    assert "6.2.1 Ridge Regression" in titles
    assert "6.2.2 The Lasso" in titles
    assert "Cross-Validation" in titles          # prerequisite found via cross-reference


def test_every_slot_fits(system):
    plan = plan_for(system)
    for s in plan["roadmap"]:
        node = next(n for n in plan["graph"]["nodes"] if n["id"] == s["id"])
        # a single chunk can't be split further; merges get 10% tolerance
        assert s["minutes"] <= plan["slot_minutes"] * 1.1 or len(node["chunk_ids"]) == 1


def test_prerequisites_come_first(system):
    plan = plan_for(system)
    seen = set()
    for s in plan["roadmap"]:
        assert set(s["after"]) <= seen, f"{s['id']} scheduled before its prerequisites"
        seen.add(s["id"])


def test_audit_only_contains_allowed_transitions(system):
    plan_for(system)
    for e in system.audit.events:
        if e["event"] == "transition":
            assert State(e["to"]) in ALLOWED[State(e["frm"])]


def test_orchestrator_rejects_illegal_transition(system):
    system.state = State.PLAN
    with pytest.raises(TransitionError):
        system._go(State.MERGE)


def test_research_budget_is_enforced(system):
    """Bounded autonomy: an evaluator that always asks for research is capped."""
    real = system.evaluator.score

    def always_tie(*a, **kw):
        res = real(*a, **kw)
        if not kw.get("researched"):
            res.suggestion, res.payload["tied"] = NextAction.CALL_RESEARCHER, [n.id for n in a[0][:2]]
        return res

    system.evaluator.score = always_tie
    plan_for(system)
    calls = [e for e in system.audit.events if e["event"] == "agent_call" and e["agent"] == "researcher"
             and e["state"] == "research"]
    assert len(calls) == system.cfg.limits.max_research_calls


def test_cache_hit_for_similar_request(system):
    system.run(UserProfile("learn lasso and ridge regression", Level.BEGINNER))
    again = system.run(UserProfile("I want to learn ridge regression and lasso", Level.BEGINNER))
    assert again["from_cache"]
    other_level = system.run(UserProfile("learn lasso and ridge regression", Level.ADVANCED))
    assert not other_level["from_cache"]


def test_feedback_known_topic_removes_it(system):
    p = UserProfile(GOAL.goal, GOAL.level)
    before = plan_for(system, p)
    assert any("Cross-Validation" in s["id"] for s in before["roadmap"])
    after = system.apply_feedback(p, {"known": ["Cross-Validation", "Resampling"]})
    assert not any("Cross-Validation" in s["id"] for s in after["roadmap"])


def test_slower_pace_means_more_slots(system):
    p = UserProfile(GOAL.goal, GOAL.level)
    base = plan_for(system, p)
    slower = system.apply_feedback(p, {"pace_factor": 1.5})
    assert slower["total_minutes"] > base["total_minutes"]


def test_alternates_offered_for_review(system):
    plan = plan_for(system)
    assert plan["awaiting_review"] and plan["alternates"]
