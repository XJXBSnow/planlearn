from planlearn.estimate import estimate_minutes
from planlearn.knowledge_graph import SEP
from planlearn.models import Level, NodeType

TUNING = "6 Linear Model Selection and Regularization > 6.2 Shrinkage Methods > 6.2.3 Selecting the Tuning Parameter"


def test_tree_from_toc(kg):
    ch6 = kg.nodes["6 Linear Model Selection and Regularization"]
    assert any("6.2 Shrinkage Methods" in c for c in ch6.children)
    assert kg.by_number["6.2.2"].endswith("6.2.2 The Lasso")


def test_cross_reference_prerequisites(kg):
    # "compute the cross-validation error ... as described in Chapter 5"
    assert "5 Resampling Methods" in kg.prerequisites_of(TUNING)
    assert any("cross-validation" in c.lower() for c in kg.why_needed(TUNING, "5 Resampling Methods"))


def test_prereqs_point_backwards(kg):
    for src, deps in kg.prereqs.items():
        for d in deps:
            assert kg._book_order(d) < kg._book_order(src)


def test_classification(kg):
    lab = next(n for n in kg.nodes if "5.3 Lab" in n and n.count(SEP) == 1)
    assert kg.classify(lab) == NodeType.CODING
    assert kg.classify(next(n for n in kg.nodes if n.endswith("Exercises"))) == NodeType.EXERCISE


def test_estimates_scale_with_level_and_type(kg):
    chunks = kg.subtree_chunks(TUNING)
    b = estimate_minutes(chunks, NodeType.CONCEPT, Level.BEGINNER)
    a = estimate_minutes(chunks, NodeType.CONCEPT, Level.ADVANCED)
    m = estimate_minutes(chunks, NodeType.MATH, Level.ADVANCED)
    assert b > a > 0 and m > a
