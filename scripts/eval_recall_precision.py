"""Formal recall/precision evaluation for planlearn, run fully offline (TF-IDF +
heuristics, no LLM, no web) against a fixed set of goals with hand-labeled expected
ISLP sections. This is the benchmark the README's "Known limitations" section calls
for ("write 15-20 goals with expected sections and track recall").

It also scores the automatable subset of the 9 evaluation metrics from the Module 4
(Tree-of-Thought) design doc: granularity, type-hint coverage, dependency integrity,
and whether the plan has a clean set of starting points. The remaining metrics
(faithfulness, clarity, essentialness, feedback-awareness) require human or LLM
judgment and are out of scope for this automated pass -- see the report for why.

Usage:
    python scripts/eval_recall_precision.py [--json eval/recall_precision_results.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as stats
from pathlib import Path
from typing import Any

from planlearn import Level, UserProfile, build_system

ROOT = Path(__file__).resolve().parent.parent
CHUNKS = ROOT / "data" / "islp_chunks.jsonl"

# ---------------------------------------------------------------------------
# Benchmark: 20 goals spanning 10 of ISLP's 13 chapters, each with 1-3 "must
# appear" sections a competent roadmap should include. Sections are full TOC
# breadcrumbs (matching KnowledgeGraph section paths); a goal counts a section
# as "found" if any roadmap leaf's breadcrumb starts with it.
# ---------------------------------------------------------------------------
BENCHMARK: list[dict[str, Any]] = [
    {"goal": "ridge regression and the lasso", "expected": [
        "6 Linear Model Selection and Regularization > 6.2 Shrinkage Methods",
        "6 Linear Model Selection and Regularization > 6.2 Shrinkage Methods > 6.2.1 Ridge Regression",
        "6 Linear Model Selection and Regularization > 6.2 Shrinkage Methods > 6.2.2 The Lasso"]},
    {"goal": "support vector machines", "expected": [
        "9 Support Vector Machines > 9.1 Maximal Margin Classifier",
        "9 Support Vector Machines > 9.2 Support Vector Classifiers",
        "9 Support Vector Machines > 9.3 Support Vector Machines"]},
    {"goal": "logistic regression", "expected": [
        "4 Classification > 4.3 Logistic Regression"]},
    {"goal": "linear discriminant analysis", "expected": [
        "4 Classification > 4.4 Generative Models for Classification > 4.4.1 Linear Discriminant Analysis for p = 1",
        "4 Classification > 4.4 Generative Models for Classification > 4.4.2 Linear Discriminant Analysis for p >1"]},
    {"goal": "decision trees", "expected": [
        "8 Tree-Based Methods > 8.1 The Basics of Decision Trees"]},
    {"goal": "random forests and boosting", "expected": [
        "8 Tree-Based Methods > 8.2 Bagging, Random Forests, Boosting, and Bayesian Additive Regression Trees > 8.2.2 Random Forests",
        "8 Tree-Based Methods > 8.2 Bagging, Random Forests, Boosting, and Bayesian Additive Regression Trees > 8.2.3 Boosting"]},
    {"goal": "principal components analysis", "expected": [
        "12 Unsupervised Learning > 12.2 Principal Components Analysis",
        "12 Unsupervised Learning > 12.2 Principal Components Analysis > 12.2.1 What Are Principal Components?"]},
    {"goal": "k-means and hierarchical clustering", "expected": [
        "12 Unsupervised Learning > 12.4 Clustering Methods > 12.4.1 K-Means Clustering",
        "12 Unsupervised Learning > 12.4 Clustering Methods > 12.4.2 Hierarchical Clustering"]},
    {"goal": "cross-validation", "expected": [
        "5 Resampling Methods > 5.1 Cross-Validation"]},
    {"goal": "the bootstrap", "expected": [
        "5 Resampling Methods > 5.2 The Bootstrap"]},
    {"goal": "regression splines", "expected": [
        "7 Moving Beyond Linearity > 7.4 Regression Splines"]},
    {"goal": "generalized additive models", "expected": [
        "7 Moving Beyond Linearity > 7.7 Generalized Additive Models"]},
    {"goal": "convolutional neural networks", "expected": [
        "10 Deep Learning > 10.3 Convolutional Neural Networks"]},
    {"goal": "recurrent neural networks", "expected": [
        "10 Deep Learning > 10.5 Recurrent Neural Networks"]},
    {"goal": "the bias-variance tradeoff", "expected": [
        "2 Statistical Learning > 2.2 Assessing Model Accuracy > 2.2.2 The Bias-Variance Trade-Off"]},
    {"goal": "survival analysis", "expected": [
        "11 Survival Analysis and Censored Data > 11.1 Survival and Censoring Times",
        "11 Survival Analysis and Censored Data > 11.5 Regression Models With a Survival Response"]},
    {"goal": "multiple testing and the false discovery rate", "expected": [
        "13 Multiple Testing > 13.3 The Family-Wise Error Rate",
        "13 Multiple Testing > 13.4 The False Discovery Rate"]},
    {"goal": "principal components regression and partial least squares", "expected": [
        "6 Linear Model Selection and Regularization > 6.3 Dimension Reduction Methods > 6.3.1 Principal Components Regression",
        "6 Linear Model Selection and Regularization > 6.3 Dimension Reduction Methods > 6.3.2 Partial Least Squares"]},
    {"goal": "best subset and stepwise selection", "expected": [
        "6 Linear Model Selection and Regularization > 6.1 Subset Selection > 6.1.1 Best Subset Selection",
        "6 Linear Model Selection and Regularization > 6.1 Subset Selection > 6.1.2 Stepwise Selection"]},
    {"goal": "neural networks for tabular data", "expected": [
        "10 Deep Learning > 10.1 Single Layer Neural Networks",
        "10 Deep Learning > 10.2 Multilayer Neural Networks"]},
]

BRACKET_SUFFIX = re.compile(r"\s*\[[^\]]*\]\s*$")


def _breadcrumbs(slot_id: str) -> list[str]:
    """A slot id is a full TOC breadcrumb, possibly with a trailing "[intro]" /
    "[part k/n]" tag, or two breadcrumbs joined by " + " (a merged pair of
    small siblings). Return the breadcrumb(s) with any such tag stripped."""
    return [BRACKET_SUFFIX.sub("", part).strip() for part in slot_id.split(" + ")]


def _chapter(breadcrumb: str) -> str:
    return breadcrumb.split(" > ", 1)[0].split(" ", 1)[0]  # leading chapter number


def evaluate_goal(system, entry: dict[str, Any]) -> dict[str, Any]:
    profile = UserProfile(goal=entry["goal"], level=Level.BEGINNER)
    plan = system.run(profile, use_cache=False)
    roadmap = plan["roadmap"]

    all_breadcrumbs = [bc for slot in roadmap for bc in _breadcrumbs(slot["id"])]
    slot_chapters = [_chapter(_breadcrumbs(slot["id"])[0]) for slot in roadmap]
    expected_chapters = {e.split(" ", 1)[0].split(".")[0] for e in entry["expected"]}

    found = [exp for exp in entry["expected"]
            if any(bc.startswith(exp) for bc in all_breadcrumbs)]
    recall = len(found) / len(entry["expected"])
    on_topic = sum(1 for c in slot_chapters if c in expected_chapters)
    precision = on_topic / len(roadmap) if roadmap else 0.0

    slot_min = profile.slot_minutes
    within_band = sum(1 for s in roadmap if 0.3 * slot_min <= s["minutes"] <= 1.15 * slot_min)
    granularity = within_band / len(roadmap) if roadmap else 0.0
    avg_minutes = stats.mean(s["minutes"] for s in roadmap) if roadmap else 0.0

    valid_types = {"concept", "coding", "math_proof", "exercise"}
    type_hint_ok = sum(1 for s in roadmap if s["type"] in valid_types)
    type_hint_coverage = type_hint_ok / len(roadmap) if roadmap else 0.0

    ids = {s["id"] for s in roadmap}
    dep_refs = [a for s in roadmap for a in s["after"]]
    dep_ok = sum(1 for a in dep_refs if a in ids)
    dependency_integrity = dep_ok / len(dep_refs) if dep_refs else 1.0

    off_topic_examples = sorted({bc for bc, c in zip((b[0] for b in
                                 (_breadcrumbs(s["id"]) for s in roadmap)), slot_chapters)
                                 if c not in expected_chapters})[:5]

    return {
        "goal": entry["goal"],
        "expected": entry["expected"],
        "found": found,
        "missed": [e for e in entry["expected"] if e not in found],
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "n_slots": len(roadmap),
        "total_minutes": plan["total_minutes"],
        "avg_slot_minutes": round(avg_minutes, 1),
        "granularity_in_band": round(granularity, 3),
        "type_hint_coverage": round(type_hint_coverage, 3),
        "dependency_integrity": round(dependency_integrity, 3),
        "has_starting_points": bool(plan["starting_points"]),
        "off_topic_chapters_sample": off_topic_examples,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(ROOT / "eval" / "recall_precision_results.json"))
    args = ap.parse_args()

    system = build_system(CHUNKS, cache_path=str(ROOT / "eval" / ".cache.json"),
                          audit_path=str(ROOT / "eval" / ".audit.jsonl"))

    results = [evaluate_goal(system, entry) for entry in BENCHMARK]

    macro_recall = stats.mean(r["recall"] for r in results)
    macro_precision = stats.mean(r["precision"] for r in results)
    summary = {
        "n_goals": len(results),
        "macro_recall": round(macro_recall, 3),
        "macro_precision": round(macro_precision, 3),
        "perfect_recall_goals": sum(1 for r in results if r["recall"] == 1.0),
        "mean_granularity_in_band": round(stats.mean(r["granularity_in_band"] for r in results), 3),
        "mean_type_hint_coverage": round(stats.mean(r["type_hint_coverage"] for r in results), 3),
        "mean_dependency_integrity": round(stats.mean(r["dependency_integrity"] for r in results), 3),
        "goals_missing_starting_points": sum(1 for r in results if not r["has_starting_points"]),
    }

    out = {"summary": summary, "results": results}
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=2))

    print(f"{'goal':<58} {'recall':>7} {'prec':>6} {'slots':>6} {'min':>7}")
    for r in results:
        print(f"{r['goal']:<58} {r['recall']:>7.2f} {r['precision']:>6.2f} "
              f"{r['n_slots']:>6} {r['total_minutes']:>7.1f}")
    print("-" * 90)
    print(f"macro recall={summary['macro_recall']:.3f}  macro precision={summary['macro_precision']:.3f}  "
          f"perfect-recall goals={summary['perfect_recall_goals']}/{summary['n_goals']}")
    print(f"mean granularity-in-band={summary['mean_granularity_in_band']:.3f}  "
          f"mean type-hint coverage={summary['mean_type_hint_coverage']:.3f}  "
          f"mean dependency integrity={summary['mean_dependency_integrity']:.3f}")
    print(f"results -> {args.json}")


if __name__ == "__main__":
    main()
