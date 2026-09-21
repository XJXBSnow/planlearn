"""python -m planlearn.cli "learn lasso and ridge regression" --level beginner"""
from __future__ import annotations

import argparse
import json

from . import Level, UserProfile, build_system


def main() -> None:
    ap = argparse.ArgumentParser(description="Plan a 20-minute-slot learning roadmap.")
    ap.add_argument("goal")
    ap.add_argument("--chunks", default="data/islp_chunks.jsonl")
    ap.add_argument("--level", choices=[l.value for l in Level], default="beginner")
    ap.add_argument("--known", nargs="*", default=[], help="topics you already know")
    ap.add_argument("--slot", type=int, default=20)
    ap.add_argument("--exercises", action="store_true")
    ap.add_argument("--llm", action="store_true", help="use Claude to refine prerequisites")
    ap.add_argument("--chroma", help="path to a Chroma index (default: TF-IDF)")
    ap.add_argument("--json", help="write full plan JSON here")
    ap.add_argument("--audit", default="runs/audit.jsonl")
    a = ap.parse_args()

    llm = retriever = None
    if a.llm:
        from .llm import AnthropicLLM
        llm = AnthropicLLM()
    if a.chroma:
        from .retrieval.chroma import ChromaRetriever
        retriever = ChromaRetriever(a.chroma)

    system = build_system(a.chunks, llm=llm, retriever=retriever,
                          cache_path="runs/cache.json", audit_path=a.audit)
    plan = system.run(UserProfile(goal=a.goal, level=Level(a.level), known_topics=a.known,
                                  slot_minutes=a.slot, include_exercises=a.exercises))

    print(f"\nGoal: {plan['goal']}  ({plan['level']}, {plan['slot_minutes']}-min slots)"
          f"{'  [cached]' if plan['from_cache'] else ''}")
    print(f"{len(plan['roadmap'])} slots, ~{plan['total_minutes']} min total\n")
    for s in plan["roadmap"]:
        title = s["id"].split(" > ")[-1]
        print(f"{s['slot']:>3}. [{s['type']:<10}] {s['minutes']:>5.1f} min  pp.{s['pages'][0]}-"
              f"{s['pages'][1]}  {title}")
    print(f"\nStart with: {', '.join(i.split(' > ')[-1] for i in plan['starting_points'][:3])}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(plan, f, indent=2, default=str)
        print(f"full plan -> {a.json}")


if __name__ == "__main__":
    main()
