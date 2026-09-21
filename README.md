# planlearn

A multi-agent planner that turns a textbook into a personalized roadmap of
**20-minute learning slots**, following the *Plan Over* design
(Orchestrator, Graph Planner, Pedagogical Evaluator, Topic Researcher).

```
python -m planlearn.cli "learn lasso and ridge regression" --level beginner --json runs/plan.json
```

Runs fully offline by default (TF-IDF retrieval, heuristic agents). Add `--llm`
for Claude-assisted prerequisite and relevance judgments, and `--chroma ./islp_db`
for embedding retrieval.

## Setup

```
pip install -e ".[dev]"            # core + tests
pip install -e ".[llm,chroma]"     # optional: Claude + Chroma
export ANTHROPIC_API_KEY=...        # only for --llm
pytest                              # 17 tests, ~5 s
```

To ingest another book: `python scripts/chunk_islp.py book.pdf data/book_chunks.jsonl`,
then (optionally) `python scripts/index_chunks.py data/book_chunks.jsonl ./book_db`.
Any PDF with bookmarks (a table of contents) works; adjust the margin-note
x-position in the chunker for other layouts.

## MCP server

`planlearn` can run as an MCP server so an MCP client (Claude Code, Claude
Desktop, etc.) can call it directly from a chat prompt instead of the CLI.

```
pip install -e ".[mcp]"
python -m planlearn.mcp_server        # stdio server, for manual testing
```

It exposes five tools: `plan_learning_roadmap` (build a roadmap for a goal),
`refine_plan` (re-plan after feedback — known topics, dropped slots, pace),
`list_levels`, `list_node_types`, and both planning tools take an optional
`node_types` filter (e.g. `["concept"]` for explanation-only slots, no
coding/math_proof/exercise). Both planning tools default to a trimmed
response (goal, roadmap, starting points, recommendations, alternates);
pass `full_output: true` to also get the raw knowledge graph and
rejected/merged bookkeeping (computed before any `node_types` filtering).

This repo ships a project-scoped [`.mcp.json`](.mcp.json) pointing at
`.venv/bin/python -m planlearn.mcp_server`, so Claude Code picks it up
automatically when run from this directory (it will prompt once to approve
the project's MCP servers). To register it globally instead:

```
claude mcp add planlearn -- /path/to/planlearn/.venv/bin/python -m planlearn.mcp_server
```

Configuration is via env vars, all optional: `PLANLEARN_CHUNKS` (textbook
chunks JSONL, default `data/islp_chunks.jsonl`), `PLANLEARN_CACHE`
(`runs/cache.json`), `PLANLEARN_AUDIT` (`runs/audit.jsonl`). Set
`ANTHROPIC_API_KEY` in the server's environment if you want tool calls with
`use_llm: true`.

## How the design maps to code

| Design doc | Code |
|---|---|
| Orchestrator: workflow, state, cache, ToT, logging | `orchestrator.py`: explicit state machine (`ALLOWED` transitions), `audit.py`, `cache.py` |
| Graph Planner: static KG, prerequisites, unwind large nodes | `agents/graph_planner.py`, `knowledge_graph.py` |
| Pedagogical Evaluator: coverage, size, time; Keep / Revise / Reject | `agents/evaluator.py` |
| Topic Researcher: trends, tie-breaking, light recommendations | `agents/researcher.py` (pluggable `WebSearch`; offline stub) |
| Bounded autonomy: agents suggest, orchestrator decides | `AgentResult.suggestion` → `Orchestrator._accept()` checks transition + budget |
| ToT with BFS, one node per step, pick 3 best | `Orchestrator.run()`: BFS frontier, each step expands one node and keeps the top `beam_width` children |
| > 20 min: keep expanding; too small: merge | TOC expansion → mechanical chunk split → `_merge_small()` |
| User in the loop | plan returns `awaiting_review`, `alternates`; `apply_feedback()` re-plans |
| Scaling ToT: limit each step | `config.SearchLimits` (beam, depth, steps, nodes, research calls) |

### Workflow

```
PLAN ──► EVALUATE ◄──► RESEARCH        (only on near-ties at the beam cut, budgeted)
            │
            ▼
         APPEND ──► EXPAND ──► EVALUATE (next frontier node, BFS)
                       │
                       ▼
                     MERGE ──► REVIEW ──► DONE ──► user feedback ──► re-plan
```

Every transition and agent call (with latency and suggestion accepted/overridden)
goes to `runs/audit.jsonl`.

## Design decisions made while building

These came from running the system and reading the audit log. Each fixed a real
failure:

1. **Knowledge graph comes from the book itself.** Tree edges come from the
   TOC path on each chunk. Prerequisite edges come from explicit cross-references
   ("as described in Chapter 5"), and the *text around the reference* is used as
   a retrieval query to decide **which part** of the prerequisite matters (e.g.
   cross-validation rather than the bootstrap).
2. **Per-node beam, not a global pool.** A single global candidate pool let
   unrelated branches crowd each other out (Lasso's parts pushed Ridge out of
   the beam). Expanding one node per step and ranking only its children, as
   in the design doc, fixed it.
3. **Choices vs. mechanics.** Choosing *which* subsections to learn is the
   evaluator's job. Splitting a kept section into ≤20-minute parts, and
   including a section's intro, are mechanical and never compete in the beam.
4. **Revise = alternates.** Plausible siblings that didn't make the beam are
   returned as `alternates` so the user can swap them in at review.
5. **Relevance gates inclusion; difficulty shapes pacing.** Level fit has a low
   weight so a beginner who asks for Ridge still gets Ridge (just slower slots).
6. **Goal sections always survive the root step**; prerequisites get a
   narrower beam (`prereq_beam=2`) to stay "just enough".
7. **Coverage rule:** a kept section always contributes at least its best child.

## Known limitations / next steps

- **Lexical relevance is the weakest link.** Offline TF-IDF sometimes misses
  foundational sections (PCA plans can skip "What Are Principal Components?")
  or picks adjacent topics (RNNs for a CNN goal). `--llm` enables an LLM judge
  in the evaluator; `--chroma` swaps in embeddings. Measure before tuning:
  write 15–20 goals with expected sections and track recall.
- **Time estimates are priors** (`estimate.py`). Calibrate with real completion
  times via `pace_factor`; eventually learn per-type rates per user.
- **Researcher is a stub offline.** Implement `WebSearch.search()` against a
  search API; trend scoring is a placeholder (result count), so replace it with
  recency/citation signals.
- **Single book.** Multi-book support needs node IDs namespaced by source and
  cross-book prerequisite linking (an LLM task).
- **Scaling (design doc §Scalability):** move the cache and audit log to shared
  storage, and retrieval to pgvector/Qdrant. The `Retriever` protocol is the seam.
