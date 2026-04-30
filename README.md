# Agentic Strategic Oversight of Classical Pathfinding

A LangGraph-based agent that supervises an A\* path planner with cross-episode
episodic memory and an LLM-driven reflective reasoner. Final project for
*Introduction to Agentic AI* (Module 14).

The system contributes:

1. **Cross-episode episodic memory** — a persistent hazard map that
   accumulates across sequential episodes, encoding environment-specific
   risk patterns that classical replanners (A\*, D\* Lite) cannot retain.
2. **LLM as interpretable strategic reasoner** — `gpt-4o-mini` produces
   natural-language risk justifications and end-of-episode reflections,
   creating a human-readable audit trail for every decision.
3. **Systematic ablation against classical baselines** — A\*, D\* Lite,
   Agent-NoReflection, and Full Agent run on identical seeds across three
   environment difficulty tiers.

## Repository layout

```
agentic-ai-final-project/
├── INSTRUCTIONS.md              # MVP scope (class deliverable)
├── INSTRUCTIONS_CONF.md         # Conference-extended scope (deferred)
├── PLAN.md                      # Phase-by-phase implementation plan
├── PAPER.md                     # 8-section IEEE-style write-up
├── requirements.txt
├── .env.example                 # Copy to .env and fill in OPENAI_API_KEY
├── src/
│   ├── environment/             # Grid world, obstacle motion, visualizer
│   ├── planners/                # base.py, a_star.py, d_star_lite.py
│   ├── agent/                   # LangGraph agent
│   │   ├── nodes/               # 7 node files (one per node)
│   │   ├── state.py             # AgentState TypedDict
│   │   └── graph.py             # StateGraph wiring + conditional edges
│   ├── memory/store.py          # JSON-Lines episodic memory store
│   └── llm/                     # OpenAI client + mock + corporate-SSL helper
├── experiments/
│   ├── configs/                 # Static, Dynamic-Low, Dynamic-High YAML
│   ├── run_experiments.py       # Seed-controlled experiment runner
│   ├── aggregate_results.py     # Per-condition summary table
│   ├── generate_figures.py      # Data-driven figures (4–7)
│   └── generate_architecture_figures.py  # Topology + flow + grid (1–3)
├── results/
│   ├── logs/run50/              # 50-episode JSONL records (canonical)
│   └── figures/run50/           # All 7 figures referenced by PAPER.md
└── tests/                       # 83 pytest cases — `pytest -q`
```

## Setup

Requires Python 3.10+. The repository ships with a `.venv/` virtualenv path
expectation — adapt as needed.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env to set:
#   LLM_PROVIDER=openai
#   LLM_MODEL=gpt-4o-mini
#   OPENAI_API_KEY=sk-...
```

### Corporate-SSL note

If you sit behind a TLS-inspecting proxy (e.g. corporate networks that
present a self-signed root CA), set `SSL_CERT_FILE=/path/to/corp-root.cer`
in your environment. `src/llm/_ssl.py` will detect it and build a merged
public-CA + corporate-root bundle at `.venv/merged-ca-bundle.pem` on first
LLM call so `httpx` validates both public APIs and inspected endpoints.

## Running the tests

```bash
.venv/bin/pytest -q
# 83 passed
```

## Reproducing the experiments

The canonical results (50 episodes × 4 variants × 3 conditions = 600 episodes)
are reproduced with:

```bash
.venv/bin/python -m experiments.run_experiments \
    --episodes 50 \
    --output results/logs/run50

.venv/bin/python -m experiments.aggregate_results \
    --input results/logs/run50

.venv/bin/python -m experiments.generate_figures \
    --input results/logs/run50 \
    --output results/figures/run50

.venv/bin/python -m experiments.generate_architecture_figures \
    --output results/figures/run50
```

For a fast-feedback smoke run (no API spend):

```bash
.venv/bin/python -m experiments.run_experiments \
    --episodes 2 --mock-llm \
    --output /tmp/smoke
```

For a single-cell debug run:

```bash
.venv/bin/python -m experiments.run_experiments \
    --variants agent_full --conditions dynamic_low \
    --episodes 5 --output /tmp/debug
```

### Cost note

The 50-episode canonical run executes roughly 2,500 LLM calls totalling
~3M tokens — under \$1 USD with `gpt-4o-mini`.

## Variants

| Variant | Description |
|---|---|
| `astar` | A\* Planner + Execution Monitor only — no replanning on obstacle detection. Theoretical lower bound in dynamic environments. |
| `dstar_lite` | D\* Lite (Koenig & Likhachev 2002) with full incremental replanning. Primary algorithmic competitor. |
| `agent_noreflection` | Full LangGraph agent with the Reflection Node disabled (raw memory accumulation, no synthesis). |
| `agent_full` | Full LangGraph agent with all 7 nodes active. Primary proposed system. |

## Environment configurations

| Tier | Density | Motion | Use |
|---|---|---|---|
| `static` | 15% | None | Baseline |
| `dynamic_low` | 15% | Random walk, 1 cell / 5 steps | Tests reactive replanning |
| `dynamic_high` | 20% | Random walk, every step | Tests memory + reflective adaptation |

## Documentation

- `PAPER.md` — the 8-section research-paper write-up with all results, figures,
  and analysis.
- `PLAN.md` — phase-by-phase implementation log (use this to orient if picking
  up the project mid-flight).
- `INSTRUCTIONS.md` / `INSTRUCTIONS_CONF.md` — the original assignment brief
  and the proposal extension targeted at a future conference submission.
