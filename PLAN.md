# Implementation Plan
## Agentic Strategic Oversight of Classical Pathfinding with Cross-Episode Episodic Memory

> **How to resume**: At the start of each session, say "check PLAN.md and continue from where we left off."
> **Status key**: `[ ]` not started · `[~]` in progress · `[x]` complete

---

## Phase 1 — Project Setup & Dependencies
- [x] Create `requirements.txt`
- [x] Create `.env.example` with API key placeholder
- [x] Create full folder structure (`src/`, `experiments/`, `results/`, `paper/`)
- [x] Verify venv and confirm all packages install cleanly

> **Environment**: Virtualenv lives at `.venv/`. Activate with `source .venv/bin/activate` or invoke binaries directly via `.venv/bin/python` / `.venv/bin/pytest`.
> **Scope**: Targeting class MVP (A* + D* Lite, 4 variants, 20×20, 50 sequential episodes). Conference extension (LPA*, Agent-NoMemory, 200 i.i.d. episodes, 50×50, Cohen's κ rating) is deferred.

## Phase 2 — Grid Environment
- [x] `src/environment/grid.py` — NxN grid, static + dynamic obstacles, stochastic motion models
- [x] `src/environment/visualizer.py` — render grid with hazard overlay and planned path
- [x] Smoke test: grid initializes, obstacles move correctly across steps

## Phase 3 — Classical Planners (Baselines)
- [x] `src/planners/base.py` — shared logging interface (all variants produce identical output format)
- [x] `src/planners/a_star.py` — A* with pluggable cost function (accepts hazard map weights)
- [x] `src/planners/d_star_lite.py` — D* Lite with incremental replanning on obstacle detection
- [x] Smoke test: A* finds correct path on static grid; D* Lite replans when obstacle appears

> LPA* is conference-only (deferred).

## Phase 4 — LangGraph Agent
- [ ] `src/agent/state.py` — `AgentState` TypedDict schema
- [ ] `src/agent/nodes/scanner.py` — Environment Scanner node
- [ ] `src/agent/nodes/planner.py` — A* Planner node (wraps Phase 3 planner)
- [ ] `src/agent/nodes/risk_evaluator.py` — LLM-backed Path Risk Evaluator node
- [ ] `src/agent/nodes/memory_manager.py` — Episodic Memory Manager node
- [ ] `src/agent/nodes/path_healer.py` — Path Healer node
- [ ] `src/agent/nodes/execution_monitor.py` — Execution Monitor node
- [ ] `src/agent/nodes/reflection.py` — Reflection Node
- [ ] `src/agent/graph.py` — wire StateGraph with conditional edges
- [ ] `src/memory/store.py` — JSON-backed episodic store (read/write utilities)
- [ ] Smoke test: Full Agent runs one episode end-to-end and produces a JSON log

## Phase 5 — Experiment Runner & Evaluation
- [ ] `experiments/configs/static.yaml` — Static environment config
- [ ] `experiments/configs/dynamic_low.yaml` — Dynamic-Low config
- [ ] `experiments/configs/dynamic_high.yaml` — Dynamic-High config
- [ ] `experiments/run_experiments.py` — seed-controlled runner for all variants
- [ ] Metric collection: SR, PER, CR, MRC, DL, MHR, token cost per episode
- [ ] `results/figures/` — generate all 7 required plots
  - [ ] Fig 1: LangGraph node architecture diagram
  - [ ] Fig 2: Agent reasoning pipeline / data flow
  - [ ] Fig 3: Grid world visualization with hazard overlay
  - [ ] Fig 4: Success rate by variant and condition (grouped bar chart)
  - [ ] Fig 5: Episode-over-episode learning curve
  - [ ] Fig 6: Cost-quality scatter (latency vs. SR)
  - [ ] Fig 7: Sample LLM justification text box

## Phase 6 — Paper & README
- [ ] `README.md` — setup instructions, venv activation, example run commands
- [ ] `PAPER.md` — full 8-section paper draft

---

## Current Status
**Active phase**: Phase 4
**Last completed**: Phase 3 (base.py, A*, D* Lite — 18 phase-3 tests; 83 total tests passing)
**Next step**: Begin Phase 4 — `src/agent/state.py` (AgentState TypedDict)
