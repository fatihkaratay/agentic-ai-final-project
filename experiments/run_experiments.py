"""
Experiment Runner — Phase 5

Runs (variant × condition × seed) combinations and writes per-episode
JSONL records to results/logs/<variant>__<condition>.jsonl.

Variants (MVP):
  astar               — A* + execution (no replanning on blocking) — proposal's lower bound
  dstar_lite          — D* Lite incremental replanning baseline
  agent_noreflection  — Full LangGraph agent with reflection disabled
  agent_full          — Full LangGraph agent

Sequential episodes within a (variant, condition) cell share the same Grid
instance so cross-episode hazard memory accumulates for agent variants
(hazard_map persists across grid.reset()). Baselines use hazard_weight=0,
so accumulation is moot.

Usage:
    .venv/bin/python -m experiments.run_experiments \\
        --episodes 50 \\
        --variants astar dstar_lite agent_noreflection agent_full \\
        --conditions static dynamic_low dynamic_high

Smoke run with mocked LLM:
    .venv/bin/python -m experiments.run_experiments --episodes 2 --mock-llm
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml
from dotenv import load_dotenv

# Ensure project root is importable when run as a script.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.agent.graph import RECURSION_LIMIT, build_agent_graph
from src.agent.state import init_state
from src.environment.grid import Grid, GridConfig
from src.llm.client import LLMClient
from src.llm.mock import MockLLMClient
from src.memory.store import EpisodicMemoryStore
from src.planners.a_star import AStarPlanner
from src.planners.d_star_lite import DStarLitePlanner


VARIANTS    = ["astar", "dstar_lite", "agent_noreflection", "agent_full"]
CONDITIONS  = ["static", "dynamic_low", "dynamic_high"]
CONFIG_DIR  = os.path.join(ROOT, "experiments", "configs")
DEFAULT_OUT = os.path.join(ROOT, "results", "logs")


# ── Episode runners ──────────────────────────────────────────────────────────

def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _run_baseline_episode(
    grid:           Grid,
    planner_kind:   str,                 # "astar" | "dstar_lite"
    max_steps:      int,
    allow_replan:   bool,
) -> Dict[str, Any]:
    """
    Step-by-step baseline runner. Mirrors execution_monitor semantics
    (move-then-step-world, terminal checks against post-step grid).
    """
    t0      = time.perf_counter()
    planner = AStarPlanner(hazard_weight=0.0) if planner_kind == "astar" \
              else DStarLitePlanner(hazard_weight=0.0)

    # Initial plan
    start, goal = grid.start, grid.goal
    result = planner.plan(start, goal, grid)
    path         = list(result.path)
    nodes_total  = result.nodes_expanded
    plan_latency = result.plan_latency_ms

    current_pos = start
    steps       = 0
    replans     = 0
    status      = "running"

    # If the initial plan fails (start currently surrounded by dynamic obstacles),
    # we still enter the loop so the world advances. D* Lite may recover on its
    # next replan; A* baseline will time out with honest non-zero steps.

    while status == "running":
        if current_pos == goal:
            status = "success"
            break
        if steps >= max_steps:
            status = "timeout"
            break

        # Walk along path. If next is blocked, replan (D* Lite) or stay (A*).
        if len(path) >= 2 and path[0] == current_pos:
            next_pos = path[1]
            if grid.is_blocked(next_pos):
                if allow_replan:
                    rr = planner.replan(current_pos, grid)
                    path         = list(rr.path)
                    nodes_total += rr.nodes_expanded
                    plan_latency += rr.plan_latency_ms
                    replans     += 1
                    if not rr.success or len(path) < 2 or path[0] != current_pos:
                        # Cannot make progress — stay; world advances; eventually times out.
                        new_pos = current_pos
                    else:
                        if grid.is_blocked(path[1]):
                            new_pos = current_pos
                        else:
                            new_pos = path[1]
                            path = path[1:]
                else:
                    new_pos = current_pos                       # A* baseline: stuck
            else:
                new_pos = next_pos
                path    = path[1:]
        else:
            # No usable path
            if allow_replan:
                rr = planner.replan(current_pos, grid)
                path         = list(rr.path)
                nodes_total += rr.nodes_expanded
                plan_latency += rr.plan_latency_ms
                replans     += 1
                if not rr.success or len(path) < 2 or path[0] != current_pos:
                    new_pos = current_pos
                else:
                    new_pos = path[1] if not grid.is_blocked(path[1]) else current_pos
                    if new_pos != current_pos:
                        path = path[1:]
            else:
                new_pos = current_pos                           # A* baseline gives up

        steps += 1
        grid.step()
        grid.agent_pos = list(new_pos)
        current_pos = new_pos

        if current_pos == goal:
            status = "success"
        elif grid.is_blocked(current_pos):
            status = "collision"
        elif steps >= max_steps:
            status = "timeout"

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "outcome":            status,
        "steps_taken":        steps,
        "replan_count":       replans,
        "heal_count":         0,
        "nodes_expanded":     nodes_total,
        "plan_latency_ms":    plan_latency,
        "wall_clock_ms":      wall_ms,
        "llm_call_count":     0,
        "llm_total_tokens":   0,
        "final_pos":          list(current_pos),
    }


def _run_agent_episode(
    grid:               Grid,
    llm_client,
    store:              EpisodicMemoryStore,
    enable_reflection:  bool,
    max_steps:          int,
    risk_threshold:     float,
    hazard_weight:      float,
    max_heals:          int,
    episode_id:         str,
) -> Dict[str, Any]:
    state = init_state(
        grid,
        risk_threshold    = risk_threshold,
        max_steps         = max_steps,
        enable_memory     = True,
        enable_reflection = enable_reflection,
    )
    state["episode_id"] = episode_id

    # Reset LLM counters so this episode's record reflects only its own usage.
    llm_client.reset_counters()

    t0    = time.perf_counter()
    graph = build_agent_graph(
        llm_client    = llm_client,
        hazard_weight = hazard_weight,
        max_heals     = max_heals,
        store         = store,
    )
    final = graph.invoke(state, config={"recursion_limit": RECURSION_LIMIT})
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "outcome":          final.get("episode_status", "unknown"),
        "steps_taken":      final.get("step_count", 0),
        "replan_count":     final.get("replan_count", 0),
        "heal_count":       final.get("heal_count", 0),
        "nodes_expanded":   None,                    # not tracked through the graph
        "plan_latency_ms":  None,                    # see episode_log entries instead
        "wall_clock_ms":    wall_ms,
        "llm_call_count":   llm_client.call_count,
        "llm_total_tokens": llm_client.total_tokens,
        "final_pos":        list(final.get("current_pos", [-1, -1])),
        "risk_score_last":  final.get("risk_score"),
        "reflection_summary": (final.get("reflection_summary") or "")[:300],
    }


# ── Driver ───────────────────────────────────────────────────────────────────

def _load_condition(name: str) -> Dict[str, Any]:
    with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), "r") as f:
        return yaml.safe_load(f)


def _build_grid(cfg: Dict[str, Any], seed: int) -> Grid:
    return Grid(
        GridConfig(
            size             = cfg["size"],
            obstacle_density = cfg["obstacle_density"],
            dynamic_ratio    = cfg["dynamic_ratio"],
            motion_model     = cfg["motion_model"],
            motion_speed     = cfg["motion_speed"],
        ),
        seed=seed,
    )


def run(
    variants:    List[str],
    conditions:  List[str],
    n_episodes:  int,
    seed_base:   int,
    output_dir:  str,
    mock_llm:    bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for cond_name in conditions:
        cfg     = _load_condition(cond_name)
        manhattan = (cfg["size"] - 1) * 2
        max_steps = manhattan * 3      # proposal: timeout = 3× optimal

        for variant in variants:
            print(f"\n=== {variant} on {cond_name} — {n_episodes} episodes ===", flush=True)

            # Per-(variant, condition) state.
            grid       = _build_grid(cfg, seed=seed_base)        # seed reset on each ep below
            store_path = os.path.join(output_dir, f"_memstore__{variant}__{cond_name}.jsonl")
            store      = EpisodicMemoryStore(path=store_path)
            store.clear()

            llm_client = MockLLMClient() if mock_llm else LLMClient() if variant.startswith("agent") else None

            log_path = os.path.join(output_dir, f"{variant}__{cond_name}.jsonl")
            with open(log_path, "w") as fh:
                for i in range(n_episodes):
                    seed = seed_base + i
                    grid.reset(seed=seed)        # fresh obstacles; hazard_map preserved (cross-ep memory)

                    if variant == "astar":
                        ep = _run_baseline_episode(
                            grid, planner_kind="astar",
                            max_steps=max_steps, allow_replan=False,
                        )
                    elif variant == "dstar_lite":
                        ep = _run_baseline_episode(
                            grid, planner_kind="dstar_lite",
                            max_steps=max_steps, allow_replan=True,
                        )
                    elif variant in ("agent_noreflection", "agent_full"):
                        ep = _run_agent_episode(
                            grid                = grid,
                            llm_client          = llm_client,
                            store               = store,
                            enable_reflection   = (variant == "agent_full"),
                            max_steps           = max_steps,
                            risk_threshold      = 0.3,
                            hazard_weight       = 2.0,
                            max_heals           = 3,
                            episode_id          = f"{variant}__{cond_name}__{i:04d}",
                        )
                    else:
                        raise ValueError(f"unknown variant: {variant}")

                    # PER = steps / Manhattan (lower bound on optimal).
                    per = (ep["steps_taken"] / manhattan) if manhattan > 0 else None

                    record = {
                        "ts":           datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "variant":      variant,
                        "condition":    cond_name,
                        "seed":         seed,
                        "episode_idx":  i,
                        "manhattan":    manhattan,
                        "max_steps":    max_steps,
                        "per":          per,
                        **ep,
                    }
                    fh.write(json.dumps(record) + "\n")
                    fh.flush()

                    print(
                        f"  [{i+1:3d}/{n_episodes}] seed={seed} "
                        f"{ep['outcome']:9s} steps={ep['steps_taken']:3d} "
                        f"replans={ep['replan_count']:2d} heals={ep['heal_count']:1d} "
                        f"llm_calls={ep['llm_call_count']:3d}",
                        flush=True,
                    )

            print(f"  → wrote {log_path}", flush=True)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run agentic-pathfinding experiments.")
    p.add_argument("--variants",   nargs="+", default=VARIANTS,   choices=VARIANTS)
    p.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    p.add_argument("--episodes",   type=int,   default=50,   help="episodes per (variant, condition)")
    p.add_argument("--seed-base",  type=int,   default=1000, dest="seed_base")
    p.add_argument("--output",     type=str,   default=DEFAULT_OUT, dest="output_dir")
    p.add_argument("--mock-llm",   action="store_true",
                   help="use deterministic mock LLM (no API calls; for smoke tests)")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    run(
        variants    = args.variants,
        conditions  = args.conditions,
        n_episodes  = args.episodes,
        seed_base   = args.seed_base,
        output_dir  = args.output_dir,
        mock_llm    = args.mock_llm,
    )


if __name__ == "__main__":
    main()
