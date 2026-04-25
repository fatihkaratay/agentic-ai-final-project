"""
Phase 3 Tests — Classical Planners (A* and D* Lite)
Run: .venv/bin/pytest tests/test_phase3_planners.py -v
"""

import numpy as np
import pytest

from src.environment.grid import Grid, GridConfig
from src.planners.base import Planner, PlanResult
from src.planners.a_star import AStarPlanner
from src.planners.d_star_lite import DStarLitePlanner


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_5x5():
    return Grid(GridConfig(size=5, obstacle_density=0.0), seed=0)

@pytest.fixture
def static_10x10():
    cfg = GridConfig(size=10, obstacle_density=0.20, dynamic_ratio=0.0,
                     motion_model="static")
    return Grid(cfg, seed=42)

@pytest.fixture(params=[AStarPlanner, DStarLitePlanner])
def planner(request):
    """Parametrised — every test using this runs against both planners."""
    return request.param(hazard_weight=2.0)


# ── PlanResult contract ──────────────────────────────────────────────────────

def test_planresult_default_is_failure():
    r = PlanResult()
    assert r.success is False and r.path == [] and r.nodes_expanded == 0


# ── Shared planner contract (both A* and D* Lite) ────────────────────────────

def test_finds_path_on_empty_grid(planner, empty_5x5):
    r = planner.plan((0, 0), (4, 4), empty_5x5)
    assert r.success
    assert r.path[0] == (0, 0) and r.path[-1] == (4, 4)
    assert len(r.path) == 9      # Manhattan-optimal: 8 steps + start
    assert r.plan_latency_ms >= 0
    assert r.nodes_expanded > 0

def test_path_is_contiguous(planner, empty_5x5):
    r = planner.plan((0, 0), (4, 4), empty_5x5)
    for a, b in zip(r.path, r.path[1:]):
        manhattan = abs(a[0] - b[0]) + abs(a[1] - b[1])
        assert manhattan == 1, f"non-adjacent step: {a} → {b}"

def test_path_avoids_blocked_cells(planner, static_10x10):
    r = planner.plan((0, 0), (9, 9), static_10x10)
    if r.success:
        for cell in r.path:
            assert not static_10x10.is_blocked(cell), f"path enters blocked {cell}"

def test_no_path_when_start_walled_in(planner):
    g = Grid(GridConfig(size=5, obstacle_density=0.0), seed=0)
    g._static_grid[0, 1] = 1
    g._static_grid[1, 0] = 1
    r = planner.plan((0, 0), (4, 4), g)
    assert r.success is False
    assert r.path == []

def test_replan_from_intermediate_position(planner, empty_5x5):
    planner.plan((0, 0), (4, 4), empty_5x5)
    r = planner.replan((2, 2), empty_5x5)
    assert r.success
    assert r.path[0] == (2, 2) and r.path[-1] == (4, 4)
    assert len(r.path) == 5     # Manhattan-optimal from (2,2) to (4,4)

def test_replan_before_plan_raises(planner, empty_5x5):
    with pytest.raises(RuntimeError):
        planner.replan((0, 0), empty_5x5)

def test_hazard_weight_steers_path(empty_5x5):
    """High hazard cost should pull the path away from a flagged row."""
    empty_5x5.hazard_map[2, :] = 1.0    # row 2 is maximally risky

    naive   = AStarPlanner(hazard_weight=0.0).plan((0, 0), (4, 4), empty_5x5)
    careful = AStarPlanner(hazard_weight=10.0).plan((0, 0), (4, 4), empty_5x5)
    assert naive.success and careful.success
    # Both must traverse row 2 at least once (it bisects the grid),
    # but the minimum is one cell.
    careful_hot = sum(1 for (r, _) in careful.path if r == 2)
    assert careful_hot == 1


# ── A* / D* Lite optimality agreement ────────────────────────────────────────

def test_astar_and_dstar_agree_on_static_grid(static_10x10):
    a = AStarPlanner(hazard_weight=0.0).plan((0, 0), (9, 9), static_10x10)
    d = DStarLitePlanner(hazard_weight=0.0).plan((0, 0), (9, 9), static_10x10)
    assert a.success == d.success
    if a.success:
        assert len(a.path) == len(d.path), \
            f"shortest-path lengths differ: A*={len(a.path)} D*L={len(d.path)}"


# ── D* Lite incremental behaviour ────────────────────────────────────────────

def test_dstar_lite_replan_after_new_obstacle():
    """Block a cell on the planned path; D* Lite must route around it."""
    g = Grid(GridConfig(size=5, obstacle_density=0.0), seed=0)
    d = DStarLitePlanner(hazard_weight=0.0)

    r1 = d.plan((0, 0), (4, 4), g)
    assert r1.success
    # Pick a non-endpoint waypoint on the path and block it.
    block_cell = r1.path[len(r1.path) // 2]
    g._static_grid[block_cell[0], block_cell[1]] = 1

    r2 = d.replan((0, 0), g)
    assert r2.success
    assert block_cell not in r2.path
    # Path through every cell must still be valid.
    for cell in r2.path:
        assert not g.is_blocked(cell)

def test_dstar_lite_matches_astar_after_change():
    """After modifying the grid, D* Lite's incremental result equals A*-from-scratch."""
    g = Grid(GridConfig(size=8, obstacle_density=0.0), seed=0)
    d = DStarLitePlanner(hazard_weight=0.0)

    d.plan((0, 0), (7, 7), g)
    g._static_grid[3, 3] = 1
    g._static_grid[3, 4] = 1
    g._static_grid[4, 3] = 1

    d_after = d.replan((0, 0), g)
    a_after = AStarPlanner(hazard_weight=0.0).plan((0, 0), (7, 7), g)
    assert d_after.success and a_after.success
    assert len(d_after.path) == len(a_after.path)

def test_dstar_lite_replan_uses_less_search_when_change_is_local():
    """A small late-stage change should expand far fewer nodes than the initial plan."""
    g = Grid(GridConfig(size=10, obstacle_density=0.0), seed=0)
    d = DStarLitePlanner(hazard_weight=0.0)
    r1 = d.plan((0, 0), (9, 9), g)
    initial_expanded = r1.nodes_expanded

    # Block one cell far from start.
    g._static_grid[8, 8] = 1
    r2 = d.replan((0, 0), g)
    assert r2.success
    assert r2.nodes_expanded < initial_expanded, (
        f"replan expanded {r2.nodes_expanded} ≥ initial {initial_expanded}; "
        "incremental advantage lost"
    )
