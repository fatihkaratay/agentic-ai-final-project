"""
Phase 2 Tests — Grid Environment
Run: .venv/bin/pytest tests/test_phase2_environment.py -v
"""

import os
import numpy as np
import pytest

from src.environment.grid import Grid, GridConfig, FREE, STATIC, DYNAMIC


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def static_grid():
    cfg = GridConfig(size=20, obstacle_density=0.15, dynamic_ratio=0.0,
                     motion_model="static")
    return Grid(cfg, seed=42)

@pytest.fixture
def dynamic_grid():
    cfg = GridConfig(size=20, obstacle_density=0.15, dynamic_ratio=0.5,
                     motion_model="random_walk", motion_speed=1.0)
    return Grid(cfg, seed=42)


# ── Grid initialisation ───────────────────────────────────────────────────────

def test_grid_correct_shape(dynamic_grid):
    assert dynamic_grid.get_grid().shape == (20, 20)

def test_start_cell_is_free(dynamic_grid):
    g = dynamic_grid.get_grid()
    r, c = dynamic_grid.start
    assert g[r, c] == FREE, "Start cell must never be blocked at initialisation"

def test_goal_cell_is_free(dynamic_grid):
    g = dynamic_grid.get_grid()
    r, c = dynamic_grid.goal
    assert g[r, c] == FREE, "Goal cell must never be blocked at initialisation"

def test_default_goal_is_bottom_right():
    cfg = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    assert grid.goal == (9, 9)

def test_obstacle_density_approximate(dynamic_grid):
    g = dynamic_grid.get_grid()
    total = 20 * 20
    n_obs = int(np.sum(g != FREE))
    density = n_obs / total
    assert 0.05 <= density <= 0.30, (
        f"Obstacle density {density:.2f} is outside expected range [0.05, 0.30]"
    )

def test_static_and_dynamic_counts(dynamic_grid):
    g = dynamic_grid.get_grid()
    n_static  = int(np.sum(dynamic_grid._static_grid == STATIC))
    n_dynamic = len(dynamic_grid._dynamic_positions)
    assert n_static > 0,  "Expected some static obstacles"
    assert n_dynamic > 0, "Expected some dynamic obstacles"


# ── Determinism ───────────────────────────────────────────────────────────────

def test_same_seed_same_grid():
    cfg  = GridConfig(size=20, obstacle_density=0.15, dynamic_ratio=0.5)
    g1   = Grid(cfg, seed=99)
    g2   = Grid(cfg, seed=99)
    assert np.array_equal(g1.get_grid(), g2.get_grid()), \
        "Same seed must produce identical grids"

def test_different_seeds_different_grids():
    cfg = GridConfig(size=20, obstacle_density=0.15, dynamic_ratio=0.5)
    g1  = Grid(cfg, seed=1)
    g2  = Grid(cfg, seed=2)
    assert not np.array_equal(g1.get_grid(), g2.get_grid()), \
        "Different seeds should produce different grids"

def test_reset_with_same_seed_restores_grid(dynamic_grid):
    original = dynamic_grid.get_grid().copy()
    dynamic_grid.reset(seed=42)
    assert np.array_equal(dynamic_grid.get_grid(), original), \
        "reset(seed=42) should restore the original layout"


# ── Static obstacles don't move ───────────────────────────────────────────────

def test_static_obstacles_do_not_move(static_grid):
    before = static_grid._static_grid.copy()
    for _ in range(10):
        static_grid.step()
    assert np.array_equal(static_grid._static_grid, before), \
        "Static grid should be unchanged after steps"

def test_no_dynamic_obstacles_in_static_config(static_grid):
    assert len(static_grid._dynamic_positions) == 0, \
        "dynamic_ratio=0.0 should produce zero dynamic obstacles"


# ── Dynamic obstacles move ────────────────────────────────────────────────────

def test_dynamic_obstacles_can_move():
    cfg  = GridConfig(size=20, obstacle_density=0.15, dynamic_ratio=0.8,
                      motion_model="random_walk", motion_speed=1.0)
    grid = Grid(cfg, seed=7)
    before = set(grid._dynamic_positions)
    for _ in range(5):
        grid.step()
    after = set(grid._dynamic_positions)
    assert before != after, \
        "Dynamic obstacles should change position over 5 steps with motion_speed=1.0"

def test_step_count_increments(dynamic_grid):
    assert dynamic_grid.step_count == 0
    dynamic_grid.step()
    dynamic_grid.step()
    assert dynamic_grid.step_count == 2

def test_dynamic_obstacles_stay_in_bounds(dynamic_grid):
    for _ in range(50):
        dynamic_grid.step()
    for r, c in dynamic_grid._dynamic_positions:
        assert 0 <= r < dynamic_grid.size, f"Row {r} out of bounds"
        assert 0 <= c < dynamic_grid.size, f"Col {c} out of bounds"


# ── is_blocked ────────────────────────────────────────────────────────────────

def test_out_of_bounds_is_blocked(dynamic_grid):
    assert dynamic_grid.is_blocked((-1, 0))
    assert dynamic_grid.is_blocked((0, -1))
    assert dynamic_grid.is_blocked((20, 0))
    assert dynamic_grid.is_blocked((0, 20))

def test_static_wall_is_blocked():
    cfg  = GridConfig(size=10, obstacle_density=0.20, dynamic_ratio=0.0)
    grid = Grid(cfg, seed=1)
    walls = [(r, c) for r in range(10) for c in range(10)
             if grid._static_grid[r, c] == STATIC]
    assert len(walls) > 0
    for pos in walls:
        assert grid.is_blocked(pos), f"Static wall at {pos} should be blocked"

def test_start_and_goal_not_blocked(dynamic_grid):
    assert not dynamic_grid.is_blocked(dynamic_grid.start)
    assert not dynamic_grid.is_blocked(dynamic_grid.goal)


# ── Hazard map ────────────────────────────────────────────────────────────────

def test_hazard_map_initialises_to_zero(dynamic_grid):
    assert np.all(dynamic_grid.hazard_map == 0.0)

def test_hazard_map_updates():
    cfg  = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    grid.update_hazard_map([(3, 3), (5, 5)], increment=0.5, decay=1.0)
    assert grid.hazard_map[3, 3] > 0.0
    assert grid.hazard_map[5, 5] > 0.0
    assert grid.hazard_map[0, 0] == 0.0

def test_hazard_map_does_not_exceed_one():
    cfg  = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    for _ in range(20):
        grid.update_hazard_map([(3, 3)], increment=0.5, decay=1.0)
    assert grid.hazard_map[3, 3] <= 1.0

def test_hazard_map_decays():
    cfg  = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    grid.update_hazard_map([(3, 3)], increment=0.5, decay=1.0)
    score_before = grid.hazard_map[3, 3]
    grid.update_hazard_map([], increment=0.0, decay=0.5)
    assert grid.hazard_map[3, 3] < score_before

def test_hazard_map_persists_across_reset():
    cfg  = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    grid.update_hazard_map([(3, 3)], increment=0.5, decay=1.0)
    grid.reset(seed=1)                       # new episode, different seed
    assert grid.hazard_map[3, 3] > 0.0, \
        "Hazard map should persist across episodes (cross-episode memory)"

def test_reset_hazard_map_clears_all():
    cfg  = GridConfig(size=10)
    grid = Grid(cfg, seed=0)
    grid.update_hazard_map([(3, 3)], increment=0.5, decay=1.0)
    grid.reset_hazard_map()
    assert np.all(grid.hazard_map == 0.0)


# ── State summary ─────────────────────────────────────────────────────────────

def test_state_summary_is_string(dynamic_grid):
    summary = dynamic_grid.get_state_summary()
    assert isinstance(summary, str) and len(summary) > 0

def test_state_summary_contains_key_info(dynamic_grid):
    summary = dynamic_grid.get_state_summary()
    assert "Grid" in summary
    assert "Agent" in summary
    assert "Goal" in summary

def test_state_summary_with_path(dynamic_grid):
    path    = [(1,1),(2,1),(3,1),(4,1)]
    summary = dynamic_grid.get_state_summary(path=path)
    assert isinstance(summary, str) and len(summary) > 0


# ── Visualizer (smoke test — no display) ─────────────────────────────────────

def test_visualizer_saves_file(tmp_path, dynamic_grid):
    from src.environment.visualizer import render_grid
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend for headless test

    path      = [(0,0),(1,0),(2,0),(3,0),(4,0)]
    save_file = str(tmp_path / "test_render.png")
    fig       = render_grid(dynamic_grid, path=path, title="Test", save_path=save_file)

    assert os.path.isfile(save_file), "Visualizer should save a PNG file"
    assert os.path.getsize(save_file) > 1000, "Saved PNG should not be empty"

    import matplotlib.pyplot as plt
    plt.close(fig)
