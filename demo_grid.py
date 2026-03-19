"""
Quick demo — renders the grid world and saves it to results/figures/.
Run: .agentic-ai-final-project-env/Scripts/python demo_grid.py
"""

import matplotlib
matplotlib.use("Agg")

from src.environment.grid import Grid, GridConfig
from src.environment.visualizer import render_grid

# ── Build a Dynamic-High grid ─────────────────────────────────────────────────
cfg = GridConfig(
    size             = 20,
    obstacle_density = 0.20,
    dynamic_ratio    = 0.5,
    motion_model     = "random_walk",
    motion_speed     = 1.0,
    start            = (0, 0),
    goal             = (19, 19),
)
grid = Grid(cfg, seed=42)

# Simulate 15 steps so obstacles have moved
for _ in range(15):
    grid.step()
    # Mark wherever dynamic obstacles were as hazardous
    grid.update_hazard_map(grid._dynamic_positions, increment=0.12, decay=0.97)

# Fake a planned path diagonally across the grid
path = [(i, i) for i in range(20)]

# ── Render and save ───────────────────────────────────────────────────────────
fig = render_grid(
    grid,
    path      = path,
    title     = f"Grid World — step {grid.step_count}  |  Dynamic-High",
    save_path = "results/figures/demo_grid.png",
)

import matplotlib.pyplot as plt
plt.close(fig)

print("Saved: results/figures/demo_grid.png")
