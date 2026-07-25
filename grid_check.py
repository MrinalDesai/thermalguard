"""Verification renders: flat wall grid, sensor number + temp in black, hi-res."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simulator import ThermalSim, WALLS, ROWS, COLS
from thermal3d import frame_to_walls

VMIN, VMAX = 20.0, 40.0

def episode(kind, seed):
    sim = ThermalSim(seed=seed)
    for i in range(110):
        if i == 40:
            if kind == "hotspot":
                sim.inject_hotspot("N", 1, 3, power=0.7)
            elif kind == "cluster":
                for dr, dc in [(0,0),(0,1),(1,0)]:
                    sim.inject_hotspot("N", 1+dr, 3+dc, power=0.5)
        frame = sim.step()
    return frame_to_walls(frame)

def render(name, kind, seed):
    mats, amb = episode(kind, seed)
    m = mats["N"]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=200)
    im = ax.imshow(m, cmap="turbo", vmin=VMIN, vmax=VMAX)
    ax.set_xticks(np.arange(-.5, COLS, 1), minor=True)
    ax.set_yticks(np.arange(-.5, ROWS, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=2)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for r in range(ROWS):
        for c in range(COLS):
            n = r * COLS + c + 1
            ax.text(c, r - 0.18, f"S{n:02d}", ha="center", va="center",
                    fontsize=13, color="black", fontweight="bold")
            ax.text(c, r + 0.18, f"{m[r, c]:.2f}°C", ha="center",
                    va="center", fontsize=12, color="black")
    title = f"{name}  —  Wall N  (ambient {amb:.2f}°C, max {np.nanmax(m):.2f}°C)"
    ax.set_title(title, fontsize=14)
    fig.colorbar(im, ax=ax, label="°C (fixed 20–40)")
    out = f"/mnt/user-data/outputs/{name}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)

render("check_normal_1", "normal", seed=11)
render("check_normal_2", "normal", seed=22)
render("check_fault_hotspot", "hotspot", seed=33)
render("check_fault_cluster", "cluster", seed=44)
