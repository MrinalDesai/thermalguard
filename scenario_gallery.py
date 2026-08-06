"""ThermalGuard scenario gallery — every demo scenario, imaged in 2D and 3D.

Scenarios:
  S1 warmup      base temperature rises 26->31C uniformly; system must stay
                 NORMAL (proves ambient-adjusted features)
  S2 hotspot     single localized source, moderate
  S3 runaway     exponential growth at one cell (capped at physical limit)
  S4 cluster     multi-cell spreading region
  S5 dead_sensor one probe fails to 85C/invalid; grid shows honest gap

For each: runs the episode, captures the final frame, renders
  scenario_<name>_2d.png   gridded per-wall values (black grid, cell temps)
  scenario_<name>_3d.png   3D enclosure view
with the trained model's anomaly score in every title.

Usage:  python scenario_gallery.py            # all five
        python scenario_gallery.py --frame 100 --cmap turbo
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np

from simulator import ThermalSim, WALLS, ROWS, COLS
from thermal3d import frame_to_walls, wall_surface, VMIN, VMAX
from ml_train import frame_features, WINDOW

SCENARIOS = ["warmup", "hotspot", "runaway", "cluster", "dead_sensor"]


def load_scorer():
    import joblib
    b = joblib.load(Path(__file__).parent / "model.joblib")
    return b["model"], b["scaler"], b["score_range"]


def run_scenario(kind, frames=110, seed=7):
    model, scaler, (lo, hi) = load_scorer()
    sim = ThermalSim(seed=seed)
    hist = deque(maxlen=WINDOW)
    inject = 40
    for i in range(frames):
        if kind == "warmup":
            sim.ambient += 0.03          # everything warms together (~3C)
            for w in WALLS:
                sim.walls[w] += 0.03
        if i == inject:
            if kind == "hotspot":
                sim.inject_hotspot("N", 1, 3, power=0.7)
            elif kind == "runaway":
                sim.inject_runaway("N", 1, 3)
            elif kind == "cluster":
                for dr, dc in [(0, 0), (0, 1), (1, 0)]:
                    sim.inject_hotspot("N", 1 + dr, 3 + dc, power=0.5)
            elif kind == "dead_sensor":
                sim.kill_sensor(0, 9)    # wall N, idx 9 -> r1 c3
        frame = sim.step()
        mats, amb = frame_to_walls(frame)
        feats = frame_features(mats, amb, hist)
        hist.append(np.stack([np.nan_to_num(mats[w], nan=amb)
                              for w in WALLS]))
        raw = -model.score_samples(scaler.transform([feats]))[0]
        score = float(np.clip((raw - lo) / (hi - lo), 0, 1))
    return mats, amb, score


def render_2d(name, mats, amb, score, cmap_name, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=150)
    for ax, w in zip(axes, WALLS):
        m = np.asarray(mats[w], float)
        shown = np.nan_to_num(m, nan=amb)
        ax.imshow(shown, cmap=cmap_name, vmin=VMIN, vmax=VMAX)
        ax.set_xticks(np.arange(-.5, COLS), minor=True)
        ax.set_yticks(np.arange(-.5, ROWS), minor=True)
        ax.grid(which="minor", color="black", linewidth=1.5)
        ax.tick_params(which="both", length=0,
                       labelbottom=False, labelleft=False)
        for r in range(ROWS):
            for c in range(COLS):
                v = m[r, c]
                ax.text(c, r, "X" if np.isnan(v) else f"{v:.1f}",
                        ha="center", va="center", fontsize=9, color="black")
        ax.set_title(f"Wall {w}", fontsize=11)
    fig.suptitle(f"{name.upper()} — score {score:.2f}   "
                 f"ambient {amb:.1f}°C", fontsize=14)
    out = outdir / f"scenario_{name}_2d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_3d(name, mats, amb, score, cmap_name, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(VMIN, VMAX)
    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    safe = {w: np.nan_to_num(np.asarray(mats[w], float), nan=amb)
            for w in WALLS}
    for w in WALLS:
        wall_surface(ax, w, safe[w], cmap, norm)
    fx, fy = np.meshgrid([0, COLS], [0, COLS])
    ax.plot_surface(fx, fy, np.zeros_like(fx), color="#444444", alpha=0.4)
    hot = max(float(np.nanmax(safe[w])) for w in WALLS)
    ax.set_title(f"{name.upper()} — score {score:.2f}  "
                 f"(amb {amb:.1f}°C, max {hot:.1f}°C)", fontsize=12)
    ax.set_box_aspect((1, 1, 0.66))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02, label="°C")
    out = outdir / f"scenario_{name}_3d.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=110)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    outdir = Path(args.outdir)

    for name in SCENARIOS:
        mats, amb, score = run_scenario(name, frames=args.frame)
        p2 = render_2d(name, mats, amb, score, args.cmap, outdir)
        p3 = render_3d(name, mats, amb, score, args.cmap, outdir)
        print(f"{name:12s} score {score:.2f}  -> {p2.name}, {p3.name}")


if __name__ == "__main__":
    main()
