"""ThermalGuard dataset viewer — see what the ML trained on.

Regenerates one representative episode per class (normal, hotspot, runaway,
cluster) with the same physics as ml_train.py, renders the four walls of a
late frame as smooth thermal images, and annotates each row with the trained
model's anomaly score for that exact frame.

Output: dataset_cases.png — a 4x4 grid (rows = cases, cols = walls),
report-figure ready.

Usage:
  python dataset_viewer.py                 # save PNG
  python dataset_viewer.py --cmap turbo    # classic thermal-camera palette
  python dataset_viewer.py --frame 100     # which frame of the episode
"""

import argparse
from collections import deque
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from simulator import ThermalSim, WALLS, ROWS, COLS
from thermal3d import frame_to_walls
from ml_train import frame_features, WINDOW

CASES = ["normal", "hotspot", "runaway", "cluster"]
VMIN, VMAX = 20.0, 40.0
UP = 12


def run_case(kind, steps, seed=7):
    """Replays one episode; returns (mats, amb, score_history) at each step."""
    import joblib
    bundle = joblib.load(Path(__file__).parent / "model.joblib")
    model, scaler = bundle["model"], bundle["scaler"]
    lo, hi = bundle["score_range"]

    sim = ThermalSim(seed=seed)
    rng = np.random.default_rng(seed)
    hist = deque(maxlen=WINDOW)
    out = []
    inject_at = 40
    for i in range(steps):
        if kind != "normal" and i == inject_at:
            w, r, c = "N", 1, 3  # fixed epicentre for comparable figures
            if kind == "hotspot":
                sim.inject_hotspot(w, r, c, power=0.7)
            elif kind == "runaway":
                sim.inject_hotspot(w, r, c, power=0.05, growth=1.06)
            elif kind == "cluster":
                for dr, dc in [(0, 0), (0, 1), (1, 0)]:
                    sim.inject_hotspot(w, min(r + dr, ROWS - 1),
                                       min(c + dc, COLS - 1), power=0.5)
        frame = sim.step()
        mats, amb = frame_to_walls(frame)
        feats = frame_features(mats, amb, hist)
        hist.append(np.stack([mats[w] for w in WALLS]))
        raw = -model.score_samples(scaler.transform([feats]))[0]
        score = float(np.clip((raw - lo) / (hi - lo), 0, 1))
        out.append((mats, amb, score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", default="inferno")
    ap.add_argument("--frame", type=int, default=110)
    ap.add_argument("--out", default="dataset_cases.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(VMIN, VMAX)

    fig, axes = plt.subplots(len(CASES), len(WALLS),
                             figsize=(13, 9), constrained_layout=True)
    for ri, kind in enumerate(CASES):
        episode = run_case(kind, steps=args.frame + 1)
        mats, amb, score = episode[args.frame]
        for ci, w in enumerate(WALLS):
            ax = axes[ri][ci]
            img = zoom(np.nan_to_num(mats[w], nan=amb), UP, order=3)
            ax.imshow(img, cmap=cmap, vmin=VMIN, vmax=VMAX)
            # sensor markers
            ys, xs = np.meshgrid(range(ROWS), range(COLS), indexing="ij")
            ax.scatter((xs + 0.5) * UP - 0.5, (ys + 0.5) * UP - 0.5,
                       c="white", s=3)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"Wall {w}", fontsize=11)
        axes[ri][0].set_ylabel(
            f"{kind.upper()}\nscore {score:.2f}", fontsize=11,
            rotation=0, ha="right", va="center", labelpad=45)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.01,
                 label="°C (fixed scale)")
    fig.suptitle(f"Training dataset classes — frame {args.frame} of each "
                 f"episode, model anomaly score per case", fontsize=13)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
