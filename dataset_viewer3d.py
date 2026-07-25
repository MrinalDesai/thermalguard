"""ThermalGuard dataset viewer 3D — the four training classes as thermal boxes.

Renders one representative frame per episode class (normal, hotspot,
runaway, cluster) as a 3D enclosure — four subplots, each annotated with
the trained model's anomaly score for that exact frame.

Usage:
  python dataset_viewer3d.py                  # save dataset_cases_3d.png
  python dataset_viewer3d.py --cmap turbo
  python dataset_viewer3d.py --frame 115
"""

import argparse

import numpy as np

from simulator import WALLS, ROWS, COLS
from thermal3d import wall_surface, VMIN, VMAX
from dataset_viewer import run_case, CASES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--frame", type=int, default=110)
    ap.add_argument("--out", default="dataset_cases_3d.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(args.cmap)
    norm = Normalize(VMIN, VMAX)

    fig = plt.figure(figsize=(14, 11))
    for i, kind in enumerate(CASES):
        episode = run_case(kind, steps=args.frame + 1)
        mats, amb, score = episode[args.frame]

        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for w in WALLS:
            wall_surface(ax, w, mats[w], cmap, norm)
        fx, fy = np.meshgrid([0, COLS], [0, COLS])
        ax.plot_surface(fx, fy, np.zeros_like(fx), color="#444444",
                        alpha=0.4)

        hot_t = max(np.nanmax(mats[w]) for w in WALLS)
        ax.set_title(f"{kind.upper()} — score {score:.2f}   "
                     f"(amb {amb:.1f}°C, max {hot_t:.1f}°C)",
                     fontsize=12)
        ax.set_box_aspect((1, 1, 0.66))
        ax.set_axis_off()
        ax.view_init(elev=22, azim=-55)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=fig.axes, fraction=0.02, pad=0.02,
                 label="°C (fixed scale)")
    fig.suptitle("ThermalGuard training classes — 3D enclosure view, "
                 f"frame {args.frame}", fontsize=14)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
