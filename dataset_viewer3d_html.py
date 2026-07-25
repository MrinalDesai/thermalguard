"""ThermalGuard interactive 3D — normal vs each fault, fully rotatable.

Generates dataset_cases_3d.html: four 3D thermal enclosures (NORMAL,
HOTSPOT, RUNAWAY, CLUSTER) in a 2x2 grid, each independently drag-rotatable
in the browser, zoomable, with the trained model's anomaly score in each
title. Fixed colour scale across all four for honest comparison.

Usage:
  python dataset_viewer3d_html.py            # writes + opens the HTML
  python dataset_viewer3d_html.py --frame 115 --cmap inferno
Requires: pip install plotly
"""

import argparse
import webbrowser
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from simulator import WALLS, ROWS, COLS
from dataset_viewer import run_case, CASES

VMIN, VMAX = 20.0, 40.0
UP = 8


def wall_surface_data(wall, mat, amb):
    """Return (x, y, z, colorvals) 2D arrays for one wall face."""
    img = zoom(np.nan_to_num(np.asarray(mat, float), nan=amb), UP, order=3)
    r, c = img.shape
    h = np.linspace(0, COLS, c)
    v = np.linspace(ROWS, 0, r)
    H, V = np.meshgrid(h, v)
    Z = V
    if wall == "N":
        X, Y = H, np.full_like(H, COLS)
    elif wall == "S":
        X, Y = COLS - H, np.zeros_like(H)
    elif wall == "E":
        X, Y = np.full_like(H, COLS), COLS - H
    else:
        X, Y = np.zeros_like(H), H
    return X, Y, Z, img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=110)
    ap.add_argument("--cmap", default="turbo",
                    help="plotly colorscale: turbo | inferno | jet")
    ap.add_argument("--out", default="dataset_cases_3d.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "scene"}] * 2] * 2,
        horizontal_spacing=0.02, vertical_spacing=0.06,
        subplot_titles=[""] * 4)

    titles = []
    for i, kind in enumerate(CASES):
        episode = run_case(kind, steps=args.frame + 1)
        mats, amb, score = episode[args.frame]
        row, col = i // 2 + 1, i % 2 + 1

        for w in WALLS:
            X, Y, Z, C = wall_surface_data(w, mats[w], amb)
            fig.add_trace(
                go.Surface(x=X, y=Y, z=Z, surfacecolor=C,
                           colorscale=args.cmap, cmin=VMIN, cmax=VMAX,
                           showscale=(i == 0 and w == "N"),
                           colorbar=dict(title="°C", len=0.5, x=1.02),
                           lighting=dict(ambient=0.9, diffuse=0.3),
                           hovertemplate="%{surfacecolor:.1f}°C<extra></extra>"),
                row=row, col=col)
        # floor
        fx, fy = np.meshgrid(np.linspace(0, COLS, 2), np.linspace(0, COLS, 2))
        fig.add_trace(go.Surface(x=fx, y=fy, z=np.zeros_like(fx),
                                 surfacecolor=np.zeros_like(fx),
                                 colorscale=[[0, "#3a3a3a"], [1, "#3a3a3a"]],
                                 showscale=False, opacity=0.5),
                      row=row, col=col)

        hot = max(float(np.nanmax(mats[w])) for w in WALLS)
        titles.append(f"<b>{kind.upper()}</b> — score {score:.2f} "
                      f"(max {hot:.1f}°C, amb {amb:.1f}°C)")

    scene_cfg = dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        zaxis=dict(visible=False), aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.66),
        camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)))
    fig.update_layout(
        height=900, width=1200,
        title=f"ThermalGuard — normal vs fault classes (frame {args.frame}). "
              f"Drag any box to rotate.",
        scene=scene_cfg, scene2=scene_cfg,
        scene3=scene_cfg, scene4=scene_cfg,
        margin=dict(l=10, r=80, t=90, b=10))
    for ann, t in zip(fig.layout.annotations, titles):
        ann.text = t
        ann.font.size = 13

    out = Path(args.out).resolve()
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved {out}")
    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
