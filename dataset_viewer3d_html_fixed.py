"""ThermalGuard interactive 3D — corrected hover behaviour.

Changes from the original:
1. Uses linear interpolation for the coloured wall surface to avoid cubic
   overshoot creating artificial hot/cold patches.
2. Disables hover on interpolated surfaces.
3. Overlays the real 4x5 sensor positions as Scatter3d markers; hover values
   now always come from the original sensor matrix, never from a hidden wall
   or an interpolated vertex.
4. Marks the hottest real sensor in each case.

Usage:
  python dataset_viewer3d_html_fixed.py
  python dataset_viewer3d_html_fixed.py --frame 115 --cmap inferno
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
EPS = 0.025  # moves markers just outside each wall to prevent z-fighting


def wall_surface_data(wall, mat, amb):
    """Return interpolated wall surface geometry and colour values."""
    raw = np.nan_to_num(np.asarray(mat, dtype=float), nan=amb)

    # Linear interpolation is visually smooth enough and does not overshoot
    # like cubic interpolation can around a sharp hotspot.
    img = zoom(raw, UP, order=1)

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
    else:  # W
        X, Y = np.zeros_like(H), H

    return X, Y, Z, img


def sensor_points(wall, mat, amb):
    """Return exact physical sensor positions and exact measured values."""
    raw = np.asarray(mat, dtype=float)

    # Sensor locations are the centres of the 4x5 grid cells.
    horizontal = np.arange(COLS, dtype=float) + 0.5
    vertical = ROWS - (np.arange(ROWS, dtype=float) + 0.5)
    H, V = np.meshgrid(horizontal, vertical)

    if wall == "N":
        X, Y = H, np.full_like(H, COLS + EPS)
    elif wall == "S":
        X, Y = COLS - H, np.full_like(H, -EPS)
    elif wall == "E":
        X, Y = np.full_like(H, COLS + EPS), COLS - H
    else:  # W
        X, Y = np.full_like(H, -EPS), H

    texts = []
    for r in range(ROWS):
        for c in range(COLS):
            value = raw[r, c]
            if np.isnan(value):
                texts.append(
                    f"Wall {wall}<br>R{r + 1} C{c + 1}<br>Sensor missing"
                )
            else:
                texts.append(
                    f"Wall {wall}<br>R{r + 1} C{c + 1}"
                    f"<br><b>{value:.1f}°C</b>"
                    f"<br>Δ ambient {value - amb:+.1f}°C"
                )

    return X.ravel(), Y.ravel(), V.ravel(), raw.ravel(), texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=110)
    ap.add_argument(
        "--cmap", default="turbo",
        help="Plotly colourscale: turbo | inferno | jet"
    )
    ap.add_argument("--out", default="dataset_cases_3d_fixed.html")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "scene"}] * 2] * 2,
        horizontal_spacing=0.02,
        vertical_spacing=0.06,
        subplot_titles=[""] * 4,
    )

    titles = []

    for i, kind in enumerate(CASES):
        episode = run_case(kind, steps=args.frame + 1)
        mats, amb, score = episode[args.frame]
        subplot_row, subplot_col = i // 2 + 1, i % 2 + 1

        hottest = None

        for wall in WALLS:
            X, Y, Z, C = wall_surface_data(wall, mats[wall], amb)

            # Interpolated surface is display-only. Disabling hover prevents
            # Plotly from selecting a hidden/rear wall or an interpolated point.
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z,
                    surfacecolor=C,
                    colorscale=args.cmap,
                    cmin=VMIN,
                    cmax=VMAX,
                    showscale=(i == 0 and wall == "N"),
                    colorbar=dict(title="°C", len=0.5, x=1.02),
                    lighting=dict(ambient=0.9, diffuse=0.3),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=subplot_row,
                col=subplot_col,
            )

            sx, sy, sz, values, hover_text = sensor_points(
                wall, mats[wall], amb
            )

            fig.add_trace(
                go.Scatter3d(
                    x=sx,
                    y=sy,
                    z=sz,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=values,
                        colorscale=args.cmap,
                        cmin=VMIN,
                        cmax=VMAX,
                        showscale=False,
                        line=dict(width=1),
                    ),
                    text=hover_text,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                    name=f"Wall {wall} sensors",
                ),
                row=subplot_row,
                col=subplot_col,
            )

            raw = np.asarray(mats[wall], dtype=float)
            if not np.all(np.isnan(raw)):
                flat_index = int(np.nanargmax(raw))
                rr, cc = np.unravel_index(flat_index, raw.shape)
                value = float(raw[rr, cc])
                wall_candidate = {
                    "wall": wall,
                    "row": rr,
                    "col": cc,
                    "value": value,
                    "x": sx[flat_index],
                    "y": sy[flat_index],
                    "z": sz[flat_index],
                }
                if hottest is None or value > hottest["value"]:
                    hottest = wall_candidate

        # Floor
        fx, fy = np.meshgrid(
            np.linspace(0, COLS, 2), np.linspace(0, COLS, 2)
        )
        fig.add_trace(
            go.Surface(
                x=fx,
                y=fy,
                z=np.zeros_like(fx),
                surfacecolor=np.zeros_like(fx),
                colorscale=[[0, "#3a3a3a"], [1, "#3a3a3a"]],
                showscale=False,
                opacity=0.5,
                hoverinfo="skip",
                showlegend=False,
            ),
            row=subplot_row,
            col=subplot_col,
        )

        # Explicit marker for the hottest real sensor.
        if hottest is not None:
            max_text = (
                f"MAX — Wall {hottest['wall']}"
                f"<br>R{hottest['row'] + 1} C{hottest['col'] + 1}"
                f"<br><b>{hottest['value']:.1f}°C</b>"
                f"<br>Δ ambient {hottest['value'] - amb:+.1f}°C"
            )
            fig.add_trace(
                go.Scatter3d(
                    x=[hottest["x"]],
                    y=[hottest["y"]],
                    z=[hottest["z"]],
                    mode="markers+text",
                    marker=dict(size=9, symbol="diamond", line=dict(width=2)),
                    text=["MAX"],
                    textposition="top center",
                    customdata=[max_text],
                    hovertemplate="%{customdata}<extra></extra>",
                    showlegend=False,
                ),
                row=subplot_row,
                col=subplot_col,
            )

        max_value = hottest["value"] if hottest else float("nan")
        titles.append(
            f"<b>{kind.upper()}</b> — score {score:.2f} "
            f"(max {max_value:.1f}°C, amb {amb:.1f}°C)"
        )

    scene_cfg = dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False),
        aspectmode="manual",
        aspectratio=dict(x=1, y=1, z=0.66),
        camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
    )

    fig.update_layout(
        height=900,
        width=1200,
        title=(
            f"ThermalGuard — normal vs fault classes (frame {args.frame}). "
            "Surface is linearly interpolated; hover dots are exact sensors."
        ),
        scene=scene_cfg,
        scene2=scene_cfg,
        scene3=scene_cfg,
        scene4=scene_cfg,
        margin=dict(l=10, r=80, t=90, b=10),
        hovermode="closest",
    )

    for ann, title in zip(fig.layout.annotations, titles):
        ann.text = title
        ann.font.size = 13

    out = Path(args.out).resolve()
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"saved {out}")

    if not args.no_open:
        webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
