"""ThermalGuard 3D thermal view — the enclosure as a thermal camera would see it.

Renders the four sensing walls as faces of a 3D box, each wall's 4x6 sensor
grid bicubic-interpolated to a smooth thermal image (fixed colour scale,
sensor positions marked). Works from the simulator today and from real
frames the moment walls are wired — same frame format throughout.

Usage:
  python thermal3d.py                          # sim, scripted hotspot, save PNG
  python thermal3d.py --live                   # interactive rotating window
  python thermal3d.py --hotspot E --row 2 --col 4
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

WALLS = ["N", "E", "S", "W"]
ROWS, COLS = 4, 6
VMIN, VMAX = 20.0, 40.0
UP = 12  # interpolation upsample factor


def smooth(mat):
    """4x6 -> smooth image via bicubic zoom, NaNs filled with ambient-ish."""
    m = np.array(mat, dtype=float)
    if np.isnan(m).any():
        fill = np.nanmean(m) if np.any(~np.isnan(m)) else VMIN
        m = np.where(np.isnan(m), fill, m)
    return zoom(m, UP, order=3)


def wall_surface(ax, wall, mat, cmap, norm):
    """Draw one wall as a coloured vertical face of the box."""
    img = smooth(mat)                      # (4*UP, 6*UP)
    colors = cmap(norm(img))
    r, c = img.shape
    # horizontal span 0..6, vertical 0..4 (row 0 at top)
    h = np.linspace(0, COLS, c + 1)
    v = np.linspace(ROWS, 0, r + 1)
    H, V = np.meshgrid(h, v)
    Z = V
    if wall == "N":
        X, Y = H, np.full_like(H, COLS)
    elif wall == "S":
        X, Y = COLS - H, np.zeros_like(H)  # flip so viewed-from-outside reads L->R
    elif wall == "E":
        X, Y = np.full_like(H, COLS), COLS - H
    else:  # W
        X, Y = np.zeros_like(H), H
    ax.plot_surface(X, Y, Z, facecolors=colors, rstride=1, cstride=1,
                    shade=False, antialiased=False)
    # sensor position markers
    for rr in range(ROWS):
        for cc in range(COLS):
            hh = (cc + 0.5) / COLS * COLS
            vv = ROWS - (rr + 0.5) / ROWS * ROWS
            if wall == "N":
                ax.scatter(hh, COLS, vv, c="white", s=4, depthshade=False)
            elif wall == "S":
                ax.scatter(COLS - hh, 0, vv, c="white", s=4, depthshade=False)
            elif wall == "E":
                ax.scatter(COLS, COLS - hh, vv, c="white", s=4, depthshade=False)
            else:
                ax.scatter(0, hh, vv, c="white", s=4, depthshade=False)


def frame_to_walls(frame):
    """96 wall sensors -> {wall: 4x6}, using simulator's deterministic layout."""
    mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
    ambient = []
    for bus in frame["buses"]:
        b = bus["bus"]
        if b == 8:
            ambient += [s["t"] for s in bus["sensors"] if s["ok"]]
            continue
        wall = WALLS[b // 2]
        half = b % 2
        for k, s in enumerate(bus["sensors"]):
            if not s["ok"]:
                continue
            idx = half * 12 + k
            mats[wall][idx // COLS, idx % COLS] = s["t"]
    amb = float(np.mean(ambient)) if ambient else float("nan")
    return mats, amb


def render(mats, amb, out=None, live=False, cmap_name="inferno"):
    import matplotlib
    if not live:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap(cmap_name)
    norm = Normalize(VMIN, VMAX)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    for w in WALLS:
        wall_surface(ax, w, mats[w], cmap, norm)

    # floor for context
    fx, fy = np.meshgrid([0, COLS], [0, COLS])
    ax.plot_surface(fx, fy, np.zeros_like(fx), color="#444444", alpha=0.4)

    hot_t = max(np.nanmax(mats[w]) for w in WALLS)
    hot_w = max(WALLS, key=lambda w: np.nanmax(mats[w]))
    r, c = np.unravel_index(np.nanargmax(mats[hot_w]), (ROWS, COLS))
    ax.set_title(f"ThermalGuard — 96-sensor enclosure   "
                 f"ambient {amb:.1f}°C   max {hot_t:.1f}°C @ Wall {hot_w} "
                 f"r{r} c{c}", fontsize=11)
    ax.set_box_aspect((1, 1, 0.66))
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(m, ax=ax, fraction=0.03, pad=0.02, label="°C (fixed scale)")

    if live:
        plt.show()
    else:
        fig.savefig(out, dpi=120, bbox_inches="tight")
        print(f"saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--hotspot", default="N")
    ap.add_argument("--row", type=int, default=1)
    ap.add_argument("--col", type=int, default=3)
    ap.add_argument("--out", default="thermal3d.png")
    ap.add_argument("--cmap", default="inferno",
                    help="inferno (FLIR ironbow-like) | turbo/jet (classic "
                         "blue-to-yellow thermal-camera look)")
    ap.add_argument("--big", action="store_true",
                    help="large multi-cell hotspot spreading across the wall")
    args = ap.parse_args()

    from simulator import ThermalSim
    sim = ThermalSim()
    for i in range(60 if args.big else 40):
        if i == 10:
            if args.big:
                # cluster of sources around the epicentre -> broad hot region
                for dr, dc, p in [(0, 0, 1.4), (0, 1, 0.9), (1, 0, 0.9),
                                  (0, -1, 0.6), (-1, 0, 0.6), (1, 1, 0.5)]:
                    rr = min(max(args.row + dr, 0), ROWS - 1)
                    cc = min(max(args.col + dc, 0), COLS - 1)
                    sim.inject_hotspot(args.hotspot, rr, cc, power=p,
                                       growth=1.01)
            else:
                sim.inject_hotspot(args.hotspot, args.row, args.col,
                                   power=0.9)
        frame = sim.step()
    mats, amb = frame_to_walls(frame)
    render(mats, amb, out=args.out, live=args.live, cmap_name=args.cmap)


if __name__ == "__main__":
    main()
