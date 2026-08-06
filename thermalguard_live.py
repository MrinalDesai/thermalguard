"""ThermalGuard LIVE — the resident application.

One window: the four-wall live heatmap. System state is shown as the
window's background colour (green/amber/orange/red/purple) plus a corner
message — no separate banner screen.

  NORMAL(green) -> WATCH(amber) -> WARNING(orange) -> CRITICAL(red)
  -> relay event -> ISOLATED(purple, latched; press R to reset)

Sources: --source sim (scripted runaway at --inject) or
         --source serial --port COM8|/dev/ttyUSB0 (real sensors)

Usage:
  python thermalguard_live.py --source sim --inject 25
  python thermalguard_live.py --source sim --timers 20 45 90   # production
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from simulator import ThermalSim, WALLS, ROWS, COLS
from thermal3d import frame_to_walls
from ml_train import AnomalyScorer
from statemachine import StateMachine, Relay, COLORS, THRESH

VMIN, VMAX = 20.0, 45.0


def frames_sim(inject_at):
    sim = ThermalSim()
    n = 0
    while True:
        n += 1
        if n == inject_at:
            sim.inject_runaway("N", 1, 3)
            print(f"[sim] runaway injected at frame {n}")
        yield sim.step()


def frames_serial(port_name):
    import serial
    port = serial.Serial(port_name, 115200, timeout=5)
    while True:
        line = port.readline().decode(errors="ignore").strip()
        if line.startswith('{"seq"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "serial"], default="sim")
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--inject", type=int, default=25)
    ap.add_argument("--timers", nargs=3, type=float, default=[5, 10, 20])
    ap.add_argument("--gpiochip", default=None)
    ap.add_argument("--line", default=None)
    ap.add_argument("--cycle", type=float, default=1.0)
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--save", default=None,
                    help="headless test: run N frames then save PNG")
    ap.add_argument("--save-frames", type=int, default=60)
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    relay = Relay(args.gpiochip, args.line)
    sm = StateMachine(*args.timers, relay)
    scorer = AnomalyScorer()
    src = frames_sim(args.inject) if args.source == "sim" \
        else frames_serial(args.port)

    plt.ion()
    import thermal3d as t3d
    t3d.UP = 6                      # coarser interpolation for live 3D speed
    from matplotlib import cm
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(16, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5],
                          left=0.03, right=0.94, top=0.80, bottom=0.05,
                          wspace=0.15, hspace=0.25)
    axes = {"N": fig.add_subplot(gs[0, 0]), "E": fig.add_subplot(gs[0, 1]),
            "S": fig.add_subplot(gs[1, 0]), "W": fig.add_subplot(gs[1, 1])}
    ax3d = fig.add_subplot(gs[:, 2], projection="3d")

    cmap = plt.get_cmap(args.cmap).copy()
    cmap.set_bad(color="#666666")
    norm = Normalize(VMIN, VMAX)

    images, texts = {}, {}
    for w, ax in axes.items():
        im = ax.imshow(np.full((ROWS, COLS), np.nan), cmap=cmap,
                       vmin=VMIN, vmax=VMAX, aspect="equal")
        ax.set_title(f"Wall {w}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        images[w] = im
        texts[w] = [[ax.text(c, r, "", ha="center", va="center",
                             fontsize=7, color="black")
                     for c in range(COLS)] for r in range(ROWS)]
    cbar = fig.colorbar(images["W"], ax=list(axes.values()),
                        fraction=0.03, pad=0.02)
    cbar.set_label("°C")

    corner = fig.text(0.015, 0.94, "", fontsize=16, fontweight="bold",
                      va="top", ha="left", color="white",
                      bbox=dict(boxstyle="round,pad=0.4",
                                facecolor="black", alpha=0.55))

    def on_key(ev):
        if ev.key in ("r", "R"):
            sm.reset()
    fig.canvas.mpl_connect("key_press_event", on_key)

    n = 0
    for frame in src:
        n += 1
        mats, amb = frame_to_walls(frame)
        score = scorer.score(mats, amb)
        state = sm.step(score, n * args.cycle)
        c = COLORS[state]

        fig.patch.set_facecolor(c)
        hot_t, hot_at = -999.0, "-"
        for w in WALLS:
            m = mats[w]
            images[w].set_data(np.ma.masked_invalid(m))
            for r in range(ROWS):
                for cc in range(COLS):
                    v = m[r, cc]
                    texts[w][r][cc].set_text(
                        "" if np.isnan(v) else f"{v:.1f}")
            if np.any(~np.isnan(m)) and np.nanmax(m) > hot_t:
                hot_t = float(np.nanmax(m))
                rr, ccx = np.unravel_index(np.nanargmax(m), m.shape)
                hot_at = f"{w} r{rr} c{ccx}"

        # --- live 3D box (right panel); keep the user's rotation ---
        elev, azim = ax3d.elev, ax3d.azim
        ax3d.clear()
        safe = {w: np.nan_to_num(np.asarray(mats[w], float), nan=amb)
                for w in WALLS}
        for w in WALLS:
            t3d.wall_surface(ax3d, w, safe[w], cmap, norm)
        fx, fy = np.meshgrid([0, COLS], [0, COLS])
        ax3d.plot_surface(fx, fy, np.zeros_like(fx), color="#444444",
                          alpha=0.4)
        ax3d.set_box_aspect((1, 1, 0.66))
        ax3d.set_axis_off()
        ax3d.view_init(elev=elev, azim=azim)

        msg = (f"{state}   score {score:.2f}   max {hot_t:.1f}°C @ {hot_at}"
               f"   Δamb {hot_t - amb:+.1f}°C   relay {relay.state}")
        if state == "ISOLATED":
            msg += "   [R = reset]"
        corner.set_text(msg)
        print(f"f{n:04d} {msg}")

        if args.save:
            if n >= args.save_frames:
                fig.savefig(args.save, dpi=110, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                print(f"saved {args.save}")
                relay.restore()
                return
        else:
            plt.pause(args.cycle)


if __name__ == "__main__":
    main()
