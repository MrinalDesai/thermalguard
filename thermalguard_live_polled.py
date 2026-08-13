"""ThermalGuard LIVE (POLLED) — for multiple acquisition boards.

Pair with the polled firmware: each board answers one frame per 'P'.
Run:  python thermalguard_live_polled.py --source serial

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

VMIN, VMAX = 22.0, 45.0


def frames_sim(inject_at):
    sim = ThermalSim()
    n = 0
    while True:
        n += 1
        if n == inject_at:
            sim.inject_runaway("N", 1, 3)
            print(f"[sim] runaway injected at frame {n}")
        yield sim.step()


MAP_PATH = Path(__file__).parent / "config" / "sensor_map.json"
REAL_ROWS, REAL_COLS = 4, 4


def load_rom_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    print(f"[warn] {MAP_PATH} missing — real sensors cannot be placed")
    return {}


def mats_from_mapped(frame, rom_map):
    """Place real sensors into 4x4 wall grids by sensor_map.json."""
    mats = {w: np.full((REAL_ROWS, REAL_COLS), np.nan) for w in WALLS}
    ambs = []
    unmapped = 0
    for bus in frame.get("buses", []):
        for s in bus.get("sensors", []):
            if not s.get("ok"):
                continue
            e = rom_map.get(s["rom"])
            if e is None:
                unmapped += 1
                continue
            if e.get("wall") == "AMB":
                ambs.append(s["t"])
            elif e.get("wall") in WALLS:
                r, c = e.get("row", 0), e.get("col", 0)
                if 0 <= r < REAL_ROWS and 0 <= c < REAL_COLS:
                    mats[e["wall"]][r, c] = s["t"] + e.get("offset", 0.0)
    amb = float(np.mean(ambs)) if ambs else         float(np.nanmean([np.nanmean(m) for m in mats.values()
                          if np.any(~np.isnan(m))]))
    return mats, amb, unmapped


def find_ports():
    """Every plausible Arduino serial port, Windows or Linux."""
    import serial.tools.list_ports
    out = []
    for p in serial.tools.list_ports.comports():
        blob = " ".join(str(x) for x in
                        (p.description, p.manufacturer, p.hwid))
        if any(k in blob for k in ("Arduino", "CH340", "CH341",
                                   "USB Serial", "CP210", "FT232")):
            out.append(p.device)
        elif p.device.startswith("/dev/ttyACM"):
            out.append(p.device)
    return sorted(set(out))


def frames_serial(port_name, cycle=1.0):
    """POLLED mode: boards stay silent until sent 'P'; one active
    stream at a time — no interleaving, any number of boards.
    Requires the polled firmware (loop waits for 'P')."""
    import serial
    import time as _t

    names = [port_name] if port_name not in (None, "auto") else find_ports()
    if not names:
        raise SystemExit("no serial ports found — boards plugged in?")
    print(f"[serial] POLLED mode, {len(names)} port(s): {', '.join(names)}")
    conns = []
    for nm in names:
        try:
            conns.append(serial.Serial(nm, 115200, timeout=3))
        except Exception as e:
            print(f"[serial] {nm} unavailable ({e}) — skipping")
    _t.sleep(2.5)                      # boards auto-reset on port open
    for c in conns:
        c.reset_input_buffer()

    merged, seq = {}, 0
    while True:
        for c in conns:
            try:
                c.reset_input_buffer()
                c.write(b'P')
                t_end = _t.time() + 2.5
                while _t.time() < t_end:
                    line = c.readline().decode(errors="ignore").strip()
                    if line.startswith('{"seq"'):
                        f = json.loads(line)
                        for bus in f.get("buses", []):
                            for s in bus.get("sensors", []):
                                if s.get("ok"):
                                    merged[s["rom"]] = s["t"]
                        break
            except Exception:
                continue
        seq += 1
        yield {"seq": seq, "buses": [{"bus": 0, "enabled": True,
               "sensors": [{"rom": r, "t": t, "ok": True}
                           for r, t in merged.items()]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "serial"], default="sim")
    ap.add_argument("--port", default="auto")
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

    rom_map = load_rom_map()
    relay = Relay(args.gpiochip, args.line)
    sm = StateMachine(*args.timers, relay)
    scorer = AnomalyScorer()
    src = frames_sim(args.inject) if args.source == "sim" \
        else frames_serial(args.port, args.cycle)

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

    g_rows = REAL_ROWS if args.source == "serial" else ROWS
    g_cols = REAL_COLS if args.source == "serial" else COLS
    images, texts = {}, {}
    for w, ax in axes.items():
        im = ax.imshow(np.full((g_rows, g_cols), np.nan), cmap=cmap,
                       vmin=VMIN, vmax=VMAX, aspect="equal")
        ax.set_title(f"Wall {w}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        images[w] = im
        texts[w] = [[ax.text(c, r, "", ha="center", va="center",
                             fontsize=7, color="black")
                     for c in range(g_cols)] for r in range(g_rows)]
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
        if args.source == "serial":
            mats, amb, unmapped = mats_from_mapped(frame, rom_map)
        else:
            mats, amb = frame_to_walls(frame)
            unmapped = 0
        try:
            score = scorer.score(mats, amb)
        except Exception:
            score = 0.0
        state = sm.step(score, n * args.cycle)
        c = COLORS[state]

        fig.patch.set_facecolor(c)
        hot_t, hot_at = -999.0, "-"
        for w in WALLS:
            m = mats[w]
            images[w].set_data(np.ma.masked_invalid(m))
            for r in range(m.shape[0]):
                for cc in range(m.shape[1]):
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
        for w in WALLS:
            m = np.asarray(mats[w], float)
            if np.all(np.isnan(m)):
                # no data on this wall: neutral grey face, no thermal paint
                grey = np.full_like(m, np.nan)
                t3d.wall_surface_blank(ax3d, w)
            else:
                fill = float(np.nanmean(m))
                t3d.wall_surface(ax3d, w,
                                 np.nan_to_num(m, nan=fill), cmap, norm)
        fx, fy = np.meshgrid([0, COLS], [0, COLS])
        ax3d.plot_surface(fx, fy, np.zeros_like(fx), color="#444444",
                          alpha=0.4)
        ax3d.set_box_aspect((1, 1, 0.66))
        ax3d.set_axis_off()
        ax3d.view_init(elev=elev, azim=azim)

        live_n = sum(int(np.sum(~np.isnan(mats[w]))) for w in WALLS)
        msg = (f"{state}   score {score:.2f}   max {hot_t:.1f}°C @ {hot_at}"
               f"   Δamb {hot_t - amb:+.1f}°C   relay {relay.state}")
        if args.source == "serial":
            msg += f"   live {live_n}" +                    (f" (+{unmapped} unmapped)" if unmapped else "")
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
