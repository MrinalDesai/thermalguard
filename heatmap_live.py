"""ThermalGuard live heatmap — Weekend 2 renderer.

Works tonight on the bench (4 sensors, classic Uno over serial) and unchanged
on the full 82-sensor build. Four wall panels side by side (the layout the
400x1280 display will use), fixed colour scale, per-cell readouts, missing
cells greyed.

Usage:
  python heatmap_live.py --source sim                    # no hardware
  python heatmap_live.py --source serial --port COM5     # bench Uno
Options:
  --vmin 20 --vmax 40      fixed colour scale (NEVER auto-scale)
  --save out.png           render N frames headless then save (testing)

Bench convenience: ROMs not found in config/sensor_map.json are auto-assigned
to Wall N row 0 onward, so the 4 bench sensors appear with zero config. For
the real build, map ROMs properly in sensor_map.json (auto-slots print a
warning so you can't forget).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

WALLS = ["N", "E", "S", "W"]
ROWS, COLS = 4, 5
MAP_PATH = Path(__file__).parent.parent / "config" / "sensor_map.json"


def load_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    return {}


class AutoMapper:
    """Wraps the ROM map; assigns unknown ROMs to free Wall N slots."""

    def __init__(self, rom_map):
        self.map = dict(rom_map)
        taken = {(v["wall"], v.get("row"), v.get("col"))
                 for v in self.map.values() if v["wall"] != "AMB"}
        self.free = [(w, r, c) for w in WALLS for r in range(ROWS)
                     for c in range(COLS) if (w, r, c) not in taken]

    def place(self, rom):
        if rom in self.map:
            return self.map[rom]
        if not self.free:
            return None
        w, r, c = self.free.pop(0)
        self.map[rom] = {"wall": w, "row": r, "col": c, "offset": 0.0}
        print(f"[auto-map] {rom} -> {w} r{r} c{c} (add to sensor_map.json)")
        return self.map[rom]


def frame_to_state(frame, mapper):
    mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
    ambient = []
    for bus in frame["buses"]:
        for s in bus["sensors"]:
            if not s["ok"]:
                continue
            info = mapper.place(s["rom"])
            if info is None:
                continue
            t = s["t"] + info.get("offset", 0.0)
            if info["wall"] == "AMB":
                ambient.append(t)
            else:
                mats[info["wall"]][info["row"], info["col"]] = t
    amb = float(np.mean(ambient)) if ambient else float("nan")
    return mats, amb


# ------------------------------------------------------------------ sources
def frames_sim():
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from simulator import ThermalSim

    sim = ThermalSim()
    n = 0
    while True:
        n += 1
        if n == 15:                       # scripted demo fault
            sim.inject_hotspot("N", 1, 2, power=0.6)
            print("[sim] hotspot injected at N r1 c2")
        yield sim.step()
        time.sleep(0.5)


def frames_serial(port_name):
    import serial  # pip install pyserial

    port = serial.Serial(port_name, 115200, timeout=5)
    while True:
        line = port.readline().decode(errors="ignore").strip()
        if line.startswith('{"seq"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # partial line at startup


# ------------------------------------------------------------------ render
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["sim", "serial"], default="sim")
    ap.add_argument("--port", default="COM5")
    ap.add_argument("--vmin", type=float, default=20.0)
    ap.add_argument("--vmax", type=float, default=40.0)
    ap.add_argument("--save", default=None,
                    help="headless: render 30 frames, save last to this PNG")
    args = ap.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mapper = AutoMapper(load_map())
    src = frames_sim() if args.source == "sim" else frames_serial(args.port)

    plt.ion()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.canvas.manager.set_window_title("ThermalGuard") if not args.save else None
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="#777777")

    images, texts = {}, {}
    for ax, w in zip(axes, WALLS):
        im = ax.imshow(np.full((ROWS, COLS), np.nan), cmap=cmap,
                       vmin=args.vmin, vmax=args.vmax, aspect="equal")
        ax.set_title(f"Wall {w}")
        ax.set_xticks(range(COLS)); ax.set_yticks(range(ROWS))
        images[w] = im
        texts[w] = [[ax.text(c, r, "", ha="center", va="center",
                             fontsize=8, color="white")
                     for c in range(COLS)] for r in range(ROWS)]
    cbar = fig.colorbar(images["W"], ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("°C (fixed scale)")

    n = 0
    for frame in src:
        mats, amb = frame_to_state(frame, mapper)
        hot_t, hot_at = -999.0, "-"
        for w in WALLS:
            m = mats[w]
            images[w].set_data(np.ma.masked_invalid(m))
            for r in range(ROWS):
                for c in range(COLS):
                    v = m[r, c]
                    texts[w][r][c].set_text("" if np.isnan(v) else f"{v:.1f}")
            if np.any(~np.isnan(m)) and np.nanmax(m) > hot_t:
                hot_t = float(np.nanmax(m))
                r, c = np.unravel_index(np.nanargmax(m), m.shape)
                hot_at = f"{w} r{r} c{c}"
        fig.suptitle(
            f"seq {frame.get('seq')}   ambient {amb:.1f}°C   "
            f"max {hot_t:.1f}°C @ {hot_at}   Δamb {hot_t - amb:+.1f}°C",
            fontsize=11)

        if args.save:
            n += 1
            if n >= 30:
                fig.savefig(args.save, dpi=110, bbox_inches="tight")
                print(f"saved {args.save}")
                return
        else:
            plt.pause(0.05)


if __name__ == "__main__":
    main()
