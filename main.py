"""ThermalGuard Edge — Linux-side pipeline (Arduino UNO Q, Qualcomm/Debian side).

Weekend-1 scope:
  frame source (hardware Bridge OR simulator) -> ROM->position mapping
  -> per-wall 4x5 matrices -> SQLite logging -> console status line.

Heatmap rendering, features, Isolation Forest, state machine bolt on top of
get_matrices() in Weekend 2 without touching anything here.

Run:
  python main.py                 # hardware via Bridge (inside App Lab app)
  python main.py --source sim    # no hardware needed; uses simulator.py
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

WALLS = ["N", "E", "S", "W"]
ROWS, COLS = 4, 5
CYCLE_S = 2.0
DB_PATH = Path(__file__).parent / "thermalguard.db"
MAP_PATH = Path(__file__).parent.parent / "config" / "sensor_map.json"


# ----------------------------------------------------------------- storage
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS frames (
               ts REAL, seq INTEGER, ambient_c REAL,
               matrices TEXT,          -- JSON: {wall: [[...]]}, NaN for missing
               raw TEXT                -- full source frame for reprocessing
           )"""
    )
    con.commit()
    return con


# ----------------------------------------------------------------- mapping
def load_map():
    """sensor_map.json: { ROM_HEX: {"wall": "N", "row": 0, "col": 0,
                                    "offset": 0.0} , ...,
                          ambient ROMs use {"wall": "AMB"} }"""
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    print(f"[warn] {MAP_PATH} missing — running unmapped (census mode)")
    return {}


def frame_to_matrices(frame, rom_map):
    """Returns ({wall: 4x5 ndarray with NaN gaps}, ambient_c, unmapped_roms)."""
    mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
    ambient, unmapped = [], []
    for bus in frame["buses"]:
        for s in bus["sensors"]:
            if not s["ok"]:
                continue
            info = rom_map.get(s["rom"])
            if info is None:
                unmapped.append(s["rom"])
                continue
            t = s["t"] + info.get("offset", 0.0)
            if info["wall"] == "AMB":
                ambient.append(t)
            else:
                mats[info["wall"]][info["row"], info["col"]] = t
    amb = float(np.mean(ambient)) if ambient else float("nan")
    return mats, amb, unmapped


# ----------------------------------------------------------------- sources
def source_hardware(port_name="COM6"):
    """Read JSON frames from the bench Uno / UNO Q serial port.
    (Bridge variant for App Lab deployment: swap this body for
    Bridge.call("get_frame") per README.)"""
    import serial  # pip install pyserial

    port = serial.Serial(port_name, 115200, timeout=5)
    while True:
        line = port.readline().decode(errors="ignore").strip()
        if line.startswith('{"seq"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def source_sim():
    from simulator import ThermalSim

    sim = ThermalSim()
    while True:
        yield sim.step()
        time.sleep(CYCLE_S)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["hardware", "sim"], default="hardware")
    ap.add_argument("--port", default="COM6")
    args = ap.parse_args()

    con = init_db()
    rom_map = load_map()
    src = source_sim() if args.source == "sim" else source_hardware(args.port)

    print(f"ThermalGuard pipeline up | source={args.source}")
    for frame in src:
        mats, amb, unmapped = frame_to_matrices(frame, rom_map)
        con.execute(
            "INSERT INTO frames VALUES (?,?,?,?,?)",
            (
                time.time(),
                frame.get("seq", -1),
                amb,
                json.dumps({w: np.where(np.isnan(m), None, m).tolist()
                            for w, m in mats.items()}),
                json.dumps(frame),
            ),
        )
        con.commit()

        live = {w: int(np.sum(~np.isnan(m))) for w, m in mats.items()}
        hot_w, hot_t = "-", float("nan")
        for w, m in mats.items():
            if np.any(~np.isnan(m)) and (np.isnan(hot_t) or np.nanmax(m) > hot_t):
                hot_w, hot_t = w, float(np.nanmax(m))
        print(
            f"seq={frame.get('seq'):>6} amb={amb:5.1f}C "
            f"live={live} max={hot_t:5.1f}C@{hot_w} "
            f"unmapped={len(unmapped)}"
        )


if __name__ == "__main__":
    main()
