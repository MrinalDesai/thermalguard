"""ThermalGuard mapping assist — name each sensor by warming it.

Workflow:
  1. Run this with the bench streaming (close other serial programs first)
  2. It learns each ROM's baseline temperature (~15s)
  3. Warm ONE probe (blower/pinch). The ROM that rises above baseline
     by THRESH is announced.
  4. Type its position:  N 1 3   (wall row col)  — walls N/E/S/W,
     rows 0-3, cols 0-3 for the 4x4 layout. Or:
        amb        -> mark as ambient reference
        skip       -> ignore this spike
  5. It writes config/sensor_map.json immediately and re-baselines.
  6. Repeat until every ROM is named. Ctrl+C anytime — progress is saved.

Usage:
  python mapping_assist.py --port COM8
  python mapping_assist.py --port COM8 --thresh 1.5   # more sensitive
"""

import argparse
import json
import statistics
import time
from collections import defaultdict, deque
from pathlib import Path

MAP_PATH = Path(__file__).parent / "config" / "sensor_map.json"
BASELINE_FRAMES = 6      # frames used to (re)learn baselines
THRESH_DEFAULT = 2.0     # degC above baseline = "this one is being warmed"


def frames(port_name):
    import serial
    port = serial.Serial(port_name, 115200, timeout=5)
    while True:
        line = port.readline().decode(errors="ignore").strip()
        if line.startswith('{"seq"'):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def temps_of(frame):
    out = {}
    for bus in frame.get("buses", []):
        for s in bus.get("sensors", []):
            if s.get("ok"):
                out[s["rom"]] = s["t"]
    return out


def load_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    return {}


def save_map(m):
    MAP_PATH.parent.mkdir(exist_ok=True)
    MAP_PATH.write_text(json.dumps(m, indent=1))


def learn_baseline(src, n=BASELINE_FRAMES):
    hist = defaultdict(list)
    print(f"[baseline] learning over {n} frames — touch nothing...")
    for _ in range(n):
        for rom, t in temps_of(next(src)).items():
            hist[rom].append(t)
    return {rom: statistics.median(v) for rom, v in hist.items()}


def pos_label(entry):
    if entry.get("wall") == "AMB":
        return "AMB"
    return f"{entry.get('wall','?')} r{entry.get('row','?')} " \
           f"c{entry.get('col','?')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM8")
    ap.add_argument("--thresh", type=float, default=THRESH_DEFAULT)
    args = ap.parse_args()

    rom_map = load_map()
    src = frames(args.port)
    base = learn_baseline(src)

    total = len(base)
    print(f"\n{total} sensors on the bus. "
          f"{sum(r in rom_map for r in base)} already mapped.")
    print("Warm ONE probe at a time. Ctrl+C to stop (progress is saved).\n")

    try:
        while True:
            unmapped = [r for r in base if r not in rom_map]
            if not unmapped:
                print("\nAll sensors mapped! sensor_map.json is complete.")
                return
            print(f"--- {len(unmapped)} unmapped remain. Warm the next "
                  f"probe... (watching, thresh +{args.thresh}°C)")

            # watch until one ROM rises above its baseline
            hot_rom = None
            while hot_rom is None:
                now = temps_of(next(src))
                risers = {r: now[r] - base[r] for r in now
                          if now[r] - base[r] >= args.thresh}
                if risers:
                    hot_rom = max(risers, key=risers.get)
                    rise = risers[hot_rom]

            already = rom_map.get(hot_rom)
            tag = f" (currently {pos_label(already)})" if already else ""
            print(f"\n>>> SPIKE: {hot_rom}  +{rise:.1f}°C{tag}")
            ans = input("    position (e.g. 'N 1 3'), 'amb', or 'skip': ") \
                .strip().lower()

            if ans == "skip" or not ans:
                print("    skipped.")
            elif ans == "amb":
                rom_map[hot_rom] = {"wall": "AMB", "offset": 0.0}
                save_map(rom_map)
                print(f"    {hot_rom} -> AMBIENT  [saved]")
            else:
                try:
                    w, r, c = ans.split()
                    w = w.upper()
                    assert w in ("N", "E", "S", "W")
                    r, c = int(r), int(c)
                    rom_map[hot_rom] = {"wall": w, "row": r, "col": c,
                                        "offset": 0.0}
                    save_map(rom_map)
                    print(f"    {hot_rom} -> {w} r{r} c{c}  [saved]  "
                          f"— label the probe '{hot_rom[-4:]}'")
                except Exception:
                    print("    didn't parse — expected like: N 1 3. "
                          "Spike ignored; warm it again.")

            print("    let it cool a few seconds...")
            time.sleep(4)
            base = learn_baseline(src, n=3)   # quick re-baseline
    except KeyboardInterrupt:
        print(f"\nStopped. {sum(r in rom_map for r in base)}/{total} "
              f"mapped — saved in {MAP_PATH}")


if __name__ == "__main__":
    main()
