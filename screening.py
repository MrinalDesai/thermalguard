"""ThermalGuard probe screening — shipment-day batch tester.

Plug a batch of probes (up to ~10) into the bench bus, run this, get a
verdict per ROM. Survivors' calibration offsets are written into
config/sensor_map.json (wall/row/col left as "UNASSIGNED" for later
mapping). Rejects go to logs/screening_rejects.txt so you physically bin
them with confidence.

Reference strategy:
  --ref median   (default) judge against the batch's stabilized median.
                 Catches stuck-85s, CRC dropouts, drift, and outliers.
  --ref ROMHEX[,ROMHEX]  judge against specific trusted sensor(s) present
                 on the bus (e.g. genuine DS18B20+ or your proven bench
                 units). Absolute accuracy instead of batch consistency.

Usage (bench Uno streaming on COM6):
  python screening.py --port COM6 --batch B01
  python screening.py --port COM6 --batch B02 --soak 300
  python screening.py --port COM6 --batch B03 --ref 28CDFF5C40240B23

Protocol per batch (~12 min):
  1. Plug probes in, tips together in still air (bundle them loosely)
  2. Run the command; it waits WARMUP then records SOAK seconds
  3. Read the verdict table; unplug batch; next batch
Rules of thumb: PASS |offset| <= 0.5 degC and stable; MARGINAL <= 1.0
(usable in non-critical positions); REJECT beyond that or any dropouts.
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

MAP_PATH = Path(__file__).parent.parent / "config" / "sensor_map.json"
LOG_DIR = Path(__file__).parent.parent / "logs"

PASS_LIMIT = 0.5      # degC vs reference
MARGINAL_LIMIT = 1.0
STABILITY_LIMIT = 0.25  # max stddev of a sensor's own readings
MIN_VALID_FRACTION = 0.95


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


def collect(port_name, seconds, label):
    """Gather per-ROM reading lists for `seconds`."""
    data = defaultdict(list)
    invalid = defaultdict(int)
    t_end = time.time() + seconds
    n = 0
    for f in frames(port_name):
        n += 1
        for bus in f["buses"]:
            for s in bus["sensors"]:
                if s["ok"]:
                    data[s["rom"]].append(s["t"])
                else:
                    invalid[s["rom"]] += 1
        remaining = int(t_end - time.time())
        print(f"\r[{label}] frame {n}  sensors {len(data)}  "
              f"{remaining:4d}s left ", end="", flush=True)
        if time.time() >= t_end:
            print()
            return data, invalid, n


def judge(data, invalid, total_frames, ref_roms):
    """Return list of dicts: rom, mean, offset, std, valid_frac, verdict."""
    means = {rom: statistics.fmean(v) for rom, v in data.items() if v}
    if ref_roms:
        present = [r for r in ref_roms if r in means]
        if not present:
            raise SystemExit(f"reference ROM(s) not on bus: {ref_roms}")
        ref = statistics.fmean([means[r] for r in present])
        ref_desc = f"reference {'+'.join(present)} = {ref:.2f}C"
    else:
        ref = statistics.median(means.values())
        ref_desc = f"batch median = {ref:.2f}C"

    rows = []
    for rom, vals in sorted(data.items()):
        mean = means[rom]
        off = ref - mean                     # ADD this to sensor reading
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        vf = len(vals) / max(1, len(vals) + invalid.get(rom, 0))
        is_ref = ref_roms and rom in ref_roms
        if is_ref:
            verdict = "REF"
        elif vf < MIN_VALID_FRACTION or std > STABILITY_LIMIT:
            verdict = "REJECT"
        elif abs(off) <= PASS_LIMIT:
            verdict = "PASS"
        elif abs(off) <= MARGINAL_LIMIT:
            verdict = "MARGINAL"
        else:
            verdict = "REJECT"
        rows.append(dict(rom=rom, mean=mean, offset=off, std=std,
                         valid=vf, verdict=verdict))
    return rows, ref_desc


def write_results(rows, batch):
    rom_map = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else {}
    LOG_DIR.mkdir(exist_ok=True)
    rejects = []
    for r in rows:
        if r["verdict"] in ("PASS", "MARGINAL"):
            entry = rom_map.get(r["rom"], {})
            entry.update({
                "wall": entry.get("wall", "UNASSIGNED"),
                "offset": round(r["offset"], 3),
                "batch": batch,
                "screen": r["verdict"],
            })
            rom_map[r["rom"]] = entry
        elif r["verdict"] == "REJECT":
            rejects.append(f"{batch} {r['rom']} off={r['offset']:+.2f} "
                           f"std={r['std']:.3f} valid={r['valid']:.0%}")
    MAP_PATH.parent.mkdir(exist_ok=True)
    MAP_PATH.write_text(json.dumps(rom_map, indent=1))
    if rejects:
        with open(LOG_DIR / "screening_rejects.txt", "a") as f:
            f.write("\n".join(rejects) + "\n")
    return sum(r["verdict"] in ("PASS", "MARGINAL") for r in rows), len(rejects)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--batch", required=True, help="label, e.g. B01")
    ap.add_argument("--warmup", type=int, default=120,
                    help="stabilisation seconds before recording")
    ap.add_argument("--soak", type=int, default=300,
                    help="recording seconds")
    ap.add_argument("--ref", default="median",
                    help="'median' or comma-separated trusted ROM hex")
    args = ap.parse_args()

    ref_roms = None if args.ref == "median" else args.ref.upper().split(",")

    print(f"Batch {args.batch}: warming up {args.warmup}s "
          f"(sensors settling to air temperature)...")
    collect(args.port, args.warmup, "warmup")          # discarded
    print(f"Recording {args.soak}s...")
    data, invalid, nframes = collect(args.port, args.soak, "soak")

    rows, ref_desc = judge(data, invalid, nframes, ref_roms)
    print(f"\nBatch {args.batch} vs {ref_desc}")
    print(f"{'ROM':<18}{'mean C':>8}{'offset':>8}{'std':>7}"
          f"{'valid':>7}  verdict")
    for r in rows:
        print(f"{r['rom']:<18}{r['mean']:>8.2f}{r['offset']:>+8.2f}"
              f"{r['std']:>7.3f}{r['valid']:>7.0%}  {r['verdict']}")

    kept, binned = write_results(rows, args.batch)
    print(f"\n{kept} written to sensor_map.json (wall=UNASSIGNED), "
          f"{binned} logged to screening_rejects.txt")
    print("Label kept probes physically with their last-4 ROM digits "
          "before unplugging!")


if __name__ == "__main__":
    main()
