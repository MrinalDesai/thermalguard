"""ThermalGuard heat-diffusion simulator.

Emits frames byte-compatible with the MCU firmware, so the entire Linux
pipeline (mapping, SQLite, heatmap, ML, state machine) runs identically on
--source sim and --source hardware. This is the disclosed simulation layer:
fault scenarios are physics-generated and injected UPSTREAM of the live
pipeline.

Scenarios (call inject_* at runtime, or script them):
  - baseline drift: slow ambient wander + per-sensor noise
  - hotspot: Gaussian source on one wall, grows and diffuses to neighbours
  - runaway: exponential-rate hotspot (the case you'd never build physically)
  - dead sensor: a ROM goes silent (ok=false)
"""

import numpy as np

WALLS = ["N", "E", "S", "W"]
ROWS, COLS = 4, 6

# Same bus layout as the firmware: 2 buses/wall, 10 sensors each -> 4x5.
# Synthetic ROMs are deterministic so sensor_map.json can be pre-generated.
def synth_rom(wall_i, idx):
    return f"28SIM{wall_i:02d}{idx:02d}00000000"[:16]


class ThermalSim:
    def __init__(self, ambient=27.5, noise=0.08, seed=42):
        self.rng = np.random.default_rng(seed)
        self.ambient = ambient
        self.noise = noise
        self.walls = {w: np.full((ROWS, COLS), ambient) for w in WALLS}
        self.seq = 0
        self.hotspots = []   # dicts: wall,row,col,power,growth
        self.dead = set()    # ROMs reporting invalid

    # ------------------------------------------------ fault injection
    def inject_hotspot(self, wall="N", row=1, col=2, power=0.15, growth=1.0):
        """power: degC added per step at the core. growth=1 steady,
        >1 escalating (thermal-runaway-like)."""
        self.hotspots.append(dict(wall=wall, row=row, col=col,
                                  power=power, growth=growth))

    def inject_runaway(self, wall="E", row=2, col=3):
        self.inject_hotspot(wall, row, col, power=0.05, growth=1.06)

    def kill_sensor(self, wall_i=0, idx=7):
        self.dead.add(synth_rom(wall_i, idx))

    def clear_faults(self):
        self.hotspots.clear()
        self.dead.clear()

    # ------------------------------------------------ physics step
    def step(self):
        self.ambient += self.rng.normal(0, 0.01)  # slow drift

        for w in WALLS:
            m = self.walls[w]
            # relax toward ambient
            m += (self.ambient - m) * 0.05
            # diffuse to neighbours (simple 4-neighbour kernel)
            pad = np.pad(m, 1, mode="edge")
            lap = (pad[:-2, 1:-1] + pad[2:, 1:-1] +
                   pad[1:-1, :-2] + pad[1:-1, 2:] - 4 * m)
            m += 0.12 * lap

        for h in self.hotspots:
            m = self.walls[h["wall"]]
            m[h["row"], h["col"]] += h["power"]
            h["power"] *= h["growth"]

        return self._frame()

    # ------------------------------------------------ firmware-format frame
    def _frame(self):
        buses = []
        for wall_i, w in enumerate(WALLS):
            m = self.walls[w]
            flat = m.flatten()
            for half in range(2):                     # 2 buses per wall
                sensors = []
                for k in range(12):
                    idx = half * 12 + k
                    rom = synth_rom(wall_i, idx)
                    dead = rom in self.dead
                    t = 85.0 if dead else float(
                        flat[idx] + self.rng.normal(0, self.noise))
                    sensors.append({"rom": rom, "t": round(t, 2),
                                    "ok": not dead})
                buses.append({"bus": wall_i * 2 + half, "enabled": True,
                              "sensors": sensors})
        # ambient bus (bus 8): two free-air probes
        amb = [{"rom": synth_rom(9, i),
                "t": round(float(self.ambient + self.rng.normal(0, self.noise)), 2),
                "ok": True} for i in range(2)]
        buses.append({"bus": 8, "enabled": True, "sensors": amb})

        self.seq += 1
        return {"seq": self.seq, "ms": self.seq * 2000, "buses": buses}


def write_sim_sensor_map(path):
    """Generate config/sensor_map.json for the simulator's synthetic ROMs."""
    import json
    m = {}
    for wall_i, w in enumerate(WALLS):
        for idx in range(24):
            m[synth_rom(wall_i, idx)] = {
                "wall": w, "row": idx // COLS, "col": idx % COLS, "offset": 0.0}
    for i in range(2):
        m[synth_rom(9, i)] = {"wall": "AMB", "offset": 0.0}
    path.write_text(json.dumps(m, indent=1))


if __name__ == "__main__":
    # quick self-test: baseline, then a hotspot appears on wall N
    from pathlib import Path
    write_sim_sensor_map(Path(__file__).parent.parent / "config" / "sensor_map.json")
    sim = ThermalSim()
    for i in range(30):
        if i == 10:
            sim.inject_hotspot("N", 1, 2, power=0.4)
        f = sim.step()
    n = np.array([[s["t"] for s in f["buses"][0]["sensors"]][:5]])
    print("wall N bus0 after hotspot:", n.round(1))
