# ThermalGuard Edge

**Edge-AI spatial thermal monitoring on Arduino UNO Q.** A 96-sensor
DS18B20 grid (4 walls × 24) reconstructs live thermal images of an
enclosure, scores them with an anomaly model trained on normal operation
only, and drives a preventive isolation output — no cloud, no PC in the
loop.

> **Status:** bench rig validated on real hardware (multi-drop 1-Wire bus,
> live heatmap, hotspot localisation). Full 96-sensor build in progress.
> Fault scenarios are physics-simulated and injected upstream of the live
> pipeline; the hardware path is validated on real sensors.

---

## Setup

```bash
# Windows (PowerShell), inside the project folder
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy scipy matplotlib pyserial scikit-learn joblib plotly
```

Arduino side: IDE 2.x, install **DallasTemperature** library (pulls in
OneWire). Upload `sketch/bench_uno.ino` for the bench rig (classic Uno) or
`sketch/sketch.ino` via App Lab for the UNO Q.

---

## Commands

### Live heatmap (2D, four wall panels)
```bash
python heatmap_live.py --source sim                 # simulator, scripted hotspot
python heatmap_live.py --source serial --port COM6  # real sensors via bench Uno
```
Fixed colour scale 20–40 °C. Unknown ROMs auto-map to free grid slots and
print an `[auto-map]` line — copy those into `config/sensor_map.json`.

### 3D thermal box (matplotlib)
```bash
python thermal3d.py                                  # save PNG, single hotspot
python thermal3d.py --live                           # interactive window
python thermal3d.py --live --big --cmap turbo        # large spreading hotspot
python thermal3d.py --hotspot E --row 2 --col 4      # place the fault
```

### Train the anomaly model
```bash
python ml_train.py                # 4800 synthetic frames, train, evaluate
python ml_train.py --episodes 40  # larger dataset
```
Trains an Isolation Forest on NORMAL episodes only, evaluates on held-out
hotspot / runaway / cluster faults, saves `model.joblib`.
Reference numbers (seeded): normal mean score 0.125, detection 91.8 % at
1.0 % false alarms (WATCH threshold 0.65).

### Visualise the training dataset
```bash
python dataset_viewer.py --cmap turbo        # 2D grid: 4 classes × 4 walls
python dataset_viewer3d.py --cmap turbo      # same, as four 3D boxes (PNG)
python dataset_viewer3d_html.py              # fully rotatable, opens browser
python dataset_viewer.py --frame 60          # early-fault (subtle) frames
```
All annotate each class with the trained model's anomaly score for the
shown frame. `dataset_viewer3d_html.py` writes `dataset_cases_3d.html` —
drag any box to rotate, hover for temperatures.

### Screen the probe batch (shipment QC)
```bash
python screening.py --port COM6 --batch B01
python screening.py --port COM6 --batch B02 --ref 28CDFF5C40240B23
```
Plug up to ~10 probes into the bench bus, tips in still air. 2 min warmup,
5 min soak, verdict per ROM (PASS / MARGINAL / REJECT). Survivors are
written to `config/sensor_map.json` with calibration offsets; rejects to
`logs/screening_rejects.txt`. **Label kept probes with their last-4 ROM
digits before unplugging.**

### Pipeline logger (baseline collection)
```bash
python main.py --source sim          # simulator frames -> SQLite
python main.py --source hardware     # real frames -> SQLite (edit port in
                                     # source_hardware() for serial fallback)
```
Logs every frame to `thermalguard.db`. Leave running overnight on real
sensors to collect the normal-operation baseline the model retrains on.

---

## Typical workflows

**Bench rig bring-up (no shipment needed):**
1. Wire 4+ sensors: all VDD→5V, all GND→GND, all DATA→D2, one 4.7 kΩ
   DATA→5V. Upload `bench_uno.ino`, Serial Monitor 115200 → `{"census":N}`.
2. Close Serial Monitor, run the live heatmap against the COM port.
3. Pinch a sensor → its cell ignites. That's the whole chain working.

**Wall commissioning (per wall):**
1. Wire the wall's probes into its terminal blocks (tinned ends).
2. Live heatmap up → all probes visible at room temperature *before taping*.
3. Warm each tip in sequence → identify ROM → record wall/row/col in
   `sensor_map.json`. Outliers (±2 °C from neighbours, stuck 85.0) swap for
   a spare on the spot.
4. Tape tips to blocks — identical method at every position.

**Retrain on real data:** append baseline frames from `thermalguard.db` to
the normal training set in `ml_train.py` and rerun. Copy `model.joblib` to
the UNO Q with the rest of `python/`.

---

## Repository layout

```
sketch/sketch.ino          UNO Q firmware: 9-bus acquisition, Bridge output
sketch/bench_uno.ino       Classic Uno bench rig: 1 bus, Serial output
python/main.py             Frame listener -> mapping -> SQLite
python/simulator.py        Heat-diffusion physics, fault injection (4×6 walls)
python/heatmap_live.py     Live 2D four-wall heatmap (serial or sim)
python/thermal3d.py        3D thermal enclosure render
python/ml_train.py         Dataset gen -> features -> Isolation Forest
python/screening.py        Probe batch QC -> offsets -> sensor_map.json
python/dataset_viewer*.py  Training-class visualisers (2D / 3D / HTML)
config/sensor_map.json     ROM -> wall/row/col/offset (the identity map)
docs/PROJECT_STATUS.md     Full build status, wiring design, plan
```

## Architecture (one paragraph)

The STM32 side of the UNO Q owns acquisition and deterministic safety:
nine 1-Wire buses, broadcast conversion, hard temperature trips that no
software layer can suppress. The Linux side owns everything above:
per-wall matrices, SQLite history, interpolated thermal rendering, an
Isolation Forest trained on normal operation only, and a persistence state
machine (NORMAL → WATCH → WARNING → CRITICAL → ISOLATED) that requires
sustained anomaly before escalating and manual acknowledgement after
isolation. The system stores measurements, not visualisations — imagery is
a view, not a record.
