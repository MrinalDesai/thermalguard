"""ThermalGuard PANEL — PyQtGraph interactive 3D renderer for UNO Q.

Architecture:
    poll thread (HTTP) --> latest frame (shared)
    Qt render timer    --> 4 x 2D heatmaps + interactive OpenGL 3D box

No matplotlib, no plt.pause(), no per-frame ax3d.clear().
The 3D scene is created once and only the small 4x4 wall meshes are recolored.

Mouse in 3D view:
    Left drag   rotate
    Middle drag pan
    Wheel       zoom

Keys:
    R reset isolation
    V reset 3D camera
    F fullscreen
    Q / Esc quit

Install on UNO Q:
    sudo apt install -y python3-pyqt5 python3-pyqtgraph python3-opengl python3-opencv

Run:
    DISPLAY=:0 python3 thermalguard_panel_interactive3d.py --url http://10.0.0.1:8000/frame.json

Test:
    python3 thermalguard_panel_interactive3d.py --source sim --inject 25
"""

import argparse
import json
import threading
import time
from pathlib import Path
import numpy as np

MAP_PATH = Path(__file__).parent / "config" / "sensor_map.json"
WALLS = ["N", "E", "S", "W"]
ROWS = COLS = 4
VMIN, VMAX = 22.0, 45.0
STATES = ["NORMAL", "WATCH", "WARNING", "CRITICAL", "ISOLATED"]
THRESH = {"WATCH": 0.65, "WARNING": 0.80, "CRITICAL": 0.90}
STATE_CSS = {
    "NORMAL": "#1f7a3d",
    "WATCH": "#c99a00",
    "WARNING": "#d66a00",
    "CRITICAL": "#b3261e",
    "ISOLATED": "#6f2b8c",
}


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0


def poll_http(url, shared, period=1.0):
    import urllib.request
    last_seq = object()
    while True:
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                frame = json.loads(r.read().decode())
            seq = frame.get("seq")
            if seq != last_seq:
                last_seq = seq
                with shared.lock:
                    shared.frame = frame
                    shared.ts = time.time()
                print(f"[poll] seq={seq}")
        except Exception as exc:
            print(f"[poll] {type(exc).__name__}: {exc}")
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def sim_thread(shared, inject_at=25):
    from simulator import ThermalSim
    sim = ThermalSim()
    n = 0
    while True:
        n += 1
        if n == inject_at:
            sim.inject_runaway("N", 1, 3)
            print(f"[sim] runaway injected at frame {n}")
        frame = sim.step()
        with shared.lock:
            shared.frame = frame
            shared.ts = time.time()
        time.sleep(1.0)


def load_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    print(f"[warn] mapping file not found: {MAP_PATH}")
    return {}


def place(frame, rom_map, sim_mode):
    mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
    ambs = []

    if sim_mode:
        from thermal3d import frame_to_walls
        m6, amb = frame_to_walls(frame)
        for w in WALLS:
            mats[w] = np.asarray(m6[w], float)[:ROWS, :COLS]
        return mats, float(amb)

    for bus in frame.get("buses", []):
        for s in bus.get("sensors", []):
            if not s.get("ok"):
                continue
            e = rom_map.get(s["rom"])
            if not e:
                continue
            if e.get("wall") == "AMB":
                ambs.append(float(s["t"]))
            elif e.get("wall") in WALLS:
                r = int(e.get("row", 0))
                c = int(e.get("col", 0))
                if 0 <= r < ROWS and 0 <= c < COLS:
                    mats[e["wall"]][r, c] = float(s["t"]) + float(e.get("offset", 0.0))

    vals = [float(np.nanmean(m)) for m in mats.values() if np.any(~np.isnan(m))]
    amb = float(np.mean(ambs)) if ambs else (float(np.mean(vals)) if vals else float("nan"))
    return mats, amb


class Scorer:
    def __init__(self):
        try:
            from ml_train import AnomalyScorer
            self.s = AnomalyScorer()
        except Exception as exc:
            print(f"[warn] scorer unavailable: {exc}")
            self.s = None

    def __call__(self, mats, amb):
        if self.s is None:
            return 0.0
        try:
            return float(self.s.score(mats, amb))
        except Exception as exc:
            print(f"[score] {type(exc).__name__}: {exc}")
            return 0.0


class StateMachine:
    def __init__(self, watch_s=5.0, warn_s=10.0, crit_s=20.0):
        self.state = "NORMAL"
        self.timers = {"WATCH": watch_s, "WARNING": warn_s, "CRITICAL": crit_s}
        self.above = None
        self.relay = "CLOSED"

    def step(self, score, now):
        if self.state == "ISOLATED":
            return self.state
        if self.state == "CRITICAL":
            self.relay = "OPEN"
            print(">>> RELAY OPEN — LOAD ISOLATED <<<")
            self.state = "ISOLATED"
            return self.state

        nxt = {"NORMAL": "WATCH", "WATCH": "WARNING", "WARNING": "CRITICAL"}[self.state]
        if score >= THRESH[nxt]:
            if self.above is None:
                self.above = now
            elif now - self.above >= self.timers[nxt]:
                self.state = nxt
                self.above = None
                print(f"[state] -> {self.state}")
        else:
            self.above = None
            if self.state != "NORMAL" and score < THRESH["WATCH"]:
                self.state = STATES[STATES.index(self.state) - 1]
        return self.state

    def reset(self):
        self.state = "NORMAL"
        self.relay = "CLOSED"
        self.above = None
        print("[state] manual reset -> NORMAL")


def turbo_lut():
    try:
        import cv2
        ramp = np.arange(256, dtype=np.uint8).reshape(256, 1)
        bgr = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)[:, 0, :]
        return bgr[:, ::-1].copy()
    except Exception:
        anchors = np.array([
            [48, 18, 59], [50, 100, 200], [30, 190, 180],
            [230, 220, 60], [200, 40, 30]
        ], dtype=float)
        x = np.linspace(0, 1, 256)
        xp = np.linspace(0, 1, len(anchors))
        lut = np.empty((256, 3), dtype=np.uint8)
        for ch in range(3):
            lut[:, ch] = np.interp(x, xp, anchors[:, ch]).astype(np.uint8)
        return lut


LUT = turbo_lut()


def temp_rgba(value):
    if not np.isfinite(value):
        return np.array([0.35, 0.35, 0.35, 1.0], dtype=np.float32)
    idx = int(np.clip(round((float(value) - VMIN) / (VMAX - VMIN) * 255), 0, 255))
    rgb = LUT[idx].astype(np.float32) / 255.0
    return np.array([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)


def make_wall_geometry(wall):
    verts, faces = [], []

    def quad(a, b, c, d):
        base = len(verts)
        verts.extend([a, b, c, d])
        faces.extend([[base, base + 1, base + 2],
                      [base, base + 2, base + 3]])

    eps = 0.01
    for r in range(ROWS):
        z1 = ROWS - r
        z0 = z1 - 1
        for c in range(COLS):
            a0, a1 = c, c + 1
            if wall == "N":
                quad((a0, 4 + eps, z1), (a1, 4 + eps, z1),
                     (a1, 4 + eps, z0), (a0, 4 + eps, z0))
            elif wall == "S":
                quad((4 - a0, -eps, z1), (4 - a1, -eps, z1),
                     (4 - a1, -eps, z0), (4 - a0, -eps, z0))
            elif wall == "E":
                quad((4 + eps, 4 - a0, z1), (4 + eps, 4 - a1, z1),
                     (4 + eps, 4 - a1, z0), (4 + eps, 4 - a0, z0))
            elif wall == "W":
                quad((-eps, a0, z1), (-eps, a1, z1),
                     (-eps, a1, z0), (-eps, a0, z0))
    return np.asarray(verts, np.float32), np.asarray(faces, np.int32)


def wall_face_colors(m):
    colors = []
    for r in range(ROWS):
        for c in range(COLS):
            rgba = temp_rgba(m[r, c])
            colors.extend([rgba, rgba])
    return np.asarray(colors, np.float32)


def make_box_edges():
    c = np.array([
        [0,0,0], [4,0,0], [4,4,0], [0,4,0],
        [0,0,4], [4,0,4], [4,4,4], [0,4,4]
    ], np.float32)
    pairs = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
             (0,4),(1,5),(2,6),(3,7)]
    pts = []
    for a, b in pairs:
        pts.extend([c[a], c[b]])
    return np.asarray(pts, np.float32)


def main():
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    pg.setConfigOptions(imageAxisOrder="row-major")

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["http", "sim"], default="http")
    ap.add_argument("--url", default="http://10.0.0.1:8000/frame.json")
    ap.add_argument("--inject", type=int, default=25)
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("--refresh-ms", type=int, default=500)
    ap.add_argument("--timers", nargs=3, type=float, default=[5, 10, 20])
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    shared = Shared()
    if args.source == "sim":
        threading.Thread(target=sim_thread, args=(shared, args.inject), daemon=True).start()
    else:
        threading.Thread(target=poll_http, args=(args.url, shared, args.poll), daemon=True).start()

    rom_map = load_map()
    scorer = Scorer()
    sm = StateMachine(*args.timers)
    started = time.time()

    app = QtWidgets.QApplication([])
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("ThermalGuard — Interactive 3D")

    central = QtWidgets.QWidget()
    win.setCentralWidget(central)
    root = QtWidgets.QVBoxLayout(central)

    status = QtWidgets.QLabel("Waiting for sensor data...")
    status.setStyleSheet("QLabel { color:white; background:#222; padding:8px; font-size:18px; font-weight:bold; }")
    root.addWidget(status)

    split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
    root.addWidget(split, 1)

    heat = pg.GraphicsLayoutWidget()
    heat.setBackground((18,18,18))
    split.addWidget(heat)

    image_items, text_items = {}, {}
    for i, wall in enumerate(WALLS):
        r0, c0 = divmod(i, 2)
        p = heat.addPlot(row=r0, col=c0, title=f"Wall {wall}")
        p.setAspectLocked(True)
        p.hideAxis("left")
        p.hideAxis("bottom")
        p.setXRange(-0.5, 3.5, padding=0)
        p.setYRange(3.5, -0.5, padding=0)

        img = pg.ImageItem(np.full((ROWS, COLS), VMIN, dtype=float),
                           axisOrder="row-major")
        img.setLookupTable(LUT)
        img.setLevels([VMIN, VMAX])
        img.setRect(-0.5, -0.5, COLS, ROWS)   # centre pixels on labels
        p.addItem(img)
        image_items[wall] = img

        labels = []
        for rr in range(ROWS):
            row_labels = []
            for cc in range(COLS):
                t = pg.TextItem("", color=(0,0,0), anchor=(0.5,0.5))
                t.setPos(cc, rr)
                p.addItem(t)
                row_labels.append(t)
            labels.append(row_labels)
        text_items[wall] = labels

    view = gl.GLViewWidget()
    view.setBackgroundColor((18,18,18))
    view.opts["distance"] = 10
    view.opts["elevation"] = 24
    view.opts["azimuth"] = -45
    split.addWidget(view)
    split.setSizes([700, 580])

    grid = gl.GLGridItem()
    grid.setSize(6,6)
    grid.setSpacing(1,1)
    grid.translate(2,2,0)
    view.addItem(grid)

    blank = np.full((ROWS, COLS), np.nan)
    wall_geoms, wall_meshes = {}, {}
    for wall in WALLS:
        verts, faces = make_wall_geometry(wall)
        wall_geoms[wall] = (verts, faces)
        mesh = gl.GLMeshItem(
            vertexes=verts,
            faces=faces,
            faceColors=wall_face_colors(blank),
            smooth=False,
            shader="shaded",
            drawEdges=True,
            edgeColor=(0.15,0.15,0.15,1.0),
        )
        view.addItem(mesh)
        wall_meshes[wall] = mesh

    edges = gl.GLLinePlotItem(
        pos=make_box_edges(),
        color=(1,1,1,0.85),
        width=2,
        mode="lines",
        antialias=True,
    )
    view.addItem(edges)

    # translucent center battery for context
    bverts = np.array([
        [1,1,0],[3,1,0],[3,3,0],[1,3,0],
        [1,1,3],[3,1,3],[3,3,3],[1,3,3]
    ], np.float32)
    bfaces = np.array([
        [0,1,2],[0,2,3],[4,5,6],[4,6,7],
        [0,1,5],[0,5,4],[1,2,6],[1,6,5],
        [2,3,7],[2,7,6],[3,0,4],[3,4,7]
    ], np.int32)
    bcols = np.tile(np.array([[0.88,0.36,0.08,0.45]], np.float32), (len(bfaces),1))
    battery = gl.GLMeshItem(
        vertexes=bverts, faces=bfaces, faceColors=bcols,
        smooth=False, shader="shaded", drawEdges=True,
        edgeColor=(0.8,0.8,0.8,0.5)
    )
    view.addItem(battery)

    last_seq = object()
    last_state = None

    def reset_camera():
        view.opts["distance"] = 10
        view.opts["elevation"] = 24
        view.opts["azimuth"] = -45
        view.update()

    class Keys(QtCore.QObject):
        def eventFilter(self, obj, event):
            if event.type() == QtCore.QEvent.KeyPress:
                k = event.key()
                if k in (QtCore.Qt.Key_Q, QtCore.Qt.Key_Escape):
                    win.close()
                    return True
                if k == QtCore.Qt.Key_R:
                    sm.reset()
                    return True
                if k == QtCore.Qt.Key_V:
                    reset_camera()
                    return True
                if k == QtCore.Qt.Key_F:
                    win.showNormal() if win.isFullScreen() else win.showFullScreen()
                    return True
            return False

    keys = Keys()
    app.installEventFilter(keys)

    def update_ui():
        nonlocal last_seq, last_state
        with shared.lock:
            frame, ts = shared.frame, shared.ts

        if frame is None:
            mats = {w: np.full((ROWS,COLS), np.nan) for w in WALLS}
            amb, score, live_n, age, seq = float("nan"), 0.0, 0, 999.0, None
        else:
            mats, amb = place(frame, rom_map, args.source == "sim")
            score = scorer(mats, amb)
            live_n = sum(int(np.sum(~np.isnan(mats[w]))) for w in WALLS)
            age = time.time() - ts
            seq = frame.get("seq")

        sm.step(score, time.time() - started)

        if seq != last_seq:
            last_seq = seq
            for wall in WALLS:
                m = np.asarray(mats[wall], float)

                # Keep missing cells visually grey in 3D; 2D uses VMIN for the image
                # but suppresses the numeric label.
                image_items[wall].setImage(
                    np.nan_to_num(m, nan=VMIN),
                    autoLevels=False,
                    levels=(VMIN, VMAX),
                )
                for rr in range(ROWS):
                    for cc in range(COLS):
                        v = m[rr,cc]
                        text_items[wall][rr][cc].setText(
                            "" if not np.isfinite(v) else f"{v:.1f}"
                        )

                verts, faces = wall_geoms[wall]
                wall_meshes[wall].setMeshData(
                    vertexes=verts,
                    faces=faces,
                    faceColors=wall_face_colors(m),
                    smooth=False,
                )

        hot_t = float("nan")
        hot_at = "-"
        for wall in WALLS:
            m = mats[wall]
            if np.any(~np.isnan(m)):
                local = float(np.nanmax(m))
                if not np.isfinite(hot_t) or local > hot_t:
                    hot_t = local
                    rr, cc = np.unravel_index(np.nanargmax(m), m.shape)
                    hot_at = f"{wall} r{rr} c{cc}"

        data_txt = f"STALE {age:.0f}s" if age >= 5 else "LIVE"
        if np.isfinite(hot_t) and np.isfinite(amb):
            detail = f"max {hot_t:.1f}C @ {hot_at}   dAmb {hot_t-amb:+.1f}C"
        else:
            detail = "max --   dAmb --"

        status.setText(
            f"{sm.state}   score {score:.2f}   {detail}   "
            f"relay {sm.relay}   live {live_n}   data {data_txt}   seq {seq}"
        )

        if sm.state != last_state:
            last_state = sm.state
            status.setStyleSheet(
                "QLabel { color:white; "
                f"background:{STATE_CSS[sm.state]}; "
                "padding:8px; font-size:18px; font-weight:bold; }"
            )

    timer = QtCore.QTimer()
    timer.timeout.connect(update_ui)
    timer.start(max(100, args.refresh_ms))

    win.resize(1280, 720)
    win.showFullScreen() if args.fullscreen else win.show()
    app.exec_()


if __name__ == "__main__":
    main()
