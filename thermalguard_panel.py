"""ThermalGuard PANEL — OpenCV renderer for the UNO Q.

No matplotlib. No plt.pause(). No 3D teardown. Architecture:

    poll thread (http) --> latest_frame (shared)
    render loop        --> NumPy canvas --> cv2.imshow

The poll thread is fully independent of display: a stalled repaint can
never starve data; a missed poll costs one second, not the stream.

Layout (landscape 1280x400, --portrait for 400x1280):
  left: four 4x4 wall heatmaps (turbo, cell values)
  right: status column + pseudo-3D box (walls as colored quads)
  background: state colour (NORMAL green ... ISOLATED purple)

Keys: R = reset isolation   F = fullscreen toggle   Q/Esc = quit

Usage (UNO Q):
  sudo apt install -y python3-opencv
  DISPLAY=:0 python3 thermalguard_panel.py --url http://10.0.0.1:8000/frame.json
Test without hardware:
  python thermalguard_panel.py --source sim --save out.png
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
STATE_BGR = {"NORMAL": (89, 170, 31), "WATCH": (0, 180, 232),
             "WARNING": (46, 119, 244), "CRITICAL": (47, 52, 209),
             "ISOLATED": (120, 31, 122)}
THRESH = {"WATCH": 0.65, "WARNING": 0.80, "CRITICAL": 0.90}


class Shared:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.ts = 0.0


def poll_http(url, shared, period=1.0):
    import urllib.request
    last_seq = -1
    while True:
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                f = json.loads(r.read().decode())
            if f.get("seq", -1) != last_seq:
                last_seq = f.get("seq", -1)
                with shared.lock:
                    shared.frame = f
                    shared.ts = time.time()
        except Exception:
            pass
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
        f = sim.step()
        with shared.lock:
            shared.frame = f
            shared.ts = time.time()
        time.sleep(1.0)


def load_map():
    if MAP_PATH.exists():
        return json.loads(MAP_PATH.read_text())
    return {}


def place(frame, rom_map, sim_mode):
    mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
    ambs = []
    if sim_mode:
        from thermal3d import frame_to_walls
        m6, amb = frame_to_walls(frame)
        for w in WALLS:
            mats[w] = np.asarray(m6[w], float)[:, :4]
        return mats, amb
    for bus in frame.get("buses", []):
        for s in bus.get("sensors", []):
            if not s.get("ok"):
                continue
            e = rom_map.get(s["rom"])
            if not e:
                continue
            if e.get("wall") == "AMB":
                ambs.append(s["t"])
            elif e.get("wall") in WALLS:
                r, c = e.get("row", 0), e.get("col", 0)
                if 0 <= r < ROWS and 0 <= c < COLS:
                    mats[e["wall"]][r, c] = s["t"] + e.get("offset", 0.0)
    vals = [np.nanmean(m) for m in mats.values() if np.any(~np.isnan(m))]
    amb = float(np.mean(ambs)) if ambs else \
        (float(np.nanmean(vals)) if vals else float("nan"))
    return mats, amb


class Scorer:
    def __init__(self):
        try:
            from ml_train import AnomalyScorer
            self.s = AnomalyScorer()
        except Exception:
            self.s = None

    def __call__(self, mats, amb):
        if self.s is None:
            return 0.0
        try:
            return float(self.s.score(mats, amb))
        except Exception:
            return 0.0


class StateMachine:
    def __init__(self, watch_s=5.0, warn_s=10.0, crit_s=20.0):
        self.state = "NORMAL"
        self.timers = {"WATCH": watch_s, "WARNING": warn_s,
                       "CRITICAL": crit_s}
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
        nxt = {"NORMAL": "WATCH", "WATCH": "WARNING",
               "WARNING": "CRITICAL"}[self.state]
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


def turbo(vals):
    import cv2
    x = np.clip((np.nan_to_num(vals, nan=VMIN) - VMIN)
                / (VMAX - VMIN), 0, 1)
    img = cv2.applyColorMap((x * 255).astype(np.uint8),
                            cv2.COLORMAP_TURBO)
    img[np.isnan(vals)] = (102, 102, 102)
    return img


def draw_wall(canvas, x, y, size, name, m, font=0.42):
    import cv2
    cell = size // COLS
    img = turbo(m)
    up = cv2.resize(img, (cell * COLS, cell * ROWS),
                    interpolation=cv2.INTER_NEAREST)
    canvas[y:y + cell * ROWS, x:x + cell * COLS] = up
    for r in range(ROWS):
        for c in range(COLS):
            cv2.rectangle(canvas, (x + c * cell, y + r * cell),
                          (x + (c + 1) * cell, y + (r + 1) * cell),
                          (0, 0, 0), 1)
            v = m[r, c]
            if not np.isnan(v):
                cv2.putText(canvas, f"{v:.1f}",
                            (x + c * cell + 3,
                             y + r * cell + cell // 2 + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, font,
                            (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Wall {name}", (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA)


def wall_bgr(m):
    v = np.nanmean(m)
    if np.isnan(v):
        return (120, 120, 120)
    img = turbo(np.array([[v]]))
    return tuple(int(c) for c in img[0, 0])


def draw_box(canvas, x, y, w, h, mats):
    """Isometric box: three visible faces textured with their wall's
    actual 4x4 cell colours (warped heatmaps), hotspots visible on
    the box itself."""
    import cv2

    def face_img(m, upscale=60):
        img = turbo(m)
        img = cv2.resize(img, (COLS * upscale, ROWS * upscale),
                         interpolation=cv2.INTER_LINEAR)
        for i in range(ROWS + 1):
            cv2.line(img, (0, i * upscale), (img.shape[1], i * upscale),
                     (30, 30, 30), 2)
        for j in range(COLS + 1):
            cv2.line(img, (j * upscale, 0), (j * upscale, img.shape[0]),
                     (30, 30, 30), 2)
        return img

    def warp(img, quad):
        srcp = np.float32([[0, 0], [img.shape[1], 0],
                           [img.shape[1], img.shape[0]],
                           [0, img.shape[0]]])
        M = cv2.getPerspectiveTransform(srcp, np.float32(quad))
        wimg = cv2.warpPerspective(img, M,
                                   (canvas.shape[1], canvas.shape[0]))
        mask = cv2.warpPerspective(
            np.full(img.shape[:2], 255, np.uint8), M,
            (canvas.shape[1], canvas.shape[0]))
        canvas[mask > 0] = wimg[mask > 0]

    d = int(w * 0.32)
    fw, fh = w - d, h - d
    # quads: TL, TR, BR, BL order
    top = [[x + d, y], [x + d + fw, y], [x + fw, y + d], [x, y + d]]
    side = [[x + fw + d, y], [x + fw + d, y + fh],
            [x + fw, y + d + fh], [x + fw, y + d]]
    front = [[x, y + d], [x + fw, y + d],
             [x + fw, y + d + fh], [x, y + d + fh]]

    warp(face_img(mats["N"]), top)
    s_img = cv2.convertScaleAbs(face_img(mats["E"]), alpha=0.78)
    warp(s_img, [side[3], side[0], side[1], side[2]])
    warp(face_img(mats["S"]), front)

    for quad in (top, side, front):
        cv2.polylines(canvas, [np.array(quad, np.int32)], True,
                      (230, 230, 230), 2)
    for wall, pos in (("N", (x + d + fw // 2 - 10, y + d // 2 + 4)),
                      ("E", (x + fw + d // 2 - 8, y + fh // 2 + d // 2)),
                      ("S", (x + fw // 2 - 10, y + d + fh // 2))):
        cv2.putText(canvas, wall, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "W: rear", (x, y + d + fh + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
                cv2.LINE_AA)


def render(mats, amb, score, sm, live_n, age, W, H, portrait):
    import cv2
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = STATE_BGR[sm.state]
    cv2.rectangle(canvas, (8, 8), (W - 8, H - 8), (18, 18, 18), -1)

    vals = [float(np.nanmax(m)) for m in mats.values()
            if np.any(~np.isnan(m))]
    hot = max(vals) if vals else float("nan")
    if portrait:
        size = min(W - 40, (H - 260) // 4 - 30)
        for i, w in enumerate(WALLS):
            draw_wall(canvas, 20, 44 + i * (size + 34), size, w, mats[w])
        sx, sy = 20, H - 200
    else:
        size = min((W - 480) // 4 - 14, H - 100)
        for i, w in enumerate(WALLS):
            draw_wall(canvas, 20 + i * (size + 14), 46, size, w, mats[w])
        draw_box(canvas, W - 290, 175, 200, 175, mats)
        sx, sy = W - 430, 46

    lines = [(sm.state, 0.95, STATE_BGR[sm.state])]
    if hot == hot:
        lines += [(f"score {score:.2f}  max {hot:.1f}C", 0.55,
                   (255, 255, 255)),
                  (f"amb {amb:.1f}C  dAmb {hot - amb:+.1f}C", 0.55,
                   (255, 255, 255))]
    else:
        lines += [(f"score {score:.2f}   no data", 0.55,
                   (200, 200, 200))]
    lines += [(f"relay {sm.relay}  live {live_n}", 0.5,
               (255, 255, 255)),
              (("data LIVE" if age < 5 else f"data STALE {age:.0f}s"),
               0.5, (160, 255, 160) if age < 5 else (0, 0, 255))]
    if sm.state == "ISOLATED":
        lines.append(("[R] reset", 0.5, (255, 255, 255)))
    yy = sy
    for txt, sc, col in lines:
        cv2.putText(canvas, txt, (sx, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    sc, col, 2 if sc > 0.8 else 1, cv2.LINE_AA)
        yy += int(40 * sc + 14)
    return canvas


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["http", "sim"], default="http")
    ap.add_argument("--url", default="http://10.0.0.1:8000/frame.json")
    ap.add_argument("--inject", type=int, default=25)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=400)
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--timers", nargs=3, type=float,
                    default=[5, 10, 20])
    ap.add_argument("--save", default=None)
    ap.add_argument("--save-frames", type=int, default=40)
    args = ap.parse_args()
    if args.portrait:
        args.w, args.h = min(args.w, args.h), max(args.w, args.h)

    shared = Shared()
    if args.source == "sim":
        threading.Thread(target=sim_thread, args=(shared, args.inject),
                         daemon=True).start()
    else:
        threading.Thread(target=poll_http, args=(args.url, shared),
                         daemon=True).start()

    rom_map = load_map()
    scorer = Scorer()
    sm = StateMachine(*args.timers)

    if not args.save:
        cv2.namedWindow("ThermalGuard", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("ThermalGuard", cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    n = 0
    t0 = time.time()
    while True:
        with shared.lock:
            frame, ts = shared.frame, shared.ts
        if frame is None:
            mats = {w: np.full((ROWS, COLS), np.nan) for w in WALLS}
            amb, score, live_n, age = float("nan"), 0.0, 0, 999
        else:
            mats, amb = place(frame, rom_map, args.source == "sim")
            score = scorer(mats, amb)
            live_n = sum(int(np.sum(~np.isnan(mats[w])))
                         for w in WALLS)
            age = time.time() - ts
        sm.step(score, time.time() - t0)
        canvas = render(mats, amb, score, sm, live_n, age,
                        args.w, args.h, args.portrait)
        n += 1
        if args.save:
            if args.source == "sim":
                time.sleep(1.0 if n < 5 else 0.2)
            if n >= args.save_frames:
                cv2.imwrite(args.save, canvas)
                print(f"saved {args.save}  state={sm.state}")
                return
            continue
        cv2.imshow("ThermalGuard", canvas)
        k = cv2.waitKey(500) & 0xFF
        if k in (ord("q"), 27):
            break
        if k in (ord("r"), ord("R")):
            sm.reset()
        if k in (ord("f"), ord("F")):
            fs = cv2.getWindowProperty("ThermalGuard",
                                       cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                "ThermalGuard", cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL if fs == cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_FULLSCREEN)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
