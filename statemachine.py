"""ThermalGuard state machine — the decision layer, demo-ready.

Consumes frames (simulator tonight, serial later), scores them with the
trained model, and walks the persistence state machine:

  NORMAL -> (score>=0.65 for WATCH_S)   -> WATCH
  WATCH  -> (score>=0.80 for WARN_S)    -> WARNING
  WARNING-> (score>=0.90 for CRIT_S)    -> CRITICAL -> relay opens -> ISOLATED
  ISOLATED -> stays latched until manual reset (press R in banner / Ctrl+C)

Demo profile timers default (5/10/20s). Production: --timers 20 45 90.

Relay: tries Linux GPIO (gpiod) on --gpiochip/--line if given; otherwise
prints RELAY events to console (logic fully demonstrable without hardware).

UI: fullscreen colored banner (tkinter) on the panel showing state, score,
hottest cell. --headless for console only.

Usage (on the UNO Q):
  DISPLAY=:0 python3 statemachine.py --inject 20
  DISPLAY=:0 python3 statemachine.py --inject 20 --gpiochip gpiochip0 --line 42
"""

import argparse
import time

import numpy as np

from simulator import ThermalSim, WALLS, ROWS, COLS
from thermal3d import frame_to_walls
from ml_train import AnomalyScorer

STATES = ["NORMAL", "WATCH", "WARNING", "CRITICAL", "ISOLATED"]
COLORS = {"NORMAL": "#1faa59", "WATCH": "#e8b400", "WARNING": "#f4772e",
          "CRITICAL": "#d1342f", "ISOLATED": "#7a1f78"}
THRESH = {"WATCH": 0.65, "WARNING": 0.80, "CRITICAL": 0.90}


class Relay:
    """Isolation output: gpiod if configured & available, else console."""

    def __init__(self, chip=None, line=None):
        self.mode = "console"
        self.state = "CLOSED"       # load powered
        if chip is not None and line is not None:
            try:
                import gpiod
                self.chip = gpiod.Chip(chip)
                self.req = self.chip.get_line(int(line))
                self.req.request(consumer="thermalguard",
                                 type=gpiod.LINE_REQ_DIR_OUT, default_val=0)
                self.mode = f"gpiod:{chip}/{line}"
            except Exception as e:
                print(f"[relay] gpiod unavailable ({e}); console mode")

    def isolate(self):
        self.state = "OPEN"
        if self.mode.startswith("gpiod"):
            self.req.set_value(1)
        print(">>> RELAY OPEN — LOAD ISOLATED <<<")

    def restore(self):
        self.state = "CLOSED"
        if self.mode.startswith("gpiod"):
            self.req.set_value(0)
        print(">>> RELAY CLOSED — load restored <<<")


class StateMachine:
    def __init__(self, watch_s, warn_s, crit_s, relay):
        self.state = "NORMAL"
        self.timers = {"WATCH": watch_s, "WARNING": warn_s,
                       "CRITICAL": crit_s}
        self.above_since = None
        self.relay = relay

    def step(self, score, now):
        if self.state == "ISOLATED":
            return self.state
        nxt = {"NORMAL": "WATCH", "WATCH": "WARNING",
               "WARNING": "CRITICAL"}[self.state] \
            if self.state != "CRITICAL" else None

        if self.state == "CRITICAL":
            self.relay.isolate()
            self.state = "ISOLATED"
            return self.state

        thr = THRESH[nxt]
        if score >= thr:
            if self.above_since is None:
                self.above_since = now
            elif now - self.above_since >= self.timers[nxt]:
                self.state = nxt
                self.above_since = None
                print(f"[state] -> {self.state}")
        else:
            self.above_since = None
            # de-escalate one level when calm
            if self.state != "NORMAL" and score < THRESH["WATCH"]:
                self.state = STATES[STATES.index(self.state) - 1]
                print(f"[state] <- {self.state}")
        return self.state

    def reset(self):
        self.relay.restore()
        self.state = "NORMAL"
        self.above_since = None
        print("[state] manual reset -> NORMAL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", type=int, default=20,
                    help="frame at which runaway is injected (sim)")
    ap.add_argument("--timers", nargs=3, type=float, default=[5, 10, 20],
                    metavar=("WATCH_S", "WARN_S", "CRIT_S"))
    ap.add_argument("--gpiochip", default=None)
    ap.add_argument("--line", default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--cycle", type=float, default=1.0)
    args = ap.parse_args()

    relay = Relay(args.gpiochip, args.line)
    sm = StateMachine(*args.timers, relay)
    scorer = AnomalyScorer()
    sim = ThermalSim()

    ui = None
    if not args.headless:
        try:
            import tkinter as tk
            ui = tk.Tk()
            ui.attributes("-fullscreen", True)
            ui.configure(bg=COLORS["NORMAL"])
            big = tk.Label(ui, text="NORMAL", font=("Arial", 64, "bold"),
                           fg="white", bg=COLORS["NORMAL"])
            big.pack(expand=True)
            sub = tk.Label(ui, text="", font=("Arial", 22),
                           fg="white", bg=COLORS["NORMAL"])
            sub.pack(pady=20)
            ui.bind("<r>", lambda e: sm.reset())
            ui.bind("<Escape>", lambda e: ui.destroy())
        except Exception as e:
            print(f"[ui] banner unavailable ({e}); console mode")
            ui = None

    n = 0
    try:
        while True:
            n += 1
            if n == args.inject:
                sim.inject_runaway("N", 1, 3)
                print(f"[sim] runaway injected at frame {n}")
            frame = sim.step()
            mats, amb = frame_to_walls(frame)
            score = scorer.score(mats, amb)
            state = sm.step(score, time.time())

            hot = max(float(np.nanmax(mats[w])) for w in WALLS)
            line = (f"f{n:04d} score {score:.2f} state {state:9s} "
                    f"max {hot:5.1f}C amb {amb:5.1f}C relay {relay.state}")
            print(line)

            if ui is not None:
                c = COLORS[state]
                ui.configure(bg=c)
                big.configure(text=state, bg=c)
                sub.configure(
                    text=f"score {score:.2f}   max {hot:.1f}°C   "
                         f"Δamb {hot - amb:+.1f}°C   relay {relay.state}"
                         + ("   [press R to reset]"
                            if state == "ISOLATED" else ""), bg=c)
                ui.update()
            time.sleep(args.cycle)
    except KeyboardInterrupt:
        pass
    finally:
        relay.restore()


if __name__ == "__main__":
    main()
