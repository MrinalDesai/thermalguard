"""ThermalGuard mapping GUI — robust multi-board version.

Designed for 4 Arduino/CH340 boards and up to 64 mapped DS18B20 sensors.

Key fixes versus the earlier mapping_gui.py:
  - dead/stale COM ports are purged instead of leaving ghost sensor values
  - per-port latest frame replaces old data atomically
  - GUI drains/merges only the newest frame from each board
  - stale-board timeout prevents old readings from looking live
  - duplicate ROMs across ports are detected and reported
  - parser tolerates boot/debug text before/after JSON
  - reconnects continue independently per port
  - no assumption that a board has exactly 16 sensors

Convention:
  walls A B C D -> N E S W
  columns V X Y Z -> 0 1 2 3
  rows 1 2 3 4 -> 0 1 2 3

Usage:
    python mapping_gui_robust.py
    python mapping_gui_robust.py --port COM9 COM10 COM11 COM12
"""

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, messagebox

MAP_PATH = Path(__file__).parent / "config" / "sensor_map.json"

ROWS, COLS = 4, 4
THRESH = 2.0
VMIN, VMAX = 24.0, 42.0
RESCAN_SEC = 3.0
STALE_SEC = 4.0
GUI_TICK_MS = 200

WALLS_UI = ["A", "B", "C", "D"]
WALL2STD = {"A": "N", "B": "E", "C": "S", "D": "W"}
STD2WALL = {v: k for k, v in WALL2STD.items()}
COLS_UI = ["V", "X", "Y", "Z"]
COL2IDX = {c: i for i, c in enumerate(COLS_UI)}

try:
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex

    _norm = Normalize(VMIN, VMAX)
    try:
        _cmap = cm.get_cmap("turbo")
    except Exception:
        import matplotlib.pyplot as _plt
        _cmap = _plt.get_cmap("turbo")

    def temp_color(t):
        return to_hex(_cmap(_norm(t)))
except Exception:
    def temp_color(t):
        x = max(0.0, min(1.0, (float(t) - VMIN) / (VMAX - VMIN)))
        r = int(255 * x)
        b = int(255 * (1.0 - x))
        return f"#{r:02x}40{b:02x}"


def find_ports():
    """Return plausible Arduino serial ports on Windows or Linux."""
    import serial.tools.list_ports

    out = []
    for p in serial.tools.list_ports.comports():
        blob = " ".join(
            str(x or "") for x in
            (p.description, p.manufacturer, p.hwid, p.product)
        )
        dev = p.device
        if any(k.lower() in blob.lower() for k in
               ("arduino", "ch340", "ch341", "usb serial",
                "cp210", "ft232", "wch")):
            out.append(dev)
        elif dev.startswith("/dev/ttyACM") or dev.startswith("/dev/ttyUSB"):
            out.append(dev)
    return sorted(set(out))


def extract_json(line):
    """Extract one JSON object even if boot/debug text surrounds it."""
    if not line:
        return None
    a = line.find("{")
    b = line.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(line[a:b + 1])
    except json.JSONDecodeError:
        return None


def reader(port_name, q, stop_evt):
    """Independent reconnecting reader for one COM port.

    Sends:
      ("frame", port, timestamp, {rom: temp})
      ("offline", port, timestamp, None)
      ("error", port, timestamp, "message")
    """
    import serial

    last_offline_notice = 0.0

    while not stop_evt.is_set():
        try:
            with serial.Serial(
                port_name,
                115200,
                timeout=1.0,
                write_timeout=1.0,
            ) as port:
                try:
                    port.reset_input_buffer()
                except Exception:
                    pass

                while not stop_evt.is_set():
                    raw = port.readline()
                    if not raw:
                        continue

                    line = raw.decode(errors="ignore").strip()
                    frame = extract_json(line)
                    if not isinstance(frame, dict):
                        continue

                    # Accept the expected ThermalGuard frame shape.
                    if "buses" not in frame:
                        continue

                    temps = {}
                    for bus in frame.get("buses", []):
                        for s in bus.get("sensors", []):
                            try:
                                if s.get("ok") and "rom" in s and "t" in s:
                                    rom = str(s["rom"]).strip()
                                    temps[rom] = float(s["t"])
                            except Exception:
                                continue

                    # A valid frame with zero sensors is still a real frame;
                    # it should clear that port's old sensor cache.
                    q.put(("frame", port_name, time.time(), temps))

        except Exception as exc:
            now = time.time()
            # Avoid flooding the GUI queue with the same reconnect failure.
            if now - last_offline_notice >= 1.5:
                q.put(("offline", port_name, now, None))
                q.put(("error", port_name, now, str(exc)))
                last_offline_notice = now
            time.sleep(1.0)


class App:
    def __init__(self, root, ports=None):
        self.root = root
        self.root.title("ThermalGuard mapping — robust 64 sensor")

        self.q = queue.Queue()
        self.fixed_ports = list(ports) if ports else None
        self.stop_evt = threading.Event()

        self.threads = {}       # port -> thread
        self.by_port = {}       # port -> {rom: temp}
        self.port_ts = {}       # port -> last valid frame timestamp
        self.port_error = {}    # port -> latest error text
        self.dead = set()

        self.temps = {}
        self.rom_port = {}
        self.duplicates = {}
        self.base = {}
        self.selected = None

        self.rom_map = {}
        if MAP_PATH.exists():
            try:
                self.rom_map = json.loads(MAP_PATH.read_text())
            except Exception as exc:
                messagebox.showwarning(
                    "Mapping file",
                    f"Could not read {MAP_PATH.name}: {exc}"
                )

        self._last_scan = 0.0
        self.scan_ports()

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", padx=6, fill="y")

        wsel = ttk.Frame(left)
        wsel.pack()
        ttk.Label(wsel, text="Wall:").pack(side="left")
        self.wall = tk.StringVar(value="A")
        for w in WALLS_UI:
            ttk.Radiobutton(
                wsel, text=w, value=w, variable=self.wall,
                command=self.redraw_grid
            ).pack(side="left")

        hdr = ttk.Frame(left)
        hdr.pack()
        ttk.Label(hdr, text="   ").grid(row=0, column=0)
        for i, cl in enumerate(COLS_UI):
            ttk.Label(
                hdr, text=cl, width=10, anchor="center"
            ).grid(row=0, column=i + 1)

        gridf = ttk.Frame(left)
        gridf.pack()
        self.cells = {}
        for r in range(ROWS):
            ttk.Label(gridf, text=str(r + 1), width=3).grid(row=r, column=0)
            for c in range(COLS):
                b = tk.Button(
                    gridf, text="—", width=10, height=3,
                    command=lambda r=r, c=c: self.assign_cell(r, c)
                )
                b.bind(
                    "<Button-3>",
                    lambda ev, r=r, c=c: self.clear_cell(r, c)
                )
                b.grid(row=r, column=c + 1, padx=2, pady=2)
                self.cells[(r, c)] = b

        leg = tk.Canvas(
            left, width=440, height=18, highlightthickness=0
        )
        leg.pack(pady=(8, 0))
        for i in range(200):
            t = VMIN + (VMAX - VMIN) * i / 199
            x0 = int(i * 2.2)
            leg.create_rectangle(
                x0, 0, x0 + 3, 18,
                fill=temp_color(t), outline=""
            )

        legl = ttk.Frame(left)
        legl.pack(fill="x")
        ttk.Label(legl, text=f"{VMIN:.0f}°C").pack(side="left")
        ttk.Label(legl, text=f"{VMAX:.0f}°C").pack(side="right")

        entryf = ttk.Frame(left)
        entryf.pack(pady=4)
        self.entry = ttk.Entry(entryf, width=12)
        self.entry.pack(side="left")
        ttk.Button(
            entryf, text="Assign", command=self.assign_entry
        ).pack(side="left", padx=4)

        btnf = ttk.Frame(left)
        btnf.pack(pady=4)
        ttk.Button(
            btnf, text="Show serial", command=self.show_serial
        ).pack(side="left", padx=3)
        ttk.Button(
            btnf, text="Rebaseline", command=self.rebaseline
        ).pack(side="left", padx=3)
        ttk.Button(
            btnf, text="Save", command=self.save
        ).pack(side="left", padx=3)

        self.status = ttk.Label(left, text="waiting for frames...")
        self.status.pack(pady=4)

        self.portlbl = ttk.Label(
            left, text="", justify="left", wraplength=470
        )
        self.portlbl.pack(fill="x")

        cols = ("rom", "temp", "rise", "pos", "port")
        self.tree = ttk.Treeview(
            main, columns=cols, show="headings",
            height=24, selectmode="browse"
        )
        for cid, txt, wd in (
            ("rom", "Serial", 155),
            ("temp", "°C", 60),
            ("rise", "Δbase", 60),
            ("pos", "Position", 80),
            ("port", "Port", 80),
        ):
            self.tree.heading(cid, text=txt)
            self.tree.column(cid, width=wd, anchor="center")

        self.tree.pack(side="left", fill="both", expand=True, padx=6)
        self.tree.tag_configure("hot", background="#ff9999")
        self.tree.tag_configure("mapped", background="#ccffcc")
        self.tree.tag_configure("duplicate", background="#ffd27f")

        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<ButtonPress-1>", self.drag_start)
        self.tree.bind("<B1-Motion>", self.drag_motion)
        self.tree.bind("<ButtonRelease-1>", self.drag_drop)
        self._drag_rom = None

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(GUI_TICK_MS, self.tick)

    def on_close(self):
        self.stop_evt.set()
        self.root.destroy()

    # ---------- ports
    def scan_ports(self):
        self._last_scan = time.time()
        wanted = self.fixed_ports if self.fixed_ports else find_ports()

        for p in wanted:
            th = self.threads.get(p)
            if th is None or not th.is_alive():
                th = threading.Thread(
                    target=reader,
                    args=(p, self.q, self.stop_evt),
                    daemon=True,
                    name=f"reader-{p}",
                )
                th.start()
                self.threads[p] = th

    def purge_stale_ports(self):
        now = time.time()
        changed = False

        for p, ts in list(self.port_ts.items()):
            if now - ts > STALE_SEC:
                if p in self.by_port:
                    del self.by_port[p]
                    changed = True
                self.dead.add(p)

        return changed

    def rebuild_union(self):
        """Merge only currently-live port caches.

        If the same ROM appears on more than one port, keep one deterministic
        copy and report the duplication instead of silently hiding it.
        """
        merged = {}
        rom_port = {}
        duplicates = {}

        for p in sorted(self.by_port):
            if p in self.dead:
                continue
            for rom, temp in self.by_port[p].items():
                if rom in merged:
                    duplicates.setdefault(rom, [rom_port[rom]]).append(p)
                    # Keep the first deterministic copy.
                    continue
                merged[rom] = temp
                rom_port[rom] = p

        self.temps = merged
        self.rom_port = rom_port
        self.duplicates = duplicates

        for rom, t in self.temps.items():
            self.base.setdefault(rom, t)

    # ---------- drag and drop
    def drag_start(self, ev):
        row = self.tree.identify_row(ev.y)
        self._drag_rom = row or None

    def drag_motion(self, ev):
        if self._drag_rom:
            self.tree.configure(cursor="hand2")

    def drag_drop(self, ev):
        self.tree.configure(cursor="")
        rom = self._drag_rom
        self._drag_rom = None
        if not rom:
            return
        target = self.root.winfo_containing(ev.x_root, ev.y_root)
        for (r, c), btn in self.cells.items():
            if btn is target:
                self.assign_cell(r, c, rom=rom)
                return

    # ---------- data pump
    def tick(self):
        changed = False

        # Drain the whole queue. Because self.by_port stores only the most recent
        # dictionary per port, old frames cannot accumulate in the UI state.
        while True:
            try:
                kind, port, ts, payload = self.q.get_nowait()
            except queue.Empty:
                break

            if kind == "frame":
                self.by_port[port] = payload
                self.port_ts[port] = ts
                self.dead.discard(port)
                self.port_error.pop(port, None)
                changed = True

            elif kind == "offline":
                self.dead.add(port)
                self.by_port.pop(port, None)   # critical: remove ghost values
                changed = True

            elif kind == "error":
                self.port_error[port] = payload

        if self.purge_stale_ports():
            changed = True

        if changed:
            self.rebuild_union()
            self.refresh()

        if time.time() - self._last_scan > RESCAN_SEC:
            self.scan_ports()

        self.root.after(GUI_TICK_MS, self.tick)

    # ---------- UI
    def refresh(self):
        hot = None
        hot_rise = THRESH

        for rom, t in self.temps.items():
            rise = t - self.base.get(rom, t)
            if rise >= hot_rise:
                hot = rom
                hot_rise = rise

        existing = set(self.tree.get_children())
        live = set(self.temps)

        for rom in existing - live:
            self.tree.delete(rom)

        for rom in sorted(self.temps):
            t = self.temps[rom]
            rise = t - self.base.get(rom, t)
            pos = self.pos_label(rom)

            if rom in self.duplicates:
                tags = ("duplicate",)
            elif rom == hot:
                tags = ("hot",)
            elif pos != "—":
                tags = ("mapped",)
            else:
                tags = ()

            vals = (
                rom,
                f"{t:.1f}",
                f"{rise:+.1f}",
                pos,
                self.rom_port.get(rom, "?"),
            )

            if rom in existing:
                self.tree.item(rom, values=vals, tags=tags)
            else:
                self.tree.insert(
                    "", "end", iid=rom, values=vals, tags=tags
                )

        if hot and self.selected != hot:
            try:
                self.tree.selection_set(hot)
                self.tree.see(hot)
            except Exception:
                pass

        n_mapped = sum(1 for rom in self.temps if rom in self.rom_map)

        status = f"{len(self.temps)} sensors | {n_mapped} mapped"
        if hot:
            status += f" | WARMING: ...{hot[-4:]}"
        if self.duplicates:
            status += f" | DUP ROMS: {len(self.duplicates)}"

        self.status.config(text=status)

        parts = []
        for p in sorted(set(self.threads) | set(self.by_port) | self.dead):
            if p in self.by_port and p not in self.dead:
                age = time.time() - self.port_ts.get(p, time.time())
                parts.append(
                    f"{p}:{len(self.by_port[p])} ({age:.1f}s)"
                )
            else:
                parts.append(f"{p}:OFFLINE")

        self.portlbl.config(text="   ".join(parts) if parts else "no boards")
        self.redraw_grid()

    def pos_label(self, rom):
        e = self.rom_map.get(rom)
        if not e:
            return "—"
        if e.get("wall") == "AMB":
            return "AMB"

        wall = e.get("wall")
        row = e.get("row")
        col = e.get("col")
        if wall not in STD2WALL or not isinstance(row, int) or not isinstance(col, int):
            return "?"
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return "?"

        return f"{STD2WALL[wall]}{row + 1}{COLS_UI[col]}"

    def redraw_grid(self):
        wstd = WALL2STD[self.wall.get()]

        by_pos = {}
        for rom, e in self.rom_map.items():
            if e.get("wall") != wstd:
                continue
            r = e.get("row")
            c = e.get("col")
            if isinstance(r, int) and isinstance(c, int):
                if 0 <= r < ROWS and 0 <= c < COLS:
                    by_pos[(r, c)] = rom

        for (r, c), btn in self.cells.items():
            rom = by_pos.get((r, c))

            if rom:
                t = self.temps.get(rom)
                if t is not None:
                    fg = (
                        "white"
                        if (t - VMIN) / (VMAX - VMIN) < 0.35
                        else "black"
                    )
                    btn.config(
                        text=f"{rom[-4:]}\n{t:.1f}°C",
                        bg=temp_color(t),
                        fg=fg,
                        activebackground=temp_color(t),
                    )
                else:
                    btn.config(
                        text=f"{rom[-4:]}\n--",
                        bg="#dddddd",
                        fg="black",
                    )
            else:
                btn.config(
                    text=f"{self.wall.get()}{r + 1}{COLS_UI[c]}\n—",
                    bg="SystemButtonFace",
                    fg="black",
                )

    def on_select(self, _ev):
        sel = self.tree.selection()
        self.selected = sel[0] if sel else None

    def assign_cell(self, r, c, rom=None):
        rom = rom or self.selected
        if not rom:
            messagebox.showinfo(
                "Pick a sensor",
                "Warm a probe, select its row, or drag a row onto a cell.",
            )
            return

        wstd = WALL2STD[self.wall.get()]

        for other, e in list(self.rom_map.items()):
            if (
                other != rom
                and e.get("wall") == wstd
                and e.get("row") == r
                and e.get("col") == c
            ):
                del self.rom_map[other]

        self.rom_map[rom] = {
            "wall": wstd,
            "row": r,
            "col": c,
            "offset": 0.0,
        }
        self.refresh()

    def clear_cell(self, r, c):
        wstd = WALL2STD[self.wall.get()]
        for rom, e in list(self.rom_map.items()):
            if (
                e.get("wall") == wstd
                and e.get("row") == r
                and e.get("col") == c
            ):
                del self.rom_map[rom]
        self.refresh()

    def assign_entry(self):
        if not self.selected:
            messagebox.showinfo("Pick a sensor", "Select a row first.")
            return

        raw = self.entry.get().strip().upper().replace(" ", "")

        if raw == "AMB":
            self.rom_map[self.selected] = {
                "wall": "AMB",
                "offset": 0.0,
            }
        else:
            try:
                if len(raw) != 3:
                    raise ValueError
                w, rr, cc = raw[0], raw[1], raw[2]
                if w not in WALLS_UI or rr not in "1234" or cc not in COL2IDX:
                    raise ValueError
                self.rom_map[self.selected] = {
                    "wall": WALL2STD[w],
                    "row": int(rr) - 1,
                    "col": COL2IDX[cc],
                    "offset": 0.0,
                }
            except Exception:
                messagebox.showerror(
                    "Bad label",
                    "Expected A1V / A 1 V / amb",
                )
                return

        self.entry.delete(0, "end")
        self.refresh()

    def show_serial(self):
        if not self.selected:
            return

        msg = self.selected
        if self.selected in self.duplicates:
            msg += "\n\nWARNING: same ROM is visible on ports:\n" + \
                   ", ".join(self.duplicates[self.selected])

        messagebox.showinfo("Serial", msg)

    def rebaseline(self):
        self.base = dict(self.temps)
        self.refresh()

    def save(self):
        MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        MAP_PATH.write_text(json.dumps(self.rom_map, indent=2))
        self.status.config(
            text=f"saved {len(self.rom_map)} entries -> {MAP_PATH.name}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        nargs="*",
        default=None,
        help="COM ports; omit to auto-detect all boards",
    )
    args = ap.parse_args()

    root = tk.Tk()
    App(root, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
