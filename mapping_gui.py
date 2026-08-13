"""ThermalGuard mapping GUI — warm a probe, click its cell.

Convention (yours): walls A B C D; columns V X Y Z (left->right);
rows 1 2 3 4 (top->bottom). Stored canonically (A->N,B->E,C->S,D->W,
V..Z->0..3, 1..4->0..3) so all other scripts work unchanged.

Cells are LIVE COLOR-MAPPED by temperature (turbo, fixed 24-42C) with a
legend bar. Warmed probe (+2C over baseline) highlights in the table and
auto-selects; click the grid cell where it sits. Entry box accepts
'A 1 V' or 'a1v'. 'amb' = ambient reference.

Reads from ALL connected boards at once. Ports are auto-detected and
rescanned every few seconds, so boards can be plugged in mid-session and
COM numbers may shuffle between reboots without breaking anything.

Usage:  python mapping_gui.py                 # auto-detect every board
        python mapping_gui.py --port COM9 COM10   # or pin them explicitly
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
RESCAN_SEC = 5.0

WALLS_UI = ["A", "B", "C", "D"]
WALL2STD = {"A": "N", "B": "E", "C": "S", "D": "W"}
STD2WALL = {v: k for k, v in WALL2STD.items()}
COLS_UI = ["V", "X", "Y", "Z"]
COL2IDX = {c: i for i, c in enumerate(COLS_UI)}

try:
    from matplotlib import cm
    from matplotlib.colors import Normalize, to_hex
    _norm = Normalize(VMIN, VMAX)
    _cmap = cm.get_cmap("turbo") if hasattr(cm, "get_cmap") else None
    if _cmap is None:
        import matplotlib.pyplot as _plt
        _cmap = _plt.get_cmap("turbo")

    def temp_color(t):
        return to_hex(_cmap(_norm(t)))
except Exception:
    def temp_color(t):
        x = max(0.0, min(1.0, (t - VMIN) / (VMAX - VMIN)))
        r = int(255 * x); b = int(255 * (1 - x))
        return f"#{r:02x}40{b:02x}"


def find_ports():
    """Every plausible Arduino serial port, on Windows or Linux."""
    import serial.tools.list_ports
    out = []
    for p in serial.tools.list_ports.comports():
        blob = " ".join(str(x) for x in
                        (p.description, p.manufacturer, p.hwid))
        if any(k in blob for k in ("Arduino", "CH340", "CH341",
                                   "USB Serial", "CP210", "FT232")):
            out.append(p.device)
        elif p.device.startswith("/dev/ttyACM"):
            out.append(p.device)
    return sorted(set(out))


def reader(port_name, q):
    """One thread per board. Reconnects on its own so a single unplugged
    board never takes down the others."""
    import serial
    while True:
        try:
            with serial.Serial(port_name, 115200, timeout=5) as port:
                port.reset_input_buffer()
                while True:
                    line = port.readline().decode(errors="ignore").strip()
                    if not line:
                        continue
                    if line.startswith('{"seq"'):
                        try:
                            frame = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        temps = {}
                        for bus in frame.get("buses", []):
                            for s in bus.get("sensors", []):
                                if s.get("ok"):
                                    temps[s["rom"]] = s["t"]
                        q.put((port_name, temps))
        except Exception:
            q.put((port_name, None))      # None = this board went away
            time.sleep(2)


class App:
    def __init__(self, root, ports=None):
        self.root = root
        root.title("ThermalGuard mapping")
        self.q = queue.Queue()
        self.fixed_ports = ports          # None => auto-detect
        self.threads = {}
        self.by_port = {}                 # port -> {rom: temp}
        self.dead = set()
        self._last_scan = 0.0
        self.scan_ports()

        self.temps = {}
        self.base = {}
        self.rom_map = json.loads(MAP_PATH.read_text()) \
            if MAP_PATH.exists() else {}
        self.selected = None

        main = ttk.Frame(root, padding=8)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", padx=6)
        wsel = ttk.Frame(left)
        wsel.pack()
        ttk.Label(wsel, text="Wall:").pack(side="left")
        self.wall = tk.StringVar(value="A")
        for w in WALLS_UI:
            ttk.Radiobutton(wsel, text=w, value=w, variable=self.wall,
                            command=self.redraw_grid).pack(side="left")

        hdr = ttk.Frame(left); hdr.pack()
        ttk.Label(hdr, text="   ").grid(row=0, column=0)
        for i, cl in enumerate(COLS_UI):
            ttk.Label(hdr, text=cl, width=10,
                      anchor="center").grid(row=0, column=i + 1)
        gridf = ttk.Frame(left)
        gridf.pack()
        self.cells = {}
        for r in range(ROWS):
            ttk.Label(gridf, text=str(r + 1),
                      width=3).grid(row=r, column=0)
            for c in range(COLS):
                b = tk.Button(gridf, text="—", width=10, height=3,
                              command=lambda r=r, c=c:
                              self.assign_cell(r, c))
                b.bind("<Button-3>", lambda ev, r=r, c=c:
                       self.clear_cell(r, c))
                b.grid(row=r, column=c + 1, padx=2, pady=2)
                self.cells[(r, c)] = b

        # ---- color legend
        leg = tk.Canvas(left, width=44 * 10, height=18,
                        highlightthickness=0)
        leg.pack(pady=(8, 0))
        for i in range(200):
            t = VMIN + (VMAX - VMIN) * i / 199
            leg.create_rectangle(i * 2.2, 0, i * 2.2 + 3, 18,
                                 fill=temp_color(t), outline="")
        legl = ttk.Frame(left); legl.pack(fill="x")
        ttk.Label(legl, text=f"{VMIN:.0f}°C").pack(side="left")
        ttk.Label(legl, text=f"{VMAX:.0f}°C").pack(side="right")

        entryf = ttk.Frame(left)
        entryf.pack(pady=4)
        self.entry = ttk.Entry(entryf, width=12)
        self.entry.pack(side="left")
        ttk.Button(entryf, text="Assign",
                   command=self.assign_entry).pack(side="left", padx=4)

        btnf = ttk.Frame(left)
        btnf.pack(pady=4)
        ttk.Button(btnf, text="Show serial",
                   command=self.show_serial).pack(side="left", padx=3)
        ttk.Button(btnf, text="Rebaseline",
                   command=self.rebaseline).pack(side="left", padx=3)
        ttk.Button(btnf, text="Save",
                   command=self.save).pack(side="left", padx=3)
        self.status = ttk.Label(left, text="waiting for frames...")
        self.status.pack(pady=4)
        self.portlbl = ttk.Label(left, text="")
        self.portlbl.pack()

        cols = ("rom", "temp", "rise", "pos", "port")
        self.tree = ttk.Treeview(main, columns=cols, show="headings",
                                 height=20, selectmode="browse")
        for cid, txt, wd in (("rom", "Serial", 150), ("temp", "°C", 60),
                             ("rise", "Δbase", 60),
                             ("pos", "Position", 80),
                             ("port", "Port", 70)):
            self.tree.heading(cid, text=txt)
            self.tree.column(cid, width=wd, anchor="center")
        self.tree.pack(side="left", fill="y", padx=6)
        self.tree.tag_configure("hot", background="#ff9999")
        self.tree.tag_configure("mapped", background="#ccffcc")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        self.tree.bind("<ButtonPress-1>", self.drag_start)
        self.tree.bind("<B1-Motion>", self.drag_motion)
        self.tree.bind("<ButtonRelease-1>", self.drag_drop)
        self._drag_rom = None

        self.root.after(300, self.tick)

    # ---------- ports
    def scan_ports(self):
        """Start a reader for any board we are not already listening to."""
        self._last_scan = time.time()
        wanted = self.fixed_ports if self.fixed_ports else find_ports()
        for p in wanted:
            if p not in self.threads:
                th = threading.Thread(target=reader, args=(p, self.q),
                                      daemon=True)
                th.start()
                self.threads[p] = th

    # ---------- drag & drop
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

    def tick(self):
        got = False
        while not self.q.empty():
            port, temps = self.q.get()
            if temps is None:
                self.dead.add(port)
            else:
                self.dead.discard(port)
                # Per-port replace, then union: a sensor dropping off one
                # board disappears, but the other boards are untouched.
                self.by_port[port] = temps
                got = True
        if got:
            merged = {}
            for p in sorted(self.by_port):
                merged.update(self.by_port[p])
            self.temps = merged
            self.rom_port = {rom: p for p in sorted(self.by_port)
                             for rom in self.by_port[p]}
            # Baseline fills in per sensor, so boards that connect late
            # still get a baseline instead of a permanent +0.0 rise.
            for rom, t in self.temps.items():
                self.base.setdefault(rom, t)
            self.refresh()
        if time.time() - self._last_scan > RESCAN_SEC:
            self.scan_ports()
        self.root.after(300, self.tick)

    def refresh(self):
        hot, hot_rise = None, THRESH
        for rom, t in self.temps.items():
            rise = t - self.base.get(rom, t)
            if rise >= hot_rise:
                hot, hot_rise = rom, rise
        existing = set(self.tree.get_children())
        live = set(self.temps)
        for rom in existing - live:
            self.tree.delete(rom)
        for rom in sorted(self.temps):
            t = self.temps[rom]
            rise = t - self.base.get(rom, t)
            pos = self.pos_label(rom)
            tags = ("hot",) if rom == hot else \
                   (("mapped",) if pos != "—" else ())
            vals = (rom, f"{t:.1f}", f"{rise:+.1f}", pos,
                    self.rom_port.get(rom, "?"))
            if rom in existing:
                self.tree.item(rom, values=vals, tags=tags)
            else:
                self.tree.insert("", "end", iid=rom, values=vals,
                                 tags=tags)
        if hot and self.selected != hot:
            self.tree.selection_set(hot)
        n_mapped = sum(1 for r in self.temps if r in self.rom_map)
        self.status.config(
            text=f"{len(self.temps)} sensors | {n_mapped} mapped"
                 + (f" | WARMING: ...{hot[-4:]}" if hot else ""))
        live_ports = [p for p in sorted(self.by_port) if p not in self.dead]
        txt = "  ".join(f"{p}:{len(self.by_port[p])}" for p in live_ports)
        if self.dead:
            txt += "   offline: " + ",".join(sorted(self.dead))
        self.portlbl.config(text=txt or "no boards")
        self.redraw_grid()

    def pos_label(self, rom):
        e = self.rom_map.get(rom)
        if not e:
            return "—"
        if e.get("wall") == "AMB":
            return "AMB"
        return f"{STD2WALL[e['wall']]}{e['row'] + 1}" \
               f"{COLS_UI[e['col']]}"

    def redraw_grid(self):
        wstd = WALL2STD[self.wall.get()]
        by_pos = {(e["row"], e["col"]): rom
                  for rom, e in self.rom_map.items()
                  if e.get("wall") == wstd}
        for (r, c), btn in self.cells.items():
            rom = by_pos.get((r, c))
            if rom:
                t = self.temps.get(rom)
                if t is not None:
                    fg = "white" if (t - VMIN) / (VMAX - VMIN) < 0.35 \
                        else "black"
                    btn.config(text=f"{rom[-4:]}\n{t:.1f}°C",
                               bg=temp_color(t), fg=fg,
                               activebackground=temp_color(t))
                else:
                    btn.config(text=f"{rom[-4:]}\n--",
                               bg="#dddddd", fg="black")
            else:
                btn.config(text=f"{self.wall.get()}{r + 1}"
                                f"{COLS_UI[c]}\n—",
                           bg="SystemButtonFace", fg="black")

    def on_select(self, _ev):
        sel = self.tree.selection()
        self.selected = sel[0] if sel else None

    def assign_cell(self, r, c, rom=None):
        rom = rom or self.selected
        if not rom:
            messagebox.showinfo("Pick a sensor",
                                "Warm a probe (auto-selects), click its "
                                "row, or drag a row onto a cell.")
            return
        wstd = WALL2STD[self.wall.get()]
        # evict whoever currently holds this cell
        for other, e in list(self.rom_map.items()):
            if other != rom and e.get("wall") == wstd \
                    and e.get("row") == r and e.get("col") == c:
                del self.rom_map[other]
        self.rom_map[rom] = {"wall": wstd, "row": r, "col": c,
                             "offset": 0.0}
        self.refresh()

    def clear_cell(self, r, c):
        wstd = WALL2STD[self.wall.get()]
        for rom, e in list(self.rom_map.items()):
            if e.get("wall") == wstd and e.get("row") == r \
                    and e.get("col") == c:
                del self.rom_map[rom]
        self.refresh()

    def assign_entry(self):
        if not self.selected:
            messagebox.showinfo("Pick a sensor", "Select a row first.")
            return
        raw = self.entry.get().strip().upper().replace(" ", "")
        if raw == "AMB":
            self.rom_map[self.selected] = {"wall": "AMB", "offset": 0.0}
        else:
            try:
                w, rr, cc = raw[0], raw[1], raw[2]
                assert w in WALLS_UI and rr in "1234" and cc in COL2IDX
                self.rom_map[self.selected] = {
                    "wall": WALL2STD[w], "row": int(rr) - 1,
                    "col": COL2IDX[cc], "offset": 0.0}
            except Exception:
                messagebox.showerror(
                    "Bad label", "Expected like: A1V or A 1 V (or amb)")
                return
        self.entry.delete(0, "end")
        self.refresh()

    def show_serial(self):
        if self.selected:
            messagebox.showinfo("Serial", self.selected)

    def rebaseline(self):
        self.base = dict(self.temps)
        self.refresh()

    def save(self):
        MAP_PATH.parent.mkdir(exist_ok=True)
        MAP_PATH.write_text(json.dumps(self.rom_map, indent=1))
        self.status.config(text=f"saved {len(self.rom_map)} entries -> "
                                f"{MAP_PATH.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", nargs="*", default=None,
                    help="COM ports; omit to auto-detect all boards")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, args.port)
    root.mainloop()


if __name__ == "__main__":
    main()
