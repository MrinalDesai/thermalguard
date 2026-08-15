"""Frame server — run on the LAPTOP (Mrinal's architecture).

Reads all acquisition boards over serial (threaded, auto-reconnect),
keeps the latest merged frame in memory, and serves it as JSON at
    http://0.0.0.0:8000/frame.json
The UNO Q polls this once per second (thermalguard_live --source http).
Pull-based: every poll is an independent request — a network hiccup
costs one poll, never a wedged stream. Local network only; no cloud.

Usage (laptop):
  python frame_server.py            # then Q polls http://10.0.0.1:8000
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LATEST = {"seq": 0, "ts": 0.0, "buses": [
    {"bus": 0, "enabled": True, "sensors": []}]}
LOCK = threading.Lock()
MERGED = {}
SEQ = [0]


def find_ports():
    import serial.tools.list_ports
    out = []
    for p in serial.tools.list_ports.comports():
        blob = " ".join(str(x) for x in
                        (p.description, p.manufacturer, p.hwid))
        if any(k in blob for k in ("Arduino", "CH340", "CH341",
                                   "USB Serial", "CP210", "FT232")):
            out.append(p.device)
    return sorted(set(out))


def reader(port_name):
    import serial
    global LATEST
    while True:
        try:
            with serial.Serial(port_name, 115200, timeout=5) as port:
                port.reset_input_buffer()
                print(f"[server] reading {port_name}")
                import sys
                relay_target = getattr(sys.modules[__name__],
                                       "RELAY_PORT_NAME", None)
                if relay_target and port_name == relay_target:
                    RELAY_CONN["port"] = port
                    print(f"[relay] armed on {port_name}")
                while True:
                    line = port.readline().decode(errors="ignore").strip()
                    if not line.startswith('{"seq"'):
                        continue
                    try:
                        f = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    with LOCK:
                        for bus in f.get("buses", []):
                            for s in bus.get("sensors", []):
                                if s.get("ok"):
                                    MERGED[s["rom"]] = (s["t"], time.time())
                        SEQ[0] += 1
                        fresh = time.time() - 30    # drop stale boards
                        LATEST = {"seq": SEQ[0], "ts": time.time(),
                                  "buses": [{"bus": 0, "enabled": True,
                                  "sensors": [
                                      {"rom": r, "t": t, "ok": True}
                                      for r, (t, ts) in MERGED.items()
                                      if ts > fresh]}]}
        except Exception:
            time.sleep(2)


RELAY_CONN = {"port": None}   # set to the relay-board's serial handle


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/relay"):
            cmd = "I" if "cmd=I" in self.path else \
                  ("C" if "cmd=C" in self.path else None)
            ok = False
            conn = RELAY_CONN.get("port")
            if cmd and conn is not None:
                try:
                    conn.write(cmd.encode())
                    ok = True
                    print(f"[relay] sent '{cmd}'")
                except Exception as e:
                    print(f"[relay] write failed: {e}")
            body = (b'{"relay":"ok"}' if ok else b'{"relay":"fail"}')
            self.send_response(200 if ok else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/frame.json":
            self.send_response(404)
            self.end_headers()
            return
        with LOCK:
            body = json.dumps(LATEST).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--relay-port", default=None,
                    help="COM port of the board wearing the relay "
                         "(e.g. COM21); /relay commands go there")
    args = ap.parse_args()
    import sys
    setattr(sys.modules[__name__], "RELAY_PORT_NAME", args.relay_port)
    ports = find_ports()
    if not ports:
        raise SystemExit("no acquisition boards found")
    print(f"[server] boards: {', '.join(ports)}")
    for p in ports:
        threading.Thread(target=reader, args=(p,), daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", 8000), H)
    print("[server] serving http://0.0.0.0:8000/frame.json")
    srv.serve_forever()


if __name__ == "__main__":
    main()
