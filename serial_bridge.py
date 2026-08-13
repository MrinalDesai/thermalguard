"""Serial-to-TCP bridge — run on the LAPTOP.

Reads every Arduino acquisition board (auto-detected COM ports, one
resilient thread each — the proven pattern) and forwards every JSON
frame line over TCP to the UNO Q, where thermalguard_live consumes
them (--source tcp). Exists because the Q's current kernel image lacks
CH34x USB-serial drivers; the network path sidesteps it.

Usage (laptop):
  python serial_bridge.py --host 192.168.101.10
"""

import argparse
import queue
import socket
import threading
import time


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


def reader(port_name, q):
    import serial
    while True:
        try:
            with serial.Serial(port_name, 115200, timeout=5) as port:
                port.reset_input_buffer()
                while True:
                    line = port.readline().decode(errors="ignore").strip()
                    if line.startswith('{"seq"'):
                        q.put(line)
        except Exception:
            time.sleep(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="UNO Q IP")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    ports = find_ports()
    if not ports:
        raise SystemExit("no acquisition boards found")
    print(f"[bridge] boards: {', '.join(ports)}")
    q = queue.Queue()
    for p in ports:
        threading.Thread(target=reader, args=(p, q), daemon=True).start()

    while True:
        try:
            print(f"[bridge] connecting to {args.host}:{args.port} ...")
            s = socket.create_connection((args.host, args.port), timeout=10)
            print("[bridge] connected — streaming")
            while True:
                line = q.get()
                s.sendall((line + "\n").encode())
        except Exception as e:
            print(f"[bridge] link lost ({e}) — retrying in 3s")
            time.sleep(3)


if __name__ == "__main__":
    main()
