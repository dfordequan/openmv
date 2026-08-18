#!/usr/bin/env python3
"""Minimal non-interactive MicroPython raw-REPL driver for the OpenMV board.

Usage:  python omv_repl.py "<python code>"          # run code, print its stdout
        python omv_repl.py -f script.py            # run a file's contents
Exits non-zero and prints the board's traceback on error.
"""
import sys, time, serial

PORT = "/dev/ttyACM0"
BAUD = 115200


def raw_repl_exec(code, port=PORT, timeout=180):
    s = serial.Serial(port, BAUD, timeout=1, write_timeout=5, dsrdtr=True)
    try:
        s.reset_input_buffer()
        s.write(b"\r\x03\x03")          # Ctrl-C twice: stop any running script
        time.sleep(0.3)
        s.reset_input_buffer()
        s.write(b"\r\x01")              # Ctrl-A: enter raw REPL
        time.sleep(0.3)
        s.read(s.in_waiting or 1)
        s.write(code.encode() + b"\x04")  # Ctrl-D: execute
        # collect until the raw-REPL end markers
        buf, t0 = b"", time.time()
        while time.time() - t0 < timeout:
            chunk = s.read(s.in_waiting or 1)
            if chunk:
                buf += chunk
                if buf.count(b"\x04") >= 2:
                    break
            else:
                time.sleep(0.05)
        s.write(b"\r\x02")              # Ctrl-B: back to friendly REPL
        return buf
    finally:
        s.close()


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "-f":
        code = open(sys.argv[2]).read()
    else:
        code = sys.argv[1]
    out = raw_repl_exec(code)
    txt = out.decode("utf-8", "replace")
    if txt.startswith("OK"):
        txt = txt[2:]
    parts = txt.split("\x04")
    stdout = parts[0].strip()
    stderr = parts[1].strip() if len(parts) > 1 else ""
    if stdout:
        print(stdout)
    if stderr:
        print("--- BOARD ERROR ---", file=sys.stderr)
        print(stderr, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
