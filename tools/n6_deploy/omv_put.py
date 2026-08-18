#!/usr/bin/env python3
"""Upload a binary file to the OpenMV board's filesystem over the raw REPL.

Usage: python omv_put.py <local_file> <remote_path> [chunk_bytes]
Streams in base64 chunks inside a single raw-REPL session, then verifies size.
"""
import sys, os, time, base64, serial

PORT, BAUD = "/dev/ttyACM0", 115200


def _wait(s, timeout=20):
    """Read until the raw-REPL response terminator (\x04>) appears."""
    buf, t0 = b"", time.time()
    while time.time() - t0 < timeout:
        c = s.read(s.in_waiting or 1)
        if c:
            buf += c
            if buf.endswith(b"\x04>") or buf.endswith(b">"):
                return buf
        else:
            time.sleep(0.002)
    return buf


def _exec(s, code, timeout=20):
    s.write(code.encode() + b"\x04")
    r = _wait(s, timeout)
    if b"Traceback" in r or b"Error" in r:
        raise RuntimeError(r.decode("utf-8", "replace")[-500:])
    return r


def main():
    local, remote = sys.argv[1], sys.argv[2]
    chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 16384
    data = open(local, "rb").read()
    total = len(data)
    print(f"uploading {local} -> {remote}  ({total:,} bytes, chunk={chunk})")

    s = serial.Serial(PORT, BAUD, timeout=1, write_timeout=30, dsrdtr=True)
    try:
        s.write(b"\r\x03\x03"); time.sleep(0.3); s.reset_input_buffer()
        s.write(b"\r\x01"); time.sleep(0.3); s.read(s.in_waiting or 1)   # raw REPL

        _exec(s, "import binascii\nf=open(%r,'wb')" % remote)
        t0 = time.time()
        for i in range(0, total, chunk):
            b64 = base64.b64encode(data[i:i + chunk]).decode()
            _exec(s, "f.write(binascii.a2b_base64('%s'))" % b64, timeout=30)
            done = min(i + chunk, total)
            el = time.time() - t0
            print(f"\r  {done:,}/{total:,} ({100*done//total}%)  {done/1024/max(el,1e-3):.0f} KB/s",
                  end="", flush=True)
        _exec(s, "f.close()")
        print()
        r = _exec(s, "import os\nprint(os.stat(%r)[6])" % remote)
        txt = r.decode("utf-8", "replace")
        size = "".join(ch for ch in txt if ch.isdigit() or ch == "\n").strip().split("\n")[0]
        print(f"remote size: {size} (expected {total})")
        ok = str(total) in txt
        print("VERIFY:", "OK" if ok else "MISMATCH")
        s.write(b"\r\x02")
        sys.exit(0 if ok else 1)
    finally:
        s.close()


if __name__ == "__main__":
    main()
