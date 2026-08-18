# n6_sweep_record.py -- DATA COLLECTION during a controlled open-loop maneuver (no network).
#   TX (N6->CF): FIXED speed + SINUSOIDAL yaw rate (small amplitude) -> a known, repeatable motion.
#   RX (CF->N6): the CF proprio vector, SAVED per frame alongside the event frame.
#   Records the network-input [1,32,32,2] frame + rx-vector + tx-action + timestamp to /sdcard/rec.bin.
#   Each 10 Hz tick DRAINS the sensor backlog to real-time (read_fresh) so the frame is the NEWEST
#   ~frame of events -- fixes the multi-second lag from reading a stale oldest-first backlog.
# Replay/test offboard with deploy/replay_dataset.py (parses the GENXSW01 format).
#
# File format:  header b'GENXSW02' | u16 G | u16 C | u16 VLEN | u16 flags
#   per frame:  u32 t_ms | u32 evt_ms | u16 tot | u16 pad | G*G*C u8 image | VLEN f32 rx_vec | 2 f32 tx_action
#   (evt_ms = newest event's hardware timestamp; the LAG audit = how (t_ms - evt_ms) grows over frames)
import csi, image, time, struct, math, gc
from machine import UART
from ulab import numpy as np

BUF, G, EV_CAP = 8192, 32, 6000
DRAIN_MAX = 20                          # read_fresh cap: max reads to drain the backlog to real-time
BLOCK_MS = 40                           # a read slower than this = buffer caught up = freshest frame
GAIN = 0.07 * 100 * 32
NOISE_FLOOR = 5                         # >FLOOR events per 10x10 block to light a cell (matches main.py)
VLEN = 4
PATH = '/sdcard/rec.bin'
DURATION_S = 20                         # <=~210 frames keeps free memory above the 3 MB gc floor, so
                                        # gc NEVER fires (no ~940 ms hitch). Raise it only if you accept
                                        # one gc pause near the end -- or ask me to shrink the per-frame garbage.
UART_ID, BAUD = 4, 115200
YAWRATE_MAX = 3.5                        # rad/s == full-scale a0 (a0 = yaw_rate / YAWRATE_MAX)
# ---- TX maneuver ----
FIXED_SPEED = 0.30                       # a1 in [-1,1] (CF maps to speed); tune for a safe cruise
YAW_AMP_DEG = 15.0                       # yaw-RATE amplitude in deg/s (small gentle wiggle)
YAW_FREQ_HZ = 0.5                        # sinusoid frequency -> 2 s period
A0_AMP = math.radians(YAW_AMP_DEG) / YAWRATE_MAX   # normalized a0 amplitude (~0.075 at 15 deg/s)

# ---------------- event camera (same obs pipeline as main.py) ----------------
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try: _c.framebuffers(3)
except Exception: pass
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)

def build_img():                        # ONE read (instrumented). Returns frame, tot, and the newest event's
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)     # hardware timestamp (ms) so we can audit the LAG.
    if n < 1:
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0, 0
    evt_ms = int(_ev[n - 1][1]) * 1000 + int(_ev[n - 1][2])   # newest event time (s*1000 + ms)
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]; tot = n - lo
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    frame = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(frame, dtype=np.uint8), tot, evt_ms

# ---------------- UART4 (TX maneuver / RX proprio) ----------------
uart = UART(UART_ID, BAUD)
rx_vec = [1.0, 0.0, 0.0, 0.5]           # last CF proprio (dummy until the CF sends 0xCF frames)
_rx = bytearray()
def send_action(a0, a1):                # N6 -> CF: 0xAA 0x55 | i16 a0*1e4 | i16 a1*1e4 | xor
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
def poll_vector():                      # CF -> N6: 0xCF 0x55 | 4*i16 *1e4 | xor  (MicroPython-safe parse)
    global rx_vec, _rx
    if uart.any():
        _rx.extend(uart.read(uart.any()))
    i = 0; n = len(_rx)
    while i + 11 <= n:
        if _rx[i] != 0xCF or _rx[i + 1] != 0x55:
            i += 1; continue
        body = _rx[i + 2:i + 10]; ck = 0
        for b in body:
            ck ^= b
        if (ck & 0xFF) == _rx[i + 10]:
            v = struct.unpack('<hhhh', body)
            rx_vec = [v[0] / 10000.0, v[1] / 10000.0, v[2] / 10000.0, v[3] / 10000.0]
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

f = open(PATH, 'wb')
f.write(b'GENXSW02' + struct.pack('<HHHH', G, 2, VLEN, 0))
print('SWEEP record: speed %.2f, yaw %.0f deg/s @ %.2f Hz, %d s -> %s'
      % (FIXED_SPEED, YAW_AMP_DEG, YAW_FREQ_HZ, DURATION_S, PATH))
gc.collect()
t0 = time.ticks_ms(); k = 0; PERIOD = 100   # 10 Hz loop -> 10 Hz TX + 10 Hz frames (matches r10 training)
try:
    while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
        tick = time.ticks_ms()
        if gc.mem_free() < 3000000:                       # gc ONLY when low (~once/recording); gc.collect
            gc.collect()                                 # costs ~940 ms on the 25 MB heap -> don't do it often
        t = time.ticks_diff(tick, t0) / 1000.0
        poll_vector()                                    # RX: refresh CF proprio
        a0 = A0_AMP * math.sin(2 * math.pi * YAW_FREQ_HZ * t)   # sinusoidal yaw rate
        a1 = FIXED_SPEED                                 # fixed speed
        send_action(a0, a1)                              # TX: the maneuver command (10 Hz)
        img, tot, evt_ms = build_img()                   # ONE read + newest-event hardware time (lag audit)
        b = bytes(img)                                   # img is already uint8 -> no redundant copy (less GC)
        if len(b) != G * G * 2:
            continue
        # store wall elapsed (WRAP-SAFE) + the newest event's hardware ms -> offline LAG = wall - aligned(evt)
        ok = (f.write(struct.pack('<IIHH', time.ticks_diff(tick, t0), evt_ms, tot & 0xFFFF, 0)) == 12
              and f.write(b) == len(b)
              and f.write(struct.pack('<4f', *rx_vec)) == 16
              and f.write(struct.pack('<2f', a0, a1)) == 8)
        if not ok:
            print('SHORT WRITE (SD full?) after %d frames -> stop' % k); break
        k += 1
        if k % 20 == 0:
            print('  %d frames  a0%+.3f rx%s tot%d free%d' % (k, a0, rx_vec, tot, gc.mem_free()))
        rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
        if rem > 0:
            time.sleep_ms(rem)
finally:
    f.close()
    send_action(0.0, 0.0)               # stop command on exit
print('done: %d frames -> %s' % (k, PATH))
