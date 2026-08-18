# n6_sweep_record_1ch.py -- SINGLE-CHANNEL (GenX320 DIFF3D graded net) data collection during a
# controlled open-loop maneuver (no network). This is the SNAPSHOT path: the sensor's on-chip
# histogram, read via csi.snapshot() in GRAYSCALE, gives the 128-centred graded net USAT(net*16+128)
# DIRECTLY (needs the STOCK post_process_histo firmware -- reflash build/OPENMV_N6/bin/firmware.bin
# built 2026-08-18). NO read_events, NO ulab histogram math -> almost zero per-frame garbage, no gc
# hitch, and snapshot() is inherently fresh so there is NO multi-second backlog lag.
#
#   TX (N6->CF): FIXED speed + SINUSOIDAL yaw rate (small amplitude) -> a known, repeatable motion.
#   RX (CF->N6): the CF proprio vector, SAVED per frame alongside the event frame.
#   Records the network-input [1,32,32,1] graded-net frame + rx-vector + tx-action + timestamp.
#
# File format = GENXSW02 with C=1 (so deploy/replay_dataset.py parses it unchanged; evt_ms=0 here
# because snapshot has no per-event hardware timestamp -- the LAG audit is meaningless for snapshot
# mode and will read ~0, which is correct: snapshot is always the freshest 20 ms integration window).
#   header b'GENXSW02' | u16 G | u16 C(=1) | u16 VLEN | u16 flags
#   per frame: u32 t_ms | u32 evt_ms(=0) | u16 tot | u16 pad | G*G*1 u8 image | VLEN f32 rx | 2 f32 tx
import csi, image, time, struct, math, gc
from machine import UART
from ulab import numpy as np

G = 32
SRC = 320                               # snapshot native size
PATH = '/sdcard/rec1.bin'
DURATION_S = 20
VLEN = 4
UART_ID, BAUD = 4, 115200
YAWRATE_MAX = 3.5
# ---- TX maneuver (identical to n6_sweep_record.py so datasets are comparable) ----
FIXED_SPEED = 0.30
YAW_AMP_DEG = 15.0
YAW_FREQ_HZ = 0.5
A0_AMP = math.radians(YAW_AMP_DEG) / YAWRATE_MAX

# ---------------- event camera: HISTO/DIFF3D snapshot (graded net) ----------------
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.snapshot(time=800)                    # settle
_g32 = image.Image(G, G, image.GRAYSCALE)   # the 32x32 obs (AREA-downsampled graded net)

def build_img():
    d = _c.snapshot()                                                     # 320x320 graded net, ~0.2 ms warm
    _g32.draw_image(d, 0, 0, x_scale=G / SRC, y_scale=G / SRC, hint=image.AREA)  # faithful 10x10 average
    b = _g32.bytearray()                                                  # 1024 raw uint8 (128-centred), no ulab
    # tot = count of pixels that moved off the 128 background (a cheap activity meter for the print)
    tot = 0
    for v in b:
        if v < 123 or v > 133:
            tot += 1
    return b, tot

# ---------------- UART4 (TX maneuver / RX proprio) ----------------
uart = UART(UART_ID, BAUD)
rx_vec = [1.0, 0.0, 0.0, 0.5]
_rx = bytearray()
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
def poll_vector():
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
f.write(b'GENXSW02' + struct.pack('<HHHH', G, 1, VLEN, 0))   # C=1
print('SWEEP-1ch record: speed %.2f, yaw %.0f deg/s @ %.2f Hz, %d s -> %s'
      % (FIXED_SPEED, YAW_AMP_DEG, YAW_FREQ_HZ, DURATION_S, PATH))
gc.collect()
t0 = time.ticks_ms(); k = 0; PERIOD = 100    # 10 Hz
try:
    while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
        tick = time.ticks_ms()
        t = time.ticks_diff(tick, t0) / 1000.0
        poll_vector()
        a0 = A0_AMP * math.sin(2 * math.pi * YAW_FREQ_HZ * t)
        a1 = FIXED_SPEED
        send_action(a0, a1)
        b, tot = build_img()
        if len(b) != G * G:
            continue
        ok = (f.write(struct.pack('<IIHH', time.ticks_diff(tick, t0), 0, tot & 0xFFFF, 0)) == 12
              and f.write(b) == len(b)
              and f.write(struct.pack('<4f', *rx_vec)) == 16
              and f.write(struct.pack('<2f', a0, a1)) == 8)
        if not ok:
            print('SHORT WRITE (SD full?) after %d frames -> stop' % k); break
        k += 1
        if k % 20 == 0:
            print('  %d frames  a0%+.3f rx%s active_px%d free%d' % (k, a0, rx_vec, tot, gc.mem_free()))
        rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
        if rem > 0:
            time.sleep_ms(rem)
finally:
    f.close()
    send_action(0.0, 0.0)
print('done: %d frames -> %s' % (k, PATH))
