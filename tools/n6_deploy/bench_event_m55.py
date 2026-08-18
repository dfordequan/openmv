# Benchmark: how fast can the M55 turn raw events into a representation?
# Run in the OpenMV IDE on the N6 (no GenX320 needed — we synthesize an event batch;
# we're timing the M55's COMPUTE, which is sensor-independent).
#
# Answers: for a typical ~2048-event read, how long does it take to (A) bin events into
# a 64x64 grid in pure MicroPython, vs (C) let the hardware histogram + a C resize do it?
# That tells you whether custom event representations are affordable on-device.

import time
import image
from ulab import numpy as np

N = 2048            # events per read (genx320 event-mode buffer size)
G = 64              # target grid
SCALE = 320 // G    # 5  (320 -> 64)
REPS = 20

# --- synthesize an event batch: cols [type, s, ms, us, x, y] (values arbitrary for timing) ---
k = np.arange(N)
xs = [int(v) for v in ((k * 37) % 320)]      # pseudo-scattered x
ys = [int(v) for v in ((k * 53) % 320)]      # pseudo-scattered y
ps = [int(v) for v in (k % 2)]               # polarity

# ========== (A) pure-Python scatter into ON/OFF 64x64 (bytearrays) ==========
def bin_python():
    on = bytearray(G * G)
    off = bytearray(G * G)
    for i in range(N):
        gx = xs[i] // SCALE
        gy = ys[i] // SCALE
        idx = gy * G + gx
        if ps[i]:
            if on[idx] < 255: on[idx] += 1
        else:
            if off[idx] < 255: off[idx] += 1
    return on, off

bin_python()  # warm
t0 = time.ticks_us()
for _ in range(REPS): bin_python()
tA = time.ticks_diff(time.ticks_us(), t0) / REPS / 1000.0

# ========== (B) vectorize the coords in ulab, loop only the scatter ==========
def bin_vec():
    gx = (k * 0) + 0  # placeholder to keep ulab import used
    X = ((k * 37) % 320)
    Y = ((k * 53) % 320)
    idx = ((Y // SCALE) * G + (X // SCALE))    # ulab vectorized index (if // supported)
    idl = [int(v) for v in idx]
    grid = bytearray(G * G)
    for i in idl:
        if grid[i] < 255: grid[i] += 1
    return grid

tB = -1.0
try:
    bin_vec()
    t0 = time.ticks_us()
    for _ in range(REPS): bin_vec()
    tB = time.ticks_diff(time.ticks_us(), t0) / REPS / 1000.0
except Exception as e:
    print("vec path unsupported:", e)

# ========== (C) hardware-histogram path: 320x320 frame -> AREA downscale to 64x64 ==========
big = image.Image(320, 320, image.GRAYSCALE)
small = image.Image(G, G, image.GRAYSCALE)
def downscale():
    small.draw_image(big, 0, 0, x_scale=G / 320, y_scale=G / 320, hint=image.AREA)

downscale()  # warm
t0 = time.ticks_us()
for _ in range(REPS): downscale()
tC = time.ticks_diff(time.ticks_us(), t0) / REPS / 1000.0

def hz(ms): return 1000.0 / ms if ms > 0 else 0

print("=== M55 event-processing benchmark (N=%d events, %dx%d grid) ===" % (N, G, G))
print("A) pure-Python ON/OFF scatter : %6.2f ms  (%5.0f Hz)" % (tA, hz(tA)))
print("B) ulab-coords + loop scatter : %6.2f ms  (%5.0f Hz)" % (tB, hz(tB)))
print("C) HW-histogram AREA downscale: %6.2f ms  (%5.0f Hz)" % (tC, hz(tC)))
print("--> A/B = cost of building a CUSTOM event repr on the M55;")
print("    C    = cost of the hardware-histogram route (no per-event work).")
