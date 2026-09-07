# actor_dummy_fps.py -- measure N6 inference FPS of the DUMMY reactive PPO actor
# (actor_dummy_32x1.bin: 32x32x1 uint8 image + vec[4] -> action mean[2], NO recurrency).
# Dummy weights -> numbers are LATENCY only, not behaviour. Run in the OpenMV IDE (or from flash).
#
# It builds zero inputs matching the model's quantized input spec (image fed as uint8, vec as
# int8 + its zero-point -- same pattern as n6_fly_timing.py), warms up, then times PURE predict()
# over N iters. No camera, no UART -> isolates the network cost. Compare to the ~40 ms RSSM predict.
import ml, time, gc
from ulab import numpy as np

MODEL = '/sdcard/actor_dummy_32x1.bin'
N = 50
us = time.ticks_us; di = time.ticks_diff

gc.collect()
t0 = us(); model = ml.Model(MODEL); print('model load: %d ms' % (di(us(), t0) // 1000))
print('input_shape:', model.input_shape)
try:
    Z = model.input_zero_point
except Exception:
    Z = [0, 0]

# inputs: [0]=image [1,32,32,1] uint8, [1]=vector [1,4] int8+zp  (mirror n6_fly_timing feeding)
img = np.zeros((1, 32, 32, 1), dtype=np.uint8)
vec = np.zeros((1, 4), dtype=np.int8) + Z[1]

# warmup (first call includes one-time alloc / cache fill -- exclude from timing)
for _ in range(5):
    out = model.predict([img, vec])
print('warmup out[0]:', out[0][0][0], out[0][0][1])

gc.collect()
t = us()
for _ in range(N):
    out = model.predict([img, vec])
dt = di(us(), t)
per = dt / N
print('--- DUMMY REACTIVE ACTOR (32x32x1, no recurrency) ---')
print('  predict avg  %7.1f us  = %.2f ms  -> %.1f Hz  (pure inference, N=%d)' % (per, per / 1000.0, 1e6 / per, N))
print('  free mem     %d' % gc.mem_free())
