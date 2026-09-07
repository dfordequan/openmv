# ppo_fps.py -- measure N6 inference FPS of PPO h64 (ppo_h64.bin), predict() ALONE (no camera/UART).
# Feedforward: image[1,64,64,3] + vector[1,4] -> action[1,2]. Run in the OpenMV IDE.
import ml, time, gc
from ulab import numpy as np
MODEL = '/sdcard/ppo_h64.bin'; N = 50
us = time.ticks_us; di = time.ticks_diff
gc.collect()
t0 = us(); model = ml.Model(MODEL); print('load %d ms' % (di(us(), t0)//1000))
print('input_shape:', model.input_shape)
img = np.zeros((1, 64, 64, 3)); vec = np.zeros((1, 4))
for _ in range(5):
    out = model.predict([img, vec])
print('warmup action:', out[0][0][0], out[0][0][1])
gc.collect(); t = us()
for _ in range(N):
    out = model.predict([img, vec])
per = di(us(), t)/N
print('--- PPO h64 predict (feedforward, 64x64x3) ---')
print('  %.1f us = %.2f ms -> %.1f Hz  (target ~14 Hz; obs pipeline: 64px snapshot+downsample+stack)' % (per, per/1000.0, 1e6/per))
print('  free %d' % gc.mem_free())
