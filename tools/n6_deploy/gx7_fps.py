# gx7_fps.py -- measure N6 inference FPS of Dreamer GX7 (gx7.bin), predict() ALONE (no camera/UART).
# Builds every input from model.input_shape (so it matches the compiled layout regardless of whether
# stoch is presented as [1,32,16] or [1,16,32]). Feeds carry back each tick. Run in the OpenMV IDE.
import ml, time, gc
from ulab import numpy as np
MODEL = '/sdcard/gx7.bin'; N = 50
us = time.ticks_us; di = time.ticks_diff
gc.collect()
t0 = us(); model = ml.Model(MODEL); print('load %d ms' % (di(us(), t0)//1000))
shp = model.input_shape
print('input_shape:', shp)          # [image, vector, deter, stoch, prevaction] in the compiled layout
ins = [np.zeros(tuple(s)) for s in shp]
S_STOCH = tuple(shp[3])             # whatever the bin wants for stoch
def feedback(out):                  # out = [action, deter_out, stoch_out]
    ins[2] = out[1]
    ins[3] = np.array(out[2]).reshape(S_STOCH)
    ins[4] = out[0]
for _ in range(5):                  # warmup
    out = model.predict(ins); feedback(out)
gc.collect(); t = us()
for _ in range(N):
    out = model.predict(ins); feedback(out)
per = di(us(), t)/N
print('--- GX7 RSSM predict (with carry feedback) ---')
print('  %.1f us = %.2f ms -> %.1f Hz  (target 10 Hz; obs pipeline adds ~40ms on top)' % (per, per/1000.0, 1e6/per))
print('  free %d' % gc.mem_free())
