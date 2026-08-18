# gc_model_test.py -- the REAL number: gc.collect() cost WITH the model loaded (main.py condition).
# The 129ms earlier was model-free; the old 940ms was with the model. This settles which it is.
import ml, gc, time
from ulab import numpy as np

m = ml.Model('/sdcard/ratezone3.bin')
Z = m.input_zero_point
img = np.zeros((1, 32, 32, 2), dtype=np.uint8)
vec = np.zeros((1, 4), dtype=np.int8) + Z[1]
d = np.zeros((1, 2048), dtype=np.int8) + Z[2]
s = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
pa = np.zeros((1, 2))
print('model loaded, running 10 inferences to populate the heap like main.py...')
for _ in range(10):
    out = m.predict([img, vec, d, s, pa])
    d = out[1]; s = np.array(out[2]).reshape((1, 16, 32)); pa = np.array(out[0], dtype=np.float)

print('free before gc:', gc.mem_free())
for i in range(3):
    t0 = time.ticks_ms(); gc.collect(); dt = time.ticks_diff(time.ticks_ms(), t0)
    print('gc.collect() WITH model: %d ms   free %d' % (dt, gc.mem_free()))
print('>>> if ~130ms: no refactor needed, just let gc fire every ~15s (130ms coast).')
print('>>> if ~900ms: we do the zero-alloc refactor (or firmware heap shrink).')
