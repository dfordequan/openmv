# n6_fly_timing.py -- find the 3.3 Hz (305 ms/frame) bottleneck. Times each step of the fly loop
# for the FLOAT-IO model, then compares predict() vs the INT8 model. Run in the IDE.
import csi, image, time, ml, gc, struct
from ulab import numpy as np

FMODEL = '/sdcard/gx2_15hz_1ms_f.bin'    # float-IO (what gx2_fly_rec uses)
IMODEL = '/sdcard/gx2_15hz.bin'          # int8 (for predict-time comparison; skip if missing)
G, SRC = 32, 320; SC = G / SRC
us = time.ticks_us; di = time.ticks_diff

gc.collect()
t0 = us(); model = ml.Model(FMODEL); print('float-IO model load: %d ms' % (di(us(), t0) // 1000))
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)
deter = np.zeros((1, 2048)); stoch = np.zeros((1, 16, 32)); prevact = np.zeros((1, 2)); vector = np.zeros((1, 4))
f = open('/sdcard/_t.bin', 'wb')

N = 12
acc = [0, 0, 0, 0, 0]
for k in range(N):
    a = us(); d = _c.snapshot(); b = us()                                  # snapshot
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA); c = us()   # downsample
    img = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1)); e = us()  # float conv
    out = model.predict([img, vector, deter, stoch, prevact]); h = us()    # predict (FLOAT-IO)
    f.write(struct.pack('<IH', k, 0)); f.write(bytes(_g.bytearray()))
    f.write(struct.pack('<4f', 0, 0, 0, 0)); f.write(struct.pack('<4f', 0, 0, 0, 0)); j = us()   # logging
    deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32)); prevact = out[0]
    acc[0] += di(b, a); acc[1] += di(c, b); acc[2] += di(e, c); acc[3] += di(h, e); acc[4] += di(j, h)
f.close()
tot = sum(acc) // N
lines = ['per-frame avg (us):',
         '  snapshot     %6d' % (acc[0] // N),
         '  downsample   %6d' % (acc[1] // N),
         '  float-conv   %6d' % (acc[2] // N),
         '  PREDICT(f32) %6d   <-- usually the big one' % (acc[3] // N),
         '  logging(SD)  %6d' % (acc[4] // N),
         '  TOTAL        %6d us = %.0f ms -> %.1f Hz' % (tot, tot / 1000.0, 1e6 / tot)]
rep = open('/sdcard/timing.txt', 'w')                 # <-- pull this file to read the result
for L in lines:
    print(L); rep.write(L + '\n')
rep.flush()

# int8 predict comparison
del model; gc.collect()
try:
    m2 = ml.Model(IMODEL); Z = m2.input_zero_point
    im = np.zeros((1, 32, 32, 2), dtype=np.uint8); ve = np.zeros((1, 4), dtype=np.int8) + Z[1]
    de = np.zeros((1, 2048), dtype=np.int8) + Z[2]; so = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]; pa = np.zeros((1, 2))
    for _ in range(3):
        o = m2.predict([im, ve, de, so, pa]); de = o[1]; so = np.array(o[2]).reshape((1, 16, 32)); pa = np.array(o[0], dtype=np.float)
    a = us()
    for _ in range(N):
        o = m2.predict([im, ve, de, so, pa]); de = o[1]; so = np.array(o[2]).reshape((1, 16, 32)); pa = np.array(o[0], dtype=np.float)
    L = '  PREDICT(int8) %5d us  <-- compare to f32 predict above' % (di(us(), a) // N)
    print(L); rep.write(L + '\n')
except Exception as _e:
    print('  int8 compare skipped:', _e)
rep.close()
print('done. results also in /sdcard/timing.txt  (pull it if not running in the IDE).')
