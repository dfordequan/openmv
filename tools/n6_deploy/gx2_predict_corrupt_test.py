# gx2_predict_corrupt_test.py -- raw 320 is clean AND every downsample hint is clean, yet the
# flight obs was corrupt. The one thing the flight loop does that the earlier tests didn't:
# call predict() BEFORE the next snapshot. So this replicates the real loop
# (snapshot -> downsample(reused _g) -> predict, repeat) and checks the TOP rows of BOTH the raw
# 320 and the 32x32 obs every iteration -- to see if predict() corrupts the following capture.
import csi, image, ml, gc, time
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
SRC, G = 320, 32
SC = G / SRC

gc.collect()
m = ml.Model(MODEL)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)          # reused across the loop, like gx2_main/gx2_obslog
Z = m.input_zero_point

def top320(d):
    b = np.frombuffer(d.bytearray(), dtype=np.uint8)
    return np.sum(np.array(b[:100 * SRC], dtype=np.float)) / (100 * SRC)   # rows 0-99
def top32(g):
    b = np.frombuffer(g.bytearray(), dtype=np.uint8)
    return np.sum(np.array(b[:10 * G], dtype=np.float)) / (10 * G)         # rows 0-9

print('iter : raw320-top  obs32-top   (clean ~128, corrupt ~0)')
for k in range(8):
    d = _c.snapshot()                            # capture (frames 1+ follow a predict)
    r = top320(d)
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.BILINEAR)
    o = top32(_g)
    print('  %2d  :  %6.1f      %6.1f    %s' % (k, r, o,
          '<-- CORRUPT' if (r < 60 or o < 60) else 'clean'))
    # now run predict (as in the flight loop) BEFORE the next snapshot
    img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))
    vec = np.zeros((1, 4), dtype=np.int8) + Z[1]
    det = np.zeros((1, 2048), dtype=np.int8) + Z[2]
    sto = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
    pa = np.zeros((1, 2))
    m.predict([img, vec, det, sto, pa])
print('>>> if iter 0 is clean but 1+ go corrupt, predict() corrupts the NEXT capture.')
