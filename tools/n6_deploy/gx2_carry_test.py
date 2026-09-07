# gx2_carry_test.py -- the loop is CLEAN when the carry is fed back as int8 (gx2_predict_corrupt_test).
# But gx2_obslog/gx2_main feed the carry back as out[1]/out[2] DIRECTLY, and those come back FLOAT
# (the log's deter block was 8192 B = float32, not 2048 B = int8). Feeding an 8192-byte float array
# into the model's 2048-byte int8 deter_in is a size/type mismatch -> likely a read overrun that
# clobbers the framebuffer. This test (a) prints the real out dtypes/sizes and (b) replicates the
# float feedback to see if the obs goes corrupt.
import csi, image, ml, gc
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
SRC, G = 320, 32
SC = G / SRC

gc.collect()
m = ml.Model(MODEL)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)
Z = m.input_zero_point

def top320(d):
    b = np.frombuffer(d.bytearray(), dtype=np.uint8)
    return np.sum(np.array(b[:100 * SRC], dtype=np.float)) / (100 * SRC)

vec = np.zeros((1, 4), dtype=np.int8) + Z[1]
det = np.zeros((1, 2048), dtype=np.int8) + Z[2]
sto = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
pa = np.zeros((1, 2))

# --- inspect what predict actually returns ---
d = _c.snapshot(); _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.BILINEAR)
img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))
out = m.predict([img, vec, det, sto, pa])
print('deter_in bytes:', len(bytes(det)), '(int8 [1,2048] = 2048)')
print('out[1] deter_out bytes:', len(bytes(out[1])), ' out[2] stoch_out bytes:', len(bytes(np.array(out[2]))))
print('  -> if out[1] is 8192, feeding it back as int8 deter_in is a 4x size mismatch\n')

# --- replicate gx2_obslog's DIRECT (float) feedback; each iter compare raw320 / live-img / saved ---
def tmean(arr, n):
    return np.sum(np.array(arr[:n], dtype=np.float)) / n
print('replicating float carry feedback; per iter: raw320 / live-img(predict) / saved-bytes:')
det = np.zeros((1, 2048), dtype=np.int8) + Z[2]
sto = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
pa = np.zeros((1, 2))
for k in range(16):
    d = _c.snapshot()
    raw = tmean(np.frombuffer(d.bytearray(), dtype=np.uint8), 100 * SRC)
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.BILINEAR)
    img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))
    live = tmean(img.reshape((G * G,)), 6 * G)               # what predict reads
    saved = tmean(np.frombuffer(bytes(img), dtype=np.uint8), 6 * G)  # what would be written
    tag = ('RAW-CORRUPT(capture)' if raw < 60 else
           'LIVE-CORRUPT(model sees it)' if live < 60 else
           'SAVE-ONLY(bytes bug)' if saved < 60 else 'clean')
    print('  %2d : raw %6.1f | live %6.1f | saved %6.1f  %s' % (k, raw, live, saved, tag))
    out = m.predict([img, vec, det, sto, pa])
    det = out[1]                                   # <-- float feedback (the suspect)
    sto = np.array(out[2]).reshape((1, 16, 32))
    pa = np.array(out[0], dtype=np.float)
print('>>> raw-corrupt=predict/carry clobbers capture; live-corrupt=model saw it; save-only=logging bug.')
