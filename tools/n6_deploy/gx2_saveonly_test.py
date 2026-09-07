# gx2_saveonly_test.py -- IS THE CORRUPTION ONLY IN THE SAVE?
# The model reads `img` directly; the log wrote `bytes(img)`. If bytes(img) mangles the top rows but
# `img` itself is clean, then the model saw a CLEAN obs and the top-band is a LOGGING artifact only.
# This compares, for ONE frame: the _g source, what predict reads (img values), and what got written
# (bytes(img)) -- all top-rows means side by side.
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

d = _c.snapshot()
_g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.BILINEAR)
img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))   # exactly as gx2_obslog builds it

def topmean(arr):     # arr = 1D uint8-ish sequence, mean of first 6 rows (192 vals)
    return np.sum(np.array(arr[:6 * G], dtype=np.float)) / (6 * G)

# 1) the _g image source
src = topmean(np.frombuffer(_g.bytearray(), dtype=np.uint8))
# 2) what PREDICT reads: img flattened back to values
pred = topmean(img.reshape((G * G,)))
# 3) what got SAVED: bytes(img)
sb = bytes(img)
save = topmean(np.frombuffer(sb, dtype=np.uint8))

print('TOP-6-rows mean of the SAME frame:')
print('  _g source            : %6.1f' % src)
print('  img  (predict reads) : %6.1f' % pred)
print('  bytes(img) (SAVED)   : %6.1f   [len=%d, want 1024]' % (save, len(sb)))
print('')
if save < 60 and pred > 100:
    print('>>> SAVE-ONLY: predict saw a CLEAN obs; the top-band is a LOGGING artifact (bytes(img) bug).')
elif pred < 60:
    print('>>> REAL: img itself is corrupt -> the model DID see the corrupted obs.')
else:
    print('>>> all clean here -- corruption needs the float-carry feedback / loop; run gx2_carry_test.')
