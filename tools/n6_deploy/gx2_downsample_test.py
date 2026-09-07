# gx2_downsample_test.py -- the raw 320 snapshot is CLEAN with the model resident (proven by
# gx2_fbfix_test), so the top-band corruption is in the 320->32 DOWNSAMPLE, not the sensor/memory.
# This feeds the SAME clean 320 frame through each draw_image hint and checks the top rows of the
# resulting 32x32 -- to find which downsample zeroes the top (and which one is clean to deploy with).
import csi, image, ml, gc
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
SRC, G = 320, 32
SC = G / SRC

gc.collect()
m = ml.Model(MODEL)                     # model resident = deploy condition
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
d = _c.snapshot()

# raw-320 sanity (should be clean top+bottom)
b = np.frombuffer(d.bytearray(), dtype=np.uint8)
rt = np.sum(np.array(b[:80 * SRC], dtype=np.float)) / (80 * SRC)
rb = np.sum(np.array(b[240 * SRC:], dtype=np.float)) / (80 * SRC)
print('raw 320: TOP %.1f  BOT %.1f  (%s)' % (rt, rb, 'clean' if rt > 100 else 'ALSO CORRUPT'))

def check(g, tag, us=None):
    bb = np.frombuffer(g.bytearray(), dtype=np.uint8)     # 32*32 row-major
    top = np.sum(np.array(bb[:6 * G], dtype=np.float)) / (6 * G)     # rows 0-5
    bot = np.sum(np.array(bb[26 * G:], dtype=np.float)) / (6 * G)    # rows 26-31
    extra = '' if us is None else ' (%d us)' % us
    print('  %-10s 32x32: TOP %5.1f | BOT %5.1f  %s%s'
          % (tag, top, bot, '<-- TOP ZEROED' if top < 60 else 'CLEAN', extra))

print('downsample 320 -> 32 (same clean frame), per hint:')
for hint, name in ((image.AREA, 'AREA'), (image.BILINEAR, 'BILINEAR'),
                   (image.BICUBIC, 'BICUBIC'), (0, 'nearest')):
    g = image.Image(G, G, image.GRAYSCALE)
    import time
    t0 = time.ticks_us()
    try:
        g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=hint)
        check(g, name, time.ticks_diff(time.ticks_us(), t0))
    except Exception as e:
        print('  %-10s FAILED: %s' % (name, e))

# also: does a bigger intermediate (320->64->32) or a full-frame draw avoid it?
g2 = image.Image(G, G, image.GRAYSCALE)
g2.draw_image(d, 0, 0, x_scale=SC, y_scale=SC)          # no hint kw at all (default path)
check(g2, 'default')
print('>>> pick the hint whose TOP is ~128 (clean). done.')
