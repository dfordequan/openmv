# gx2_fbfix_test.py -- localize + fix the top-band framebuffer corruption.
#
# Root-cause hypothesis: ml.Model grabs a 6.3 MB PERSISTENT alloc; if it can't fit the GC heap it
# spills into the shared UMA pool the 320x320 snapshot framebuffer comes from -> the DMA fills row 0
# at the LOW address that now overlaps the model -> the top ~44% of every frame reads back as zeroed
# memory. This test shows (a) WHERE the model landed (GC heap vs UMA) and (b) whether the raw 320
# capture is CLEAN, for three memory strategies.
#
# RUN IN THE IDE THREE TIMES, changing MODE each run (0 -> 1 -> 2), and compare the printout:
#   MODE 0 = baseline (model-first, no gc)   -- should reproduce the corruption
#   MODE 1 = gc.collect() right before load  -- defragments the GC heap so the model fits there
#   MODE 2 = camera-first (fb reserved first) -- model allocates around the framebuffer
import csi, image, ml, gc
from ulab import numpy as np

MODE  = 0                       # <<< change to 0, then 1, then 2, running once each
MODEL = '/sdcard/gx2_15hz.bin'
SRC   = 320

def bands(d, tag):
    b = np.frombuffer(d.bytearray(), dtype=np.uint8)     # 320*320 uint8 (row-major, row0 first)
    print('  ' + tag + ':')
    for y0, y1, name in ((0, 80, 'rows   0- 80 TOP'), (80, 160, 'rows  80-160'),
                         (160, 240, 'rows 160-240'), (240, 320, 'rows 240-320 BOT')):
        m = np.sum(np.array(b[y0 * SRC:y1 * SRC], dtype=np.float)) / ((y1 - y0) * SRC)
        print('     %-18s mean %5.1f  %s' % (name, m, '<-- ZEROED (corrupt)' if m < 60 else ''))

print('=== MODE %d ===   free at start: %d' % (MODE, gc.mem_free()))

if MODE == 2:
    # camera FIRST: reserve the 320x320 framebuffer before the model exists
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
    _c.snapshot(time=800)
    f0 = gc.mem_free(); m = ml.Model(MODEL)
else:
    if MODE == 1:
        gc.collect()            # defragment/free the GC heap so the 6.3 MB model fits there
    f0 = gc.mem_free(); m = ml.Model(MODEL)
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
    _c.snapshot(time=800)

drop = (f0 - gc.mem_free()) / 1e6
print('  model loaded: GC free dropped %.1f MB  ->  model landed in %s'
      % (drop, 'GC HEAP  (framebuffer SAFE)' if drop > 3 else 'UMA POOL  (collides with framebuffer!)'))

d = _c.snapshot()
bands(d, 'raw 320 snapshot with model resident')
print('>>> CLEAN if all four bands ~128; CORRUPT if the TOP band is ~0. done MODE %d' % MODE)
