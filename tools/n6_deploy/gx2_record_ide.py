# gx2_record_ide.py -- record a single-channel GenX320 NET dataset for ANALYSIS (to decide the
# per-frame normalizer / the deploy contrast). RUN IN THE OpenMV IDE and MOVE the camera or wave
# objects in front of it so there is REAL motion (a static scene has almost no net).
#
# WHY contrast=1 + full 320x320: contrast=1 makes snapshot() the UNSATURATED net (net+128), and
# recording the full 320 lets the host analyzer derive the 32x32 obs under ANY contrast, AREA/BILINEAR
# downsample, ordering (saturate-then-pool vs pool-then-saturate), and AGC -- then compare to the sim
# training frames and read off the real p90. A contrast=16 recording would bake in saturation and
# destroy the magnitude info we need to make the decision.
#
# Format:  b'GX2NET01' | u16 SRC | u16 G | u16 N | u16 contrast
#   per frame: u32 t_ms | SRC*SRC uint8   (= net + 128, unsaturated)
import csi, image, time, struct, gc
from ulab import numpy as np

N = 100                    # frames (~100 * 100KB = ~10 MB on /sdcard); ~7 s at 14 Hz
SRC, G = 320, 32
PATH = '/sdcard/gx2net.bin'

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.framebuffers(1)
# Try to drop to contrast=1 (unsaturated net). Older firmware (this board) has no set_contrast, so
# we fall back to the sensor default (16) -- the analyzer DETECTS the actual mapping from the data.
CONTRAST = 0
try:
    if _c.set_contrast(1):
        CONTRAST = 1
except Exception as _e:
    print('set_contrast unavailable (%s) -> recording at DEFAULT contrast; analyzer will detect it' % _e)
_c.snapshot(time=800)      # settle
_g = image.Image(G, G, image.GRAYSCALE); SC = G / SRC

f = open(PATH, 'wb')
f.write(b'GX2NET01' + struct.pack('<HHHH', SRC, G, N, CONTRAST))
print('RECORDING %d frames (contrast=%s, full %dx%d) -> %s' % (N, CONTRAST or 'default', SRC, SRC, PATH))
print('>>> MOVE the camera / wave objects in front of it to create motion <<<')
t0 = time.ticks_ms()
for k in range(N):
    d = _c.snapshot()                                              # 320x320 unsaturated net
    f.write(struct.pack('<I', time.ticks_diff(time.ticks_ms(), t0)))
    f.write(d.bytearray())                                         # 102400 uint8 (net+128)
    if k % 20 == 0:
        _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)   # quick 32x32 for live stats
        v = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float)  # ulab: no .astype()
        near128 = 100.0 * np.sum(np.array((v >= 126.0) * (v <= 130.0), dtype=np.float)) / (G * G)  # |v-128|<=2, no abs
        # raw min/mean/max reveal the mapping: ~128 background => graded(USAT); ~0 background => raw net
        print('  k%d  raw: min%.0f mean%.1f max%.0f  near128%%%.0f  free%d' %
              (k, np.min(v), np.mean(v), np.max(v), near128, gc.mem_free()))
    if gc.mem_free() < 6000000:        # the 100KB/frame bytearray is garbage; collect only when low
        gc.collect()
f.close()
print('DONE: %d frames -> %s.  Pull it via the IDE (right-click the SD file -> Save), then I analyze.' % (N, PATH))
