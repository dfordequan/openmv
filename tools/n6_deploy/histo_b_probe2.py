# histo_b_probe2.py -- FAST re-probe. Questions:
#  (1) Is the HISTO snapshot integration a FIXED period, or does it accumulate since last grab?
#      -> snapshot after a controlled sleep; if denser with longer sleep, it accumulates.
#  (2) At a realistic fast cadence, is the frame sparse (sim ~26% occ) or FLOODED (~100%)?
# Uses C-accelerated AREA downscale (fast) instead of ulab pooling.
import csi, image, time, gc
from ulab import numpy as np

G, W, H = 32, 320, 320
THR = 6.0                                  # |net| > THR counts as an active pixel
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE)
_c.framesize((W, H))
_c.snapshot(time=800)
_s32 = image.Image(G, G, image.GRAYSCALE)

def raw_occ():                             # occupancy on the full 320x320 net (frac |net|>THR)
    d = _c.snapshot()
    S = d.to_ndarray('f'); net = S - 128.0
    amag = np.maximum(net, -net)
    return 100.0 * int(np.sum(np.array(amag > THR, dtype=np.float))) / float(W * H)

# ---- (1) window dependence: occupancy vs sleep before snapshot ----
print('--- window test: occupancy(320) vs dwell before grab ---')
for ms in (2, 10, 30, 100, 300):
    _c.snapshot()                          # flush/reset the accumulator
    time.sleep_ms(ms)
    print('  dwell %4d ms -> raw_occ %.1f%%' % (ms, raw_occ()))

# ---- (2) fast free-running loop: occupancy(32 via AREA) + saturation + timing ----
print('\n--- fast loop (AREA downscale, split, no model) ---')
GAINS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
NF = 30
tsnap = tbuild = 0
ons = []; offs = []; raws = []
for i in range(NF):
    t0 = time.ticks_ms()
    disp = _c.snapshot()
    t1 = time.ticks_ms()
    _s32.draw_image(disp, 0, 0, x_scale=G / float(W), y_scale=G / float(H), hint=image.AREA)
    S = _s32.to_ndarray('f'); net = S - 128.0
    on = np.maximum(net, 0.0); off = np.maximum(-net, 0.0)
    amag = np.maximum(net, -net)
    t2 = time.ticks_ms()
    tsnap += time.ticks_diff(t1, t0); tbuild += time.ticks_diff(t2, t1)
    ons.append(on); offs.append(off)
    raws.append(100.0 * int(np.sum(np.array(amag > 0.5, dtype=np.float))) / float(G * G))

print('timing: snapshot %.1f ms | build %.1f ms | %.1f Hz cap (no model)' %
      (tsnap / NF, tbuild / NF, 1000.0 / max((tsnap + tbuild) / NF, 1)))
rmean = sum(raws) / NF
print('AREA-32 occupancy (frac pixels with |net|>0): %.1f%%   (sim target ~26%%)' % rmean)

NPIX = float(NF * G * G)
print('\n GAIN   occ_pix%   sat%   meanNZ   (sim: occ~26, sat~10)')
for g in GAINS:
    pixocc = sat = nzsum = nzcnt = 0
    for k in range(NF):
        o = np.array(np.minimum(offs[k] * g, 255.0), dtype=np.uint8)
        n = np.array(np.minimum(ons[k] * g, 255.0), dtype=np.uint8)
        pixocc += int(np.sum(np.array((o + n) > 0, dtype=np.float)))
        sat += int(np.sum(np.array(o >= 255, dtype=np.float))) + int(np.sum(np.array(n >= 255, dtype=np.float)))
        s = float(np.sum(o)) + float(np.sum(n)); c = int(np.sum(np.array(o > 0, dtype=np.float))) + int(np.sum(np.array(n > 0, dtype=np.float)))
        nzsum += s; nzcnt += c
    print(' %.2f    %5.1f    %5.1f   %5.0f' %
          (g, 100.0 * pixocc / NPIX, 100.0 * sat / (2 * NPIX), nzsum / max(nzcnt, 1)))
print('free', gc.mem_free())
