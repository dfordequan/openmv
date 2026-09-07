# genx_framerate_probe.py -- find how low the GenX320 snapshot() cost can go via set_framerate().
# The end-to-end pipeline is snapshot-bound (~42 ms = 70%); snapshot() is SENSOR-GATED (blocks for
# the next histogram frame). set_framerate() writes EHC_INTEGRATION_PERIOD + the CPI readout cadence
# (genx320.c set_framerate), so a higher rate = faster snapshot -- at the cost of a shorter
# integration window = FEWER events per frame. This sweep measures BOTH so we can pick the point.
#
# snapshot-TIME is objective (no motion needed). The DENSITY column needs MOTION -- wave the board /
# a hand in front of it throughout, or that column reads ~0 and only the timing is meaningful.
import csi, image, time
from ulab import numpy as np

G, SRC = 32, 320; SC = G / SRC
us = time.ticks_us; di = time.ticks_diff
RATES = [None, 30, 50, 75, 100, 150, 200, 300]   # None = leave default (no set_framerate call)

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

def density_pct():
    # % of downsampled pixels that deviate from the 128 background (event content proxy)
    b = np.frombuffer(_g.bytearray(), dtype=np.uint8)
    dev = (b >= 136) + (b <= 120)                # |v-128| >= 8
    return 100.0 * np.sum(np.array(dev, dtype=np.float)) / (G * G)

print('GENX320 framerate sweep -- WAVE something in front of the camera for the density column')
print(' req_fps |  snapshot_ms | downsample_ms | density%% (needs motion)')
for r in RATES:
    ok = 'default'
    if r is not None:
        try:
            _c.set_framerate(r); ok = str(r)
        except Exception as e:
            print(' %7s | set_framerate FAILED: %s' % (str(r), e)); continue
    for _ in range(4):                            # discard settle frames
        _c.snapshot()
    ts = 0; td = 0; dens = 0.0; N = 12
    for _ in range(N):
        t = us(); d = _c.snapshot(); ts += di(us(), t)
        t = us(); _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA); td += di(us(), t)
        dens += density_pct()
    print(' %7s | %10.1f   | %11.1f   | %6.1f' % (ok, ts / N / 1000.0, td / N / 1000.0, dens / N))
print('done. snapshot_ms is objective; density needs consistent motion to compare fairly.')
