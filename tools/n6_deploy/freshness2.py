# freshness2.py -- measure event LAG behind wall-clock: triple-buffer(3, keep NEWEST) vs FIFO(10).
# 40 ms sleep/read simulates inference. Events are chronological -> ev[n-1] is newest, ev[0] oldest.
# If a config keeps the newest, the newest event stays ~pinned to now (lag ~constant).
# If it queues oldest (FIFO), the newest returned event falls further behind each read (lag GROWS).
import csi, image, time
from ulab import numpy as np
BUF = 8192
def ts_us(row):   # row = _ev[i] -> absolute microseconds (python int, exact)
    return int(row[1]) * 1000000 + int(row[2]) * 1000 + int(row[3])
for FB in (3, 10):
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
    try: _c.framebuffers(FB)
    except Exception as e: print('FB', FB, e)
    _ev = np.zeros((BUF, 6), dtype=np.uint16)
    _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)   # clear
    time.sleep_ms(50)
    lags = []; ref_ts = ref_wall = None
    for k in range(25):
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        w = time.ticks_ms()
        if n >= 2:
            newest = ts_us(_ev[n - 1])
            if ref_ts is None:
                ref_ts = newest; ref_wall = w
            ts_elapsed = (newest - ref_ts) / 1000.0      # ms of sensor-time advanced
            wall_elapsed = time.ticks_diff(w, ref_wall)   # ms of wall-time advanced
            lags.append(wall_elapsed - ts_elapsed)        # how far newest event lags wall clock (ms)
        time.sleep_ms(40)                                 # simulate inference
    if len(lags) >= 3:
        drift = lags[-1] - lags[2]                        # growth of lag over the run
        print('FB=%2d %-22s: lag start %.0f ms -> end %.0f ms  (DRIFT %+.0f ms) -> %s' %
              (FB, 'triple (keep NEW)' if FB <= 3 else 'FIFO (keep OLD)',
               lags[2], lags[-1], drift,
               'FRESH (stays pinned)' if drift < 150 else 'STALE (falls behind)'))
    else:
        print('FB=%2d: too few events to measure' % FB)
print('done')
