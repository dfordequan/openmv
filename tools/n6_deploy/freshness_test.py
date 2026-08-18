# freshness_test.py -- keep-NEWEST (triple-buffer) vs keep-OLDEST (FIFO). No motion needed.
# Accumulate 500ms of events, then drain as fast as possible. FIFO queues the whole backlog
# (many reads, keeps OLD); triple-buffer keeps only the latest few (drops stale, keeps NEW).
import csi, image, time
from ulab import numpy as np
BUF = 4096
for FB in (3, 10):
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
    try:
        _c.framebuffers(FB)
    except Exception as e:
        print('FB', FB, 'set err', e)
    _ev = np.zeros((BUF, 6), dtype=np.uint16)
    _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)   # clear
    time.sleep_ms(500)                              # accumulate a 500 ms backlog
    total = 0; reads = 0
    t0 = time.ticks_ms()
    for k in range(20):
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        reads += 1; total += n
        if n < 150:                                 # buffer effectively drained
            break
    dt = time.ticks_diff(time.ticks_ms(), t0)
    mode = 'triple-buffer (keep NEWEST)' if FB <= 3 else 'FIFO (keep OLDEST)'
    print('FB=%2d %-28s: drain took %2d reads, %6d events, %4d ms  -> %s' %
          (FB, mode, reads, total, dt,
           'small backlog = fresh' if total < 3 * BUF else 'LARGE backlog = stale/queued'))
print('done')
