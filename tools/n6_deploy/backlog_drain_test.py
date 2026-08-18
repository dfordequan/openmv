# backlog_drain_test.py -- the decisive test: can read_events DRAIN a backlog to real-time, or does
# every read block ~one frame (=> read_fresh can't work)? Build a known 3 s backlog, then tight-drain
# and watch each read's newest EVENT timestamp climb from "3 s ago" toward "now".
import csi, time
from ulab import numpy as np

BUF = 8192
ev = np.zeros((BUF, 6), dtype=np.uint16)
c = csi.CSI(cid=csi.GENX320); c.reset()
c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)

def ts(r):
    return int(r[1]) * 1000000 + int(r[2]) * 1000 + int(r[3])

c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)          # clear whatever is queued
print('building a 3 s backlog (NOT reading for 3 s)...')
time.sleep_ms(3000)

print('tight-drain -- read# : n  read_ms  newest_event_ts')
w0 = time.ticks_ms()
last = 0
for k in range(60):
    t0 = time.ticks_ms()
    n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)
    dt = time.ticks_diff(time.ticks_ms(), t0)
    if n < 1:
        print('  read %2d: n=0' % k); break
    ne = ts(ev[n - 1])
    print('  read %2d: n=%5d  %3dms  newest=%.3fs' % (k, n, dt, ne / 1e6))
    if dt > 40:
        print('  ^ this read BLOCKED (%dms) -> buffer is CAUGHT UP (or blocks per-frame)' % dt)
        break
    last = ne
print('drained the 3 s backlog in %d ms wall.' % time.ticks_diff(time.ticks_ms(), w0))
print('VERDICT: if the fast reads race newest_ts from ~3s-ago up to ~now -> read_fresh WORKS.')
print('         if every read is ~100ms and n~one frame -> can NOT catch up by reading (need another fix).')
