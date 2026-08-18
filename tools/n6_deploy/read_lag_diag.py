# read_lag_diag.py -- measure the read_events STANDING BACKLOG (temporal lag) with hardware
# event timestamps, and reveal how read_events actually buffers. NO model, NO UART.
# Event columns: [0]type [1]s [2]ms [3]us [4]X [5]Y  -> ts = s*1e6 + ms*1e3 + us (microseconds).
import csi, time
from ulab import numpy as np

BUF = 8192
ev = np.zeros((BUF, 6), dtype=np.uint16)
c = csi.CSI(cid=csi.GENX320); c.reset()
c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)

def ts(row):                            # event absolute timestamp, microseconds (big int -> no overflow)
    return int(row[1]) * 1000000 + int(row[2]) * 1000 + int(row[3])

time.sleep_ms(300)                       # let the sensor accumulate a bit

# ---- (1) ONE read: how many events? oldest-first or newest-first? within-read time span? ----
n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)
print('=== single read ===')
if n >= 2:
    a, b = ts(ev[0]), ts(ev[n - 1])
    print('n=%d  ev[0]=%.3fs  ev[n-1]=%.3fs  span=%.3fs  order=%s' %
          (n, a / 1e6, b / 1e6, (b - a) / 1e6, 'OLDEST-first' if b >= a else 'NEWEST-first'))
oldest0 = min(ts(ev[0]), ts(ev[n - 1]))

# ---- (2) TIGHT-DRAIN: a read that returns FAST = draining backlog; one that BLOCKS (~frame) = caught up.
#         Terminate on the first blocking read (or a 3 s hard cap so it can't hang).
print('=== tight-drain (fast reads = backlog; first slow read = caught up) ===')
total = n; reads = 1; newest = oldest0; fast = 0; w0 = time.ticks_ms()
sizes = []
while time.ticks_diff(time.ticks_ms(), w0) < 3000:      # 3 s hard cap
    t0 = time.ticks_ms()
    n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)
    dt = time.ticks_diff(time.ticks_ms(), t0)
    if n < 1:
        break
    total += n; reads += 1
    if n >= 2:
        newest = max(ts(ev[0]), ts(ev[n - 1]))
    if len(sizes) < 10:
        sizes.append((n, dt))                            # (event_count, read_time_ms)
    if dt > 40:                                          # this read BLOCKED -> we've caught up to real-time
        break
    fast += 1                                            # this read returned immediately -> it drained backlog
w1 = time.ticks_ms()
print('first reads (n, ms):', sizes)
print('FAST (backlog) reads: %d   total drained: %d events in %d ms' % (fast, total, time.ticks_diff(w1, w0)))
print('EVENT-TIME oldest->newest drained = %.2f s   <== THE STANDING BACKLOG / LAG' % ((newest - oldest0) / 1e6))
print('caught-up read: n=%d took %d ms' % (n, dt))

# ---- (3) mimic the RECORDER loop (flush + sleep30 + build @100ms) -- does the LAG GROW over ticks? ----
print('=== recorder-mimic (flush + sleep30 + build): event-time vs wall (30 ticks) ===')
n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)
base_evt = max(ts(ev[0]), ts(ev[n - 1])) if n >= 2 else 0
base_wall = time.ticks_ms()
for k in range(30):
    tick = time.ticks_ms()
    c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)          # flush (discard) -- same as the recorder
    time.sleep_ms(30)
    n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)      # build read = the frame the recorder saves
    if n >= 2:
        ne = max(ts(ev[0]), ts(ev[n - 1]))
        wall = time.ticks_diff(time.ticks_ms(), base_wall)
        evt = (ne - base_evt) / 1000.0                  # ms of event-time advanced since start
        if k % 3 == 0:
            print('  tick %2d: wall %5dms  event-time +%6.0fms  LAG %+6.0fms  n=%d' %
                  (k, wall, evt, wall - evt, n))
    rem = 100 - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
print('  LAG rising over ticks => recorder consumes slower than sensor produces => backlog builds.')
