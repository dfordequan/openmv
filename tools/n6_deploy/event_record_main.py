# N6 RAW EVENT RECORDER  ->  /sdcard/events.bin
# Records a few seconds of the RAW GenX320 event stream so we can analyze it offline on the laptop
# (true event rate, drops, and replay through the frame builder to pick CLIP_DEV) -- read with
# event_read.py.  Uses the max EVT_res (65536) to minimize on-sensor DROP_ON_FULL.
#
# File format:  8-byte magic b'GENXEV01', then repeated records:
#     <uint32 n><uint32 t_us>  followed by  n*6 uint16  (the events[:n] rows)
# Event columns: [0]type(polarity) [1]s [2]ms [3]us [4]X 0-319 [5]Y 0-319
#
# NOTE: raw events are BIG (~12 MB/s at 1 M ev/s) and SD write speed can itself bottleneck (causing
# extra drops), so keep DURATION_S short. The us timestamps let the reader compute the TRUE rate and
# spot time gaps (= drops) even if writing is bursty.
import csi, time, struct, os
from ulab import numpy as np

EVT_RES = 65536          # max event buffer (power of two, 1024..65536)
DURATION_S = 5           # recording length in seconds  (raise carefully -- file grows fast)
PATH = '/sdcard/events.bin'

c = csi.CSI(cid=csi.GENX320); c.reset()
c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, EVT_RES)
ev = np.zeros((EVT_RES, 6), dtype=np.uint16)

f = open(PATH, 'wb'); f.write(b'GENXEV01')
print('recording %d s of raw events -> %s  (EVT_res=%d)' % (DURATION_S, PATH, EVT_RES))
t0 = time.ticks_ms(); tot = 0; recs = 0
try:
    bad = 0
    while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
        n = c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, ev)
        if n < 1 or n > EVT_RES:          # guard: skip error codes / impossible counts (was corrupting the file)
            bad += 1
            continue
        b = bytes(ev[:n])
        if len(b) != n * 12:              # guard: write only if the byte count matches n exactly
            bad += 1
            continue
        try:                              # guard: a short/failed write (SD full) STOPS cleanly, no corruption
            if f.write(struct.pack('<II', n, time.ticks_us())) != 8 or f.write(b) != len(b):
                print('SHORT WRITE (SD full?) after %d reads -> stopping' % recs); break
        except Exception as e:
            print('write error:', e, '-> stopping'); break
        tot += n; recs += 1
    if bad:
        print('skipped %d bad reads (invalid n or byte count)' % bad)
finally:
    f.close()
dt = time.ticks_diff(time.ticks_ms(), t0) / 1000.0
print('DONE: %d events over %d reads in %.2f s  ->  %.0f ev/s (CAPTURED),  file %.1f MB' %
      (tot, recs, dt, tot / max(dt, 1e-6), os.stat(PATH)[6] / 1e6))
print('pull it:  mpremote connect /dev/ttyACM0 fs cp :%s ./events.bin' % PATH)
