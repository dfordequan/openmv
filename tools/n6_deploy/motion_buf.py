# motion_buf.py -- occupancy per BUF UNDER MOTION -> answers "why not 2048?".
# WAVE HAND ACROSS LENS CONTINUOUSLY the whole run. Compares occ@32 to sim target ~26%.
import csi, image, time
from ulab import numpy as np
G = 32
time.sleep_ms(3000)                         # lead-in: start waving now
for BUF in (2048, 4096, 8192):
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
    try: _c.framebuffers(10)
    except Exception: pass
    _ev = np.zeros((BUF, 6), dtype=np.uint16)
    _sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)
    time.sleep_ms(200)
    nf = 20; tread = nsum = 0; occs = []
    for i in range(nf):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        tread += time.ticks_diff(time.ticks_ms(), t0); nsum += n
        if n < 1:
            occs.append(0.0); continue
        _ev[:n, 4] = _ev[:n, 4] // 10; _ev[:n, 5] = _ev[:n, 5] // 10
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
        A = _act.to_ndarray('f')
        occs.append(100.0 * int(np.sum(np.array(A > 0, dtype=np.float))) / (G * G))
    occs.sort()
    rd = tread / nf
    print('BUF %5d: ev/read %6.0f  read %5.1f ms (~%4.1f Hz obs)  occ median %.1f%%  peak %.1f%%  (sim ~26%%)' %
          (BUF, nsum / nf, rd, 1000.0 / max(rd, 1), occs[nf // 2], occs[-1]))
print('done -- stop waving')
