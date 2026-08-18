# buf_sweep.py -- obs-rate vs event-capture knob for the read_events path. Static.
import csi, image, time
from ulab import numpy as np
G = 32
for BUF in (2048, 4096, 8192, 16384):
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
    try: _c.framebuffers(10)
    except Exception: pass
    _ev = np.zeros((BUF, 6), dtype=np.uint16)
    _sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)
    time.sleep_ms(150)
    nf = 15; tread = tbuild = nsum = occ = 0
    for i in range(nf):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        t1 = time.ticks_ms(); tread += time.ticks_diff(t1, t0); nsum += n
        if n < 1: continue
        _ev[:n, 4] = _ev[:n, 4] // 10; _ev[:n, 5] = _ev[:n, 5] // 10
        _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
        A = _act.to_ndarray('f'); t2 = time.ticks_ms(); tbuild += time.ticks_diff(t2, t1)
        occ += 100.0 * int(np.sum(np.array(A > 0, dtype=np.float))) / (G * G)
    loop = (tread + tbuild) / nf
    print('BUF %5d: ev/read %6.0f  read %5.1f  build %4.1f  OBS %5.1f ms = %4.1f Hz  occ %.1f%%  (+infer~46 -> %.1f Hz full)' %
          (BUF, nsum / nf, tread / nf, tbuild / nf, loop, 1000.0 / max(loop, 1), occ / nf, 1000.0 / max(loop + 46, 1)))
print('done')
