# stc_afk_probe.py -- does STC/AFK kill the flicker flood on the read_events path?
# Static scene: baseline events/read is mostly flicker+noise. If a filter collapses events/read
# and speeds the loop, it solves the flood (real edges survive under motion).
import csi, image, time
from ulab import numpy as np

G = 32; BUF = 8192
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try:
    _c.framebuffers(10)                     # FIFO to avoid drops during processing
except Exception as e:
    print('framebuffers:', e)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE)
_act = image.Image(G, G, image.GRAYSCALE)

def measure(name, nf=12):
    time.sleep_ms(150)
    tread = tbuild = nsum = occ = 0
    for i in range(nf):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        t1 = time.ticks_ms(); tread += time.ticks_diff(t1, t0); nsum += n
        if n < 1:
            continue
        _ev[:n, 4] = _ev[:n, 4] // 10; _ev[:n, 5] = _ev[:n, 5] // 10
        _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
        A = _act.to_ndarray('f')
        t2 = time.ticks_ms(); tbuild += time.ticks_diff(t2, t1)
        occ += 100.0 * int(np.sum(np.array(A > 0, dtype=np.float))) / (G * G)
    loop = (tread + tbuild) / nf
    print('%-22s ev/read %6.0f  read %6.1f  build %5.1f  loop %5.1f ms = %4.1f Hz  occ@32 %.1f%%' %
          (name, nsum / nf, tread / nf, tbuild / nf, loop, 1000.0 / max(loop, 1), occ / nf))

measure('baseline (no filter)')

try:
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_TRAIL_ONLY, 2)
    measure('STC_TRAIL_ONLY(2ms)')
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_DISABLE)
except Exception as e:
    print('STC_TRAIL_ONLY err', e)

try:
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_ONLY, 1)
    measure('STC_ONLY(1ms)')
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_DISABLE)
except Exception as e:
    print('STC_ONLY err', e)

try:
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_TRAIL, 1, 2)
    measure('STC_TRAIL(1,2ms)')
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_DISABLE)
except Exception as e:
    print('STC_TRAIL err', e)

# AFK: indoor lights ~100/120 Hz + harmonics -> filter a broad low band
try:
    _c.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 90, 300)
    measure('AFK(90-300Hz)')
except Exception as e:
    print('AFK err', e)

# AFK + STC_TRAIL together
try:
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_TRAIL, 1, 2)
    measure('AFK + STC_TRAIL')
    _c.ioctl(csi.IOCTL_GENX320_SET_STC, csi.GENX320_STC_DISABLE)
    _c.ioctl(csi.IOCTL_GENX320_SET_AFK, 0)
except Exception as e:
    print('AFK+STC err', e)
print('done')
