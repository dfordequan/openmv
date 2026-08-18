# readevt_probe.py -- baseline for the TRUE 2-channel path (EVENT mode read_events).
# Measures events/read, occupancy@32, saturation, per-stage timing, vs sim (occ~26%, sat~10%).
# This is the path ratezone3_main.py uses (7 Hz). Confirms it gives sim-like sparse 2-ch data.
import csi, image, time, gc
from ulab import numpy as np

G = 32
for BUF in (16384, 65536):
    _c = csi.CSI(cid=csi.GENX320); _c.reset()
    _c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
    _ev = np.zeros((BUF, 6), dtype=np.uint16)
    _sig = image.Image(G, G, image.GRAYSCALE)
    _act = image.Image(G, G, image.GRAYSCALE)
    time.sleep_ms(200)

    NF = 20; tread = tbuild = 0; nsum = 0
    ons = []; offs = []
    for i in range(NF):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        t1 = time.ticks_ms(); tread += time.ticks_diff(t1, t0); nsum += n
        if n < 1:
            ons.append(np.zeros((G, G))); offs.append(np.zeros((G, G))); continue
        _ev[:n, 4] = _ev[:n, 4] // 10; _ev[:n, 5] = _ev[:n, 5] // 10
        _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
        S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
        on = np.maximum((A + net) * 0.5, 0.0); off = np.maximum((A - net) * 0.5, 0.0)
        t2 = time.ticks_ms(); tbuild += time.ticks_diff(t2, t1)
        ons.append(on); offs.append(off)

    NPIX = float(NF * G * G)
    # raw occupancy (any polarity nonzero, pre-gain) + count stats
    pixocc = 0; cmax = 0.0; csum = 0.0; ccnt = 0
    for k in range(NF):
        o = offs[k]; n_ = ons[k]
        pixocc += int(np.sum(np.array((o + n_) > 0, dtype=np.float)))
        m = float(np.max(o + n_)); cmax = m if m > cmax else cmax
        tot = o + n_; csum += float(np.sum(tot)); ccnt += int(np.sum(np.array(tot > 0, dtype=np.float)))
    print('BUF %d: events/read %.0f  read %.1fms  build %.1fms  loop~%.1fms=%.1fHz' %
          (BUF, nsum / NF, tread / NF, tbuild / NF, (tread + tbuild) / NF, 1000.0 / max((tread + tbuild) / NF, 1)))
    print('   raw occupancy@32 (pre-gain) %.1f%% (sim ~26%%)  maxcount %.0f  meanNZcount %.1f' %
          (100.0 * pixocc / NPIX, cmax, csum / max(ccnt, 1)))
    # gain sweep for saturation
    for g in (0.07 * 100 * 32, 0.02 * 100 * 32, 0.5, 2.0):
        sat = pocc = 0
        for k in range(NF):
            o = np.array(np.minimum(offs[k] * g, 255.0), dtype=np.uint8)
            n_ = np.array(np.minimum(ons[k] * g, 255.0), dtype=np.uint8)
            sat += int(np.sum(np.array(o >= 255, dtype=np.float))) + int(np.sum(np.array(n_ >= 255, dtype=np.float)))
            pocc += int(np.sum(np.array((o + n_) > 0, dtype=np.float)))
        print('     GAIN %.1f -> occ_pix %.1f%%  sat %.1f%% (sim sat~10%%)' %
              (g, 100.0 * pocc / NPIX, 100.0 * sat / (2 * NPIX)))
    gc.collect()
