# N6 EVENT CALIBRATION RECORDER (step 2)  -- deploy as main.py on the N6.
# Point the GenX320 at a representative scene WITH representative motion. This measures the RAW
# per-pixel event-count distribution (the 64x64 ON/OFF frame BEFORE any clip/normalize), so we can
# choose CLIP_DEV for event_frame_builder: set CLIP_DEV ~ the ON p99 under active motion, so a strong
# edge maps to ~1.0 (matching how a strong edge saturates in the sim proxy the policy trained on).
# Logs /sdcard/event_calib.csv and prints a rolling summary. Ctrl-C (or IDE stop) to end -> saves.
import csi, image, time
from ulab import numpy as np

BUF, WIN_MS, G = 16384, 20, 64           # BIG buffer: 2048 saturates under motion -> DROP_ON_FULL
#   silently drops most events (N6_HARDWARE.md §9). 16384 captures far more; 20 ms window gives it
#   time to fill. If frames are STILL sparse vs the raw stream, switch to the hardware histogram mode
#   (on-sensor accumulation, NO drop) -- the robust path for motion.
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(320, 320, image.GRAYSCALE)   # signed accumulator (real polarity)
_act = image.Image(320, 320, image.GRAYSCALE)   # total-activity accumulator
_s64 = image.Image(G, G, image.GRAYSCALE); _a64 = image.Image(G, G, image.GRAYSCALE)


def raw_frame():
    """~WIN_MS of events -> RAW (unclipped) ON/OFF count frames [64,64] + total events."""
    first = True; t0 = time.ticks_ms(); tot = 0
    while time.ticks_diff(time.ticks_ms(), t0) < WIN_MS:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            continue
        _sig.draw_event_histogram(_ev[:n], clear=first, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=first, brightness=0, contrast=1)
        first = False; tot += n
    _s64.draw_image(_sig, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    _a64.draw_image(_act, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    S = _s64.to_ndarray('f'); A = _a64.to_ndarray('f'); net = S - 128.0
    on = np.maximum((A + net) * 0.5, 0.0)          # raw ON count (no clip/normalize)
    off = np.maximum((A - net) * 0.5, 0.0)         # raw OFF count
    return on, off, tot


def pct(x, q):
    xs = np.sort(x.flatten())
    return float(xs[int(q * (xs.size - 1))])


def save_frame(prefix):
    """Save the 64x64 downscaled histograms the model sees, contrast-stretched (histeq) so sparse
    events are visible. _a64 = total activity (ON+OFF); _s64 = signed polarity (128=zero, bright=ON,
    dark=OFF). Grayscale PNGs -> avoids the RGB set_pixel firmware quirk."""
    for im, tag in ((_a64, 'act'), (_s64, 'sig')):
        c = im.copy()
        try:
            c.histeq()          # stretch so the few active pixels stand out
        except Exception:
            pass
        p = prefix + '_' + tag + '.png'
        try:
            c.save(p)
        except Exception:
            c.save(p.replace('.png', '.bmp'))
    print('  saved', prefix + '_{act,sig}.png')


def main():
    f = open('/sdcard/event_calib.csv', 'w')
    f.write('t_s,events,on_p50,on_p95,on_p99,on_max,off_p99,off_max,active_pct\n')
    print('EVENT CALIB recording -> /sdcard/event_calib.csv  (point at the scene; stop to save)')
    t0 = time.ticks_ms(); k = 0
    # running aggregates (so the final suggestion is stable, not one noisy frame)
    agg_p99 = 0.0; agg_max = 0.0; nagg = 0
    saved = 0; next_save = time.ticks_ms()   # save a frame every ~3 s, up to SAVE_N
    SAVE_N = 5
    try:
        while True:
            on, off, tot = raw_frame()
            flat = on.flatten()
            active = float(np.sum(flat > 0.5)) / flat.size * 100.0
            on_p99 = pct(on, 0.99); on_max = float(np.max(on))
            # save a few actual frames (contrast-stretched) to eyeball the representation
            if saved < SAVE_N and active > 0.5 and time.ticks_diff(time.ticks_ms(), next_save) >= 0:
                save_frame('/sdcard/evframe_%02d' % saved)
                saved += 1; next_save = time.ticks_ms() + 3000
            row = (time.ticks_diff(time.ticks_ms(), t0) / 1000.0, tot,
                   pct(on, 0.5), pct(on, 0.95), on_p99, on_max,
                   pct(off, 0.99), float(np.max(off)), active)
            f.write('%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n' % row); f.flush()
            # aggregate only frames with real activity (ignore idle frames)
            if active > 1.0:
                agg_p99 += on_p99; agg_max += on_max; nagg += 1
            k += 1
            if k % 10 == 0:
                sugg = (agg_p99 / nagg) if nagg else on_p99
                print('t%5.0fs  ev/frame %5d  ON p95=%.1f p99=%.1f max=%.1f  active=%4.1f%%  '
                      '-> CLIP_DEV ~= %.1f' % (row[0], tot, row[3], on_p99, on_max, active, sugg))
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        if nagg:
            print('\nDONE. Over %d active frames:  suggested CLIP_DEV ~= %.1f (ON p99),  '
                  'or %.1f (ON max).' % (nagg, agg_p99 / nagg, agg_max / nagg))
        print('CSV saved: /sdcard/event_calib.csv')


if __name__ == '__main__':
    main()
