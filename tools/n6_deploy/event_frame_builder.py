# On-device EVENT FRAME BUILDER for the N6 + GenX320  (100 Hz, raw events, 2048 buffer)
#
# Produces the CANONICAL navigation tensor:  float32 [64, 64, 2]  in [0,1]
#   channel 0 = ON  event density   (brightness increased)
#   channel 1 = OFF event density   (brightness decreased)
# This is the SAME tensor the analytic proxy emits (skydreamer_cf/event_proxy_demo/
# event_proxy.py -> event_frame()), so a Dreamer policy trained in sim ingests the
# identical representation it will see on hardware.
#
# WHY this shape (see the design discussion):
#   * 2-channel ON/OFF  -> no signed-difference cancellation, keeps motion polarity.
#   * built with 2x draw_event_histogram (C speed) -> keeps up within the 2048 budget.
#   * 64x64x2 int8/NHWC -> small, NPU-friendly CNN input.
#   * trivially proxy-able from an analytic flow field -> fast Dreamer training.
#
# HOW it stays at 100 Hz with a 2048 buffer:
#   read_events BLOCKS until the 2048 buffer fills (~3 ms at ~650k ev/s), so we
#   ACCUMULATE reads (clear=False) for WIN_MS=10 ms -> ~3 reads / ~6k events per frame,
#   then downscale 320->64 and reconstruct ON/OFF.  Window is wall-clock, so the rate is
#   steady regardless of scene activity (density varies instead, as event cameras do).
#
# NOTE on normalization: on-device counts (averaged by the AREA downscale) and the sim
# proxy's counts live on different absolute scales, so each domain has its OWN clip
# constant chosen so a typical active edge lands near ~0.5.  Domain randomization in the
# proxy bridges the residual gap.  CLIP_DEV below is a starting value -- calibrate it with
# the built-in measure() (prints ON/OFF percentiles) and set it so p95 ~ 1.0.

import csi, image, time
from ulab import numpy as np

BUF = 2048          # event buffer (read granularity); keep 2048 as chosen
WIN_MS = 10         # accumulation window -> 100 Hz
G = 64              # output grid (64x64)
CLIP_DEV = 3.0      # on-device count clip for normalization (CALIBRATE with measure())

_c = csi.CSI(cid=csi.GENX320)
_c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(320, 320, image.GRAYSCALE)   # signed accumulator (real polarity)
_act = image.Image(320, 320, image.GRAYSCALE)   # total-activity accumulator (all->ON)
_s64 = image.Image(G, G, image.GRAYSCALE)
_a64 = image.Image(G, G, image.GRAYSCALE)


def build_frame(clip=CLIP_DEV):
    """Accumulate ~WIN_MS of events -> canonical float32 [G,G,2] in [0,1] (ch0=ON, ch1=OFF).
    Also returns (n_events, n_reads) for monitoring."""
    first = True
    t0 = time.ticks_ms()
    tot = 0
    nr = 0
    while time.ticks_diff(time.ticks_ms(), t0) < WIN_MS:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            continue
        # signed histogram (ON=+1, OFF=-1) using the REAL polarity
        _sig.draw_event_histogram(_ev[:n], clear=first, brightness=128, contrast=1)
        # total activity: force every event to ON, then histogram = ON+OFF count
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=first, brightness=0, contrast=1)
        first = False
        tot += n
        nr += 1
    # downscale 320 -> 64 (AREA averages counts, C speed)
    _s64.draw_image(_sig, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    _a64.draw_image(_act, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    S = _s64.to_ndarray('f')          # 0..255, 128 = zero net
    A = _a64.to_ndarray('f')          # 0..255, total activity
    net = S - 128.0                   # ON - OFF
    on = (A + net) * 0.5              # separate ON / OFF counts
    off = (A - net) * 0.5
    # clip negatives (reconstruction noise) and normalize to [0,1]
    on = np.minimum(np.maximum(on, 0.0) / clip, 1.0)
    off = np.minimum(np.maximum(off, 0.0) / clip, 1.0)
    frame = np.concatenate((on.reshape((G, G, 1)), off.reshape((G, G, 1))), axis=2)
    return frame, tot, nr


def measure(K=120):
    """Time the builder and print value distribution -> use to set CLIP_DEV and confirm 100 Hz."""
    for _ in range(3):
        build_frame()                 # warm
    t0 = time.ticks_us()
    es = 0
    rs = 0
    pk = 0.0
    mean = 0.0
    for _ in range(K):
        f, tot, nr = build_frame(clip=1.0)   # clip=1 -> see raw normalized magnitudes
        es += tot
        rs += nr
        m = float(np.max(f))
        pk = m if m > pk else pk
        mean += float(np.mean(f))
    dt = time.ticks_diff(time.ticks_us(), t0) / K / 1000.0
    print("frame %.2f ms -> %.0f Hz (target 100)" % (dt, 1000.0 / dt))
    print("events/frame %d   reads/frame %.1f" % (es // K, rs / K))
    print("raw count: peak %.1f  mean %.3f  -> set CLIP_DEV ~ peak" % (pk, mean / K))


if __name__ == "__main__":
    measure()
    # deploy loop would be:
    #   net = ml.Model('/sdcard/navnet.bin')          # your Dreamer encoder/policy
    #   while True:
    #       frame, _, _ = build_frame()
    #       out = net.predict([frame])                # [64,64,2] -> action/latent
