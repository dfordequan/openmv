# obs_viz.py -- shows EXACTLY the [32,32,2] observation the network receives, in the OpenMV IDE.
# NO inference, NO UART. Just main.py's build_img, rendered faithfully so you can SEE the net's input.
#   green = ON (ch1),  red = OFF (ch0),  yellow = BOTH at that pixel,  black = no event.
# It is the REAL obs: GAIN=224 saturates on ~1 event, so pixels are near-binary -- that IS what the
# network sees (a coarse 32x32 near-binary event mask), upscaled 10x here only so it's visible.
import csi, image, time, gc
from ulab import numpy as np

BUF, G, DRAIN_MAX, EV_CAP = 8192, 32, 20, 6000   # DRAIN_MAX = read_fresh cap (drain backlog to real-time)
BLOCK_MS = 40                          # a read slower than this = buffer caught up = freshest frame
GAIN = 0.07 * 100 * 32                  # SAME as main.py -> identical obs values
NOISE_FLOOR = 5                         # require > NOISE_FLOOR events in a 10x10 block to light the cell.
                                        # Kills the single-event noise the SUM-bin amplifies (1 sensor px
                                        # -> whole 32x32 cell). 0 = off (old behaviour). Tuned to 5 on-board.
UP = 10                                 # upscale 32 -> 320 for viewing (nearest = blocky = true pixels)

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try: _c.framebuffers(3)
except Exception: pass
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)
_c32 = image.Image(G, G, image.RGB565)          # the exact obs, in colour
_disp = image.Image(G * UP, G * UP, image.RGB565)  # upscaled for the IDE frame view

def read_fresh():                       # drain stale backlog to real-time (see main.py); returns newest frame
    for _ in range(DRAIN_MAX):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            return 0
        if time.ticks_diff(time.ticks_ms(), t0) > BLOCK_MS:
            return n
    return n

def build_obs():                        # read_fresh (drains to real-time) -> (on,off) 0-255 32x32, event count
    n = read_fresh()
    if n < 1:
        return None, None, 0
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]; tot = n - lo
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)   # net = ON-OFF
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)     # total = ON+OFF
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    # ON/OFF event COUNT per 32x32 bin, then subtract the noise floor -> only blocks with >FLOOR
    # events survive (isolated single-event noise -> 0), then *GAIN saturates real edges (near-binary).
    on = np.minimum(np.maximum((A + net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)   # ch1 = ON  (0-255)
    off = np.minimum(np.maximum((A - net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)  # ch0 = OFF (0-255)
    return on, off, tot

clock = time.clock()
print('obs_viz: showing the exact [32,32,2] network input.  green=ON red=OFF yellow=both black=none')
while True:
    clock.tick()
    on, off, tot = build_obs()
    if on is not None:
        for y in range(G):
            oy = on[y]; fy = off[y]                      # ulab rows (avoid .tolist(); works on all ulab)
            for x in range(G):
                _c32.set_pixel((x, y), (int(fy[x]), int(oy[x]), 0))  # location is a (x,y) TUPLE; color=(R=OFF,G=ON,B)
        _disp.draw_image(_c32, 0, 0, x_scale=UP, y_scale=UP)        # blocky upscale = the true 32x32 grid
        _disp.flush()
        both = 100.0 * int(np.sum(np.array((on > 0) * (off > 0), dtype=np.float))) / (G * G)   # yellow pixels
        occ = 100.0 * int(np.sum(np.array((on + off) > 0, dtype=np.float))) / (G * G)
        print('ev%d occ%d%% both/yellow%d%% fps%.1f' % (tot, int(occ), int(both), clock.fps()))
    else:
        print('no events  fps%.1f' % clock.fps())
