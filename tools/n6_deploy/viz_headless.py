# viz_headless.py -- FAITHFUL behavior probe: EVENT mode + read_events (bit-identical input to main.py),
# no display, prints the predicted heading every tick. Use this to judge behavior without the
# grayscale-frame approximation viz_deploy.py uses for the IDE picture.  Needs /sdcard/forest13.bin.
import csi, image, time, ml, math, gc
from ulab import numpy as np

MODEL = '/sdcard/forest13.bin'
BUF, WIN_MS, G = 16384, 20, 64
CLIP = 1.0
GOAL = (1.0, 0.0)

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(320, 320, image.GRAYSCALE); _act = image.Image(320, 320, image.GRAYSCALE)
_s64 = image.Image(G, G, image.GRAYSCALE); _a64 = image.Image(G, G, image.GRAYSCALE)


def build_img():                        # SAME as main.py: read_events -> [1,64,64,2] uint8, + count
    first = True; t0 = time.ticks_ms(); tot = 0
    while time.ticks_diff(time.ticks_ms(), t0) < WIN_MS:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            continue
        _sig.draw_event_histogram(_ev[:n], clear=first, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=first, brightness=0, contrast=1)
        first = False; tot += n
    if tot == 0:
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0
    _s64.draw_image(_sig, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    _a64.draw_image(_act, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    S = _s64.to_ndarray('f'); A = _a64.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5, 0.0) / CLIP, 1.0)
    off = np.minimum(np.maximum((A - net) * 0.5, 0.0) / CLIP, 1.0)
    frame = np.concatenate((on.reshape((1, G, G, 1)), off.reshape((1, G, G, 1))), axis=3)
    return np.array(frame * 255, dtype=np.uint8), tot


model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point
OUT_INT = ('b' in model.output_dtype[1])


def requant(q, so, zo, si, zi):
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)


VLEN = model.input_shape[1][1]          # vector length (forest13=3; ratezone=4)
def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))


deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
vector = q_vec([GOAL[0], GOAL[1]] + [0.0] * (VLEN - 2))

# The raw action is bang-bang jitter -- DON'T read it per-tick. What steers the drone is the RATE-LIMITED
# yaw (step_steer: yaw_rate = clip(KYAW*rel, +-YMAX)); an EMA of it = the smoothed steering tendency.
# INTERPRETABLE CHECK: point at clutter on one side and watch whether yaw_ema leans AWAY from the busy side
# (busy RIGHT [R>L] -> should steer LEFT -> yaw_ema > 0).  KYAW/YMAX from the forest course.
KYAW, YMAX, ALPHA = 2.5, 3.5, 0.12
yaw_ema = 0.0
gc.collect(); GC_FLOOR = gc.mem_free() // 4   # gc.collect() ~940ms on the N6 -> only when heap is low
print('read_events. cols: rel=raw action angle | yawEMA=smoothed yaw deg/s (+=LEFT) | L/R event mass')
while True:
    if gc.mem_free() < GC_FLOOR: gc.collect()   # cheap check; rare collect
    img, tot = build_img()
    L = int(np.sum(np.array(img[0, :, :G // 2, :], dtype=np.float)))
    R = int(np.sum(np.array(img[0, :, G // 2:, :], dtype=np.float)))
    out = model.predict([img, vector, deter, stoch, prevact])
    a = out[0]; a0 = float(a[0][0]); a1 = float(a[0][1])
    rel = math.atan2(a1, a0)                             # radians
    yaw_rate = max(-YMAX, min(YMAX, KYAW * rel))         # same law as step_steer (rate-limited)
    yaw_ema = (1 - ALPHA) * yaw_ema + ALPHA * yaw_rate
    deter = requant(out[1], OUT_SC[1], OUT_ZP[1], IN_SC[2], IN_ZP[2])
    s_t = np.array(out[2]).reshape((1, 16, 32))          # PLAIN reshape (fixed)
    stoch = requant(s_t, OUT_SC[2], OUT_ZP[2], IN_SC[3], IN_ZP[3])
    prevact = np.array(a, dtype=np.float)
    side = 'L>R' if L > R * 1.2 else ('R>L' if R > L * 1.2 else ' ~ ')
    lean = 'LEFT ' if yaw_ema > 0.3 else ('RIGHT' if yaw_ema < -0.3 else '  .  ')
    print('rel %+6.1f  yawEMA %+5.2f %s  ev %5d  L%6d R%6d %s'
          % (math.degrees(rel), yaw_ema, lean, tot, L, R, side))
