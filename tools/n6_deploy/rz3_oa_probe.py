# rz3_oa_probe.py -- BENCH obstacle-avoidance sanity: does steering oppose the obstacle side?
# Hold the board still; move an object (or hand) mostly on ONE side of the lens.
# 'side' = (right_mass - left_mass)/(total) in [-1(all LEFT) .. +1(all RIGHT)].
# If OA works, a0 (yaw) should consistently push AWAY from 'side' (opposite sign).
import csi, image, time, ml, gc
from ulab import numpy as np
MODEL = '/sdcard/ratezone3.bin'
BUF, G, DRAIN_MAX, EV_CAP = 8192, 32, 3, 6000
GAIN = 0.07 * 100 * 32
TICKS = 150
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try: _c.framebuffers(3)
except Exception: pass
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)

def build():
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev); d = 0
    while n >= BUF and d < DRAIN_MAX:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev); d += 1
    if n < 1:
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0, 0.0, 0.0
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5, 0.0) * GAIN, 255.0)
    mass = on + off                                    # (32,32) activity
    L = float(np.sum(mass[:, :G // 2])); R = float(np.sum(mass[:, G // 2:]))
    tot = L + R
    side = (R - L) / tot if tot > 1 else 0.0           # -1 all LEFT .. +1 all RIGHT
    occ = 100.0 * int(np.sum(np.array(mass > 0, dtype=np.float))) / (G * G)
    fr = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(fr, dtype=np.uint8), occ, side, tot

model = ml.Model(MODEL); Z = model.input_zero_point
deter = np.zeros((1, 2048), dtype=np.int8) + Z[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
prevact = np.zeros((1, 2))
def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / model.input_scale[1] + 0.5) + Z[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, model.input_shape[1][1]))
vector = q_vec([1.0, 0.0, 0.0, 0.5])                   # goal straight ahead
print('OA probe: move object on ONE side. cols: tick occ%% side(-L/+R) | a0(yaw) a1(spd) | agree?')
gc.collect()
build(); model.predict([np.zeros((1, G, G, 2), dtype=np.uint8), vector, deter, stoch, prevact])
agree = 0; cnt = 0
for k in range(TICKS):
    img, occ, side, tot = build()
    out = model.predict([img, vector, deter, stoch, prevact])
    a = out[0]; a0 = float(a[0][0]); a1 = float(a[0][1])
    deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32)); prevact = np.array(a, dtype=np.float)
    tag = ''
    if abs(side) > 0.2 and occ > 6:                    # only judge when there IS a clear one-sided obstacle
        cnt += 1
        ok = (side > 0 and a0 < -0.1) or (side < 0 and a0 > 0.1)   # steer AWAY from side
        agree += 1 if ok else 0
        tag = 'AWAY' if ok else 'TOWARD'
    if k % 3 == 0:
        print('t%3d occ%3.0f side%+.2f  a[%+.2f %+.2f]  %s' % (k, occ, side, a0, a1, tag))
print('judged %d ticks; steered AWAY %d (%.0f%%) -- >70%% = OA polarity OK' %
      (cnt, agree, 100.0 * agree / max(cnt, 1)))
