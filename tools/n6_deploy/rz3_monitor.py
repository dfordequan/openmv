# rz3_monitor.py -- live view of ratezone3 prediction + loop rate (terminal or OpenMV IDE).
# Same pipeline as main.py (BUF=8192, drain-to-latest, real inference) but bounded + prints every tick.
# a0 = yaw_rate/3.5 (steer; -=right? see DEPLOYMENT frames), a1 = speed_cmd. occ vs sim ~26%.
import csi, image, time, ml, gc
from ulab import numpy as np

MODEL = '/sdcard/ratezone3.bin'
BUF, G, DRAIN_MAX, EV_CAP = 8192, 32, 3, 6000   # EV_CAP: freshest N events -> caps build time + occ
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
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0, d, 0.0
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5, 0.0) * GAIN, 255.0)
    fr = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    occ = 100.0 * int(np.sum(np.array((on + off) > 0, dtype=np.float))) / (G * G)
    return np.array(fr, dtype=np.uint8), n, d, occ

model = ml.Model(MODEL)
IN_ZP = model.input_zero_point
deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
# fixed dummy goal vector [cos,sin,yawrate/3.5,v/6] = straight ahead, mid speed
def q_vec(v):
    IN_SC = model.input_scale
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, model.input_shape[1][1]))
vector = q_vec([1.0, 0.0, 0.0, 0.5])

print('rz3 monitor: %d ticks. cols: tick | ev | drain | occ%% | a0(yaw) a1(spd) | ms | Hz' % TICKS)
gc.collect()
build(); model.predict([np.zeros((1, G, G, 2), dtype=np.uint8), vector, deter, stoch, prevact])  # warm
for k in range(TICKS):
    t0 = time.ticks_ms()
    img, n, d, occ = build()
    out = model.predict([img, vector, deter, stoch, prevact])
    a = out[0]; a0 = float(a[0][0]); a1 = float(a[0][1])
    deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32)); prevact = np.array(a, dtype=np.float)
    ms = time.ticks_diff(time.ticks_ms(), t0)
    if k % 3 == 0:
        print('t%3d ev%5d dr%d occ%4.0f  a[%+.2f %+.2f]  %3dms %4.1fHz' % (k, n, d, occ, a0, a1, ms, 1000.0 / max(ms, 1)))
print('done')
