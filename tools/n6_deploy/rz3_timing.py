# bounded rz3 loop timing (no forever loop -> mpremote run returns clean output). 30 real iterations.
import csi, image, time, ml, gc
from ulab import numpy as np
BUF, G, GAIN = 16384, 32, 0.07 * 100 * 32
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)
m = ml.Model('/sdcard/ratezone3.bin')
IN_SC, IN_ZP = m.input_scale, m.input_zero_point
OUT_SC, OUT_ZP = m.output_scale, m.output_zero_point
OUT_INT = ('b' in m.output_dtype[1])
def rq(q, so, zo, si, zi):
    r = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(r / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)
def build():
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
    if n < 1: return np.zeros((1, G, G, 2), dtype=np.uint8), 0, 0
    tb = time.ticks_ms()
    _ev[:n, 4] = _ev[:n, 4] // 10; _ev[:n, 5] = _ev[:n, 5] // 10
    _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
    _ev[:n, 0] = 1
    _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5, 0.0) * GAIN, 255.0)
    fr = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(fr, dtype=np.uint8), n, time.ticks_diff(time.ticks_ms(), tb)
vec = np.zeros((1, 4), dtype=np.int8) + IN_ZP[1]
d = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]; s = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]; pa = np.zeros((1, 2))
gc.collect()
build(); m.predict([np.zeros((1, G, G, 2), dtype=np.uint8), vec, d, s, pa])   # warm
tb_sum = tp_sum = 0; t0 = time.ticks_ms()
for i in range(30):
    img, n, tb = build()
    ta = time.ticks_ms()
    out = m.predict([img, vec, d, s, pa])
    tp_sum += time.ticks_diff(time.ticks_ms(), ta); tb_sum += tb
    d = out[1]                                        # scales match -> no requant
    s = np.array(out[2]).reshape((1, 16, 32))
    pa = np.array(out[0], dtype=np.float)
dt = time.ticks_diff(time.ticks_ms(), t0) / 30.0
print('rz3 FULL loop: %.1f ms = %.1f Hz  (build %.1f + infer %.1f + carry/rest)  free %d' %
      (dt, 1000.0 / dt, tb_sum / 30.0, tp_sum / 30.0, gc.mem_free()))
