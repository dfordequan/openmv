# n6_gray_latency.py -- how fast can the policy run on the N6 with the GRAYSCALE (single-channel)
# obs? Measures three things and combines them:
#   (A) INFERENCE latency of the deployed models (2-ch, but the graph is arch-bound: a 1-ch model
#       differs only at conv0 -> sub-ms, so this IS the 1-ch inference number to within noise).
#   (B) the GRAYSCALE OBS pipeline: snapshot() + downsample 320->32 (BILINEAR vs AREA).
#   (C) implied full-loop rate = obs + inference.
import ml, gc, time, csi, image
from ulab import numpy as np

# ---------- (B) grayscale obs pipeline ----------
print('=== (B) GRAYSCALE OBS (snapshot + downsample 320->32) ===')
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((320, 320))
_c.snapshot(time=800)                       # settle
_g = image.Image(32, 32, image.GRAYSCALE)

def _t(fn, N=30):
    fn(); fn()                              # warm
    t0 = time.ticks_us()
    for _ in range(N):
        fn()
    return time.ticks_diff(time.ticks_us(), t0) / N / 1000.0

def snap():
    global _d
    _d = _c.snapshot()
def down_bilinear():
    _g.draw_image(_d, 0, 0, x_scale=32/320, y_scale=32/320, hint=image.BILINEAR)
def down_area():
    _g.draw_image(_d, 0, 0, x_scale=32/320, y_scale=32/320, hint=image.AREA)

t_snap = _t(snap)
t_bil = _t(down_bilinear)
t_area = _t(down_area)
print('snapshot        %.2f ms' % t_snap)
print('downsample BILINEAR %.2f ms' % t_bil)
print('downsample AREA     %.2f ms' % t_area)
obs_bil = t_snap + t_bil
obs_area = t_snap + t_area
print('-> obs total: BILINEAR %.2f ms | AREA %.2f ms' % (obs_bil, obs_area))
del _c; gc.collect()

# ---------- (A) inference latency ----------
print('')
print('=== (A) INFERENCE latency (deployed models) ===')
def bench(path, ch=2, N=30):
    m = ml.Model(path)
    Z = m.input_zero_point
    img = np.zeros((1, 32, 32, ch), dtype=np.uint8)
    vec = np.zeros((1, 4), dtype=np.int8) + Z[1]
    d = np.zeros((1, 2048), dtype=np.int8) + Z[2]
    s = np.zeros((1, 16, 32), dtype=np.int8) + Z[3]
    pa = np.zeros((1, 2))
    for _ in range(5):                      # warm + populate heap
        out = m.predict([img, vec, d, s, pa])
        d = out[1]; s = np.array(out[2]).reshape((1, 16, 32)); pa = np.array(out[0], dtype=np.float)
    gc.collect()
    t0 = time.ticks_us()
    for _ in range(N):
        out = m.predict([img, vec, d, s, pa])
        d = out[1]; s = np.array(out[2]).reshape((1, 16, 32)); pa = np.array(out[0], dtype=np.float)
    dt = time.ticks_diff(time.ticks_us(), t0) / N / 1000.0
    name = path.split('/')[-1]
    print('%-18s infer %.1f ms  (%.1f Hz)' % (name, dt, 1000.0 / dt))
    del m; gc.collect()
    return dt

t_inf = None
for p in ('/sdcard/ratezone3.bin', '/sdcard/yawzone10hz.bin'):
    try:
        t_inf = bench(p)
    except Exception as e:
        print('%s FAILED: %s' % (p, e))

# ---------- (C) full-loop rate ----------
if t_inf is not None:
    print('')
    print('=== (C) FULL LOOP (grayscale obs + inference) ===')
    loop_bil = obs_bil + t_inf
    loop_area = obs_area + t_inf
    print('BILINEAR obs: %.1f ms/loop -> %.1f Hz' % (loop_bil, 1000.0 / loop_bil))
    print('AREA     obs: %.1f ms/loop -> %.1f Hz' % (loop_area, 1000.0 / loop_area))
    print('(inference is arch-bound; a 1-ch model differs only at conv0 -> ~same as above)')
