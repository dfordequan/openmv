# actor_dummy_pipeline.py -- END-TO-END FPS of the reactive PPO actor on the N6: the FULL obs
# pipeline (snapshot -> AREA downsample -> uint8 [1,32,32,1]) + predict, timed per step. No UART,
# no recurrency, no PERIOD pacing -> runs flat-out so you see the true achievable rate.
#
# Contrast with actor_dummy_fps.py, which timed predict() ALONE (6.9 ms) to compare vs the RSSM's
# 40 ms. THIS one adds the camera cost back so you get the real deploy FPS. Dummy weights -> the
# actions are meaningless; only the timing is real.
#
# NOTE: the int8 actor takes the uint8 image DIRECTLY ([1,32,32,1]); the (x-128)/32 affine lives
# INSIDE the graph, so there is NO float-conversion step here (unlike the float-IO RSSM main).
import csi, image, time, ml, gc
from ulab import numpy as np

MODEL = '/sdcard/actor_dummy_32x1.bin'
G, SRC = 32, 320
SC = G / SRC
GC_EVERY = 30
us = time.ticks_us; di = time.ticks_diff

# ---- load model first, then size the framebuffer around it ----
model = ml.Model(MODEL)
print('model loaded. input_shape:', model.input_shape)
try:
    Z = model.input_zero_point
except Exception:
    Z = [0, 0]

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.framebuffers(1)
_c.snapshot(time=800)                      # settle
_g = image.Image(G, G, image.GRAYSCALE)

# dummy goal vector (int8 + zero-point), constant -- irrelevant with dummy weights
vec = np.zeros((1, 4), dtype=np.int8) + Z[1]

_t_snap = _t_down = _t_reshape = 0
def build_img():
    global _t_snap, _t_down, _t_reshape
    _t = us(); d = _c.snapshot(); _t_snap = di(us(), _t)                       # 320x320 graded net
    _t = us(); _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA); _t_down = di(us(), _t)
    _t = us()
    img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))  # uint8 straight in
    _t_reshape = di(us(), _t)
    return img

print('actor_dummy_pipeline up -- END-TO-END (snapshot+downsample+predict), no pacing')
gc.collect()
k = 0; _hb = time.ticks_ms(); _acc = 0
while True:
    _t = us()
    if k % GC_EVERY == 0:
        gc.collect()
    t_gc = di(us(), _t)

    img = build_img()                                                          # sets _t_snap/_t_down/_t_reshape

    _t = us(); out = model.predict([img, vec]); t_pred = di(us(), _t)

    work = t_gc + _t_snap + _t_down + _t_reshape + t_pred
    _acc += work; k += 1
    print('--- frame %d   END-TO-END %d us -> %.1f Hz ---' % (k, work, 1e6 / max(work, 1)))
    print('  [0] gc          %6d us' % t_gc)
    print('  [1] snapshot    %6d us   (sensor integration + readout)' % _t_snap)
    print('  [2] downsample  %6d us   (AREA 320->32)' % _t_down)
    print('  [3] reshape     %6d us   (uint8 [1,32,32,1])' % _t_reshape)
    print('  [4] predict     %6d us   (NPU inference)' % t_pred)

    if k % 15 == 0:
        avg = _acc / 15; _acc = 0
        print('==> 15-frame avg %.2f ms -> %.1f Hz end-to-end   free%d' % (avg / 1000.0, 1e6 / avg, gc.mem_free()))
