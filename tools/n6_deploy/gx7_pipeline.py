# gx7_pipeline.py -- END-TO-END operation FPS for Dreamer GX7: real camera obs + RSSM carry feedback,
# no UART (dummy goal), no PERIOD pacing (runs flat-out). Per-step timing = the real deploy loop cost.
import csi, image, time, ml, gc
from ulab import numpy as np
MODEL = '/sdcard/gx7.bin'; G, SRC = 32, 320; SC = G / SRC; GC_EVERY = 200
us = time.ticks_us; di = time.ticks_diff

model = ml.Model(MODEL)
shp = model.input_shape; print('input_shape:', shp)
S_DETER = tuple(shp[2]); S_STOCH = tuple(shp[3])
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

deter = np.zeros(S_DETER); stoch = np.zeros(S_STOCH); prevact = np.zeros((1, 2))
vector = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float).reshape((1, 4))

_t_snap = _t_down = _t_conv = 0
def build_img():
    global _t_snap, _t_down, _t_conv
    _t = us(); d = _c.snapshot(); _t_snap = di(us(), _t)
    _t = us(); _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA); _t_down = di(us(), _t)
    _t = us(); r = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1)); _t_conv = di(us(), _t)
    return r

print('gx7_pipeline END-TO-END (camera + carry), flat-out')
gc.collect(); k = 0; _acc = 0
while True:
    _t = us()
    if k % GC_EVERY == 0: gc.collect()
    t_gc = di(us(), _t)
    img = build_img()
    _t = us(); out = model.predict([img, vector, deter, stoch, prevact]); t_pred = di(us(), _t)
    _t = us(); deter = out[1]; stoch = np.array(out[2]).reshape(S_STOCH); prevact = out[0]; t_carry = di(us(), _t)
    work = t_gc + _t_snap + _t_down + _t_conv + t_pred + t_carry
    _acc += work; k += 1
    print('--- f%d  %d us -> %.1f Hz | snap %d down %d conv %d predict %d carry %d gc %d' % (
        k, work, 1e6/max(work,1), _t_snap, _t_down, _t_conv, t_pred, t_carry, t_gc))
    if k % 15 == 0:
        avg = _acc/15; _acc = 0
        print('==> 15-frame avg %.1f ms -> %.1f Hz end-to-end  free%d' % (avg/1000.0, 1e6/avg, gc.mem_free()))
