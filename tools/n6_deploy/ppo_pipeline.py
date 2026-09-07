# ppo_pipeline.py -- END-TO-END operation FPS for PPO h64: real camera + 3-frame stack, no UART.
# v2: PRE-ALLOCATED persistent image buffer + in-place channel shift (no 49KB/frame alloc) -> avoids
# the heap fragmentation that crashed v1. Frame stack: shift channels down, newest -> last channel.
import csi, image, time, ml, gc
from ulab import numpy as np
MODEL = '/sdcard/ppo_h64.bin'; G, SRC = 64, 320; SC = G / SRC; KSTACK = 3; GC_EVERY = 15
us = time.ticks_us; di = time.ticks_diff

model = ml.Model(MODEL); print('input_shape:', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)
vector = np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float).reshape((1, 4))

_img = np.zeros((1, G, G, KSTACK))     # PERSISTENT -- filled in place every frame (no per-frame alloc)
_first = True; _t_snap = _t_down = _t_stack = 0
def build_img():
    global _first, _t_snap, _t_down, _t_stack
    _t = us(); d = _c.snapshot(); _t_snap = di(us(), _t)
    _t = us(); _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA); _t_down = di(us(), _t)
    _t = us()
    newf = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((G, G))
    if _first:
        for i in range(KSTACK): _img[0, :, :, i] = newf
        _first = False
    else:
        for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]   # shift down
        _img[0, :, :, KSTACK - 1] = newf                                       # newest last
    _t_stack = di(us(), _t)
    return _img

print('ppo_pipeline v2 END-TO-END (persistent buffer), flat-out')
gc.collect(); k = 0; _acc = 0
while True:
    _t = us()
    if k % GC_EVERY == 0: gc.collect()
    t_gc = di(us(), _t)
    img = build_img()
    _t = us(); out = model.predict([img, vector]); t_pred = di(us(), _t)
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    work = t_gc + _t_snap + _t_down + _t_stack + t_pred
    _acc += work; k += 1
    print('--- f%d  %d us -> %.1f Hz | snap %d down %d stack %d predict %d gc %d | a[%+.2f,%+.2f]' % (
        k, work, 1e6/max(work,1), _t_snap, _t_down, _t_stack, t_pred, t_gc, a0, a1))
    if k % 15 == 0:
        avg = _acc/15; _acc = 0
        print('==> 15-frame avg %.1f ms -> %.1f Hz end-to-end  free%d' % (avg/1000.0, 1e6/avg, gc.mem_free()))
