# alloc_test3.py -- v3: kill the 66KB //10 by drawing the histogram at 320 and DOWNSCALING (C, no alloc),
# plus in-place math + in-place clamp. Measure bytes/frame. Goal <~15KB -> gc once per ~2min.
import csi, image, gc
from ulab import numpy as np

BUF, G = 8192, 32
GAIN = 0.07 * 100 * 32
NOISE_FLOOR = 5
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)

# preallocated ONCE
_sig3 = image.Image(320, 320, image.GRAYSCALE)   # net @320
_act3 = image.Image(320, 320, image.GRAYSCALE)   # total @320
_sig2 = image.Image(G, G, image.GRAYSCALE)        # downscaled net
_act2 = image.Image(G, G, image.GRAYSCALE)        # downscaled total
_net = np.zeros((G, G)); _onb = np.zeros((G, G)); _offb = np.zeros((G, G))
_frame = np.zeros((1, G, G, 2), dtype=np.uint8)
SC = G / 320.0

def build_v3():
    global _net, _onb, _offb
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
    if n < 1:
        return _frame
    ev = _ev[:n]                                                        # view (~free)
    _sig3.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)   # net @320 (NO //10)
    _sig2.draw_image(_sig3, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)   # downscale 320->32 (C)
    ev[:, 0] = 1                                                        # all ON
    _act3.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)     # total @320
    _act2.draw_image(_act3, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    _net[:] = _sig2.to_ndarray('f'); _net -= 128.0
    _onb[:] = _act2.to_ndarray('f')
    _offb[:] = _onb
    _onb += _net; _onb *= 0.5; _onb -= NOISE_FLOOR; _onb *= GAIN
    _offb -= _net; _offb *= 0.5; _offb -= NOISE_FLOOR; _offb *= GAIN
    _onb[:] = np.clip(_onb, 0.0, 255.0)                                 # clip works on 2D (~9KB temp)
    _offb[:] = np.clip(_offb, 0.0, 255.0)
    _frame[0, :, :, 0] = _offb
    _frame[0, :, :, 1] = _onb
    return _frame

def measure(fn, name, warm=3, N=30):
    try:
        for _ in range(warm):
            fn()
    except Exception as e:
        print('%s FAILED -> %s' % (name, e)); return
    gc.collect(); f0 = gc.mem_free()
    for _ in range(N):
        fn()
    used = (f0 - gc.mem_free()) / N
    print('%-12s %6.0f bytes/frame -> gc every ~%d frames (~%.0f s @10Hz)' %
          (name, used, (gc.mem_free() - 3000000) / max(used, 1), (gc.mem_free() - 3000000) / max(used, 1) / 10))

print('measuring v3 (move camera)...')
measure(build_v3, 'v3 320+area')
# does float->uint8 assignment CLAMP or wrap? (if clamp, we can drop the explicit clamp lines -> less alloc)
_t = np.zeros((2,), dtype=np.uint8); _f = np.zeros((2,))
_f[0] = 300.0; _f[1] = -5.0; _t[:] = _f
print('float->uint8 cast of [300,-5] =', list(_t), '(255,0 = clamps; else wraps)')
