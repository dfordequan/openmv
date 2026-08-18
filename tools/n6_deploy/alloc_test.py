# alloc_test.py -- measure the per-frame heap allocation (the gc-forcing leak) of the current obs
# pipeline vs a preallocated/in-place refactor, and time gc.collect(). NO model, NO UART.
# Goal: drive bytes/frame low enough that gc ~never fires -> flyable continuous loop.
import csi, image, time, gc
from ulab import numpy as np

BUF, G, EV_CAP = 8192, 32, 6000
GAIN = 0.07 * 100 * 32
NOISE_FLOOR = 5
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)

# ---------- v1: the current pipeline (allocates a temp per ulab op) ----------
def build_v1():
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
    if n < 1:
        return 0
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    frame = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(frame, dtype=np.uint8)

# ---------- v2: preallocated buffers + in-place ops (reused every call) ----------
_net = np.zeros((G, G)); _onb = np.zeros((G, G)); _offb = np.zeros((G, G))
_frame = np.zeros((1, G, G, 2), dtype=np.uint8)
def build_v2():
    global _net, _onb, _offb                             # augmented assign (+=, *=) rebinds -> need global
    n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
    if n < 1:
        return _frame
    _ev[:n, 4] //= 10; _ev[:n, 5] //= 10                 # in-place floor-div (no temp)
    _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
    _ev[:n, 0] = 1
    _act.draw_event_histogram(_ev[:n], clear=True, brightness=0, contrast=1)
    _net[:] = _sig.to_ndarray('f'); _net -= 128.0        # net = sig-128
    _onb[:] = _act.to_ndarray('f')                       # onb = total
    _offb[:] = _onb                                      # offb = total
    _onb += _net; _onb *= 0.5; _onb -= NOISE_FLOOR; _onb *= GAIN     # ON pre-clip
    _offb -= _net; _offb *= 0.5; _offb -= NOISE_FLOOR; _offb *= GAIN  # OFF pre-clip
    _onb[:] = np.clip(_onb, 0.0, 255.0)                  # 1 temp
    _offb[:] = np.clip(_offb, 0.0, 255.0)                # 1 temp
    _frame[0, :, :, 0] = _offb                           # pack into reused uint8 output (cast)
    _frame[0, :, :, 1] = _onb
    return _frame

def measure(fn, name, warm=3, N=30):
    try:
        for _ in range(warm):
            fn()
    except Exception as e:
        print('%s: FAILED -> %s' % (name, e)); return
    gc.collect(); f0 = gc.mem_free()
    for _ in range(N):
        fn()
    used = (f0 - gc.mem_free()) / N
    frames_to_gc = (gc.mem_free() - 3000000) / max(used, 1)
    print('%-14s %6.0f bytes/frame -> gc every ~%d frames (~%.0f s @10Hz)' %
          (name, used, frames_to_gc, frames_to_gc / 10.0))

print('measuring (move the camera for realistic event counts)...')
measure(build_v1, 'v1 current')
measure(build_v2, 'v2 in-place')
t0 = time.ticks_ms(); gc.collect(); dt = time.ticks_diff(time.ticks_ms(), t0)
print('gc.collect() cost: %d ms   (this is the in-flight freeze)' % dt)
print('free', gc.mem_free())
