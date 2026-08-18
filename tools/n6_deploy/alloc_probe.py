# alloc_probe.py -- isolate WHICH operation allocates, so we know exactly what to eliminate.
import csi, image, gc
from ulab import numpy as np

BUF, G = 8192, 32
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE)
_S = np.zeros((G, G))

n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)          # one read; measure ops on these n events
print('n =', n, 'events  (each op repeated, bytes/call):')

def probe(name, fn, N=20):
    for _ in range(2):
        fn()
    gc.collect(); f0 = gc.mem_free()
    for _ in range(N):
        fn()
    print('  %-26s %7.0f B/call' % (name, (f0 - gc.mem_free()) / N))

def op_rowslice():
    x = _ev[:n]                                           # row prefix -> view or copy?
def op_colslice():
    x = _ev[:n, 4]                                        # column -> strided, likely copy
def op_colassign():
    _ev[:n, 4] = _ev[:n, 4] // 10                         # the //10 coord op
def op_colinplace():
    _ev[:n, 4] //= 10                                     # in-place variant
def op_tondarray():
    x = _sig.to_ndarray('f')                             # image -> ndarray
def op_ndcopy():
    _S[:] = _sig.to_ndarray('f')                         # to_ndarray + copy into preallocated
def op_draw():
    _sig.draw_event_histogram(_ev[:n], clear=True, brightness=128, contrast=1)
def op_clip():
    x = np.clip(_S, 0.0, 255.0)

probe('row slice _ev[:n]', op_rowslice)
probe('col slice _ev[:n,4]', op_colslice)
probe('col assign =//10', op_colassign)
probe('col //= 10', op_colinplace)
probe('to_ndarray', op_tondarray)
probe('to_ndarray->_S[:]', op_ndcopy)
probe('draw_event_histogram', op_draw)
probe('np.clip', op_clip)
