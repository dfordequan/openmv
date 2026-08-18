# raw_dump.py -- what does the HISTO-firmware GRAYSCALE snapshot ACTUALLY contain?
import csi, image, time
from ulab import numpy as np

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE)
_c.framesize((320, 320))
_c.snapshot(time=800)

d = _c.snapshot()
S = d.to_ndarray('f')                       # (320,320) as float, values 0..255 (unsigned)
print('shape', S.shape, 'min %.0f max %.0f mean %.1f' % (float(np.min(S)), float(np.max(S)), float(np.mean(S))))

# value histogram over 0..255 in 16-wide bins
hist = [0] * 16
flat = S.reshape((320 * 320,))
for b in range(16):
    lo = b * 16; hi = lo + 16
    hist[b] = int(np.sum(np.array((flat >= lo) * (flat < hi), dtype=np.float)))
print('value histogram (bin=16 wide, 0..255):')
for b in range(16):
    print('  [%3d-%3d) %6d  %s' % (b * 16, b * 16 + 16, hist[b], '#' * (hist[b] // 800)))

# sample an 8x8 patch of raw pixel values from the centre
print('centre 8x8 patch (raw uint):')
u = d.to_ndarray('b')                        # signed read too, to compare
for r in range(156, 164):
    row = []
    for cx in range(156, 164):
        row.append(int(S[r][cx]))
    print('  ', row)

# how many exactly-0, exactly-128, exactly-255
print('exact 0: %d   ~128(120-136): %d   exact255: %d   /102400' % (
    int(np.sum(np.array(flat < 1, dtype=np.float))),
    int(np.sum(np.array((flat > 119) * (flat < 137), dtype=np.float))),
    int(np.sum(np.array(flat > 254, dtype=np.float)))))
