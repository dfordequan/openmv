# motion_dump.py -- LEAN binary-vs-graded test WITH real events. WAVE HAND / MOVE BOARD during run.
#  intermediate values (16..240) appear -> GRADED histogram (B may live);  strictly 0/255 -> BINARY (B dead)
import csi, image, time
from ulab import numpy as np

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((320, 320)); _c.snapshot(time=600)

NF = 80
peak_lit = peak_inter = peak_max = 0
sum_inter = sum_sat = sum_lit = 0
for i in range(NF):
    S = _c.snapshot().to_ndarray('f').reshape((320 * 320,))
    lit = int(np.sum(np.array(S > 0, dtype=np.float)))
    inter = int(np.sum(np.array((S >= 16) * (S <= 240), dtype=np.float)))
    sat = int(np.sum(np.array(S >= 255, dtype=np.float)))
    mx = int(np.max(S))
    sum_lit += lit; sum_inter += inter; sum_sat += sat
    if lit > peak_lit: peak_lit = lit
    if inter > peak_inter: peak_inter = inter
    if mx > peak_max: peak_max = mx
    if i % 16 == 0:
        print('f%3d lit %5d inter %4d sat %4d max %3d' % (i, lit, inter, sat, mx))

print('\n=== peaks over %d frames ===' % NF)
print('peak lit %d (%.2f%%)  peak intermediate(16-240) %d  peak max-value %d' %
      (peak_lit, 100.0 * peak_lit / 102400.0, peak_inter, peak_max))
print('avg/frame: lit %.0f  intermediate %.0f  saturated(255) %.0f' %
      (sum_lit / NF, sum_inter / NF, sum_sat / NF))
print('VERDICT: intermediate avg %.0f -> %s' %
      (sum_inter / NF, 'GRADED (has counts/net, B may live)' if sum_inter / NF > 20 else 'BINARY-ish (mostly 0/255)'))
