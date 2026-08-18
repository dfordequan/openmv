# Single-step, drift-free resolver for the stoch carry relayout (finding C).
# deter_out = core(deter=0, stoch_in, prevact=0) is a PURE, unsaturated function of stoch_in (independent
# of the image). Feed a known one-hot pattern P_k two ways -- reshape vs transpose -- and see which makes
# the board's deter_out track the fp32 reference across patterns.
#   fp32 deter[3] across k: k0=0.053 k1=0.072 k2=0.082 k3=0.106 (monotonic up)  <- the correct relayout matches
import ml
from ulab import numpy as np
m = ml.Model('/sdcard/forest13.bin')
IN_SC, IN_ZP = m.input_scale, m.input_zero_point
OUT_SC, OUT_ZP = m.output_scale, m.output_zero_point
G = 64
img = np.zeros((1, G, G, 2), dtype=np.uint8)          # deter path ignores the image, but predict needs one
vec = np.zeros((1, 3), dtype=np.int8) + IN_ZP[1]
d0 = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]     # deter_in = 0
pa = np.zeros((1, 2))

def qstoch(a16x32):                                    # float (16,32) 0/1 -> int8 stoch input
    z = np.floor(a16x32 / IN_SC[3] + 0.5) + IN_ZP[3]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, 16, 32))

def pat(k):                                            # known one-hot (32,16): group g -> class (g*7+k*3)%16
    P = np.zeros((32, 16))
    for g in range(32):
        P[g, (g * 7 + k * 3) % 16] = 1.0
    return P

def deter_out(stoch_in):
    o = m.predict([img, vec, d0, stoch_in, pa])
    return ((np.array(o[1], dtype=np.float) - OUT_ZP[1]) * OUT_SC[1]).flatten()

print('OUT_SC[1]=%.5f (deter quantum)  IN_SC[3]=%.5f' % (OUT_SC[1], IN_SC[3]))
print('fp32 ref deter[3]:  k0=0.053 k1=0.072 k2=0.082 k3=0.106  (correct relayout tracks this trend)')
print('k   RESHAPE  [2]     [3]     [4]      TRANSPOSE [2]     [3]     [4]')
for k in range(4):
    P = pat(k)
    dr = deter_out(qstoch(P.reshape((16, 32))))            # reshape relayout
    dt = deter_out(qstoch(np.array(P.transpose())))        # transpose relayout -> (16,32)
    print('k%d   R  %+.3f %+.3f %+.3f     T  %+.3f %+.3f %+.3f'
          % (k, dr[2], dr[3], dr[4], dt[2], dt[3], dt[4]))
