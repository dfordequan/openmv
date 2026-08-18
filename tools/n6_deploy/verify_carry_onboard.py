# VERIFY the manual-recurrency carry on the N6 -- resolves deploy findings C, D, B.
#   Run:  mpremote connect /dev/ttyACM0 run verify_carry_onboard.py
#
# Pairs with the host reference deploy/carry_golden_host.py (which prints A_ref for the SAME synthetic
# input sequence, fp32, soft=True, symmetric carry). Here we run that identical sequence through the
# COMPILED model under BOTH carry relayouts and print each action trajectory:
#   - the trajectory that matches A_ref  => that relayout is CORRECT           (finding C)
#   - model.output_dtype printed                                              (finding D: OUT_INT)
#   - stoch output softmax(fractional, rows~1) vs one-hot(0/1)                (finding B confirm)
import ml
from ulab import numpy as np

MODEL = '/sdcard/forest13.bin'
G, T = 64, 16
m = ml.Model(MODEL)
print('in_shape ', m.input_shape, ' out_shape', m.output_shape)
print('in_dtype ', m.input_dtype, ' out_dtype', m.output_dtype)
IN_SC, IN_ZP = m.input_scale, m.input_zero_point
OUT_SC, OUT_ZP = m.output_scale, m.output_zero_point
OUT_INT = ('b' in m.output_dtype[1])
print('D) OUT_INT (state outputs int8?) =', OUT_INT)


def synth_img(t):                       # MUST match deploy/carry_golden_host.py exactly
    img = np.zeros((1, G, G, 2), dtype=np.uint8)
    c0 = 6 + 3 * t
    for r in range(12, 52):
        for c in range(c0, c0 + 4):
            img[0, r, c, 0] = 180
    return img


def q_vec(v3):
    z = np.floor(np.array(v3, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, 3))


def requant(q, so, zo, si, zi):
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)


VEC = q_vec((1.0, 0.0, 0.0))
_checked_B = [False]


def dequant_stoch(out2):                # dequant the (32,16) stoch output to real for the B check
    if OUT_INT:
        return (np.array(out2, dtype=np.float) - OUT_ZP[2]) * OUT_SC[2]
    return np.array(out2, dtype=np.float)


def run(mode):
    deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
    stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
    prevact = np.zeros((1, 2))
    print('--- relayout: %s ---' % mode)
    for t in range(T):
        out = m.predict([synth_img(t), VEC, deter, stoch, prevact])
        a = out[0]
        print('t%02d  [% .4f, % .4f]' % (t, float(a[0][0]), float(a[0][1])))
        if not _checked_B[0]:           # finding B: is the stoch output softmax or one-hot?
            sm = dequant_stoch(out[2]).reshape((32, 16))
            row = sm[0]
            frac = sum(1 for x in row if 0.02 < float(x) < 0.98)
            print('B) stoch row0 sum=%.3f  #fractional(0.02..0.98)=%d/16  => %s'
                  % (float(np.sum(row)), frac, 'SOFTMAX(soft build)' if frac >= 3 else 'ONE-HOT(hard)'))
            _checked_B[0] = True
        deter = requant(out[1], OUT_SC[1], OUT_ZP[1], IN_SC[2], IN_ZP[2])
        s = out[2].reshape((32, 16))
        if mode == 'transpose':
            s = np.array(s.transpose()).reshape((1, 16, 32))
        else:
            s = np.array(s).reshape((1, 16, 32))
        stoch = requant(s, OUT_SC[2], OUT_ZP[2], IN_SC[3], IN_ZP[3])
        prevact = np.array(a, dtype=np.float)


run('transpose')
run('reshape')
print('C) compare BOTH trajectories to A_ref from deploy/carry_golden_host.py;')
print('   the matching one is the correct carry relayout.')
