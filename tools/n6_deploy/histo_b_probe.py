# histo_b_probe.py -- OPTION B diagnostic for ratezone3 on the 20 Hz HISTO-snapshot firmware.
# Captures real frames, reconstructs ON/OFF by net-split (OFF=relu(-net), ON=relu(+net)),
# SUM-pools 320->32 per-channel (matches the sim proxy's per-channel .add), runs the model,
# and prints: per-stage timing + a GAIN sweep (occupancy% / saturation%) vs the sim targets.
# SIM TRAIN TARGETS (H=32): saturation ~10% of pixels pegged at 255; occupancy ~26% nonzero.
import csi, image, time, ml, gc
from ulab import numpy as np

MODEL = '/sdcard/ratezone3.bin'
G, W, H = 32, 320, 320
NF = 12                                   # frames to capture
GAIN_GUESS = 0.03                         # nominal gain used for the live inference pass
GAINS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.35, 0.6]   # sweep

# ---- sensor: GRAYSCALE 320x320 snapshot (128=no event, >128 ON, <128 OFF) ----
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE)
_c.framesize((W, H))
_c.snapshot(time=800)                     # settle

def pool(a):                              # (320,320) -> (32,32) SUM over each 10x10 block
    a = a.reshape((G, 10, W)); a = np.sum(a, axis=1)      # (32,320) sum 10 rows
    a = a.reshape((G, G, 10)); a = np.sum(a, axis=2)      # (32,32)  sum 10 cols
    return a

def build():                             # -> (on32, off32) pre-gain floats, + snap/build ms
    t0 = time.ticks_ms()
    disp = _c.snapshot()
    t1 = time.ticks_ms()
    S = disp.to_ndarray('f'); net = S - 128.0            # (320,320) signed
    on = pool(np.maximum(net, 0.0))                       # per-channel split BEFORE pooling
    off = pool(np.maximum(-net, 0.0))
    t2 = time.ticks_ms()
    return on, off, time.ticks_diff(t1, t0), time.ticks_diff(t2, t1)

# ---- model + quant ----
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
VLEN = model.input_shape[1][1]
def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))
deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
vector = q_vec([1.0, 0.0, 0.0, 0.5])      # goal ahead, mid speed

def to_frame(on, off, gain):              # (32,32)x2 floats -> [1,32,32,2] uint8 (ch0=OFF, ch1=ON)
    o = np.minimum(off * gain, 255.0); n = np.minimum(on * gain, 255.0)
    return np.concatenate((o.reshape((1, G, G, 1)), n.reshape((1, G, G, 1))), axis=3)

gc.collect()
ons = []; offs = []
tsnap = tbuild = tinf = 0
print('capturing %d frames...' % NF)
for i in range(NF):
    on, off, ts, tb = build()
    ons.append(on); offs.append(off); tsnap += ts; tbuild += tb
    img = np.array(to_frame(on, off, GAIN_GUESS), dtype=np.uint8)
    ta = time.ticks_ms()
    out = model.predict([img, vector, deter, stoch, prevact])
    tinf += time.ticks_diff(time.ticks_ms(), ta)
    a = out[0]; deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = np.array(a, dtype=np.float)
    print('  f%d a[%.2f,%.2f]' % (i, float(a[0][0]), float(a[0][1])))

N = float(NF * G * G * 2)                  # total elements over all frames (2 channels)
NP = float(NF * G * G)                     # total pixels (either channel)
print('\n--- TIMING (avg ms/frame) ---')
print('snapshot %.1f | build(split+pool) %.1f | infer %.1f | LOOP~ %.1f ms = %.1f Hz' %
      (tsnap / NF, tbuild / NF, tinf / NF,
       (tsnap + tbuild + tinf) / NF, 1000.0 / max((tsnap + tbuild + tinf) / NF, 1)))

print('\n--- GAIN SWEEP (sim target: sat~10%%, occ~26%%) ---')
print(' GAIN    occ_elem%%  occ_pix%%  sat%%   meanNZ')
for g in GAINS:
    occ = sat = nzsum = nzcnt = pixocc = 0
    for k in range(NF):
        o = np.minimum(offs[k] * g, 255.0); n = np.minimum(ons[k] * g, 255.0)
        ou = np.array(o, dtype=np.uint8); nu = np.array(n, dtype=np.uint8)
        occ += int(np.sum(np.array(ou > 0, dtype=np.float))) + int(np.sum(np.array(nu > 0, dtype=np.float)))
        sat += int(np.sum(np.array(ou >= 255, dtype=np.float))) + int(np.sum(np.array(nu >= 255, dtype=np.float)))
        pixocc += int(np.sum(np.array((ou + nu) > 0, dtype=np.float)))
        nzsum += float(np.sum(ou)) + float(np.sum(nu)); nzcnt += int(np.sum(np.array(ou > 0, dtype=np.float))) + int(np.sum(np.array(nu > 0, dtype=np.float)))
    mnz = nzsum / max(nzcnt, 1)
    print(' %.3f   %6.1f    %6.1f   %5.1f   %5.0f' %
          (g, 100.0 * occ / N, 100.0 * pixocc / NP, 100.0 * sat / N, mnz))
print('free', gc.mem_free())
