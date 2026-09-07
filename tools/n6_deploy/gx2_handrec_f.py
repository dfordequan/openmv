# gx2_handrec_f.py -- hand-record with the FLOAT-I/O model (gx2_15hz_f.bin). All model I/O is float32
# now, so the carry feeds back DIRECTLY (deter=out[1], stoch=reshape(out[2])) with NO quantization --
# this is the fix for the broken int8-in/float-out carry. Fixed 1052 B frames (obs logged as uint8).
import csi, image, ml, gc, time, struct
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz_f.bin'
LOGPATH = '/sdcard/gx2hand_f.bin'
G, SRC = 32, 320
SC = G / SRC
N = 300
PERIOD = 70
GOAL = [1.0, 0.0, 0.0, 0.5]      # dummy straight-ahead goal
VLEN = len(GOAL)

gc.collect()
m = ml.Model(MODEL)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

vector = np.array(GOAL, dtype=np.float).reshape((1, VLEN))    # FLOAT vector (no quant)
deter = np.zeros((1, 2048))                                   # FLOAT carry
stoch = np.zeros((1, 16, 32))
prevact = np.zeros((1, 2))

f = open(LOGPATH, 'wb')
f.write(b'GX2HR001' + struct.pack('<HHHH', G, N, VLEN, 0))
print('HAND-RECORD [FLOAT-IO model, direct carry] %d frames -> %s' % (N, LOGPATH))
print('>>> MOVE the board by hand to create events <<<')
t0 = time.ticks_ms()
for k in range(N):
    tick = time.ticks_ms()
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    obs_u8 = _g.bytearray()                                   # 1024 uint8 (logged)
    img = np.array(np.frombuffer(obs_u8, dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1))  # FLOAT image
    out = m.predict([img, vector, deter, stoch, prevact])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    f.write(struct.pack('<I', time.ticks_diff(tick, t0)))
    f.write(bytes(obs_u8))
    f.write(struct.pack('<%df' % VLEN, *GOAL))
    f.write(struct.pack('<2f', a0, a1))
    deter = out[1]                                            # DIRECT float feedback (the fix)
    stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = out[0]
    if k % 20 == 0:
        b = np.frombuffer(obs_u8, dtype=np.uint8)
        act = 100.0 * np.sum(np.array((b < 123) + (b > 133), dtype=np.float)) / (G * G)
        print('  k%d  a[%+.2f,%+.2f]  events%%%.0f  free%d' % (k, a0, a1, act, gc.mem_free()))
    if gc.mem_free() < 2000000:
        gc.collect()
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
f.close()
print('DONE: %d frames -> %s. Pull it and I verify board a0 now tracks fp32.' % (N, LOGPATH))
