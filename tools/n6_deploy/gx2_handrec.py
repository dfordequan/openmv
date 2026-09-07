# gx2_handrec.py -- hand-move the board and record the REAL obs the model sees + the action it emits,
# with a fixed DUMMY goal bearing (no CF/UART needed). Fixed-size frames so the log parses cleanly
# (the earlier variable-size carry logging is dropped -- the carry is still fed back internally for
# the recurrence, just not written).
#
# Format: b'GX2HR001' | u16 G | u16 N | u16 VLEN | u16 pad
#   per frame: u32 t_ms | G*G u8 obs | VLEN f32 goal | 2 f32 action   (= 4+1024+16+8 = 1052 B)
import csi, image, ml, gc, time, struct
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
LOGPATH = '/sdcard/gx2hand_fix.bin'    # distinct name: this is the REQUANT-carry-fixed recording
G, SRC = 32, 320
SC = G / SRC
N = 300                          # frames (~1052 B each -> ~0.3 MB); stop early with Ctrl-C if needed
PERIOD = 70                      # ~14 Hz target (IDE preview will slow it; that's fine)
GOAL = [1.0, 0.0, 0.0, 0.5]      # DUMMY goal: cos=1,sin=0 (straight ahead), yawrate 0, v/6=0.5. edit me.

gc.collect()
m = ml.Model(MODEL)
IN_SC, IN_ZP = m.input_scale, m.input_zero_point
VLEN = m.input_shape[1][1]
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))

def q_carry(xf, si, zi, shape):    # requantize FLOAT out[1]/out[2] -> int8 for the int8 carry inputs
    z = np.floor(np.array(xf, dtype=np.float) / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape(shape)

vector = q_vec(GOAL)
deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))

f = open(LOGPATH, 'wb')
f.write(b'GX2HR001' + struct.pack('<HHHH', G, N, VLEN, 0))
print('HAND-RECORD [REQUANT CARRY FIX] %d frames, dummy goal %s -> %s' % (N, GOAL, LOGPATH))
print('>>> MOVE the board by hand (translate/rotate) to create events <<<')
t0 = time.ticks_ms()
for k in range(N):
    tick = time.ticks_ms()
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    img = np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))
    out = m.predict([img, vector, deter, stoch, prevact])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    # log: t, obs, goal, action  (fixed 1052 B)
    f.write(struct.pack('<I', time.ticks_diff(tick, t0)))
    f.write(bytes(img))
    f.write(struct.pack('<%df' % VLEN, *GOAL))
    f.write(struct.pack('<2f', a0, a1))
    # feed carry back -- REQUANTIZE float out[1]/out[2] to int8 (THE FIX)
    deter = q_carry(out[1], IN_SC[2], IN_ZP[2], (1, 2048))
    stoch = q_carry(out[2], IN_SC[3], IN_ZP[3], (1, 16, 32))
    prevact = np.array(out[0], dtype=np.float)
    if k % 20 == 0:
        b = np.frombuffer(_g.bytearray(), dtype=np.uint8)
        act = 100.0 * np.sum(np.array((b < 123) + (b > 133), dtype=np.float)) / (G * G)   # % pixels w/ event
        print('  k%d  a[%+.2f,%+.2f]  events%%%.0f  free%d' % (k, a0, a1, act, gc.mem_free()))
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
f.close()
print('DONE: %d frames -> %s.  Pull it and I parse/plot obs+action (fixed 1052 B frames).' % (N, LOGPATH))
