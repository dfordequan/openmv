# gx2_obslog.py -- run the gx2 policy on-board AND log the EXACT model input (obs + int8 carry +
# goal vector) and the board's OUTPUT action per frame. Replay it offline (fp32 / int8-onnxruntime)
# on the SAME inputs to separate three hypotheses for the left-circle bias (board a0 +0.45 vs
# fp32-on-my-recording -0.07): (1) faint flight frames, (2) the real goal vector, (3) Neural-ART int8.
# Logging the int8 carry per frame removes the recurrent carry-drift confound -> exact per-step check.
#
# It also SENDS the action over UART4 (so it flies / drives the CF if wired). For a bench test, just
# HAND-TRANSLATE the board at ~0.5 m/s to mimic the slow-flight visual regime that circled.
#
# Format: b'GX2LOG01' | u16 G | u16 DET | u16 STO | u16 VLEN | u16 N
#   per frame: u32 t_ms | G*G u8 obs | DET i8 deter_in | STO i8 stoch_in
#              | 2 f32 prevact | VLEN f32 goal_vec | 2 f32 action_out
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
LOGPATH = '/sdcard/gx2log.bin'
G, SRC = 32, 320
UART_ID, BAUD = 4, 115200
PERIOD = 70
N = 200                     # ~14 s at 14 Hz  (~200 * 3.6KB = ~0.7 MB)
SC = G / SRC

# ---- model first (framebuffer sizes around it), then camera with a single fb ----
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
VLEN = model.input_shape[1][1]
DET = 2048
STO = 16 * 32

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.framebuffers(1)
try: _c.set_contrast(1)                 # if available -> unsaturated; else default 16 (graded) is fine
except Exception: pass
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

def build_img():
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    return np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1))

def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))

deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
goal_vec = [1.0, 0.0, 0.0, 0.5]
vector = q_vec(goal_vec)

uart = UART(UART_ID, BAUD)
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
_rx = bytearray()
def poll_vector():
    global goal_vec, vector, _rx
    if uart.any():
        _rx.extend(uart.read(uart.any()))
    i = 0; n = len(_rx)
    while i + 11 <= n:
        if _rx[i] != 0xCF or _rx[i + 1] != 0x55:
            i += 1; continue
        body = _rx[i + 2:i + 10]; ck = 0
        for b in body:
            ck ^= b
        if (ck & 0xFF) == _rx[i + 10]:
            vv = struct.unpack('<hhhh', body)
            goal_vec = [vv[0] / 1e4, vv[1] / 1e4, vv[2] / 1e4, vv[3] / 1e4]
            vector = q_vec(goal_vec)
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

f = open(LOGPATH, 'wb')
f.write(b'GX2LOG01' + struct.pack('<HHHHH', G, DET, STO, VLEN, N))
print('LOGGING %d frames -> %s   (translate ~0.5 m/s to mimic the slow flight)' % (N, LOGPATH))
gc.collect()
t0 = time.ticks_ms()
for k in range(N):
    tick = time.ticks_ms()
    poll_vector()
    img = build_img()
    # --- log the INPUTS (exact tensors going into predict) ---
    f.write(struct.pack('<I', time.ticks_diff(tick, t0)))
    f.write(bytes(img))                              # G*G u8
    f.write(bytes(deter))                            # DET i8
    f.write(bytes(stoch))                            # STO i8
    f.write(struct.pack('<2f', float(prevact[0][0]), float(prevact[0][1])))
    f.write(struct.pack('<%df' % VLEN, *goal_vec))
    # --- run + log the OUTPUT ---
    out = model.predict([img, vector, deter, stoch, prevact])
    action = out[0]; a0 = float(action[0][0]); a1 = float(action[0][1])
    f.write(struct.pack('<2f', a0, a1))
    send_action(a0, a1)
    deter = out[1]
    stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = np.array(action, dtype=np.float)
    if k % 20 == 0:
        print('  k%d a[%+.2f,%+.2f] brg(cos%.2f sin%.2f) free%d' %
              (k, a0, a1, goal_vec[0], goal_vec[1], gc.mem_free()))
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
f.close()
send_action(0.0, 0.0)
print('DONE: %d frames -> %s.  Pull it and I replay fp32/int8 on the exact inputs.' % (N, LOGPATH))
