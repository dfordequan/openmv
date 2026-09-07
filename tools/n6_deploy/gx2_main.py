# gx2_main.py -- deploy eval_GX2_15hz_3ms (task yawzoneh32r7w20gxv3): SINGLE-CHANNEL GenX320
# DIFF3D graded-net obs via csi.snapshot() (NOT read_events). Needs the STOCK firmware (graded
# USAT(net*16+128) snapshot -- reflash openmv/build/OPENMV_N6/bin/firmware.bin). No ulab histogram
# math, no per-frame garbage, no gc hitch, no backlog lag.
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz.bin'
G, SRC = 32, 320
UART_ID, BAUD = 4, 115200
YAWRATE_MAX = 3.5
PERIOD = 70                              # r7 == 100Hz/7 ~= 14.3 Hz control
SC = G / SRC

# ---- load the model FIRST (6.3 MB), so the framebuffer sizes around it ----
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point
OUT_INT = ('b' in model.output_dtype[1])
VLEN = model.input_shape[1][1]

# ---- camera: HISTO/DIFF3D grayscale SNAPSHOT (single-channel graded net) ----
# framebuffers(1): the 320x320 histo needs the WHOLE fb pool as one buffer. With the default
# multi-buffering, each buffer = pool/N < 320*320 once the model is loaded -> "Frame buffer overflow".
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.framebuffers(1)
_c.snapshot(time=800)                    # settle
_g = image.Image(G, G, image.GRAYSCALE)

def build_img():
    d = _c.snapshot()                                                 # 320x320 graded net (128-centred)
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)    # faithful 10x10 average -> 32x32
    # graded net is already the model's input (uint8, 128-centred) -> [1,32,32,1], no GAIN/ON/OFF math
    return np.frombuffer(_g.bytearray(), dtype=np.uint8).reshape((1, G, G, 1)), 0

def requant(q, so, zo, si, zi):          # kept as a fallback if deter/stoch in/out scales differ
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)

def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))

deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
goal_vec = [1.0, 0.0, 0.0, 0.5]          # [cos(bearing), sin(bearing), yawrate/3.5, v/6.0]
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
            vals = struct.unpack('<hhhh', body)
            goal_vec = [vals[0] / 10000.0, vals[1] / 10000.0, vals[2] / 10000.0, vals[3] / 10000.0]
            vector = q_vec(goal_vec)
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

print('gx2 loop up. model in', model.input_shape, '-> UART%d @ %d ~15Hz' % (UART_ID, BAUD))
gc.collect(); GC_FLOOR = 1000000
k = 0; _hb = time.ticks_ms()
while True:
    tick = time.ticks_ms()
    if gc.mem_free() < GC_FLOOR:
        gc.collect()
    poll_vector()
    img, tot = build_img()
    out = model.predict([img, vector, deter, stoch, prevact])
    action = out[0]; a0 = float(action[0][0]); a1 = float(action[0][1])
    send_action(a0, a1)
    deter = out[1]
    stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = np.array(action, dtype=np.float)
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a[%.2f,%.2f] %.1fHz free%d' % (k, a0, a1, hz, gc.mem_free()))
