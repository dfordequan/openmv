# gx7_main.py -- deploy Dreamer GX7 (dreamer_gx7, task yawzoneh32r10w20gxv1sc) on the N6.
# Same manual-recurrency float-IO scheme as gx2_main_f.py: image[1,32,32,1] + vector[4] +
# carry(deter[2048]+stoch[32x16]+prevact[2]); carry fed back directly as float32.
#
# DIFFERENCES vs gx2 (from the 2026-08-25 bundle README):
#   * 10 Hz decisions (act-repeat 10 over 100 Hz)         -> PERIOD = 100
#   * NO output low-pass and NO speed cap. Two experiments showed a low-pass COLLAPSES deterministic
#     performance -> do NOT add one. A_LP=1.0 (raw pass-through), A1_MAX=1.0 (no cap). Knobs kept ONLY
#     as escape hatches. IF the real drone shows yaw cancellation, the FIRST lever is NOT a filter --
#     it's the tau=0.08s yaw-rate lag being wrong on hardware: measure the CF yaw-rate STEP RESPONSE
#     and match tau (add the CF-side 1st-order filter if the loop is faster than 80ms). Only if that
#     is correct and it still cancels should you touch A_LP.
#   * CF-side speed map is vmin 0.5 / vmax 1.0 m/s (v1 profile) -- that's firmware, not here.
#   * yaw-rate lag tau=0.08 s is applied CF-side (a0*3.5 rad/s through a 1st-order lag).
# The soft-argmax stoch (temp=8) is baked into the bin; carry is float32 I/O.
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/gx7.bin'
G, SRC = 32, 320
UART_ID, BAUD = 4, 115200
PERIOD = 100                             # 10 Hz (act-repeat 10 @ 100 Hz control)
GC_EVERY = 200                           # periodic collect (heap ~24MB, ~35KB/frame churn)
SC = G / SRC
VLEN = 4
A_LP = 1.0                               # 1.0 = NO low-pass (README: do not filter gx7)
A1_MAX = 1.0                             # 1.0 = NO speed cap

model = ml.Model(MODEL)
_shp = model.input_shape; print('input_shape:', _shp)   # [image,vector,deter,stoch,prevaction]
S_DETER = tuple(_shp[2]); S_STOCH = tuple(_shp[3])       # match compiled layout (stoch may be (1,16,32))
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(2)  # snapshot 40->22ms (fb2 overlaps readout+integration, measured 2026-09-01)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

def build_img():
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    return np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1))

deter = np.zeros(S_DETER); stoch = np.zeros(S_STOCH); prevact = np.zeros((1, 2))
a0f = 0.0; a1f = 0.0
goal_vec = [1.0, 0.0, 0.0, 0.5]   # CF sends: [cos(brg), sin(brg), yaw_rate/3.5, |v_EKF|/vmax]
# NOTE: slot[3] is EKF HORIZONTAL SPEED (norm of vx,vy from the state estimate), NOT the last
# commanded speed. The CF must feed the state estimate here (confirmed vs training observe()).
vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))

uart = UART(UART_ID, BAUD)
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
_rx = bytearray(); _rx_seen = False
def poll_vector():
    global goal_vec, vector, _rx, _rx_seen
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
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))
            _rx_seen = True
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

print('gx7_main [Dreamer GX7, float-IO, 10Hz, no-filter] up. model in', model.input_shape)
gc.collect()
k = 0; _hb = time.ticks_ms(); _started = False
while True:
    tick = time.ticks_ms()
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    if not _started:                                  # hold belief fresh until first UART (mission start)
        deter = np.zeros(S_DETER); stoch = np.zeros(S_STOCH); prevact = np.zeros((1, 2))
        a0f = 0.0; a1f = 0.0
        if _rx_seen:
            _started = True
            print('>>> first UART RX -> belief RESET (mission start)')
    img = build_img()
    out = model.predict([img, vector, deter, stoch, prevact])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    a0f = A_LP * a0 + (1 - A_LP) * a0f                # A_LP=1.0 -> a0f == a0 (no filtering)
    a1f = A_LP * a1 + (1 - A_LP) * a1f
    if a1f > A1_MAX: a1f = A1_MAX
    send_action(a0f, a1f)
    deter = out[1]; stoch = np.array(out[2]).reshape(S_STOCH); prevact = out[0]
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 10 == 0:
        now = time.ticks_ms(); hz = 10000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz free%d' % (k, a0, a1, goal_vec[0], goal_vec[1], hz, gc.mem_free()))
