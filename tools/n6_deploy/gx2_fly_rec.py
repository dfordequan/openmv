# gx2_fly_rec.py -- FLY (predict + send UART action) AND RECORD at the same time. Same deploy config
# as gx2_main_f.py (float-IO bin, direct float carry, a0 low-pass, belief reset on first UART).
# Records a fixed MISSION_S-second window that STARTS on the first UART goal (mission start); only
# those frames are logged (belief is reset fresh at that instant). Also logs the per-frame loop dt so
# the real update frequency is in the data.
#
# Format: b'GX2FLY02' | u16 G | u16 pad | u16 VLEN | u16 flags   (N not fixed -> parse by file size)
#   per frame: u32 t_ms | u16 dt_ms | G*G u8 obs | VLEN f32 goal | 2 f32 raw_action | 2 f32 sent_action
#   (= 4+2+1024+16+8+8 = 1062 B fixed)
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz_1ms_real.bin'   # real-data-calibrated bin (matches gx2_main_f)
LOGPATH = '/sdcard/gx2fly.bin'
# NOTE: left UNCAPPED (no A1_MAX) on purpose -- this recording is to CONFIRM the speed<->obs-density
# hypothesis, so we want to capture the full speed range incl. any race regime. The deploy
# gx2_main_f.py has the A1_MAX speed cap; this diagnostic does not.
G, SRC = 32, 320
UART_ID, BAUD = 4, 115200
PERIOD = 70
GC_EVERY = 30                            # collect() every N frames (NOT per-frame mem_free poll ~200ms!)
SC = G / SRC
VLEN = 4
MISSION_S = 10                           # record this many seconds AFTER the first UART
KMAX = 4000                              # safety cap (stops if UART never arrives)
A_LP = 0.15

model = ml.Model(MODEL)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

def build_img_bytes():
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    b = _g.bytearray()
    img = np.array(np.frombuffer(b, dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1))
    return img, b

deter = np.zeros((1, 2048)); stoch = np.zeros((1, 16, 32)); prevact = np.zeros((1, 2))
a0f = 0.0; a1f = 0.0
goal_vec = [1.0, 0.0, 0.0, 0.5]
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
            goal_vec = [vals[0] / 1e4, vals[1] / 1e4, vals[2] / 1e4, vals[3] / 1e4]
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))
            _rx_seen = True
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

f = open(LOGPATH, 'wb')
f.write(b'GX2FLY02' + struct.pack('<HHHH', G, 0, VLEN, 0))
print('gx2_fly_rec [FLOAT-IO, LP, reset] armed -> waiting for first UART, then logging %ds' % MISSION_S)
gc.collect()
k = 0; nlog = 0; _hb = time.ticks_ms(); _started = False
t0 = time.ticks_ms(); t_mission = None; last_tick = t0
while k < KMAX:
    tick = time.ticks_ms()
    dt = time.ticks_diff(tick, last_tick); last_tick = tick
    if k % GC_EVERY == 0:                              # periodic collect, NOT per-frame mem_free walk
        gc.collect()
    poll_vector()
    if not _started:                                  # hold belief fresh until mission start
        deter = np.zeros((1, 2048)); stoch = np.zeros((1, 16, 32)); prevact = np.zeros((1, 2))
        a0f = 0.0; a1f = 0.0
        if _rx_seen:
            _started = True; t_mission = tick
            print('>>> first UART RX -> belief RESET, recording %ds window START' % MISSION_S)
    img, obs_b = build_img_bytes()
    out = model.predict([img, vector, deter, stoch, prevact])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    a0f = A_LP * a0 + (1 - A_LP) * a0f
    a1f = A_LP * a1 + (1 - A_LP) * a1f
    send_action(a0f, a1f)
    if _started:                                      # log ONLY the mission window
        f.write(struct.pack('<IH', time.ticks_diff(tick, t_mission), dt & 0xFFFF))
        f.write(bytes(obs_b))
        f.write(struct.pack('<%df' % VLEN, *goal_vec))
        f.write(struct.pack('<4f', a0, a1, a0f, a1f))
        nlog += 1
    deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32)); prevact = out[0]
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d st%d dt%dms a0 %+.2f->%+.2f brg(%.2f,%.2f) %.1fHz' % (k, _started, dt, a0, a0f, goal_vec[0], goal_vec[1], hz))
    if _started and time.ticks_diff(tick, t_mission) >= MISSION_S * 1000:
        break
f.close()
send_action(0.0, 0.0)
if nlog:
    print('DONE: logged %d frames over %ds -> %.1f Hz avg -> %s' % (nlog, MISSION_S, nlog / MISSION_S, LOGPATH))
else:
    print('DONE: no UART received (nothing logged). Check the CF is sending 0xCF frames.')
