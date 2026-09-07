# ppo_headpen_fly_rec.py -- FLY (predict + send UART action) AND RECORD, for PPO HEADPEN (2026-08-27).
# Same deploy config as ppo_headpen_main.py (feedforward, 64x64x3, persistent buffer, no filtering,
# SLOW 1.0-1.5 m/s, yaw a0*1.0 rad/s). Records a fixed MISSION_S window from the first UART goal;
# logs per-frame single 64x64 obs (rebuild 3-stacks offline), goal, and RAW action [a0,a1] -- so the
# log is a flight trace AND a task-real calibration source.
#
# Format: b'PPOFLY01' | u16 G(=64) | u16 VLEN | u16 pad
#   per frame: u32 t_ms | u16 dt_ms | G*G u8 obs(newest frame) | VLEN f32 goal | 2 f32 action
#   (= 4+2+4096+16+8 = 4126 B fixed)
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np
try:
    import framestack                     # custom N6 fw (2026-09-01): zero-alloc C frame-stack build
    _HAS_FS = True
except ImportError:
    _HAS_FS = False                        # stock fw -> ulab fallback

MODEL = '/sdcard/ppo_headpen.bin'
LOGPATH = '/sdcard/headpenfly.bin'
G, SRC = 64, 320; SC = G / SRC; KSTACK = 3
UART_ID, BAUD = 4, 115200
PERIOD = 70                               # ~14.3 Hz
GC_EVERY = 15
VLEN = 4
MISSION_S = 10                            # record this many seconds AFTER the first UART
KMAX = 4000

model = ml.Model(MODEL); print('ppo_headpen_fly_rec: model in', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(2)   # snapshot 40->22ms
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

_img = np.zeros((1, G, G, KSTACK)); _first = True   # persistent; ch0=oldest ... ch(K-1)=newest
def build_img():
    global _first
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    ob = _g.bytearray()                              # uint8 newest frame (for logging; ~4KB/frame)
    if _HAS_FS:                                      # zero-alloc C stack build (custom fw)
        (framestack.fill if _first else framestack.push)(_img, _g)
        _first = False
    else:
        newf = np.array(np.frombuffer(ob, dtype=np.uint8), dtype=np.float).reshape((G, G))
        if _first:
            for i in range(KSTACK): _img[0, :, :, i] = newf
            _first = False
        else:
            for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]
            _img[0, :, :, KSTACK - 1] = newf
    return _img, ob

goal_vec = [1.0, 0.0, 0.0, 0.5]           # CF: [cos(brg), sin(brg), yaw_rate/1.0, |v_EKF|/1.5]
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

f = open(LOGPATH, 'wb'); f.write(b'PPOFLY01' + struct.pack('<HHH', G, VLEN, 0))
print('ppo_headpen_fly_rec armed -> waiting for first UART, then flying + logging %ds' % MISSION_S)
gc.collect()
k = 0; nlog = 0; _hb = time.ticks_ms(); _started = False
t0 = time.ticks_ms(); t_mission = None; last_tick = t0
while k < KMAX:
    tick = time.ticks_ms(); dt = time.ticks_diff(tick, last_tick); last_tick = tick
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    if (not _started) and _rx_seen:
        _started = True; t_mission = tick
        print('>>> first UART RX -> flying + recording %ds window START' % MISSION_S)
    img, ob = build_img()
    out = model.predict([img, vector])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])   # already tanh(mean); raw action sent
    send_action(a0, a1)                                   # NO filtering
    if _started:
        f.write(struct.pack('<IH', time.ticks_diff(tick, t_mission), dt & 0xFFFF))
        f.write(bytes(ob))
        f.write(struct.pack('<%df' % VLEN, *goal_vec))
        f.write(struct.pack('<2f', a0, a1))
        nlog += 1
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d st%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz' % (k, _started, a0, a1, goal_vec[0], goal_vec[1], hz))
    if _started and time.ticks_diff(tick, t_mission) >= MISSION_S * 1000:
        break
f.close()
send_action(0.0, 0.0)
if nlog:
    print('DONE: logged %d frames over %ds -> %.1f Hz avg -> %s' % (nlog, MISSION_S, nlog / MISSION_S, LOGPATH))
else:
    print('DONE: no UART received (nothing logged). Check the CF is sending 0xCF frames.')
