# ppo_headpen_main.py -- deploy PPO h64 HEADPEN (heading-penalty variant, 2026-08-27) on the N6.
# FEEDFORWARD (no recurrency): image[1,64,64,3] float NHWC + vector[4] -> action[2]=tanh(mean).
# Fixes the "doesn't return to goal" behavior (gated heading penalty in training). Isaac 10/10,
# empty-lab 5/5, int8 closed-loop 20/20.
#
# CHANGED CONSTANTS vs the old ppo_h64 (these are CF-SIDE maps -- set them in firmware!):
#   * SLOW: vmin 1.0 / vmax 1.5 m/s   ->  v = 1.0 + (a1+1)/2 * 0.5   (3 m/s^2 slew)   [was 1.5/3.0]
#   * GENTLE yaw: rate = a0 * 1.0 rad/s (~57 deg/s)                                    [was a0*3.5]
#   * vector = [cos(brg), sin(brg), yaw_rate_phys/1.0, v_EKF/1.5]                       [note /1.0, /1.5]
#   * 3-frame stack OLDEST->NEWEST (ch0=oldest, ch2=newest), NHWC flatten. no output filtering.
#   * decision rate trained 14.3 Hz (board ~10 Hz acceptable, ~1.4x hold stretch).
# Persistent image buffer (no per-frame 49KB alloc -> no heap fragmentation).
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np
try:
    import framestack                     # custom N6 fw (2026-09-01): zero-alloc C frame-stack build.
    _HAS_FS = True                         # push/fill do the channel shift + uint8->float in C -> no
except ImportError:                        # per-frame ulab temps -> ~260B/frame churn -> GC never fires
    _HAS_FS = False                        # in flight. Stock fw: ulab fallback (~50KB/frame -> GC hitch).

MODEL = '/sdcard/ppo_headpen.bin'
G, SRC = 64, 320; SC = G / SRC; KSTACK = 3
UART_ID, BAUD = 4, 115200
PERIOD = 70                               # ~14.3 Hz target
GC_EVERY = 2000 if _HAS_FS else 15         # framestack: churn ~0 -> collect rarely (safety); and the
# custom fw's GC heap is 4M not 24M so a collect is ~63ms not ~610ms (no control freeze either way).
VLEN = 4

model = ml.Model(MODEL); print('ppo_headpen_main: model in', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(2)   # 2 = overlap readout/
# integration -> snapshot 40.8->22ms (measured on-board 2026-09-01, no overflow w/ model loaded)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

_img = np.zeros((1, G, G, KSTACK))        # persistent; ch0=oldest ... ch(K-1)=newest
_first = True
def build_img():
    global _first
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    if _HAS_FS:                                                    # zero-alloc C path (custom fw)
        (framestack.fill if _first else framestack.push)(_img, _g)  # fill=all channels (reset); push=shift+newest-last
        _first = False
    else:                                                          # ulab fallback (stock fw): allocates ~50KB/frame
        newf = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((G, G))
        if _first:
            for i in range(KSTACK): _img[0, :, :, i] = newf
            _first = False
        else:
            for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]   # shift down (drop oldest)
            _img[0, :, :, KSTACK - 1] = newf                                       # newest last
    return _img

goal_vec = [1.0, 0.0, 0.0, 0.5]   # CF sends: [cos(brg), sin(brg), yaw_rate/1.0, |v_EKF|/1.5]
# slot[2] = physical yaw rate rad/s (÷1.0); slot[3] = EKF horizontal speed ÷ vmax(1.5). EKF, not cmd.
vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))

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
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

print('ppo_headpen_main [feedforward, 64x64x3, slow 1.0-1.5m/s, yaw 57deg/s] up.')
gc.collect()
k = 0; _hb = time.ticks_ms()
while True:
    tick = time.ticks_ms()
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    img = build_img()
    out = model.predict([img, vector])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])   # already tanh(mean)
    send_action(a0, a1)                                   # NO output filtering
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz free%d' % (k, a0, a1, goal_vec[0], goal_vec[1], hz, gc.mem_free()))
