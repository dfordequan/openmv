# ppo_main.py -- deploy PPO h64 (ppo_h64, task yawzoneh64r7w20gxv3) on the N6.
# FEEDFORWARD (no recurrency, no carry): image[1,64,64,3] + vector[4] -> action[2]=tanh(mean).
# The 3-frame STACK is the only temporal context. Bin is float-IO (raw uint8-ish frame fed as
# float; the (x-128)/32 affine + tanh are baked into the graph, so out[0] is the final action).
#
# From the 2026-08-25 bundle README:
#   * image 64x64x1 GenX DIFF3D net frame, 20 ms window, 3-FRAME STACK -> 64x64x3 (newest LAST).
#   * ~14.3 Hz decisions (act-repeat 7 @ 100 Hz)          -> PERIOD = 70
#   * NO output filtering (tanh only, already in the graph). vmin 1.5 / vmax 3.0 m/s (CF side).
#   * yaw-rate lag tau=0.08 s applied CF-side.
#
# >>> VALIDATE ON BOARD: the 3-channel stack build (ulab 4D slice-assign) is the one untested op.
#     If `img[0,:,:,i] = f` raises, use the np.concatenate fallback noted below.
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np
try:
    import framestack                     # custom N6 fw (2026-09-01): zero-alloc C frame-stack build
    _HAS_FS = True
except ImportError:
    _HAS_FS = False                        # stock fw -> ulab fallback (churns ~50KB/frame -> GC hitch)

MODEL = '/sdcard/ppo_h64.bin'
G, SRC = 64, 320                          # 64px model input (NOT 32 like gx2/gx7)
KSTACK = 3
UART_ID, BAUD = 4, 115200
PERIOD = 70                               # ~14.3 Hz (act-repeat 7 @ 100 Hz)
GC_EVERY = 2000 if _HAS_FS else 15         # framestack: churn ~0 -> collect rarely; else 64px allocs heavy
SC = G / SRC                              # 0.2 (320 -> 64)
VLEN = 4

model = ml.Model(MODEL)
print('ppo_main: model in', model.input_shape)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(2)  # snapshot 40->22ms (fb2 overlaps readout+integration, measured 2026-09-01)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

# PERSISTENT image buffer + in-place channel shift -- NO 49KB/frame alloc (avoids heap fragmentation
# that otherwise degrades the loop ~12->7 Hz over ~30 frames, confirmed in ppo_pipeline). Stack order:
# channels shift down each frame, newest -> last channel (matches training).
_img = np.zeros((1, G, G, KSTACK))
_first = True
def build_img():
    global _first
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    if _HAS_FS:                                                    # zero-alloc C path (custom fw)
        (framestack.fill if _first else framestack.push)(_img, _g)  # fill=all channels (reset); push=shift+newest-last
        _first = False
    else:                                                          # ulab fallback (stock fw): ~50KB/frame
        newf = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((G, G))
        if _first:
            for i in range(KSTACK): _img[0, :, :, i] = newf   # fill with first frame (== env reset)
            _first = False
        else:
            for i in range(KSTACK - 1): _img[0, :, :, i] = _img[0, :, :, i + 1]   # shift down
            _img[0, :, :, KSTACK - 1] = newf                                       # newest last
    return _img

goal_vec = [1.0, 0.0, 0.0, 0.5]   # CF sends: [cos(brg), sin(brg), yaw_rate/3.5, |v_EKF|/vmax], vmax=3.0
# slot[3] = EKF horizontal speed (norm of vx,vy state estimate)/3.0, NOT the last commanded speed.
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

print('ppo_main [feedforward, 64x64x3 stack, 14Hz] up.')
gc.collect()
k = 0; _hb = time.ticks_ms()
while True:
    tick = time.ticks_ms()
    if k % GC_EVERY == 0:
        gc.collect()
    poll_vector()
    img = build_img()
    out = model.predict([img, vector])            # feedforward: no carry
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])   # already tanh(mean)
    send_action(a0, a1)                            # README: NO output filtering
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a[%+.2f,%+.2f] brg(%.2f,%.2f) %.1fHz free%d' % (k, a0, a1, goal_vec[0], goal_vec[1], hz, gc.mem_free()))
