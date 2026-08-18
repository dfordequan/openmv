# N6 FOREST POLICY -- ALWAYS-ON deploy loop.  Copy to the board as /sdcard/main.py (auto-runs on boot).
# Pipeline:  live GenX320 events -> [64,64,2] ON/OFF frame -> ml.Model (manual recurrency)
#            -> 2-D heading action -> sent out over UART bus 4 every tick.  Loops forever.
#
# Needs /sdcard/forest13.bin  (stedgeai-compiled model; generated on the host from the checkpoint).
# Model I/O (compiled):  in  = image(1,64,64,2 uint8 NHWC), vector(1,3 int8), deter(1,2048 int8),
#                              stoch(1,16,32 int8), prevaction(1,2 float)
#                        out = action(1,2 float), deter(1,2048 int8), stoch(1,32,16 int8)
# Carry (deter/stoch) is held in RAM and fed back each tick (transpose stoch 32x16->16x32 + int8 requant).
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

# ---------------- config ----------------
MODEL = '/sdcard/forest13.bin'
BUF, WIN_MS, G = 16384, 20, 64          # big buffer (avoids DROP_ON_FULL under motion); 20ms window
CLIP = 1.0                              # event-frame normalization. From temp/events.bin (real GenX320
                                       # capture): busy-frame ON p99 ~22 counts/cell (SUM) / ~25 (AREA
                                       # downscale) ~= 0.9 -> CLIP~1, NOT 3. Training target: ~8% active,
                                       # ~1% saturated. Lock exactly at FLIGHT SPEED with event_calib_main.py
                                       # (events scale with image speed; the bin was hand-recorded).
UART_ID, BAUD = 4, 115200              # UART bus 4
GOAL = (1.0, 0.0)                       # dummy egocentric goal bearing (straight ahead) until CF sends pose

# ---------------- event camera ----------------
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(320, 320, image.GRAYSCALE); _act = image.Image(320, 320, image.GRAYSCALE)
_s64 = image.Image(G, G, image.GRAYSCALE); _a64 = image.Image(G, G, image.GRAYSCALE)

def build_img():                        # -> [1,64,64,2] uint8 (NHWC), + event count
    first = True; t0 = time.ticks_ms(); tot = 0
    while time.ticks_diff(time.ticks_ms(), t0) < WIN_MS:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            continue
        _sig.draw_event_histogram(_ev[:n], clear=first, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=first, brightness=0, contrast=1)
        first = False; tot += n
    if tot == 0:                        # no events this window: emit a CLEAN zero frame, never a
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0   # stale/uninitialized image buffer
    _s64.draw_image(_sig, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    _a64.draw_image(_act, 0, 0, x_scale=G / 320.0, y_scale=G / 320.0, hint=image.AREA)
    S = _s64.to_ndarray('f'); A = _a64.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5, 0.0) / CLIP, 1.0)
    off = np.minimum(np.maximum((A - net) * 0.5, 0.0) / CLIP, 1.0)
    frame = np.concatenate((on.reshape((1, G, G, 1)), off.reshape((1, G, G, 1))), axis=3)
    return np.array(frame * 255, dtype=np.uint8), tot

# ---------------- model + quant params ----------------
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point       # [img,vec,deter,stoch,prevact]
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point   # [action,deter,stoch]
OUT_INT = ('b' in model.output_dtype[1])                        # state outputs are int8?

def requant(q, so, zo, si, zi):
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)

# quantize the (fixed) goal-bearing vector to the model's int8 input encoding
VLEN = model.input_shape[1][1]          # vector length (forest13=3; ratezone=4 with the rate cmd)
def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))

# ---------------- carry (manual recurrency), init at zero-point ----------------
deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
vector = q_vec([GOAL[0], GOAL[1]] + [0.0] * (VLEN - 2))         # goal ahead; pad yawrate/rate dims with 0

# ---------------- UART bus 4 ----------------
uart = UART(UART_ID, BAUD)
def send(a0, a1):                       # frame: 0xAA 0x55 | int16 cos*1e4 | int16 sin*1e4 | xor
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))

print('forest13 deploy loop up. model in', model.input_shape, '-> UART%d @ %d' % (UART_ID, BAUD))

# ---------------- forever loop ----------------
gc.collect(); GC_FLOOR = gc.mem_free() // 4   # gc.collect() ~940ms on the N6 -> only when heap is low
k = 0
while True:
    if gc.mem_free() < GC_FLOOR: gc.collect()   # cheap check; rare collect (keeps predict()'s arena happy)
    img, tot = build_img()
    out = model.predict([img, vector, deter, stoch, prevact])
    action = out[0]                                             # float (1,2) heading [cos,sin]
    a0 = float(action[0][0]); a1 = float(action[0][1])
    send(a0, a1)
    # feed the belief state back (step 3): PLAIN reshape stoch out(1,32,16) -> in(1,16,32), requant.
    # (Both stoch tensors are internally (16,32) w/ the same flat order -- the compiled graph reshapes
    #  stoch_in straight to 512, no transpose; a transpose here would SCRAMBLE the 32 groups x 16 classes.)
    deter = requant(out[1], OUT_SC[1], OUT_ZP[1], IN_SC[2], IN_ZP[2])
    s_t = np.array(out[2]).reshape((1, 16, 32))
    stoch = requant(s_t, OUT_SC[2], OUT_ZP[2], IN_SC[3], IN_ZP[3])
    prevact = np.array(action, dtype=np.float)
    k += 1
    if k % 25 == 0:
        print('k%d ev%d action[%.2f,%.2f] free%d' % (k, tot, a0, a1, gc.mem_free()))
