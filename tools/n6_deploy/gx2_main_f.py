# gx2_main_f.py -- deploy the FLOAT-I/O gx2 model (gx2_15hz_f.bin). All model I/O is float32, so the
# recurrent carry feeds back DIRECTLY (deter=out[1], stoch=reshape(out[2])) with a consistent
# representation -- this is the fix for the broken int8-in/float-out carry that garbled the belief
# state (and drove the left-circle). No q_vec / q_carry / requant needed.
#
# Obs = GenX320 DIFF3D graded-net snapshot (needs STOCK firmware: USAT(net*16+128)). snapshot -> AREA
# downsample 320->32 -> float [1,32,32,1]. UART4: TX action (N6->CF), RX goal vector (CF->N6).
#
# TIMING BUILD: per-frame prints show how long each step takes (us). Logic is UNCHANGED -- only
# time.ticks_us() calls and print()s were added. The per-step numbers are measured BEFORE the prints,
# so they are accurate; note the prints themselves add serial overhead, so the real loop is a bit
# slower than the "WORK" sum shown.
import csi, image, time, ml, struct, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz_1ms_real.bin' # REAL-DATA-calibrated float-IO bin (fixes the a1=-1 crawl;
                                        # old synthetic-calib was gx2_15hz_1ms_f.bin). v3: gx2_15hz_f.bin
G, SRC = 32, 320
UART_ID, BAUD = 4, 115200
YAWRATE_MAX = 3.5
PERIOD = 70                              # r7 == 100Hz/7 ~= 14.3 Hz control
GC_EVERY = 30                            # collect() every N frames (NOT per-frame mem_free poll)
SC = G / SRC
VLEN = 4
# a0 (yaw) is near-bang-bang (+-1) and CANCELS to a tiny net yaw (~1-7 deg/s) on the real drone, so
# the correct-but-weak toward-goal intent never becomes real turning -> "never yaws back". Low-pass
# the SENT command to extract that net intent as a steady yaw. ALP~0.15 -> ~0.7s time constant.
# Set A_LP=1.0 to disable (raw bang-bang). Filter is on the OUTPUT only; the policy's recurrence
# still gets the raw action as prevact.
A_LP = 0.15
# a1 (speed) SPEED CAP -- breaks the race->obs-saturation->chaos runaway seen in the 0823 flights.
# CF maps v_cmd = 1.5 + (a1+1)/2*4.5 m/s, so a1=+1 -> 6 m/s: that saturates/blurs the 20ms event
# frame (OOD obs) AND exceeds the CFB model's <3 m/s validity -> belief goes chaotic, yaw jitters,
# speed sticks at the rail = circling. The GOOD flight stayed moderate. Cap the SENT speed here.
# A1_MAX=0.3 -> ~4.4 m/s; lower toward -0.33 for a strict 3 m/s. Cap is on OUTPUT only (prevact raw).
A1_MAX = 0.3

# ---- load the model FIRST (framebuffer sizes around it) ----
model = ml.Model(MODEL)

# ---- camera: grayscale 320 snapshot, single framebuffer ----
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
_c.framebuffers(1)
_c.snapshot(time=800)                    # settle
_g = image.Image(G, G, image.GRAYSCALE)

# sub-step timings for build_img (us), filled each call so the loop can print them
_t_snap = _t_down = _t_conv = 0
def build_img():
    global _t_snap, _t_down, _t_conv
    # -- build STEP a: snapshot the 320x320 graded net (sensor-gated ~ integration window) --
    _t = time.ticks_us()
    d = _c.snapshot()                                                 # 320x320 graded net (128-centred)
    _t_snap = time.ticks_diff(time.ticks_us(), _t)
    # -- build STEP b: AREA downsample 320 -> 32 --
    _t = time.ticks_us()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)    # faithful 10x10 average -> 32x32
    _t_down = time.ticks_diff(time.ticks_us(), _t)
    # -- build STEP c: uint8 -> float32 [1,32,32,1] (model does /255-0.5 internally) --
    _t = time.ticks_us()
    r = np.array(np.frombuffer(_g.bytearray(), dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1))
    _t_conv = time.ticks_diff(time.ticks_us(), _t)
    return r

# ---- FLOAT carry + vector (no quantization) ----
deter = np.zeros((1, 2048))
stoch = np.zeros((1, 16, 32))
prevact = np.zeros((1, 2))
a0f = 0.0; a1f = 0.0                      # low-pass state for the SENT command
goal_vec = [1.0, 0.0, 0.0, 0.5]          # [cos(bearing), sin(bearing), yawrate/3.5, v/6.0]
vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))

uart = UART(UART_ID, BAUD)
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
_rx = bytearray()
_rx_seen = False                         # set True on the first valid goal frame (= mission start)
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
            vector = np.array(goal_vec, dtype=np.float).reshape((1, VLEN))   # FLOAT vector (no quant)
            _rx_seen = True
            i += 11
        else:
            i += 1
    _rx = _rx[i:]

print('gx2_main_f [FLOAT-IO, direct carry] up. model in', model.input_shape, '-> UART%d @ %d' % (UART_ID, BAUD))
gc.collect()
k = 0; _hb = time.ticks_ms(); _started = False
_us = time.ticks_us
while True:
    tick = time.ticks_ms()

    # ===== STEP 0: garbage collection (periodic, NOT per-frame mem_free poll) =====
    # NOTE: gc.mem_free() alone was ~200 ms/frame here -- it walks the 24 MB DRAM GC-heap
    # bitmap every call, and we were nowhere near GC_FLOOR so collect() never even ran. So we
    # DON'T poll mem_free() any more; we just collect() every GC_EVERY frames to reclaim the
    # per-frame ndarray churn (img / reshaped stoch / predict outputs) at a controlled time.
    _t = _us()
    if k % GC_EVERY == 0:
        gc.collect()
    t_gc = time.ticks_diff(_us(), _t)

    # ===== STEP 1: poll UART4 for the goal vector (CF -> N6) =====
    _t = _us()
    poll_vector()
    t_poll = time.ticks_diff(_us(), _t)

    # ===== STEP 1b: hold belief fresh until first UART (mission start); usually skipped in flight =====
    if not _started:
        deter = np.zeros((1, 2048)); stoch = np.zeros((1, 16, 32)); prevact = np.zeros((1, 2))
        a0f = 0.0; a1f = 0.0
        if _rx_seen:
            _started = True
            print('>>> first UART RX -> recurrent belief RESET (mission start)')

    # ===== STEP 2: build obs (snapshot + downsample + uint8->float) =====
    _t = _us()
    img = build_img()
    t_build = time.ticks_diff(_us(), _t)

    # ===== STEP 3: model inference (predict) =====
    _t = _us()
    out = model.predict([img, vector, deter, stoch, prevact])
    t_pred = time.ticks_diff(_us(), _t)
    action = out[0]; a0 = float(action[0][0]); a1 = float(action[0][1])
    a0f = A_LP * a0 + (1 - A_LP) * a0f                # low-pass -> net toward-goal yaw survives
    a1f = A_LP * a1 + (1 - A_LP) * a1f
    if a1f > A1_MAX: a1f = A1_MAX                     # speed cap -> anti-race-runaway (see A1_MAX note)

    # ===== STEP 4: send action over UART4 (N6 -> CF) =====
    _t = _us()
    send_action(a0f, a1f)
    t_send = time.ticks_diff(_us(), _t)

    # ===== STEP 5: advance recurrent carry (direct float feedback) =====
    _t = _us()
    deter = out[1]                                    # DIRECT float feedback (consistent recurrence)
    stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = out[0]                                  # policy recurrence gets the RAW action
    t_carry = time.ticks_diff(_us(), _t)

    # ===== PER-FRAME TIMING PRINT (which step costs how much) =====
    work = t_gc + t_poll + t_build + t_pred + t_send + t_carry
    print('--- frame %d   WORK %d us -> %.1f Hz (before PERIOD pacing & prints) ---' % (k, work, 1e6 / max(work, 1)))
    print('  [0] gc            %6d us' % t_gc)
    print('  [1] poll_vector   %6d us   (UART RX goal)' % t_poll)
    print('  [2] build_img     %6d us   (snapshot %d + downsample %d + float %d)' % (t_build, _t_snap, _t_down, _t_conv))
    print('  [3] predict       %6d us   (model inference)' % t_pred)
    print('  [4] send_action   %6d us   (UART TX)' % t_send)
    print('  [5] carry update  %6d us' % t_carry)

    # ===== STEP 6: pace to PERIOD (sleep the remainder) =====
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 15 == 0:
        now = time.ticks_ms(); hz = 15000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d a0 %+.2f->%+.2f a1 %+.2f->%+.2f %.1fHz free%d' % (k, a0, a0f, a1, a1f, hz, gc.mem_free()))
