# ratezone3_main.py -- ALWAYS-ON deploy loop for BEST_ratezone3 (npunative, 32x32) on the OpenMV N6.
# Copy to the board as /sdcard/main.py to auto-run.  Implements runs/ratezone3/DEPLOYMENT.md Part 1.
#
# PIPELINE (33 Hz):  GenX320 events --30ms--> [1,32,32,2] uint8 (ch0=OFF, ch1=ON, SUM-binned, gain g)
#   -> ml.Model manual-recurrency -> action[2]=[yaw_rate/3.5, speed_cmd] -> UART4 to the Crazyflie.
#   Vector obs [cos,sin, yaw_rate/3.5, v/6.0] comes FROM the Crazyflie over UART4 (dummy until wired).
#
# >>> SENSOR CAVEAT: the doc specifies the HISTO3D firmware edit (separate on-sensor ON/OFF histograms).
# That firmware isn't built here, so this uses read_events + a direct 32x32 histogram per polarity
# (SUM-binned, drop-resistant with a big EVT buffer).  Functionally equivalent ON/OFF SUM counts; if
# heavy-motion event drops show up, do the HISTO3D edit (genx320.c) per the doc.
#
# >>> GAIN g: calibrate once per venue so median(nonzero(img)) ~= 76 (see DEPLOYMENT.md).  GAIN below
# folds the doc's `counts*g*32`; retune GAIN on the real scene.
import csi, image, time, ml, struct, math, gc
from machine import UART
from ulab import numpy as np

MODEL = '/sdcard/ratezone3.bin'
# BUF is the OCCUPANCY knob (verified on-board 2026-08-13 under hand motion):
#   BUF 2048 -> occ ~13%,  4096 -> ~17%,  8192 -> ~24-30% == the sim's ~26% target.
# Smaller BUF is faster ONLY on a static desk (slow event fill); under real motion the buffer
# fills in ~2-5 ms for ALL sizes, so 8192 is NOT slower when it matters -- pick it for fidelity.
BUF, WIN_MS, G = 8192, 30, 32
GAIN = 0.07 * 100 * 32                  # counts*g*32 with g~0.07; *100 folds avg->sum (retune per venue)
NOISE_FLOOR = 5                         # require > NOISE_FLOOR events in a 10x10 block to light a cell.
                                        # The 320->32 SUM-bin makes 1 sensor px light a whole 32x32 cell;
                                        # this floor kills that single-event noise (tuned on obs_viz). 0=off.
UART_ID, BAUD = 4, 115200
YAWRATE_MAX = 3.5
DRAIN_MAX = 20                         # read_fresh cap: max reads to drain the backlog to real-time
BLOCK_MS = 40                          # a read slower than this BLOCKED (one frame) = buffer caught up = fresh
EV_CAP = 6000                           # process only the FRESHEST EV_CAP events/frame: caps BUILD time
                                        # AND caps occupancy near the sim's ~26% under heavy motion.
                                        # On-board 2026-08-13: ev~6-7k -> occ~26%; retune in flight.
AFK_ENABLE = False                     # anti-flicker: enable ONLY in venues with flickering lights
AFK_FMIN, AFK_FMAX = 90, 300           # Hz band to suppress (mains ~100/120 Hz + low harmonics)

# ---------------- event camera ----------------
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try:
    _c.framebuffers(3)                 # triple-buffer = latest-priority (per driver docs); NB: on-board
except Exception:                       # tests showed framebuffers() does NOT gate the read_events path
    pass                                # -- freshness is enforced by DRAIN_MAX below, not this call.
if AFK_ENABLE:                          # STC filter returns "Sensor control failed" on this board -> unused
    try: _c.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, AFK_FMIN, AFK_FMAX)
    except Exception as _e: print('AFK enable failed:', _e)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
# 32x32 histograms -> draw_event_histogram SUM-bins the 320-coord events straight into 32x32 (10x10 blocks)
_sig = image.Image(G, G, image.GRAYSCALE)   # signed (128 + ON-OFF)
_act = image.Image(G, G, image.GRAYSCALE)   # magnitude (ON+OFF)

def read_fresh():                       # DRAIN TO REAL-TIME. read_events returns OLDEST-first up to BUF, so a
    # slow tick (SD/math) lets >BUF events pile up and we then read STALE (this caused a ~25 s lag). Keep
    # reading (discarding the backlog) until a read BLOCKS (>BLOCK_MS ~ one frame) -- THAT is the newest frame.
    for _ in range(DRAIN_MAX):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            return 0
        if time.ticks_diff(time.ticks_ms(), t0) > BLOCK_MS:
            return n                     # blocked -> caught up to real-time
    return n                            # flood: couldn't catch up in DRAIN_MAX reads -> enable AFK

def build_img():                        # -> [1,32,32,2] uint8 (ch0=OFF, ch1=ON), + event count
    n = read_fresh()
    if n < 1:
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0
    lo = n - EV_CAP if n > EV_CAP else 0    # keep only the FRESHEST EV_CAP events (chronological)
    ev = _ev[lo:n]; tot = n - lo
    # draw_event_histogram uses LITERAL event coords (0-319), no scaling -> pre-divide by 10 so the
    # 320-space events land in the 32x32 image = a 10x10-block SUM-bin (fast; no AREA downscale).
    ev[:, 4] = ev[:, 4] // 10
    ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)   # net = ON-OFF
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)     # total = ON+OFF
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.maximum((A + net) * 0.5 - NOISE_FLOOR, 0.0)   # ON  = (total+net)/2, minus the per-bin noise floor
    off = np.maximum((A - net) * 0.5 - NOISE_FLOOR, 0.0)  # OFF = (total-net)/2, minus the noise floor
    on = np.minimum(on * GAIN, 255.0)
    off = np.minimum(off * GAIN, 255.0)
    # DEPLOYMENT.md order: ch0 = OFF, ch1 = ON
    frame = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(frame, dtype=np.uint8), tot

# ---------------- model + quant params ----------------
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point       # [img,vec,deter,stoch,prevact]
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point   # [action,deter,stoch]
OUT_INT = ('b' in model.output_dtype[1])
VLEN = model.input_shape[1][1]          # 4

def requant(q, so, zo, si, zi):
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)

def q_vec(v):                            # float vector -> int8 input encoding
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))

# ---------------- carry (manual recurrency), ZEROED at mission start ----------------
deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
# vector [cos_bearing, sin_bearing, yaw_rate/3.5, v/6.0] -- from the CF; dummy = goal ahead, mid speed
goal_vec = [1.0, 0.0, 0.0, 0.5]
vector = q_vec(goal_vec)

# ---------------- UART4 (to/from Crazyflie) ----------------
uart = UART(UART_ID, BAUD)
# N6 -> CF action frame:  0xAA 0x55 | int16 yawrate/3.5*1e4 | int16 speed*1e4 | xor
def send_action(a0, a1):
    p = struct.pack('<hh', int(max(-1, min(1, a0)) * 10000), int(max(-1, min(1, a1)) * 10000))
    ck = 0
    for b in p:
        ck ^= b
    uart.write(b'\xAA\x55' + p + bytes([ck & 0xFF]))
# CF -> N6 vector frame:  0xCF 0x55 | int16 cos*1e4 | int16 sin*1e4 | int16 yawrate/3.5*1e4 | int16 v/6.0*1e4 | xor
_rx = bytearray()
def poll_vector():                       # non-blocking; updates goal_vec if a full valid frame arrived
    global goal_vec, vector, _rx
    if uart.any():
        _rx.extend(uart.read(uart.any()))
    # MicroPython bytearray has NO .pop(0) and NO slice-delete -> scan with an index, then keep the
    # unparsed tail via reassignment (both del _rx[:k] and _rx.pop(0) raise TypeError on-device).
    i = 0; n = len(_rx)
    while i + 11 <= n:
        if _rx[i] != 0xCF or _rx[i + 1] != 0x55:
            i += 1; continue             # resync: drop one byte, keep scanning
        body = _rx[i + 2:i + 10]; ck = 0
        for b in body:
            ck ^= b
        if (ck & 0xFF) == _rx[i + 10]:
            vals = struct.unpack('<hhhh', body)
            goal_vec = [vals[0] / 10000.0, vals[1] / 10000.0, vals[2] / 10000.0, vals[3] / 10000.0]
            vector = q_vec(goal_vec)
            i += 11                      # consume the whole frame
        else:
            i += 1                       # bad checksum -> resync by one byte
    _rx = _rx[i:]                        # keep only the unparsed tail (a partial frame survives)

print('ratezone3 loop up. model in', model.input_shape, '-> UART%d @ %d @33Hz' % (UART_ID, BAUD))

# ---------------- 33 Hz forever loop (action held between inferences) ----------------
gc.collect(); GC_FLOOR = 1000000   # absolute 1MB floor (heap is huge; //4 collected every tick = 940ms killer)
PERIOD = 30                              # ms  (~33 Hz)
k = 0; _hb = time.ticks_ms()
while True:
    tick = time.ticks_ms()
    if gc.mem_free() < GC_FLOOR: gc.collect()
    poll_vector()                        # latest goal bearing / rate / speed from the CF
    img, tot = build_img()               # ~30 ms window (this IS most of the period)
    out = model.predict([img, vector, deter, stoch, prevact])
    action = out[0]; a0 = float(action[0][0]); a1 = float(action[0][1])
    send_action(a0, a1)                  # a0=yaw_rate/3.5, a1=speed_cmd
    # carry feedback WITHOUT requant: deter IN/OUT quant match exactly, stoch to 0.4% (negligible for a
    # ~one-hot). Feeding the int8 outputs straight back saves ~40ms/tick vs requant().  PLAIN reshape.
    deter = out[1]
    stoch = np.array(out[2]).reshape((1, 16, 32))
    prevact = np.array(action, dtype=np.float)
    # pace to 33 Hz (build_img already consumed ~30ms; sleep any remainder)
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
    k += 1
    if k % 33 == 0:
        now = time.ticks_ms(); hz = 33000.0 / max(time.ticks_diff(now, _hb), 1); _hb = now
        print('k%d ev%d a[%.2f,%.2f] %.1fHz free%d' % (k, tot, a0, a1, hz, gc.mem_free()))
