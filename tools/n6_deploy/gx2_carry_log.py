# gx2_carry_log.py -- DIAGNOSTIC: log the recurrent carry to find where board (Neural-ART) diverges
# from fp32/int8-onnx. NO UART, NO drone -- hand-move the board. Fixed 10 s from script start, fresh
# (zero) belief at start, fixed dummy goal. Logs the model INPUT carry (deter_in/stoch_in/prevact_in)
# + obs + goal, and the raw action OUT, every frame -> offline we replay each step through fp32/int8
# with the board's EXACT carry to see if the per-step matches (kernel diff) or only drifts (recurrence).
#
# Format: b'GX2CAR01' | u16 G | u16 DET | u16 STO | u16 VLEN
#   per frame: u32 t_ms | u16 dt_ms | G*G u8 obs | VLEN f32 goal
#              | DET f32 deter_in | STO f32 stoch_in | 2 f32 prevact_in | 2 f32 raw_action_out
#   (= 4+2+1024+16+8192+2048+8+8 = 11302 B)
import csi, image, time, ml, struct, gc
from ulab import numpy as np

MODEL = '/sdcard/gx2_15hz_1ms_real.bin'   # real-data-calibrated float-IO bin (was gx2_15hz_1ms_f.bin)
LOGPATH = '/sdcard/gx2carry_real.bin'
G, SRC = 32, 320; SC = G / SRC
VLEN = 4; DET = 2048; STO = 16 * 32
PERIOD = 70
GC_EVERY = 30                            # collect() every N frames (NOT per-frame mem_free poll)
MISSION_S = 10                           # fixed window from START (no UART)
GOAL = [1.0, 0.0, 0.0, 0.5]

model = ml.Model(MODEL)
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)

deter = np.zeros((1, 2048)); stoch = np.zeros((1, 16, 32)); prevact = np.zeros((1, 2))   # fresh belief
vector = np.array(GOAL, dtype=np.float).reshape((1, VLEN))

f = open(LOGPATH, 'wb')
f.write(b'GX2CAR01' + struct.pack('<HHHH', G, DET, STO, VLEN))
print('gx2_carry_log: logging carry for %ds, MOVE the board by hand -> %s' % (MISSION_S, LOGPATH))
gc.collect()
k = 0; t0 = time.ticks_ms(); last = t0
while time.ticks_diff(time.ticks_ms(), t0) < MISSION_S * 1000:
    tick = time.ticks_ms(); dt = time.ticks_diff(tick, last); last = tick
    if k % GC_EVERY == 0:                 # periodic collect, NOT a per-frame mem_free() walk (~200ms!)
        gc.collect()
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    obs_b = _g.bytearray()
    img = np.array(np.frombuffer(obs_b, dtype=np.uint8), dtype=np.float).reshape((1, G, G, 1))
    # log INPUTS (exact carry going INTO this predict)
    f.write(struct.pack('<IH', time.ticks_diff(tick, t0), dt & 0xFFFF))
    f.write(bytes(obs_b))
    f.write(struct.pack('<%df' % VLEN, *GOAL))
    f.write(bytes(deter)); f.write(bytes(stoch))
    f.write(struct.pack('<2f', float(prevact[0][0]), float(prevact[0][1])))
    # predict + log OUTPUT action
    out = model.predict([img, vector, deter, stoch, prevact])
    a0 = float(out[0][0][0]); a1 = float(out[0][0][1])
    f.write(struct.pack('<2f', a0, a1))
    # advance carry (direct float feedback)
    deter = out[1]; stoch = np.array(out[2]).reshape((1, 16, 32)); prevact = out[0]
    k += 1
    if k % 10 == 0:
        b = np.frombuffer(obs_b, dtype=np.uint8)
        act = 100.0 * np.sum(np.array((b < 123) + (b > 133), dtype=np.float)) / (G * G)
        print('  k%d dt%dms a[%+.2f,%+.2f] ev%%%.0f free%d' % (k, dt, a0, a1, act, gc.mem_free()))
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
f.close()
print('DONE: %d frames over %ds -> %s. Pull it; I replay each step through fp32/int8.' % (k, MISSION_S, LOGPATH))
