# viz_deploy.py -- LIVE VISUALIZER for the N6 event policy (open in the OpenMV IDE, hit Run).
# Shows the GenX320 event frame + draws the predicted heading as an arrow, and prints the action.
#
# NOTE: this uses the GenX320 GRAYSCALE FRAME mode (snapshot()) so the OpenMV IDE can display it
# (128 = no event, bright = ON, dark = OFF). main.py uses EVENT mode + read_events (2 channels: ON
# magnitude + OFF magnitude); here we derive the 2-ch model input from the single signed frame
# (on=max(net,0), off=max(-net,0)), a close approximation -- fine for eyeballing, not bit-identical.
#
# forward = image UP. action=[a0,a1] egocentric heading; rel=atan2(a1,a0) (>0 = LEFT).
# Arrow: rel=0 -> up, +90 -> left, -90 -> right.  Needs /sdcard/forest13.bin.
import csi, image, time, ml, math, gc
from ulab import numpy as np

MODEL = '/sdcard/forest13.bin'
G = 64                                     # model input size
CLIP = 1.0
GOAL = (1.0, 0.0)

# ---------------- event camera in GRAYSCALE FRAME mode (displayable via snapshot) ----------------
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE)
_c.framesize((320, 320))                   # 320x320 event-accumulation frames (tuple form for this build)
_c.snapshot(time=1000)                     # let it settle
W, H = _c.width(), _c.height()
_s64 = image.Image(G, G, image.GRAYSCALE)


def build_img(frame):                      # frame = 320x320 grayscale snapshot -> [1,64,64,2] uint8
    _s64.draw_image(frame, 0, 0, x_scale=G / float(W), y_scale=G / float(H), hint=image.AREA)
    S = _s64.to_ndarray('f'); net = (S - 128.0)                     # signed: >0 ON, <0 OFF
    on = np.minimum(np.maximum(net, 0.0) / CLIP, 1.0)
    off = np.minimum(np.maximum(-net, 0.0) / CLIP, 1.0)
    frame2 = np.concatenate((on.reshape((1, G, G, 1)), off.reshape((1, G, G, 1))), axis=3)
    amag = np.maximum(net, -net)                                    # |net|  (ulab has no np.abs)
    active = int(np.sum(np.array(amag > 6.0, dtype=np.float)))
    return np.array(frame2 * 255, dtype=np.uint8), active


# ---------------- model + quant params (same as main.py) ----------------
model = ml.Model(MODEL)
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point
OUT_INT = ('b' in model.output_dtype[1])


def requant(q, so, zo, si, zi):
    real = (np.array(q, dtype=np.float) - zo) * so if OUT_INT else np.array(q, dtype=np.float)
    z = np.floor(real / si + 0.5) + zi
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)


VLEN = model.input_shape[1][1]             # vector length (forest13=3, ratezone=4: +rate cmd)
def q_vec(v):
    z = np.floor(np.array(v, dtype=np.float) / IN_SC[1] + 0.5) + IN_ZP[1]
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8).reshape((1, VLEN))


deter = np.zeros((1, 2048), dtype=np.int8) + IN_ZP[2]
stoch = np.zeros((1, 16, 32), dtype=np.int8) + IN_ZP[3]
prevact = np.zeros((1, 2))
vector = q_vec([GOAL[0], GOAL[1]] + [0.0] * (VLEN - 2))   # goal ahead; pad extra dims (yawrate/rate) with 0

cx, cy = W // 2, H // 2
L = int(0.30 * min(W, H))
clock = time.clock()
print('viz up. model in', model.input_shape, ' frame', W, 'x', H)

def us():                                                   # microsecond stopwatch helper
    return time.ticks_us()

gc.collect(); GC_FLOOR = gc.mem_free() // 4                 # collect ONLY when heap is low: gc.collect()
#                                          costs ~940 ms on the N6 (huge/slow heap) -> never every tick.

while True:
    clock.tick()
    t0 = us()
    if gc.mem_free() < GC_FLOOR: gc.collect()               # cheap check; collect rarely (avoids the crash)
    t1 = us(); disp = _c.snapshot()                         # grab the event frame (IDE shows this)
    t2 = us(); img, act = build_img(disp)                   # data processing: downscale + ON/OFF + uint8
    t3 = us(); out = model.predict([img, vector, deter, stoch, prevact])   # INFERENCE (NPU + M55)
    t4 = us()
    a = out[0]; a0 = float(a[0][0]); a1 = float(a[0][1])
    rel = math.atan2(a1, a0)                                # desired heading change (rad); >0 = LEFT
    # carry feedback (same as main.py): PLAIN reshape (NOT transpose -- see main.py note)
    deter = requant(out[1], OUT_SC[1], OUT_ZP[1], IN_SC[2], IN_ZP[2])
    s_t = np.array(out[2]).reshape((1, 16, 32))
    stoch = requant(s_t, OUT_SC[2], OUT_ZP[2], IN_SC[3], IN_ZP[3])
    prevact = np.array(a, dtype=np.float)
    t5 = us()

    # ---- overlay the prediction on the displayed frame ----
    disp.draw_cross((cx, cy), color=255, size=6)            # this firmware wants a TUPLE position
    disp.draw_line((cx, cy, cx, cy - int(0.13 * H)), color=90)      # faint forward reference (up)
    tx = int(cx - L * math.sin(rel)); ty = int(cy - L * math.cos(rel))
    disp.draw_line((cx, cy, tx, ty), color=255, thickness=3)        # predicted heading arrow
    disp.draw_circle((tx, ty, 5), color=255, fill=True)
    disp.draw_string((4, 4), 'hdg %+5.1fdeg  a[% .2f,% .2f]' % (math.degrees(rel), a0, a1), color=255)
    disp.draw_string((4, 16), 'active %d  fps %.1f' % (act, clock.fps()), color=255)
    t6 = us()
    ms = lambda x, y: time.ticks_diff(y, x) / 1000.0
    disp.draw_string((4, 28), 'infer %dms  build %dms  fps %.1f'
                     % (ms(t3, t4), ms(t2, t3), clock.fps()), color=255)
    # full per-stage breakdown to the terminal
    print('hdg %+6.1f a[% .2f,% .2f] | gc %4.0f  snap %4.0f  build %4.0f  INFER %4.0f  carry %4.0f  draw %4.0f ms  fps %.1f'
          % (math.degrees(rel), a0, a1, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4), ms(t4, t5), ms(t5, t6), clock.fps()))
