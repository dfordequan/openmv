# Onboard LIVE-EVENT test of the forest event-policy on the N6.
# GenX320 events -> [64,64,2] ON/OFF frame -> ml.Model (manual recurrency) -> 2D heading action.
# Measures real end-to-end latency. NOTE: weights are the mirror/proxy-trained forest2 stand-in +
# dummy goal-bearing vector, so ACTIONS ARE NOT MEANINGFUL — this is a "does the full onboard loop
# run, and how fast" test. Run: mpremote connect /dev/ttyACM0 run n6_forest_run.py
import csi, image, time, ml
from ulab import numpy as np

BUF, WIN_MS, G, CLIP = 2048, 10, 64, 3.0
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(320, 320, image.GRAYSCALE); _act = image.Image(320, 320, image.GRAYSCALE)
_s64 = image.Image(G, G, image.GRAYSCALE); _a64 = image.Image(G, G, image.GRAYSCALE)

def build_onoff(clip=CLIP):
    first = True; t0 = time.ticks_ms(); tot = 0
    while time.ticks_diff(time.ticks_ms(), t0) < WIN_MS:
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1: continue
        _sig.draw_event_histogram(_ev[:n], clear=first, brightness=128, contrast=1)
        _ev[:n, 0] = 1
        _act.draw_event_histogram(_ev[:n], clear=first, brightness=0, contrast=1)
        first = False; tot += n
    _s64.draw_image(_sig, 0, 0, x_scale=G/320.0, y_scale=G/320.0, hint=image.AREA)
    _a64.draw_image(_act, 0, 0, x_scale=G/320.0, y_scale=G/320.0, hint=image.AREA)
    S = _s64.to_ndarray('f'); A = _a64.to_ndarray('f'); net_ = S - 128.0
    on = np.minimum(np.maximum((A + net_) * 0.5, 0.0) / clip, 1.0)
    off = np.minimum(np.maximum((A - net_) * 0.5, 0.0) / clip, 1.0)
    return on, off, tot

model = ml.Model('/sdcard/forest_fast.bin')
print("in_shape:", model.input_shape, "in_dtype:", model.input_dtype)
print("out_shape:", model.output_shape, "out_dtype:", model.output_dtype)
print("out_scale:", model.output_scale, "out_zp:", model.output_zero_point)

# --- quantization params (inputs: [img,vec,deter,stoch,prevact]; outputs: [action,deter,stoch]) ---
IN_SC, IN_ZP = model.input_scale, model.input_zero_point
OUT_SC, OUT_ZP = model.output_scale, model.output_zero_point
SC_D_IN, ZP_D_IN = IN_SC[2], IN_ZP[2]   # deter input
SC_S_IN, ZP_S_IN = IN_SC[3], IN_ZP[3]   # stoch input
SC_D_OUT, ZP_D_OUT = OUT_SC[1], OUT_ZP[1]
SC_S_OUT, ZP_S_OUT = OUT_SC[2], OUT_ZP[2]
OUT_IS_INT = ('b' in model.output_dtype[1])   # True if predict returns int8 state (needs dequant)

def requant(q, sc_out, zp_out, sc_in, zp_in):
    """out-encoding int8 (or float) -> real -> in-encoding int8."""
    real = (np.array(q, dtype=np.float) - zp_out) * sc_out if OUT_IS_INT else np.array(q, dtype=np.float)
    z = np.floor(real / sc_in + 0.5) + zp_in
    return np.array(np.minimum(np.maximum(z, -128), 127), dtype=np.int8)

# STEP 3: proper manual-recurrency carry (state fed back each tick). Init at zero-point.
deter = np.zeros((1, 2048), dtype=np.int8) + ZP_D_IN
stoch = np.zeros((1, 16, 32), dtype=np.int8) + ZP_S_IN
prevact = np.zeros((1, 2))
vector = np.zeros((1, 3), dtype=np.int8) + IN_ZP[1]   # dummy goal until step 4 (CF pose over UART)

def make_img():                         # live GenX320 -> [1,64,64,2] uint8 (NHWC), 0..255
    on, off, tot = build_onoff()
    frame = np.concatenate((on.reshape((1, G, G, 1)), off.reshape((1, G, G, 1))), axis=3)
    return np.array(frame * 255, dtype=np.uint8), tot

lat = []
for k in range(60):
    img, tot = make_img()
    t0 = time.ticks_us()
    out = model.predict([img, vector, deter, stoch, prevact])
    dt = time.ticks_diff(time.ticks_us(), t0) / 1000.0
    action = out[0]                                   # float (1,2) heading command
    # --- feed the belief state back (transpose stoch 32x16 -> 16x32, re-encode int8) ---
    deter = requant(out[1], SC_D_OUT, ZP_D_OUT, SC_D_IN, ZP_D_IN)          # (1,2048)
    s_t = np.array(out[2].reshape((32, 16)).transpose()).reshape((1, 16, 32))  # -> input layout
    stoch = requant(s_t, SC_S_OUT, ZP_S_OUT, SC_S_IN, ZP_S_IN)            # (1,16,32)
    prevact = np.array(action, dtype=np.float)                            # float, fed directly
    if k >= 8: lat.append(dt)
    if k % 5 == 0:
        print("k%d  ev %d  infer %.1f ms  action=%s  deter_mean=%.1f" %
              (k, tot, dt, [round(float(a), 2) for a in action.flatten()[:2]],
               float(np.mean(np.array(deter, dtype=np.float)))))
lat.sort(); med = lat[len(lat)//2]
print("\nLIVE-EVENT loop (state fed back): infer median %.1f ms -> %.1f Hz [n=%d]" %
      (med, 1000.0/med, len(lat)))
print("If 'action' now VARIES with the scene, the recurrent state is live. If it blows up/freezes,")
print("the stoch transpose or the requant scale is off -> check against the host fp32 rollout.")
