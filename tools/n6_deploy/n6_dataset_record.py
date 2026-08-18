# n6_dataset_record.py -- record the NETWORK-INPUT dataset for offboard testing.
# Runs main.py's exact obs pipeline (BUF=8192, drain-to-latest, EV_CAP, GAIN, ch0=OFF ch1=ON) but
# DOES NOT run the model or touch UART -- instead it writes each built [1,32,32,2] frame + vector to
# /sdcard/rec.bin.  Replay it on the laptop through the model with replay_dataset.py.
#
# File format:  16-byte header  b'GENXDS01' | uint16 G | uint16 C | uint16 VLEN | uint16 reserved
#   then per frame:  uint16 tot(events) | uint16 pad | G*G*C uint8 image | VLEN float32 vector
import csi, image, time, struct, gc
from ulab import numpy as np

BUF, G, DRAIN_MAX, EV_CAP = 8192, 32, 20, 6000   # DRAIN_MAX = read_fresh cap (drain backlog to real-time)
BLOCK_MS = 40                          # a read slower than this = buffer caught up = freshest frame
GAIN = 0.07 * 100 * 32
NOISE_FLOOR = 5                        # >FLOOR events per 10x10 block to light a cell (matches main.py)
VLEN = 4
PATH = '/sdcard/rec.bin'
DURATION_S = 30                       # recording length (200-300 frames @ ~10 Hz ~= 0.5 MB)
GOAL = [1.0, 0.0, 0.0, 0.5]           # vector recorded per frame (dummy = goal ahead, mid speed)

_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, BUF)
try: _c.framebuffers(3)
except Exception: pass
_ev = np.zeros((BUF, 6), dtype=np.uint16)
_sig = image.Image(G, G, image.GRAYSCALE); _act = image.Image(G, G, image.GRAYSCALE)

def read_fresh():                     # drain stale backlog to real-time (see main.py); returns newest frame
    for _ in range(DRAIN_MAX):
        t0 = time.ticks_ms()
        n = _c.ioctl(csi.IOCTL_GENX320_READ_EVENTS, _ev)
        if n < 1:
            return 0
        if time.ticks_diff(time.ticks_ms(), t0) > BLOCK_MS:
            return n
    return n

def build_img():                      # read_fresh (drains to real-time) -> [1,G,G,2] uint8, tot
    n = read_fresh()
    if n < 1:
        return np.zeros((1, G, G, 2), dtype=np.uint8), 0
    lo = n - EV_CAP if n > EV_CAP else 0
    ev = _ev[lo:n]; tot = n - lo
    ev[:, 4] = ev[:, 4] // 10; ev[:, 5] = ev[:, 5] // 10
    _sig.draw_event_histogram(ev, clear=True, brightness=128, contrast=1)
    ev[:, 0] = 1
    _act.draw_event_histogram(ev, clear=True, brightness=0, contrast=1)
    S = _sig.to_ndarray('f'); A = _act.to_ndarray('f'); net = S - 128.0
    on = np.minimum(np.maximum((A + net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    off = np.minimum(np.maximum((A - net) * 0.5 - NOISE_FLOOR, 0.0) * GAIN, 255.0)
    frame = np.concatenate((off.reshape((1, G, G, 1)), on.reshape((1, G, G, 1))), axis=3)
    return np.array(frame, dtype=np.uint8), tot

vecb = struct.pack('<%df' % VLEN, *GOAL)
f = open(PATH, 'wb')
f.write(b'GENXDS01' + struct.pack('<HHHH', G, 2, VLEN, 0))
print('recording %d s of network-input frames -> %s' % (DURATION_S, PATH))
gc.collect()
t0 = time.ticks_ms(); k = 0
try:
    while time.ticks_diff(time.ticks_ms(), t0) < DURATION_S * 1000:
        img, tot = build_img()
        b = bytes(np.array(img, dtype=np.uint8))       # G*G*2 = 2048 bytes, [1,G,G,2] C-order
        if len(b) != G * G * 2:
            continue
        if f.write(struct.pack('<HH', tot & 0xFFFF, 0)) != 4 or f.write(b) != len(b) or f.write(vecb) != len(vecb):
            print('SHORT WRITE (SD full?) after %d frames -> stop' % k); break
        k += 1
        if k % 20 == 0:
            print('  %d frames  tot%d  free%d' % (k, tot, gc.mem_free()))
finally:
    f.close()
print('done: %d frames -> %s' % (k, PATH))
print('pull it:  python omv_repl.py  (or IDE)  then cp :/sdcard/rec.bin ./rec.bin')
