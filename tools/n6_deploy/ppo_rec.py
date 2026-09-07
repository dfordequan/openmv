# ppo_rec.py -- record REAL 64x64 GenX obs frames for PPO int8 recalibration (and analysis).
# No UART/drone needed: hand-move the board (or fly it) to create a representative event stream.
# The host recalibrate_int8.py builds 3-frame stacks from consecutive frames as the calib set.
#
# Format: b'PPOREC01' | u16 G(=64) | u16 pad ; per frame: u32 t_ms | G*G u8 obs   (= 4+4096 = 4100 B)
import csi, image, time, gc, struct
from ulab import numpy as np
G, SRC = 64, 320; SC = G / SRC
N = 300; PERIOD = 70
LOGPATH = '/sdcard/ppo_obs64.bin'
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC)); _c.framebuffers(1)
_c.snapshot(time=800)
_g = image.Image(G, G, image.GRAYSCALE)
f = open(LOGPATH, 'wb'); f.write(b'PPOREC01' + struct.pack('<HH', G, 0))
print('ppo_rec: %d frames of 64x64 obs -> %s  (MOVE the board)' % (N, LOGPATH))
gc.collect(); t0 = time.ticks_ms()
for k in range(N):
    tick = time.ticks_ms()
    if k % 30 == 0:
        gc.collect()
    d = _c.snapshot()
    _g.draw_image(d, 0, 0, x_scale=SC, y_scale=SC, hint=image.AREA)
    ob = _g.bytearray()
    f.write(struct.pack('<I', time.ticks_diff(tick, t0))); f.write(bytes(ob))
    if k % 30 == 0:
        b = np.frombuffer(ob, dtype=np.uint8)
        act = 100.0 * np.sum(np.array((b < 123) + (b > 133), dtype=np.float)) / (G * G)
        print('  k%d ev%%%.0f free%d' % (k, act, gc.mem_free()))
    rem = PERIOD - time.ticks_diff(time.ticks_ms(), tick)
    if rem > 0:
        time.sleep_ms(rem)
f.close()
print('DONE: %d frames -> %s. Pull it; feed to recalibrate_int8.py ppo.' % (N, LOGPATH))
