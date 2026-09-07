# uart_rx_monitor.py -- print the goal-bearing vector the N6 receives from the drone over UART4.
# No camera, no model. Frame contract (CF -> N6):  0xCF 0x55 | i16 cos | i16 sin | i16 yawrate/3.5
#                                                  | i16 v/6.0 | xor   (each value * 1e4)
import time, struct, math
from machine import UART

UART_ID, BAUD = 4, 115200
uart = UART(UART_ID, BAUD)
_rx = bytearray()
last = None
n_frames = 0
n_bad = 0
t_hb = time.ticks_ms()

print('UART%d @ %d -- waiting for 0xCF 0x55 frames from the drone...' % (UART_ID, BAUD))
while True:
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
            v = struct.unpack('<hhhh', body)
            cos, sin, yaw, spd = v[0] / 1e4, v[1] / 1e4, v[2] / 1e4, v[3] / 1e4
            brg = math.degrees(math.atan2(sin, cos))          # +ve = goal to drone-LEFT
            last = (cos, sin, yaw, spd, brg)
            n_frames += 1
            print('RX  bearing %+6.1f deg  (cos %+.3f sin %+.3f)  yawrate/3.5 %+.3f  v/6 %+.3f'
                  % (brg, cos, sin, yaw, spd))
            i += 11
        else:
            n_bad += 1
            i += 1
    _rx = _rx[i:]
    # heartbeat every ~2 s so you can tell it's alive even with no data / bad frames
    if time.ticks_diff(time.ticks_ms(), t_hb) > 2000:
        if n_frames == 0:
            print('...no valid frames yet (%d bad bytes, %d in buffer). Check wiring / baud / that the CF is sending.'
                  % (n_bad, len(_rx)))
        t_hb = time.ticks_ms()
    time.sleep_ms(20)
