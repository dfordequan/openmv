# snapshot_bench.py -- measure GenX320 snapshot() time vs framebuffers count + set_framerate.
# Goal: find whether double-buffering (free-run) and/or a 50fps/20ms rate cut the ~33ms snapshot.
import csi, image, time
us = time.ticks_us; di = time.ticks_diff
SRC = 320
_c = csi.CSI(cid=csi.GENX320)

def cfg(nfb, fps):
    _c.reset()
    _c.pixformat(csi.GRAYSCALE); _c.framesize((SRC, SRC))
    _c.framebuffers(nfb)
    fr = 'default'
    if fps is not None:
        try:
            _c.set_framerate(fps); fr = str(fps)
        except Exception as e:
            fr = 'FRfail'
    _c.snapshot(time=800)                      # settle
    return fr

def bench(N=25):
    for _ in range(5):                         # warm / fill buffers
        _c.snapshot()
    ts = []
    for _ in range(N):
        t = us(); _c.snapshot(); ts.append(di(us(), t))
    ts = sorted(ts)
    return ts[0], ts[len(ts)//2], sum(ts)//len(ts)   # min, median, mean (us)

print('config          | snapshot ms  min / median / mean')
for nfb, fps in [(1, None), (1, 50), (1, 100), (2, None), (2, 50), (2, 100), (3, 50)]:
    try:
        fr = cfg(nfb, fps); mn, md, me = bench()
        print(' fb%d fr=%-7s | %5.1f / %5.1f / %5.1f' % (nfb, fr, mn/1000, md/1000, me/1000))
    except Exception as e:
        print(' fb%d fr=%s ERR: %s' % (nfb, fps, e))
    time.sleep_ms(150)
print('done')
