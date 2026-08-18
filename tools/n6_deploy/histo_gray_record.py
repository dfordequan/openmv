# histo_gray_record.py -- record the GenX320 grayscale event histogram (single-channel snapshot mode)
# for REFERENCE. Saves the 320x320 snapshot + a 32x32 AREA-downsample (upscaled 10x for viewing) as BMPs
# on /sdcard. NO read_events, NO ulab histogram math -- this is the sensor's native on-chip histogram.
#
# NOTE ON CURRENT FIRMWARE: post_process_histo was modded to RAW passthrough, so the snapshot is the
# RAW histogram (~binary: 0 background, events near 0/255), NOT the stock graded 128-centred net.
# For the graded net either (a) reconstruct in SW: net=snapshot.to_ndarray('b') (int8); gray=clip(net*16+128,0,255),
# or (b) restore the stock USAT(int8*contrast+128) in drivers/sensors/genx320.c and reflash.
import csi, image, time

N = 15
_c = csi.CSI(cid=csi.GENX320); _c.reset()
_c.pixformat(csi.GRAYSCALE); _c.framesize((320, 320))
_c.snapshot(time=800)                                    # settle
_s32 = image.Image(32, 32, image.GRAYSCALE)              # the 32x32 obs-resolution downsample
_up = image.Image(320, 320, image.GRAYSCALE)             # 32 upscaled (nearest) so it's viewable

print('recording %d grayscale histogram frames -> /sdcard/gray*.bmp' % N)
for k in range(N):
    t0 = time.ticks_us()
    d = _c.snapshot()                                    # 320x320 single-channel histogram
    t1 = time.ticks_us()
    _s32.draw_image(d, 0, 0, x_scale=32 / 320, y_scale=32 / 320, hint=image.AREA)   # faithful downsample
    t2 = time.ticks_us()
    d.save('/sdcard/gray320_%02d.bmp' % k)               # full-res histogram (reference)
    _up.draw_image(_s32, 0, 0, x_scale=10, y_scale=10)   # nearest upscale to see the 32x32 obs
    _up.save('/sdcard/gray32_%02d.bmp' % k)
    print('  f%d  snapshot %d us  downsample(AREA) %d us' % (k, time.ticks_diff(t1, t0), time.ticks_diff(t2, t1)))
    time.sleep_ms(100)
print('done -> /sdcard/gray320_*.bmp (full) and gray32_*.bmp (32x32 obs, upscaled). Pull via the IDE.')
