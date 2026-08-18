"""Make a GIF from a raw event capture (events.bin). Animates 64x64 ON/OFF frames (ON=blue, OFF=red)
reconstructed from ~CHUNK_MS event windows, stepped across the whole recording.
Usage: python event_gif.py events.bin [n_frames=120] [chunk_ms=16] [fps=15]
"""
import sys, struct
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'events.bin'
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 120
CHUNK_MS = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
FPS = int(sys.argv[4]) if len(sys.argv) > 4 else 15
SENS, G = 320, 64

data = open(PATH, 'rb').read()
assert data[:8] == b'GENXEV01', 'bad magic'
off = 8; blocks = []; ts = []
while off + 8 <= len(data):
    n, t = struct.unpack_from('<II', data, off); off += 8
    if not (1 <= n <= 65536):                        # corrupt header -> stop (salvage valid prefix)
        print(f'  stop: corrupt record header n={n} at read {len(blocks)}'); break
    nb = n * 6 * 2
    if off + nb > len(data):
        break
    b = np.frombuffer(data, np.uint16, count=n * 6, offset=off).reshape(n, 6); off += nb
    if ((b[:, 4] > 319) | (b[:, 5] > 319) | (b[:, 0] > 3)).mean() > 0.3:   # mostly-invalid -> corrupt
        print(f'  stop: {len(blocks)} reads, next is corrupt (bad x/y/type)'); break
    blocks.append(b); ts.append(t)
allev = np.concatenate(blocks, 0); tot = allev.shape[0]
codes, cnts = np.unique(allev[:, 0], return_counts=True); order = np.argsort(cnts)[::-1]
ON, OFF = int(codes[order[0]]), int(codes[order[1]])
print(f'{PATH}: {tot} events. ON={ON} OFF={OFF}  (ticks_us wrapped -> using event-count windows)')

# CONTIGUOUS (non-overlapping) windows that TILE the whole stream -> gap-free playback in order.
# chunk = stride = tot / N_FRAMES, so more frames -> shorter windows -> crisper + continuous.
CHUNK = max(tot // N_FRAMES, 1000)
STRIDE = CHUNK
gx_all = (allev[:, 4] * G // SENS).clip(0, G - 1)
gy_all = (allev[:, 5] * G // SENS).clip(0, G - 1)
is_on = allev[:, 0] == ON; is_off = allev[:, 0] == OFF

nf = min(N_FRAMES, (tot // STRIDE))
if nf < 1:
    raise SystemExit(f'too few events ({tot}) for even one {CHUNK}-event frame; record more or lower N_FRAMES')
def build(s, e):
    on = np.zeros((G, G)); of = np.zeros((G, G))
    mo = is_on[s:e]; mf = is_off[s:e]
    np.add.at(on, (gy_all[s:e][mo], gx_all[s:e][mo]), 1)
    np.add.at(of, (gy_all[s:e][mf], gx_all[s:e][mf]), 1)
    return on, of
# global scale from ~80 sampled frames (so density changes stay visible; avoids holding all frames)
samp = []
for i in np.linspace(0, nf - 1, min(80, nf)).astype(int):
    on, of = build(i * STRIDE, i * STRIDE + CHUNK); samp.append(max(on.max(), of.max()))
scale = max(float(np.percentile(samp, 90)), 1.0)
print(f'{nf} frames, chunk={CHUNK} ev, stride={STRIDE} ev (CONTIGUOUS), scale={scale:.1f}')

from PIL import Image
UP = 2                                               # 64 -> 128 px for visibility
imgs = []
for i in range(nf):
    on, of = build(i * STRIDE, i * STRIDE + CHUNK)
    rgb = np.zeros((G, G, 3), np.uint8)
    rgb[..., 2] = np.clip(on / scale * 255, 0, 255)      # ON  = blue
    rgb[..., 0] = np.clip(of / scale * 255, 0, 255)      # OFF = red
    rgb[..., 1] = np.clip(0.12 * (on + of) / scale * 255, 0, 255)
    imgs.append(Image.fromarray(rgb, 'RGB').resize((G * UP, G * UP), Image.NEAREST))
out = PATH.rsplit('.', 1)[0] + '.gif'
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0)
import os
print(f'saved {out}  ({os.path.getsize(out)/1e6:.1f} MB, {nf} frames @ {FPS}fps = {nf/FPS:.0f}s)')
