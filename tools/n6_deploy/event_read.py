"""Read a raw event capture (events.bin from event_record_main.py) OFFLINE on the laptop.
Prints stats (true rate, drops), identifies ON/OFF polarity codes, reconstructs 64x64 ON/OFF frames
over a chosen window, reports the raw count distribution (-> CLIP_DEV), and saves a viz PNG.
Usage: python event_read.py events.bin [win_ms]
"""
import sys, struct
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else 'events.bin'
WIN_MS = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
SENS, G = 320, 64

data = open(PATH, 'rb').read()
assert data[:8] == b'GENXEV01', 'bad magic (not an events.bin from event_record_main.py)'
off = 8
recs = []                                   # list of (t_us, events[n,6])
while off + 8 <= len(data):
    n, t_us = struct.unpack_from('<II', data, off); off += 8
    nb = n * 6 * 2
    if off + nb > len(data):
        break
    ev = np.frombuffer(data, np.uint16, count=n * 6, offset=off).reshape(n, 6); off += nb
    recs.append((t_us, ev))
assert recs, 'no records parsed'
allev = np.concatenate([r[1] for r in recs], 0).astype(np.int64)
tot = allev.shape[0]

# ---- stats ----
t_us = np.array([r[0] for r in recs], np.int64)
wall = (t_us[-1] - t_us[0]) / 1e6 if len(t_us) > 1 else 0.0
# sensor timestamp (cols 1,2,3 = s, ms, us) -> absolute microseconds
ets = allev[:, 1] * 1_000_000 + allev[:, 2] * 1000 + allev[:, 3]
span = (ets.max() - ets.min()) / 1e6 if tot > 1 else 0.0
print(f'file {PATH}: {len(recs)} reads, {tot} events')
print(f'  wall duration {wall:.2f}s  -> captured {tot/max(wall,1e-6):.0f} ev/s')
print(f'  sensor-timestamp span {span:.2f}s  -> {tot/max(span,1e-6):.0f} ev/s (in-stream)')
print(f'  events/read: mean {tot/len(recs):.0f}  max {max(r[1].shape[0] for r in recs)}  '
      f'(if max hits EVT_res=65536, the buffer was saturating -> drops)')
codes, cnts = np.unique(allev[:, 0], return_counts=True)
print('  polarity codes (col0):', dict(zip(codes.tolist(), cnts.tolist())))

# ON = most common code, OFF = second (GenX320 PIX_ON/PIX_OFF); adjust if needed
order = np.argsort(cnts)[::-1]
ON_CODE = int(codes[order[0]]); OFF_CODE = int(codes[order[1]]) if len(codes) > 1 else ON_CODE
print(f'  -> treating ON=col0=={ON_CODE}, OFF=col0=={OFF_CODE}')

# ---- reconstruct 64x64 ON/OFF frames from fixed EVENT-COUNT chunks (robust; per-event ts unreliable) ----
rate = tot / max(span, 1e-6)
CHUNK = max(int(rate * WIN_MS / 1000), 2000)     # ~WIN_MS worth of events
def frame(off):
    sub = allev[off:off + CHUNK]
    gx = (sub[:, 4] * G // SENS).clip(0, G - 1); gy = (sub[:, 5] * G // SENS).clip(0, G - 1)
    on = np.zeros((G, G)); off_ = np.zeros((G, G))
    m_on = sub[:, 0] == ON_CODE; m_off = sub[:, 0] == OFF_CODE
    np.add.at(on, (gy[m_on], gx[m_on]), 1); np.add.at(off_, (gy[m_off], gx[m_off]), 1)
    return on, off_, sub.shape[0]
spots = {'start': 0, 'middle': tot // 2, 'end': max(tot - CHUNK, 0)}
print(f'\nreconstructing frames from {CHUNK}-event chunks (~{WIN_MS:.0f}ms each):')
frames = {}
for name, o in spots.items():
    on, off_, n = frame(o); frames[name] = (on, off_)
    nz = on[on > 0]; p = lambda q: float(np.percentile(nz, q)) if nz.size else 0.0
    print(f'  {name:6s}: ON per-cell p50={p(50):.0f} p95={p(95):.0f} p99={p(99):.0f} max={on.max():.0f}'
          f'  active={100*np.mean(on > 0):.0f}%  (CLIP_DEV~p99)')
print('  NOTE: SUM-binned counts here; on-device AREA downscale AVERAGES over ~25 px -> divide by ~25.')

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(13, 4.5))
for j, name in enumerate(['start', 'middle', 'end']):
    on, off_ = frames[name]; mx = max(on.max(), off_.max(), 1)
    rgb = np.zeros((G, G, 3)); rgb[..., 2] = on / mx; rgb[..., 0] = off_ / mx
    rgb[..., 1] = 0.12 * (on + off_) / mx
    ax[j].imshow(rgb); ax[j].axis('off'); ax[j].set_title(f'{name}  (ON=blue OFF=red, ~{WIN_MS:.0f}ms)')
fig.suptitle(f'{PATH}: reconstructed 64x64 event frames  ({tot} events, {rate:.0f} ev/s)')
out = PATH.rsplit('.', 1)[0] + '_frames.png'
fig.tight_layout(); fig.savefig(out, dpi=110); print('\nsaved', out)
