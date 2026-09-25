import numpy as np, re
from scipy.signal import welch

rows = np.load('te_out/te_full.npy', allow_pickle=True)
U = {r['drive']: r['u'] for r in rows if (r['source'], r['target']) == (3, 2)}

def val(s):
    return float(re.search(r'[\d.]+', s).group())

print(f"{'drive':>7} {'f0 (Hz)':>8} {'T/2 (ms)':>9} {'L4I>L4E u':>10}")
for d in sorted(U, key=val):
    x = np.load(f'te_out/counts_{d}.npz')['counts'][:, 2, :].mean(axis=0)
    f, P = welch(x - x.mean(), fs=1000.0, nperseg=4096)
    b = (f > 20) & (f < 150)
    f0 = f[b][P[b].argmax()]
    half = 1000.0 / (2 * f0)
    print(f"{val(d):>7.2f} {f0:>8.1f} {half:>9.2f} {U[d]:>10}")
