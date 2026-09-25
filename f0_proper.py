import glob, re
import numpy as np
from scipy.signal import welch

for d in sorted(glob.glob('data_background_rate_big/*/'),
                key=lambda s: float(re.search(r'[\d.]+', s).group()))[::6]:
    a = np.loadtxt(d + 'trial_0/measurements/pop_activities/pop_activity_2.dat')
    f, P = welch(a - a.mean(), fs=5000.0, nperseg=65536)
    b = (f > 1) & (f < 200)          # no artificial lower cutoff
    f0 = f[b][P[b].argmax()]
    print(f"{float(re.search(r'[0-9.]+', d).group()):>6.2f}  f0 = {f0:6.2f} Hz  "
          f"period = {1000/f0:7.1f} ms")
