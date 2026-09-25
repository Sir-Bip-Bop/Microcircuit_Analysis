import glob, re
import numpy as np

def val(s):
    return float(re.search(r"[\d.]+", s).group())

ds = sorted(glob.glob("data_background_rate_big/*/"), key=val)

print(f"{'drive':>7} {'mean':>7} {'min':>7} {'max':>7}  per-trial")
for d in ds[:20]:
    trials = sorted(glob.glob(d + "trial*"))
    r = np.array([
        np.loadtxt(t + "/measurements/pop_activities/pop_activity_0.dat").mean()
        for t in trials
    ])
    per = " ".join(f"{v:.2f}" for v in r)
    print(f"{val(d):>7.2f} {r.mean():>7.3f} {r.min():>7.3f} {r.max():>7.3f}  {per}")
