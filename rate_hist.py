import numpy as np
z = np.load('state.npz')
ref = z['rates'][:, :, z['ref_pop']]
dr = z['drives']

low = dr < 7.8
v = ref[low].ravel()
print(f"{v.size} trials below 7.8 spikes/s")
print("percentiles:", np.round(np.percentile(v, [0, 5, 25, 50, 75, 95, 100]), 3))
print(f"exact zeros: {(v == 0).sum()}")
h, e = np.histogram(v, bins=20)
for c, lo, hi in zip(h, e[:-1], e[1:]):
    print(f"  [{lo:5.2f},{hi:5.2f})  {'#' * c}  {c}")
