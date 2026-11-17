"""Planted secreted signals for the drawn channels: x = R_i * sum over senders j within 30 um of exp(-d_ij / 15 um) L_j."""
import json

import numpy as np

from graticcs.paths import BREAST
from plant.common import add_signals, drawn, save, secreted_inputs, zscore

D = BREAST
BETAS = [0.05, 0.1, 0.2, 0.4]
plant = drawn("secreted")
S, m, names, tnames, W, Y0 = secreted_inputs(D)
sig = {}
for chn, tgt, sgn in plant:
    c = names.index(chn)
    L = np.maximum(S["L"][:, c].astype(np.float32), 0); R = np.maximum(S["R"][:, c].astype(np.float32), 0)
    x = R * (W @ L); sig[chn] = zscore(x)
    print(f"{chn:22s} -> {tgt:10s} exposed {(x > 0).mean():.3f}")
for b in BETAS:
    save(D, "spike_sec_rand_beta", add_signals(Y0, tnames, plant, sig, b), b)
json.dump({"plant": plant, "betas": BETAS, "tnames": tnames}, open(D / "spike_sec_rand_meta.json", "w"), indent=1)
