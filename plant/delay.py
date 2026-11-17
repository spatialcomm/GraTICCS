"""Planted signals with a delay between ligand exposure and response.

The response follows a past ligand state L_past = rho L_now + sqrt(1 - rho^2) L_perm, where L_perm permutes the ligand
within imaging field and cell type; the model only sees L_now.
python -m plant.delay contact | secreted
"""
import json
import sys

import numpy as np

from graticcs.paths import BREAST
from plant.common import add_signals, contact_inputs, drawn, save, secreted_inputs, within_group_permutation, zscore

KIND = sys.argv[1]
D = BREAST
plant = drawn(KIND)
zc = np.load(D / "gt_inputs.npz")
if KIND == "contact":
    z, meta, names, tnames, W, Y0 = contact_inputs(D); L_now, R = z["L_mo"], z["R_mo"]; prefix = "spike_crand_lag_"
else:
    S, m, names, tnames, W, Y0 = secreted_inputs(D); L_now, R = S["L"], S["R"]; prefix = "spike_sec_lag_"
assert len(zc["fov"]) == len(Y0)
perm = within_group_permutation(zc["fov"], zc["ctype"])

past = {}
for rho in [0.7, 0.5, 0.3]:
    for b in [0.2, 0.4]:
        sig = {}
        for chn, tgt, sgn in plant:
            c = names.index(chn); Ln = L_now[:, c].astype(np.float64)
            Lp = zscore(rho * Ln + np.sqrt(1 - rho ** 2) * Ln[perm]); past[(rho, chn)] = Lp.astype(np.float32)
            sig[chn] = zscore(np.maximum(R[:, c].astype(np.float64), 0) * (W @ np.maximum(Lp, 0)))
        save(D, f"{prefix}r{rho}_b", add_signals(Y0, tnames, plant, sig, b), b)
    print("rho", rho, "corr(L_past, L_now):",
          [round(float(np.corrcoef(past[(rho, p[0])], L_now[:, names.index(p[0])])[0, 1]), 3) for p in plant])
np.savez_compressed(D / f"{prefix}past_ligands.npz", **{f"r{r}__{c.replace('->', '__')}": v for (r, c), v in past.items()})
json.dump({"plant": plant, "tnames": tnames}, open(D / f"{prefix}meta.json", "w"), indent=1)
