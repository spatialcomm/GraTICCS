"""Planted signals that follow a past ligand state L_past = rho L + sqrt(1 - rho^2) L_perm."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp
from graticcs.paths import BREAST
D = BREAST
sm = json.load(open(D / "spike_meta.json")); PLANT = sm["plant"]
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json"))
names = [c["name"] for c in meta["channels"]]; tnames = meta["prot_readout"] + meta["rna_programs"]
n = len(z["fold"]); src, dst, con = z["src"], z["dst"], z["contact"]; ci = con > 0
A = sp.csr_matrix((np.ones(ci.sum(), np.float32), (dst[ci], src[ci])), shape=(n, n))
Y0 = np.c_[z["Yp"], z["Yr"]].astype(np.float64)
rng = np.random.default_rng(7)
groups = {}
for i, key in enumerate(zip(z["fov"], z["ctype"])): groups.setdefault(key, []).append(i)
perm = np.arange(n)
for idx in groups.values():
    idx = np.array(idx); perm[idx] = idx[rng.permutation(len(idx))]
past = {}
for rho in [0.7, 0.5, 0.3]:
    for b in [0.2, 0.4]:
        Y = Y0.copy()
        for chn, tgt, sgn in PLANT:
            c = names.index(chn); Ln = z["L_mo"][:, c].astype(np.float64); Lp = rho * Ln + np.sqrt(1 - rho ** 2) * Ln[perm]
            Lp = (Lp - Lp.mean()) / (Lp.std() + 1e-12); past[(rho, chn)] = Lp.astype(np.float32)
            x = np.maximum(z["R_mo"][:, c], 0) * (A @ np.maximum(Lp, 0)); x = (x - x.mean()) / (x.std() + 1e-12)
            t = tnames.index(tgt); Y[:, t] += sgn * b * Y0[:, t].std() * x
        np.savez_compressed(D / f"spike_lag_r{rho}_b{b}.npz", Y=Y.astype(np.float32))
    print("rho", rho, "corr(L_past, L_now) for planted:", [round(float(np.corrcoef(past[(rho, p[0])], z["L_mo"][:, names.index(p[0])])[0, 1]), 3) for p in PLANT])
np.savez_compressed(D / "spike_lag_past_ligands.npz", **{f"r{r}__{c.replace('->', '__')}": v for (r, c), v in past.items()})
print("saved")
