"""Planted responses whose form differs from the model (switch-like, binary gate, proximity)."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp
from graticcs.paths import BREAST
D = BREAST
sm = json.load(open(D / "spike_meta.json")); PLANT = sm["plant"]
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json"))
names = [c["name"] for c in meta["channels"]]; tnames = meta["prot_readout"] + meta["rna_programs"]
n = len(z["fold"]); src, dst, con, dist = z["src"], z["dst"], z["contact"] > 0, z["dist"]
A = sp.csr_matrix((np.ones(con.sum(), np.float32), (dst[con], src[con])), shape=(n, n))
W = sp.csr_matrix((np.exp(-dist / 8.0).astype(np.float32), (dst, src)), shape=(n, n))
Y0 = np.c_[z["Yp"], z["Yr"]].astype(np.float64)
def zs(x): return (x - x.mean()) / (x.std() + 1e-12)
def maxnb(L):
    m = np.zeros(n); np.maximum.at(m, dst[con], L[src[con]]); return m
sig = {}
for chn, tgt, sgn in PLANT:
    c = names.index(chn); L = np.maximum(z["L_mo"][:, c], 0).astype(np.float64); R = np.maximum(z["R_mo"][:, c], 0).astype(np.float64)
    x = R * (A @ L); K = np.median(x[x > 0]); h = x ** 2 / (K ** 2 + x ** 2)
    g = ((R > np.quantile(R[R > 0], 0.75)) & (maxnb(L) > np.quantile(L[L > 0], 0.75))).astype(float)
    dcy = R * (W @ L)
    sig[chn] = {"hill": zs(h), "gate": zs(g), "decay": zs(dcy)}
    std_form = zs(x)
    print(f"{chn:22s} gate-on {g.mean():.3f}  corr with standard form: hill {np.corrcoef(h, x)[0,1]:.2f} gate {np.corrcoef(g, x)[0,1]:.2f} decay {np.corrcoef(dcy, x)[0,1]:.2f}")
for form in ["hill", "gate", "decay"]:
    for b in [0.2, 0.4]:
        Y = Y0.copy()
        for chn, tgt, sgn in PLANT:
            t = tnames.index(tgt); Y[:, t] += sgn * b * Y0[:, t].std() * sig[chn][form]
        np.savez_compressed(D / f"spike_mis_{form}_b{b}.npz", Y=Y.astype(np.float32))
print("saved")
