"""Planted secreted signals."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp
from graticcs.paths import BREAST
D = BREAST
PLANT = [("IL1B->IL1R1_IL1RAP", "NF-kB p65", 1), ("CXCL12->CXCR4", "Ki-67", 1), ("SPP1->CD44", "iNOS", 1),
         ("CCL5->CCR5", "GZMB", 1), ("VEGFA->NRP1", "ICAM1", 1), ("CXCL10->CXCR3", "CD38", 1)]
BETAS = [0.05, 0.1, 0.2, 0.4]
S = np.load(D / "gt_sec_inputs.npz"); m = json.load(open(D / "gt_sec_channels.json"))
names = [c["name"] for c in m["channels"]]; T = m["prot_readout"] + m["rna_programs"]; n = S["L"].shape[0]
W = sp.csr_matrix((np.exp(-S["dist"] / 15.0), (S["dst"], S["src"])), shape=(n, n))
Y0 = np.c_[S["Yp"], S["Yr"]].astype(np.float64); ex = {}
for chn, tgt, sg in PLANT:
    c = names.index(chn); L = np.maximum(S["L"][:, c].astype(np.float32), 0); R = np.maximum(S["R"][:, c].astype(np.float32), 0)
    x = R * (W @ L); ex[chn] = (x - x.mean()) / (x.std() + 1e-12); print(f"{chn:22s} -> {tgt:10s} exposed {(x > 0).mean():.3f}")
for b in BETAS:
    Y = Y0.copy()
    for chn, tgt, sg in PLANT:
        t = T.index(tgt); Y[:, t] += sg * b * Y0[:, t].std() * ex[chn]
    np.savez_compressed(D / f"spike_sec_beta{b}.npz", Y=Y.astype(np.float32))
json.dump({"plant": PLANT, "betas": BETAS, "tnames": T}, open(D / "spike_sec_meta.json", "w"), indent=1); print("saved")
