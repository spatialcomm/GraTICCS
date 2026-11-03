"""Planted contact signals with RNA-defined ligand and receptor."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp
from graticcs.paths import BREAST

D = BREAST
PLANT = [("CD274->PDCD1", "GZMB", -1), ("HLA-DRA->CD4", "Ki-67", +1), ("ICAM1->ITGAM_ITGB2", "iNOS", +1),
         ("CD70->CD27", "CD38", +1), ("LGALS9->HAVCR2", "GZMA", -1), ("TNFSF13B->CD40", "NF-kB p65", +1),
         ("HLA-A->CD8A", "CYTOTOX", +1)]
BETAS = [0.1, 0.4]
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json"))
names = [c["name"] for c in meta["channels"]]; tnames = meta["prot_readout"] + meta["rna_programs"]
n = len(z["fold"]); src, dst, con = z["src"], z["dst"], z["contact"]
ci = con > 0
A = sp.csr_matrix((np.ones(ci.sum(), np.float32), (dst[ci], src[ci])), shape=(n, n))
Y0 = np.c_[z["Yp"], z["Yr"]].astype(np.float64)
expo = {}
for chn, tgt, sgn in PLANT:
    c = names.index(chn); L = np.maximum(z["L_rna"][:, c], 0); R = np.maximum(z["R_rna"][:, c], 0)
    x = R * (A @ L); expo[chn] = (x - x.mean()) / (x.std() + 1e-12)
    print(f"{chn:22s} -> {tgt:9s} exposed cells {(x > 0).mean():.3f}")
for b in BETAS:
    Y = Y0.copy()
    for chn, tgt, sgn in PLANT:
        t = tnames.index(tgt); Y[:, t] += sgn * b * Y0[:, t].std() * expo[chn]
    np.savez_compressed(D / f"spike_rna_beta{b}.npz", Y=Y.astype(np.float32))
json.dump({"plant": PLANT, "betas": BETAS, "tnames": tnames}, open(D / "spike_rna_meta.json", "w"), indent=1)
print("saved")
