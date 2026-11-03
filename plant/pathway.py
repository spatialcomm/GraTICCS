"""Planted contact signals on pathway/TF readouts allowed by the receptor family."""
import json
from pathlib import Path
import numpy as np, scipy.sparse as sp
from graticcs.paths import BREAST
D = BREAST; PW = D / "pathway_tf"
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); pm = json.load(open(PW / "meta.json"))
names = [c["name"] for c in meta["channels"]]; T = meta["prot_readout"] + pm["readouts"]; Mk = np.load(PW / "mask_contact.npy")
Y0 = np.c_[z["Yp"], np.load(PW / "Y_pathway_tf.npz")["Y"]].astype(np.float64); npr = z["Yp"].shape[1]
want = ["CD274->PDCD1", "HLA-DRA->CD4", "ICAM1->ITGAM_ITGB2", "TNFSF13B->CD40", "LGALS9->HAVCR2", "CDH1->EGFR", "CD70->CD27"]
PLANT = []
for w in want:
    if w not in names: continue
    c = names.index(w); allowed = [T[npr + k] for k in np.flatnonzero(Mk[c])]
    if allowed: PLANT.append((w, allowed[0], +1))
print("planted (channel, readout):", PLANT)
n = len(z["fold"]); src, dst, con = z["src"], z["dst"], z["contact"]; ci = con > 0
A = sp.csr_matrix((np.ones(ci.sum(), np.float32), (dst[ci], src[ci])), shape=(n, n))
for b in [0.2]:
    Y = Y0.copy()
    for chn, tgt, sg in PLANT:
        c = names.index(chn); x = np.maximum(z["R_mo"][:, c], 0) * (A @ np.maximum(z["L_mo"][:, c], 0)); x = (x - x.mean()) / (x.std() + 1e-12)
        t = T.index(tgt); Y[:, t] += sg * b * Y0[:, t].std() * x
    np.savez_compressed(D / f"spike_pw_beta{b}.npz", Y=Y.astype(np.float32))
json.dump({"plant": PLANT, "betas": [0.2], "tnames": T}, open(D / "spike_pw_meta.json", "w"), indent=1); print("saved")
