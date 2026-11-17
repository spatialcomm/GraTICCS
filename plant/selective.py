"""Planted signals carried by only some of the neighbours that express the ligand.

python -m plant.selective sender_type            contact: only senders of one cell type contribute
python -m plant.selective contact_extent         contact: each sender contributes in proportion to the shared boundary
python -m plant.selective secreted_sender_type   secreted: only senders of one cell type contribute

Sender type: candidate types (not 'other') carry 10-60% of the channel's L x R total and leave enough receivers exposed
when the sum is restricted to them; one candidate per channel is drawn with seed 20261112, and channels without a candidate
are not planted. Secreted distance selectivity is part of every secreted planting (exp(-d / 15 um) kernel, plant/secreted.py).
"""
import json
import sys

import numpy as np

from graticcs.paths import BREAST
from plant.common import add_signals, save, table, zscore

KIND = sys.argv[1]
D = BREAST
z = np.load(D / "gt_inputs.npz"); cmeta = json.load(open(D / "gt_channels.json")); TYPES = cmeta["types"]
ct = z["ctype"]; n = len(ct)

if KIND == "secreted_sender_type":
    S = np.load(D / "gt_sec_inputs.npz"); meta = json.load(open(D / "gt_sec_channels.json"))
    src, dst, kern = S["src"], S["dst"], np.exp(-S["dist"] / 15.0)
    Lm, Rm, Y0 = S["L"], S["R"], np.c_[S["Yp"], S["Yr"]].astype(np.float64)
    P = table("planted_random_secreted.csv")
    betas, min_exposed, prefix = [0.2], 0.02, "spike_sec_ctype"
else:
    meta = cmeta; ci = z["contact"] > 0
    src, dst, area = z["src"][ci], z["dst"][ci], z["contact"][ci]
    kern = None
    Lm, Rm, Y0 = z["L_mo"], z["R_mo"], np.c_[z["Yp"], z["Yr"]].astype(np.float64)
    P = table("planted_random_contact.csv")
    betas, min_exposed = [0.1, 0.2, 0.4], 0.005
    prefix = "spike_carea" if KIND == "contact_extent" else "spike_ctype"
names = [c["name"] for c in meta["channels"]]; tnames = meta["prot_readout"] + meta["rna_programs"]

rng = np.random.default_rng(20261112)
plant, sig = [], {}
for r in P.itertuples():
    c = names.index(r.channel)
    L = np.maximum(Lm[:, c].astype(np.float32), 0); R = np.maximum(Rm[:, c].astype(np.float32), 0)
    if KIND == "contact_extent":
        e = np.maximum(Lm[:, c], 0)[src] * np.maximum(Rm[:, c], 0)[dst]
        x = np.bincount(dst, e * area, n); sig[r.channel] = zscore(x)
        plant.append((r.channel, r.readout, int(r.sign)))
        print(f"{r.channel:22s} -> {r.readout:9s} exposed {(x > 0).mean():.3f} "
              f"corr with equal weights {np.corrcoef(x, np.bincount(dst, e, n))[0, 1]:.2f}")
        continue
    contrib = L[src] * R[dst] if kern is None else kern * L[src] * R[dst]
    share = np.bincount(ct[src], contrib, len(TYPES)) / contrib.sum(); x_all = np.bincount(dst, contrib, n)
    cand = []
    for t in range(len(TYPES)):
        if TYPES[t] == "other" or not 0.10 <= share[t] <= 0.60:
            continue
        xt = np.bincount(dst, contrib * (ct[src] == t), n)
        if (xt > 0).mean() >= min_exposed:
            cand.append((t, xt))
    if not cand:
        print(f"{r.channel:22s} no candidate sender type"); continue
    t, xt = cand[rng.integers(len(cand))]
    sig[r.channel] = zscore(xt); plant.append([r.channel, r.readout, int(r.sign), TYPES[t]])
    print(f"{r.channel:22s} -> {r.readout:9s} sender {TYPES[t]:11s} share {share[t]:.2f} exposed {(xt > 0).mean():.3f} "
          f"corr with all senders {np.corrcoef(xt, x_all)[0, 1]:.2f}")

for b in betas:
    save(D, f"{prefix}_beta", add_signals(Y0, tnames, plant, sig, b), b)
info = {"plant": plant, "betas": betas, "tnames": tnames}
if KIND == "contact_extent":
    info["contact_median"] = float(np.median(area))
json.dump(info, open(D / f"{prefix}_meta.json", "w"), indent=1)
print("planted", len(plant))
