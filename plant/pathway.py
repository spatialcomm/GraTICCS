"""Drawn contact channels planted on pathway/TF readouts: each channel acts on the TF of its first CollecTRI route, or, when
only literature supports it, on the first readout its receptor family allows. Y = [protein readouts, pathway/TF readouts]."""
import json

import numpy as np

from graticcs.paths import BREAST
from plant.common import add_signals, contact_inputs, save, table, zscore

D = BREAST; PW = D / "pathway_tf"
z, meta, names, _, A, _ = contact_inputs(D)
pm = json.load(open(PW / "meta.json")); T = meta["prot_readout"] + pm["readouts"]; Mk = np.load(PW / "mask_contact.npy")
Y0 = np.c_[z["Yp"], np.load(PW / "Y_pathway_tf.npz")["Y"]].astype(np.float64); npr = z["Yp"].shape[1]
routes = table("planted_random_route_refs.csv"); routes = routes[routes.modality == "contact"]

plant = []
for r in routes.itertuples():
    c = names.index(r.channel); allowed = {T[npr + k] for k in np.flatnonzero(Mk[c])}
    if isinstance(r.collectri_routes, str) and r.collectri_routes:
        tf = "TF_" + r.collectri_routes.split(";")[0].split("/")[1].split(" ")[0]
    else:
        tf = sorted(allowed)[0] if allowed else None
    if tf is None:
        continue
    assert tf in allowed, (r.channel, tf)
    plant.append((r.channel, tf, int(r.sign)))
print("planted (channel, readout):", plant)

sig = {}
for chn, tgt, sgn in plant:
    c = names.index(chn)
    sig[chn] = zscore(np.maximum(z["R_mo"][:, c], 0) * (A @ np.maximum(z["L_mo"][:, c], 0)))
save(D, "spike_crand_pw_beta", add_signals(Y0, T, plant, sig, 0.2), 0.2)
json.dump({"plant": plant, "betas": [0.2], "tnames": T}, open(D / "spike_crand_pw_meta.json", "w"), indent=1)
