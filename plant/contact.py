"""Planted contact signals for the drawn channels: x = R_i * sum over contacting senders j of L_j.

python -m plant.contact                          # breast, protein-defined L and R where available
python -m plant.contact --rna                    # breast, RNA-defined L and R
python -m plant.contact --xenium                 # renal section, first draw
python -m plant.contact --xenium --draw planted_random_xenium_batch2.csv --prefix spike_crand2_beta
"""
import argparse
import json

import numpy as np

from graticcs.paths import BREAST, XENIUM
from plant.common import add_signals, contact_inputs, drawn, save, zscore

ap = argparse.ArgumentParser()
ap.add_argument("--rna", action="store_true")
ap.add_argument("--xenium", action="store_true")
ap.add_argument("--draw", help="csv of drawn pairs under results/")
ap.add_argument("--prefix")
args = ap.parse_args()

D = XENIUM if args.xenium else BREAST
plant = drawn("xenium" if args.xenium else "contact", args.draw)
if args.rna:
    lk, rk, betas, prefix = "L_rna", "R_rna", [0.1, 0.4], "spike_crand_rna_beta"
else:
    lk, rk, prefix = "L_mo", "R_mo", "spike_crand_beta"
    betas = [0.1, 0.4] if args.xenium else [0.05, 0.1, 0.2, 0.4]
prefix = args.prefix or prefix

z, meta, names, tnames, A, Y0 = contact_inputs(D)
sig = {}
for chn, tgt, sgn in plant:
    c = names.index(chn)
    x = np.maximum(z[rk][:, c], 0) * (A @ np.maximum(z[lk][:, c], 0)); sig[chn] = zscore(x)
    print(f"{chn:22s} -> {tgt:9s} exposed cells {(x > 0).mean():.3f}")
for b in betas:
    save(D, prefix, add_signals(Y0, tnames, plant, sig, b), b)
json.dump({"plant": plant, "betas": betas, "tnames": tnames}, open(D / (prefix.replace("_beta", "") + "_meta.json"), "w"), indent=1)
