"""Agreement between the predicted and the introduced effect in individual receivers, for every drawn contact channel.

Predicted: the channel's pair scores summed over each receiver's incoming contact edges (12 evaluation fields, three-seed
GraTICCS ensemble). Introduced: the standardized exposure R+ x sum of contacting L+ used for planting. Pearson r over all
receivers of the 12 fields and over exposed receivers only.

python -m analysis.per_cell [beta]   ->  results/percell_agreement_crand_b{beta}.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from analysis.pair_level import D, R, cdst, ch, pair_scores, real, z

B = sys.argv[1] if len(sys.argv) > 1 else "0.1"
plant = json.load(open(D / "spike_crand_meta.json"))["plant"]
S = pair_scores([R / f"G_crand_b{B}_s{s}" for s in range(3)])
n = len(z["fov"]); rc = np.unique(cdst); cc = z["contact"] > 0
names = ch.name.to_numpy(); rows = []
for lr, tgt, sgn in plant:
    c = int(np.flatnonzero(names == lr)[0]); j = list(real).index(c)
    Lp = np.maximum(z["L_mo"][:, c], 0); Rp = np.maximum(z["R_mo"][:, c], 0)
    x = Rp * np.bincount(z["dst"][cc], weights=Lp[z["src"][cc]], minlength=n); ze = (x - x.mean()) / x.std()
    pred = np.bincount(cdst, weights=S[:, j], minlength=n)
    ex = rc[x[rc] > 0]
    rows.append(dict(channel=lr, readout=tgt, sign=sgn, receivers=len(rc), exposed=len(ex),
                     r_all=pearsonr(pred[rc], ze[rc])[0], r_exposed=pearsonr(pred[ex], ze[ex])[0] if len(ex) > 2 else np.nan))
d = pd.DataFrame(rows); d.to_csv(R / f"percell_agreement_crand_b{B}.csv", index=False)
print(d.round(3).to_string(index=False))
print("median r", round(d.r_all.median(), 3), "range", round(d.r_all.min(), 3), round(d.r_all.max(), 3))
