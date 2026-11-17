"""Simplest channel-level comparator on the contact draw: no background model and no training.

For every channel (real and decoy), x_i = R_i x sum of L_j over contacting neighbours j, the planted form itself, is
correlated with each readout across all cells; importance is the squared correlation, with the same channel x readout
mask as the model. Variant 'cell type removed' centres x and readouts within cell type first. Recovery, additional
selections and AUROC follow analysis/recovery.py. Writes results/naive_baseline_crand.csv.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from analysis.summary import CH
from graticcs.data import load, target_mask
from graticcs.paths import BREAST, RESULTS

B = BREAST; R = RESULTS
ch = CH["c"]; real = ~ch.decoy.to_numpy(); BETAS = (0.05, 0.1, 0.2, 0.4)
PLANT = json.load(open(B / "spike_crand_meta.json"))["plant"]; PL = {p[0] for p in PLANT}
pidx = {c: ch.index[(ch.name == c) & ~ch.decoy][0] for c in PL}
d = load("mo"); Mk = target_mask(d); T = d["tnames"]
z = np.load(B / "gt_inputs.npz"); n = len(z["fold"]); ci = z["contact"] > 0; s_, d_ = z["src"][ci], z["dst"][ci]; ctype = z["ctype"]
X = np.zeros((n, len(ch)), np.float32)
Lall = np.maximum(z["L_mo"], 0); Rall = np.maximum(z["R_mo"], 0)
for c in range(len(ch)):
    X[:, c] = Rall[:, c] * np.bincount(d_, weights=Lall[:, c][s_], minlength=n)


def centre(A, by=None):
    A = A.astype(np.float64)
    if by is not None:
        for k in np.unique(by):
            A[by == k] -= A[by == k].mean(0)
    A -= A.mean(0); sd = A.std(0); sd[sd == 0] = 1
    return A / sd


def evaluate(imp):
    ib = imp.max(1); q = np.ones(len(ch))
    for st in ("protein", "rna"):
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        p = [(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]; q[rl] = multipletests(p, method="fdr_bh")[1]
    rec = strict = 0
    for chn, tgt, sg in PLANT:
        c = pidx[chn]; t = T.index(tgt); st = ch.stratum[c]
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        others = rl[~ch.name[rl].isin(PL)]
        pp = (1 + (imp[dec, t] >= imp[c, t]).sum()) / (1 + len(dec))
        rec += pp <= 0.05 and int((imp[others, t] > imp[c, t]).sum()) + 1 <= 3
        strict += pp <= 0.05 and int((imp[rl, t] > imp[c, t]).sum()) + 1 <= 3
    hits = set(ch.name[real & (q <= 0.1)]) - PL; planted = ch.name[real].isin(PL).to_numpy()
    return dict(recovered=int(rec), recovered_strict=int(strict), FP=len(hits), channel_AUROC=round(roc_auc_score(planted, ib[real]), 3))


rows = []
for variant, by in (("raw", None), ("cell type removed", ctype)):
    Xs = centre(X, by)
    for b in BETAS:
        Ys = centre(np.load(B / f"spike_crand_beta{b}.npz")["Y"], by)
        rows.append(dict(variant=variant, beta=b, **evaluate(((Xs.T @ Ys) / n) ** 2 * Mk)))
    Y0 = centre(np.c_[z["Yp"], z["Yr"]], by)
    rows.append(dict(variant=variant, beta=0.0, **evaluate(((Xs.T @ Y0) / n) ** 2 * Mk)))
out = pd.DataFrame(rows); out.to_csv(R / "naive_baseline_crand.csv", index=False)
print(out.to_string(index=False))
