"""Dose-response check of the top-ranked original-tissue channels, each on its strongest readout.

Receptor-positive receivers are grouped by imaging field, cell type, neighborhood-density quintile (sum of exp(-d/15 um)
over secreted edges, or number of contacts) and receptor-input decile. Within groups, ranks of neighboring ligand input
and of the readout are correlated and pooled into rho, with a permutation test within groups (1,000 permutations).
Channels with fewer than 50 receptor-positive receivers are tested at the ligand level. Same-receptor decoys are scored
the same way. Supported: q <= 0.1 (BH within the batch), sign of rho as predicted, at most one decoy as large.

python -m analysis.dose_response secreted 0 20 dose_response_top20_v2.csv
python -m analysis.dose_response secreted 20 50 dose_response_sec_rank21_50_v2.csv
python -m analysis.dose_response contact 0 10 dose_response_contact_top10_v2.csv
python -m analysis.dose_response contact 10 50 dose_response_contact_rank11_50_v2.csv
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from statsmodels.stats.multitest import multipletests

from graticcs.paths import BREAST, RESULTS

D, R = BREAST, RESULTS
KIND, START, END, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
NPERM = 1000; rng = np.random.default_rng(0)
z = np.load(D / "gt_inputs.npz"); ctype = z["ctype"]
fov = pd.read_parquet(D / "cells.parquet", columns=["fov"]).fov.to_numpy(); n = len(fov)
if KIND == "contact":
    RUNS = [R / f"G_real_s{s}" for s in range(3)]
    ch = pd.DataFrame(json.load(open(D / "gt_channels.json"))["channels"]); ci = z["contact"] > 0
    S = dict(L=z["L_mo"], R=z["R_mo"]); Y = np.c_[z["Yp"], z["Yr"]].astype(np.float64)
    W = sp.csr_matrix((np.ones(ci.sum()), (z["dst"][ci], z["src"][ci])), shape=(n, n))
    top = pd.read_csv(R / "channel_best_rank_contact_G.csv").iloc[START:END]
else:
    RUNS = [R / f"G2_sec_real_s{s}" for s in range(3)]
    S = np.load(D / "gt_sec_inputs.npz"); ch = pd.DataFrame(json.load(open(D / "gt_sec_channels.json"))["channels"])
    Y = np.c_[S["Yp"], S["Yr"]].astype(np.float64)
    W = sp.csr_matrix((np.exp(-S["dist"] / 15.0).astype(np.float64), (S["dst"], S["src"])), shape=(n, n))
    top = pd.read_csv(R / "channel_best_rank_secreted_G2.csv").iloc[START:END]
fo = [json.load(open(p)) for d in RUNS for p in sorted(d.glob("fold*_mo.json"))]; T = fo[0]["targets"]
V = np.mean([torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]["V"].numpy() for d in RUNS for k in range(5)], 0)
dens = np.asarray(W.sum(1)).ravel()


def qbin(x, k):
    cuts = np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(np.unique(cuts) if KIND == "contact" else cuts, x, side="right")


def rho(c, t, m, use_rec, nperm=NPERM):
    key = pd.Series(fov[m]).astype(str) + "_" + pd.Series(ctype[m]).astype(str) + "_" + pd.Series(qbin(dens[m], 5)).astype(str)
    if use_rec:
        key = key + "_" + pd.Series(qbin(S["R"][m, c], 10)).astype(str)
    df = pd.DataFrame(dict(g=pd.factorize(key)[0], e=W[m] @ np.maximum(S["L"][:, c], 0), y=Y[m, t]))
    df = df[df.groupby("g").g.transform("size") >= 5]; gg = df.g.to_numpy(); sz = df.groupby("g").g.transform("size").to_numpy()
    a = df.groupby("g").e.rank().to_numpy() - (sz + 1) / 2; b = df.groupby("g").y.rank().to_numpy() - (sz + 1) / 2
    den = np.sqrt((a * a).sum() * (b * b).sum()); obs = (a * b).sum() / den if den > 0 else np.nan
    null = np.empty(nperm); s = np.argsort(gg, kind="stable")
    for i in range(nperm):
        o = np.lexsort((rng.random(len(gg)), gg)); ap = np.empty_like(a); ap[s] = a[o]; null[i] = (ap * b).sum() / den
    p = (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + nperm) if nperm else np.nan
    return obs, p, int(len(df))


rows = []
for _, r in top.iterrows():
    c = int(ch.index[(ch.name == r.channel) & ~ch.decoy][0]); t = T.index(r.best_readout)
    m = np.flatnonzero(S["R"][:, c] > 0); use_rec = len(m) >= 50
    if not use_rec:
        m = np.arange(n)
    obs, p, nrec = rho(c, t, m, use_rec)
    drho = []
    for d in ch.index[ch.decoy & (ch.of == r.channel)]:
        md = np.flatnonzero(S["R"][:, d] > 0) if use_rec else np.arange(n)
        drho.append(rho(int(d), t, md, use_rec, 0)[0] if len(md) >= 50 else np.nan)
    rows.append(dict(rank=r.channel_rank_by_best_event, channel=r.channel, readout=r.best_readout, best_event=r.best_rank, q_c=round(r.q, 4),
                     model_sign=float(np.sign(V[c, t])), level="receptor" if use_rec else "ligand", rho=obs, p=p, n_receivers=nrec,
                     decoy_rhos=";".join(f"{x:.4f}" for x in drho), decoy_mean=np.nanmean(drho),
                     decoys_as_large=int(sum((np.sign(x) == np.sign(obs)) and abs(x) >= abs(obs) for x in drho if np.isfinite(x)))))
d = pd.DataFrame(rows); ok = d.p.notna() & (d.n_receivers >= 50)
d["q"] = np.nan; d.loc[ok, "q"] = multipletests(d.p[ok], method="fdr_bh")[1]
d["supported_data"] = (d.q <= 0.1) & (np.sign(d.rho) == d.model_sign) & (d.decoys_as_large <= 1)
d.to_csv(R / OUT, index=False)
print(d.drop(columns="decoy_rhos").round(4).to_string(index=False))
