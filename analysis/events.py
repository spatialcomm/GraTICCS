"""Event scores in the original tissue, shared by the channel ranking, the case selection and the case figures.

For every real channel: decoy-calibrated q (modality-stratified), its strongest readout t*, and for every held-out
sender -> receiver event the contribution to t*, averaged over seeds:
    contrib = a_mix * softplus(L_j) softplus(R_i) * sigmoid(G[type_i, c]) * V[c, t*]
The event score is (1 - q_c) |contrib|; events are ranked over all real channels.
"""
import json

import numpy as np
import pandas as pd
import torch
from statsmodels.stats.multitest import multipletests

from graticcs.paths import BREAST

D = BREAST


def channels(kind):
    meta = json.load(open(D / ("gt_channels.json" if kind == "contact" else "gt_sec_channels.json")))
    ch = pd.DataFrame(meta["channels"])
    if "decoy_modality" not in ch:
        ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
    prot_real = set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])
    ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"), np.where(ch.name.isin(prot_real), "protein", "rna"))
    return ch


def inputs(kind):
    """src, dst, contact, dist, L, R of the candidate events of one modality."""
    if kind == "contact":
        z = np.load(D / "gt_inputs.npz")
        return z["src"], z["dst"], z["contact"], z["dist"], z["L_mo"], z["R_mo"]
    S = np.load(D / "gt_sec_inputs.npz")
    return S["src"], S["dst"], S["contact"], S["dist"], S["L"], S["R"]


def calibrate(ch, runs):
    fo = [json.load(open(p)) for d in runs for p in sorted(d.glob("fold*_mo.json"))]
    T = fo[0]["targets"]; imp = np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)
    ib, bt = imp.max(1), imp.argmax(1); p = np.ones(len(ch)); q = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); p[rl] = pv; q[rl] = multipletests(pv, method="fdr_bh")[1]
    return T, bt, p, q


def scores(kind, runs):
    """Returns a dict with channel table, calibration, held-out edges, mean contributions, scores and global ranks."""
    ch = channels(kind); T, bt, p, q = calibrate(ch, runs)
    src, dst, con, dist, L, Rr = inputs(kind)
    ctype = np.load(D / "gt_inputs.npz")["ctype"]
    real = np.flatnonzero(~ch.decoy.to_numpy())
    sp_ = lambda a: np.log1p(np.exp(np.clip(a.astype(np.float32), -10, 10)))
    C = np.zeros((len(src), len(real)), np.float32); nseen = np.zeros(len(src), np.float32)
    for d in runs:
        for k in range(5):
            A = np.load(d / f"fold{k}_mo_attention.npz"); e, a = A["edge"], A["a"].astype(np.float32)
            st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
            P = torch.softmax(st["P"], 1).numpy()[real]; G = st["G"].numpy()[:, real]; V = st["V"].numpy()[real, bt[real]]
            for b in range(0, len(e), 200_000):
                eb, ab = e[b:b + 200_000], a[b:b + 200_000]
                C[eb] += (ab @ P.T) * sp_(L[src[eb]][:, real]) * sp_(Rr[dst[eb]][:, real]) / (1 + np.exp(-G[ctype[dst[eb]]])) * V[None, :]
            nseen[e] += 1
    edges = np.flatnonzero(nseen > 0); Cm = C[edges] / nseen[edges][:, None]; del C
    score = np.abs(Cm) * (1 - q[real])[None, :]
    order = np.argsort(-score, axis=None)
    rank = np.empty(score.size, np.int64); rank[order] = np.arange(1, score.size + 1)
    return dict(ch=ch, T=T, bt=bt, p=p, q=q, real=real, edges=edges, contrib=Cm, score=score, rank=rank.reshape(score.shape),
                src=src, dst=dst, contact=con, dist=dist, L=L, R=Rr, ctype=ctype)
