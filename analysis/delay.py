"""Simulated signaling delay, contact channels (three seeds; rho = 1 uses the standard planting).

Channel recovery follows analysis/recovery.py (main and strict rule). Cell-pair AP is computed on 100,000 contact edges
sampled with a fixed seed; positives are planted-channel events whose sender has positive past ligand state and whose
receiver has positive receptor, regenerated for each rho. Coexpression scores use the measured (current) L and R.
Writes results/lag_eval_crand.csv; secreted delay recovery is in analysis/recovery.py.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

from graticcs.data import target_mask
from graticcs.paths import BREAST, RESULTS

D = BREAST; R = RESULTS
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); ch = pd.DataFrame(meta["channels"])
sm = json.load(open(D / "spike_crand_meta.json")); T = sm["tnames"]
M = target_mask({"L": z["L_mo"], "tnames": T, "channels": meta["channels"]})
real_prot = set(ch.loc[ch.decoy & (ch.get("decoy_modality") == "protein"), "of"])
ch["stratum"] = np.where(ch.decoy, np.where(ch.get("decoy_modality") == "protein", "protein", "rna"),
                         np.where(ch.name.isin(real_prot), "protein", "rna"))
real = np.flatnonzero(~ch.decoy.to_numpy())
NATURAL = {"HLA-A->CD8A", "HLA-DRA->CD4"}
plant = [p for p in sm["plant"] if p[0] not in NATURAL]; planted = {p[0] for p in plant}
src, dst, con = z["src"], z["dst"], z["contact"]
ctype = z["ctype"]; L, Rm = z["L_mo"], z["R_mo"]
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
ci = np.flatnonzero(con > 0); samp = np.sort(np.random.default_rng(0).choice(len(ci), 100000, replace=False))
csrc, cdst = src[ci][samp], dst[ci][samp]
coexpr = np.maximum(L[csrc][:, real], 0) * np.maximum(Rm[cdst][:, real], 0)
keep = ~np.isin(real, [ch.index[ch.name == n][0] for n in NATURAL])
gid = ci[samp]; pos = -np.ones(len(src), np.int64); pos[gid] = np.arange(len(gid))
past = np.load(D / "spike_crand_lag_past_ligands.npz")


def importance(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    return ds / s0


def qvals(imp):
    ib = imp.max(1); q = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    return q


def pair_scores(dirs, q):
    S = np.zeros((len(samp), len(real)), np.float32)
    for d in dirs:
        for k in range(5):
            st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]; G, V = st["G"].numpy(), st["V"].numpy()
            P = torch.softmax(st["P"], 1).numpy()[real]; Vn = np.linalg.norm((V * M)[real], axis=1)
            A = np.load(d / f"fold{k}_mo_attention.npz"); p_ = pos[A["edge"]]; m = p_ >= 0
            e, a = p_[m], A["a"][m].astype(np.float32) @ P.T; se = gid[e]
            s = sp_(L[src[se]][:, real]) * sp_(Rm[dst[se]][:, real]); g = 1 / (1 + np.exp(-G[ctype[dst[se]]][:, real]))
            S[e] += a * s * g * Vn[None, :] / len(dirs)
    return (1 - q[real])[None, :] * S


def recovery(imp):
    rec = strict = 0
    for chn, tgt, sgn in plant:
        c = ch.index[ch.name == chn][0]; t = T.index(tgt); st = ch.stratum[c]
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        others = rl[~ch.name[rl].isin(planted)]
        pp = (1 + (imp[dec, t] >= imp[c, t]).sum()) / (1 + len(dec))
        rec += int(pp <= 0.05 and int((imp[others, t] > imp[c, t]).sum()) + 1 <= 3)
        strict += int(pp <= 0.05 and int((imp[rl, t] > imp[c, t]).sum()) + 1 <= 3)
    return rec, strict


rows = []
for rho in [1.0, 0.7, 0.5, 0.3]:
    for b in [0.2, 0.4]:
        dirs = [R / (f"G_crand_b{b}_s{s}" if rho == 1.0 else f"G_crand_lag_r{rho}_b{b}_s{s}") for s in range(3)]
        if any(len(list(d.glob("fold*_mo.json"))) < 5 for d in dirs):
            print("missing", rho, b); continue
        imp = importance(dirs); q = qvals(imp); rec, strict = recovery(imp)
        seeds = [recovery(importance([d]))[0] for d in dirs]
        truth = np.zeros(coexpr.shape, bool)
        for chn, _, _ in plant:
            k = int(np.flatnonzero(ch.name[real].to_numpy() == chn)[0]); c = real[k]
            Lp = L[:, c] if rho == 1.0 else past[f"r{rho}__{chn.replace('->', '__')}"]
            truth[:, k] = (np.maximum(Lp[csrc], 0) * np.maximum(Rm[cdst, c], 0)) > 0
        y = truth[:, keep].ravel(); S = pair_scores(dirs, q)
        rows.append(dict(rho=rho, beta=b, n=len(plant), recovered=rec, recovered_strict=strict, seeds=seeds,
                         ours_AP=average_precision_score(y, S[:, keep].ravel()),
                         coexpr_now_AP=average_precision_score(y, coexpr[:, keep].ravel()), random_AP=float(y.mean())))
out = pd.DataFrame(rows); out.to_csv(R / "lag_eval_crand.csv", index=False)
print(out.to_string(index=False))
