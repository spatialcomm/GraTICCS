"""NicheNet ligand activities (standard workflow in Python) on the NicheNet-consistent planted data, with cell-type
niches and with ligand-specific niches."""
import json
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy import stats
from graticcs.paths import BREAST, RESULTS
D = BREAST; ND = D / "nichenet_spike"; R = RESULTS
meta = json.load(open(ND / "meta.json")); union = meta["union_genes"]; ui = {g: k for k, g in enumerate(union)}; cand = meta["candidates"]
NN = pd.read_parquet(D / "nichenet_ligand_target.parquet"); lrn = pd.read_parquet(D / "nichenet_lr_network.parquet")
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); rawmean = np.asarray(X.mean(0)).ravel()
X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
xneg = rawmean / (pd.read_parquet(D / "cells.parquet", columns=["negprobe_total"]).negprobe_total.mean() / 50)
z = np.load(D / "gt_inputs.npz"); ctype = z["ctype"]; types = json.load(open(D / "gt_channels.json"))["types"]
S = np.load(D / "gt_sec_inputs.npz"); src, dst = S["src"], S["dst"]
bg = [g for g in NN.index if g in gi and xneg[gi[g]] >= 1.0]; bgi = np.array([gi[g] for g in bg])
Lbg = Lc[:, bgi].tocsr()
print("background genes", len(bg), "union (planted-target) genes", len(union))
detf = lambda g, m: float((Lc[m][:, gi[g]] > 0).mean()) if g in gi else 0.0


def de(E_union, niche, other):
    """logFC and Welch p for background genes; union genes taken from the (possibly spiked) E_union."""
    A = Lbg[niche]; B = Lbg[other]
    ma, mb = np.asarray(A.mean(0)).ravel(), np.asarray(B.mean(0)).ravel()
    va = np.asarray(A.power(2).mean(0)).ravel() - ma ** 2; vb = np.asarray(B.power(2).mean(0)).ravel() - mb ** 2
    for k, g in enumerate(bg):
        if g in ui:
            col = E_union[:, ui[g]].astype(np.float32); ma[k], mb[k] = col[niche].mean(), col[other].mean()
            va[k], vb[k] = col[niche].var(), col[other].var()
    t = (ma - mb) / np.sqrt(va / niche.sum() + vb / other.sum() + 1e-12)
    dfree = niche.sum() + other.sum() - 2; p = 2 * stats.t.sf(np.abs(t), dfree)
    return ma - mb, p


rows = []
for b in meta["betas"]:
    E = np.load(ND / f"E_beta{b}.npy")
    for si, sname in enumerate(types):
        sender = ctype == si; rec = ~sender
        near = np.zeros(len(ctype), bool); near[dst[sender[src]]] = True
        niche = rec & near; other = rec & ~near
        if niche.sum() < 200 or other.sum() < 200: continue
        lfc, p = de(E, niche, other)
        ok = (lfc > 0.1) & (p < 0.01); order = np.argsort(-lfc); gs = set(np.array(bg)[order[ok[order]]][:250])
        ind = np.array([g in gs for g in bg], float)
        cands = [L for L in cand if detf(L, sender) >= 0.05 and any(detf(r, rec) >= 0.05 for r in lrn.loc[lrn["from"] == L, "to"])]
        act = {L: float(np.corrcoef(NN.loc[bg, L].to_numpy(), ind)[0, 1]) if ind.sum() > 0 else np.nan for L in cands}
        ranked = sorted(act, key=lambda L: -np.nan_to_num(act[L], nan=-9))
        for L in meta["plant"]:
            rows.append(dict(beta=b, sender=sname, ligand=L, candidate=L in act, n_candidates=len(cands), geneset=len(gs),
                             activity=act.get(L, np.nan), rank=ranked.index(L) + 1 if L in act else np.nan))
    print("beta", b, "done", flush=True)
df = pd.DataFrame(rows)
best_sender = {L: types[int(np.argmax([float(np.asarray(Lc[ctype == k][:, gi[L]].mean())) for k in range(len(types))]))] for L in meta["plant"]}
summ = df[df.apply(lambda r: r.sender == best_sender[r.ligand], axis=1)].copy(); summ["recovered"] = summ["rank"] <= 3
print("main sender per ligand:", best_sender)
print(summ[["beta", "ligand", "sender", "candidate", "n_candidates", "geneset", "activity", "rank", "recovered"]].to_string(index=False))
print(summ.groupby("beta").recovered.sum())
df.to_csv(R / "nichenet_all.csv", index=False); summ.to_csv(R / "nichenet_summary.csv", index=False)


rows = []
for b in meta["betas"]:
    E = np.load(ND / f"E_beta{b}.npy")
    for L in meta["plant"]:
        v = np.asarray(Lc[:, gi[L]].todense()).ravel(); high = v >= np.quantile(v[v > 0], 0.9) if (v > 0).any() else v > 0
        near = np.zeros(len(v), bool); near[dst[high[src]]] = True
        niche = near & ~high; other = ~near & ~high
        lfc, p = de(E, niche, other)
        ok = (lfc > 0.1) & (p < 0.01); order = np.argsort(-lfc); gs = set(np.array(bg)[order[ok[order]]][:250])
        ind = np.array([g in gs for g in bg], float)
        act = {c: float(np.corrcoef(NN.loc[bg, c].to_numpy(), ind)[0, 1]) for c in cand} if ind.sum() else {}
        ranked = sorted(act, key=lambda c: -np.nan_to_num(act[c], nan=-9))
        r = ranked.index(L) + 1 if L in act else np.nan
        rows.append(dict(beta=b, ligand=L, niche_cells=int(niche.sum()), geneset=len(gs), rank=r, n=len(cand), recovered=bool(r <= 3)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); print(df.groupby("beta").recovered.sum()); df.to_csv(R / "nichenet_ligand_niche.csv", index=False)
