"""Signals planted according to the NicheNet prior (top-50 targets of six ligands)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from graticcs.paths import BREAST
D = BREAST; OUTD = D / "nichenet_spike"; OUTD.mkdir(exist_ok=True)
PLANT = ["CXCL12", "SPP1", "CXCL10", "IL1B", "CCL5", "VEGFA"]; BETAS = [0.0, 0.2, 0.4]; NTOP = 50
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); rawmean = np.asarray(X.mean(0)).ravel()
X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
cells = pd.read_parquet(D / "cells.parquet", columns=["negprobe_total"]); xneg = rawmean / (cells.negprobe_total.mean() / 50)
NN = pd.read_parquet(D / "nichenet_ligand_target.parquet"); lrn = pd.read_parquet(D / "nichenet_lr_network.parquet")
import liana as li
res = li.rs.select_resource("consensus").apply(lambda s: s.str.upper())
lr_genes = set(g for col in ["ligand", "receptor"] for x in res[col] for g in x.split("_")) | set(lrn["from"]) | set(lrn["to"])
smeta = json.load(open(D / "gt_sec_channels.json"))
cand = sorted({c["ligand"] for c in smeta["channels"] if not c["decoy"] and c["ligand"] in NN.columns})
log("candidate ligands (secreted channels in NicheNet):", len(cand))
targets = {}
for L in cand:
    w = NN[L]; w = w[[g for g in w.index if g in gi and g not in lr_genes and xneg[gi[g]] >= 1.0]].sort_values(ascending=False)[:NTOP]
    targets[L] = w
union = sorted({g for w in targets.values() for g in w.index}); ui = {g: k for k, g in enumerate(union)}
E0 = Lc[:, [gi[g] for g in union]].toarray().astype(np.float32); sdg = E0.std(0) + 1e-6
log("union target genes", len(union))
S = np.load(D / "gt_sec_inputs.npz"); n = E0.shape[0]
W = sp.csr_matrix((np.exp(-S["dist"] / 15.0), (S["dst"], S["src"])), shape=(n, n))
def zr(g):
    v = np.asarray(Lc[:, gi[g]].todense()).ravel(); return (v - v.mean()) / (v.std() + 1e-9)
expo = {}
for L in PLANT:
    recs = [r for r in lrn.loc[lrn["from"] == L, "to"] if r in gi]
    Rv = np.max(np.column_stack([np.maximum(zr(r), 0) for r in recs]), axis=1) if recs else np.zeros(n)
    x = Rv * (W @ np.maximum(zr(L), 0)); expo[L] = (x - x.mean()) / (x.std() + 1e-12)
    log(f"{L}: receptors in panel {recs[:6]}, exposed cells {(x > 0).mean():.3f}")
det = np.asarray((Lc > 0).mean(0)).ravel()
candg = np.array([i for i, g in enumerate(genes) if g not in ui and g not in lr_genes and det[i] > 0.02])
sub = Lc[:, candg]; var = np.asarray(sub.power(2).mean(0)).ravel() - np.asarray(sub.mean(0)).ravel() ** 2
M = Lc[:, candg[np.argsort(-var)[:2000]]].toarray(); M -= M.mean(0); M /= M.std(0) + 1e-6
pcs = TruncatedSVD(50, random_state=0).fit_transform(M).astype(np.float32); del M; pcs /= pcs.std(0) + 1e-6
np.save(OUTD / "alt_rna_pc.npy", pcs)
for b in BETAS:
    E = E0.copy()
    for L in PLANT:
        w = targets[L]; cols = [ui[g] for g in w.index]
        E[:, cols] += (b * sdg[cols] * (w.to_numpy() / w.max()))[None, :] * expo[L][:, None]
    Ez = (E - E.mean(0)) / (E.std(0) + 1e-6)
    Y = np.column_stack([Ez[:, [ui[g] for g in targets[L].index]] @ targets[L].to_numpy() / targets[L].sum() for L in cand]).astype(np.float32)
    np.savez_compressed(OUTD / f"Y_beta{b}.npz", Y=Y); np.save(OUTD / f"E_beta{b}.npy", E.astype(np.float16))
    log("saved beta", b)
json.dump({"plant": PLANT, "betas": BETAS, "readouts": [f"NN_{L}" for L in cand], "candidates": cand, "union_genes": union,
           "targets": {L: list(targets[L].index) for L in cand}}, open(OUTD / "meta.json", "w"), indent=1)
log("done")
