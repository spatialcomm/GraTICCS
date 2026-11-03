"""Rank all held-out candidate events by (1 - q_c) |contribution| for channels with literature support.

python -m analysis.case_events <contact|secreted> <run>[,<run>...] <out.csv>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from statsmodels.stats.multitest import multipletests
from graticcs.paths import BREAST, RESULTS
R = RESULTS; D = BREAST
cls, runs, outp = sys.argv[1], [R / x for x in sys.argv[2].split(",")], sys.argv[3]
z = np.load(D / "gt_inputs.npz"); ctype, fold = z["ctype"], z["fold"]
if cls == "contact":
    meta = json.load(open(D / "gt_channels.json")); src, dst, con, dist = z["src"], z["dst"], z["contact"], z["dist"]; L, Rr = z["L_mo"], z["R_mo"]
else:
    S = np.load(D / "gt_sec_inputs.npz"); meta = json.load(open(D / "gt_sec_channels.json")); src, dst, con, dist = S["src"], S["dst"], S["contact"], S["dist"]; L, Rr = S["L"], S["R"]
ch = pd.DataFrame(meta["channels"])
if "decoy_modality" not in ch: ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"), np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
lit = json.load(open(D / "lit_channel_matches.json"))[cls]; TYPES = json.load(open(D / "gt_channels.json"))["types"]
fo = [json.load(open(p)) for d in runs for p in sorted(d.glob("fold*_mo.json"))]
T = fo[0]["targets"]; s0 = np.sum([f["sse_nuisance"] for f in fo], 0); imp = np.sum([f["dsse_knockout"] for f in fo], 0) / s0
ib, bt = imp.max(1), imp.argmax(1); p = np.ones(len(ch)); q = np.ones(len(ch))
for st in ["protein", "rna"]:
    dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
    pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); p[rl] = pv; q[rl] = multipletests(pv, method="fdr_bh")[1]
real = np.flatnonzero(~ch.decoy.to_numpy())
sp_ = lambda a: np.log1p(np.exp(np.clip(a.astype(np.float32), -10, 10)))
Cfull = np.zeros((len(src), len(real)), np.float32); nseen = np.zeros(len(src), np.float32)
for d in runs:
    for k in range(5):
        A = np.load(d / f"fold{k}_mo_attention.npz"); e, a = A["edge"], A["a"].astype(np.float32)
        st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
        P = torch.softmax(st["P"], 1).numpy()[real]; G = st["G"].numpy()[:, real]; V = st["V"].numpy()[real, bt[real]]
        for b in range(0, len(e), 200_000):
            eb, ab = e[b:b + 200_000], a[b:b + 200_000]
            Cfull[eb] += (ab @ P.T) * sp_(L[src[eb]][:, real]) * sp_(Rr[dst[eb]][:, real]) / (1 + np.exp(-G[ctype[dst[eb]]])) * V[None, :]
        nseen[e] += 1
edges = np.flatnonzero(nseen > 0); Cm = Cfull[edges] / nseen[edges][:, None]; del Cfull
score = np.abs(Cm) * (1 - q[real])[None, :]
order = np.argsort(-score, axis=None); N = score.size
rank_all = np.empty(N, np.int64); rank_all[order] = np.arange(1, N + 1)
rank_all = rank_all.reshape(score.shape)
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: i for i, g in enumerate(genes)}
import scipy.sparse as sp
X = sp.load_npz(D / "rna_counts.npz").tocsr()
cells = pd.read_parquet(D / "cells.parquet", columns=["cell_id", "fov"])
rows = []
for j, c in enumerate(real):
    nm = ch.name[c]
    if nm not in lit: continue
    lg = [x for x in ch.ligand[c].split("_") if x in gi]; rg = [x for x in ch.receptor[c].split("_") if x in gi]
    top = np.argsort(-score[:, j])[:200]
    for t in top:
        e = edges[t]; sj, ri = src[e], dst[e]
        lc = int(sum(X[sj, gi[x]] for x in lg)) if lg else -1; rc = int(sum(X[ri, gi[x]] for x in rg)) if rg else -1
        rows.append(dict(channel=nm, n_papers=len(lit[nm]["papers"]), stratum=ch.stratum[c], p=p[c], q=q[c], best_readout=T[bt[c]],
                         rank_all=int(rank_all[t, j]), n_events=N, rank_in_channel=int((score[:, j] > score[t, j]).sum() + 1),
                         contrib=float(Cm[t, j]), score=float(score[t, j]), edge=int(e), sender=cells.cell_id[sj], receiver=cells.cell_id[ri],
                         fov=int(cells.fov[ri]), sender_type=TYPES[ctype[sj]], receiver_type=TYPES[ctype[ri]],
                         ligand_rna=lc, receptor_rna=rc, L_input=float(L[sj, c]), R_input=float(Rr[ri, c]), contact=float(con[e]), dist_um=float(dist[e])))
out = pd.DataFrame(rows); out.to_csv(outp, index=False)
summ = out.sort_values("rank_all").groupby("channel").first().sort_values("rank_all")
print(f"events ranked: {N:,} ({len(edges):,} held-out edges x {len(real)} real channels)")
print(summ[["n_papers", "q", "best_readout", "rank_all", "sender_type", "receiver_type", "ligand_rna", "receptor_rna", "dist_um"]].head(25).to_string())
