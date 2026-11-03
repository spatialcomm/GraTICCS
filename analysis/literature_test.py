"""Literature-reported response genes in the 50 top-ranked distinct receivers versus receptor-positive
controls of the same cell types (one-sided Mann-Whitney on ranks; BH across the tested channels).

python -m analysis.literature_test <cases.csv> <contact|secreted> <out.csv>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from graticcs.paths import BREAST
D = BREAST
cases, cls, outp = pd.read_csv(sys.argv[1]), sys.argv[2], sys.argv[3]
lit = json.load(open(D / "lit_channel_matches.json"))[cls]
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: i for i, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr(); lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
ctype = np.load(D / "gt_inputs.npz")["ctype"]
cells = pd.read_parquet(D / "cells.parquet", columns=["cell_id"]); idx = pd.Series(np.arange(len(cells)), index=cells.cell_id)
rows = []
for chn, cc in cases.groupby("channel"):
    lg, rg = chn.split("->")[0].split("_"), chn.split("->")[1].split("_")
    up = sorted({g for g, ev in lit[chn]["effectors"].items() if g in gi and g not in lg + rg and any(d.startswith("up") for d, _ in ev)})
    if len(up) < 2: continue
    top = idx[cc.drop_duplicates("receiver").sort_values("rank_all").head(50).receiver].to_numpy()
    lu = np.log1p(X[:, [gi[g] for g in up]].toarray() / lib[:, None] * 1e3); zt = np.zeros_like(lu)
    for t in np.unique(ctype):
        m = ctype == t; zt[m] = (lu[m] - lu[m].mean(0)) / (lu[m].std(0) + 1e-9)
    sc = zt.mean(1)
    recx = np.asarray(X[:, [gi[g] for g in rg if g in gi]].sum(1)).ravel() if any(g in gi for g in rg) else np.ones(len(sc))
    ctrl = np.flatnonzero(np.isin(ctype, ctype[top]) & (recx > 0) & ~np.isin(np.arange(len(sc)), top))
    mw = mannwhitneyu(sc[top], sc[ctrl], alternative="greater")
    rows.append(dict(channel=chn, n_papers=len(lit[chn]["papers"]), q_channel=cc.q.iloc[0], best_rank=int(cc.rank_all.min()),
                     n_response_genes=len(up), response_genes=",".join(up), top50_mean=sc[top].mean(), ctrl_mean=sc[ctrl].mean(),
                     effect=sc[top].mean() - sc[ctrl].mean(), mw_p=mw.pvalue))
out = pd.DataFrame(rows); out["mw_q"] = multipletests(out.mw_p, method="fdr_bh")[1]
out = out.sort_values("mw_p"); out.to_csv(outp, index=False)
print(out[["channel", "n_papers", "q_channel", "best_rank", "n_response_genes", "effect", "mw_p", "mw_q"]].to_string(index=False))
print(f"channels tested {len(out)}; p<=0.05: {(out.mw_p <= 0.05).sum()} (expected by chance {0.05 * len(out):.1f}); q<=0.1: {(out.mw_q <= 0.1).sum()}")
