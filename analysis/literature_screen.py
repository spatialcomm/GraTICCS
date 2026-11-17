"""Screen every real original-tissue channel with one fixed rule for literature-reported receiver responses.

Gene sets: (lit) receiver genes reported as up-regulated in the curated literature records; (nichenet) the ligand's top 30
panel targets. Receivers: the 50 distinct receivers of the best-scored held-out events on the channel's strongest readout.
Controls: cells of the same imaging field, cell type and decile of the channel's receptor input.
Pass: BH q <= 0.1 on the one-sided Wilcoxon test of per-receiver differences (within data set and gene-set type), median
> 0, at least 55% of receivers above their controls, mean > 0 without the three largest, at least 30 matched receivers,
and at most one of the same-receptor decoys with an effect as large. Writes results/screen_all_channels.csv.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import false_discovery_control, wilcoxon

from graticcs.paths import BREAST, RAW, RESULTS, XENIUM

B, XE, R = BREAST, XENIUM, RESULTS
REC = RAW / "literature_records"
ALIAS = {"IL8": "CXCL8", "WARS": "WARS1", "TNF-ALPHA": "TNF", "TNFA": "TNF", "IFN-GAMMA": "IFNG", "TGF-B1": "TGFB1", "TGFB": "TGFB1",
         "PD-L1": "CD274", "PD-1": "PDCD1", "PDL1": "CD274", "PD1": "PDCD1", "GALECTIN-9": "LGALS9", "TIM-3": "HAVCR2"}
sym = lambda g: ALIAS.get(str(g).strip().upper(), str(g).strip().upper())
split = lambda s: [sym(x) for x in re.split(r"[;/,+]| and ", str(s or "")) if x.strip()]
recs = [json.load(open(REC / fn, encoding="utf-8-sig")) for fn in os.listdir(REC)]
nn = pd.read_parquet(B / "nichenet_ligand_target.parquet"); nn.index = nn.index.astype(str).str.upper(); nn.columns = nn.columns.astype(str).str.upper()
sp_ = lambda a: np.log1p(np.exp(np.clip(a.astype(np.float32), -10, 10)))


def gene_sets(ligand, receptor, gi):
    lg, rg = ligand.split("_"), receptor.split("_"); out = {}; up = set()
    for r in recs:
        e = r.get("main_ccc_edge") or {}
        if any(x in split(e.get("ligand")) for x in lg) and any(p in split(e.get("receptor")) for p in rg):
            for f in r.get("downstream_effectors") or []:
                if isinstance(f, dict) and str(f.get("measured_in", "")).startswith("receiver") and str(f.get("direction", "")).lower().startswith("up"):
                    g = sym(f.get("gene", ""))
                    if g in gi and g not in lg + rg:
                        up.add(g)
    if len(up) >= 3:
        out["lit"] = sorted(up)
    lig = [x for x in lg if x in nn.columns]
    if lig:
        w = nn[lig].mean(1); w = w[[g for g in w.index if g in gi and g not in lg + rg]]; w = w[w > 0].sort_values(ascending=False).head(30)
        if len(w) >= 10:
            out["nichenet"] = list(w.index)
    return out


def run_set(tag, run, src, dst, L, Rr, ctype, fov, X, genes, meta):
    gi = {g: i for i, g in enumerate(genes)}; lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); ch = pd.DataFrame(meta["channels"])
    types = meta["types"]; T = list(meta["prot_readout"]) + list(meta.get("rna_programs", []))
    dirs = [R / f"{run}_s{s}" for s in range(3)]; fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]
    bt = (np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)).argmax(1)
    fits = []
    for d in dirs:
        for k in range(5):
            A = np.load(d / f"fold{k}_mo_attention.npz"); e = A["edge"]; st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
            fits.append(dict(e=e, a=A["a"].astype(np.float32), s=src[e], d=dst[e], t=ctype[dst[e]], P=torch.softmax(st["P"], 1).numpy(),
                             G=st["G"].numpy(), V=st["V"].numpy()))
    cache = {}

    def z_gene(g):
        if g not in cache:
            x = np.log1p(np.asarray(X[:, gi[g]].todense()).ravel() / lib * 1e3); o = np.zeros(len(x), np.float32)
            for t in np.unique(ctype):
                m = ctype == t; o[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-9)
            cache[g] = o
        return cache[g]

    def receivers(c, b):
        C = np.zeros(len(src), np.float32); n = np.zeros(len(src), np.float32)
        for f in fits:
            C[f["e"]] += (f["a"] @ f["P"][c]) * sp_(L[f["s"], c]) * sp_(Rr[f["d"], c]) / (1 + np.exp(-f["G"][f["t"], c])) * f["V"][c, b]
            n[f["e"]] += 1
        s = np.abs(C) / np.maximum(n, 1); best = np.argpartition(-s, 200)[:200]; best = best[np.argsort(-s[best])]
        return pd.unique(dst[best])[:50]

    def diffs(c, top, y):
        rec = Rr[:, c].astype(np.float32); dec = np.zeros(len(rec), np.int64)
        for t in np.unique(ctype):
            m = ctype == t; dec[m] = np.searchsorted(np.quantile(rec[m], np.linspace(0.1, 0.9, 9)), rec[m])
        key = fov.astype(np.int64) * 100000 + ctype.astype(np.int64) * 100 + dec
        m = np.isin(key, np.unique(key[top])); is_top = np.zeros(len(y), bool); is_top[top] = True; m &= ~is_top
        g = pd.Series(y[m]).groupby(key[m]).mean(); k = key[top]; ok = pd.Series(k).isin(g.index).values
        return y[top][ok] - g.reindex(k[ok]).values

    rows = []
    for c in ch.index[~ch.decoy]:
        sets = gene_sets(ch.ligand[c], ch.receptor[c], gi)
        if not sets:
            continue
        b = int(bt[c]); top = receivers(c, b)
        for kind, gs in sets.items():
            y = np.mean([z_gene(g) for g in gs], 0); d = diffs(c, top, y)
            if len(d) < 10:
                continue
            rows.append(dict(data=tag, kind=kind, channel=ch.name[c], c=c, readout=T[b] if b < len(T) else str(b), n_genes=len(gs),
                             genes=",".join(gs), n_used=len(d), mean=d.mean(), median=np.median(d), frac_pos=(d > 0).mean(),
                             drop3=np.sort(d)[:-3].mean(), wil_p=wilcoxon(d, alternative="greater").pvalue,
                             types=str(pd.Series(np.array(types)[ctype[top]]).value_counts().head(4).to_dict())))
    df = pd.DataFrame(rows)
    for kind, g in df.groupby("kind"):
        df.loc[g.index, "wil_q"] = false_discovery_control(g.wil_p.values)
    df["decoys_ge"] = np.nan; df["decoys_n"] = np.nan; df["decoys"] = ""
    for i in df.index[(df.wil_p <= 0.05) & (df["mean"] > 0)]:
        r = df.loc[i]; c = int(r.c); y = np.mean([z_gene(g) for g in r.genes.split(",")], 0); dd = []
        for d_ in ch.index[ch.decoy & (ch.of == ch.name[c])]:
            x = diffs(d_, receivers(d_, int(bt[c])), y)
            if len(x) >= 10:
                dd.append(x.mean())
        df.loc[i, ["decoys_ge", "decoys_n"]] = [sum(v >= r["mean"] for v in dd), len(dd)]; df.loc[i, "decoys"] = " ".join(f"{v:+.2f}" for v in dd)
    df["PASS"] = (df.wil_q <= 0.1) & (df["median"] > 0) & (df.frac_pos >= 0.55) & (df.drop3 > 0) & (df.n_used >= 30) & (df.decoys_ge <= 1)
    print(tag, "tested", df.groupby("kind").size().to_dict(), "pass", int(df.PASS.sum()), flush=True)
    return df


z = np.load(B / "gt_inputs.npz"); genes = [g.upper() for g in json.load(open(B / "genes.json"))["genes"]]; X = sp.load_npz(B / "rna_counts.npz").tocsr()
S = np.load(B / "gt_sec_inputs.npz"); types = json.load(open(B / "gt_channels.json"))["types"]
out = [run_set("breast_secreted", "G2_sec_real", S["src"], S["dst"], S["L"], S["R"], z["ctype"], z["fov"], X, genes,
               {**json.load(open(B / "gt_sec_channels.json")), "types": types}),
       run_set("breast_contact", "G_real", z["src"], z["dst"], z["L_mo"], z["R_mo"], z["ctype"], z["fov"], X, genes, json.load(open(B / "gt_channels.json")))]
import scanpy as sc
a = sc.read_10x_h5(XE / "cell_feature_matrix.h5", gex_only=True); a.var_names = a.var_names.astype(str); zx = np.load(XE / "gt_inputs.npz")
out.append(run_set("xenium_contact", "G_xen_real", zx["src"], zx["dst"], zx["L_mo"], zx["R_mo"], zx["ctype"], zx["fov"], a.X.tocsr(),
                   [g.upper() for g in a.var_names], json.load(open(XE / "gt_channels.json"))))
pd.concat(out).to_csv(R / "screen_all_channels.csv", index=False)
