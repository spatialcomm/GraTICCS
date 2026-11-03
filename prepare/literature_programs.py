"""Ligand-specific readouts from prior knowledge: literature-reported downstream genes and NicheNet top-50 targets."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from graticcs.paths import BREAST, RAW
D = BREAST; OUTD = D / os.environ.get("PRIOR_OUT", "prior_programs"); OUTD.mkdir(exist_ok=True)
kb = json.load(open(os.environ.get("KB", RAW / "literature_kb.json")))
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
smeta = json.load(open(D / "gt_sec_channels.json")); cmeta = json.load(open(D / "gt_channels.json")); ch = smeta["channels"]
chan_genes = {g for m in (cmeta, smeta) for c in m["channels"] for s in ("ligand", "receptor") for g in c[s].split("_")}
NN = pd.read_parquet(D / "nichenet_ligand_target.parquet")
real_ligs = sorted({c["ligand"] for c in ch if not c["decoy"]})
lit = {}
for L in real_ligs:
    if L not in kb: continue
    up = {g.upper(): v for g, v in kb[L]["up"].items() if g.upper() in gi and g.upper() not in chan_genes}
    if len(up) >= 5: lit[L] = pd.Series(up, dtype=float)
ligs = sorted(lit); print("ligands with literature programs:", ligs)
nn = {}
for L in ligs:
    w = NN[L] if L in NN.columns else None
    if w is None: continue
    w = w[[g for g in w.index if g in gi and g not in chan_genes]].sort_values(ascending=False)[:50]; nn[L] = w
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
def program(w):
    M = Lc[:, [gi[g] for g in w.index]].toarray(); M = (M - M.mean(0)) / (M.std(0) + 1e-6)
    return (M @ w.to_numpy()) / w.sum()
of_lig = {c["name"]: c["ligand"] for c in ch if not c["decoy"]}
for tag, progs in [("lit", lit), ("nn", nn)]:
    names = [L for L in ligs if L in progs]
    Y = np.column_stack([program(progs[L]) for L in names]).astype(np.float32); Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    Mk = np.zeros((len(ch), len(names)), np.float32)
    for k, c in enumerate(ch):
        L = of_lig.get(c["of"], c["ligand"])
        if L in names: Mk[k, names.index(L)] = 1
    np.savez_compressed(OUTD / f"Y_{tag}.npz", Y=Y); np.save(OUTD / f"mask_{tag}.npy", Mk)
    json.dump({"readouts": [f"{tag.upper()}_{L}" for L in names], "ligands": names,
               "genes": {L: list(progs[L].index) for L in names}}, open(OUTD / f"meta_{tag}.json", "w"), indent=1)
    print(tag, "programs", len(names), "channels with a program", int(Mk.any(1).sum()))
lit_g = {L: set(lit[L].index) for L in ligs}; nn_g = {L: set(nn[L].index) for L in nn}
print("literature vs NicheNet gene overlap per ligand:", {L: len(lit_g[L] & nn_g.get(L, set())) for L in ligs})
