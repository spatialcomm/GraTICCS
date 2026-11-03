"""Contact channels, decoys, readouts and background inputs for the CosMx breast section -> gt_inputs.npz, gt_channels.json."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from graticcs.lr_map import PROT2GENE, GENE2PROT, CONTACT_LIGANDS
from graticcs.paths import BREAST, RAW

D = BREAST
CELL_TYPES = RAW / "cells_typed.parquet"
UM = 0.5119 / 4256 * 1000
PROT_READOUT = ["GZMB", "GZMA", "Ki-67", "ICOS", "4-1BB", "iNOS", "IL-1b", "NF-kB p65", "HLA-DR", "IDO1", "CD38", "ICAM1", "PD-L1"]
RNA_PROGRAMS = {
    "IFN_GAS": ["STAT1", "IRF1", "GBP1", "GBP4", "GBP5", "CXCL9", "CXCL10", "CXCL11", "IDO1", "PSMB9", "TAP1", "WARS1", "SOCS1", "CIITA"],
    "NFKB": ["NFKBIA", "TNFAIP3", "ICAM1", "BIRC3", "RELB", "NFKB1", "NFKB2", "TNFAIP2", "CXCL8", "SOD2", "CCL20"],
    "PROLIF": ["MKI67", "TOP2A", "CENPF", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "CDK1", "CCNB1", "MCM2"],
    "CYTOTOX": ["GZMB", "GZMA", "GZMK", "PRF1", "NKG7", "GNLY", "CST7", "CTSW"],
    "TACT": ["ICOS", "TNFRSF9", "CD69", "IL2RA", "TNFRSF4", "TNFRSF18", "CRTAM", "NR4A1", "NR4A2", "EGR2"],
    "EXHAUST": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4", "ENTPD1", "CXCL13"],
}
N_DECOY = 5
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

c = pd.read_parquet(D / "cells.parquet")
typed = pd.read_parquet(CELL_TYPES, columns=["cell_id", "ctype"])
c = c.merge(typed, on="cell_id", how="left", validate="one_to_one"); c["ctype"] = c.ctype.fillna("other")
n = len(c); types = sorted(c.ctype.unique()); tcode = c.ctype.map({t: k for k, t in enumerate(types)}).to_numpy()
pcols = [x for x in c.columns if x.startswith("P_") and not x.startswith("P_Channel") and "IgG" not in x]
pnames = [x[2:] for x in pcols]
P = np.log1p(c[pcols].to_numpy(float)); P = (P - P.mean(0)) / (P.std(0) + 1e-9)
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); det = np.asarray((X > 0).mean(0)).ravel()
X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
log("loaded", n, "cells")

import liana as li
res = li.rs.select_resource("consensus").apply(lambda s: s.str.upper())
lr_genes = set(g for col in ["ligand", "receptor"] for x in res[col] for g in x.split("_"))
cont = res[res.ligand.map(lambda x: any(g in CONTACT_LIGANDS for g in x.split("_")))].drop_duplicates().reset_index(drop=True)


def rna_vec(g):
    v = np.asarray(Lc[:, gi[g]].todense()).ravel(); return (v - v.mean()) / (v.std() + 1e-9)


def feat(gene, rna_only):
    vals = []
    for g in gene.split("_"):
        if not rna_only and g in GENE2PROT and GENE2PROT[g] in pnames:
            vals.append(P[:, pnames.index(GENE2PROT[g])]); continue
        if g not in gi or det[gi[g]] < 0.002: return None
        vals.append(rna_vec(g))
    return np.min(np.column_stack(vals), axis=1).astype(np.float32)


pool = np.array([i for i, g in enumerate(genes) if g not in lr_genes and det[i] >= 0.005])
mean_all = np.asarray(Lc.mean(0)).ravel()
Fpool = np.column_stack([mean_all[pool], np.log10(det[pool] + 1e-4)]); mu, sd = Fpool.mean(0), Fpool.std(0)
Fz = (Fpool - mu) / sd
rng_p = np.random.default_rng(1)
channels = []; Lm = {"mo": [], "rna": []}; Rm = {"mo": [], "rna": []}; used = set()
for row in cont.itertuples():
    Lmo, Rmo = feat(row.ligand, False), feat(row.receptor, False)
    Lr, Rr = feat(row.ligand, True), feat(row.receptor, True)
    if Lmo is None or Rmo is None or Lr is None or Rr is None: continue
    name = f"{row.ligand}->{row.receptor}"
    channels.append({"name": name, "ligand": row.ligand, "receptor": row.receptor, "decoy": False, "of": name})
    Lm["mo"].append(Lmo); Rm["mo"].append(Rmo); Lm["rna"].append(Lr); Rm["rna"].append(Rr)
    lig_is_prot = all(g in GENE2PROT and GENE2PROT[g] in pnames for g in row.ligand.split("_"))
    if lig_is_prot:
        ppool = [q for q in pnames if PROT2GENE.get(q, "") not in lr_genes and q not in PROT_READOUT
                 and not q.startswith("Channel")]
        for q in list(rng_p.choice(ppool, size=min(5, len(ppool)), replace=False)):
            dv = P[:, pnames.index(q)].astype(np.float32); gq = PROT2GENE.get(q, "")
            dr = rna_vec(gq).astype(np.float32) if gq in gi else np.zeros(n, np.float32)
            channels.append({"name": f"PDECOY:{q}->{row.receptor}", "ligand": gq or q, "receptor": row.receptor,
                             "decoy": True, "decoy_modality": "protein", "of": name})
            Lm["mo"].append(dv); Rm["mo"].append(Rmo); Lm["rna"].append(dr); Rm["rna"].append(Rr)
    lg = row.ligand.split("_")[0]
    if lg not in gi: continue
    f = (np.array([mean_all[gi[lg]], np.log10(det[gi[lg]] + 1e-4)]) - mu) / sd
    order = np.argsort(((Fz - f) ** 2).sum(1))
    picks = [pool[k] for k in order[:300] if pool[k] not in used][:N_DECOY]
    for k in picks:
        used.add(k); dv = rna_vec(genes[k]).astype(np.float32)
        channels.append({"name": f"DECOY:{genes[k]}->{row.receptor}", "ligand": genes[k], "receptor": row.receptor,
                         "decoy": True, "of": name})
        Lm["mo"].append(dv); Rm["mo"].append(Rmo); Lm["rna"].append(dv); Rm["rna"].append(Rr)
log("channels", len(channels), "real", sum(not ch["decoy"] for ch in channels))

chan_genes = {g for ch in channels for side in ("ligand", "receptor") for g in ch[side].split("_")}
rec_genes = {g for ch in channels for g in ch["receptor"].split("_")}
dropped = [p for p in PROT_READOUT if PROT2GENE.get(p) in rec_genes]
PROT_READOUT = [p for p in PROT_READOUT if p not in dropped]
log("protein readouts dropped (are channel receptors):", dropped, "kept:", PROT_READOUT)
Yp = P[:, [pnames.index(p) for p in PROT_READOUT]].astype(np.float32)
prog = {k: [g for g in v if g in gi and det[gi[g]] >= 0.002 and g not in chan_genes] for k, v in RNA_PROGRAMS.items()}
log("RNA program sizes after removing channel genes:", {k: len(v) for k, v in prog.items()})
Yr = np.column_stack([np.mean([rna_vec(g) for g in v], axis=0) for v in prog.values()]).astype(np.float32)
excl = {g for v in RNA_PROGRAMS.values() for g in v}
excl |= {g for ch in channels for side in ("ligand", "receptor") for g in ch[side].split("_")}
excl |= {PROT2GENE.get(p, "") for p in PROT_READOUT}
cand = np.array([i for i, g in enumerate(genes) if g not in excl and det[i] > 0.02])
sub = Lc[:, cand]; var = np.asarray(sub.power(2).mean(0)).ravel() - np.asarray(sub.mean(0)).ravel() ** 2
hv = cand[np.argsort(-var)[:2000]]
M = Lc[:, hv].toarray(); M -= M.mean(0); M /= M.std(0) + 1e-6
rna_pc = TruncatedSVD(50, random_state=0).fit_transform(M).astype(np.float32); del M
rna_pc /= rna_pc.std(0) + 1e-6
prot_in_idx = [k for k, p in enumerate(pnames) if p not in PROT_READOUT]
Pin = P[:, prot_in_idx].astype(np.float32)
xy = c[["CenterX_global_px", "CenterY_global_px"]].to_numpy(float) * UM
e = pd.read_parquet(D / "contact_edges.parquet"); idx = pd.Series(np.arange(n), index=c.cell_id)
a = idx.reindex(e.cell_id_a).to_numpy(); b = idx.reindex(e.cell_id_b).to_numpy(); ok = ~(np.isnan(a) | np.isnan(b))
a, b, w = a[ok].astype(int), b[ok].astype(int), e.contact_um.to_numpy()[ok].astype(np.float32)
fov = c.fov.to_numpy()
pairs = cKDTree(xy).query_pairs(15.0, output_type="ndarray")
pairs = np.sort(pairs[fov[pairs[:, 0]] == fov[pairs[:, 1]]], 1)
ckey = np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)
near = pairs[~np.isin(pairs[:, 0].astype(np.int64) * n + pairs[:, 1], ckey)]
src = np.r_[a, b, near[:, 0], near[:, 1]]; dst = np.r_[b, a, near[:, 1], near[:, 0]]
contact = np.r_[w, w, np.zeros(2 * len(near), np.float32)]
dist = np.linalg.norm(xy[src] - xy[dst], axis=1).astype(np.float32)
fc = pd.Series(xy[:, 0]).groupby(fov).mean().sort_values()
fold_of = {f: k for k, part in enumerate(np.array_split(fc.index.to_numpy(), 5)) for f in part}
fold = np.array([fold_of[f] for f in fov])
covar = np.column_stack([np.log(c.Area.to_numpy(float)), np.log(lib), np.log1p(c.negprobe_total.to_numpy(float))]).astype(np.float32)
covar = (covar - covar.mean(0)) / covar.std(0)
np.savez_compressed(D / "gt_inputs.npz", rna_pc=rna_pc, prot_in=Pin, ctype=tcode, covar=covar, Yp=Yp, Yr=Yr,
                    L_mo=np.column_stack(Lm["mo"]), R_mo=np.column_stack(Rm["mo"]),
                    L_rna=np.column_stack(Lm["rna"]), R_rna=np.column_stack(Rm["rna"]),
                    src=src, dst=dst, contact=contact, dist=dist, fold=fold, fov=fov, xy=xy.astype(np.float32))
json.dump({"channels": channels, "types": types, "prot_readout": PROT_READOUT, "rna_programs": list(RNA_PROGRAMS),
           "rna_program_genes": prog, "prot_inputs": [pnames[k] for k in prot_in_idx],
           "n_contact_edges": int(2 * len(a)), "n_near_edges": int(2 * len(near))},
          open(D / "gt_channels.json", "w"), indent=1)
log("saved; edges contact", 2 * len(a), "near", 2 * len(near))
