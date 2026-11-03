"""Contact channels, decoys and readouts for the Xenium renal cell carcinoma section."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, scanpy as sc, shapely
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from graticcs.lr_map import CONTACT_LIGANDS
from graticcs.paths import XENIUM

D = XENIUM
TILE = 1000.0; TOL = 1.0; N_DECOY = 5
PMAP = {"PD-1": "PDCD1", "VISTA": "VSIR", "PD-L1": "CD274", "LAG-3": "LAG3", "CD16": "FCGR3A", "GranzymeB": "GZMB", "CD163": "CD163",
        "CD4": "CD4", "CD20": "MS4A1", "CD8A": "CD8A", "CD3E": "CD3E", "CD138": "SDC1", "HLA-DR": "HLA-DRA", "CD11c": "ITGAX", "CD68": "CD68",
        "PCNA": "PCNA", "Ki-67": "MKI67", "Beta-catenin": "CTNNB1", "CD31": "PECAM1", "PTEN": "PTEN", "PanCK": "KRT8", "Vimentin": "VIM",
        "alphaSMA": "ACTA2", "CD45": "PTPRC", "E-Cadherin": "CDH1", "CD45RA": "PTPRC_RA", "CD45RO": "PTPRC_RO"}
G2P = {v: k for k, v in PMAP.items()}
PROT_READOUT = ["GranzymeB", "Ki-67", "PCNA", "HLA-DR", "PD-L1"]
RNA_PROGRAMS = {"CYTOTOX": ["GZMA", "GZMB", "GZMK", "PRF1", "NKG7", "GNLY", "CST7", "CTSW"],
                "PROLIF": ["MKI67", "TOP2A", "CENPF", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "CDK1", "CCNB1", "MCM2", "PCNA"]}
TYPES = {"epithelial": ["PanCK", "E-Cadherin"], "T4": ["CD3E", "CD4"], "T8": ["CD3E", "CD8A"], "B": ["CD20"], "plasma": ["CD138"],
         "myeloid": ["CD68", "CD163", "CD11c"], "endothelial": ["CD31"], "stromal": ["alphaSMA"]}
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)

a = sc.read_10x_h5(D / "cell_feature_matrix.h5", gex_only=False); a.var_names = a.var_names.astype(str)
cells = pd.read_parquet(D / "cells.parquet"); assert (cells.cell_id.astype(str).to_numpy() == a.obs_names.to_numpy()).all()
n = a.n_obs; xy = cells[["x_centroid", "y_centroid"]].to_numpy(float)
isg = (a.var.feature_types == "Gene Expression").to_numpy(); isp = (a.var.feature_types == "Protein Expression").to_numpy()
X = a.X[:, isg].tocsr().astype(np.float32); genes = [g.upper() for g in a.var_names[isg]]; gi = {g: k for k, g in enumerate(genes)}
Pr = a.X[:, isp].toarray().astype(np.float64); pnames = list(a.var_names[isp])
P = np.log1p(np.maximum(Pr, 0)); P = (P - P.mean(0)) / (P.std(0) + 1e-9)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); det = np.asarray((X > 0).mean(0)).ravel()
X = X.multiply((1e2 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
log("cells", n, "genes", len(genes), "proteins", len(pnames))
sc_ = np.column_stack([P[:, [pnames.index(m) for m in ms]].mean(1) for ms in TYPES.values()])
imm = P[:, pnames.index("CD45")]
for k, t in enumerate(TYPES):
    if t in ("T4", "T8", "B", "plasma", "myeloid"): sc_[:, k] += 0.5 * imm
best = sc_.argmax(1); ctype_names = list(TYPES) + ["other"]
ctype = np.where(sc_.max(1) > 0.5, best, len(TYPES)); log("types", pd.Series(np.array(ctype_names)[ctype]).value_counts().to_dict())
tile = (np.floor(xy[:, 0] / TILE) * 1000 + np.floor(xy[:, 1] / TILE)).astype(np.int64)
fov = pd.factorize(tile)[0]
fc = pd.Series(xy[:, 0]).groupby(fov).mean().sort_values()
fold_of = {f: k for k, part in enumerate(np.array_split(fc.index.to_numpy(), 5)) for f in part}; fold = np.array([fold_of[f] for f in fov])
b = pd.read_parquet(D / "cell_boundaries.parquet"); cid = {c: i for i, c in enumerate(cells.cell_id.astype(str))}
b["i"] = b.cell_id.astype(str).map(cid); b = b.dropna(subset=["i"]); b["i"] = b.i.astype(int)
geoms = [None] * n
for i, g in b.groupby("i", sort=False):
    pts = g[["vertex_x", "vertex_y"]].to_numpy(float)
    if len(pts) >= 3:
        pg = shapely.Polygon(pts); geoms[i] = pg if pg.is_valid else shapely.make_valid(pg).buffer(0)
geoms = np.array(geoms, dtype=object); ok = np.flatnonzero([g is not None and not g.is_empty for g in geoms])
tree = shapely.STRtree(geoms[ok]); pr = tree.query(geoms[ok], predicate="dwithin", distance=TOL)
ea, eb = ok[pr[0]], ok[pr[1]]; keep = (ea < eb) & (fov[ea] == fov[eb]); ea, eb = ea[keep], eb[keep]
inter = shapely.area(shapely.intersection(shapely.buffer(geoms[ea], TOL / 2), shapely.buffer(geoms[eb], TOL / 2)))
w = (inter / TOL).astype(np.float32)
log("contact pairs", len(ea), "median contact um", float(np.median(w)), "median degree", float(np.median(np.bincount(np.r_[ea, eb], minlength=n))))
pairs = cKDTree(xy).query_pairs(15.0, output_type="ndarray"); pairs = np.sort(pairs[fov[pairs[:, 0]] == fov[pairs[:, 1]]], 1)
ck = np.minimum(ea, eb).astype(np.int64) * n + np.maximum(ea, eb); near = pairs[~np.isin(pairs[:, 0].astype(np.int64) * n + pairs[:, 1], ck)]
src = np.r_[ea, eb, near[:, 0], near[:, 1]]; dst = np.r_[eb, ea, near[:, 1], near[:, 0]]
contact = np.r_[w, w, np.zeros(2 * len(near), np.float32)]; dist = np.linalg.norm(xy[src] - xy[dst], axis=1).astype(np.float32)
import liana as li
res = li.rs.select_resource("consensus").apply(lambda s: s.str.upper()).drop_duplicates()
lr_genes = set(g for col in ["ligand", "receptor"] for x in res[col] for g in x.split("_"))
def rna_vec(g):
    v = np.asarray(Lc[:, gi[g]].todense()).ravel(); return ((v - v.mean()) / (v.std() + 1e-9)).astype(np.float32)
def feat(gene, rna_only):
    vals = []
    for g in gene.split("_"):
        if not rna_only and g in G2P and G2P[g] in pnames: vals.append(P[:, pnames.index(G2P[g])].astype(np.float32)); continue
        if g not in gi: return None
        vals.append(rna_vec(g))
    return np.min(np.column_stack(vals), axis=1)
cont = res[res.ligand.map(lambda x: any(g in CONTACT_LIGANDS for g in x.split("_")))]
pool = np.array([i for i, g in enumerate(genes) if g not in lr_genes and det[i] >= 0.005])
ppool = [q for q in pnames if PMAP.get(q, "") not in lr_genes and q not in PROT_READOUT]
mean_all = np.asarray(Lc.mean(0)).ravel(); Fp = np.column_stack([mean_all[pool], np.log10(det[pool] + 1e-4)]); mu, sdv = Fp.mean(0), Fp.std(0)
Fz = (Fp - mu) / sdv; rng = np.random.default_rng(1)
channels = []; Lm = {"mo": [], "rna": []}; Rm = {"mo": [], "rna": []}; used = set()
for row in cont.itertuples():
    Lmo, Rmo, Lr, Rr = feat(row.ligand, False), feat(row.receptor, False), feat(row.ligand, True), feat(row.receptor, True)
    if Lmo is None or Rmo is None: continue
    Lr = Lr if Lr is not None else np.zeros(n, np.float32); Rr = Rr if Rr is not None else np.zeros(n, np.float32)
    name = f"{row.ligand}->{row.receptor}"
    channels.append({"name": name, "ligand": row.ligand, "receptor": row.receptor, "decoy": False, "of": name})
    Lm["mo"].append(Lmo); Rm["mo"].append(Rmo); Lm["rna"].append(Lr); Rm["rna"].append(Rr)
    if all(g in G2P and G2P[g] in pnames for g in row.ligand.split("_")):
        for q in rng.choice(ppool, size=min(N_DECOY, len(ppool)), replace=False):
            gq = PMAP.get(q, q); dv = P[:, pnames.index(q)].astype(np.float32); dr = rna_vec(gq) if gq in gi else np.zeros(n, np.float32)
            channels.append({"name": f"PDECOY:{q}->{row.receptor}", "ligand": gq, "receptor": row.receptor, "decoy": True, "decoy_modality": "protein", "of": name})
            Lm["mo"].append(dv); Rm["mo"].append(Rmo); Lm["rna"].append(dr); Rm["rna"].append(Rr)
    lg = row.ligand.split("_")[0]
    if lg in gi:
        f = (np.array([mean_all[gi[lg]], np.log10(det[gi[lg]] + 1e-4)]) - mu) / sdv
        for k in [pool[j] for j in np.argsort(((Fz - f) ** 2).sum(1))[:200] if pool[j] not in used][:N_DECOY]:
            used.add(k); dv = rna_vec(genes[k])
            channels.append({"name": f"DECOY:{genes[k]}->{row.receptor}", "ligand": genes[k], "receptor": row.receptor, "decoy": True, "decoy_modality": "rna", "of": name})
            Lm["mo"].append(dv); Rm["mo"].append(Rmo); Lm["rna"].append(dv); Rm["rna"].append(Rr)
log("channels", len(channels), "real", sum(not c["decoy"] for c in channels), [c["name"] for c in channels if not c["decoy"]])
chan_genes = {g for c in channels for s in ("ligand", "receptor") for g in c[s].split("_")}
rec_genes = {g for c in channels for g in c["receptor"].split("_")}
PROT_READOUT = [p for p in PROT_READOUT if PMAP.get(p) not in rec_genes]
Yp = P[:, [pnames.index(p) for p in PROT_READOUT]].astype(np.float32)
prog = {k: [g for g in v if g in gi and g not in chan_genes] for k, v in RNA_PROGRAMS.items()}; prog = {k: v for k, v in prog.items() if len(v) >= 2}
Yr = np.column_stack([np.mean([rna_vec(g) for g in v], axis=0) for v in prog.values()]).astype(np.float32)
excl = chan_genes | {g for v in RNA_PROGRAMS.values() for g in v} | {PMAP.get(p, "") for p in PROT_READOUT}
cand = np.array([i for i, g in enumerate(genes) if g not in excl and det[i] > 0.01])
M = Lc[:, cand].toarray(); M -= M.mean(0); M /= M.std(0) + 1e-6
rna_pc = TruncatedSVD(min(30, len(cand) - 1), random_state=0).fit_transform(M).astype(np.float32); del M; rna_pc /= rna_pc.std(0) + 1e-6
prot_in = P[:, [k for k, p in enumerate(pnames) if p not in PROT_READOUT]].astype(np.float32)
covar = np.column_stack([np.log(cells.cell_area.to_numpy(float) + 1), np.log(lib), np.log1p(cells.control_probe_counts.to_numpy(float))]).astype(np.float32)
covar = (covar - covar.mean(0)) / (covar.std(0) + 1e-9)
np.savez_compressed(D / "gt_inputs.npz", rna_pc=rna_pc, prot_in=prot_in, ctype=ctype, covar=covar, Yp=Yp, Yr=Yr,
                    L_mo=np.column_stack(Lm["mo"]), R_mo=np.column_stack(Rm["mo"]), L_rna=np.column_stack(Lm["rna"]), R_rna=np.column_stack(Rm["rna"]),
                    src=src, dst=dst, contact=contact, dist=dist, fold=fold, fov=fov, xy=xy.astype(np.float32))
json.dump({"channels": channels, "types": ctype_names, "prot_readout": PROT_READOUT, "rna_programs": list(prog), "rna_program_genes": prog,
           "prot_inputs": [p for p in pnames if p not in PROT_READOUT], "n_contact_edges": int(2 * len(ea)), "n_near_edges": int(2 * len(near))},
          open(D / "gt_channels.json", "w"), indent=1)
log("saved; readouts", PROT_READOUT, list(prog))
