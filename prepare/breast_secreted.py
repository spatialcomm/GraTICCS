"""Secreted channels (30 um neighbourhoods) and their decoys -> gt_sec_inputs.npz, gt_sec_channels.json."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.spatial import cKDTree
from graticcs.lr_map import PROT2GENE, GENE2PROT, CONTACT_LIGANDS, SECRETED_FAMILY
from graticcs.paths import BREAST

D = BREAST
R_SEC = 30.0; N_DECOY = 5
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
c = pd.read_parquet(D / "cells.parquet"); n = len(c)
pcols = [x for x in c.columns if x.startswith("P_") and not x.startswith("P_Channel") and "IgG" not in x]; pnames = [x[2:] for x in pcols]
P = np.log1p(c[pcols].to_numpy(float)); P = (P - P.mean(0)) / (P.std(0) + 1e-9)
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); rawmean = np.asarray(X.mean(0)).ravel()
X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data); Lc = X.tocsc(); del X
xneg = rawmean / (c.negprobe_total.mean() / 50); det = np.asarray((Lc > 0).mean(0)).ravel()
cmeta = json.load(open(D / "gt_channels.json")); old_genes = {g for ch in cmeta["channels"] for s in ("ligand", "receptor") for g in ch[s].split("_")}
import liana as li
res = li.rs.select_resource("consensus").apply(lambda s: s.str.upper()).drop_duplicates()
lr_genes = set(g for col in ["ligand", "receptor"] for x in res[col] for g in x.split("_"))
issec = res.ligand.map(lambda x: all(SECRETED_FAMILY.match(g) for g in x.split("_")) and not any(g in CONTACT_LIGANDS for g in x.split("_")))
xl = lambda x: min(xneg[gi[g]] if g in gi else 0 for g in x.split("_"))
protlig = res.ligand.map(lambda x: all(g in GENE2PROT and GENE2PROT[g] in pnames for g in x.split("_")) and not any(g in CONTACT_LIGANDS for g in x.split("_")))
sec = res[(issec & (res.ligand.map(xl) >= 3) & (res.receptor.map(xl) >= 3)) | (protlig & ~res.ligand.map(lambda x: any(g in CONTACT_LIGANDS for g in x.split("_"))))].drop_duplicates()
log("secreted real pairs", len(sec), "ligands", sec.ligand.nunique())
def rna_vec(g):
    v = np.asarray(Lc[:, gi[g]].todense()).ravel(); return ((v - v.mean()) / (v.std() + 1e-9)).astype(np.float32)
def feat(gene):
    vals = []
    for g in gene.split("_"):
        if g in GENE2PROT and GENE2PROT[g] in pnames: vals.append(P[:, pnames.index(GENE2PROT[g])].astype(np.float32)); continue
        if g not in gi: return None
        vals.append(rna_vec(g))
    return np.min(np.column_stack(vals), axis=1)
pool = np.array([i for i, g in enumerate(genes) if g not in lr_genes and xneg[i] >= 1.5])
Fp = np.column_stack([np.log(xneg[pool] + 1e-3), det[pool]]); mu, sdv = Fp.mean(0), Fp.std(0); Fz = (Fp - mu) / sdv
ppool = [q for q in pnames if PROT2GENE.get(q, "") not in lr_genes and not q.startswith("Channel")]
rng = np.random.default_rng(2); channels, Lm, Rm, used = [], [], [], set()
for row in sec.itertuples():
    Lv, Rv = feat(row.ligand), feat(row.receptor)
    if Lv is None or Rv is None: continue
    name = f"{row.ligand}->{row.receptor}"
    channels.append({"name": name, "ligand": row.ligand, "receptor": row.receptor, "decoy": False, "of": name}); Lm.append(Lv); Rm.append(Rv)
    lig_prot = all(g in GENE2PROT and GENE2PROT[g] in pnames for g in row.ligand.split("_"))
    if lig_prot:
        for q in rng.choice(ppool, size=N_DECOY, replace=False):
            channels.append({"name": f"PDECOY:{q}->{row.receptor}", "ligand": PROT2GENE.get(q, q), "receptor": row.receptor,
                             "decoy": True, "decoy_modality": "protein", "of": name}); Lm.append(P[:, pnames.index(q)].astype(np.float32)); Rm.append(Rv)
    else:
        lg = row.ligand.split("_")[0]
        f = (np.array([np.log(xneg[gi[lg]] + 1e-3), det[gi[lg]]]) - mu) / sdv
        picks = [pool[k] for k in np.argsort(((Fz - f) ** 2).sum(1))[:400] if pool[k] not in used][:N_DECOY]
        for k in picks:
            used.add(k); channels.append({"name": f"DECOY:{genes[k]}->{row.receptor}", "ligand": genes[k], "receptor": row.receptor,
                                          "decoy": True, "decoy_modality": "rna", "of": name}); Lm.append(rna_vec(genes[k])); Rm.append(Rv)
log("channels", len(channels), "real", sum(not x["decoy"] for x in channels))
RNA_PROGRAMS = {
    "IFN_GAS": ["STAT1", "IRF1", "GBP1", "GBP4", "GBP5", "CXCL9", "CXCL10", "CXCL11", "IDO1", "PSMB9", "TAP1", "WARS1", "SOCS1", "CIITA"],
    "NFKB": ["NFKBIA", "TNFAIP3", "ICAM1", "BIRC3", "RELB", "NFKB1", "NFKB2", "TNFAIP2", "CXCL8", "SOD2", "CCL20"],
    "PROLIF": ["MKI67", "TOP2A", "CENPF", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "CDK1", "CCNB1", "MCM2"],
    "CYTOTOX": ["GZMB", "GZMA", "GZMK", "PRF1", "NKG7", "GNLY", "CST7", "CTSW"],
    "TACT": ["ICOS", "TNFRSF9", "CD69", "IL2RA", "TNFRSF4", "TNFRSF18", "CRTAM", "NR4A1", "NR4A2", "EGR2"],
    "EXHAUST": ["PDCD1", "HAVCR2", "LAG3", "TIGIT", "TOX", "CTLA4", "ENTPD1", "CXCL13"],
}
sec_genes = {g for ch in channels for s in ("ligand", "receptor") for g in ch[s].split("_")}
excl = old_genes | sec_genes
prog = {k: [g for g in v if g in gi and det[gi[g]] >= 0.002 and g not in excl] for k, v in RNA_PROGRAMS.items()}
prog = {k: v for k, v in prog.items() if len(v) >= 3}
log("RNA programs after excluding all channel genes:", {k: len(v) for k, v in prog.items()})
Yr = np.column_stack([np.mean([rna_vec(g) for g in v], axis=0) for v in prog.values()]).astype(np.float32)
rec_genes = {g for ch in channels for g in ch["receptor"].split("_")}
z0 = np.load(D / "gt_inputs.npz"); PROT_READOUT = cmeta["prot_readout"]
keep_p = [k for k, p in enumerate(PROT_READOUT) if PROT2GENE.get(p) not in rec_genes]
log("protein readouts kept (not a secreted receptor):", [PROT_READOUT[k] for k in keep_p])
Yp = z0["Yp"][:, keep_p]
xy = z0["xy"]; fov = z0["fov"]
pairs = cKDTree(xy).query_pairs(R_SEC, output_type="ndarray"); pairs = pairs[fov[pairs[:, 0]] == fov[pairs[:, 1]]]
src = np.r_[pairs[:, 0], pairs[:, 1]]; dst = np.r_[pairs[:, 1], pairs[:, 0]]
csrc, cdst, ccon = z0["src"], z0["dst"], z0["contact"]
ck = pd.Series(ccon[ccon > 0], index=csrc[ccon > 0].astype(np.int64) * n + cdst[ccon > 0])
con = ck.reindex(src.astype(np.int64) * n + dst).fillna(0).to_numpy(np.float32)
dist = np.linalg.norm(xy[src] - xy[dst], axis=1).astype(np.float32)
np.savez_compressed(D / "gt_sec_inputs.npz", L=np.column_stack(Lm).astype(np.float16), R=np.column_stack(Rm).astype(np.float16),
                    src=src, dst=dst, contact=con, dist=dist, Yp=Yp, Yr=Yr)
json.dump({"channels": channels, "prot_readout": [PROT_READOUT[k] for k in keep_p], "rna_programs": list(prog), "rna_program_genes": prog,
           "radius_um": R_SEC, "n_edges": int(len(src))}, open(D / "gt_sec_channels.json", "w"), indent=1)
log("saved; secreted edges", len(src))
