"""PROGENy pathway and CollecTRI TF activities (ULM, decoupler) with all channel genes removed, and the
channel x readout masks from the receptor-family rules."""
import json, re, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, decoupler as dc
from graticcs.paths import BREAST
D = BREAST; OUTD = D / "pathway_tf"; OUTD.mkdir(exist_ok=True)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
from graticcs.pathways import PATH_TF, families
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]
cmeta = json.load(open(D / "gt_channels.json")); smeta = json.load(open(D / "gt_sec_channels.json"))
chan_genes = {g for m in (cmeta, smeta) for c in m["channels"] for s in ("ligand", "receptor") for g in c[s].split("_")}
X = sp.load_npz(D / "rna_counts.npz").tocsr().astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data)
keep = np.array([g not in chan_genes for g in genes]); A = ad.AnnData(X=X[:, keep], var=pd.DataFrame(index=np.array(genes)[keep]))
log("expression for activity: genes", A.n_vars, "(removed channel genes:", int((~keep).sum()), ")")
prog = dc.op.progeny(organism="human", top=500); tfs = sorted({t for v in PATH_TF.values() for t in v})
col = dc.op.collectri(organism="human"); col = col[col.source.isin(tfs)]
dc.mt.ulm(A, prog, tmin=5); P = A.obsm["score_ulm"].copy(); del A.obsm["score_ulm"]
dc.mt.ulm(A, col, tmin=5); Tf = A.obsm["score_ulm"].copy()
log("PROGENy", P.shape, "TFs", Tf.shape, list(Tf.columns))
Y = pd.concat([P.add_prefix("PW_"), Tf.add_prefix("TF_")], axis=1); Y = (Y - Y.mean()) / (Y.std() + 1e-9)
names = list(Y.columns); np.savez_compressed(OUTD / "Y_pathway_tf.npz", Y=Y.to_numpy(np.float32))
def mask_for(channels):
    Mk = np.zeros((len(channels), len(names)), np.float32)
    for ci, c in enumerate(channels):
        fam = families(c["receptor"]); allowed = set()
        for f in fam:
            if f in P.columns: allowed.add(f"PW_{f}")
            allowed |= {f"TF_{t}" for t in PATH_TF.get(f, [])}
        for t, nm in enumerate(names):
            if nm in allowed: Mk[ci, t] = 1
    return Mk
Mc = mask_for(cmeta["channels"]); Ms = mask_for(smeta["channels"])
np.save(OUTD / "mask_contact.npy", Mc); np.save(OUTD / "mask_secreted.npy", Ms)
json.dump({"readouts": names}, open(OUTD / "meta.json", "w"), indent=1)
rc = sum(Mc[i].any() for i, c in enumerate(cmeta["channels"]) if not c["decoy"]); rs = sum(Ms[i].any() for i, c in enumerate(smeta["channels"]) if not c["decoy"])
log(f"real contact channels with >=1 allowed readout: {rc}/{sum(not c['decoy'] for c in cmeta['channels'])}; secreted: {rs}/{sum(not c['decoy'] for c in smeta['channels'])}")
