"""Expression-based pair scores (co-expression, CellPhoneDB via LIANA, LIANA+ bivariate, COMMOT) on the
12 evaluation FOVs -> results/baseline_scores_12fov.npz."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, torch
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests
from torch_geometric.utils import softmax
from graticcs.paths import BREAST, RESULTS

D = BREAST; OUT = RESULTS
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); ch = pd.DataFrame(meta["channels"])
sm = json.load(open(D / "spike_meta.json")); T = meta["prot_readout"] + meta["rna_programs"]
from graticcs.data import target_mask
M = target_mask({"L": z["L_mo"], "tnames": T, "channels": meta["channels"]})
real_prot = set(ch.loc[ch.decoy & (ch.get("decoy_modality") == "protein"), "of"])
ch["stratum"] = np.where(ch.decoy, np.where(ch.get("decoy_modality") == "protein", "protein", "rna"), np.where(ch.name.isin(real_prot), "protein", "rna"))
real = np.flatnonzero(~ch.decoy.to_numpy()); planted = [ch.index[ch.name == p[0]][0] for p in sm["plant"]]
src, dst, con, dist, fold, fov = z["src"], z["dst"], z["contact"], z["dist"], z["fold"], z["fov"]
ctype = z["ctype"]; ntypes = int(ctype.max()) + 1; oh = np.eye(ntypes, dtype=np.float32)[ctype]; types = meta["types"]
L, R = z["L_mo"], z["R_mo"]; sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
rng = np.random.default_rng(0); fovs = np.sort(rng.choice(np.unique(fov), 12, replace=False))
cells = np.flatnonzero(np.isin(fov, fovs)); ci = np.flatnonzero((con > 0) & np.isin(fov[dst], fovs))
csrc, cdst = src[ci], dst[ci]
log("subset cells", len(cells), "contact edges", len(ci))
Ls = L[csrc][:, real]; Rs = R[cdst][:, real]
coexpr = np.maximum(Ls, 0) * np.maximum(Rs, 0)
truth = (coexpr > 0) & np.isin(real, planted)[None, :]
log("events", truth.size, "positives", int(truth.sum()), "random AP", truth.mean())
scores = {"co-expression (L x R)": coexpr}

genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: k for k, g in enumerate(genes)}
need = sorted({g for c in real for side in ("ligand", "receptor") for g in ch.loc[c, side].split("_") if g in gi})
X = sp.load_npz(D / "rna_counts.npz").tocsr()[cells].astype(np.float32)
lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1); X = X.multiply((1e4 / lib)[:, None]).tocsr(); X.data = np.log1p(X.data)
A = ad.AnnData(X=X[:, [gi[g] for g in need]].tocsr(), obs=pd.DataFrame({"ctype": pd.Categorical(np.array(types)[ctype[cells]])},
               index=[f"c{i}" for i in cells]), var=pd.DataFrame(index=need))
A.obsm["spatial"] = z["xy"][cells].astype(np.float64)
lr_df = pd.DataFrame({"ligand": ch.ligand[real].to_numpy(), "receptor": ch.receptor[real].to_numpy()})
lr_df = lr_df[lr_df.apply(lambda r: all(g in gi for g in r.ligand.split("_") + r.receptor.split("_")), axis=1)].drop_duplicates()
pos_of = pd.Series(np.arange(len(cells)), index=cells)
ls_, ld_ = pos_of[csrc].to_numpy(), pos_of[cdst].to_numpy()
col_of = {(ch.ligand[c], ch.receptor[c]): k for k, c in enumerate(real)}


def fill(get):
    """build E x C score matrix from a per-(ligand, receptor) getter returning a length-E vector or None."""
    S = np.zeros((len(ci), len(real)), np.float32)
    for (lg, rc), k in col_of.items():
        v = get(lg, rc)
        if v is not None: S[:, k] = np.nan_to_num(np.asarray(v, dtype=np.float32).ravel())
    return S

import graticcs.compat
import liana as li
try:
    li.mt.cellphonedb(A, groupby="ctype", resource=lr_df, expr_prop=0.1, use_raw=False, n_perms=100, verbose=False)
    r = A.uns["liana_res"]; tsrc = A.obs.ctype.to_numpy()[ls_]; tdst = A.obs.ctype.to_numpy()[ld_]
    idx = r.set_index(["source", "target", "ligand_complex", "receptor_complex"]).lr_means
    def g_cpdb(lg, rc):
        key = pd.MultiIndex.from_arrays([tsrc, tdst, [lg] * len(tsrc), [rc] * len(tsrc)])
        return idx.reindex(key).to_numpy()
    scores["CellPhoneDB (LIANA, type level)"] = fill(g_cpdb); log("cellphonedb done")
except Exception as e: log("cellphonedb failed:", repr(e)[:300])
try:
    li.ut.spatial_neighbors(A, bandwidth=30.0, cutoff=0.1, set_diag=False, spatial_key="spatial")
    lr = li.mt.bivariate(A, resource=lr_df[~lr_df.ligand.str.contains("_") & ~lr_df.receptor.str.contains("_")], local_name="cosine", global_name=None, nz_prop=0.001, use_raw=False, verbose=False)
    vi = {v: k for k, v in enumerate(lr.var_names)}; LX = lr.X.toarray() if sp.issparse(lr.X) else np.asarray(lr.X)
    scores["LIANA+ bivariate cosine"] = fill(lambda lg, rc: LX[ld_, vi[f"{lg}^{rc}"]] if f"{lg}^{rc}" in vi else None)
    log("liana bivariate done")
except Exception as e: log("bivariate failed:", repr(e)[:300])
try:
    import commot as ct
    dfl = lr_df.assign(pathway="p").reset_index(drop=True)
    ct.tl.spatial_communication(A, database_name="u", df_ligrec=dfl[["ligand", "receptor", "pathway"]].set_axis([0, 1, 2], axis=1), dis_thr=30.0,
                                heteromeric=True, heteromeric_delimiter="_", pathway_sum=False)
    def g_commot(lg, rc):
        k = f"commot-u-{lg}-{rc}"
        return np.asarray(A.obsp[k][ls_, ld_]).ravel() if k in A.obsp else None
    scores["COMMOT"] = fill(g_commot); log("commot done")
except Exception as e: log("commot failed:", repr(e)[:300])


scores["co-expression (RNA only)"] = np.maximum(z["L_rna"][csrc][:, real], 0) * np.maximum(z["R_rna"][cdst][:, real], 0)
np.savez_compressed(OUT / "baseline_scores_12fov.npz", ci=ci, real=real, **{k.replace(" ", "_"): v.astype(np.float16) for k, v in scores.items()})
log("saved baseline scores:", list(scores))
