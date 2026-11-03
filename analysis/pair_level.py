"""Pair-level average precision: GraTICCS (1 - q) x event contribution against the expression-based scores,
and the expression-based scores weighted by 1 - q (protein- and RNA-defined planted truth)."""
import json
import torch
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests
from torch_geometric.utils import softmax
from graticcs.paths import BREAST, RESULTS

D = BREAST; R = RESULTS
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); ch = pd.DataFrame(meta["channels"])
T = meta["prot_readout"] + meta["rna_programs"]
from graticcs.data import target_mask
M = target_mask({"L": z["L_mo"], "tnames": T, "channels": meta["channels"]})
real_prot = set(ch.loc[ch.decoy & (ch.get("decoy_modality") == "protein"), "of"])
ch["stratum"] = np.where(ch.decoy, np.where(ch.get("decoy_modality") == "protein", "protein", "rna"), np.where(ch.name.isin(real_prot), "protein", "rna"))
B = np.load(R / "baseline_scores_12fov.npz"); ci = B["ci"]; real = B["real"]
src, dst, con, dist, fold = z["src"], z["dst"], z["contact"], z["dist"], z["fold"]
csrc, cdst = src[ci], dst[ci]; ctype = z["ctype"]; oh = np.eye(int(ctype.max()) + 1, dtype=np.float32)[ctype]
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
NATURAL = {"HLA-A->CD8A", "HLA-DRA->CD4"}; keep = ~ch.name[real].isin(NATURAL).to_numpy()
base = {k: B[k].astype(np.float32) for k in B.files if k not in ("ci", "real")}


def truth_for(meta_file, Lkey, Rkey):
    sm = json.load(open(D / meta_file)); names = ch.name[real].to_numpy()
    pl = np.isin(names, [p[0] for p in sm["plant"] if p[0] not in NATURAL])
    return (np.maximum(z[Lkey][csrc][:, real], 0) * np.maximum(z[Rkey][cdst][:, real], 0) > 0) & pl[None, :], sm


pos = -np.ones(len(src_ := z["src"]), np.int64); pos[ci] = np.arange(len(ci))


def ens(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    ib = (ds / s0).max(1); q = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    S = np.zeros((len(ci), len(real)), np.float32)
    for d in dirs:
        for k in range(5):
            st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]; G, V = st["G"].numpy(), st["V"].numpy()
            P = torch.softmax(st["P"], 1).numpy()[real]; Vn = np.linalg.norm((V * M)[real], axis=1)
            A = np.load(d / f"fold{k}_mo_attention.npz"); p_ = pos[A["edge"]]; m = p_ >= 0
            e, a = p_[m], A["a"][m].astype(np.float32) @ P.T
            s = sp_(z["L_mo"][csrc[e]][:, real]) * sp_(z["R_mo"][cdst[e]][:, real]); g = 1 / (1 + np.exp(-G[ctype[cdst[e]]][:, real]))
            S[e] += a * s * g * Vn[None, :] / len(dirs)
    return (1 - q[real])[None, :] * S


def channel_evidence(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    ib = (ds / s0).max(1); q = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    return (1 - q)[real]


if __name__ == "__main__":
    rows = []
    for label, mf, Lk, Rk, runs in [
            ("protein-defined truth", "spike_meta.json", "L_mo", "R_mo", lambda b: [R / f"G_spike_b{b}_s{s}" for s in (0, 1, 2)]),
            ("RNA-defined truth", "spike_rna_meta.json", "L_rna", "R_rna", lambda b: [R / f"G_spikeRNA_b{b}"])]:
        truth, sm = truth_for(mf, Lk, Rk); y = truth[:, keep].ravel()
        print(f"\n=== {label}: events {y.size}, positives {int(y.sum())}, random AP {y.mean():.5f}")
        for k, S in base.items():
            rows.append(dict(truth=label, method=k, beta="any", AP=average_precision_score(y, S[:, keep].ravel())))
        for b in sm["betas"]:
            dirs = [d for d in runs(b) if len(list(d.glob("fold*_mo_attention.npz"))) == 5]
            if not dirs: continue
            S = ens(dirs); rows.append(dict(truth=label, method=f"GraTICCS-GT ({len(dirs)}-seed)", beta=b, AP=average_precision_score(y, S[:, keep].ravel())))
        print(pd.DataFrame([r for r in rows if r["truth"] == label]).to_string(index=False))
    pd.DataFrame(rows).to_csv(R / "final_pair_level_comparison_GT.csv", index=False)

    rows = []
    for label, mf, Lk, Rk, runs, evid in [
            ("protein-defined", "spike_meta.json", "L_mo", "R_mo", lambda b: [R / f"G_spike_b{b}_s{s}" for s in (0, 1, 2)],
             {"co-expression": base["co-expression_(L_x_R)"]}),
            ("RNA-defined", "spike_rna_meta.json", "L_rna", "R_rna", lambda b: [R / f"G_spikeRNA_b{b}"],
             {"RNA co-expression": base["co-expression_(RNA_only)"], "COMMOT": base["COMMOT"]})]:
        truth, sm = truth_for(mf, Lk, Rk); y = truth[:, keep].ravel()
        for b in sm["betas"]:
            dirs = [d for d in runs(b) if len(list(d.glob("fold*_mo.json"))) == 5]
            if not dirs: continue
            p = channel_evidence(dirs)
            for en, E in evid.items():
                rows.append(dict(truth=label, beta=b, method=f"{en} alone", AP=average_precision_score(y, E[:, keep].ravel())))
                rows.append(dict(truth=label, beta=b, method=f"GraTICCS (1 - q) x {en}", AP=average_precision_score(y, (p[None, :] * E)[:, keep].ravel())))
    res = pd.DataFrame(rows); print(res.to_string(index=False)); res.to_csv(R / "hybrid_pair_level_GT.csv", index=False)
