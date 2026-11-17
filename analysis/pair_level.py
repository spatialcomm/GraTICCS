"""Cell-pair ranking for the planted contact signals on the 12 cached imaging fields.

Event score: (1 - q_c) x neighbour weight x softplus(L) softplus(R) x sigmoid(G) x ||V M||, each event scored by the fold
model that held out its receiver. Neighbour weight: saved attention (GraTICCS), 1/degree (local model),
2 sigmoid(T[sender type]) / degree (sender-type rule) or shared boundary / total boundary (contact-extent rule).

python -m analysis.pair_level contact          protein- and RNA-defined truth, GraTICCS against expression baselines
python -m analysis.pair_level local            matched local model, protein-defined truth
python -m analysis.pair_level hybrid           expression baselines weighted by GraTICCS channel evidence 1 - q
python -m analysis.pair_level sender_type      selective-sender tests: truth follows the generating rule;
python -m analysis.pair_level contact_extent   AP over all events and among co-expressing events of planted channels
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

from graticcs.data import target_mask
from graticcs.paths import BREAST, RESULTS

D = BREAST; R = RESULTS
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); ch = pd.DataFrame(meta["channels"])
T = meta["prot_readout"] + meta["rna_programs"]
M = target_mask({"L": z["L_mo"], "tnames": T, "channels": meta["channels"]})
real_prot = set(ch.loc[ch.decoy & (ch.get("decoy_modality") == "protein"), "of"])
ch["stratum"] = np.where(ch.decoy, np.where(ch.get("decoy_modality") == "protein", "protein", "rna"),
                         np.where(ch.name.isin(real_prot), "protein", "rna"))
B = np.load(R / "baseline_scores_12fov.npz"); ci = B["ci"]; real = B["real"]
src, dst, con, dist, fold = z["src"], z["dst"], z["contact"], z["dist"], z["fold"]
csrc, cdst = src[ci], dst[ci]; ctype = z["ctype"]
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
NATURAL = {"HLA-A->CD8A", "HLA-DRA->CD4"}; keep = ~ch.name[real].isin(NATURAL).to_numpy()
base = {k: B[k].astype(np.float32) for k in B.files if k not in ("ci", "real")}
pos = -np.ones(len(src), np.int64); pos[ci] = np.arange(len(ci))
deg = np.bincount(dst[con > 0], minlength=len(fold)).clip(min=1)
csum = np.bincount(dst[con > 0], weights=con[con > 0], minlength=len(fold)).clip(min=1e-6)


def channel_q(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    ib = (ds / s0).max(1); q = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    return q


def pair_scores(dirs, weight="attention"):
    """weight: attention | local | sender_type | contact_extent."""
    q = channel_q(dirs)
    S = np.zeros((len(ci), len(real)), np.float32); fk = fold[cdst]
    for d in dirs:
        for k in range(5):
            st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]; G, V = st["G"].numpy(), st["V"].numpy()
            Vn = np.linalg.norm((V * M)[real], axis=1)
            if weight == "attention":
                P = torch.softmax(st["P"], 1).numpy()[real]
                A = np.load(d / f"fold{k}_mo_attention.npz"); p_ = pos[A["edge"]]; m = p_ >= 0
                e, w = p_[m], A["a"][m].astype(np.float32) @ P.T
            else:
                e = np.flatnonzero(fk == k)
                if weight == "local":
                    w = (1.0 / deg[cdst[e]])[:, None]
                elif weight == "sender_type":
                    w = (1.0 / deg[cdst[e]])[:, None] * 2 / (1 + np.exp(-st["T"].numpy()[ctype[csrc[e]]][:, real]))
                else:
                    w = (con[ci][e] / csum[cdst[e]])[:, None]
            s = sp_(z["L_mo"][csrc[e]][:, real]) * sp_(z["R_mo"][cdst[e]][:, real])
            g = 1 / (1 + np.exp(-G[ctype[cdst[e]]][:, real]))
            S[e] += w * s * g * Vn[None, :] / len(dirs)
    return (1 - q[real])[None, :] * S


def truth_for(meta_file, Lkey, Rkey):
    sm = json.load(open(D / meta_file)); names = ch.name[real].to_numpy()
    pl = np.isin(names, [p[0] for p in sm["plant"] if p[0] not in NATURAL])
    return (np.maximum(z[Lkey][csrc][:, real], 0) * np.maximum(z[Rkey][cdst][:, real], 0) > 0) & pl[None, :], sm


def seeds(prefix):
    return [R / f"{prefix}_s{s}" for s in (0, 1, 2)]


def contact():
    rows = []
    for label, mf, Lk, Rk, pre in [("protein-defined truth", "spike_crand_meta.json", "L_mo", "R_mo", "G_crand"),
                                   ("RNA-defined truth", "spike_crand_rna_meta.json", "L_rna", "R_rna", "G_crand_rna")]:
        truth, sm = truth_for(mf, Lk, Rk); y = truth[:, keep].ravel()
        print(f"{label}: events {y.size}, positives {int(y.sum())}, random AP {y.mean():.5f}")
        for k, S in base.items():
            rows.append(dict(truth=label, method=k, beta="any", AP=average_precision_score(y, S[:, keep].ravel())))
        for b in sm["betas"]:
            dirs = [d for d in seeds(f"{pre}_b{b}") if len(list(d.glob("fold*_mo_attention.npz"))) == 5]
            if not dirs:
                continue
            S = pair_scores(dirs)
            rows.append(dict(truth=label, method=f"GraTICCS-GT ({len(dirs)}-seed)", beta=b, AP=average_precision_score(y, S[:, keep].ravel())))
    out = pd.DataFrame(rows); out.to_csv(R / "final_pair_level_comparison_crand.csv", index=False); print(out.to_string(index=False))


def local():
    truth, sm = truth_for("spike_crand_meta.json", "L_mo", "R_mo"); y = truth[:, keep].ravel()
    gt = pd.read_csv(R / "final_pair_level_comparison_crand.csv"); gt = gt[gt.truth == "protein-defined truth"]
    rows = [dict(method=r.method, beta=r.beta, AP=r.AP) for r in gt.itertuples()]
    for b in sm["betas"]:
        dirs = seeds(f"L_crand_b{b}")
        if not all(len(list(d.glob("fold*_mo.pt"))) == 5 for d in dirs):
            print("missing local", b); continue
        rows.append(dict(method="local model (3-seed)", beta=b, AP=average_precision_score(y, pair_scores(dirs, "local")[:, keep].ravel())))
    out = pd.DataFrame(rows); out.to_csv(R / "pair_level_local_vs_graticcs_crand.csv", index=False); print(out.to_string(index=False))


def hybrid():
    rows = []
    for label, mf, Lk, Rk, pre, evid in [
            ("protein-defined", "spike_crand_meta.json", "L_mo", "R_mo", "G_crand", {"co-expression": base["co-expression_(L_x_R)"]}),
            ("RNA-defined", "spike_crand_rna_meta.json", "L_rna", "R_rna", "G_crand_rna",
             {"RNA co-expression": base["co-expression_(RNA_only)"], "COMMOT": base["COMMOT"]})]:
        truth, sm = truth_for(mf, Lk, Rk); y = truth[:, keep].ravel()
        for b in sm["betas"]:
            dirs = [d for d in seeds(f"{pre}_b{b}") if len(list(d.glob("fold*_mo.json"))) == 5]
            if not dirs:
                continue
            p = (1 - channel_q(dirs))[real]
            for en, E in evid.items():
                rows.append(dict(truth=label, beta=b, method=f"{en} alone", AP=average_precision_score(y, E[:, keep].ravel())))
                rows.append(dict(truth=label, beta=b, method=f"GraTICCS (1 - q) x {en}", AP=average_precision_score(y, (p[None, :] * E)[:, keep].ravel())))
    out = pd.DataFrame(rows); out.to_csv(R / "hybrid_pair_level_crand.csv", index=False); print(out.to_string(index=False))


def selective(test):
    exp = {"sender_type": "ctype", "contact_extent": "carea"}[test]
    sm = json.load(open(D / f"spike_{exp}_meta.json")); TYPES = meta["types"]
    names = ch.name[real].to_numpy()
    co = (np.maximum(z["L_mo"][csrc][:, real], 0) * np.maximum(z["R_mo"][cdst][:, real], 0)) > 0
    truth = np.zeros_like(co); cols = []
    for p in sm["plant"]:
        k = int(np.flatnonzero(names == p[0])[0]); cols.append(k)
        truth[:, k] = co[:, k] & ((ctype[csrc] == TYPES.index(p[3])) if exp == "ctype" else (con[ci] >= sm["contact_median"]))
    cols = np.array(cols); y = truth[:, keep].ravel(); yc = truth[:, cols]; mc = co[:, cols]
    print(f"events {y.size}, positives {int(y.sum())}; chance AP among co-expressing planted events {yc[mc].mean():.3f}")

    def both(S, method, b):
        return dict(method=method, beta=b, AP=average_precision_score(y, S[:, keep].ravel()),
                    AP_within_coexpressing=average_precision_score(yc[mc], S[:, cols][mc]))

    rows = [both(base["co-expression_(L_x_R)"], "co-expression (L x R)", "any")]
    for b in sm["betas"]:
        for method, pre, w in (("GraTICCS", "G", "attention"), ("local model", "L", "local"),
                               ("local + sender type", "T", "sender_type"), ("weighted by contact amount", "W", "contact_extent")):
            dirs = seeds(f"{pre}_{exp}_b{b}")
            if not all(len(list(d.glob("fold*_mo.json"))) == 5 for d in dirs):
                continue
            rows.append(both(pair_scores(dirs, w), f"{method} (3-seed)", b))
    out = pd.DataFrame(rows); out.to_csv(R / f"{exp}_pair_level.csv", index=False); print(out.to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1]
    {"contact": contact, "local": local, "hybrid": hybrid}.get(a, lambda: selective(a))()
