"""Why the local model has the higher AP over all secreted cell pairs (Supplementary Note 4).

The pair score (1 - q_c) x w_e,c x pair term x sigmoid(G) ||V M|| is split into its parts, which are exchanged between
GraTICCS and the local model (same events, truth and folds as analysis/secreted_pairs.py, distance test, beta 0.2):
  w: GraTICCS attention mixed per channel, or 1/degree;  f: sigmoid(G[receiver type]) x ||V M||;  q: channel evidence.
The pair term is softplus(L) softplus(R); --coexpression replaces it by max(L, 0) max(R, 0), a diagnostic only.

python -m analysis.overall_ap decompose [--coexpression]     sec_pair_decompose_sec_rand[_relu].csv, sec_pair_attnmass_*
python -m analysis.overall_ap crosschannel [--coexpression]  sec_pair_crosschannel_sec_rand.csv
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score as aps
from statsmodels.stats.multitest import multipletests

from graticcs.data import target_mask
from graticcs.paths import BREAST, RESULTS

D = BREAST; R = RESULTS
EXP, BETA = "sec_rand", "0.2"
MODE = sys.argv[1]; RELU = "--coexpression" in sys.argv
z = np.load(D / "gt_inputs.npz"); fov, ctype = z["fov"], z["ctype"]
S = np.load(D / "gt_sec_inputs.npz"); meta = json.load(open(D / "gt_sec_channels.json")); ch = pd.DataFrame(meta["channels"])
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
real = np.flatnonzero(~ch.decoy.to_numpy())
B12 = np.load(R / "baseline_scores_12fov.npz"); f12 = np.unique(fov[z["dst"][B12["ci"]]])
ei = np.flatnonzero(np.isin(fov[S["dst"]], f12)); src, dst, dist = S["src"][ei], S["dst"][ei], S["dist"][ei]
pos = -np.ones(len(S["src"]), np.int64); pos[ei] = np.arange(len(ei))
Lr = S["L"][:, real].astype(np.float32); Rr = S["R"][:, real].astype(np.float32)
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
plant = json.load(open(D / f"spike_{EXP}_meta.json"))["plant"]; names = ch.name[real].to_numpy()
deg = np.bincount(S["dst"], minlength=len(fov)).astype(np.float32)

Y = np.zeros((len(ei), len(real)), bool); CO = np.zeros_like(Y); pc = []
for p in plant:
    c = int(np.flatnonzero(names == p[0])[0]); pc.append(c)
    co = (np.maximum(Lr[src, c], 0) > 0) & (np.maximum(Rr[dst, c], 0) > 0); CO[:, c] = co; Y[:, c] = co
med = np.median(np.broadcast_to(dist[:, None], Y.shape)[Y]); Y &= (dist <= med)[:, None]
y = Y.ravel(); co = CO.ravel()


def qvals(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    ib = (ds / s0).max(1); q = np.ones(len(ch))
    for st in ("protein", "rna"):
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    return q[real], fo[0]["targets"]


G = [R / f"G2_{EXP}_b{BETA}_s{s}" for s in range(3)]; L = [R / f"L_{EXP}_b{BETA}_s{s}" for s in range(3)]
qG, tn = qvals(G); qL, _ = qvals(L); M = target_mask({"L": S["L"][:1], "tnames": tn, "channels": meta["channels"]})
s_ = (np.maximum(Lr[src], 0) * np.maximum(Rr[dst], 0)) if RELU else sp_(Lr[src]) * sp_(Rr[dst])
SUF = "_relu" if RELU else ""
acc = {k: np.zeros(s_.shape, np.float32) for k in ["aG_fG", "aG_fL", "uni_fG", "uni_fL", "aG_only"]}
amass = []
for sd in range(3):
    for k in range(5):
        stG = torch.load(G[sd] / f"fold{k}_mo.pt", map_location="cpu")["lr"]; stL = torch.load(L[sd] / f"fold{k}_mo.pt", map_location="cpu")["lr"]
        A = np.load(G[sd] / f"fold{k}_mo_attention.npz"); p_ = pos[A["edge"]]; m = p_ >= 0; e = p_[m]
        a = A["a"][m].astype(np.float32) @ torch.softmax(stG["P"], 1).numpy()[real].T
        u = (1.0 / np.maximum(deg[dst[e]], 1))[:, None]
        fac = {}
        for nm, st in (("G", stG), ("L", stL)):
            Vn = np.linalg.norm((st["V"].numpy() * M)[real], axis=1); g = 1 / (1 + np.exp(-st["G"].numpy()[ctype[dst[e]]][:, real]))
            fac[nm] = g * Vn[None, :]
        se = s_[e]
        acc["aG_fG"][e] += a * se * fac["G"] / 3; acc["aG_fL"][e] += a * se * fac["L"] / 3
        acc["uni_fG"][e] += u * se * fac["G"] / 3; acc["uni_fL"][e] += u * se * fac["L"] / 3
        acc["aG_only"][e] += a * se / 3
        for c in pc:
            amass.append(dict(seed=sd, fold=k, channel=names[c], attn_share_coexpr=float(a[CO[e, c], c].sum() / a[:, c].sum()),
                              uniform_share_coexpr=float(u[CO[e, c], 0].sum() / u[:, 0].sum()),
                              attn_share_true=float(a[Y[e, c], c].sum() / a[:, c].sum()), uniform_share_true=float(u[Y[e, c], 0].sum() / u[:, 0].sum())))
QG, QL = (1 - qG)[None, :], (1 - qL)[None, :]
pd.set_option("display.width", 250)


def decompose():
    variants = {
        "GraTICCS full: qG x attn x fG": QG * acc["aG_fG"], "local full: qL x 1/deg x fL": QL * acc["uni_fL"],
        "attn x fG (no q)": acc["aG_fG"], "1/deg x fL (no q)": acc["uni_fL"],
        "qG x 1/deg x fG (GraTICCS factors, uniform weights)": QG * acc["uni_fG"],
        "qL x attn x fL (local factors, GraTICCS attention)": QL * acc["aG_fL"],
        "qL x attn x fG": QL * acc["aG_fG"], "qG x attn x fL": QG * acc["aG_fL"],
        "attn only (x LR)": acc["aG_only"], "co-expression softplus only": s_,
        "qG x co-expression": QG * s_, "qL x co-expression": QL * s_}
    rows = []
    for nm, sc in variants.items():
        v = sc.ravel()
        rows.append(dict(score=nm, AP=aps(y, v), AP_within_coexpressing=aps(y[co], v[co]),
                         mean_AP_per_planted_channel=np.mean([aps(Y[:, c], sc[:, c]) for c in pc])))
    d = pd.DataFrame(rows); print(d.round(4).to_string(index=False))
    d.to_csv(R / f"sec_pair_decompose_{EXP}{SUF}.csv", index=False)
    am = pd.DataFrame(amass).groupby("channel").mean(numeric_only=True).drop(columns=["seed", "fold"])
    print("attention share on co-expressing edges, mean over planted channels", am.mean().round(3).to_dict())
    am.to_csv(R / f"sec_pair_attnmass_{EXP}{SUF}.csv")


def crosschannel():
    SC = {"GraTICCS": QG * acc["aG_fG"], "local": QL * acc["uni_fL"]}
    K = int(Y.sum()); isP = np.zeros(len(names), bool); isP[pc] = True
    rows = []
    for nm, sc in SC.items():
        v = sc.ravel(); top = np.argsort(-v)[:K]; tc = top % sc.shape[1]; te = top // sc.shape[1]
        r = dict(model=nm, AP=aps(y, v))
        r["AP_planted_channels_only"] = aps(Y[:, pc].ravel(), sc[:, pc].ravel())
        p99 = np.array([np.quantile(sc[sc[:, c] > 0, c], 0.99) if (sc[:, c] > 0).any() else 1.0 for c in range(sc.shape[1])])
        r["AP_after_per_channel_scale"] = aps(y, (sc / p99[None, :]).ravel())
        r["AP_planted_only_after_scale"] = aps(Y[:, pc].ravel(), (sc[:, pc] / p99[None, pc]).ravel())
        r["top_K_from_nonplanted_channels"] = float((~isP[tc]).mean())
        r["top_K_true"] = float(Y[te, tc].mean())
        r["top_K_planted_but_false"] = float((isP[tc] & ~Y[te, tc]).mean())
        r["top_K_planted_false_coexpr"] = float((isP[tc] & ~Y[te, tc] & CO[te, tc]).mean())
        rows.append(r)
        print(nm, "most frequent non-planted channels among the top K events:",
              pd.Series(names[tc[~isP[tc]]]).value_counts().head(8).to_dict())
    d = pd.DataFrame(rows); print(d.round(4).to_string(index=False))
    d.to_csv(R / f"sec_pair_crosschannel_{EXP}.csv", index=False)


{"decompose": decompose, "crosschannel": crosschannel}[MODE]()
