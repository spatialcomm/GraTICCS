"""Cell-pair ranking for the planted secreted signals on the 12 imaging fields of the contact benchmark.

Events: every secreted edge (30 um) whose receiver lies in those fields x every real secreted channel.
distance:    positive = planted channel, ligand > 0 at the sender, receptor > 0 at the receiver, and distance <= the median
             distance of such co-expressing planted events (the exp(-d / 15 um) kernel puts most weight there)
sender_type: positive = planted channel, co-expressing, sender of the chosen type
Scores as in analysis/pair_level.py, with the neighbour weight from the saved attention (GraTICCS), 1/degree (local),
2 sigmoid(T[sender type]) / degree (sender-type rule) or the normalised distance kernel (distance rule).

python -m analysis.secreted_pairs distance | sender_type
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
EXP = {"distance": "sec_rand", "sender_type": "sec_ctype"}[sys.argv[1]]; BETA = "0.2"
z = np.load(D / "gt_inputs.npz"); fov, ctype = z["fov"], z["ctype"]
types = json.load(open(D / "gt_channels.json"))["types"]
S = np.load(D / "gt_sec_inputs.npz"); meta = json.load(open(D / "gt_sec_channels.json")); ch = pd.DataFrame(meta["channels"])
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
real = np.flatnonzero(~ch.decoy.to_numpy())
B12 = np.load(R / "baseline_scores_12fov.npz"); f12 = np.unique(fov[z["dst"][B12["ci"]]])
ei = np.flatnonzero(np.isin(fov[S["dst"]], f12)); src, dst, dist = S["src"][ei], S["dst"][ei], S["dist"][ei]
pos = -np.ones(len(S["src"]), np.int64); pos[ei] = np.arange(len(ei))
Lr = S["L"][:, real].astype(np.float32); Rr = S["R"][:, real].astype(np.float32)
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10)))
sm = json.load(open(D / f"spike_{EXP}_meta.json")); plant = sm["plant"]; names = ch.name[real].to_numpy()
deg = np.bincount(S["dst"], minlength=len(fov)).astype(np.float32)
ker = np.exp(-S["dist"] / 15.0); ksum = np.bincount(S["dst"], weights=ker, minlength=len(fov))

Y = np.zeros((len(ei), len(real)), bool); CO = np.zeros_like(Y)
for p in plant:
    c = int(np.flatnonzero(names == p[0])[0])
    co = (np.maximum(Lr[src, c], 0) > 0) & (np.maximum(Rr[dst, c], 0) > 0); CO[:, c] = co
    Y[:, c] = co & (ctype[src] == types.index(p[3])) if EXP == "sec_ctype" else co
if EXP != "sec_ctype":
    med = np.median(np.broadcast_to(dist[:, None], Y.shape)[Y]); Y &= (dist <= med)[:, None]
y = Y.ravel(); co = CO.ravel()
print(f"{EXP}: events {y.size}, positives {int(y.sum())}; chance AP among co-expressing planted events {y[co].mean():.3f}")


def qvals(dirs):
    s0 = ds = 0
    for d in dirs:
        fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
        s0 = s0 + np.sum([f["sse_nuisance"] for f in fo], 0); ds = ds + np.sum([f["dsse_knockout"] for f in fo], 0)
    ib = (ds / s0).max(1); q = np.ones(len(ch))
    for st in ("protein", "rna"):
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q[rl] = multipletests(pv, method="fdr_bh")[1]
    return q, fo[0]["targets"]


def score(prefix, mode):
    dirs = [d for d in (R / f"{prefix}_s{s}" for s in range(3)) if len(list(d.glob("fold*_mo.json"))) == 5]
    if len(dirs) < 3:
        return None
    q, tn = qvals(dirs); M = target_mask({"L": S["L"][:1], "tnames": tn, "channels": meta["channels"]})
    s = sp_(Lr[src]) * sp_(Rr[dst]); Sc = np.zeros(s.shape, np.float32)
    for d in dirs:
        for k in range(5):
            st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
            G, V = st["G"].numpy(), st["V"].numpy(); Vn = np.linalg.norm((V * M)[real], axis=1)
            if mode == "attention":
                A = np.load(d / f"fold{k}_mo_attention.npz"); p_ = pos[A["edge"]]; m = p_ >= 0
                e = p_[m]; w = A["a"][m].astype(np.float32) @ torch.softmax(st["P"], 1).numpy()[real].T
            else:
                e = np.flatnonzero(z["fold"][dst] == k); g_e = ei[e]
                if mode == "sender_type":
                    Tw = st["T"].numpy()[:, real]
                    w = (1.0 / np.maximum(deg[S["dst"][g_e]], 1))[:, None] * 2 / (1 + np.exp(-Tw[ctype[src[e]]]))
                elif mode == "local":
                    w = (1.0 / np.maximum(deg[S["dst"][g_e]], 1))[:, None]
                else:
                    w = (ker[g_e] / np.maximum(ksum[S["dst"][g_e]], 1e-12))[:, None]
            gate = 1 / (1 + np.exp(-G[ctype[dst[e]]][:, real]))
            Sc[e] += w * s[e] * gate * Vn[None, :] / len(dirs)
    return ((1 - q[real])[None, :] * Sc).ravel()


rows = []
coexpr = (np.maximum(Lr[src], 0) * np.maximum(Rr[dst], 0)).ravel()
for name, sc in [("co-expression (L x R)", coexpr),
                 ("co-expression x distance kernel", (coexpr.reshape(Y.shape) * np.exp(-dist / 15.0)[:, None]).ravel())]:
    rows.append(dict(method=name, AP=average_precision_score(y, sc), AP_within_coexpressing=average_precision_score(y[co], sc[co])))
for name, prefix, mode in [("GraTICCS", f"G2_{EXP}_b{BETA}", "attention"), ("local model", f"L_{EXP}_b{BETA}", "local"),
                           ("local + sender type", f"T_{EXP}_b{BETA}", "sender_type"), ("local + distance kernel", f"W_{EXP}_b{BETA}", "distance")]:
    sc = score(prefix, mode)
    if sc is None:
        print("incomplete", name); continue
    rows.append(dict(method=f"{name} (3-seed)", AP=average_precision_score(y, sc), AP_within_coexpressing=average_precision_score(y[co], sc[co])))
df = pd.DataFrame(rows); print(df.to_string(index=False))
df.to_csv(R / f"sec_pair_{EXP}.csv", index=False)
