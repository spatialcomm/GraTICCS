"""HoloNet on the planted contact signals (five regions); recovered = top three by layer attention among 30 LR views."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import HoloNet as hn
from graticcs.paths import BREAST, RESULTS
D = BREAST; R = RESULTS
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json")); sm = json.load(open(D / "spike_meta.json"))
names = [c["name"] for c in meta["channels"]]; real = [k for k, c in enumerate(meta["channels"]) if not c["decoy"]]
T = sm["tnames"]; planted = [p for p in sm["plant"] if p[0] not in {"HLA-A->CD8A", "HLA-DRA->CD4"}]
rng = np.random.default_rng(0)
pc = [names.index(p[0]) for p in planted]
others = [k for k in real if k not in pc]; chosen = pc + list(rng.choice(others, 30 - len(pc), replace=False))
xy, fov, ctype = z["xy"], z["fov"], z["ctype"]
tcount = pd.Series(ctype == json.load(open(D / "gt_channels.json"))["types"].index("T")).groupby(fov).sum().sort_values(ascending=False)
import os
NREG = int(os.environ.get("NREG", "5")); regions = []
for f in tcount.index[:NREG]:
    regions.append(np.flatnonzero(fov == f)[:2500])
log("regions", NREG, "cells", [len(r) for r in regions])
def build(sel):
    P = xy[sel]; d = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
    K = np.where((d <= 30) & (d > 0), np.exp(-d / 15.0), 0).astype(np.float32)
    ce = np.stack([sp_(z["L_mo"][sel, c])[:, None] * K * sp_(z["R_mo"][sel, c])[None, :] for c in chosen])
    ce = torch.as_tensor(ce).permute(0, 2, 1).contiguous()
    X = torch.as_tensor(np.eye(int(ctype.max()) + 1, dtype=np.float32)[ctype[sel]])
    return X, hn.pr.adj_normalize(adj=ce, cell_type_tensor=X, only_between_cell_type=True)
sp_ = lambda a: np.log1p(np.exp(np.clip(a, -10, 10))).astype(np.float32)
lr_df = pd.DataFrame({"LR_Pair": [names[c] for c in chosen]})
BUILT = [build(sel) for sel in regions]; log("built CE tensors")
dev = "cuda" if torch.cuda.is_available() else "cpu"
rows = []
for b in [0.0, 0.2, 0.4]:
    Y = np.c_[z["Yp"], z["Yr"]] if b == 0 else np.load(D / f"spike_beta{b}.npz")["Y"]
    for chn, tgt, sgn in planted:
        scores = []
        for sel, (X, adj) in zip(regions, BUILT):
            y = Y[sel, T.index(tgt)]; y = (y - y.mean()) / (y.std() + 1e-9)
            models = hn.pr.mgc_repeat_training(X, adj, torch.as_tensor(y[:, None], dtype=torch.float32), repeat_num=10, max_epoch=300, device=dev, hide_repeat_tqdm=True)
            att = np.stack([np.abs(m.mgc.layer_attention.data.cpu().numpy()).ravel() for m in models])
            att = (att - att.min(1, keepdims=True)) / (att.max(1, keepdims=True) - att.min(1, keepdims=True) + 1e-12)
            scores.append(att.mean(0))
        score = np.mean(scores, 0); rank = int((score > score[chosen.index(names.index(chn))]).sum()) + 1
        rows.append(dict(beta=b, channel=chn, target=tgt, holonet_rank=rank, n_views=len(chosen), n_regions=len(regions), recovered=rank <= 3)); log(rows[-1])
df = pd.DataFrame(rows); print(df.groupby("beta").recovered.sum()); df.to_csv(R / f"holonet_spike_nreg{NREG}.csv", index=False)
