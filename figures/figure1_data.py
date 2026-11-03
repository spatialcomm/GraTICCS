"""Numbers behind the Figure 1 walk-through (TNFSF13B -> CD40) and the per-pair recovery table."""
import json, sys, os
from pathlib import Path
import numpy as np, pandas as pd, torch
from analysis import summary as g22
from analysis.pair_level import D, R, ch, ci, csrc, cdst, ens, real, z
LR1 = "TNFSF13B->CD40"
runs = [f"G_spike_b0.1_s{s}" for s in range(3)]; dirs = [R / r for r in runs]
fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]; T = fo[0]["targets"]
imp = np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)
sm = json.load(open(D / "spike_meta.json")); plant = [x for x in sm["plant"] if x[0] not in g22.NATURAL]
o = g22.evaluate("c", runs, D / "spike_meta.json", exclude_natural=True)
chs = g22.CH["c"]; c1 = int(np.flatnonzero(chs.name.to_numpy() == LR1)[0]); st = chs.stratum[c1]
t1 = T.index(dict((a, b) for a, b, _ in plant)[LR1])
dec = chs.index[chs.decoy & (chs.stratum == st)].to_numpy(); own = chs.index[(chs.of == LR1) & chs.decoy].to_numpy()
V = np.mean([torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]["V"].numpy() for d in dirs for k in range(5)], 0)
pl_idx = [int(np.flatnonzero(chs.name.to_numpy() == a)[0]) for a, _, _ in plant]
eff = np.sign(V[pl_idx]) * imp[pl_idx]
realn = [c for c in o["q"]]; qv = np.array([o["q"][c] for c in realn])
S = ens(dirs); sc = S[:, list(real).index(c1)]
ib = imp.max(1)
np.savez(R / "fig1_walkthrough.npz", LR1=LR1, target=T[t1], T=np.array(T), d_lr1=ib[c1], d_own=ib[own], d_dec=ib[dec],
         p=o["p"][LR1], q=o["q"][LR1], n_dec=len(dec), n_real=int((~chs.decoy).sum()),
         plant=np.array([a for a, _, _ in plant]), plant_t=np.array([b for _, b, _ in plant]), plant_s=np.array([s for _, _, s in plant]), eff=eff,
         real_names=np.array(realn), real_q=qv, sc=sc, ci=ci, csrc=csrc, cdst=cdst)
print(LR1, T[t1], "delta", imp[c1, t1], "own", imp[own, t1].round(6), "p", o["p"][LR1], "q", o["q"][LR1])

gs = json.load(open(R / "gt_summary.json")); rows = []
for grp, key, bs in [("breast · contact", "contact_spike_b{}", (0.05, 0.1, 0.2, 0.4)), ("breast · secreted", "secreted_spike_b{}", (0.05, 0.1, 0.2, 0.4)),
                     ("kidney · contact", "xenium_spike_b{}", (0.1, 0.4))]:
    for b in bs:
        for d in gs[key.format(b)]["detail"]:
            rows.append(dict(group=grp, beta=b, pair=d[0], response=d[1], p=d[3], rank=d[4], found=d[5], fp=gs[key.format(b)]["fp"]))
pd.DataFrame(rows).to_csv(R / "fig1_recovery.csv", index=False); print("recovery rows", len(rows))
