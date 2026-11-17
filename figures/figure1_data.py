"""Numbers for Figure 1E: channel evidence, decoy comparison and cell-pair scores for HLA-E->KLRK1 in the contact draw at
beta = 0.1 (three seeds). Writes results/fig1_walkthrough_crand.npz."""
import json

import numpy as np
import torch

from analysis.pair_level import ci, cdst, csrc, pair_scores, real
from analysis.summary import CH, NATURAL, evaluate
from graticcs.paths import BREAST, RESULTS

D, R = BREAST, RESULTS
LR1 = "HLA-E->KLRK1"
runs = [f"G_crand_b0.1_s{s}" for s in range(3)]; dirs = [R / r for r in runs]
fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]; T = fo[0]["targets"]
imp = np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)
plant = [x for x in json.load(open(D / "spike_crand_meta.json"))["plant"] if x[0] not in NATURAL]
o = evaluate("c", runs, D / "spike_crand_meta.json", exclude_natural=True)
chs = CH["c"]; c1 = int(np.flatnonzero(chs.name.to_numpy() == LR1)[0]); st = chs.stratum[c1]
t1 = T.index(dict((a, b) for a, b, _ in plant)[LR1])
dec = chs.index[chs.decoy & (chs.stratum == st)].to_numpy(); own = chs.index[(chs.of == LR1) & chs.decoy].to_numpy()
V = np.mean([torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]["V"].numpy() for d in dirs for k in range(5)], 0)
pl_idx = [int(np.flatnonzero(chs.name.to_numpy() == a)[0]) for a, _, _ in plant]
eff = np.sign(V[pl_idx]) * imp[pl_idx]                      # planted channels x readouts, signed
realn = list(o["q"]); qv = np.array([o["q"][c] for c in realn])
sc = pair_scores(dirs)[:, list(real).index(c1)]
ib = imp.max(1)
np.savez(R / "fig1_walkthrough_crand.npz", LR1=LR1, target=T[t1], T=np.array(T), d_lr1=ib[c1], d_own=ib[own], d_dec=ib[dec],
         p=o["p"][LR1], q=o["q"][LR1], n_dec=len(dec), n_real=int((~chs.decoy).sum()),
         plant=np.array([a for a, _, _ in plant]), plant_t=np.array([b for _, b, _ in plant]), plant_s=np.array([s for _, _, s in plant]), eff=eff,
         real_names=np.array(realn), real_q=qv, sc=sc, ci=ci, csrc=csrc, cdst=cdst)
print(LR1, T[t1], "p", o["p"][LR1], "q", o["q"][LR1])
