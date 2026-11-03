"""Seed-ensemble decoy calibration of one analysis.

python calibrate.py <channels.json> <readout_names.json|-> <run>[,<run>...] [spike_meta.json]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from graticcs.paths import RESULTS
R = RESULTS
ch = pd.DataFrame(json.load(open(sys.argv[1]))["channels"])
if "decoy_modality" not in ch: ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
dirs = [R / d for d in sys.argv[3].split(",")]
fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]
nf = {d.name: len(list(d.glob("fold*_mo.json"))) for d in dirs}; print("folds per run:", nf)
T = fo[0]["targets"] if sys.argv[2] == "-" else json.load(open(sys.argv[2]))["readouts"]
s0 = np.sum([f["sse_nuisance"] for f in fo], 0); s1 = np.sum([f["sse_full"] for f in fo], 0); imp = np.sum([f["dsse_knockout"] for f in fo], 0) / s0
print(f"ensemble LR-branch held-out gain {1 - s1.sum() / s0.sum():.3%}")
ib, bt = imp.max(1), imp.argmax(1); rows = []
for st in ["protein", "rna"]:
    dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
    if not len(rl) or not len(dec): continue
    pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); q = multipletests(pv, method="fdr_bh")[1]
    rows += [dict(channel=ch.name[c], stratum=st, best_readout=T[bt[c]], imp=ib[c], p=p, q=qq) for c, p, qq in zip(rl, pv, q)]
tab = pd.DataFrame(rows).sort_values("p"); print(tab.head(15).to_string(index=False))
print("real channels with q<=0.1:", list(tab.channel[tab.q <= 0.1]))
if len(sys.argv) > 4:
    sm = json.load(open(sys.argv[4])); rec = 0
    for chn, tgt, sgn in sm["plant"]:
        c = ch.index[ch.name == chn][0]; t = T.index(tgt); st = ch.stratum[c]
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        p = (1 + (imp[dec, t] >= imp[c, t]).sum()) / (1 + len(dec)); rk = int((imp[rl, t] > imp[c, t]).sum()) + 1
        ok = p <= 0.05 and rk <= 3; rec += ok; print(f"  planted {chn} -> {tgt}: p={p:.4f} rank={rk} {'RECOVERED' if ok else ''}")
    print(f"recovered {rec}/{len(sm['plant'])}")
