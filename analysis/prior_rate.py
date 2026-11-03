"""Discovery rate among real channels with a prior program versus decoys.

python -m analysis.prior_rate <channels.json> <mask.npy> <run>[,<run>...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from graticcs.paths import RESULTS
R = RESULTS
ch = pd.DataFrame(json.load(open(sys.argv[1]))["channels"]); M = np.load(sys.argv[2]).any(1)
if "decoy_modality" not in ch: ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
fo = [json.load(open(p)) for d in sys.argv[3].split(",") for p in sorted((R / d).glob("fold*_mo.json"))]
s0 = np.sum([f["sse_nuisance"] for f in fo], 0); ib = (np.sum([f["dsse_knockout"] for f in fo], 0) / s0).max(1)
ps, qs = [], []
for st in ["protein", "rna"]:
    dec = ch.index[ch.decoy & (ch.stratum == st) & M]; rl = ch.index[~ch.decoy & (ch.stratum == st) & M]
    if not len(rl): continue
    pv = np.array([(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]); ps += list(pv); qs += list(multipletests(pv, method="fdr_bh")[1])
ps = np.array(ps); k = int((ps <= 0.05).sum())
print(f"real channels with program {len(ps)}; p<=0.05: {k} ({k/len(ps):.1%}); binomial p vs 5% = {binomtest(k, len(ps), 0.05, alternative='greater').pvalue:.2f}; q<=0.1: {int((np.array(qs) <= 0.1).sum())}")
