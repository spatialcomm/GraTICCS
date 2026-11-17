"""Graph transformer against the matched local model, imaging field by imaging field.

For each held-out field the full-model squared error is summed over the three seeds; fields where GraTICCS is lower are
counted and the paired differences tested with a two-sided Wilcoxon signed-rank test.
Writes results/s19_per_fov_crand.csv and results/s19_summary_crand.csv.
"""
import json

import pandas as pd
from scipy.stats import wilcoxon

from graticcs.paths import RESULTS

R = RESULTS
SETTINGS = {"planted, beta = 0.05": ("G_crand_b0.05", "L_crand_b0.05"), "planted, beta = 0.1": ("G_crand_b0.1", "L_crand_b0.1"),
            "planted, beta = 0.2": ("G_crand_b0.2", "L_crand_b0.2"), "planted, beta = 0.4": ("G_crand_b0.4", "L_crand_b0.4"),
            "binary gate, beta = 0.2": ("G_crand_mis_gate_b0.2", "L_crand_mis_gate_b0.2"), "original tissue": ("G_real", "L_real")}


def per_fov(prefix):
    acc = {}
    for s in range(3):
        fs = sorted((R / f"{prefix}_s{s}").glob("fold*_mo.json"))
        if len(fs) < 5:
            return None
        for f in fs:
            for fov, sse0, sse1, _ in json.load(open(f))["per_fov"]:
                a = acc.setdefault(int(fov), [0.0, 0.0]); a[0] += sse0; a[1] += sse1
    return pd.DataFrame([(k, v[0], v[1]) for k, v in acc.items()], columns=["fov", "sse_background", "sse_full"]).set_index("fov")


rows, summary = [], []
for name, (gp, lp) in SETTINGS.items():
    g, l = per_fov(gp), per_fov(lp)
    if g is None or l is None:
        print("incomplete", name); continue
    d = g.join(l, lsuffix="_gt", rsuffix="_local"); d["setting"] = name
    d["rel_diff"] = (d.sse_full_gt - d.sse_full_local) / d.sse_full_local       # negative: GraTICCS lower
    rows.append(d.reset_index()); w = wilcoxon(d.sse_full_gt, d.sse_full_local)
    summary.append(dict(setting=name, n_fov=len(d), fov_gt_better=int((d.sse_full_gt < d.sse_full_local).sum()),
                        error_ratio_gt_over_local=d.sse_full_gt.sum() / d.sse_full_local.sum(), median_rel_diff=d.rel_diff.median(),
                        wilcoxon_p=w.pvalue))
pd.concat(rows).to_csv(R / "s19_per_fov_crand.csv", index=False)
s = pd.DataFrame(summary); s.to_csv(R / "s19_summary_crand.csv", index=False)
print(s.to_string(index=False))
