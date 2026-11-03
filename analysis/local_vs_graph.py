"""Graph transformer versus matched local model per held-out FOV (Supplementary Table S19)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from graticcs.paths import RESULTS
R = RESULTS
SET = {"standard b=0.05": "std_b0.05", "standard b=0.1": "std_b0.1", "binary gate b=0.2": "gate_b0.2", "unspiked": "real"}


def per_fov(prefix):
    acc = {}
    for s in range(3):
        for f in sorted((R / f"{prefix}_s{s}").glob("fold*_mo.json")):
            for fov, sse0, sse1, *_ in json.load(open(f))["per_fov"]:
                a = acc.setdefault(int(fov), [0.0, 0.0]); a[0] += sse0; a[1] += sse1
    return pd.DataFrame([(k, v[0], v[1]) for k, v in acc.items()], columns=["fov", "sse_background", "sse_full"]).set_index("fov")


rows, summ = [], []
for name, tag in SET.items():
    g, l = per_fov(f"arch_graticcs_{tag}"), per_fov(f"arch_graticcs_local_{tag}")
    d = g.join(l, lsuffix="_gt", rsuffix="_local"); d["setting"] = name
    d["rel_diff"] = (d.sse_full_gt - d.sse_full_local) / d.sse_full_local
    rows.append(d.reset_index())
    w = wilcoxon(d.sse_full_gt, d.sse_full_local)
    summ.append(dict(setting=name, n_fov=len(d), fov_gt_better=int((d.sse_full_gt < d.sse_full_local).sum()),
                     error_ratio_gt_over_local=d.sse_full_gt.sum() / d.sse_full_local.sum(), median_rel_diff=d.rel_diff.median(),
                     mean_rel_diff=d.rel_diff.mean(), wilcoxon_W=w.statistic, wilcoxon_p=w.pvalue))
pd.concat(rows).to_csv(R / "s19_per_fov.csv", index=False)
s = pd.DataFrame(summ); s.to_csv(R / "s19_summary.csv", index=False); print(s.to_string(index=False))
