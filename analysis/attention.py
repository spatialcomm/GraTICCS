"""How far learned attention departs from equal weights over contact neighbours, how reproducible it is across seeds, and
what it prefers. Uses the saved head-averaged attention (fold*_mo_attention.npz) for receivers with at least two neighbours.

Per seed: normalised entropy H / log(deg) (1 = equal weights), effective share exp(H) / deg, and R^2 of log(a * deg) on
sender type, receiver type, log contact length and distance. Seed agreement: correlation of log(a * deg) across seeds.
Group preferences: mean a * deg (1 = equal share) by sender type and by quartile of distance and contact length.

python -m analysis.attention G_real G_crand_b0.2 G_ctype_b0.2 G_carea_b0.2
"""
import sys

import numpy as np
import pandas as pd

from graticcs.paths import BREAST, RESULTS

R = RESULTS
z = np.load(BREAST / "gt_inputs.npz"); src, dst, ct = z["src"], z["dst"], z["ctype"]
con, dist = z["contact"], z["dist"]; nt = int(ct.max()) + 1


def load(run):
    E, A = [], []
    for k in range(5):
        f = np.load(R / run / f"fold{k}_mo_attention.npz"); E.append(f["edge"]); A.append(f["a"].astype(np.float64).mean(1))
    e = np.concatenate(E); a = np.concatenate(A); o = np.argsort(e)
    return e[o], a[o]


def r2(X, y):
    X = np.c_[np.ones(len(y)), X]; b = np.linalg.lstsq(X, y, rcond=None)[0]
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


rows, groups = [], []
for pre in sys.argv[1:]:
    runs = [f"{pre}_s{s}" for s in range(3) if (R / f"{pre}_s{s}" / "fold4_mo_attention.npz").exists()]
    data = [load(r) for r in runs]; e = data[0][0]
    assert all(np.array_equal(d[0], e) for d in data), "edge sets differ across seeds"
    d_ = dst[e]; ndeg = np.bincount(d_, minlength=len(ct)); deg = ndeg[d_]; m = deg >= 2
    for run, (_, a) in zip(runs, data):
        ent = np.bincount(d_, -a * np.log(a + 1e-12), len(ct)); rec = np.unique(d_[m])
        la = np.log(a * deg + 1e-12)[m]; Xs = np.eye(nt)[ct[src[e]]][m]; Xr = np.eye(nt)[ct[d_]][m]
        Xc = np.log1p(con[e])[m, None]; Xd = dist[e][m, None]
        rows.append(dict(run=run, receivers=len(rec), norm_entropy_median=np.median(ent[rec] / np.log(ndeg[rec])),
                         eff_share_median=np.median(np.exp(ent[rec]) / ndeg[rec]),
                         max_over_equal_median=np.median(pd.Series(a * deg).groupby(d_).max().loc[rec]),
                         R2_sender_type=r2(Xs, la), R2_receiver_type=r2(Xr, la), R2_contact=r2(Xc, la), R2_distance=r2(Xd, la),
                         R2_all=r2(np.c_[Xs, Xr, Xc, Xd], la)))
        r = pd.Series((a * deg)[m])
        for name, key in (("sender type", ct[src[e]][m]), ("distance quartile", pd.qcut(dist[e][m], 4, labels=False)),
                          ("contact quartile", pd.qcut(con[e][m], 4, labels=False, duplicates="drop"))):
            for k, v in r.groupby(key).mean().items():
                groups.append(dict(run=run, grouping=name, group=int(k), relative_to_equal=v))
    la_s = [np.log(a * deg + 1e-12)[m] for _, a in data]
    cc = [np.corrcoef(la_s[i], la_s[j])[0, 1] for i in range(len(la_s)) for j in range(i + 1, len(la_s))]
    rows.append(dict(run=f"{pre} seed agreement", seed_corr_mean=np.mean(cc) if cc else np.nan))
out = pd.DataFrame(rows); tag = "_".join(sys.argv[1:])
out.to_csv(R / f"attention_uniformity_{tag}.csv", index=False)
pd.DataFrame(groups).to_csv(R / f"attention_groups_{tag}.csv", index=False)
print(out.round(3).to_string(index=False))
