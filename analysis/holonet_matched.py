"""GraTICCS on the same 30 channels and readouts as the HoloNet comparison (analysis/holonet.py).

Each planted channel is ranked by knockout importance (three-seed mean of held-out folds) on its planted readout; top
three counts as recovered, without decoys, as for HoloNet. Writes results/graticcs_on_holonet_crand.csv.
"""
import json

import numpy as np
import pandas as pd

from graticcs.paths import BREAST, RESULTS

D = BREAST; R = RESULTS
meta = json.load(open(D / "gt_channels.json")); sm = json.load(open(D / "spike_crand_meta.json"))
names = [c["name"] for c in meta["channels"]]; real = [k for k, c in enumerate(meta["channels"]) if not c["decoy"]]
planted = [p for p in sm["plant"] if p[0] not in {"HLA-A->CD8A", "HLA-DRA->CD4"}]
rng = np.random.default_rng(0)                      # same channel draw as analysis/holonet.py
pc = [names.index(p[0]) for p in planted]
others = [k for k in real if k not in pc]; chosen = pc + list(rng.choice(others, 30 - len(pc), replace=False))
hol = pd.read_csv(R / "holonet_crand_nreg5.csv")


def importance(tag):
    fo = [json.load(open(p)) for p in sorted((R / tag).glob("fold*_mo.json"))]
    assert len(fo) == 5, tag
    return np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0), fo[0]["targets"]


rows = []
for b, pre in [(0.0, "G_real_s"), (0.2, "G_crand_b0.2_s"), (0.4, "G_crand_b0.4_s")]:
    imps = [importance(f"{pre}{s}") for s in range(3)]; tn = imps[0][1]
    imp = np.mean([i for i, _ in imps], 0)
    for chn, tgt, sgn in planted:
        t = tn.index(tgt); me = imp[names.index(chn), t]
        sc_all = imp[chosen, t]; sc_ex = imp[[k for k in chosen if k not in pc or k == names.index(chn)], t]
        rank_s = int((sc_all > me).sum()) + 1; rank = int((sc_ex > me).sum()) + 1
        h = hol[(hol.beta == b) & (hol.channel == chn)].iloc[0]
        rows.append(dict(beta=b, channel=chn, target=tgt, graticcs_rank=rank, graticcs_rank_strict=rank_s,
                         holonet_rank=int(h.holonet_rank_excl), holonet_rank_strict=int(h.holonet_rank),
                         graticcs_top3=rank <= 3, holonet_top3=bool(h.recovered),
                         graticcs_top3_strict=rank_s <= 3, holonet_top3_strict=bool(h.recovered_strict)))
df = pd.DataFrame(rows); df.to_csv(R / "graticcs_on_holonet_crand.csv", index=False)
print(df.groupby("beta")[["graticcs_top3", "holonet_top3", "graticcs_top3_strict", "holonet_top3_strict"]].sum())
