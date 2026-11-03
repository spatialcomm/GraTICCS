"""GraTICCS on the same 30 channels and readouts as the HoloNet comparison, rank-only criterion."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from graticcs.paths import BREAST, RESULTS
D = BREAST; R = RESULTS
meta = json.load(open(D / "gt_channels.json")); sm = json.load(open(D / "spike_meta.json"))
names = [c["name"] for c in meta["channels"]]; real = [k for k, c in enumerate(meta["channels"]) if not c["decoy"]]
planted = [p for p in sm["plant"] if p[0] not in {"HLA-A->CD8A", "HLA-DRA->CD4"}]
rng = np.random.default_rng(0)
pc = [names.index(p[0]) for p in planted]
others = [k for k in real if k not in pc]; chosen = pc + list(rng.choice(others, 30 - len(pc), replace=False))
hol = pd.read_csv(R / "holonet_spike_nreg5.csv")


def importance(tag):
    fo = [json.load(open(p)) for p in sorted((R / tag).glob("fold*_mo.json"))]
    assert len(fo) == 5, tag
    s0 = np.sum([f["sse_nuisance"] for f in fo], 0)
    return np.sum([f["dsse_knockout"] for f in fo], 0) / s0, fo[0]["targets"]


rows = []
for b, pre in [(0.0, "G_real_s"), (0.2, "G_spike_b0.2_s"), (0.4, "G_spike_b0.4_s")]:
    imps = [importance(f"{pre}{s}") for s in range(3)]; tn = imps[0][1]
    imp = np.mean([i for i, _ in imps], 0)
    for chn, tgt, sgn in planted:
        t = tn.index(tgt); sc = imp[chosen, t]; me = sc[chosen.index(names.index(chn))]
        rank = int((sc > me).sum()) + 1
        h = hol[(hol.beta == b) & (hol.channel == chn)].iloc[0]
        rows.append(dict(beta=b, channel=chn, target=tgt, graticcs_rank=rank, holonet_rank=int(h.holonet_rank),
                         graticcs_top3=rank <= 3, holonet_top3=bool(h.recovered)))
df = pd.DataFrame(rows); print(df.to_string(index=False))
print(df.groupby("beta")[["graticcs_top3", "holonet_top3"]].sum())
df.to_csv(R / "graticcs_on_holonet_set.csv", index=False)
