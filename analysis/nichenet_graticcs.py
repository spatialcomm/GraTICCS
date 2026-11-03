"""GraTICCS on the NicheNet-consistent planted data, judged per ligand on its own NicheNet program."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from graticcs.paths import BREAST, RESULTS
D = BREAST; ND = D / "nichenet_spike"; R = RESULTS
meta = json.load(open(ND / "meta.json")); T = meta["readouts"]; cand = meta["candidates"]
ch = pd.DataFrame(json.load(open(D / "gt_sec_channels.json"))["channels"])
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
rows = []
for b in meta["betas"]:
    d = R / f"{os.environ.get('NNPREFIX', 'G2_nn_spike_b')}{b}"; fo = [json.load(open(p)) for p in sorted(d.glob("fold*_mo.json"))]
    if len(fo) < 5: print("beta", b, "incomplete:", len(fo)); continue
    s0 = np.sum([f["sse_nuisance"] for f in fo], 0); imp = np.sum([f["dsse_knockout"] for f in fo], 0) / s0
    for L in meta["plant"]:
        t = T.index(f"NN_{L}")
        lig_score = {g: imp[ch.index[~ch.decoy & (ch.ligand == g)], t].max() for g in cand if (~ch.decoy & (ch.ligand == g)).any()}
        cL = ch.index[~ch.decoy & (ch.ligand == L)]; st = ch.stratum[cL[np.argmax(imp[cL, t])]]
        dec = ch.index[ch.decoy & (ch.stratum == st)]
        p = (1 + (imp[dec, t] >= lig_score[L]).sum()) / (1 + len(dec))
        rank = 1 + sum(v > lig_score[L] for g, v in lig_score.items() if g != L)
        rows.append(dict(beta=b, ligand=L, p=round(p, 4), rank=rank, n_ligands=len(lig_score), recovered=bool(p <= 0.05 and rank <= 3)))
df = pd.DataFrame(rows); print(df.to_string(index=False)); print(df.groupby("beta").recovered.sum())
df.to_csv(R / os.environ.get("NNOUT", "graticcs_nichenet_spike.csv"), index=False)
