"""GraTICCS on the NicheNet-consistent planted data (three seeds, folds pooled), judged per ligand on its own NicheNet
program readout. Ligand score: maximum importance over the ligand's channels; p against decoys of the same modality;
recovered when p <= 0.05 and the ligand ranks among the top three of all candidate ligands.
Writes results/graticcs_nichenet_crand.csv.
"""
import json

import numpy as np
import pandas as pd

from graticcs.paths import BREAST, RESULTS

D = BREAST; ND = D / "nichenet_spike_crand"; R = RESULTS
meta = json.load(open(ND / "meta.json")); T = meta["readouts"]; cand = meta["candidates"]
ch = pd.DataFrame(json.load(open(D / "gt_sec_channels.json"))["channels"])
ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                         np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
rows = []
for b in meta["betas"]:
    fo = [json.load(open(p)) for s in range(3) for p in sorted((R / f"G2_nn_crand_b{b}_s{s}").glob("fold*_mo.json"))]
    if len(fo) < 15:
        print("beta", b, "incomplete:", len(fo)); continue
    imp = np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)
    for L in meta["plant"]:
        t = T.index(f"NN_{L}")
        lig_score = {g: imp[ch.index[~ch.decoy & (ch.ligand == g)], t].max() for g in cand if (~ch.decoy & (ch.ligand == g)).any()}
        cL = ch.index[~ch.decoy & (ch.ligand == L)]; st = ch.stratum[cL[np.argmax(imp[cL, t])]]
        dec = ch.index[ch.decoy & (ch.stratum == st)]
        p = (1 + (imp[dec, t] >= lig_score[L]).sum()) / (1 + len(dec))
        rank = 1 + sum(v > lig_score[L] for g, v in lig_score.items() if g != L)
        rows.append(dict(beta=b, ligand=L, p=round(p, 4), rank=rank, n_ligands=len(lig_score), recovered=bool(p <= 0.05 and rank <= 3)))
df = pd.DataFrame(rows); df.to_csv(R / "graticcs_nichenet_crand.csv", index=False)
print(df.to_string(index=False))
