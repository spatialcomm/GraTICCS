"""The 50 best-scored events of selected channels, for the case figures (scoring as analysis/events.py).

python -m analysis.top_events secreted G2_sec_real_s0,G2_sec_real_s1,G2_sec_real_s2 results/case_events_secreted.csv "IL1B->SIGIRR,CXCL9->CXCR3,TIMP3->CD44"
python -m analysis.top_events contact G_real_s0,G_real_s1,G_real_s2 results/case_events_contact.csv "HLA-DQA1->CD4"
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.events import D, scores
from graticcs.paths import RESULTS

kind, runs, out, want = sys.argv[1], [RESULTS / x for x in sys.argv[2].split(",")], sys.argv[3], sys.argv[4].split(",")
s = scores(kind, runs); ch, real, score, edges = s["ch"], s["real"], s["score"], s["edges"]
TYPES = json.load(open(D / "gt_channels.json"))["types"]
allflat = score.ravel()
cells = pd.read_parquet(D / "cells.parquet", columns=["cell_id", "fov"]); cid = cells.cell_id.to_numpy(); fovs = cells.fov.to_numpy()
X = sp.load_npz(D / "rna_counts.npz").tocsr(); gi = {g.upper(): k for k, g in enumerate(json.load(open(D / "genes.json"))["genes"])}
src, dst, ctype = s["src"], s["dst"], s["ctype"]
rows = []
for nm in want:
    j = int(np.flatnonzero(ch.name[real].to_numpy() == nm)[0]); c = real[j]
    lg = [g for g in ch.ligand[c].split("_") if g in gi]; rg = [g for g in ch.receptor[c].split("_") if g in gi]
    for k, e in enumerate(np.argsort(-score[:, j])[:50]):
        ed = edges[e]; a, b = src[ed], dst[ed]
        rows.append(dict(channel=nm, stratum=ch.stratum[c], p=s["p"][c], q=s["q"][c], best_readout=s["T"][s["bt"][c]],
                         rank_all=int((allflat > score[e, j]).sum()) + 1, n_events=int(allflat.size), rank_in_channel=k + 1,
                         contrib=float(s["contrib"][e, j]), score=float(score[e, j]), edge=int(ed), sender=cid[a], receiver=cid[b],
                         fov=int(fovs[b]), sender_type=TYPES[ctype[a]], receiver_type=TYPES[ctype[b]],
                         ligand_rna=int(sum(X[a, gi[g]] for g in lg)), receptor_rna=int(sum(X[b, gi[g]] for g in rg)),
                         L_input=float(s["L"][a, c]), R_input=float(s["R"][b, c]), contact=float(s["contact"][ed]), dist_um=float(s["dist"][ed])))
pd.DataFrame(rows).to_csv(out, index=False)
