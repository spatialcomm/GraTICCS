"""Recovery of planted signals, three seeds pooled.

Recovered: decoy p <= 0.05 on the planted readout and a top-three rank among real channels of the same modality, with
the other planted channels left out of the ranking (main rule) or kept in it (strict rule). Additional selections: real
channels outside the planted set with q <= 0.1. Channel AUROC uses the best-readout importance.

python -m analysis.recovery contact | secreted | xenium | xenium_batch2 | forms | secreted_delay
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from analysis.summary import CH
from graticcs.paths import BREAST, RESULTS, XENIUM

R = RESULTS


def evaluate(ch, plant, tags):
    real = ~ch.decoy.to_numpy(); PL = {p[0] for p in plant}
    fo = [json.load(open(p)) for t in tags for p in sorted((R / t).glob("fold*_mo.json"))]
    assert len(fo) == 5 * len(tags), tags
    T = fo[0]["targets"]; s0 = np.sum([f["sse_nuisance"] for f in fo], 0); s1 = np.sum([f["sse_full"] for f in fo], 0)
    imp = np.sum([f["dsse_knockout"] for f in fo], 0) / s0; ib = imp.max(1); q = np.ones(len(ch))
    for st in ("protein", "rna"):
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        p = [(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]; q[rl] = multipletests(p, method="fdr_bh")[1]
    det = []
    for chn, tgt, *_ in plant:
        c = ch.index[(ch.name == chn) & ~ch.decoy][0]; t = T.index(tgt); st = ch.stratum[c]
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        others = rl[~ch.name[rl].isin(PL)]
        pp = (1 + (imp[dec, t] >= imp[c, t]).sum()) / (1 + len(dec))
        rk = int((imp[others, t] > imp[c, t]).sum()) + 1; rk_strict = int((imp[rl, t] > imp[c, t]).sum()) + 1
        det.append(dict(channel=chn, readout=tgt, stratum=st, p=round(pp, 4), rank=rk, rank_strict=rk_strict,
                        recovered=bool(pp <= 0.05 and rk <= 3), recovered_strict=bool(pp <= 0.05 and rk_strict <= 3)))
    hits = set(ch.name[real & (q <= 0.1)]) - PL; ligs = {c.split("->")[0] for c in PL}
    planted = ch.name[real].isin(PL).to_numpy()
    return dict(recovered=sum(d["recovered"] for d in det), recovered_strict=sum(d["recovered_strict"] for d in det),
                FP=len(hits), FP_shared_ligand=sum(h.split("->")[0] in ligs for h in hits),
                channel_AUROC=round(roc_auc_score(planted, ib[real]), 3), error_full_over_nuisance=round(s1.sum() / s0.sum(), 5),
                sse_full=float(s1.sum())), det


def complete(tags):
    return all(len(list((R / t).glob("fold*_mo.json"))) == 5 for t in tags)


def table(ch, plant, runs, keys):
    """runs: list of (key values, run tags). Returns summary and per-channel tables."""
    rows, dets = [], []
    for kv, tags in runs:
        if not complete(tags):
            print("incomplete", kv); continue
        o, det = evaluate(ch, plant, tags); o["seeds"] = [evaluate(ch, plant, [t])[0]["recovered"] for t in tags]
        rows.append(dict(**dict(zip(keys, kv)), n=len(plant), **o)); dets += [dict(**dict(zip(keys, kv)), **d) for d in det]
    return pd.DataFrame(rows), pd.DataFrame(dets)


def s3(base):
    return [f"{base}_s{s}" for s in range(3)]


def main(which):
    if which in ("contact", "secreted", "xenium", "xenium_batch2"):
        kind, meta, models, betas, out = {
            "contact": ("c", BREAST / "spike_crand_meta.json", {"GraTICCS": "G_crand", "local": "L_crand"}, (0.05, 0.1, 0.2, 0.4), "planted_random_contact"),
            "secreted": ("s", BREAST / "spike_sec_rand_meta.json", {"GraTICCS": "G2_sec_rand", "local": "L_sec_rand"}, (0.05, 0.1, 0.2, 0.4), "planted_random_secreted"),
            "xenium": ("x", XENIUM / "spike_crand_meta.json", {"GraTICCS": "G_xen_crand"}, (0.1, 0.4), "planted_random_xenium"),
            "xenium_batch2": ("x", XENIUM / "spike_crand2_meta.json", {"GraTICCS": "G_xen_crand2"}, (0.1, 0.4), "planted_random_xenium_batch2"),
        }[which]
        plant = json.load(open(meta))["plant"]
        r, d = table(CH[kind], plant, [((m, b), s3(f"{pre}_b{b}")) for m, pre in models.items() for b in betas], ["model", "beta"])
        if {"GraTICCS", "local"} <= set(r.model):
            sse = r.pivot(index="beta", columns="model", values="sse_full")
            r["error_GraTICCS_over_local"] = r.beta.map(sse.GraTICCS / sse.local).round(4)
    elif which == "forms":
        plant = json.load(open(BREAST / "spike_crand_meta.json"))["plant"]; out = "misspec_crand"
        runs = [(("GraTICCS", f, b), s3(f"G_crand_mis_{f}_b{b}")) for f in ("hill", "gate", "decay") for b in (0.2, 0.4)]
        runs.append((("local", "gate", 0.2), s3("L_crand_mis_gate_b0.2")))
        r, d = table(CH["c"], plant, runs, ["model", "form", "beta"])
    elif which == "secreted_delay":
        plant = json.load(open(BREAST / "spike_sec_rand_meta.json"))["plant"]; out = "lag_eval_secreted"
        runs = [((rho, b), s3(f"G2_sec_rand_b{b}") if rho == 1.0 else s3(f"G2_sec_lag_r{rho}_b{b}"))
                for rho in (1.0, 0.7, 0.5, 0.3) for b in (0.2, 0.4)]
        r, d = table(CH["s"], plant, runs, ["rho", "beta"])
    else:
        raise SystemExit(__doc__)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    if which == "secreted_delay":
        r.to_csv(R / f"{out}.csv", index=False); d.to_csv(R / f"{out}_detail.csv", index=False)
    else:
        r.to_csv(R / f"{out}_summary.csv", index=False); d.to_csv(R / f"{out}_detail.csv", index=False)
    print(r.drop(columns="sse_full").to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
