"""Channel calibration shared by the analyses, and the original-tissue summary (prediction gain per seed and pooled,
channels with q <= 0.1) -> results/original_tissue_summary.json."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from statsmodels.stats.multitest import multipletests
from graticcs.paths import BREAST, XENIUM, RESULTS
R = RESULTS; B = BREAST; X = XENIUM
NATURAL = {"HLA-A->CD8A", "HLA-DRA->CD4"}


def load_ch(f):
    ch = pd.DataFrame(json.load(open(f))["channels"])
    if "decoy_modality" not in ch: ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
    ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"),
                             np.where(ch.name.isin(set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])), "protein", "rna"))
    return ch


CH = {"c": load_ch(B / "gt_channels.json"), "s": load_ch(B / "gt_sec_channels.json"), "x": load_ch(X / "gt_channels.json")}


def evaluate(kind, runs, meta=None, exclude_natural=False):
    ch = CH[kind]; dirs = [R / r for r in runs]
    fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]
    assert len(fo) == 5 * len(dirs), (runs, len(fo))
    T = fo[0]["targets"]; s0 = np.sum([f["sse_nuisance"] for f in fo], 0); s1 = np.sum([f["sse_full"] for f in fo], 0)
    imp = np.sum([f["dsse_knockout"] for f in fo], 0) / s0; ib = imp.max(1); q = np.ones(len(ch)); p = np.ones(len(ch))
    for st in ["protein", "rna"]:
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        if not len(rl) or not len(dec): continue
        p[rl] = [(1 + (ib[dec] >= ib[c]).sum()) / (1 + len(dec)) for c in rl]; q[rl] = multipletests(p[rl], method="fdr_bh")[1]
    real = ~ch.decoy.to_numpy(); hits = set(ch.name[real & (q <= 0.1)])
    out = dict(gain=1 - s1.sum() / s0.sum(), n_q10=len(hits), q=dict(zip(ch.name[real], q[real])), p=dict(zip(ch.name[real], p[real])),
               best=dict(zip(ch.name[real], [T[i] for i in imp.argmax(1)[real]])), stratum=dict(zip(ch.name[real], ch.stratum[real])))
    if meta:
        sm = json.load(open(meta)); plant = [x for x in sm["plant"] if not (exclude_natural and x[0] in NATURAL)]
        rec = 0; detail = []
        for chn, tgt, sgn in plant:
            c = ch.index[ch.name == chn][0]; t = T.index(tgt); st = ch.stratum[c]
            dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
            pp = (1 + (imp[dec, t] >= imp[c, t]).sum()) / (1 + len(dec)); rk = int((imp[rl, t] > imp[c, t]).sum()) + 1
            rec += pp <= 0.05 and rk <= 3; detail.append((chn, tgt, st, round(pp, 4), rk, bool(pp <= 0.05 and rk <= 3)))
        pl = {x[0] for x in sm["plant"]}; ligs = {x[0].split("->")[0] for x in sm["plant"]}
        fp = hits - pl; out.update(rec=f"{rec}/{len(plant)}", n_rec=rec, fp=len(fp), fp_shared_ligand=sum(f.split("->")[0] in ligs for f in fp), detail=detail)
    return out


if __name__ == "__main__":
    s3 = lambda base: [f"{base}_s{s}" for s in range(3)]
    runs = dict(contact_real=("c", "G_real"), contact_real_niche=("c", "G_real_niche"), contact_pathway_real=("c", "G_pw_contact"),
                contact_local_real=("c", "L_real"), secreted_real=("s", "G2_sec_real"), secreted_pathway_real=("s", "G2_pw_sec"),
                xenium_real=("x", "G_xen_real"))
    rows = {}
    for name, (kind, base) in runs.items():
        o = evaluate(kind, s3(base)); o["seed_gain"] = [evaluate(kind, [r])["gain"] for r in s3(base)]; rows[name] = o
        print(f"{name:24s} gain {o['gain']:+.3%}  seeds " + " ".join(f"{g:+.2%}" for g in o["seed_gain"]) + f"  q<=0.1 {o['n_q10']}")
    json.dump(rows, open(R / "original_tissue_summary.json", "w"), indent=1, default=float)
