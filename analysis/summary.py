"""Recovery, additional selections and q-values for every analysis in the paper -> results/gt_summary.json."""
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
    pref = sys.argv[1] if len(sys.argv) > 1 else "G"
    s3 = lambda base: [f"{base}_s{s}" for s in range(3)]
    if pref == "G":
        runs = {f"contact_spike_b{b}": ("c", s3(f"G_spike_b{b}"), B / "spike_meta.json", True) for b in (0.05, 0.1, 0.2, 0.4)}
        runs.update({f"contact_spike_b{b}_seed{s}": ("c", [f"G_spike_b{b}_s{s}"], B / "spike_meta.json", True) for b in (0.05, 0.1, 0.2, 0.4) for s in range(3)})
        runs.update({f"secreted_spike_b{b}": ("s", [f"G2_sec_spike_b{b}"], B / "spike_sec_meta.json", False) for b in (0.05, 0.1, 0.2, 0.4)})
        runs.update({f"xenium_spike_b{b}": ("x", [f"G_xen_spike_b{b}"], X / "spike_meta.json", False) for b in (0.1, 0.4)})
        runs["pathway_spike_b0.2"] = ("c", ["G_pw_contact_spike_b0.2"], B / "spike_pw_meta.json", False)
        for f in ("gate", "hill", "decay"):
            for b in (0.2, 0.4):
                runs[f"mis_{f}_b{b}"] = ("c", s3("arch_graticcs_gate_b0.2") if (f, b) == ("gate", 0.2) else s3(f"G_mis_{f}_b{b}"), B / "spike_meta.json", True)
        for r in (0.3, 0.5, 0.7):
            for b in (0.2, 0.4): runs[f"lag_r{r}_b{b}"] = ("c", [f"G_lag_r{r}_b{b}"], B / "spike_meta.json", True)
        for b in (0.2, 0.4): runs[f"lag_r1.0_b{b}"] = ("c", [f"G_spike_b{b}_s0"], B / "spike_meta.json", True)
        for b in (0.1, 0.4): runs[f"spikeRNA_b{b}"] = ("c", [f"G_spikeRNA_b{b}"], B / "spike_rna_meta.json", True)
        runs.update(contact_real=("c", s3("G_real"), None, False), contact_real_niche=("c", s3("G_real_niche"), None, False),
                    contact_pw_real=("c", s3("G_pw_contact"), None, False), secreted_real=("s", s3("G2_sec_real"), None, False),
                    secreted_pw_real=("s", s3("G2_pw_sec"), None, False), xenium_real=("x", s3("G_xen_real"), None, False))
        for s in range(3): runs[f"contact_real_seed{s}"] = ("c", [f"G_real_s{s}"], None, False)
        for s in range(3): runs[f"secreted_pw_real_seed{s}"] = ("s", [f"G2_pw_sec_s{s}"], None, False)
    rows = {}
    for name, (kind, rr, meta, exn) in runs.items():
        o = evaluate(kind, rr, meta, exn); rows[name] = o
        print(f"{name:28s} gain {o['gain']:+.3%}  q<=0.1 {o['n_q10']:3d}" + (f"  recovered {o['rec']}  FP {o['fp']} (shared ligand {o['fp_shared_ligand']})  {o['detail']}" if meta else ""))
    json.dump({k: {kk: vv for kk, vv in v.items()} for k, v in rows.items()}, open(R / "gt_summary.json", "w"), indent=1, default=float)
