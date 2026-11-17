"""Literature check on decoy channels: same response genes, ranking and control rules as analysis.literature_test.

The channels checked are those with a positive effect and p <= 0.05 in the literature test. Primary control: the decoys that
share the channel's receptor; secondary: all decoys of the same modality, each with its own receptor-positive controls.

python -m analysis.lit_decoy_control"""
import json
import numpy as np, pandas as pd, scipy.sparse as sp, torch
from scipy.stats import mannwhitneyu
from graticcs.paths import BREAST, RESULTS
D, R = BREAST, RESULTS
pick = lambda f: pd.read_csv(R / f).query("mw_p <= 0.05 and effect > 0").channel.tolist()
CASES = [("secreted", "G2_sec_real", "lit_enrichment_secreted_real_G2.csv", pick("lit_enrichment_secreted_real_G2.csv")),
         ("contact", "G_real", "lit_enrichment_contact_real.csv", pick("lit_enrichment_contact_real.csv"))]
z = np.load(D / "gt_inputs.npz"); ctype = z["ctype"]
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: i for i, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr(); lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
sp_ = lambda a: np.log1p(np.exp(np.clip(a.astype(np.float32), -10, 10)))
rows = []
for cls, run, ref, chans in CASES:
    if cls == "contact":
        meta = json.load(open(D / "gt_channels.json")); src, dst, L, Rr = z["src"], z["dst"], z["L_mo"], z["R_mo"]
    else:
        S = np.load(D / "gt_sec_inputs.npz"); meta = json.load(open(D / "gt_sec_channels.json")); src, dst, L, Rr = S["src"], S["dst"], S["L"], S["R"]
    ch = pd.DataFrame(meta["channels"]); ch["mod"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"), "")
    ch.loc[ch.name.str.startswith("PDECOY"), "mod"] = "protein"
    dirs = [R / f"{run}_s{s}" for s in range(3)]; fo = [json.load(open(p)) for d in dirs for p in sorted(d.glob("fold*_mo.json"))]
    bt = (np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)).argmax(1)
    fits = []
    for d in dirs:
        for k in range(5):
            A = np.load(d / f"fold{k}_mo_attention.npz"); e = A["edge"]; st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
            fits.append(dict(e=e, a=A["a"].astype(np.float32), s=src[e], d=dst[e], t=ctype[dst[e]], P=torch.softmax(st["P"], 1).numpy(), G=st["G"].numpy(), V=st["V"].numpy()))
    def top_receivers(c):
        C = np.zeros(len(src), np.float32); n = np.zeros(len(src), np.float32)
        for f in fits:
            C[f["e"]] += (f["a"] @ f["P"][c]) * sp_(L[f["s"], c]) * sp_(Rr[f["d"], c]) / (1 + np.exp(-f["G"][f["t"], c])) * f["V"][c, bt[c]]; n[f["e"]] += 1
        s = np.abs(C) / np.maximum(n, 1); best = np.argpartition(-s, 200)[:200]; best = best[np.argsort(-s[best])]
        return pd.unique(dst[best])[:50]
    lit = json.load(open(D / "lit_channel_matches.json"))[cls]; refd = pd.read_csv(R / ref).set_index("channel")
    all_dec = {}
    for chn in chans:
        lg, rg = chn.split("->")[0].split("_"), chn.split("->")[1].split("_")
        up = sorted({g for g, ev in lit[chn]["effectors"].items() if g in gi and g not in lg + rg and any(x.startswith("up") for x, _ in ev)})
        lu = np.log1p(X[:, [gi[g] for g in up]].toarray() / lib[:, None] * 1e3); zt = np.zeros_like(lu)
        for t in np.unique(ctype):
            m = ctype == t; zt[m] = (lu[m] - lu[m].mean(0)) / (lu[m].std(0) + 1e-9)
        sc = zt.mean(1)
        def test(top, receptor):
            rgx = [g for g in receptor.split("_") if g in gi]
            recx = np.asarray(X[:, [gi[g] for g in rgx]].sum(1)).ravel() if rgx else np.ones(len(sc))
            ctrl = np.flatnonzero(np.isin(ctype, ctype[top]) & (recx > 0) & ~np.isin(np.arange(len(sc)), top))
            return sc[top].mean() - sc[ctrl].mean(), mannwhitneyu(sc[top], sc[ctrl], alternative="greater").pvalue
        c = ch.index[ch.name == chn][0]; eff, p = test(top_receivers(c), ch.receptor[c])
        print(f"{chn}: effect {eff:+.4f} p {p:.3g} | table effect {refd.effect[chn]:+.4f} p {refd.mw_p[chn]:.3g}", flush=True)
        rows.append(dict(channel=chn, comparator="real", decoy=chn, modality="", shares_receptor=True, effect=eff, mw_p=p))
        same = ch.index[ch.decoy & (ch.of == chn)]; mod = ch["mod"][same[0]]
        for d in ch.index[ch.decoy & (ch["mod"] == mod)]:
            if d not in all_dec: all_dec[d] = top_receivers(d)
            ed, pd_ = test(all_dec[d], ch.receptor[d])
            rows.append(dict(channel=chn, comparator="decoy", decoy=ch.name[d], modality=mod, shares_receptor=ch.receptor[d] == ch.receptor[c], effect=ed, mw_p=pd_))
        print(f"  decoys done: {sum(1 for r in rows if r['channel'] == chn) - 1}", flush=True)
    del fits
out = pd.DataFrame(rows); out.to_csv(R / "lit_decoy_control.csv", index=False)
sumr = []
for chn, g in out.groupby("channel", sort=False):
    r = g[g.comparator == "real"].iloc[0]; d = g[g.comparator == "decoy"]; s = d[d.shares_receptor]
    sumr.append(dict(channel=chn, n_same_receptor=len(s), real_effect=r.effect, real_p=r.mw_p, same_receptor_effects=" ".join(f"{x:+.3f}" for x in s.effect), same_receptor_p=" ".join(f"{x:.2g}" for x in s.mw_p),
                     n_same_receptor_p05_pos=int(((s.mw_p <= 0.05) & (s.effect > 0)).sum()), n_all_decoys=len(d), all_decoy_median_effect=d.effect.median(),
                     frac_decoys_effect_ge_real=(d.effect >= r.effect).mean(), emp_p_effect=(1 + (d.effect >= r.effect).sum()) / (1 + len(d)), emp_p_mw=(1 + (d.mw_p <= r.mw_p).sum()) / (1 + len(d))))
sumr = pd.DataFrame(sumr); sumr.to_csv(R / "lit_decoy_control_summary.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 60); print(sumr.to_string(index=False))
