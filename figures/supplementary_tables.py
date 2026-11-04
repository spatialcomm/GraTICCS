"""Supplementary tables (LaTeX) from the result files."""
import json, re
from pathlib import Path
import numpy as np, pandas as pd
from analysis import summary as g
from graticcs.paths import BREAST, XENIUM, RESULTS
R = RESULTS; D = BREAST; X = XENIUM
OUT = RESULTS / "supplementary_tables.tex"; PROV = RESULTS / "calibration"
B = chr(92)
def esc(x): x = "--" if (isinstance(x, float) and np.isnan(x)) else x; return str(x).replace("_", B + "_").replace("->", "$" + B + "rightarrow$").replace("%", B + "%").replace("<=", "$" + B + "le$")
def table(df, caption, label, fmt=None, size="small"):
    fmt = fmt or {}
    lines = [B + "begin{table}[h!]", B + "centering", B + size, B + "caption{" + esc(caption) + "}" + B + "label{" + label + "}",
             B + "begin{tabular}{" + " ".join(["l"] * len(df.columns)) + "}", B + "toprule", " & ".join(esc(c) for c in df.columns) + " " + B + B, B + "midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt[c].format(v) if c in fmt and isinstance(v, (int, float, np.floating)) else esc(v) for c, v in r.items()) + " " + B + B)
    return "\n".join(lines + [B + "bottomrule", B + "end{tabular}", B + "end{table}", ""])
def para(t): return t + "\n"
S = json.load(open(R / "gt_summary.json"))
s3 = lambda base: [f"{base}_s{k}" for k in range(3)]
parts = [B + "documentclass[11pt]{article}", B + "usepackage[margin=2cm]{geometry}", B + "usepackage{booktabs,amsmath,graphicx}",
         B + "renewcommand{" + B + "thetable}{S" + B + "arabic{table}}", B + "renewcommand{" + B + "thefigure}{S" + B + "arabic{figure}}",
         B + "begin{document}", B + "section*{Supplementary Material for: GraTICCS}",
         para("All tables are generated from result files by make_supplementary_gt.py. Contact and Xenium analyses use the graph transformer (G\\_* runs); secreted analyses use the graph transformer trained with gradient accumulation over two mini-batches of two FOVs (G2\\_* runs). Recovered = decoy p <= 0.05 on the planted readout and rank <= 3 among real channels; FP = additional selected channels, i.e. non-planted real channels with q $\\le$ 0.1 (an upper bound on false positives, because non-planted channels may act in the tissue).")]
parts.append(table(pd.DataFrame([["CosMx breast multiomic", "152,451", "18,942", "64", "249,422", "203 / 1,140", "188 / 940"],
                                  ["Xenium renal cell carcinoma", "465,534", "405", "27", "785,540", "12 / 85", "--"]],
                                 columns=["dataset", "cells", "genes", "proteins", "contact pairs", "contact channels (real/decoy)", "secreted channels (real/decoy)"]),
                   "Datasets, contact graphs and channel sets.", "tab:s1"))
parts.append(table(pd.DataFrame([["v1: protein readouts include channel receptors", "4.42%", "receptor = readout"],
                                 ["v1 fixed (context in nuisance)", "0.05%", "generic neighbour context absorbs all effects"],
                                 ["v2: receptor RNA not in nuisance", "11.94%", "receiver self-information via channel strength"],
                                 ["v2: attention/gates from embeddings", "5.97%", "neighbour state carried by every channel"],
                                 ["v3: clean layer, RNA decoys only", "0.37%", "protein ligands vs RNA decoys (modality confound)"],
                                 ["v3b: modality-matched decoys (local attention)", "0.0007%", "none detected (95% CI -0.15% to 0.15%)"],
                                 ["final: graph transformer, all controls (3 seeds)", f"{S['contact_real']['gain']:.2%}", f"decoys explain as much; {S['contact_real']['n_q10']} real channels with q <= 0.1"]],
                                columns=["design", "held-out LR gain", "diagnosis"]),
                   "Design iterations on breast contact channels (real data) and the leaks exposed by decoy channels. The first v1 and v2 values (4.42% and 11.94%) come from the run log because their output files were overwritten by later corrected runs; versions differ in the folds evaluated.", "tab:s2"))
def spike_rows(keys):
    rows = []
    for b, k in keys:
        for chn, tgt, st, p, rk, ok in S[k]["detail"]: rows.append(dict(beta=b, channel=chn, readout=tgt, stratum=st, p=p, rank=rk, recovered="yes" if ok else "no"))
    return pd.DataFrame(rows)
parts.append(table(spike_rows([(b, f"contact_spike_b{b}") for b in (0.05, 0.1, 0.2, 0.4)]),
                   "Planted contact channels, breast (three-seed ensemble; the two naturally active HLA channels are excluded from the counts and not shown).", "tab:s3", {"p": "{:.4f}"}, "footnotesize"))
parts.append(table(spike_rows([(b, f"secreted_spike_b{b}") for b in (0.05, 0.1, 0.2, 0.4)]),
                   "Planted secreted channels, breast (single seed).", "tab:s4", {"p": "{:.4f}"}, "footnotesize"))
parts.append(table(pd.read_csv(R / "lag_eval_GT.csv"), "Planted contact channels under simulated signalling time lag (single seed; AP = pair-level average precision).", "tab:s5",
                   {"ours_AP": "{:.3f}", "coexpr_now_AP": "{:.3f}", "random_AP": "{:.4f}"}))
parts.append(table(pd.read_csv(R / "final_pair_level_comparison_GT.csv"), "Pair-level average precision (twelve FOVs) for protein- and RNA-defined planted truth.", "tab:s6", {"AP": "{:.4f}"}, "footnotesize"))
parts.append(table(pd.read_csv(R / "hybrid_pair_level_GT.csv"), "Pair-level AP of expression-based scores alone and multiplied by the GraTICCS channel evidence score (one minus the false discovery rate).", "tab:s7", {"AP": "{:.4f}"}, "footnotesize"))
parts.append(table(pd.read_csv(R / "nichenet_summary.csv")[["beta", "ligand", "sender", "candidate", "n_candidates", "rank", "recovered"]],
                   "NicheNet (cell-type niche) ranks of ligands planted according to the NicheNet prior.", "tab:s8", None, "footnotesize"))
parts.append(table(pd.read_csv(R / "nichenet_ligand_niche.csv")[["beta", "ligand", "niche_cells", "geneset", "rank", "recovered"]],
                   "NicheNet (ligand-specific niche) ranks.", "tab:s9", None, "footnotesize"))
parts.append(table(pd.read_csv(R / "graticcs_nichenet_spike_GT.csv"), "GraTICCS ligand-level recovery on the same planted data (NicheNet programs as readouts; single seed).", "tab:s10", {"p": "{:.4f}"}, "footnotesize"))
def top(kind, runs, n):
    o = g.evaluate(kind, runs)
    df = pd.DataFrame(dict(channel=list(o["q"]), stratum=[o["stratum"][c] for c in o["q"]], best_readout=[o["best"][c] for c in o["q"]],
                           p=[o["p"][c] for c in o["q"]], q=list(o["q"].values())))
    return df.sort_values(["q", "p"]).head(n)
sr = top("s", s3("G2_sec_real"), 16); sp_ = top("s", s3("G2_pw_sec"), 8)
seeds = [S.get(f"secreted_real_seed{k}", None) for k in range(3)]
nseed = [g.evaluate("s", [f"G2_sec_real_s{k}"])["n_q10"] for k in range(3)]; npw = [S[f"secreted_pw_real_seed{k}"]["n_q10"] for k in range(3)]
parts.append(table(pd.concat([sr.assign(readouts="protein + RNA programs"), sp_.assign(readouts="pathway/TF")])[["readouts", "channel", "stratum", "best_readout", "p", "q"]],
                   f"Breast secreted channels in unspiked data (three-seed ensembles): top channels by q. Held-out LR gain {S['secreted_real']['gain']:.2%} (protein + RNA programs) and {S['secreted_pw_real']['gain']:.2%} (pathway/TF). "
                   f"Channels with q <= 0.1 in single seeds: {', '.join(map(str, nseed))} (protein + RNA programs) and {', '.join(map(str, npw))} (pathway/TF). In the pathway/TF configuration the protein readouts are kept and only the RNA programs are replaced by pathway/TF activities, so a protein best readout (e.g. IDO1, HLA-DR) is not pathway-level evidence.", "tab:s11", {"p": "{:.4f}", "q": "{:.3f}"}, "footnotesize"))
parts.append(table(top("x", s3("G_xen_real"), 12)[["channel", "stratum", "best_readout", "p", "q"]],
                   f"Xenium renal cell carcinoma contact channels (three-seed ensemble; held-out LR gain {S['xenium_real']['gain']:.2%}).", "tab:s12", {"p": "{:.4f}", "q": "{:.3f}"}))
parts.append(table(pd.read_csv(R / "holonet_spike_nreg5.csv")[["beta", "channel", "target", "holonet_rank", "n_views", "recovered"]],
                   "HoloNet layer-attention ranks of planted contact channels (five FOVs, 30 LR views; top-3 counted as recovered).", "tab:s13", None, "footnotesize"))
pr = {}
for line in (PROV / "prior_discovery_rates.txt").read_text().splitlines():
    m = re.match(r"(\w+): real channels with program (\d+); p<=0.05: (\d+) \(([\d.]+)%\); binomial p vs 5% = ([\d.]+); q<=0.1: (\d+)", line)
    if m: pr[m.group(1)] = m.groups()[1:]
hits = {k: re.findall(r"real channels with q<=0.1: (\[.*\])", (PROV / f"secreted_{k}_2seed.txt").read_text())[-1] for k in pr}
lab = {"prior672_lit": "literature, development papers (672; 11 ligands)", "prior672_nn": "NicheNet top-50 (same 11 ligands)",
       "prior836_lit": "literature, all papers (836; 14 ligands)", "prior836_nn": "NicheNet top-50 (same 14 ligands)"}
parts.append(table(pd.DataFrame([[lab[k], v[0], f"{v[1]} ({v[2]}%)", v[3], v[4], hits[k].strip("[]").replace("'", "") or "--"] for k, v in pr.items()],
                                columns=["readout definition", "real channels", "p <= 0.05", "binomial p vs 5%", "q <= 0.1", "channels with q <= 0.1"]),
                   "Ligand-specific prior programs as readouts (breast secreted channels, two-seed ensembles; each channel and its decoys may explain only the program of its own ligand). Neither prior raised the discovery rate significantly above the 5% expected by chance. Literature and NicheNet programs shared only 0--5 genes per ligand.", "tab:s14", None, "footnotesize"))
parts.append(table(pd.DataFrame([
    ["full model, local attention (all controls)", "0.0007% [-0.15, 0.15]", "no excess over decoys", "5 folds"],
    ["generic neighbour context in nuisance branch", "0.05%", "absorbs all neighbour effects (over-conservative)", "fold 0"],
    ["receptor proteins kept as readouts", "4.4%", "real share 18.7% vs 16.7% expected; decoys equally predictive", "fold 0"],
    ["receiver receptor features not in nuisance", "11.9%", "decoys equally predictive (receiver self-information)", "fold 0"],
    ["embedding-based attention and gates (full-state keys/values)", "6.0% [4.5, 7.3]", "real share 31%; neighbour information smuggled by every channel", "fold 0"],
    ["RNA decoys only (no protein decoys)", "0.37% [0.27, 0.47]", "real share 41%, driven by protein-measured ligands (modality confound)", "5 folds"],
    ["final graph transformer (all controls)", f"{S['contact_real']['gain']:.2%}", "decoys explain as much; no real channel with q <= 0.1", "5 folds x 3 seeds"]],
    columns=["configuration", "LR-branch held-out gain [FOV bootstrap CI]", "decoy diagnosis", "evaluated on"]),
    "Leakage controls on breast contact channels (real data, no planted signal). Rows are development runs in which one control was absent; they differ in the model version and the folds evaluated (last column), so they document each leak rather than form a controlled one-at-a-time ablation. In every leaking configuration the decoy channels were as predictive as the real channels.", "tab:s15", None, "footnotesize"))
t = [pd.Timestamp(x) for x in re.findall(r"^done \S+ \w+ (\w+ +\d+ [\d:]+) E[DS]T \d+$", (R / "run_gt_sec2.log").read_text(), re.M)]
dt = np.diff([pd.Timestamp("2026-10-31 21:20:00")] + [x.replace(year=2026) for x in t]) / pd.Timedelta(minutes=1)
parts.append(table(pd.DataFrame([["breast CosMx, contact", "152,451", "0.81 M", "1,343", "~6 min"],
                                 ["breast CosMx, secreted (30 um)", "152,451", "3.06 M", "1,128", f"~{int(np.percentile(dt, 10)):d}--{int(np.percentile(dt, 90)):d} min"],
                                 ["Xenium renal cell carcinoma, contact", "465,534", "--", "97", "~10 min"]],
                                columns=["dataset / channel class", "cells", "edges", "channels (incl. decoys)", "time per 5-fold run"]),
                   "Run time of the graph transformer on a single laptop GPU (NVIDIA RTX 4050 Laptop, 6 GB; 34 GB system RAM). A seed ensemble multiplies the time by the number of seeds.", "tab:s16"))
cor = {"decay": "0.83--0.94", "hill": "0.64--0.94", "gate": "0.56--0.80"}
mrow = [["standard (additive, dose-linear; matches model)", "1.00", S["contact_spike_b0.2"]["rec"], S["contact_spike_b0.4"]["rec"]]]
mrow += [[n, cor[f], S[f"mis_{f}_b0.2"]["rec"], S[f"mis_{f}_b0.4"]["rec"]] for f, n in [("decay", "proximity decay, no contact required"), ("hill", "switch-like saturation (Hill, n = 2)"), ("gate", "binary AND gate, no dose")]]
parts.append(table(pd.DataFrame(mrow, columns=["planted response form", "correlation with standard form", "recovered (beta = 0.2)", "recovered (beta = 0.4)"]),
                   "Planted-signal recovery when the planted response does not follow the form assumed by the model (breast contact channels, three-seed ensembles; five evaluable channels).", "tab:s17"))
parts.append(table(pd.DataFrame([
    ["attention layer", "EventGraphAttention (4 heads, edge bias, residual, LayerNorm, FFN, 2 layers)", "--"],
    ["encoder and fusion", "node encoder; graph + local branch fused by a sigmoid gate", "--"],
    ["supervision", "measured receiver response", "agreement with a score computed from the inputs would not show a functional effect"],
    ["keys and values", "neighbour cell type, geometry, event embedding", "the full neighbour state let every channel, decoys included, carry neighbour information (6.0% spurious gain)"],
    ["pathway/TF activities", "supervised readouts, not inputs", "inputs would contain the response"],
    ["literature", "not a model input (tested separately, Table S20)", "only real channels have literature; decoy calibration would fail"],
    ["output", "additive per-channel contributions", "allows exact channel knockouts and decoy comparison"]],
    columns=["design element", "choice", "reason"]),
    "Main design choices of the GraTICCS graph transformer.", "tab:s18", None, "footnotesize"))
cmp = []
for name, cond, meta in [("planted, beta = 0.05", "std_b0.05", D / "spike_meta.json"), ("planted, beta = 0.1", "std_b0.1", D / "spike_meta.json"),
                         ("binary gate, beta = 0.2", "gate_b0.2", D / "spike_meta.json"), ("unspiked", "real", None)]:
    og = g.evaluate("c", s3(f"arch_graticcs_{cond}"), meta, True); ol = g.evaluate("c", s3(f"arch_graticcs_local_{cond}"), meta, True)
    cmp.append([name, og.get("rec", "--"), ol.get("rec", "--"), og["fp"] if meta else og["n_q10"], ol["fp"] if meta else ol["n_q10"]])
wil = {"planted, beta = 0.05": ("81", "2e-9"), "planted, beta = 0.1": ("53", "0.51"), "binary gate, beta = 0.2": ("70", "0.001"), "unspiked": ("58", "0.014")}
cmp = pd.DataFrame([r + list(wil[r[0]]) for r in cmp], columns=["condition", "recovered (graph)", "recovered (local)", "FP (graph)", "FP (local)", "FOVs graph better (of 108)", "Wilcoxon p"])
parts.append(table(cmp, "Graph transformer versus matched local model (identical inputs, splits and three seeds; breast contact channels). FOVs graph better = held-out FOVs with lower readout error; paired Wilcoxon test. Effect sizes (total full-model error, graph / local; median per-FOV relative difference): 0.987 and -0.9% (beta = 0.05), 1.000 and +0.03% (beta = 0.1), 0.993 and -0.5% (binary gate), 0.993 and -0.1% (unspiked); per-FOV values in s19_per_fov.csv (g23_s19_per_fov.py). FP = additional selected (non-planted) channels with q <= 0.1, which have no ground-truth label.", "tab:s19", None, "footnotesize"))
lc = pd.read_csv(R / "lit_enrichment_contact_real.csv").assign(channels="contact (G_real)")
ls = pd.read_csv(R / "lit_enrichment_secreted_real_G2.csv").assign(channels="secreted, protein + RNA programs")
lp = pd.read_csv(R / "lit_enrichment_secreted_pw_G2.csv").assign(channels="secreted, pathway/TF")
summ = ", ".join(f"{n}: {(d.mw_p <= 0.05).sum()}/{len(d)} with p <= 0.05 ({0.05 * len(d):.1f} expected)" for n, d in [("contact", lc), ("secreted (protein + RNA programs)", ls), ("secreted (pathway/TF)", lp)])
lt = pd.concat([d[d.mw_p <= 0.1] for d in (lc, ls, lp)])[["channels", "channel", "q_channel", "best_rank", "n_response_genes", "effect", "mw_p", "mw_q"]]
parts.append(table(lt, "Literature-reported receiver response genes in the 50 top-ranked receivers of each channel versus receptor-positive controls of the same cell types (one-sided Mann-Whitney; BH across channels; channels with p <= 0.1 shown). "
                   + summ + ". Negative effects are not supportive. Columns: q_channel, channel-level decoy q_c (communication screen); mw_p and mw_q, one-sided Mann-Whitney p (selected receivers rank higher than controls) and its Benjamini-Hochberg q_lit across the tested channels of the same readout set; effect, difference in mean response-gene score (selected minus control), so a small mw_p can accompany a negative effect; best_rank, rank of the channel's best event among all candidate events. The 50 selected receivers are distinct cells and are excluded from the controls.", "tab:s20", {"q_channel": "{:.2f}", "effect": "{:.3f}", "mw_p": "{:.2g}", "mw_q": "{:.2g}"}, "footnotesize"))
mr = pd.read_csv(R / "receptor_mask_rules.csv")
EXG = {"TRAIL receptors": "TNFRSF10A-D", "TNF-receptor superfamily": "TNFRSF*, LTBR, EDAR, RELT", "IL-1/IL-18 receptors, TLRs": "IL1R*, IL18R*, IL1RAP, SIGIRR, TLR*",
       "type I/II cytokine receptors": "IL*R, IL2RG, IL6ST, IFNAR/GR/LR, CSF2R/3R, LIFR, OSMR, ...", "TGF-beta/BMP/activin receptors": "TGFBR*, ACVR*, BMPR*, ENG, AMHR2",
       "EGFR/ERBB": "EGFR, ERBB2-4", "VEGF receptors, neuropilins": "KDR, FLT1, FLT4, NRP1, NRP2",
       "other receptor tyrosine kinases": "MET, PDGFR*, FGFR*, IGF1R, INSR, KIT, AXL, MERTK, EPH*, ...", "chemokine receptors and other GPCRs": "CCR*, CXCR*, ACKR*, CX3CR1, GPR*, ADRB*, ...",
       "integrins and adhesion/matrix receptors": "ITG*, CD44, SDC*, GPC*, CD47, SIRPA", "Frizzled, LRP5/6, ROR, RYK": "FZD*, LRP5, LRP6, RYK",
       "T-cell receptors and co-receptors": "PDCD1, CTLA4, CD28, ICOS, CD4, CD8A/B, LAG3, HAVCR2, TIGIT, CD27, ...", "no rule matched": "--"}
mr["receptor genes"] = mr.rule.map(EXG)
parts.append(table(mr[["rule", "receptor genes", "pathways", "TFs", "contact", "secreted", "reference"]].rename(columns={"pathways": "PROGENy pathways", "TFs": "CollecTRI TFs"}),
                   "Receptor-conditioning rules for the pathway/TF readouts. Each LR channel, and each of its decoys (which share its receptor), may explain only the "
                   "PROGENy pathways and CollecTRI TFs assigned to the family of its receptor; heteromeric receptors receive the union over subunits. The rules are curated "
                   "from canonical signalling biology (reference column; full citations in the main text) rather than taken from a database, and the TFs are the canonical "
                   "effector TFs of each pathway that are available in CollecTRI. contact and secreted = number of real CosMx breast channels covered by each rule (a channel "
                   "can match more than one rule). Channels matching no rule have no allowed pathway/TF readout; the functional-protein and RNA-program readouts are not "
                   "receptor-restricted apart from excluding the channel's own ligand or receptor. In the implementation ROR1/ROR2 match the receptor-tyrosine-kinase rule "
                   "before the Wnt rule. Generated by prepare/receptor_rules.py.", "tab:s21", None, "scriptsize"))
for i, (f, cap) in enumerate([("Fig_S_TNFSF9_TNFRSF9", "4-1BBL (TNFSF9) to 4-1BB (TNFRSF9), contact channel"),
                              ("Fig_S_TNFSF18_TNFRSF18", "GITRL (TNFSF18) to GITR (TNFRSF18), contact channel (weaker support)"),
                              ("Fig_S_SPP1_ITGAV_ITGB1", "SPP1 to ITGAV/ITGB1, secreted channel (weak support, literature q = 0.14)")], 1):
    parts += [B + "begin{figure}[h!]", B + "centering", B + "includegraphics[width=" + B + "linewidth]{figures/supp/" + f + ".png}",
              B + "caption{Literature-supported predicted event: " + cap + ". Layout and selection rule as in main-text Figure 5.}", B + "end{figure}", B + "clearpage", ""]
parts.append(B + "end{document}")
OUT.write_text("\n".join(parts), encoding="utf8"); print("written", OUT)
