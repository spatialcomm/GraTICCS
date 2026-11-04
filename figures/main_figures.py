"""Figures 2 (design iterations) and 4 (pair-level AP)."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
from graticcs.paths import RESULTS, FIGURES
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = RESULTS; F = FIGURES
LRC, LEAK, GREY, INK, INK2 = "#2166ac", "#542788", "#bdbdbd", "#000000", "#333333"
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "lines.solid_capstyle": "butt", "svg.fonttype": "none", "pdf.fonttype": 42})


def bars(ax, labels, vals, colors, fmt="{:.3f}", log=False):
    x = np.arange(len(vals)); ax.bar(x, vals, color=colors, width=0.68)
    for xi, v in zip(x, vals):
        ax.text(xi, v * (1.15 if log else 1) + (0 if log else max(vals) * 0.02), fmt.format(v), ha="center", va="bottom", color=INK, fontsize=7)
    ax.set_xticks(x, labels, rotation=0, fontsize=7)
    if log: ax.set_yscale("log")


def save(fig, name):
    for ext in ("png", "pdf", "svg"): fig.savefig(F / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)


steps = [("v1\nreceptor =\nreadout", 4.42, LEAK), ("v1 fixed", 0.05, GREY), ("v2\nreceiver\nreceptor RNA", 11.94, LEAK),
         ("v2\nembedding\nattention", 5.97, LEAK), ("v3\nRNA decoys\nonly", 0.37, LEAK), ("v3b\nlocal\nattention", 0.0007, GREY),
         ("final\ngraph\ntransformer", 0.59, LRC)]
fig, ax = plt.subplots(figsize=(6.6, 2.7))
bars(ax, [s[0] for s in steps], [s[1] for s in steps], [s[2] for s in steps], fmt="{:.2f}%")
ax.texts[5].set_text("0.0007%")
ax.set_ylabel("held-out gain of the LR branch (%)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=LEAK, label="gain traced to a leak (matched by decoys)"), Patch(color=GREY, label="intermediate design, no leak"),
                   Patch(color=LRC, label="final model: gain matched by decoys, no real channel with q ≤ 0.1")],
          frameon=False, fontsize=7, loc="upper right")
fig.tight_layout(); save(fig, "fig2_leak_audit")

fp = pd.read_csv(R / "final_pair_level_comparison_GT.csv"); hy = pd.read_csv(R / "hybrid_pair_level_GT.csv")
p = fp[fp.truth == "protein-defined truth"]
labs = {"GraTICCS-GT (3-seed)": "GraTICCS", "co-expression_(L_x_R)": "co-expression", "COMMOT": "COMMOT",
        "CellPhoneDB_(LIANA,_type_level)": "CellPhoneDB", "LIANA+_bivariate_cosine": "LIANA+\nbivariate", "co-expression_(RNA_only)": "RNA\nco-expr."}
rows = [("GraTICCS-GT (3-seed)", p[(p.method == "GraTICCS-GT (3-seed)") & (p.beta.astype(str) == "0.1")].AP.iloc[0])]
rows += [(m, p[p.method == m].AP.iloc[0]) for m in ["co-expression_(L_x_R)", "COMMOT", "CellPhoneDB_(LIANA,_type_level)", "LIANA+_bivariate_cosine", "co-expression_(RNA_only)"]]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
bars(axs[0], [labs[r[0]] for r in rows], [r[1] for r in rows], [LRC] + [GREY] * 5, log=True)
axs[0].axhline(0.0028, color=INK2, lw=0.8, ls=(0, (3, 2))); axs[0].text(0.99, 0.97, "dashed line: random (0.0028)", transform=axs[0].transAxes, fontsize=6.5, color=INK2, ha="right", va="top")
axs[0].set_ylim(top=0.6)
axs[0].set_ylabel("pair-level average precision (log)"); axs[0].set_title("A   protein-defined planted truth", loc="left", fontsize=8, fontweight="bold")
h = hy[(hy.truth == "RNA-defined") & (hy.beta == 0.1)].set_index("method").AP
hv = [("RNA\nco-expr.", h["RNA co-expression alone"], GREY), ("× GraTICCS\n1 − q", h["GraTICCS (1 - q) x RNA co-expression"], LRC),
      ("COMMOT", h["COMMOT alone"], GREY), ("× GraTICCS\n1 − q", h["GraTICCS (1 - q) x COMMOT"], LRC)]
bars(axs[1], [v[0] for v in hv], [v[1] for v in hv], [v[2] for v in hv])
axs[1].set_ylabel("pair-level average precision"); axs[1].set_ylim(0, 0.185); axs[1].set_title("B   RNA-defined planted truth", loc="left", fontsize=8, fontweight="bold")
fig.tight_layout(); save(fig, "fig4_pair_level")
