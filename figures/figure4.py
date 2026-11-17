"""Figure 4: ranking contributing against non-contributing senders among neighbours that express the ligand, in the four
selective-sender tests (beta = 0.2, three seeds). Metric: AP among coexpressing events of the planted channels; the dashed
line is the fraction of contributing events. Reads the outputs of analysis.pair_level and analysis.secreted_pairs.

python -m figures.figure4 [out.png]
"""
import sys
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graticcs.paths import FIGURES, RESULTS

R = RESULTS
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES / "fig4_selective_senders.png"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, INK2, MUTE, GT, LOC, RULE, RULE2, CO = "#000000", "#333333", "#666666", "#2166ac", "#7f7f7f", "#a6a6a6", "#c8c8c8", "#dcdcdc"

ct = pd.read_csv(R / "ctype_pair_level.csv").set_index("method").AP_within_coexpressing
ca = pd.read_csv(R / "carea_pair_level.csv").set_index("method").AP_within_coexpressing
sd = pd.read_csv(R / "sec_pair_sec_rand.csv").set_index("method").AP_within_coexpressing
st = pd.read_csv(R / "sec_pair_sec_ctype.csv").set_index("method").AP_within_coexpressing
ROWS = ["GraTICCS", "local model (equal weights)", "local + sender-type rule", "local + spatial rule", "co-expression"]
COL = [GT, LOC, RULE, RULE2, CO]
panels = [
    ("A", "Contact: one sender cell type contributes", 0.261, "sender-type rule", "contact-extent rule",
     [ct["GraTICCS (3-seed)"], ct["local model (3-seed)"], ct["local + sender type (3-seed)"], ct["weighted by contact amount (3-seed)"], ct["co-expression (L x R)"]]),
    ("B", "Contact: contribution grows with shared boundary", 0.513, "sender-type rule", "contact-extent rule",
     [ca["GraTICCS (3-seed)"], ca["local model (3-seed)"], ca["local + sender type (3-seed)"], ca["weighted by contact amount (3-seed)"], ca["co-expression (L x R)"]]),
    ("C", "Secreted: contribution decays with distance", 0.500, "sender-type rule", "distance rule",
     [sd["GraTICCS (3-seed)"], sd["local model (3-seed)"], sd["local + sender type (3-seed)"], sd["local + distance kernel (3-seed)"], sd["co-expression (L x R)"]]),
    ("D", "Secreted: one sender cell type contributes", 0.363, "sender-type rule", "distance rule",
     [st["GraTICCS (3-seed)"], st["local model (3-seed)"], st["local + sender type (3-seed)"], st["local + distance kernel (3-seed)"], st["co-expression (L x R)"]]),
]
fig = plt.figure(figsize=(7.2, 3.5))
W, H, X0, Y0, DX, DY = 0.255, 0.30, 0.215, 0.565, 0.49, 0.455
for k, (letter, title, chance, r1, r2, vals) in enumerate(panels):
    x, y = X0 + DX * (k % 2), Y0 - DY * (k // 2)
    ax = fig.add_axes([x, y, W, H])
    labels = ["GraTICCS", "local (equal weights)", "local + " + r1, "local + " + r2, "co-expression"]
    yy = range(len(vals))[::-1]
    ax.barh(list(yy), vals, height=0.62, color=COL, edgecolor="white", linewidth=0.6, zorder=2)
    for yi, v, c in zip(yy, vals, COL):
        xt = v + 0.012 if abs(v - chance) > 0.035 or v > chance else chance + 0.012   # keep labels clear of the chance line
        ax.text(xt, yi, f"{v:.2f}", va="center", ha="left", fontsize=6.2, color=INK if c == GT else INK2,
                fontweight="bold" if c == GT else "normal", zorder=3)
    ax.axvline(chance, color=INK, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(chance, len(vals) - 0.35, "chance", fontsize=5.6, color=MUTE, ha="center", va="bottom")
    ax.set_yticks(list(yy), labels, fontsize=6.4)
    for tl in ax.get_yticklabels():
        if tl.get_text() == "GraTICCS": tl.set_fontweight("bold"); tl.set_color(GT)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(labelsize=6, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(0.6); ax.spines["bottom"].set_linewidth(0.6)
    ax.set_xlabel("AP among neighbours that express the ligand", fontsize=6.4, labelpad=1.5)
    fig.text(x - 0.20, y + H + 0.035, letter, fontsize=10, fontweight="bold", va="bottom")
    fig.text(x - 0.175, y + H + 0.04, title, fontsize=7.4, va="bottom")
OUT.parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix("." + ext), dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
