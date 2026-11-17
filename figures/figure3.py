"""Figure 3: recovery of the drawn planted signals (three-seed GraTICCS ensembles; filled = recovered, open = missed).
Reads results/planted_random_{contact,secreted}_detail.csv from analysis.recovery.

python -m figures.figure3 [out.png]
"""
import sys
from pathlib import Path
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graticcs.paths import FIGURES, RESULTS

R = RESULTS
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES / "fig3_random_recovery.png"
plt.rcParams.update({"font.family": "Arial", "font.size": 9.5, "pdf.fonttype": 42, "svg.fonttype": "none"})
INK, INK2, MUTE, LRC = "#000000", "#333333", "#555555", "#2166ac"
BETAS = [0.05, 0.1, 0.2, 0.4]; COLS = ["0.25%", "1%", "4%", "16%"]
short = lambda c: c.replace("->", "→").replace("_", "/")

panels = [("A", "Contact pairs, breast", pd.read_csv(R / "planted_random_contact_detail.csv")),
          ("B", "Secreted pairs, breast", pd.read_csv(R / "planted_random_secreted_detail.csv"))]
nmax = max(d[d.model == "GraTICCS"].channel.nunique() for *_, d in panels)
fig = plt.figure(figsize=(7.2, 0.17 * nmax + 1.0))
for k, (letter, title, d) in enumerate(panels):
    d = d[d.model == "GraTICCS"]; chans = list(dict.fromkeys(d.channel))
    x0 = 0.03 + 0.5 * k
    fig.text(x0 - 0.025, 0.945, letter, fontsize=10, fontweight="bold", va="bottom")
    fig.text(x0, 0.95, title, fontsize=7.6, va="bottom")
    ax = fig.add_axes([x0 + 0.22, 0.045, 0.2, 0.89]); ax.axis("off")
    ax.set_xlim(-0.6, 3.6); ax.set_ylim(nmax + 0.1, -2.3)
    for j, c in enumerate(BETAS):
        ax.text(j, -1.9, COLS[j], fontsize=6.4, ha="center", va="center", color=INK2)
        ax.text(j, -0.9, f"{int(d[d.beta == c].recovered.sum())}/{len(chans)}", fontsize=6.6, color=LRC, fontweight="bold", ha="center", va="center")
    ax.text(-0.75, -0.9, "recovered", fontsize=6.6, fontweight="bold", ha="right", va="center")
    for r, c in enumerate(chans):
        row = d[d.channel == c]; ro = row.readout.iloc[0].replace("IL-1b", "IL-1β")
        ax.text(-0.75, r + 0.4, f"{short(c)}  ({ro})", fontsize=6.0, ha="right", va="center", color=INK2)
        for j, b in enumerate(BETAS):
            f = bool(row[row.beta == b].recovered.iloc[0])
            ax.scatter([j], [r + 0.4], s=15, c=LRC if f else "white", edgecolors=LRC, linewidths=0.8, zorder=3)
fig.text(0.5, 0.008, "● recovered   ○ missed   columns: variance of the added signal relative to the original readout; "
         "readout in parentheses", fontsize=6.0, color=MUTE, ha="center")
for ext in ("png", "pdf"):
    fig.savefig(OUT.with_suffix("." + ext), dpi=300, facecolor="white")
print("saved", OUT)
