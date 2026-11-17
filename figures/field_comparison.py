"""Supplementary figure: prediction error of GraTICCS relative to the matched local model in each of the 108 held-out
imaging fields, 100 x (SSE GraTICCS - SSE local) / SSE local; bars = IQR, diamond = median. Reads analysis.per_field output.

python -m figures.field_comparison [out.png]
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graticcs.paths import FIGURES, RESULTS
R = RESULTS
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else FIGURES / "field_comparison.png"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "pdf.fonttype": 42})
d = pd.read_csv(R / "s19_per_fov_crand.csv"); s = pd.read_csv(R / "s19_summary_crand.csv")
order = list(s.setting)
lab = {"planted, beta = 0.05": "planted, 0.25%", "planted, beta = 0.1": "planted, 1%", "planted, beta = 0.2": "planted, 4%",
       "planted, beta = 0.4": "planted, 16%", "binary gate, beta = 0.2": "binary rule, 4%", "original tissue": "original tissue"}
fig, ax = plt.subplots(figsize=(5.6, 2.6)); rng = np.random.default_rng(0)
for k, st in enumerate(order[::-1]):
    v = 100 * d.loc[d.setting == st, "rel_diff"].to_numpy(); y = k + rng.uniform(-0.22, 0.22, len(v))
    ax.scatter(v, y, s=5, c="#2166ac", alpha=0.45, linewidths=0, zorder=2)
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    ax.plot([q1, q3], [k, k], color="#000000", lw=1.6, zorder=3); ax.scatter([q2], [k], marker="D", s=18, c="white", edgecolors="#000000", linewidths=0.8, zorder=4)
    r = s.set_index("setting").loc[st]
    ax.text(1.02, k, f"{int(r.fov_gt_better)}/108 lower,  p = {r.wilcoxon_p:.1g}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.2, color="#333333")
ax.axvline(0, color="#000000", lw=0.7, ls=(0, (3, 2)), zorder=1)
ax.set_yticks(range(len(order)), [lab[o] for o in order[::-1]]); ax.tick_params(axis="y", length=0)
ax.set_xlabel("difference in prediction error, GraTICCS vs local model (%)")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
lo, hi = np.percentile(100 * d.rel_diff, [0.5, 99.5]); ax.set_xlim(min(lo, -0.5) * 1.1, max(hi, 0.5) * 1.1)
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04); fig.savefig(OUT.with_suffix(".pdf"), facecolor="white", bbox_inches="tight", pad_inches=0.04)
print("saved", OUT)
