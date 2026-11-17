"""Case figure in two panels:
  <out>_A.png  one predicted event in the tissue; cells filled with immunofluorescence, ligand and receptor drawn in the
               measurement used by the model (protein markers or RNA transcripts)
  <out>_B.png  the dose-response comparison of analysis.dose_response for the ligand and its same-receptor decoys
The event shown is, among the channel's 50 best-scored events, the one with the most cells in its display window.

python -m figures.case_figure <events.csv> <channel> <out_prefix> <contact|secreted> <dose_response.csv>
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from graticcs.lr_map import GENE2PROT
from graticcs.paths import BREAST, RAW, RESULTS
D, R, RAW = BREAST, RESULTS, RAW / "cosmx_breast"
ev_path, chn, outp, cls, dr_path = sys.argv[1:6]
UM = 0.5119 / 4256 * 1000
RED, NAVY, GREY, BAND = "#e8412c", "#3C5488", "#b0b0b0", "#3a3a3a"
GOLD = "#f5c518"
BGC, FG, EDGE, CONT, AXBG = "#000000", "#f2f2f2", "#3a3a3a", "#ffffff", "#0b0c10"
import os
from matplotlib.colors import LinearSegmentedColormap
_cm = os.environ.get("CMAP_NAME", "viridis")
CMAP = LinearSegmentedColormap.from_list("navyblues", ["#102a4c", "#1f4e8c", "#2171b5", "#4292c6", "#6baed6"])
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.linewidth": 0.8, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "axes.edgecolor": FG, "axes.facecolor": AXBG, "figure.facecolor": BGC})
lig_genes, rec_genes = chn.split("->")[0].split("_"), chn.split("->")[1].split("_")
LNAME = {"IL1B": "IL-1β", "LGALS9": "Galectin-9"}.get(lig_genes[0], lig_genes[0])
cells = pd.read_parquet(D / "cells.parquet"); n = len(cells); idx = pd.Series(np.arange(n), index=cells.cell_id)
CX, CY = cells.CenterX_global_px.to_numpy(), cells.CenterY_global_px.to_numpy()
z = np.load(D / "gt_inputs.npz"); types = json.load(open(D / "gt_channels.json"))["types"]; ctype = z["ctype"]
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: i for i, g in enumerate(genes)}
X = sp.load_npz(D / "rna_counts.npz").tocsr()
# ---------- event: densest display window among the 50 best-ranked events
ev = pd.read_csv(ev_path); ev = ev[ev.channel == chn].sort_values("rank_all")
def window(a, b):
    mx, my = (CX[a] + CX[b]) / 2, (CY[a] + CY[b]) / 2; return mx, my, max(20 / UM, 1.7 * np.hypot(CX[a] - CX[b], CY[a] - CY[b]))
def ncell(e):
    mx, my, h = window(idx[e.sender], idx[e.receiver]); return int(((abs(CX - mx) < h) & (abs(CY - my) < h)).sum())
top = ev.head(50).copy(); top["ncell"] = [ncell(e) for _, e in top.iterrows()]
cs = top.sort_values(["ncell", "rank_all"], ascending=[False, True]).iloc[0]
si, ri, fov = idx[cs.sender], idx[cs.receiver], int(cs.fov); cx, cy, half = window(si, ri)
csrc, cdst = z["src"][z["contact"] > 0], z["dst"][z["contact"] > 0]; nb = np.unique(csrc[cdst == ri])
pcol_of = lambda g: (f"P_{GENE2PROT[g]}" if GENE2PROT.get(g) and f"P_{GENE2PROT[g]}" in cells.columns else None)
pct_all = lambda col: pd.Series(cells[col].to_numpy(float)).rank(pct=True).to_numpy() * 100
RO = cs.best_readout; ro_pct = pct_all("P_" + RO)
lig_prot = [pcol_of(g) for g in lig_genes]; lig_is_prot = all(p is not None for p in lig_prot)
rec_prot = [pcol_of(g) for g in rec_genes]; rec_is_prot = all(p is not None for p in rec_prot)
PNAME = {"IL-1b": "IL-1β", "Tim-3": "TIM-3"}
TNAME = lambda t: "unassigned type" if t == "other" else (t + " cell" if t in ("T", "B") else t)
ordinal = lambda v: f"{int(round(v))}" + ("th" if 10 <= int(round(v)) % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(int(round(v)) % 10, "th"))
# ---------- panel A
poly = pd.read_csv(RAW / "BreastCancer-polygons.csv"); poly = poly[poly.fov == fov]
pcol = "cellID" if "cellID" in poly.columns else "cell_ID"
tx = pd.read_parquet(D / "tx_by_fov" / f"fov{fov:03d}.parquet"); tx = tx[(abs(tx.x_global_px - cx) < half) & (abs(tx.y_global_px - cy) < half)]
fig = plt.figure(figsize=(6.2, 6.6)); ax = fig.add_axes([0.05, 0.22, 0.92, 0.75])
fig.text(0.01, 0.985, "A", fontsize=12, fontweight="bold", va="top", color=FG); ax.set_facecolor(AXBG)
def _rng(col):
    v = cells[col].to_numpy(float); lo, hi = np.percentile(v, [2, 99.5]); return np.clip((v - lo) / (hi - lo), 0, 1)
IF = np.c_[_rng("Mean.CD45"), _rng("Mean.PanCK"), _rng("Mean.DAPI")] ** 0.6
shown = []
for cell, g in poly.groupby(pcol):
    xy = g[["x_global_px", "y_global_px"]].to_numpy()
    if (abs(xy[:, 0].mean() - cx) > half * 1.2) or (abs(xy[:, 1].mean() - cy) > half * 1.2): continue
    key = f"c_1_{fov}_{int(cell)}"
    if key not in idx.index: continue
    k = idx[key]; shown.append(k); ec, lw, ls, zz = EDGE, 0.5, "-", 1
    fa = 0.62
    if k == si: ec, lw, zz, fa = GOLD, 2.6, 4, 0.85
    elif k == ri: ec, lw, zz, fa = RED, 2.6, 4, 0.85
    elif k in nb: ec, lw, ls, zz = CONT, 1.0, (0, (3, 2)), 2
    ax.add_patch(Polygon(xy, closed=True, facecolor=(*IF[k], fa), edgecolor=ec, linewidth=lw, linestyle=ls, zorder=zz))
shown = np.array(shown)
cell_of_tx = tx.cell_ID.map(lambda c: idx.get(f"c_1_{fov}_{int(c)}", -1))
def plot_tx(mask, marker, fc, size, zz):
    t = tx[mask]; ax.scatter(t.x_global_px, t.y_global_px, marker=marker, s=size, c=fc, edgecolors="#000000", linewidths=0.6, zorder=zz); return int(mask.sum())
leg = []
if lig_is_prot:
    lp = np.mean([pct_all(p) for p in lig_prot], 0); hi = shown[lp[shown] >= 75]
    ax.scatter(CX[hi], CY[hi], marker="D", s=(lp[hi] - 70) * 3.0, c=GOLD, edgecolors="#000000", linewidths=0.8, zorder=6)
    leg.append(Line2D([], [], marker="D", ls="", mfc=GOLD, mec="k", ms=6, label=f"{LNAME} protein at or above the 75th percentile (larger = higher)"))
else:
    nl = plot_tx(tx.target.str.upper().isin(lig_genes) & (cell_of_tx >= 0), "^", GOLD, 26, 6)
    leg.append(Line2D([], [], marker="^", ls="", mfc=GOLD, mec="k", ms=7, label=f"{'+'.join(lig_genes)} RNA transcripts (n={nl})"))
if not rec_is_prot:
    nr = plot_tx(tx.target.str.upper().isin(rec_genes) & (cell_of_tx == ri), "s", RED, 26, 6)
    leg.append(Line2D([], [], marker="s", ls="", mfc=RED, mec="k", ms=6, label=f"{'+'.join(rec_genes)} RNA transcripts in receiver (n={nr})"))
leg += [Line2D([], [], color=GOLD, lw=2.4, label="Sender"), Line2D([], [], color=RED, lw=2.4, label="Receiver"),
        Line2D([], [], color=CONT, lw=1.0, ls=(0, (3, 2)), label="Cells contacting the receiver")]
ax.add_patch(FancyArrowPatch((CX[si], CY[si]), (CX[ri], CY[ri]), arrowstyle="-|>", mutation_scale=16, color="#ffffff", lw=1.8, shrinkA=10, shrinkB=10, zorder=7))
def pct_of(col, k):
    v = cells[col].to_numpy(float); return (v < v[k]).mean() * 100
def feat(gl, k):
    return [f"{PNAME.get(GENE2PROT[g], GENE2PROT[g])} protein: {ordinal(pct_of(pcol_of(g), k))} pct." if pcol_of(g) else (lambda v: f"{g} RNA: {v} transcript" + ("" if v == 1 else "s"))(int(X[k, gi[g]]) if g in gi else 0) for g in gl]
def box(x, y, title, lines, col):
    ax.text(x, y, "\n".join([title] + lines), color=col, fontsize=7.5, fontweight="bold", ha="center", va="center", zorder=8, linespacing=1.25,
            bbox=dict(boxstyle="round,pad=0.35", fc="#000000", ec=col, lw=1.4))
off = half * 0.62; sx, sy, rx, ry = CX[si], CY[si], CX[ri], CY[ri]
box(sx + (sx - rx) * 0.2, sy + np.sign(sy - ry + 1e-9) * off, "Sender", [TNAME(types[ctype[si]])] + feat(lig_genes, si), GOLD)
box(rx + (rx - sx) * 0.2, ry + np.sign(ry - sy + 1e-9) * off, "Receiver", [TNAME(types[ctype[ri]])] + feat(rec_genes, ri) + [f"{RO} protein: {ordinal(pct_of('P_' + RO, ri))} pct."], RED)
ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
x0, y0 = cx - half * 0.93, cy - half * 0.93
ax.plot([x0, x0 + 10 / UM], [y0, y0], color="#ffffff", lw=2.5, zorder=9, solid_capstyle="butt")
ax.text(x0 + 5 / UM, y0 + half * 0.04, "10 µm", color=FG, ha="center", va="bottom", fontsize=7.5, zorder=9,
        bbox=dict(boxstyle="square,pad=0.1", fc="#000000", ec="none"))
ax.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.0, -0.015), fontsize=7.2, frameon=False, handlelength=1.8, labelcolor=FG)
fig.text(0.66, 0.135, "Cell fill: immunofluorescence\n(red CD45, green PanCK, blue DAPI)", fontsize=7.2, color=FG, va="top")
fig.savefig(outp + "_A.png", dpi=300, facecolor=BGC, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
# ---------- panel B: dose-response
if cls == "contact":
    chdf = pd.DataFrame(json.load(open(D / "gt_channels.json"))["channels"]); ci = z["contact"] > 0
    Lm, Rm, Y = z["L_mo"], z["R_mo"], np.c_[z["Yp"], z["Yr"]].astype(np.float64)
    W = sp.csr_matrix((np.ones(ci.sum()), (z["dst"][ci], z["src"][ci])), shape=(n, n)); T = json.load(open(R / "G_real_s0" / "fold0_mo.json"))["targets"]
else:
    S = np.load(D / "gt_sec_inputs.npz"); chdf = pd.DataFrame(json.load(open(D / "gt_sec_channels.json"))["channels"])
    Lm, Rm, Y = S["L"], S["R"], np.c_[S["Yp"], S["Yr"]].astype(np.float64)
    W = sp.csr_matrix((np.exp(-S["dist"] / 15.0).astype(np.float64), (S["dst"], S["src"])), shape=(n, n)); T = json.load(open(R / "G2_sec_real_s0" / "fold0_mo.json"))["targets"]
dens = np.asarray(W.sum(1)).ravel(); fovs = cells.fov.to_numpy(); t = T.index(RO)
LMOD = "protein" if lig_is_prot else "RNA"
qbin = lambda x, k: np.searchsorted(np.unique(np.quantile(x, np.linspace(0, 1, k + 1)[1:-1])), x, side="right")
def curve(c):
    m = np.flatnonzero(Rm[:, c] > 0)
    g = pd.factorize(pd.Series(fovs[m]).astype(str) + "_" + pd.Series(ctype[m]).astype(str) + "_" + pd.Series(qbin(dens[m], 5)).astype(str)
                     + "_" + pd.Series(qbin(Rm[m, c], 10)).astype(str))[0]
    df = pd.DataFrame(dict(g=g, e=W[m] @ np.maximum(Lm[:, c], 0), y=Y[m, t])); df = df[df.groupby("g").g.transform("size") >= 5]
    df["ue"] = df.groupby("g").e.rank(pct=True); df["uy"] = df.groupby("g").y.rank(pct=True) - 0.5
    df["bin"] = np.minimum((df.ue * 10).astype(int), 9); return df.groupby("bin").uy.agg(["mean", "sem"]), len(df)
creal = int(chdf.index[(chdf.name == chn) & ~chdf.decoy][0]); decs = list(chdf.index[chdf.decoy & (chdf.of == chn)])
res = pd.read_csv(dr_path).set_index("channel").loc[chn]
dc = pd.concat([curve(int(d))[0]["mean"].rename(i) for i, d in enumerate(decs)], axis=1)
s_, nrec = curve(creal)
fig, ax = plt.subplots(figsize=(3.4, 3.2)); fig.subplots_adjust(left=0.22, right=0.97, top=0.93, bottom=0.17)
dc = dc * 100; s_ = s_ * 100
ax.fill_between(dc.index + 1, dc.min(axis=1), dc.max(axis=1), color=BAND, alpha=0.9, lw=0, label=f"Decoy ligands, range ({len(decs)})")
ax.plot(dc.index + 1, dc.mean(axis=1), color=GREY, lw=1.4, ls="--", label="Decoy ligands, mean")
ax.errorbar(s_.index + 1, s_["mean"], yerr=s_["sem"], color=RED, lw=2.0, marker="o", ms=3.8, capsize=2, mec="#000000", mew=0.5, label=f"{LNAME} (real ligand)")
ax.axhline(0, color=FG, lw=0.6, ls=":")
ax.set_xticks(range(1, 11)); ax.set_xlim(0.5, 10.5)
ax.set_xlabel(f"Neighboring {LNAME} {LMOD}\n(decile within matched group, low to high)"); ax.set_ylabel(f"Receiver {RO} protein\n(percentile points above matched-group median)")
fig.text(0.01, 0.985, "B", fontsize=12, fontweight="bold", va="top", color=FG)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=7, frameon=False, labelcolor=FG)
ax.text(0.5, -0.27, f"rank correlation ρ = {res.rho:.3f}, q = {res.q:.3f}\ndecoy mean ρ = {res.decoy_mean:.3f}\n{'+'.join(rec_genes)}+ receivers, n = {nrec:,}",
        transform=ax.transAxes, ha="center", va="top", fontsize=7)
fig.savefig(outp + "_B.png", dpi=300, facecolor=BGC, bbox_inches="tight", pad_inches=0.05); plt.close(fig)
json.dump(dict(channel=chn, sender=cs.sender, receiver=cs.receiver, sender_type=types[ctype[si]], receiver_type=types[ctype[ri]], fov=fov,
               rank_shown=int(cs.rank_all), best_rank=int(ev.rank_all.min()), n_events=int(cs.n_events), dist_um=float(cs.dist_um),
               receiver_readout_pct=float(pct_of("P_" + RO, ri)), ligand_protein=bool(lig_is_prot), receptor_protein=bool(rec_is_prot), n_receivers=nrec),
          open(outp + ".json", "w"), indent=1)
print("saved", outp)
