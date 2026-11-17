"""Figure 1: GraTICCS workflow (A input, B target and controls, C model, D the two tasks, E downstream analysis).
Panel E follows the planted HLA-E->KLRK1 channel of the contact draw (beta = 0.1, three seeds); run figures.figure1_data first.

python -m figures.figure1 [out.png]
"""
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection, PathCollection
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon
from matplotlib.transforms import Bbox
from scipy.stats import pearsonr

from graticcs.paths import BREAST, FIGURES, RAW, RESULTS

D, R, RAW = BREAST, RESULTS, RAW / "cosmx_breast"
OUT = sys.argv[1] if len(sys.argv) > 1 else str(FIGURES / "fig0_pipeline_overview.png")
UM = 0.1203; FW, FH = 11.8, 14.5
plt.rcParams.update({"lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt", "font.family": "Arial", "font.size": 9.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
INK, INK2, MUTE, RULE = "#000000", "#333333", "#555555", "#bdbdbd"
NUIS, LIGHT = "#3a3a3a", "#dedede"           # background branch: charcoal boxes, light grey sublayers
LRC = "#2166ac"                              # LR branch, use 1 (communication)
CAL = "#542788"                              # responses, planted signals, use 2 (downstream response)
TYPES = json.load(open(D / "gt_channels.json"))["types"]
# matplotlib tab10 / ColorBrewer Paired; checked for normal and red-green colour-blind separation (all epithelial pairs dE >= 20)
TCOL = {"epithelial": "#d62728", "T": "#1f77b4", "B": "#6a3d9a", "myeloid": "#e377c2", "stromal": "#bcbd22",
        "endothelial": "#b2df8a", "other": "#c7c7c7"}
tcol = np.array([matplotlib.colors.to_rgb(TCOL[t]) for t in TYPES])
DIV = plt.get_cmap("RdBu_r")
W1 = np.load(R / "fig1_walkthrough_crand.npz", allow_pickle=False); LR1 = str(W1["LR1"]); pretty = lambda c: c.replace("->", " → ").replace("_", "/")
short = lambda c: pretty(c).replace(" → ", "→")

# ------------------------------------------------------------------ data
z = np.load(D / "gt_inputs.npz"); fov, xy, ctype, fold = z["fov"], z["xy"], z["ctype"], z["fold"]
src, dst, con, dist = z["src"], z["dst"], z["contact"], z["dist"]
cells = pd.read_parquet(D / "cells.parquet"); cid = cells.cell_id.to_numpy()
ch = pd.DataFrame(json.load(open(D / "gt_channels.json"))["channels"])
FOV_A = 64
ci, csrc, cdst, sc = W1["ci"], W1["csrc"], W1["cdst"], W1["sc"]
FOV_E = int(pd.Series(fov[cdst[np.argsort(-sc)[:200]]]).value_counts().index[0])        # field of view with most top-scoring LR1 pairs
poly = pd.read_csv(RAW / "BreastCancer-polygons.csv", usecols=["fov", "cell", "x_global_px", "y_global_px"])
poly = poly[poly.fov.isin([FOV_A, FOV_E])]
verts = {c: g[["x_global_px", "y_global_px"]].to_numpy() * UM for c, g in poly.groupby("cell", sort=False)}
# planted truth and model prediction of LR1's effect on its response, per receiver
c_pl = list(ch.name).index(LR1); prot = json.load(open(D / "gt_channels.json"))["prot_readout"]; tnames = prot + json.load(open(D / "gt_channels.json"))["rna_programs"]
lr1_t = str(W1["target"]); Lp = np.maximum(z["L_mo"][:, c_pl], 0); Rp = np.maximum(z["R_mo"][:, c_pl], 0); cc_ = con > 0
expo = Rp * np.bincount(dst[cc_], weights=Lp[src[cc_]], minlength=len(fov)); ze = (expo - expo.mean()) / expo.std()
y0_ = np.c_[z["Yp"], z["Yr"]][:, tnames.index(lr1_t)]
pred = np.bincount(cdst, weights=sc, minlength=len(fov)); rc_all = np.unique(cdst)                # receivers of the 12 evaluation FOVs
true_eff = 0.1 * y0_.std() * ze                                                                   # planted increment at beta = 0.1
lr1_t = lr1_t.replace("IL-1b", "IL-1β")                                                          # display name only


def draw_cells(ax, f, alpha=0.9, lw=0.25, fill=None):
    m = np.flatnonzero(fov == f); vs, fc = [], []
    for i in m:
        v = verts.get(cid[i])
        if v is None: continue
        vs.append(v); fc.append((*tcol[ctype[i]], alpha) if fill is None else fill(i))
    ax.add_collection(PolyCollection(vs, facecolors=fc, edgecolors="#ffffff", linewidths=lw))
    return m


def clean(ax, ls=6.8):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=ls); ax.xaxis.label.set_color(INK2); ax.yaxis.label.set_color(INK2)


def tiles(ax, M, cmap, vmin, vmax, lw=0.5):
    """matrix drawn as tiles separated by white gaps (row 0 at top)"""
    q = ax.pcolormesh(np.ma.masked_invalid(M), cmap=cmap, vmin=vmin, vmax=vmax, edgecolors="white", linewidth=lw)
    ax.set_xlim(0, M.shape[1]); ax.set_ylim(M.shape[0], 0); return q


fig = plt.figure(figsize=(FW, FH), dpi=100); fig.patch.set_facecolor("white")
AR = FW / FH


def box(x0, y0, x1, y1, letter, title, ec="#6b6b6b", tc=INK, lw=0.9):
    fig.add_artist(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.008", transform=fig.transFigure,
                                  fill=False, ec=ec, lw=lw, zorder=0, mutation_aspect=AR))
    fig.text(x0 + 0.006, y1 - 0.004, letter, fontsize=15.0, fontweight="bold", color=INK, va="top", ha="left")
    fig.text((x0 + x1) / 2, y1 - 0.007, title, fontsize=10.5, fontweight="bold", color=tc, va="top", ha="center")


def lab(x, y, text, ha="left", c=INK, fs=7.4, **kw):
    return fig.text(x, y, text, fontsize=fs, fontweight="bold", color=c, va="baseline", ha=ha, **kw)


def farrow(pts, c="#4d4d4d", lw=1.3, ms=11):
    xs, ys = zip(*pts)
    fig.add_artist(matplotlib.lines.Line2D(xs, ys, transform=fig.transFigure, color=c, lw=lw, zorder=10, solid_joinstyle="round"))
    fig.add_artist(FancyArrowPatch(pts[-2], pts[-1], transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms, color=c, lw=0, zorder=11, shrinkA=0, shrinkB=0))


ICONS = []


def icon_ax(rect, xl=(0, 10), yl=(0, 4)):
    a = fig.add_axes(rect); a.set_xlim(*xl); a.set_ylim(*yl); a.set_aspect("equal"); a.axis("off"); ICONS.append(a); return a


def receptor(a, x, y, c=INK, s=1.0):                    # Y-shaped receptor opening to the left
    a.plot([x, x - 0.5 * s], [y, y], color=c, lw=1.4); a.plot([x - 0.5 * s, x - 0.85 * s], [y, y + 0.35 * s], color=c, lw=1.4)
    a.plot([x - 0.5 * s, x - 0.85 * s], [y, y - 0.35 * s], color=c, lw=1.4)


# ================================================================== panel boxes
PA = (0.006, 0.735, 0.494, 0.948); PB = (0.506, 0.735, 0.994, 0.948)
PC = (0.006, 0.392, 0.430, 0.726); PD = (0.442, 0.392, 0.994, 0.726)
PE = (0.006, 0.200, 0.994, 0.383)
box(*PA, "A", "Spatial multi-omics input")
box(*PB, "B", "Prediction target and two controls")
box(*PC, "C", "Graph transformer learns LR → response")
box(*PD, "D", "Two tasks from the trained model")
box(*PE, "E", "Downstream analysis")

# ================================================================== A  input
X = sp.load_npz(D / "rna_counts.npz").tocsr(); genes = json.load(open(D / "genes.json"))["genes"]; lib = np.maximum(np.asarray(X.sum(1)).ravel(), 1)
mapax = fig.add_axes([0.158, 0.768, 0.145, 0.140]); m = draw_cells(mapax, FOV_A)
x0, y0 = xy[m].min(0); mapax.set_xlim(x0 - 5, x0 + 512); mapax.set_ylim(y0 - 5, y0 + 512); mapax.set_aspect("equal"); mapax.axis("off")
mapax.plot([x0 + 400, x0 + 500], [y0 - 14, y0 - 14], color=INK, lw=1.5, clip_on=False); mapax.text(x0 + 450, y0 - 52, "100 µm", ha="center", fontsize=7.2)
for t in TYPES: mapax.scatter([], [], s=16, marker="s", color=TCOL[t], label=t)
mapax.legend(loc="upper left", bbox_to_anchor=(-0.06, -0.06), ncol=4, fontsize=6.6, frameon=False, handletextpad=0.05, columnspacing=0.35, borderaxespad=0)
lab(0.158, 0.916, "tissue (CosMx breast)")
order_t = [TYPES.index(t) for t in ["epithelial", "T", "B", "myeloid", "stromal", "endothelial"]]
pick = np.concatenate([m[ctype[m] == k][np.argsort(-lib[m[ctype[m] == k]])[:4]] for k in order_t if (ctype[m] == k).any()])
show_g = ["KRT19", "KRT8", "EPCAM", "IGKC", "HLA-DRA", "LYZ", "C1QB", "COL1A1", "LUM", "SPARC", "PECAM1"]
Rm = np.log1p(X[pick][:, [genes.index(g) for g in show_g]].toarray() / lib[pick, None] * 1e3); Rm = Rm / (np.quantile(Rm, 0.98, axis=0) + 1e-6)
show_p = ["P_EpCAM", "P_CD3", "P_CD20", "P_CD68", "P_HLA-DR", "P_SMA", "P_Vimentin", "P_CD31"]
Pm = np.log1p(cells.loc[pick, show_p].to_numpy(float)); Pm = (Pm - Pm.min(0)) / (np.quantile(Pm, 0.98, axis=0) - Pm.min(0) + 1e-6)
HX, HW = 0.052, 0.070; nR, nP = len(show_g), len(show_p); H0, H1 = 0.770, 0.902; rh = (H1 - H0 - 0.007) / (nR + nP)
heatA = []
for M, labels, name, y, n in [(Rm, show_g, "RNA", H1 - nR * rh, nR), (Pm, [p[2:] for p in show_p], "protein", H0, nP)]:
    h = fig.add_axes([HX, y, HW, n * rh]); im = tiles(h, np.clip(M.T, 0, 1), "viridis", 0, 1, lw=0.4); heatA.append(h)
    h.set_yticks(np.arange(n) + 0.5, labels, fontsize=7.0, color=INK); h.set_xticks([])
    h.tick_params(length=0, pad=1.2); [s.set_visible(False) for s in h.spines.values()]
    h.text(1.05, 0.5, name, transform=h.transAxes, rotation=270, fontsize=7.2, color=INK, va="center", ha="left", fontweight="bold")
strip = fig.add_axes([HX, H1 + 0.003, HW, 0.005]); strip.imshow(tcol[ctype[pick]][None, :, :], aspect="auto"); strip.axis("off")
lab(0.014, 0.916, "same-cell RNA + protein")
cb = fig.add_axes([HX + 0.008, H0 - 0.013, HW - 0.016, 0.004]); fig.colorbar(im, cax=cb, orientation="horizontal")
cb.tick_params(labelsize=7.0, length=1.5, pad=1); cb.set_xticks([0, 1], ["low", "high"])

ce = np.flatnonzero((con > 0) & (fov[dst] == FOV_A)); deg = np.bincount(dst[ce], minlength=len(fov))
cand = np.flatnonzero((fov == FOV_A) & (deg >= 5) & (deg <= 8) & np.isin(ctype, [TYPES.index("T"), TYPES.index("myeloid")]))
ntype = np.array([len(np.unique(ctype[src[ce][dst[ce] == c]])) for c in cand]); fc_ = xy[fov == FOV_A].mean(0)
good = cand[ntype >= ntype.max() - 1]; RCV = int(good[int(np.argmin(np.linalg.norm(xy[good] - fc_, axis=1)))])
axB = fig.add_axes([0.335, 0.770, 0.145, 0.140]); draw_cells(axB, FOV_A, alpha=0.55)
cx, cy = xy[RCV]; W = 48; axB.set_xlim(cx - W, cx + W); axB.set_ylim(cy - W, cy + W); axB.set_aspect("equal")
for s_ in axB.spines.values(): s_.set_color(INK2); s_.set_linewidth(0.8)
axB.set_xticks([]); axB.set_yticks([])
win = (fov == FOV_A) & (np.abs(xy[:, 0] - cx) < W) & (np.abs(xy[:, 1] - cy) < W)
cw = ce[win[src[ce]] & win[dst[ce]]]
axB.add_collection(LineCollection([[xy[s], xy[d]] for s, d in zip(src[cw], dst[cw])], colors=INK2, linewidths=0.5, alpha=0.45, zorder=3))
S = np.load(D / "gt_sec_inputs.npz"); se = np.flatnonzero(S["dst"] == RCV)
axB.add_collection(LineCollection([[xy[s], xy[RCV]] for s in S["src"][se]], colors=LRC, linewidths=0.7, alpha=0.7, linestyles=(0, (2, 1.5)), zorder=3))
axB.add_patch(Circle((cx, cy), 30, fill=False, ec=LRC, lw=1.0, ls=(0, (4, 2)), zorder=4))
mine = ce[dst[ce] == RCV]
axB.add_collection(LineCollection([[xy[s], xy[RCV]] for s in src[mine]], colors=INK, linewidths=2.0, zorder=5))
axB.scatter(*xy[win].T, s=4, c=INK, zorder=4, linewidths=0)
axB.scatter([cx], [cy], s=50, c="white", edgecolors=INK, linewidths=1.3, zorder=6)
axB.text(cx + 3, cy + 6, "$i$", fontsize=8.0, fontweight="bold", zorder=7, bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
axB.text(cx + 21, cy + 22, "30 µm", fontsize=7.2, color=LRC, zorder=7)
axB.plot([cx + 22, cx + 42], [cy - W + 4, cy - W + 4], color=INK, lw=1.5, zorder=7); axB.text(cx + 32, cy - W + 6, "20 µm", ha="center", fontsize=7.2, zorder=7)
lab(0.335, 0.916, "neighbours of receiver $i$")
mapax.add_patch(Rectangle((cx - W, cy - W), 2 * W, 2 * W, fill=False, ec=INK, lw=0.9, zorder=8))
mapax.apply_aspect(); axB.apply_aspect(); inv = fig.transFigure.inverted()
cone = [inv.transform(mapax.transData.transform(q)) for q in [(cx + W, cy + W), (cx + W, cy - W)]]
cone = [cone[0], inv.transform(axB.transAxes.transform((0, 1))), inv.transform(axB.transAxes.transform((0, 0))), cone[1]]
fig.add_artist(Polygon(cone, closed=True, transform=fig.transFigure, fc="#7f7f7f", alpha=0.15, ec="none", zorder=0))   # zoom cone
axB.plot([], [], color=INK, lw=2.0, label="contact"); axB.plot([], [], color=LRC, lw=0.9, ls=(0, (2, 1.5)), label="secreted (≤ 30 µm)")
axB.legend(loc="upper left", bbox_to_anchor=(0.24, -0.015), fontsize=7.2, frameon=False, handlelength=2.0, borderaxespad=0, ncol=1)

# ================================================================== B  prediction target and the two controls
det = np.asarray((X > 0).mean(0)).ravel(); mexp = np.log10(np.asarray(X.mean(0)).ravel() + 1e-4)


def chip(x, y, text, c):
    fig.text(x, y, text, fontsize=7.2, fontweight="bold", color="white", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.22", fc=c, ec="none"))


# (1) target: the receiver response
txB = [[], [], []]                                                              # B texts, moved with their column
Yp = z["Yp"]; PW = np.load(D / "pathway_tf" / "Y_pathway_tf.npz")["Y"]; pwn = json.load(open(D / "pathway_tf" / "meta.json"))["readouts"]
mD = np.flatnonzero(fov == FOV_A); mD = mD[np.argsort(ctype[mD], kind="stable")]
sel = [(Yp, prot, ["GZMB", "Ki-67", "IDO1", "HLA-DR", "NF-kB p65", "CD38"], "protein"),
       (PW, pwn, ["PW_JAK-STAT", "PW_NFkB", "PW_TNFa", "PW_TGFb", "PW_MAPK"], "pathway"),
       (PW, pwn, ["TF_STAT1", "TF_IRF1", "TF_RELA", "TF_SMAD3", "TF_JUN"], "TF")]
blocks = [(Ysrc[mD][:, [nm.index(p) for p in ps]], [p.split("_")[-1] if "_" in p else p for p in ps], gl) for Ysrc, nm, ps, gl in sel]
M = np.concatenate([b[0] for b in blocks], 1); M = (M - M.mean(0)) / (M.std(0) + 1e-6)
DX, DY, DW, DH = 0.556, 0.778, 0.092, 0.122
axD = fig.add_axes([DX, DY, DW, DH])
axD.imshow(np.clip(M.T, -2.2, 2.2), aspect="auto", cmap=DIV, vmin=-2.2, vmax=2.2, interpolation="nearest")
labs = sum([b[1] for b in blocks], []); axD.set_yticks(range(len(labs)), labs, fontsize=6.4, color=INK2); axD.set_xticks([])
axD.tick_params(length=0, pad=1); [s.set_visible(False) for s in axD.spines.values()]
o = 0
for Mb, lb, nm in blocks:
    axD.axhline(o - 0.5, color="white", lw=1.4)
    txB[0].append(fig.text(0.511, DY + DH - (o + len(lb) / 2) / len(labs) * DH, nm, fontsize=7.2, color=INK, ha="center", va="center", fontweight="bold", rotation=90))
    o += len(lb)
st = fig.add_axes([DX, DY + DH + 0.002, DW, 0.004]); st.imshow(tcol[ctype[mD]][None, :, :], aspect="auto"); st.axis("off")
axD.set_xlabel("receiver cells", fontsize=7.2, color=MUTE, labelpad=2)
txB[0].append(lab(0.514, 0.916, "Target: receiver response $\\mathbf{y}_i$", c=INK))

# (2) negative control: decoy ligands
txB[1].append(lab(0.664, 0.916, "Negative control: decoys", c=INK))
nc = icon_ax([0.664, 0.852, 0.150, 0.056], (0, 15), (0, 5.6))
nc.add_patch(Circle((0.7, 4.5), 0.45, fc=LRC, ec=INK, lw=0.6)); nc.text(1.4, 4.5, "real ligand", va="center", fontsize=7.2)
for k_ in range(5): nc.add_patch(Circle((0.5 + 0.62 * k_, 1.4), 0.27, fc="white", ec=LRC, lw=0.8))
nc.text(3.6, 1.4, "decoys", va="center", fontsize=7.2, color=LRC)
nc.add_patch(FancyArrowPatch((5.6, 4.5), (8.2, 3.1), arrowstyle="-|>", mutation_scale=7, color=INK, lw=1.0))
nc.add_patch(FancyArrowPatch((5.6, 1.4), (8.2, 2.7), arrowstyle="-|>", mutation_scale=7, color=MUTE, lw=0.8, linestyle=(0, (2, 2))))
nc.text(6.9, 4.5, "binds", fontsize=7.2, color=INK, ha="center"); nc.text(6.6, 0.5, "not a known ligand", fontsize=6.4, color=MUTE, ha="center")
receptor(nc, 9.6, 2.9, s=1.4); nc.text(10.3, 4.6, "same\nreceptor", fontsize=7.2, va="center")
nc.text(10.3, 1.0, "Δ expected\nby chance", fontsize=7.2, color=LRC, va="center", ha="left")
axC = fig.add_axes([0.700, 0.778, 0.092, 0.058]); clean(axC, 6.0)
axC.scatter(mexp, det, s=0.5, c=RULE, linewidths=0, rasterized=True)
gidx = {g: i for i, g in enumerate(genes)}
for lg, c in [("HLA-DRA", "#6a3d9a"), ("LGALS9", "#1f77b4"), ("CD274", "#d62728")]:
    rows = ch[(ch.of == ch.loc[(~ch.decoy) & (ch.ligand == lg)].iloc[0].of) & ch.decoy]
    dg = [g for g in rows.ligand if g in gidx]
    axC.scatter(mexp[[gidx[g] for g in dg]], det[[gidx[g] for g in dg]], s=7, facecolors="white", edgecolors=c, linewidths=0.7, zorder=3)
    axC.scatter(mexp[gidx[lg]], det[gidx[lg]], s=14, c=c, edgecolors=INK, linewidths=0.4, zorder=4)
axC.set_xlabel("mean expression (log$_{10}$)", fontsize=7.2, labelpad=1); axC.set_ylabel("detection", fontsize=7.2, labelpad=1)
axC.set_yscale("log"); axC.set_xlim(-2.4, 0.6); axC.set_ylim(0.008, 0.9)
txB[1].append(fig.text(0.664, 0.840, "RNA decoys match each\nreal ligand's expression", fontsize=7.2, color=MUTE, ha="left", va="bottom"))

# (3) positive control: planted signals
txB[2].append(lab(0.834, 0.916, "Positive control: planted signals", c=INK))
pc = icon_ax([0.834, 0.858, 0.150, 0.050], (0, 15), (0, 5))


def tissue_icon(a, x0, hl=False):
    for r_ in range(3):
        for c_ in range(3):
            hot = hl and (r_, c_) in [(0, 1), (1, 2), (2, 0)]
            a.add_patch(Circle((x0 + 0.7 * c_ + (0.35 if r_ % 2 else 0), 1.2 + 0.75 * r_), 0.32, fc=CAL if hot else "#d0d0d0", ec="white", lw=0.4))


tissue_icon(pc, 0.3); pc.text(1.0, 0.2, "real data", fontsize=6.2, ha="center", color=MUTE)
pc.text(2.8, 2.0, "+", fontsize=10.0, ha="center", va="center")
pc.add_patch(Circle((3.7, 2.6), 0.22, fc=LRC, ec="none")); receptor(pc, 5.0, 2.6, s=0.8)
pc.add_patch(FancyArrowPatch((5.2, 2.6), (5.9, 2.6), arrowstyle="-|>", mutation_scale=6, color=INK, lw=0.8))
pc.add_patch(Rectangle((6.05, 1.6), 0.45, 2.0, fc=CAL, ec="none")); pc.text(5.0, 0.2, "known effect", fontsize=6.2, ha="center", color=MUTE)
pc.text(7.3, 2.0, "=", fontsize=10.0, ha="center", va="center")
tissue_icon(pc, 8.0, hl=True); pc.text(9.2, 0.2, "truth known", fontsize=6.2, ha="center", color=CAL)
pc.add_patch(FancyArrowPatch((10.6, 2.0), (11.6, 2.0), arrowstyle="-|>", mutation_scale=7, color=INK, lw=0.9))
pc.text(11.8, 2.0, "recovered", fontsize=7.0, va="center", fontweight="bold", color=LRC)
txB[2].append(fig.text(0.836, 0.846, "$y_{i,t}\\leftarrow y_{i,t}+\\beta\\,\\sigma_t\\,z\\!\\left(R_i\\sum_{j\\sim i}L_j\\right)$", fontsize=7.4, color=INK, va="top"))
y1_ = y0_ + 0.4 * y0_.std() * ze
sp_ax = fig.add_axes([0.880, 0.778, 0.095, 0.046]); clean(sp_ax, 6.0)
ex = expo > 0; qb = np.quantile(ze[ex], np.linspace(0, 1, 9)); bi = np.clip(np.searchsorted(qb, ze, side="right") - 1, 0, 7)
xs_ = [0] + [k + 1 for k in range(8)]
m0 = [y0_[~ex].mean()] + [y0_[ex & (bi == k)].mean() for k in range(8)]; m1 = [y1_[~ex].mean()] + [y1_[ex & (bi == k)].mean() for k in range(8)]
sp_ax.plot(xs_, m0, "-o", color="#9e9e9e", ms=2.0, lw=1.0, label="real")
sp_ax.plot(xs_, m1, "-o", color=CAL, ms=2.0, lw=1.2, label="planted")
sp_ax.set_xticks([0, 8], ["none", "high"]); sp_ax.set_xlabel("$L\\times R$ exposure", fontsize=7.2, labelpad=1)
sp_ax.set_ylabel(lr1_t, fontsize=7.2, labelpad=1)
sp_ax.legend(fontsize=7.0, frameon=False, loc="upper left", handletextpad=0.3, borderaxespad=0, handlelength=1.2)
txB[0].append(fig.text(0.514, 0.759, "source: proteins, CosMx protein panel;\npathways, PROGENy; TFs, CollecTRI;\nreceptor → pathway rules, Table S20", fontsize=7.2, color=MUTE, ha="left", va="top", linespacing=1.2))
txB[1].append(fig.text(0.664, 0.759, "source: LR pairs, LIANA+ consensus\nresource; decoys, expression-matched\npanel genes", fontsize=7.2, color=MUTE, ha="left", va="top", linespacing=1.2))
txB[2].append(fig.text(0.834, 0.759, "simulated on the real data; $\\beta^2$ = planted /\noriginal response variance (0.25–16%)", fontsize=7.2, color=MUTE, ha="left", va="top", linespacing=1.2))

# ================================================================== C  graph transformer
UX, YLO, YHI = 56.0, 6.5, 61.5; U = 0.0785; EY0 = 0.398; EW = U * UX / FW; EH = U * (YHI - YLO) / FH; EX0 = (PC[0] + PC[2]) / 2 - EW / 2 + 0.004   # centred in C
E = fig.add_axes([EX0, EY0, EW, EH]); E.set_xlim(0, UX); E.set_ylim(YLO, YHI); E.axis("off")
fx = lambda x: EX0 + EW * x / UX; fy = lambda y: EY0 + EH * (y - YLO) / (YHI - YLO)
inset = lambda x0, y0, x1, y1: fig.add_axes([fx(x0), fy(y0), fx(x1) - fx(x0), fy(y1) - fy(y0)])


def mtext(ax, x, y, s, **kw):
    """centred label mixing plain words and $math$: plain runs stay whole text objects (editable as one box in PowerPoint)"""
    rend_ = fig.canvas.get_renderer(); px = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
    py = ax.transData.transform((0, 1))[1] - ax.transData.transform((0, 0))[1]
    va = kw.pop("va", "baseline"); pt = kw.get("fontsize", 7) / 72 * fig.dpi / py
    yb = y - 0.74 * pt if va == "top" else y - 0.36 * pt if va == "center" else y
    parts = [p_ for p_ in re.split(r"(\$[^$]*\$)", s) if p_.strip()]
    ts = [ax.text(x, yb, p_.strip(), ha="left", va="baseline", **kw) for p_ in parts]
    ws = [t_.get_window_extent(rend_).width / px for t_ in ts]; gap = kw.get("fontsize", 7) * 0.28 / 72 * fig.dpi / px
    x_ = x - (sum(ws) + gap * (len(ws) - 1)) / 2
    for t_, w_ in zip(ts, ws): t_.set_x(x_); x_ += w_ + gap


def blk(x, y, w, h, title, fc="white", ec=NUIS, tc=INK, lw=1.2, fs=6.6, ty=None, stack=0):
    for s_ in range(stack, 0, -1):                                  # stacked cards = parallel heads
        E.add_patch(FancyBboxPatch((x + 0.7 * s_, y + 0.7 * s_), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc,
                                   ec=ec if fc == "white" else "white", lw=lw * 0.8, alpha=1 if fc == "white" else 0.55, zorder=1.5))
    E.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec, lw=lw, zorder=2))
    mtext(E, x + w / 2, (y + h / 2) if ty is None else ty, title, va="center" if ty is None else "top", fontsize=fs, fontweight="bold", color=tc, zorder=3)


def arr(x1, y1, x2, y2, c=INK, ls="-", rad=0.0, lw=1.0):
    E.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, color=c, lw=lw, linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))


nb = src[mine]
LX, LW_ = 2.5, 21.0; lc = LX + LW_ / 2
blk(LX, 15.0, LW_, 4.4, "cell + neighbour features", fc=LIGHT, ec=LIGHT, tc=NUIS)
E.add_patch(FancyBboxPatch((LX - 0.6, 21.3), LW_ + 1.2, 19.6, boxstyle="round,pad=0,rounding_size=1.0", fc="#f4f4f4", ec="#a0a0a0", lw=0.9, zorder=1))
blk(LX + 2.2, 22.6, LW_ - 3.0, 4.8, "graph attention", fc=NUIS, ec=NUIS, tc="white", stack=2)
blk(LX + 2.2, 28.6, LW_ - 3.0, 2.2, "Add & Norm", fc=LIGHT, ec=LIGHT, tc=NUIS, fs=6.0)
blk(LX + 2.2, 32.4, LW_ - 3.0, 3.2, "Feed-forward", fc=NUIS, ec=NUIS, tc="white", fs=6.2)
blk(LX + 2.2, 37.2, LW_ - 3.0, 2.2, "Add & Norm", fc=LIGHT, ec=LIGHT, tc=NUIS, fs=6.0)
E.text(LX - 1.3, 31.1, "×2", ha="right", va="center", fontsize=8.0, fontweight="bold", color=NUIS)
for y0__, y1__ in [(21.8, 29.7), (31.6, 38.3)]:
    E.plot([lc + 1.0, LX + 0.9, LX + 0.9], [y0__, y0__, y1__], color=NUIS, lw=0.9, zorder=3); arr(LX + 0.9, y1__, LX + 2.2, y1__, c=NUIS, lw=0.9)
for dx_, lab_ in [(-5.0, "Q"), (1.0, "K"), (7.0, "V")]:
    arr(lc + dx_, 19.4, lc + dx_, 22.6, c=NUIS); E.text(lc + dx_ - 0.9, 20.35, lab_, fontsize=7.2, color=NUIS, ha="right", va="center", fontweight="bold")
for y0__, y1__ in [(27.4, 28.6), (30.8, 32.4), (35.6, 37.2), (39.4, 42.6)]: arr(lc + 1.0, y0__, lc + 1.0, y1__, c=NUIS)
blk(LX, 42.6, LW_, 3.6, "graph + single-cell fusion", fc=NUIS, ec=NUIS, tc="white", fs=6.4)
blk(LX, 48.4, LW_, 4.0, "background $\\hat{\\mathbf{y}}^{0}_i$", fc="white", ec=NUIS, tc=NUIS, fs=6.6)
arr(lc, 46.2, lc, 48.4, c=NUIS)
E.text(-1.0, 31.1, "background branch", rotation=90, ha="center", va="center", fontsize=7.2, fontweight="bold", color=NUIS)
RX, RW = 30.5, 22.0; rc_x = RX + RW / 2
blk(RX, 14.3, RW, 7.4, "$s^{c}_{ji}=L_j\\times R_i$", ec=LRC, ty=20.95, fs=6.6)
blk(RX, 23.2, RW - 1.4, 8.4, "attention $a^{c}_{ji}$", ec=LRC, stack=2, ty=30.75, fs=6.4)
blk(RX, 33.5, RW, 4.4, "heads × cell-type gate $g^{c}$", ec=LRC, fs=6.0)
blk(RX, 39.8, RW, 6.6, "effect on responses $\\mathbf{v}_c$", ec=CAL, ty=45.65, fs=6.2, tc=CAL)
blk(RX, 48.4, RW, 4.0, "LR terms $\\Delta\\hat{\\mathbf{y}}^{c}_i$", fc="white", ec=LRC, tc=LRC, fs=6.6)
arr(rc_x - 0.7, 21.7, rc_x - 0.7, 23.2, c=LRC); E.text(rc_x + 0.2, 22.45, "K, V", fontsize=7.2, color=LRC, ha="left", va="center", fontweight="bold")
for y0__, y1__ in [(31.6, 33.5), (37.9, 39.8), (46.4, 48.4)]: arr(rc_x, y0__, rc_x, y1__, c=LRC)
E.text(54.4, 31.1, "LR branch", rotation=270, ha="center", va="center", fontsize=7.2, fontweight="bold", color=LRC)
arr(lc + 1.0, 41.0, RX, 28.5, c=NUIS, ls=(0, (3, 2)), rad=-0.25); E.text(26.5, 37.8, "Q", fontsize=7.2, color=NUIS, ha="center", fontweight="bold")
arr(lc, 10.0, lc, 15.0, c=INK2); arr(rc_x, 10.0, rc_x, 14.3, c=INK2)
E.text(lc, 8.6, "cells + neighbours (A)", fontsize=7.2, color=INK2, ha="center", va="center")
E.text(rc_x, 8.6, "LR pairs + decoys (B)", fontsize=7.2, color=INK2, ha="center", va="center")
real_c = np.flatnonzero(~ch.decoy.to_numpy())[:14]
sv = np.log1p(np.exp(np.clip(z["L_mo"][src[mine]][:, real_c], -10, 10))) * np.log1p(np.exp(np.clip(z["R_mo"][dst[mine]][:, real_c], -10, 10)))
h1 = inset(RX + 3.0, 15.0, RX + RW - 3.0, 18.7); tiles(h1, sv / sv.max(), "viridis", 0, 1); h1.set_xticks([]); h1.set_yticks([])
k_ = int(fold[RCV]); A = np.load(R / "G_real_s0" / f"fold{k_}_mo_attention.npz"); amap = dict(zip(A["edge"], A["a"].astype(np.float32)))
att = np.array([amap.get(e, np.full(4, np.nan)) for e in mine])
h2 = inset(RX + 4.0, 24.9, RX + 16.0, 28.6); tiles(h2, att, "viridis", 0, 1); h2.set_yticks([])
nh_ = att.shape[1]; h2.set_xticks([0.5, nh_ / 2, nh_ - 0.5], ["h1", "…", f"h{nh_}"], fontsize=7.0, color=INK2); h2.tick_params(length=0, pad=1)
for r, j in enumerate(nb): h2.add_patch(Rectangle((-0.55, r), 0.4, 1, color=TCOL[TYPES[ctype[j]]], clip_on=False))
st_ = torch.load(R / "G_pw_contact_s0" / "fold0_mo.pt", map_location="cpu")["lr"]; Vpw = st_["V"].numpy()[:, -41:]
pwm = np.load(D / "pathway_tf" / "mask_contact.npy").astype(float); rc_ = np.flatnonzero(~ch.decoy.to_numpy()); VM = (Vpw * pwm)[rc_]; rows_m = rc_[np.argsort(-np.abs(VM).sum(1))[:12]]
cols_m = np.flatnonzero(pwm[rows_m].sum(0) > 0); vm = (Vpw * pwm)[rows_m][:, cols_m]; vm = vm / (np.abs(vm).max(1, keepdims=True) + 1e-12)
h3 = inset(RX + 3.0, 40.3, RX + RW - 3.0, 43.9)
tiles(h3, np.where(pwm[rows_m][:, cols_m] > 0, vm, np.nan), DIV, -1, 1, lw=0.3); h3.set_facecolor("#eeeeee"); h3.set_xticks([]); h3.set_yticks([])
for hh in (h1, h2, h3): [s.set_visible(False) for s in hh.spines.values()]
SX, SY, SR = 28.0, 54.2, 0.95                                                  # sum node: circle with a drawn plus, arrows meet its outline
E.add_patch(Circle((SX, SY), SR, fc="white", ec=INK, lw=1.1, zorder=5))
E.plot([SX - 0.5, SX + 0.5], [SY, SY], color=INK, lw=1.1, zorder=6); E.plot([SX, SX], [SY - 0.5, SY + 0.5], color=INK, lw=1.1, zorder=6)
arr(lc, 52.4, SX - SR, SY, c=NUIS, rad=-0.3); arr(rc_x, 52.4, SX + SR, SY, c=LRC, rad=0.3)
E.add_patch(FancyBboxPatch((8.0, 56.6), 40.0, 4.0, boxstyle="round,pad=0,rounding_size=0.8", fc=INK, ec=INK, zorder=2))
E.text(28.0, 58.6, "$\\hat{\\mathbf{y}}_i=\\hat{\\mathbf{y}}^{0}_i+\\sum_c\\Delta\\hat{\\mathbf{y}}^{c}_i\\;\\approx\\;\\mathbf{y}_i$", ha="center", va="center", fontsize=8.2, color="white", zorder=3)
arr(SX, SY + SR, SX, 56.6, c=INK)

# ================================================================== D  two tasks: follow one LR pair L -> R through three steps each
fig.add_artist(FancyBboxPatch((0.452, 0.688), 0.532, 0.020, boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
                              fc="#f0f0f0", ec="#c8c8c8", lw=0.8, mutation_aspect=AR))
fig.text(0.718, 0.698, "the trained model, applied to one candidate LR pair (ligand L on sender cells → receptor R on the receiver)",
         fontsize=7.6, ha="center", va="center", color=INK)
fig.add_artist(matplotlib.lines.Line2D([0.452, 0.984], [0.540, 0.540], transform=fig.transFigure, color=RULE, lw=0.8, ls=(0, (3, 3))))
lab(0.452, 0.664, "Task 1   Predict CCC scores", c=INK, fs=8.2)
lab(0.452, 0.517, "Task 2   Predict downstream response changes", c=INK, fs=8.2)
STEPX = [0.455, 0.632, 0.810]                       # step headings, left edge of each column
CX = [0.5407, 0.7180, 0.8953]                       # column centres: icons and captions share them
ROWY = (0.6025, 0.4555)                             # vertical centre of the icon band in task 1 and task 2


HEADS, CAPS = [], []


def step_head(k, y, text, c):
    """numbered heading centred over column k (badge + text as one group)"""
    t_ = fig.text(CX[k] + 0.007, y, text, fontsize=7.4, fontweight="bold", color=INK, va="bottom", ha="center")
    b_ = Circle((CX[k], y + 0.0045), 0.0045, transform=fig.transFigure, fc=INK, ec="none"); fig.add_artist(b_)   # numbered badge
    n_ = fig.text(CX[k], y + 0.0045, str(k + 1), fontsize=7.0, fontweight="bold", color="white", ha="center", va="center")
    HEADS.append((t_, b_, n_))


def net(cx, y, widths, c=INK2, dashed=False, cross=False, labels=None):
    """one receiver (R) with three senders (L); arrow width = contribution; limits hug the drawing so it centres"""
    a = icon_ax([cx - 0.035, y, 0.070, 0.072], (-0.1, 7.9), (0.3, 8.75))
    snd = [(1.6, 6.6), (1.2, 2.0), (6.1, 7.75)]
    RC, RS, RR = np.array([6.3, 3.9]), 0.95, 1.3
    ppu = 0.070 * FW * 72 / 8.0                                  # points per data unit (axes width / x range)
    for (xx, yy), w_ in zip(snd, widths):
        a.add_patch(Circle((xx, yy), RS, fc="#e3e3e3", ec=INK2, lw=0.7, zorder=3)); a.text(xx, yy, "L", ha="center", va="center", fontsize=7.2, color=LRC, fontweight="bold", zorder=4)
        u_ = (RC - (xx, yy)) / np.linalg.norm(RC - (xx, yy))
        lw_ = 0.6 + 1.6 * w_                                     # tail on the sender outline, rounded tip on the receiver outline
        a.add_patch(FancyArrowPatch(tuple(np.array((xx, yy)) + u_ * (RS + 0.35 / ppu)), tuple(RC - u_ * (RR - 0.8 / ppu)), arrowstyle="-|>",
                                    mutation_scale=6 + 2.5 * w_, shrinkA=0, shrinkB=0, joinstyle="round", capstyle="butt",
                                    color=c if not dashed else "#a8a8a8", lw=lw_, linestyle=(0, (3, 2)) if dashed else "-", zorder=2))
    a.add_patch(Circle(tuple(RC), RR, fc="#e3e3e3", ec=INK2, lw=0.8, zorder=3)); a.text(*RC, "R", ha="center", va="center", fontsize=7.6, fontweight="bold", zorder=4)
    if cross:
        for (xx, yy) in snd:
            mx, my = (xx + 6.3) / 2, (yy + 3.9) / 2
            a.plot([mx - 0.35, mx + 0.35], [my - 0.35, my + 0.35], color="#d62728", lw=1.4); a.plot([mx - 0.35, mx + 0.35], [my + 0.35, my - 0.35], color="#d62728", lw=1.4)
    if labels:                                          # next to each sender, clear of the arrows
        for (xx, yy), (dx_, dy_, ha_), t_ in zip(snd, [(0, -1.5, "center"), (0, -1.5, "center"), (-1.2, 0, "right")], labels):
            a.text(xx + dx_, yy + dy_, t_, fontsize=7.2, color=c, ha=ha_, va="center")
    return a


# ---- Task 1
step_head(0, 0.646, "Read pair contributions", LRC)
T1 = [net(CX[0], 0.567, [0.6, 0.6, 0.6], c=MUTE)]
CAPS.append(fig.text(CX[0], 0.556, "contribution of each sender", fontsize=7.0, ha="center", va="center"))
step_head(1, 0.646, "Check against decoys", LRC)
hd = icon_ax([CX[1] - 0.055, 0.570, 0.110, 0.068], (0, 12), (0, 8))
xs_ = np.linspace(0.6, 6.4, 14); hs_ = 4.6 * np.exp(-((xs_ - 2.8) / 1.5) ** 2)
for xv, hv in zip(xs_, hs_): hd.add_patch(Rectangle((xv, 1.4), 0.36, hv, fc="#bdbdbd", ec="none"))
hd.plot([0.3, 11.7], [1.4, 1.4], color=INK, lw=0.7); hd.plot([10.0, 10.0], [1.4, 6.0], color=LRC, lw=2.4)
hd.text(2.9, 0.2, "decoys", ha="center", fontsize=7.2, color=MUTE); hd.text(10.0, 0.2, "L → R", ha="center", fontsize=7.2, color=LRC, fontweight="bold")
hd.text(6.0, 7.6, "error increase when removed", ha="center", fontsize=7.2, color=INK2, va="top")
T1.append(hd)
CAPS.append(fig.text(CX[1], 0.556, "LR-pair evidence = 1 − FDR", fontsize=7.0, ha="center", va="center"))
step_head(2, 0.646, "CCC score per cell pair", LRC)
T1.append(net(CX[2], 0.567, [1.0, 0.45, 0.1], c=LRC, labels=["high", "medium", "low"]))
CAPS.append(fig.text(CX[2], 0.556, "CCC score = evidence × contribution", fontsize=7.0, ha="center", va="center"))

# ---- Task 2
step_head(0, 0.498, "Full prediction", CAL)
T2 = [net(CX[0], 0.421, [0.6, 0.6, 0.6], c=MUTE)]
CAPS.append(fig.text(CX[0], 0.410, "predicted response with L → R", fontsize=7.0, ha="center", va="center"))
step_head(1, 0.498, "Remove L → R", CAL)
T2.append(net(CX[1], 0.421, [0.6, 0.6, 0.6], dashed=True, cross=True))
CAPS.append(fig.text(CX[1], 0.410, "predicted response without L → R", fontsize=7.0, ha="center", va="center"))
step_head(2, 0.498, "Response changes", CAL)
rb = fig.add_axes([CX[2] - 0.022, 0.424, 0.090, 0.064]); rb.set_xlim(-1.15, 1.15); rb.set_ylim(-0.6, 2.6); rb.axis("off")
for k_, (nm, v_) in enumerate([("NF-κB activity", -0.95), ("STAT1 activity", 0.45), ("response protein", -0.5)][::-1]):
    rb.barh(k_, v_, color="#2166ac" if v_ < 0 else "#b2182b", height=0.62)
    rb.text(-1.2, k_, nm, ha="right", va="center", fontsize=7.2)
rb.axvline(0, color=INK, lw=0.7); rb.text(-0.6, -0.55, "decrease", ha="center", fontsize=7.2, color="#2166ac", va="top"); rb.text(0.6, -0.55, "increase", ha="center", fontsize=7.2, color="#b2182b", va="top")
T2.append(rb)
CAPS.append(fig.text(CX[2], 0.410, "change of each response, in every receiver", fontsize=7.0, ha="center", va="center"))
# step arrows are drawn after layout, centred in the gap between neighbouring icons (see save section)

# ================================================================== E  downstream analysis, one row (left: task 1, right: task 2)
mE = np.flatnonzero(fov == FOV_E); keepE = fov[cdst] == FOV_E

names, qv = W1["real_names"], W1["real_q"]; plset = set(W1["plant"])
o_ = np.argsort(qv)[:10][::-1]
tE = [lab(0.016, 0.348, "Evidence per LR pair", c=INK)]
f1a = fig.add_axes([0.092, 0.224, 0.095, 0.116]); clean(f1a, 6.4)
pv = 1 - qv[o_]; cols = [LRC if names[k] == LR1 else ("#4d4d4d" if names[k] in plset else "#bdbdbd") for k in o_]
f1a.hlines(np.arange(len(o_)), 0, pv, color=cols, lw=1.1); f1a.scatter(pv, np.arange(len(o_)), c=cols, s=14, zorder=3)
f1a.set_yticks(np.arange(len(o_)), [short(names[k]) for k in o_], fontsize=7.0)
for tl, k in zip(f1a.get_yticklabels(), o_):
    if names[k] == LR1: tl.set_color(LRC); tl.set_fontweight("bold")
f1a.axvline(0.9, color=INK, lw=0.7, ls=(0, (3, 2))); f1a.set_xlim(0, 1.02); f1a.set_xticks([0, 0.5, 1]); f1a.set_xlabel("evidence (1 − FDR)", fontsize=7.2, labelpad=1)

tE.append(lab(0.174, 0.348, "CCC score per cell pair", c=INK))
e_idx = np.flatnonzero(keepE); top_ = e_idx[np.argsort(-sc[e_idx])[:150]]
cpos = np.r_[xy[csrc[top_]], xy[cdst[top_]]]; loE, hiE = xy[mE].min(0) + 2, xy[mE].max(0) - 2; HALF = np.array([128.0, 75.0])
cent = [np.clip(p_, loE + HALF, hiE - HALF) for p_ in cpos]; inwin = lambda c_, p_: (np.abs(p_ - c_) < HALF - 5).all(-1)
cc = cent[int(np.argmax([(inwin(c_, xy[csrc[top_]]) & inwin(c_, xy[cdst[top_]])).sum() for c_ in cent]))]
tw = top_[inwin(cc, xy[csrc[top_]]) & inwin(cc, xy[cdst[top_]])]; part = set(csrc[tw]) | set(cdst[tw])
fb = fig.add_axes([0.176, 0.226, 0.200, 0.1135])
draw_cells(fb, FOV_E, lw=0.3, fill=lambda i: (*tcol[ctype[i]], 0.85) if i in part else (0.91, 0.91, 0.91, 1))
w_ = sc[tw] / sc[tw].max()
fb.add_collection(LineCollection([[xy[a], xy[b]] for a, b in zip(csrc[tw], cdst[tw])], colors=INK, linewidths=1.3, zorder=4))
fb.scatter(*xy[np.unique(csrc[tw])].T, marker="^", s=12, c=INK, edgecolors="white", linewidths=0.4, zorder=5)
fb.scatter(*xy[np.unique(cdst[tw])].T, marker="o", s=10, c="white", edgecolors=INK, linewidths=0.7, zorder=5)
fb.set_xlim(cc[0] - HALF[0], cc[0] + HALF[0]); fb.set_ylim(cc[1] - HALF[1], cc[1] + HALF[1]); fb.set_aspect("equal")
for s_ in fb.spines.values(): s_.set_color(RULE)
fb.set_xticks([]); fb.set_yticks([])
fb.plot([cc[0] + HALF[0] - 35, cc[0] + HALF[0] - 5], [cc[1] - HALF[1] + 6, cc[1] - HALF[1] + 6], color=INK, lw=1.4, zorder=7)
fb.text(cc[0] + HALF[0] - 20, cc[1] - HALF[1] + 8, "30 µm", ha="center", va="bottom", fontsize=7.0, zorder=7)
fb.set_xlabel("top-scoring cell pairs of " + short(LR1) + ";  ▲ sender,  ○ receiver", fontsize=7.2, labelpad=2)

# in silico blockade of LR1: predicted change of its response in every receiver of one field of view
tE.append(lab(0.452, 0.348, "Removing " + short(LR1) + " from the model", c=INK))
pe = pred[mE]; scale = np.quantile(pe[pe > 0], 0.98) if (pe > 0).any() else 1.0
cmapE = plt.get_cmap("Blues")
ea = fig.add_axes([0.456, 0.208, 0.150, 0.134])
draw_cells(ea, FOV_E, lw=0.2, fill=lambda i: (*cmapE(0.25 + 0.75 * min(pred[i] / scale, 1.0))[:3], 1.0) if pred[i] > 0.12 * scale else (0.92, 0.92, 0.92, 1))
lo, hi_ = xy[mE].min(0), xy[mE].max(0); ea.set_xlim(lo[0], hi_[0]); ea.set_ylim(lo[1], hi_[1]); ea.set_aspect("equal")
ea.set_xticks([]); ea.set_yticks([]); ea.patch.set_visible(False); [s_.set_visible(False) for s_ in ea.spines.values()]
ea.set_xlabel("per-cell drop of " + lr1_t + " when removed", fontsize=7.2)
cbE = fig.add_axes([0.610, 0.226, 0.004, 0.095]); fig.colorbar(plt.cm.ScalarMappable(cmap=cmapE, norm=plt.Normalize(0, 1)), cax=cbE)
cbE.tick_params(labelsize=7.0, length=1.5, pad=1); cbE.set_yticks([0, 1], ["0", "max"]); cbE.set_ylabel("predicted drop of " + lr1_t, fontsize=7.0, labelpad=1)

# the same blockade summarised by receiver cell type (all receivers of the evaluation fields of view)
tE.append(lab(0.690, 0.348, "Predicted change by receiver type", c=INK))
vb = pred[rc_all] / np.quantile(pred[rc_all], 0.99); tb = ctype[rc_all]
ty_ = [k for k in range(len(TYPES)) if (tb == k).sum() >= 20]; ty_ = sorted(ty_, key=lambda k: np.median(vb[tb == k]))
eg = fig.add_axes([0.745, 0.228, 0.235, 0.110]); clean(eg, 6.4)
bp = eg.boxplot([vb[tb == k] for k in ty_], vert=False, showfliers=False, widths=0.6, patch_artist=True,
                medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(color=INK2, lw=0.6), capprops=dict(color=INK2, lw=0.6))
for p_, k in zip(bp["boxes"], ty_): p_.set_facecolor(TCOL[TYPES[k]]); p_.set_edgecolor(INK2); p_.set_linewidth(0.6)
eg.set_yticks(np.arange(1, len(ty_) + 1), [TYPES[k] for k in ty_], fontsize=7.2); eg.set_xlim(0, 1.08 * max(l_.get_xdata().max() for l_ in bp["caps"]))
eg.set_xlabel("predicted drop of " + lr1_t + " (max = 1)", fontsize=7.2, labelpad=1)

r_ = pearsonr(pred[rc_all], true_eff[rc_all])[0]                                   # per-cell agreement

# ------------------------------------------------------------------ save: cut maps to their windows 
def unclip(ax):
    (x0_, x1_), (y0_, y1_) = sorted(ax.get_xlim()), sorted(ax.get_ylim()); bx = Bbox([[x0_, y0_], [x1_, y1_]])
    inside = lambda P: (P[:, 0] >= x0_) & (P[:, 0] <= x1_) & (P[:, 1] >= y0_) & (P[:, 1] <= y1_)
    for c in list(ax.collections):
        if isinstance(c, PolyCollection):
            fcs_ = c.get_facecolor(); vs_, keep = [], []
            for k, p_ in enumerate(c.get_paths()):
                q_ = p_.clip_to_bbox(bx)
                if len(q_.vertices) > 2: vs_.append(q_.vertices); keep.append(k)
            c.set_verts(vs_)
            if len(fcs_) > 1: c.set_facecolor(fcs_[keep])
        elif isinstance(c, LineCollection):
            segs_ = c.get_segments(); ok_ = [inside(np.asarray(s_)).all() for s_ in segs_]; lws = np.atleast_1d(c.get_linewidth()); ecs = c.get_edgecolor()
            c.set_segments([s_ for s_, o2 in zip(segs_, ok_) if o2])
            if len(lws) == len(segs_): c.set_linewidth(lws[np.array(ok_)])
            if len(ecs) == len(segs_): c.set_edgecolor(ecs[np.array(ok_)])
        elif isinstance(c, PathCollection):
            off = np.asarray(c.get_offsets()); m_ = inside(off); c.set_offsets(off[m_])


for ax_ in (mapax, axB, fb, ea): unclip(ax_)
for ax_ in ICONS:
    for ch_ in ax_.get_children(): ch_.set_clip_on(False)

# ------------------------------------------------------------------ layout pass: align plots with their titles, centre the D icons
fig.canvas.draw(); rend = fig.canvas.get_renderer(); toF = fig.transFigure.inverted()
tight = lambda a: a.get_tightbbox(rend).transformed(toF)                          # includes tick labels, legends, icon text
def cbox(a):                                                                     # drawn content only (icons have no frame)
    ex_ = [c.get_window_extent(rend) for c in a.get_children() if c.get_visible() and c is not a.patch and not isinstance(c, matplotlib.spines.Spine)
           and isinstance(c, (matplotlib.patches.Patch, matplotlib.text.Text, matplotlib.lines.Line2D, matplotlib.collections.Collection))
           and not (isinstance(c, matplotlib.text.Text) and not c.get_text())]
    return Bbox.union([e_ for e_ in ex_ if e_.width > 0 or e_.height > 0]).transformed(toF)
box_ = lambda a: cbox(a) if a in ICONS else tight(a)


def move(axs, dx=0.0, dy=0.0):
    for a in axs:
        p = a.get_position(); a.set_position([p.x0 + dx, p.y0 + dy, p.width, p.height])


def flush(a, x, right=None):
    """left edge of the plot including its labels at x (right edge kept, or moved to `right`)"""
    p = a.get_position(); dx = x - tight(a).x0; a.set_position([p.x0 + dx, p.y0, p.width - dx, p.height])
    if right is not None:
        p = a.get_position(); a.set_position([p.x0, p.y0, p.width + right - tight(a).x1, p.height])


move(heatA + [strip, cb], 0.014 - min(tight(a).x0 for a in heatA))               # A: graphics share top and bottom,
YA = 0.756                                                                       #    the row beneath them shares its top
for a, x_ in ((mapax, -0.06), (axB, -0.02)):
    pa_ = a.get_position(); a.get_legend().set_bbox_to_anchor((x_, (YA - pa_.y0) / pa_.height), transform=a.transAxes)
pc_ = cb.get_position(); cb.set_position([pc_.x0, YA - pc_.height, pc_.width, pc_.height])
for a in (axD, axC, sp_ax):                                                      # B: plot bottoms and x labels on one line
    p = a.get_position(original=True); a.set_position([p.x0, 0.782, p.width, p.y1 - 0.782])
    a.xaxis.set_label_coords(0.5, (0.771 - 0.782) / (p.y1 - 0.782)); a.xaxis.label.set_fontsize(5.2); a.xaxis.label.set_va("top")
flush(axC, 0.664); flush(sp_ax, 0.834)
extB = lambda o: o.get_window_extent(rend).transformed(toF)                      # B: three columns, equal gaps
colsB = [([axD, st], txB[0]), ([nc, axC], txB[1]), ([pc, sp_ax], txB[2])]
bbB = [Bbox.union([box_(a) for a in axs_] + [extB(o) for o in txs_]) for axs_, txs_ in colsB]
gapB = (0.986 - 0.512 - sum(b_.width for b_ in bbB)) / 2; xB = 0.512
for (axs_, txs_), b_ in zip(colsB, bbB):
    dx_ = xB - b_.x0; move(axs_, dx_); xB += b_.width + gapB
    for o in txs_: o.set_x(o.get_position()[0] + dx_)
YB_E, YT_E, YL_E = 0.228, 0.338, 0.214                                           # E: equal plot heights, x labels on one line
for a in (fb, ea): a.set_anchor("W")
for a, w in ((f1a, None), (fb, 0.30), (ea, 0.30), (eg, None)):
    p = a.get_position(original=True); a.set_position([p.x0, YB_E, p.width if w is None else w, YT_E - YB_E])
for a in (f1a, fb, ea, eg):
    a.xaxis.set_label_coords(0.5, (YL_E - YB_E) / (YT_E - YB_E)); a.xaxis.label.set_va("top"); a.xaxis.label.set_fontsize(5.8)
pe_ = ea.get_position(); cbE.set_position([pe_.x1 + 0.006, YB_E + 0.1 * (YT_E - YB_E), 0.004, 0.8 * (YT_E - YB_E)])
grpE = [[f1a], [fb], [ea, cbE], [eg]]                                          # E: four plots, equal gaps, titles flush left
bbE = [Bbox.union([tight(a) for a in g_]) for g_ in grpE]
gapE = (0.982 - 0.016 - sum(b_.width for b_ in bbE)) / (len(grpE) - 1); xE = 0.016
for g_, b_, t_ in zip(grpE, bbE, tE):
    move(g_, xE - b_.x0); t_.set_x(xE); xE += b_.width + gapE
ext_ = lambda o: o.get_window_extent(rend).transformed(toF)                      # D: heading, icon and caption flush left per column
for o in [h[0] for h in HEADS] + CAPS: o.set_ha("left")
rowsD = ((T1, ROWY[0], LRC), (T2, ROWY[1], CAL))
WD = [max(max(0.014 + ext_(HEADS[3 * r + k][0]).width, box_(rowsD[r][0][k]).width, ext_(CAPS[3 * r + k]).width) for r in range(2)) for k in range(3)]
XL_D, XR_D = 0.452 + 0.035, 0.984 - 0.035                                     # step columns inset from the task titles
gapD = (XR_D - XL_D - sum(WD)) / 2; COLX = [XL_D, XL_D + WD[0] + gapD, XL_D + WD[0] + WD[1] + 2 * gapD]
for r, (row, cy, c_) in enumerate(rowsD):
    for k, (a, x) in enumerate(zip(row, COLX)):
        tb = box_(a); move([a], x - tb.x0, cy - (tb.y0 + tb.y1) / 2)
        t_, b_, n_ = HEADS[3 * r + k]; b_.center = (x + 0.0045, b_.center[1]); n_.set_x(x + 0.0045); t_.set_x(x + 0.014)
        CAPS[3 * r + k].set_x(x)
x1D = [[box_(row[k]).x1 for k in range(2)] for row, _, _ in rowsD]              # arrows: one length for all four,
L_ = min(COLX[k + 1] - x1 for xs_ in x1D for k, x1 in enumerate(xs_)) - 2 * 0.012  # each centred in its own gap
for (row, cy, c_), xs_ in zip(rowsD, x1D):
    for k in range(2):
        mid = (xs_[k] + COLX[k + 1]) / 2; farrow([(mid - L_ / 2, cy), (mid + L_ / 2, cy)], c=INK2, lw=1.1, ms=9)



CROP = Bbox([[0, (PE[1] - 0.004) * FH], [FW, (PA[3] + 0.004) * FH]])                                 # the page ends below E
fig.savefig(OUT, dpi=300, facecolor="white", bbox_inches=CROP); fig.savefig(str(OUT).replace(".png", ".pdf"), facecolor="white", bbox_inches=CROP)
print("saved", OUT, "per-cell r", round(r_, 3))
