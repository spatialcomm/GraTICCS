"""Map of one secreted channel's top contributing pairs in the FOV with most contribution (Supplementary Figure S4).

python -m figures.case_map <channel> <readout> <run>[,<run>...] <out.png>"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from graticcs.paths import BREAST, RAW, RESULTS
D, R = BREAST, RESULTS
COSMX = RAW / "cosmx_breast"
chn, readout, dirs, outp = sys.argv[1], sys.argv[2], [R / x for x in sys.argv[3].split(",")], sys.argv[4]
CH, RD = {"IL1B->IL1R1_IL1RAP": "IL-1β → IL1R1/IL1RAP"}, {"PW_NFkB": "NF-κB pathway activity"}
z = np.load(D / "gt_inputs.npz"); S = np.load(D / "gt_sec_inputs.npz"); sm = json.load(open(D / "gt_sec_channels.json"))
names = [c["name"] for c in sm["channels"]]; c = names.index(chn); types = json.load(open(D / "gt_channels.json"))["types"]
T = sm["prot_readout"] + json.load(open(D / "pathway_tf" / "meta.json"))["readouts"]; t = T.index(readout)
Y = np.c_[S["Yp"], np.load(D / "pathway_tf" / "Y_pathway_tf.npz")["Y"]]
ctype, fold, fov, xy = z["ctype"], z["fold"], z["fov"], z["xy"]; oh = np.eye(len(types), dtype=np.float32)[ctype]
src, dst, dist, con = S["src"], S["dst"], S["dist"], S["contact"]
sp_ = lambda a: np.log1p(np.exp(np.clip(a.astype(np.float32), -10, 10)))
contrib = np.zeros(len(src)); nseen = np.zeros(len(src))
for d in dirs:
    for k in range(5):
        A = np.load(d / f"fold{k}_mo_attention.npz"); e, a = A["edge"], A["a"]
        st = torch.load(d / f"fold{k}_mo.pt", map_location="cpu")["lr"]
        P = torch.softmax(st["P"], 1).numpy()[c]; g = 1 / (1 + np.exp(-st["G"].numpy()[ctype[dst[e]], c]))
        contrib[e] += (a @ P) * sp_(S["L"][src[e], c]) * sp_(S["R"][dst[e], c]) * g * st["V"].numpy()[c, t]; nseen[e] += 1
contrib = contrib / np.maximum(nseen, 1)
best = pd.Series(np.abs(contrib)).groupby(fov[dst]).sum().idxmax()
cells = pd.read_parquet(D / "cells.parquet", columns=["cell_id", "fov"]); fov_id = int(cells.fov.iloc[np.flatnonzero(fov == best)[0]])
poly = pd.read_csv(COSMX / "BreastCancer-polygons.csv", usecols=["fov", "cell", "x_global_px", "y_global_px"]); poly = poly[poly.fov == fov_id]
idx = pd.Series(np.arange(len(cells)), index=cells.cell_id); UM = 0.5119 / 4256 * 1000
fig, ax = plt.subplots(figsize=(7.2, 7)); pal = plt.get_cmap("tab10"); verts, cols = [], []
for cell, g in poly.groupby("cell"):
    if cell in idx.index: verts.append(g[["x_global_px", "y_global_px"]].to_numpy() * UM); cols.append(pal(ctype[idx[cell]] % 10))
ax.add_collection(PolyCollection(verts, facecolors=[(*cc[:3], 0.35) for cc in cols], edgecolors="#52514e", linewidths=0.2))
m = fov[dst] == best; top = np.argsort(-(contrib * m))[:80]; top = top[m[top] & (contrib[top] > 0)]
ax.add_collection(LineCollection([[xy[src[i]], xy[dst[i]]] for i in top], colors="#eb6834", linewidths=1.4, alpha=0.9))
snd = np.unique(src[top]); ax.scatter(xy[snd, 0], xy[snd, 1], marker="^", s=40, c="#eb6834", edgecolors="#0b0b0b", linewidths=0.4, zorder=4, label="sender")
rv = np.unique(dst[top]); sc = ax.scatter(xy[rv, 0], xy[rv, 1], c=Y[rv, t], cmap="Blues", s=30, edgecolors="#0b0b0b", linewidths=0.4, zorder=3)
plt.colorbar(sc, ax=ax, shrink=0.6, label=f"receiver {RD.get(readout, readout)} (z)")
ax.scatter([], [], marker="o", s=30, c="#6baed6", edgecolors="#0b0b0b", linewidths=0.4, label=f"receiver (shade = {RD.get(readout, readout)})")
ax.plot([], [], color="#eb6834", lw=1.4, label="top sender-receiver link")
for k, nm in enumerate(types): ax.scatter([], [], color=pal(k % 10), alpha=0.5, label=nm)
ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=5); ax.autoscale(); ax.set_aspect("equal"); ax.set_xlabel("µm"); ax.set_ylabel("µm")
ax.set_title(f"{CH.get(chn, chn)}, {RD.get(readout, readout)}: top contributing pairs (FOV {fov_id})", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(outp, dpi=200)
ctab = pd.crosstab(np.array(types)[ctype[src[top]]], np.array(types)[ctype[dst[top]]])
print("saved", outp, "FOV", fov_id, "links", len(top)); print("sender type x receiver type of drawn links:"); print(ctab)
