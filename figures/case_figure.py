"""Figure 6 and Supplementary Figures S1-S3: one literature-supported event with real cell outlines and transcripts."""
import json, os, sys, re
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp, matplotlib
from graticcs.paths import BREAST, RAW
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from graticcs.lr_map import GENE2PROT
D = BREAST
COSMX = RAW / "cosmx_breast"
REC = COSMX / "literature_records"
cases, chn, outp = pd.read_csv(sys.argv[1]), sys.argv[2], sys.argv[3]
row_i = int(sys.argv[4]) if len(sys.argv) > 4 else 0; cls = sys.argv[5] if len(sys.argv) > 5 else "secreted"
UM = 0.5119 / 4256 * 1000
BG, INK, GOLD, RED, CORAL, MUTE = "#000000", "#f2f2f2", "#f5c518", "#e8412c", "#ff6b5a", "#b0b0b0"
_cc = cases[cases.channel == chn].sort_values("rank_all")
_vis = _cc[(_cc.ligand_rna >= 2) & (_cc.receptor_rna >= 1)] if os.environ.get("VIS", "1") == "1" else _cc
cs = (_vis if len(_vis) > row_i else _cc).iloc[row_i]
lit = json.load(open(D / "lit_channel_matches.json"))[cls][chn]
lig_genes, rec_genes = chn.split("->")[0].split("_"), chn.split("->")[1].split("_")
genes = [g.upper() for g in json.load(open(D / "genes.json"))["genes"]]; gi = {g: i for i, g in enumerate(genes)}
up = sorted({g for g, ev in lit["effectors"].items() if g in gi and g not in lig_genes + rec_genes and any(d.startswith("up") for d, _ in ev)})
def short_cite(c, doi):
    if not c: return ""
    m_au = re.match(r"^([A-Z][A-Za-z'\-]+)\s+[A-Z]{1,3}", c); au = (m_au.group(1) + " et al.,") if m_au else ""; yr = re.search(r"(19|20)\d{2}", c); parts = [x.strip() for x in c.split(". ")]
    jn = next((parts[i - 1] for i, x in enumerate(parts) if yr and yr.group(0) in x and i > 0), "")
    jn = jn if len(jn) < 45 else jn[:42] + "..."
    return f"{au} {jn} {yr.group(0) if yr else ''}".strip() + (f" (doi:{doi})" if doi else "")
cites, relays = [], []
for pp in lit["papers"]:
    r = json.load(open(REC / pp["file"], encoding="utf-8-sig"))
    if any(isinstance(e, dict) and e.get("measured_in") == "receiver" for e in r.get("downstream_effectors") or []):
        c = (r.get("reference") or {}).get("citation", ""); c = short_cite(c, (r.get("reference") or {}).get("doi"))
        if c and c not in cites: cites.append(c)
    for x in r.get("intracellular_relay") or []:
        if isinstance(x, dict) and x.get("to"): relays.append(str(x["to"]))
relay = pd.Series(relays).value_counts().index[0] if relays else ""
cells = pd.read_parquet(D / "cells.parquet"); idx = pd.Series(np.arange(len(cells)), index=cells.cell_id)
z = np.load(D / "gt_inputs.npz"); types = json.load(open(D / "gt_channels.json"))["types"]; ctype = z["ctype"]
X = sp.load_npz(D / "rna_counts.npz").tocsr()
allc = cases[cases.channel == chn].drop_duplicates("receiver").sort_values("rank_all").head(50)
top_r = idx[allc.receiver].to_numpy()
lib = np.asarray(X.sum(1)).ravel(); Xu = X[:, [gi[g] for g in up]].toarray() / np.maximum(lib, 1)[:, None] * 1e3
lu = np.log1p(Xu); zt = np.zeros_like(lu)
for t in np.unique(ctype):
    m = ctype == t; zt[m] = (lu[m] - lu[m].mean(0)) / (lu[m].std(0) + 1e-9)
gscore = zt.mean(1)
rec_expr = np.asarray(X[:, [gi[g] for g in rec_genes if g in gi]].sum(1)).ravel()
ctrl = np.flatnonzero(np.isin(ctype, ctype[top_r]) & (rec_expr > 0) & ~np.isin(np.arange(len(ctype)), top_r))
mw = mannwhitneyu(gscore[top_r], gscore[ctrl], alternative="greater")
if os.environ.get("PICK", "response") == "response" and len(_vis):
    _cand = _vis.head(50).copy(); _cand["resp"] = gscore[idx[_cand.receiver].to_numpy()]; cs = _cand.sort_values("resp", ascending=False).iloc[0]
si, ri, fov = idx[cs.sender], idx[cs.receiver], int(cs.fov)
csrc, cdst = z["src"][z["contact"] > 0], z["dst"][z["contact"] > 0]; nb = np.unique(csrc[cdst == ri])
pct = (gscore[ctype == ctype[ri]] < gscore[ri]).mean() * 100
cx, cy = (cells.CenterX_global_px[si] + cells.CenterX_global_px[ri]) / 2, (cells.CenterY_global_px[si] + cells.CenterY_global_px[ri]) / 2
half = max(20 / UM, 1.7 * np.hypot(cells.CenterX_global_px[si] - cells.CenterX_global_px[ri], cells.CenterY_global_px[si] - cells.CenterY_global_px[ri]))
poly = pd.read_csv(COSMX / "BreastCancer-polygons.csv", usecols=["fov", "cellID", "x_global_px", "y_global_px"]) if False else None
poly = pd.read_csv(COSMX / "BreastCancer-polygons.csv"); poly = poly[poly.fov == fov]
pcol = "cellID" if "cellID" in poly.columns else "cell_ID" if "cell_ID" in poly.columns else "cell"
tx = pd.read_parquet(D / "tx_by_fov" / f"fov{fov:03d}.parquet")
tx = tx[(abs(tx.x_global_px - cx) < half) & (abs(tx.y_global_px - cy) < half)]
def rng(col):
    v = cells[col].to_numpy(float); lo, hi = np.percentile(v, [2, 99.5]); return np.clip((v - lo) / (hi - lo), 0, 1)
rgb = np.c_[rng("Mean.CD45"), rng("Mean.PanCK"), rng("Mean.DAPI")] ** 0.6
fig = plt.figure(figsize=(10, 9.4), facecolor=BG); ax = fig.add_axes([0.04, 0.17, 0.68, 0.70], facecolor="#0b0c10")
cid_of = lambda c: f"c_1_{fov}_{int(c)}"
for cell, g in poly.groupby(pcol):
    xy = g[["x_global_px", "y_global_px"]].to_numpy()
    if (abs(xy[:, 0].mean() - cx) > half * 1.2) or (abs(xy[:, 1].mean() - cy) > half * 1.2): continue
    key = cid_of(cell) if pcol != "cell" else cell
    if key not in idx.index: continue
    k = idx[key]; ec, lw, ls, fa = "#3a3a3a", 0.4, "-", 0.62
    if k == si: ec, lw, fa = GOLD, 2.2, 0.85
    elif k == ri: ec, lw, fa = RED, 2.2, 0.85
    elif k in nb: ec, lw, ls = "#ffffff", 1.1, (0, (3, 2))
    ax.add_patch(Polygon(xy, closed=True, facecolor=(*rgb[k], fa), edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3 if k in (si, ri) else 2 if k in nb else 1))
cell_of_tx = tx.cell_ID.map(lambda c: idx.get(cid_of(c), -1))
def plot_tx(mask, marker, color, size, z):
    t = tx[mask]; ax.scatter(t.x_global_px, t.y_global_px, marker=marker, s=size, c=color, edgecolors="#000000", linewidths=0.4, zorder=z)
    return int(mask.sum())
n_l = plot_tx(tx.target.str.upper().isin(lig_genes) & (cell_of_tx == si), "^", GOLD, 30, 6)
n_r = plot_tx(tx.target.str.upper().isin(rec_genes) & (cell_of_tx == ri), "s", RED, 26, 6)
dmask = tx.target.str.upper().isin(up) & cell_of_tx.isin(np.r_[ri, nb])
n_d = plot_tx(dmask, "o", CORAL, 14, 5); n_dg = tx[dmask].target.nunique()
sx, sy, rx, ry = cells.CenterX_global_px[si], cells.CenterY_global_px[si], cells.CenterY_global_px[ri] * 0 + cells.CenterX_global_px[ri], cells.CenterY_global_px[ri]
ax.add_patch(FancyArrowPatch((sx, sy), (rx, ry), arrowstyle="-|>", mutation_scale=22, color="white", lw=2.4, shrinkA=14, shrinkB=14, zorder=7))
def prot(gl, k):
    out = []
    for g in gl:
        pc = GENE2PROT.get(g)
        if pc and f"P_{pc}" in cells.columns:
            v = cells[f"P_{pc}"].to_numpy(float); out.append(f"{pc} protein p{(v < v[k]).mean() * 100:.0f}")
    return out
def box(x, y, title, lines, col):
    ax.text(x, y, "\n".join([title] + lines), color=col, fontsize=9.5, fontweight="bold", ha="center", va="center", zorder=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="#000000", ec=col, lw=1.4))
lr_raw = lambda k, gl: "+".join(gl) + "=" + str(int(sum(X[k, gi[g]] for g in gl if g in gi)))
off = half * 0.62
box(sx + (sx - rx) * 0.2, sy + np.sign(sy - ry + 1e-9) * off, "SENDER", [types[ctype[si]], cs.sender, lr_raw(si, lig_genes)] + prot(lig_genes, si), GOLD)
box(rx + (rx - sx) * 0.2, ry + np.sign(ry - sy + 1e-9) * off, "RECEIVER", [types[ctype[ri]], cs.receiver, lr_raw(ri, rec_genes)] + prot(rec_genes, ri), RED)
ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half, cy + half); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
for s_ in ax.spines.values(): s_.set_color("#333333")
ax.plot([cx - half * 0.92, cx - half * 0.92 + 10 / UM], [cy - half * 0.92] * 2, color="white", lw=3); ax.text(cx - half * 0.92 + 5 / UM, cy - half * 0.86, "10 µm", color="white", ha="center", fontsize=8)
leg = [Line2D([], [], marker="^", ls="", color=GOLD, mec="k", ms=9, label=f"Ligand {'+'.join(lig_genes)} transcripts in sender (n={n_l}; measured positions)"),
       Line2D([], [], marker="s", ls="", color=RED, mec="k", ms=8, label=f"Receptor {'+'.join(rec_genes)} transcripts in receiver (n={n_r})"),
       Line2D([], [], marker="o", ls="", color=CORAL, mec="k", ms=7, label=f"Literature response genes in receiver + contacts ({n_dg} gene{"s" if n_dg != 1 else ""}, {n_d} transcript{"s" if n_d != 1 else ""})"),
       Line2D([], [], color=GOLD, lw=2.2, label="Sender (segmentation outline)"), Line2D([], [], color=RED, lw=2.2, label="Receiver (segmentation outline)"),
       Line2D([], [], color="white", lw=1.1, ls=(0, (3, 2)), label="Receiver's contacting cells")]
ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7.6, frameon=True, facecolor="#000000", edgecolor="#555555", labelcolor=INK)
ia = fig.add_axes([0.80, 0.43, 0.18, 0.36], facecolor="#0b0c10")
rng_ = np.random.default_rng(0)
def mci(v):
    bs = [rng_.choice(v, len(v)).mean() for _ in range(2000)]; return v.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
for x_, v_, c_ in [(1, gscore[ctrl], "#9a9a9a"), (2, gscore[top_r], CORAL)]:
    m_, lo_, hi_ = mci(v_); ia.bar(x_, m_, width=0.55, color=c_, alpha=0.85, edgecolor=INK)
    ia.errorbar(x_, m_, yerr=[[m_ - lo_], [hi_ - m_]], color=INK, capsize=6, lw=1.4)
ia.axhline(0, color="#777777", lw=0.8)
ia.set_xticks([1, 2], [f"controls\n(n={len(ctrl):,})", "top-50\nreceivers"], color=INK, fontsize=8); ia.tick_params(colors=INK, labelsize=7.5)
ia.set_ylabel("literature response-gene score (z), mean ± 95% CI", color=INK, fontsize=7.5)
ia.set_title(f"one-sided Mann-Whitney p = {mw.pvalue:.2g}\nshown receiver: top {max(100 - pct, 1):.0f}% of its type", color=INK, fontsize=8)
for s_ in ia.spines.values(): s_.set_color("#555555")
top_pct = 100 * cs.rank_all / cs.n_events
fmt_pct = lambda v: f"{v:.{max(0, -int(np.floor(np.log10(v))) + 1)}f}"
fig.text(0.5, 0.99, f"Predicted CCC event (rank {cs.rank_all:,} of {cs.n_events:,}; top {fmt_pct(top_pct)}%): "
         f"\n{'+'.join(lig_genes)} ({types[ctype[si]]}) → {'+'.join(rec_genes)} ({types[ctype[ri]]})", color=INK, ha="center", va="top", fontsize=12, fontweight="bold")
fig.text(0.5, 0.925, f"{chn.split('->')[1].replace('_', '/')} → {relay or 'receiver response'}  |  CosMx breast RNA+protein, FOV {fov}\n"
         f"literature response genes: {', '.join(up[:12])}{' …' if len(up) > 12 else ''}", color=MUTE, ha="center", va="top", fontsize=8.5)
foot = (f"GraTICCS graph transformer, held-out FOVs (5-fold spatial cross-fitting, seed ensemble). Channel {chn}: decoy p = {cs.p:.3g}, q = {cs.q:.2f} "
        f"(1 − q = {1 - cs.q:.2f}); best readout {cs.best_readout}. Sender-receiver distance {cs.dist_um:.1f} µm.\n"
        f"Cell fill = per-cell mean immunofluorescence (red CD45, green PanCK, blue DAPI); transcripts drawn at their measured CosMx coordinates.\n"
        f"Literature (836 curated records, exact {chn.replace('_', '/')} match; {len(lit['papers'])} papers): " + "; ".join(cites[:3]) +
        ".\nResponse genes = receiver effectors reported as up-regulated in these papers and measured on the panel. Event shown = among the 50 best-ranked events" + (" with >=2 ligand and >=1 receptor transcripts" if os.environ.get("VIS", "1") == "1" else "") + ", the one whose receiver has the highest response-gene score;\nthe inset test uses the 50 best-ranked receivers without regard to response. Supportive, not causal, evidence.")
fig.text(0.5, 0.012, foot, color=MUTE, ha="center", va="bottom", fontsize=7.2, wrap=True)
fig.savefig(outp, dpi=220, facecolor=BG); print("saved", outp, "| n_up genes", len(up), "| MW p", mw.pvalue, "| receiver pct", round(pct))
json.dump(dict(channel=chn, sender=cs.sender, receiver=cs.receiver, fov=fov, rank_all=int(cs.rank_all), n_events=int(cs.n_events), p=float(cs.p), q=float(cs.q),
               best_readout=cs.best_readout, response_genes=up, n_resp_genes_local=int(n_dg), n_resp_tx_local=int(n_d), mw_p=float(mw.pvalue),
               receiver_percentile=float(pct), citations=cites, relay=relay), open(Path(outp).with_suffix(".json"), "w"), indent=1)
