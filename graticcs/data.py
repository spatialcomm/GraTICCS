import json
import os
from pathlib import Path

import numpy as np
import torch

from graticcs.lr_map import PROT2GENE
from graticcs.paths import BREAST

DATA = Path(os.environ.get("GRATICCS_DATASET", BREAST))
dev = "cuda" if torch.cuda.is_available() else "cpu"

FAMILY = {"HLA-DR": lambda L: any(g.startswith("HLA-D") for g in L), "PD-L1": lambda L: bool({"CD274", "PDCD1LG2"} & set(L)),
          "ICAM1": lambda L: bool({"ICAM1", "ICAM2", "ICAM3"} & set(L)), "CD38": lambda L: "CD38" in L}


def load(cfg="mo", data=DATA):
    z = np.load(data / "gt_inputs.npz")
    meta = json.load(open(data / "gt_channels.json"))
    n = len(z["fold"]); types = meta["types"]
    oh = np.eye(len(types), dtype=np.float32)[z["ctype"]]
    if cfg == "mo":
        x = np.c_[z["rna_pc"], z["prot_in"], oh, z["covar"]]; Y = np.c_[z["Yp"], z["Yr"]]
        tnames = meta["prot_readout"] + meta["rna_programs"]; L, R = z["L_mo"], z["R_mo"]
    else:
        x = np.c_[z["rna_pc"], z["covar"]]; Y = z["Yr"].copy(); tnames = meta["rna_programs"]; L, R = z["L_rna"], z["R_rna"]
    return dict(x=x.astype(np.float32), Y=Y.astype(np.float32), L=L, R=R, src=z["src"], dst=z["dst"], contact=z["contact"],
                dist=z["dist"], fold=z["fold"], fov=z["fov"], Yp=z["Yp"], n=n, tnames=tnames, channels=meta["channels"])


def batches(fovs, per=4, shuffle=False):
    # fixed FOV groups so the subgraph cache stays bounded; only the group order is shuffled
    fovs = np.sort(np.array(fovs)); groups = [fovs[s:s + per] for s in range(0, len(fovs), per)]
    order = np.random.permutation(len(groups)) if shuffle else range(len(groups))
    for i in order:
        yield groups[i]


class Graph:
    def __init__(self, d):
        self.d = d; self.node_by_fov = {f: np.flatnonzero(d["fov"] == f) for f in np.unique(d["fov"])}
        efov = d["fov"][d["dst"]]; order = np.argsort(efov, kind="stable")
        u, start = np.unique(efov[order], return_index=True); end = np.r_[start[1:], len(order)]
        self.edge_by_fov = {f: order[s:e] for f, s, e in zip(u, start, end)}

    def sub(self, fovs, Ystd):
        d = self.d
        nodes = np.concatenate([self.node_by_fov[f] for f in fovs])
        edges = np.concatenate([self.edge_by_fov.get(f, np.empty(0, int)) for f in fovs])
        remap = -np.ones(d["n"], np.int64); remap[nodes] = np.arange(len(nodes))
        src, dst = remap[d["src"][edges]], remap[d["dst"][edges]]
        con = d["contact"][edges]; dist = d["dist"][edges]
        eattr = np.c_[np.log1p(con), dist / 15.0, (con > 0).astype(np.float32)].astype(np.float32)
        ci = np.flatnonzero(con > 0)
        # spillover covariate: contact-weighted mean of the neighbours' observed readouts
        w = con[ci]; spill = np.zeros((len(nodes), Ystd.shape[1]), np.float32); tot = np.zeros(len(nodes), np.float32)
        np.add.at(spill, dst[ci], w[:, None] * Ystd[nodes][src[ci]]); np.add.at(tot, dst[ci], w)
        spill /= np.maximum(tot, 1e-6)[:, None]
        Ls = d["L"][nodes][src[ci]]; Rs = d["R"][nodes][dst[ci]]
        s = (np.log1p(np.exp(np.clip(Ls, -10, 10))) * np.log1p(np.exp(np.clip(Rs, -10, 10)))).astype(np.float32)
        T = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt)
        return dict(nodes=nodes, x=T(d["x"][nodes]), y=T(Ystd[nodes]), ei=T(np.vstack([src, dst]), torch.long),
                    eattr=T(eattr), spill=T(spill), csrc=T(src[ci], torch.long), cdst=T(dst[ci], torch.long),
                    ceattr=T(eattr[ci][:, :2]), s=T(s, torch.float16), cedge=edges[ci])


def comp_feats(g, ctype, ntypes):
    """Neighbourhood composition: contact-weighted and 15 um cell-type fractions, contact counts."""
    src, dst = g["ei"][0].numpy(), g["ei"][1].numpy(); con = np.expm1(g["eattr"][:, 0].numpy())
    nodes = g["nodes"]; n = len(nodes); oh = np.eye(ntypes, dtype=np.float32)[ctype[nodes]]
    cw = np.zeros((n, ntypes), np.float32); np.add.at(cw, dst, con[:, None] * oh[src])
    tot = np.zeros(n, np.float32); np.add.at(tot, dst, con)
    allf = np.zeros((n, ntypes), np.float32); np.add.at(allf, dst, oh[src]); cnt = allf.sum(1)
    ncon = np.zeros(n, np.float32); np.add.at(ncon, dst, (con > 0).astype(np.float32))
    return np.c_[cw / np.maximum(tot, 1e-6)[:, None], allf / np.maximum(cnt, 1)[:, None], np.log1p(ncon), np.log1p(tot), np.log1p(cnt)]


def target_mask(d):
    """Channel x readout mask: a channel may not explain its own ligand/receptor protein (or that protein's family)."""
    if "pw_mask" in d:
        dd = dict(d); dd["tnames"] = d["tnames"][:d["n_prot"]]; dd.pop("pw_mask")
        return np.c_[target_mask(dd), d["pw_mask"]].astype(np.float32)
    C = d["L"].shape[1]; M = np.ones((C, len(d["tnames"])), np.float32)
    for c, ch in enumerate(d["channels"]):
        lig = ch["ligand"].split("_"); rec = ch["receptor"].split("_")
        for t, name in enumerate(d["tnames"]):
            gene = PROT2GENE.get(name)
            if gene and (gene in lig or gene in rec): M[c, t] = 0
            if name in FAMILY and (FAMILY[name](lig) or FAMILY[name](rec)): M[c, t] = 0
    return M


def train_stage(params, step_fn, eval_fn, tr_fovs, va_fovs, module, per=4, accum=1, max_epochs=40, patience=5):
    """Early-stopped training; `accum` mini-batches of `per` FOVs per optimiser step."""
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4); best, bad, state = np.inf, 0, None
    for ep in range(max_epochs):
        opt.zero_grad(); n = 0
        for b in batches(tr_fovs, per, shuffle=True):
            (step_fn(b) / accum).backward(); n += 1
            if n % accum == 0: opt.step(); opt.zero_grad()
        if n % accum: opt.step(); opt.zero_grad()
        v = eval_fn(va_fovs)
        if v < best - 1e-5: best, bad = v, 0; state = {k: x.detach().clone() for k, x in module.state_dict().items()}
        else: bad += 1
        if bad >= patience: break
    module.load_state_dict(state)
    return best, ep + 1


def fold_split(d, k):
    fold = d["fold"]; test_f, val_f = k, (k + 1) % 5
    fov_of = lambda m: np.unique(d["fov"][m])
    trm = (fold != test_f) & (fold != val_f)
    return fov_of(trm), fov_of(fold == val_f), fov_of(fold == test_f), trm


def to_dev(g):
    return {k: (v.to(dev).float() if v.dtype == torch.float16 else v.to(dev)) if torch.is_tensor(v) else v for k, v in g.items()}
