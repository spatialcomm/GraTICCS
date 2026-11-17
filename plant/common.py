"""Shared pieces of the planted-signal scripts.

A planted signal for channel c on readout t adds  sign * beta * sd(y_t) * z(x)  to the measured readout, where x is the
receiver's exposure to the channel (R_i times the summed ligand of its senders) and z standardises over cells.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from graticcs.paths import RESULTS


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


DRAWS = Path(__file__).resolve().parent / "draws"


def table(f):
    """A table of the draw: results/ first, else the version used in the paper (plant/draws/)."""
    return pd.read_csv(RESULTS / f if (RESULTS / f).exists() else DRAWS / f)


def drawn(kind, file=None):
    """(channel, readout, sign) of the drawn planted pairs."""
    d = table(file or f"planted_random_{kind}.csv")
    return [(r.channel, r.readout, int(r.sign)) for r in d.itertuples()]


def contact_inputs(D):
    z = np.load(D / "gt_inputs.npz"); meta = json.load(open(D / "gt_channels.json"))
    names = [c["name"] for c in meta["channels"]]; tnames = meta["prot_readout"] + meta["rna_programs"]
    n = len(z["fold"]); ci = z["contact"] > 0
    A = sp.csr_matrix((np.ones(ci.sum(), np.float32), (z["dst"][ci], z["src"][ci])), shape=(n, n))   # receiver x sender
    Y0 = np.c_[z["Yp"], z["Yr"]].astype(np.float64)
    return z, meta, names, tnames, A, Y0


def secreted_inputs(D):
    S = np.load(D / "gt_sec_inputs.npz"); m = json.load(open(D / "gt_sec_channels.json"))
    names = [c["name"] for c in m["channels"]]; tnames = m["prot_readout"] + m["rna_programs"]; n = S["L"].shape[0]
    W = sp.csr_matrix((np.exp(-S["dist"] / 15.0), (S["dst"], S["src"])), shape=(n, n))
    Y0 = np.c_[S["Yp"], S["Yr"]].astype(np.float64)
    return S, m, names, tnames, W, Y0


def add_signals(Y0, tnames, plant, sig, beta):
    Y = Y0.copy()
    for p in plant:
        chn, tgt, sgn = p[:3]
        t = tnames.index(tgt); Y[:, t] += sgn * beta * Y0[:, t].std() * sig[chn]
    return Y.astype(np.float32)


def save(D, prefix, Y, beta):
    np.savez_compressed(D / f"{prefix}{beta}.npz", Y=Y)


def within_group_permutation(fov, ctype, seed=7):
    """Permutation of cells within each (imaging field, cell type) group."""
    rng = np.random.default_rng(seed); groups = {}
    for i, key in enumerate(zip(fov, ctype)):
        groups.setdefault(key, []).append(i)
    perm = np.arange(len(fov))
    for idx in groups.values():
        idx = np.array(idx); perm[idx] = idx[rng.permutation(len(idx))]
    return perm
