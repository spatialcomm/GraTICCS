"""Planted contact signals whose response does not follow the model's additive form R x sum of contact L.

hill:  saturating response x^2 / (K^2 + x^2), x = R * sum of contact L, K = median of x > 0
gate:  binary AND gate without dose, 1[R > q75(R | R > 0)] * 1[max contact-neighbour L > q75(L | L > 0)]
decay: proximity rather than contact, R * sum over all neighbours within 15 um of L exp(-d / 8 um)
"""
import numpy as np
import scipy.sparse as sp

from graticcs.paths import BREAST
from plant.common import add_signals, contact_inputs, drawn, save, zscore

D = BREAST
plant = drawn("contact")
z, meta, names, tnames, A, Y0 = contact_inputs(D)
n = len(z["fold"]); src, dst, con, dist = z["src"], z["dst"], z["contact"] > 0, z["dist"]
W = sp.csr_matrix((np.exp(-dist / 8.0).astype(np.float32), (dst, src)), shape=(n, n))


def max_neighbour(L):
    m = np.zeros(n); np.maximum.at(m, dst[con], L[src[con]]); return m


sig = {}
for chn, tgt, sgn in plant:
    c = names.index(chn)
    L = np.maximum(z["L_mo"][:, c], 0).astype(np.float64); R = np.maximum(z["R_mo"][:, c], 0).astype(np.float64)
    x = R * (A @ L); K = np.median(x[x > 0]); h = x ** 2 / (K ** 2 + x ** 2)
    g = ((R > np.quantile(R[R > 0], 0.75)) & (max_neighbour(L) > np.quantile(L[L > 0], 0.75))).astype(float)
    dcy = R * (W @ L)
    sig[chn] = {"hill": zscore(h), "gate": zscore(g), "decay": zscore(dcy)}
    print(f"{chn:22s} gate on {g.mean():.3f}; corr with additive form: hill {np.corrcoef(h, x)[0, 1]:.2f} "
          f"gate {np.corrcoef(g, x)[0, 1]:.2f} decay {np.corrcoef(dcy, x)[0, 1]:.2f}")
for form in ["hill", "gate", "decay"]:
    for b in [0.2, 0.4]:
        save(D, f"spike_crand_mis_{form}_b", add_signals(Y0, tnames, plant, {c: s[form] for c, s in sig.items()}, b), b)
