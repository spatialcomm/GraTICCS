"""liana 1.7 indexes AnnData with pandas objects that anndata 0.12 rejects; numpy 2 aliases for COMMOT."""
import numpy as np, pandas as pd, anndata as ad
np.Inf = np.inf; np.NaN = np.nan
_orig = ad.AnnData.__getitem__
def _getitem(self, idx):
    conv = lambda i: list(i) if isinstance(i, (pd.Series, pd.Index)) else i
    idx = tuple(conv(i) for i in idx) if isinstance(idx, tuple) else conv(idx)
    return _orig(self, idx)
ad.AnnData.__getitem__ = _getitem
