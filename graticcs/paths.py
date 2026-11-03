import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _env(name, default):
    return Path(os.environ.get(name, default))


BREAST = _env("GRATICCS_BREAST", ROOT / "data" / "breast")
XENIUM = _env("GRATICCS_XENIUM", ROOT / "data" / "xenium_kidney")
RAW = _env("GRATICCS_RAW", ROOT / "raw")
RESULTS = _env("GRATICCS_RESULTS", ROOT / "results")
FIGURES = _env("GRATICCS_FIGURES", ROOT / "figures")
