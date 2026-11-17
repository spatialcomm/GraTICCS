"""Published evidence behind each drawn planted pair: the receptor-to-pathway assignment, the pathway effector TF and the
CollecTRI regulation of the readout gene with its PubMed IDs, PROGENy where it applies, and literature records.
Writes results/planted_random_route_refs.csv.
"""
import re

import decoupler as dc
import pandas as pd

from graticcs.pathways import PATH_TF, families
from graticcs.paths import RESULTS
from plant.common import table

GENE = {"GZMB": "GZMB", "GZMA": "GZMA", "Ki-67": "MKI67", "iNOS": "NOS2", "IL-1b": "IL1B", "NF-kB p65": "RELA",
        "HLA-DR": "HLA-DRA", "IDO1": "IDO1", "CD38": "CD38", "ICAM1": "ICAM1", "PD-L1": "CD274"}
INHIBITORY = {"PDCD1", "HAVCR2", "CTLA4", "LAG3", "TIGIT", "BTLA", "CD96", "CD160"}
col = dc.op.collectri(organism="human"); prog = dc.op.progeny(organism="human", top=500)
ref_col = "references" if "references" in col.columns else None

rows = []
for mod in ("contact", "secreted"):
    for r in table(f"planted_random_{mod}.csv").itertuples():
        g = GENE.get(r.readout, r.readout); inhib = any(p in INHIBITORY for p in r.receptor.split("_")); fam = families(r.receptor)
        tfs, pmids = [], set()
        for f in sorted(fam):
            for t in PATH_TF.get(f, []):
                e = col[(col.source == t) & (col.target == g)]
                if len(e):
                    w = float(e.weight.iloc[0]); tfs.append(f"{f}/{t} ({'+' if w > 0 else '-'})")
                    if ref_col:
                        pmids |= {x.strip() for x in str(e[ref_col].iloc[0]).replace(",", ";").split(";") if x.strip() and x.strip() != "nan"}
        pg = [f"{f} ({'+' if w > 0 else '-'})" for f in sorted(fam) for w in prog[(prog.source == f) & (prog.target == g)].weight]
        lit = []
        for rec in str(r.lit_records).split(";"):
            m = re.match(r"(doi|pmcid)_(.+?)__", rec)
            if m:
                lit.append(("doi:" + m.group(2).replace("_", "/", 1)) if m.group(1) == "doi" else m.group(2).replace("pmcid_", ""))
        rows.append(dict(modality=mod, literature=";".join(lit), channel=r.channel, readout=r.readout, readout_gene=g, sign=r.sign,
                         inhibitory_receptor=inhib, pathway_families=";".join(sorted(fam)), collectri_routes="; ".join(tfs),
                         progeny=";".join(pg), n_pmid=len(pmids), pmids=";".join(sorted(pmids, key=lambda x: (len(x), x)))))
d = pd.DataFrame(rows); d.to_csv(RESULTS / "planted_random_route_refs.csv", index=False)
print(d.drop(columns="pmids").to_string(index=False))
