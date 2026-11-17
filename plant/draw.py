"""Pool of channel-readout pairs eligible for planted signals, and the random draw from it.

A pair is supported when literature records, or PROGENy / CollecTRI with a single direction, link the receptor's pathway to
the readout, and the ligand-receptor pair itself is listed in an expert-curated resource (CellPhoneDB, CellChatDB,
ICELLNET, connectomeDB2020, Guide to Pharmacology). Eligible pairs also need 2-15% exposed receivers, no recovery in the
unplanted tissue, and must not be among the channels examined in the biological assessment.

python -m plant.draw contact pool [runs_dir]   # writes results/planted_pool_contact.csv
python -m plant.draw contact draw              # writes results/planted_random_contact.csv

The pool was built before the final configuration was fixed, from the original-tissue runs and the biological screen
available at that time (four attention heads). Those screen lists are in plant/draws/assessed_at_draw/; runs_dir points to
the corresponding original-tissue runs. The pools and draws used in the paper are in plant/draws/.
"""
import json
import sys
from pathlib import Path

import liana as li
import numpy as np
import pandas as pd
import scipy.sparse as sp
import decoupler as dc

from graticcs.paths import BREAST, RESULTS, XENIUM
from graticcs.pathways import PATH_TF, families

KIND = sys.argv[1]
D = XENIUM if KIND == "xenium" else BREAST
R = RESULTS
SEED = 20261112
INHIBITORY = {"PDCD1", "HAVCR2", "CTLA4", "LAG3", "TIGIT", "BTLA", "CD96", "CD160"}
if KIND == "xenium":
    GENE = {"GranzymeB": "GZMB", "Ki-67": "MKI67", "PCNA": "PCNA", "HLA-DR": "HLA-DRA", "PD-L1": "CD274"}
else:
    GENE = {"GZMB": "GZMB", "GZMA": "GZMA", "Ki-67": "MKI67", "iNOS": "NOS2", "IL-1b": "IL1B", "NF-kB p65": "RELA",
            "HLA-DR": "HLA-DRA", "IDO1": "IDO1", "CD38": "CD38", "ICAM1": "ICAM1", "PD-L1": "CD274"}

# channels examined in the biological assessment are not used for planting
ASSESSED = {
    "contact": ["dose_response_contact_top10_v2.csv", "dose_response_contact_rank11_50_v2.csv"],
    "secreted": ["dose_response_top20_v2.csv", "dose_response_sec_rank21_50_v2.csv"],
    "xenium": [],
}[KIND]
# channels and genes of the original-tissue checks (contact) and dominant ligands (secreted) are left out
NATURAL = {"HLA-A->CD8A", "HLA-DRA->CD4"} if KIND == "contact" else set()
EXCL_GENE = {"HLA-DQA1", "HLA-DQB1", "HLA-DRB1", "HLA-DRA", "CD4", "LGALS9", "HAVCR2", "CD274", "PDCD1"} if KIND != "secreted" else set()
EXCL_LIG = {"IL1B", "FN1", "CXCL10", "CCL5"} if KIND == "secreted" else set()
REAL_RUN = {"contact": "G_real", "secreted": "G2_sec_real", "xenium": "G_xen_real"}[KIND]

prog = dc.op.progeny(organism="human", top=500)
col = dc.op.collectri(organism="human")
pw = {(r.source, r.target): float(r.weight) for r in prog.itertuples()}
tf = {(r.source, r.target): float(r.weight) for r in col.itertuples()}

CUR = {}
for res in ("cellphonedb", "cellchatdb", "icellnet", "connectomedb2020", "guide2pharma"):
    for a, b in li.resource.select_resource(res)[["ligand", "receptor"]].itertuples(index=False):
        CUR.setdefault(a, []).append((set(b.split("_")), res))


def curated(lig, rec):
    rs = set(rec.split("_")) - {lig}
    return ";".join(sorted({res for b, res in CUR.get(lig, []) if b <= rs or rs <= b}))


def pathway_support(fam, gene):
    rows = []
    for f in sorted(fam):
        if (f, gene) in pw:
            rows.append(("PROGENy", pw[(f, gene)]))
        for t in PATH_TF.get(f, []):
            if (t, gene) in tf:
                rows.append(("CollecTRI", tf[(t, gene)]))
    return rows


def load_inputs():
    m = json.load(open(D / ("gt_sec_channels.json" if KIND == "secreted" else "gt_channels.json")))
    ch = pd.DataFrame(m["channels"])
    if "decoy_modality" not in ch:
        ch["decoy_modality"] = np.where(ch.name.str.startswith("PDECOY"), "protein", "rna")
    prot_real = set(ch.loc[ch.decoy & (ch.decoy_modality == "protein"), "of"])
    ch["stratum"] = np.where(ch.decoy, ch.decoy_modality.fillna("rna"), np.where(ch.name.isin(prot_real), "protein", "rna"))
    if KIND == "secreted":
        S = np.load(D / "gt_sec_inputs.npz"); L, Rc = S["L"], S["R"]; n = L.shape[0]
        W = sp.csr_matrix((np.exp(-S["dist"] / 15.0), (S["dst"], S["src"])), shape=(n, n))
    else:
        S = np.load(D / "gt_inputs.npz"); L, Rc = S["L_mo"], S["R_mo"]; n = L.shape[0]; ci = S["contact"] > 0
        W = sp.csr_matrix((np.ones(ci.sum(), np.float32), (S["dst"][ci], S["src"][ci])), shape=(n, n))
    return m, ch, L, Rc, W


def pool():
    m, ch, L, Rc, W = load_inputs()
    runs = Path(sys.argv[3]) if len(sys.argv) > 3 else R
    fo = [json.load(open(p)) for s in range(3) for p in sorted((runs / f"{REAL_RUN}_s{s}").glob("fold*_mo.json"))]
    assert len(fo) == 15
    T = fo[0]["targets"]
    imp = np.sum([f["dsse_knockout"] for f in fo], 0) / np.sum([f["sse_nuisance"] for f in fo], 0)
    lit = {} if KIND == "xenium" else json.load(open(D / "lit_channel_matches.json"))[KIND]
    prot = {v: k for k, v in GENE.items()}
    prog_of = {g: p for p, gs in m["rna_program_genes"].items() for g in gs}
    screen = Path(__file__).resolve().parent / "draws" / "assessed_at_draw"
    assessed = set(pd.concat([pd.read_csv(screen / f) for f in ASSESSED]).channel) if ASSESSED else set()

    rows = []
    for i, c in ch[~ch.decoy].iterrows():
        nm, rec, lig = c["name"], c["receptor"], c["ligand"]
        subs = set(rec.split("_")) | set(lig.split("_"))
        r = np.maximum(Rc[:, i].astype(np.float32), 0)
        nb = W @ np.maximum(L[:, i].astype(np.float32), 0)
        x = r * nb; e = x > 0
        info = dict(lr_curated=curated(lig, rec),
                    lit_records=";".join(sorted({f for ev in lit.get(nm, {}).get("effectors", {}).values() for _, f in ev})),
                    exposed=float(e.mean()), corr_own=float(np.corrcoef(x, r)[0, 1]) if e.any() else np.nan,
                    lig_cv=float(nb[e].std() / nb[e].mean()) if e.any() else np.nan)
        sup = {}   # readout -> {source: set of signs}
        for g, ev in lit.get(nm, {}).get("effectors", {}).items():
            for t in {prot.get(g), prog_of.get(g)} - {None}:
                sup.setdefault(t, {}).setdefault("literature", set()).update({+1 if d == "up" else -1 for d, _ in ev})
        fam = families(rec); inhib = any(p in INHIBITORY for p in rec.split("_"))
        if fam:
            for t, g in GENE.items():
                if g in subs:
                    continue
                for src, w in pathway_support(fam, g):
                    if w:
                        sup.setdefault(t, {}).setdefault(src, set()).add(int(np.sign(w)) * (-1 if inhib else 1))
        st = c["stratum"]
        dec = ch.index[ch.decoy & (ch.stratum == st)]; rl = ch.index[~ch.decoy & (ch.stratum == st)]
        for t, srcs in sup.items():
            signs = set().union(*srcs.values()); ti = T.index(t)
            p = (1 + (imp[dec, ti] >= imp[i, ti]).sum()) / (1 + len(dec))
            rk = int((imp[rl, ti] > imp[i, ti]).sum()) + 1
            rows.append(dict(channel=nm, ligand=lig, receptor=rec, readout=t, sources=";".join(sorted(srcs)),
                             sign=signs.pop() if len(signs) == 1 else 0, stratum=st, base_p=round(p, 3), base_rank=rk,
                             active=bool(p <= 0.05 and rk <= 3), top50=nm in assessed,
                             validation=bool(nm in NATURAL or subs & EXCL_GENE), **info))
    d = pd.DataFrame(rows)
    ok = ((d.sign != 0) & d.exposed.between(0.02, 0.15) & ~d.active & ~d.top50 & ~d.validation
          & (d.corr_own < 0.75) & (d.lr_curated.fillna("") != ""))
    if KIND == "secreted":
        ok &= ~d.ligand.isin(EXCL_LIG) & (d.lig_cv >= 0.6)
    d["eligible"] = ok
    d.to_csv(R / f"planted_pool_{KIND}.csv", index=False)
    e = d[d.eligible]
    print(len(d), "supported pairs;", len(e), "eligible;", e.channel.nunique(), "channels;", e.ligand.nunique(), "ligands;",
          e.receptor.nunique(), "receptors;", e.readout.nunique(), "readouts")


def one_batch(e, skip):
    """Walk the shuffled pool; one readout per channel, ligands and receptors not repeated within the batch."""
    used_l, used_r, out = set(), set(), []
    for r in e.itertuples():
        rs = set(r.receptor.split("_"))
        if r.channel in skip or r.ligand in used_l or rs & used_r:
            continue
        out.append(r); used_l.add(r.ligand); used_r |= rs
        if len(out) == N_MAX:
            break
    return pd.DataFrame(out).drop(columns="Index")


N_MAX = 20


def draw():
    """At most 20 pairs. When the pool has no more channels than that, every channel is tested, and channels that
    conflict with an earlier pick go to a further batch (planted_random_<kind>_batch2.csv, ...)."""
    e = pd.read_csv(R / f"planted_pool_{KIND}.csv")
    e = e[e.eligible].sample(frac=1, random_state=SEED).reset_index(drop=True)
    batches = [one_batch(e, set())]
    if e.channel.nunique() <= N_MAX:
        while (done := set(pd.concat(batches).channel)) != set(e.channel):
            batches.append(one_batch(e, done))
    for k, o in enumerate(batches):
        o.to_csv(R / (f"planted_random_{KIND}.csv" if k == 0 else f"planted_random_{KIND}_batch{k + 1}.csv"), index=False)
        print(f"batch {k + 1}\n", o[["channel", "readout", "sign", "sources", "exposed", "base_p", "base_rank"]].to_string())


if __name__ == "__main__":
    {"pool": pool, "draw": draw}[sys.argv[2]]()
