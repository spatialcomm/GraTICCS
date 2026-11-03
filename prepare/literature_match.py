"""Match the curated literature records to the real LR channels and list measurable receiver effectors."""
import json, re, os, sys
from collections import defaultdict, Counter
from pathlib import Path
from graticcs.paths import BREAST, RAW
B = RAW
REC = RAW / "literature_records"
ALIAS = {'IL8': 'CXCL8', 'WARS': 'WARS1', 'TNF-ALPHA': 'TNF', 'TNFA': 'TNF', 'IFN-GAMMA': 'IFNG', 'TGF-B1': 'TGFB1', 'TGFB': 'TGFB1',
         'PD-L1': 'CD274', 'PD-1': 'PDCD1', 'PDL1': 'CD274', 'PD1': 'PDCD1', 'GALECTIN-9': 'LGALS9', 'TIM-3': 'HAVCR2'}
sym = lambda g: ALIAS.get(str(g).strip().upper(), str(g).strip().upper())
def split(s): return [sym(x) for x in re.split(r"[;/,+]| and ", str(s or "")) if x.strip()]
D = BREAST; genes = {g.upper() for g in json.load(open(D / "genes.json"))["genes"]}
recs = {fn: json.load(open(REC / fn, encoding="utf-8-sig")) for fn in os.listdir(REC)}
out = {}
for tag, f in [("contact", "gt_channels.json"), ("secreted", "gt_sec_channels.json")]:
    meta = json.load(open(D / f)); ch = [c for c in meta["channels"] if not c["decoy"]]
    hits = defaultdict(lambda: {"papers": [], "eff": defaultdict(list)})
    for fn, r in recs.items():
        e = r.get("main_ccc_edge") or {}; L, Rr = split(e.get("ligand")), split(e.get("receptor"))
        for c in ch:
            if c["ligand"] in L and any(p in Rr for p in c["receptor"].split("_")):
                h = hits[c["name"]]; ref = r.get("reference") or {}
                h["papers"].append({"file": fn, "doi": ref.get("doi"), "pmid": ref.get("pmid"), "title": ref.get("title")})
                for eff in r.get("downstream_effectors") or []:
                    if isinstance(eff, dict) and eff.get("measured_in") == "receiver":
                        h["eff"][sym(eff.get("gene", ""))].append((str(eff.get("direction", "")).lower(), fn))
    out[tag] = {k: {"papers": v["papers"], "effectors": {g: x for g, x in v["eff"].items()}} for k, v in hits.items()}
    print(f"== {tag}: real channels {len(ch)}; with >=1 paper on the exact L->R: {len(hits)}")
    for nm, h in sorted(hits.items(), key=lambda x: -len(x[1]["papers"]))[:25]:
        meas = sorted(((g, Counter(d[:2] for d, _ in x)) for g, x in h["eff"].items() if g in genes), key=lambda t: -sum(t[1].values()))
        print(f"  {nm:28s} papers {len(h['papers']):2d}  measurable effectors {len(meas):2d}: " + ", ".join(f"{g}{dict(c)}" for g, c in meas[:6]))
json.dump(out, open(D / "lit_channel_matches.json", "w", encoding="utf8"), indent=1, ensure_ascii=False)
