"""Table of the receptor-family rules (Supplementary Table S21)."""
import json, re
from pathlib import Path
import pandas as pd
from graticcs.paths import BREAST, RESULTS
D = BREAST; R = RESULTS
from graticcs.pathways import PATH_TF, families
RULES = [("TRAIL receptors", r"^TNFRSF10[A-D]$", "Walczak 2013"),
         ("TNF-receptor superfamily", r"^(TNFRSF|LTBR|EDAR|RELT)", "Hayden and Ghosh 2014"),
         ("IL-1/IL-18 receptors, TLRs", r"^(IL1R|IL18R|IL1RAP|IL1RL|SIGIRR|TLR|MYD88)", "Weber et al. 2010; Kawai and Akira 2010"),
         ("type I/II cytokine receptors", r"^(IL\d+R|IL2RG|IL6ST|IFNAR|IFNGR|IFNLR|CSF2R|CSF3R|LIFR|OSMR|CNTFR|GHR|PRLR|EPOR|MPL|CRLF2|IL31RA)", "Villarino et al. 2017"),
         ("TGF-beta/BMP/activin receptors", r"^(TGFBR|ACVR|BMPR|ENG|AMHR2)", "Massague 2012"),
         ("EGFR/ERBB", r"^(EGFR|ERBB)", "Lemmon and Schlessinger 2010"),
         ("VEGF receptors, neuropilins", r"^(KDR|FLT1|FLT4|NRP1|NRP2)$", "Simons et al. 2016"),
         ("other receptor tyrosine kinases", r"^(MET|PDGFR|FGFR|IGF1R|INSR|KIT|AXL|MERTK|TYRO3|NTRK|RET|ROR|DDR|EPH|CSF1R|ALK|LTK|MST1R|TEK|TIE)", "Lemmon and Schlessinger 2010"),
         ("chemokine receptors and other GPCRs", r"^(CCR|CXCR|ACKR|CX3CR|XCR|GPR|ADRB|ADRA|PTGER|S1PR|LPAR|CMKLR|FPR|C3AR|C5AR|EDNR|AGTR|CALCR|RAMP|ADORA|P2RY)", "Thelen 2001"),
         ("integrins and adhesion/matrix receptors", r"^(ITG|CD44$|SDC|GPC|CD47$|SIRPA)", "Hynes 2002"),
         ("Frizzled, LRP5/6, ROR, RYK", r"^(FZD|LRP5$|LRP6$|ROR1|ROR2|RYK)", "Clevers and Nusse 2012"),
         ("T-cell receptors and co-receptors", "listed genes (PDCD1, CTLA4, CD28, ICOS, CD4, CD8A/B, LAG3, HAVCR2, TIGIT, CD226, CD96, CD2, CD27, TNFRSF9/4/18, CD3D/E, KLRK1, ...)", "Smith-Garvin et al. 2009")]
probe = {"TRAIL receptors": "TNFRSF10B", "TNF-receptor superfamily": "TNFRSF1A", "IL-1/IL-18 receptors, TLRs": "IL1R1",
         "type I/II cytokine receptors": "IL6R", "TGF-beta/BMP/activin receptors": "TGFBR1", "EGFR/ERBB": "EGFR", "VEGF receptors, neuropilins": "KDR",
         "other receptor tyrosine kinases": "MET", "chemokine receptors and other GPCRs": "CXCR4", "integrins and adhesion/matrix receptors": "ITGB1",
         "Frizzled, LRP5/6, ROR, RYK": "FZD1", "T-cell receptors and co-receptors": "PDCD1"}


def rule_of(gene):
    for name, pat, _ in RULES[:-1]:
        if re.match(pat, gene): return name
    return RULES[-1][0] if families(gene) == {"TCELL"} else None


rows = []
cnt = {}
for kind, f in [("contact", "gt_channels.json"), ("secreted", "gt_sec_channels.json")]:
    chs = [c for c in json.load(open(D / f))["channels"] if not c["decoy"]]
    for c in chs:
        hit = {rule_of(g) for g in c["receptor"].split("_")} - {None}
        for h in (hit or {"no rule (no pathway/TF readout)"}): cnt[(kind, h)] = cnt.get((kind, h), 0) + 1
    cnt[(kind, "total")] = len(chs)
for name, pat, ref in RULES:
    fam = families(probe[name]); pws = sorted(f for f in fam if f != "TCELL")
    tfs = sorted({t for f in fam for t in PATH_TF.get(f, [])})
    rows.append(dict(rule=name, receptor_genes=pat, pathways=", ".join(pws) if pws else "-", TFs=", ".join(tfs), reference=ref,
                     contact=cnt.get(("contact", name), 0), secreted=cnt.get(("secreted", name), 0)))
rows.append(dict(rule="no rule matched", receptor_genes="-", pathways="none", TFs="none", reference="-",
                 contact=cnt.get(("contact", "no rule (no pathway/TF readout)"), 0), secreted=cnt.get(("secreted", "no rule (no pathway/TF readout)"), 0)))
df = pd.DataFrame(rows); df.to_csv(R / "receptor_mask_rules.csv", index=False)
print(df[["rule", "pathways", "TFs", "contact", "secreted"]].to_string(index=False))
print("real channels: contact", cnt[("contact", "total")], "secreted", cnt[("secreted", "total")])
