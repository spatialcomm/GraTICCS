"""Receptor-family rules: which PROGENy pathways and CollecTRI TFs a receptor may explain."""
import re

PATH_TF = {"JAK-STAT": ["STAT1", "STAT3", "IRF1", "STAT5A"], "NFkB": ["RELA", "NFKB1", "REL"], "TNFa": ["RELA", "NFKB1"],
           "TGFb": ["SMAD3", "SMAD4", "SMAD2"], "MAPK": ["JUN", "FOS", "ELK1", "EGR1"], "PI3K": ["FOXO1", "FOXO3"],
           "EGFR": ["JUN", "FOS", "ELK1"], "VEGF": ["ETS1", "JUN"], "WNT": ["TCF7", "LEF1", "TCF7L2"], "Hypoxia": ["HIF1A", "EPAS1"],
           "p53": ["TP53"], "Trail": ["RELA"], "Estrogen": ["ESR1"], "Androgen": ["AR"], "TCELL": ["NFATC1", "NFATC2", "JUN", "FOS", "RELA"]}
def families(rec):
    """receptor gene (or complex) -> set of PROGENy pathways / 'TCELL' by textbook family rules."""
    out = set()
    for g in rec.split("_"):
        if re.match(r"^TNFRSF10[A-D]$", g): out |= {"Trail", "NFkB"}
        elif re.match(r"^(TNFRSF|LTBR|EDAR|RELT)", g): out |= {"TNFa", "NFkB"}
        elif re.match(r"^(IL1R|IL18R|IL1RAP|IL1RL|SIGIRR|TLR|MYD88)", g): out |= {"NFkB", "MAPK"}
        elif re.match(r"^(IL\d+R|IL2RG|IL6ST|IFNAR|IFNGR|IFNLR|CSF2R|CSF3R|LIFR|OSMR|CNTFR|GHR|PRLR|EPOR|MPL|CRLF2|IL31RA)", g): out |= {"JAK-STAT"}
        elif re.match(r"^(TGFBR|ACVR|BMPR|ENG|AMHR2)", g): out |= {"TGFb"}
        elif re.match(r"^(EGFR|ERBB)", g): out |= {"EGFR", "MAPK", "PI3K"}
        elif re.match(r"^(KDR|FLT1|FLT4|NRP1|NRP2)$", g): out |= {"VEGF", "MAPK", "PI3K"}
        elif re.match(r"^(MET|PDGFR|FGFR|IGF1R|INSR|KIT|AXL|MERTK|TYRO3|NTRK|RET|ROR|DDR|EPH|CSF1R|ALK|LTK|MST1R|TEK|TIE)", g): out |= {"MAPK", "PI3K"}
        elif re.match(r"^(CCR|CXCR|ACKR|CX3CR|XCR|GPR|ADRB|ADRA|PTGER|S1PR|LPAR|CMKLR|FPR|C3AR|C5AR|EDNR|AGTR|CALCR|RAMP|ADORA|P2RY)", g): out |= {"MAPK", "PI3K"}
        elif re.match(r"^(ITG|CD44$|SDC|GPC|CD47$|SIRPA)", g): out |= {"PI3K", "MAPK"}
        elif re.match(r"^(FZD|LRP5$|LRP6$|ROR1|ROR2|RYK)", g): out |= {"WNT"}
        elif g in {"PDCD1", "CTLA4", "CD28", "ICOS", "CD4", "CD8A", "CD8B", "LAG3", "HAVCR2", "TIGIT", "CD226", "CD96", "CD2", "CD27",
                   "TNFRSF9", "TNFRSF4", "TNFRSF18", "CD3E", "CD3D", "KLRK1", "KIR2DS4", "KIR3DL2", "CD247", "BTLA", "CD160", "CD5", "CD6"}:
            out |= {"TCELL"}
    return out
