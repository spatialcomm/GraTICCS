"""CosMx 64-plex protein names -> gene symbols, and ligand classes."""
import re

PROT2GENE = {
    "4-1BB": "TNFRSF9", "B7-H3": "CD276", "Bcl-2": "BCL2", "Beta-catenin": "CTNNB1", "CCR7": "CCR7", "CD11b": "ITGAM",
    "CD11c": "ITGAX", "CD123": "IL3RA", "CD127": "IL7R", "CD138": "SDC1", "CD14": "CD14", "CD16": "FCGR3A",
    "CD163": "CD163", "CD19": "CD19", "CD20": "MS4A1", "CD27": "CD27", "CD3": "CD3E", "CD31": "PECAM1", "CD34": "CD34",
    "CD38": "CD38", "CD39": "ENTPD1", "CD4": "CD4", "CD40": "CD40", "CD45": "PTPRC", "CD56": "NCAM1", "CD68": "CD68",
    "CD8": "CD8A", "CTLA4": "CTLA4", "EGFR": "EGFR", "EpCAM": "EPCAM", "FABP4": "FABP4", "FOXP3": "FOXP3",
    "Fibronectin": "FN1", "GITR": "TNFRSF18", "GZMA": "GZMA", "GZMB": "GZMB", "HLA-DR": "HLA-DRA", "Her2": "ERBB2",
    "ICAM1": "ICAM1", "ICOS": "ICOS", "IDO1": "IDO1", "IL-18": "IL18", "IL-1b": "IL1B", "Ki-67": "MKI67",
    "LAG3": "LAG3", "LAMP1": "LAMP1", "NF-kB p65": "RELA", "PD-1": "PDCD1", "PD-L1": "CD274", "PD-L2": "PDCD1LG2",
    "SMA": "ACTA2", "STING": "STING1", "TCF7": "TCF7", "Tim-3": "HAVCR2", "VISTA": "VSIR", "Vimentin": "VIM",
    "iNOS": "NOS2", "p53": "TP53", "pan-RAS": "KRAS", "CD45RA": "PTPRC_RA", "CD15": "FUT4", "IgD": "IGHD",
}
GENE2PROT = {v: k for k, v in PROT2GENE.items()}
# membrane-bound ligands (contact channels)
CONTACT_LIGANDS = {"CD274", "PDCD1LG2", "HLA-DRA", "HLA-DRB1", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
                   "ICAM1", "ICAM2", "ICAM3", "VCAM1", "CD276", "CD40LG", "CD70", "TNFSF9", "TNFSF18", "CD80", "CD86",
                   "PECAM1", "NCAM1", "VSIR", "CD58", "CD47", "CEACAM1", "JAG1", "JAG2", "DLL1", "DLL4", "EFNB1",
                   "EFNB2", "SEMA4D", "CD200", "NECTIN2", "PVR", "LGALS9", "HLA-A", "HLA-B", "HLA-C", "HLA-E", "CD48",
                   "ICOSLG", "TNFSF4", "TNFSF13B", "CDH1", "EPCAM", "SELE", "SELP", "ITGB2", "CD99", "CD6", "ALCAM"}
SECRETED_FAMILY = re.compile(r"^(CCL\d+|CXCL\d+|XCL\d|CX3CL1|IL\d+[A-Z]?|IFN[A-Z]\d*|TGFB\d|TNF|LTA|LTB|TNFSF\d+[A-Z]?|EGF|AREG|EREG|"
                             r"HBEGF|TGFA|BTC|EPGN|NRG\d|VEGF[A-D]|PGF|PDGF[A-D]|FGF\d+|HGF|IGF\d|WNT\d+[A-Z]?|BMP\d+|GDF\d+|"
                             r"INHB[A-E]|NODAL|CSF\d|LIF|OSM|CNTF|CLCF1|KITLG|SPP1|MIF|ANGPT\d|ANGPTL\d|SEMA3[A-G]|EDN\d|"
                             r"GAS6|PROS1|NAMPT|LGALS\d|CXCL\d+|C3|C5|SAA\d|APOE|PTN|MDK|NPPC|ADM|CALCA|GRN|THBS\d|POSTN|TIMP\d)$")
