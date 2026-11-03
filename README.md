# GraTICCS

GraTICCS infers cell–cell communication (CCC) from same-cell spatial RNA and protein data. A graph transformer predicts each
receiver cell's measured response (functional proteins, gene programs, or pathway/TF activities) from a background branch
and one term per ligand–receptor (LR) channel. Every channel is tested against decoy channels that keep the receptor but
replace the ligand by an expression-matched gene, and sensitivity is measured by planting known LR effects into the data.

## Layout

```
graticcs/        model, data loading, paths, receptor-family rules
prepare/         model inputs: channels and decoys, readouts, pathway/TF activities, literature programs
plant/           planted-signal targets (standard, lag, misspecified forms, secreted, Xenium, NicheNet prior)
train_contact.py, train_secreted.py
calibrate.py     decoy calibration of one analysis (p, q per channel; planted recovery)
analysis/        summaries, pair-level AP, local-model comparison, literature test, NicheNet / HoloNet comparisons
figures/         paper figures and supplementary tables
scripts/         prepare_data.sh, run_experiments.sh, run_analysis.sh
```

## Data

Paths are set with environment variables (defaults in brackets):

| variable | content |
|---|---|
| `GRATICCS_BREAST` [`data/breast`] | CosMx breast section: `cells.parquet`, `rna_counts.npz`, `genes.json`, `contact_edges.parquet`, NicheNet v2 tables as parquet |
| `GRATICCS_XENIUM` [`data/xenium_kidney`] | Xenium renal cell carcinoma section, same files |
| `GRATICCS_RAW` [`raw`] | cell-type labels (`cells_typed.parquet`), CosMx polygons and transcripts (`cosmx_breast/`), literature records and knowledge base |
| `GRATICCS_RESULTS` [`results`] | training runs and tables |
| `GRATICCS_FIGURES` [`figures`] | figure output |

The CosMx human breast cancer RNA + protein data and the Xenium renal cell carcinoma data are public (NanoString / 10x Genomics).
The NicheNet v2 prior is from Zenodo record 7074291.

## Running

```bash
pip install -r requirements.txt
bash scripts/prepare_data.sh
bash scripts/run_experiments.sh     # about 6-13 min per 5-fold contact run on one laptop GPU
bash scripts/run_analysis.sh
```

A single run, for example the contact model on data with a planted signal of strength 0.1:

```bash
python train_contact.py --out G_spike_b0.1_s0 --seed 0 --spike 0.1
python calibrate.py data/breast/gt_channels.json - G_spike_b0.1_s0 data/breast/spike_meta.json
```

`--arch local` trains the matched local model (no graph layers in the background branch, uniform instead of learned LR attention).
Seeds are fixed, but GPU kernels are not deterministic, so reruns can differ slightly in the last digits.
