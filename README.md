# GraTICCS

GraTICCS infers cell–cell communication from spatial RNA and protein measurements of the same cells. A graph transformer
predicts each receiver's measured response from its tissue environment (background branch) and adds one contribution per
ligand–receptor (LR) channel and sender (LR branch, 16 attention heads). Each channel is calibrated against decoy channels
that keep the receptor and replace the ligand, and recovery is tested with signals of known origin added to the measured
responses.

## Layout

```
graticcs/     model, data loading, paths, receptor and protein maps
prepare/      model inputs: channels and decoys, readouts, pathway/TF activities, literature matches
plant/        drawn planted pairs (plant/draws/) and the planted-signal targets
train_contact.py, train_secreted.py
calibrate.py  decoy calibration of one analysis
analysis/     recovery, cell-pair ranking, selective-sender tests, comparisons, biological assessment
figures/      Figures 1, 3, 4, the per-field comparison and the case figures
scripts/      prepare_data.sh, run_experiments.sh, run_analysis.sh
```

## Data

Paths are set with environment variables (defaults in brackets):

| variable | content |
|---|---|
| `GRATICCS_BREAST` [`data/breast`] | CosMx breast section: `cells.parquet`, `rna_counts.npz`, `genes.json`, `contact_edges.parquet`, NicheNet v2 tables as parquet |
| `GRATICCS_XENIUM` [`data/xenium_kidney`] | Xenium renal cell carcinoma section, same files, and `cell_feature_matrix.h5` |
| `GRATICCS_RAW` [`raw`] | cell-type labels (`cells_typed.parquet`), CosMx polygons and transcripts (`cosmx_breast/`), literature records (`literature_records/`) |
| `GRATICCS_RESULTS` [`results`] | training runs and tables |
| `GRATICCS_FIGURES` [`figures`] | figure output |

The CosMx human breast cancer RNA and protein data (Bruker Spatial Biology) and the Xenium renal cell carcinoma data
(10x Genomics) are public. The NicheNet v2 prior is Zenodo record 7074291.

## Running

```bash
pip install -r requirements.txt
bash scripts/prepare_data.sh
bash scripts/run_experiments.sh     # a 5-fold contact run takes 6-13 min on one laptop GPU
bash scripts/run_analysis.sh
```

One run, for example the contact model on the drawn planted signals at strength 0.1:

```bash
python train_contact.py --out G_crand_b0.1_s0 --seed 0 --spike 0.1 --spike-prefix spike_crand_beta
python -m analysis.recovery contact
```

`--arch local` trains the matched local model (local background, equal neighbour weights); `--arch local_type`,
`local_area` and `local_dist` replace the attention by fixed weighting rules. Seeds are fixed, but GPU kernels are not
deterministic, so retrained runs can differ slightly in the last digits.
