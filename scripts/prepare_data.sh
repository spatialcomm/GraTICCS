#!/usr/bin/env bash
# Model inputs and planted-signal targets. See README for the files each step expects.
set -e
cd "$(dirname "$0")/.."
X=${GRATICCS_XENIUM:-data/xenium_kidney}

python -m prepare.breast_contact
python -m prepare.breast_secreted
GRATICCS_DATASET=$X python -m prepare.xenium
python -m prepare.pathway_tf
python -m prepare.literature_match
python -m prepare.transcripts_by_fov          # only needed for the case figures

# planted signals for the drawn pairs in plant/draws/ (python -m plant.draw <kind> pool|draw repeats the draw)
python -m plant.contact
python -m plant.contact --rna
python -m plant.contact --xenium
python -m plant.contact --xenium --draw planted_random_xenium_batch2.csv --prefix spike_crand2_beta
python -m plant.secreted
python -m plant.delay contact
python -m plant.delay secreted
python -m plant.forms
python -m plant.route_refs
python -m plant.pathway
python -m plant.selective sender_type
python -m plant.selective contact_extent
python -m plant.selective secreted_sender_type
python -m plant.nichenet
