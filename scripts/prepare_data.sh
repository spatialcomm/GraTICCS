#!/usr/bin/env bash
# Model inputs and planted-signal targets. See README for the files each step expects.
set -e
cd "$(dirname "$0")/.."
python -m prepare.breast_contact
python -m prepare.breast_secreted
GRATICCS_DATASET=${GRATICCS_XENIUM:-data/xenium_kidney} python -m prepare.xenium
python -m prepare.pathway_tf
python -m prepare.literature_match
KB=${GRATICCS_RAW:-raw}/literature_kb.json PRIOR_OUT=prior_programs python -m prepare.literature_programs
KB=${GRATICCS_RAW:-raw}/literature_kb_836.json PRIOR_OUT=prior_programs_836 python -m prepare.literature_programs

for m in contact contact_rna secreted lag misspecified pathway nichenet xenium; do python -m plant.$m; done
