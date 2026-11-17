#!/usr/bin/env bash
# Tables and figures of the paper from the runs of run_experiments.sh.
set -e
cd "$(dirname "$0")/.."
B=${GRATICCS_BREAST:-data/breast}
X=${GRATICCS_XENIUM:-data/xenium_kidney}
O=results/calibration; mkdir -p $O
s3() { echo "$1_s0,$1_s1,$1_s2"; }

# original tissue: decoy calibration and prediction gains
python calibrate.py $B/gt_channels.json - $(s3 G_real) > $O/contact_real_3seed.txt
python calibrate.py $B/gt_channels.json - $(s3 G_real_niche) > $O/contact_real_niche_3seed.txt
python calibrate.py $B/gt_channels.json - $(s3 G_pw_contact) > $O/contact_pathway_real_3seed.txt
python calibrate.py $B/gt_sec_channels.json - $(s3 G2_sec_real) > $O/secreted_real_3seed.txt
python calibrate.py $B/gt_sec_channels.json - $(s3 G2_pw_sec) > $O/secreted_pathway_real_3seed.txt
python calibrate.py $X/gt_channels.json - $(s3 G_xen_real) > $O/xenium_real_3seed.txt
python -m analysis.summary

# planted signals: channel recovery
for a in contact secreted xenium xenium_batch2 forms secreted_delay; do python -m analysis.recovery $a; done
python -m analysis.delay
python -m analysis.naive_baseline
python -m analysis.per_field
python -m analysis.attention G_real G_crand_b0.2 G_ctype_b0.2 G_carea_b0.2

# cell pairs
python -m analysis.baselines
python -m analysis.pair_level contact
python -m analysis.pair_level local
python -m analysis.pair_level hybrid
python -m analysis.pair_level sender_type
python -m analysis.pair_level contact_extent
python -m analysis.secreted_pairs distance
python -m analysis.secreted_pairs sender_type
python -m analysis.overall_ap decompose
python -m analysis.overall_ap decompose --coexpression
python -m analysis.overall_ap crosschannel
python -m analysis.overall_ap crosschannel --coexpression
python -m analysis.per_cell 0.1

# other methods
python -m analysis.holonet                  # needs HoloNet 0.1.0 (separate environment)
python -m analysis.holonet_matched
python -m analysis.nichenet
python -m analysis.nichenet_graticcs

# biological assessment in the original tissue
python -m analysis.channel_rank contact $(s3 G_real) results/channel_best_rank_contact_G.csv
python -m analysis.channel_rank secreted $(s3 G2_sec_real) results/channel_best_rank_secreted_G2.csv
python -m analysis.dose_response secreted 0 20 dose_response_top20_v2.csv
python -m analysis.dose_response secreted 20 50 dose_response_sec_rank21_50_v2.csv
python -m analysis.dose_response contact 0 10 dose_response_contact_top10_v2.csv
python -m analysis.dose_response contact 10 50 dose_response_contact_rank11_50_v2.csv
python -m analysis.case_events contact $(s3 G_real) results/cases_contact_real.csv
python -m analysis.case_events secreted $(s3 G2_sec_real) results/cases_secreted_real_G2.csv
python -m analysis.literature_test results/cases_contact_real.csv contact results/lit_enrichment_contact_real.csv
python -m analysis.literature_test results/cases_secreted_real_G2.csv secreted results/lit_enrichment_secreted_real_G2.csv
python -m analysis.lit_decoy_control
python -m analysis.literature_screen
python -m analysis.top_events secreted $(s3 G2_sec_real) results/case_events_secreted.csv "IL1B->SIGIRR,CXCL9->CXCR3,TIMP3->CD44"
python -m analysis.top_events contact $(s3 G_real) results/case_events_contact.csv "HLA-DQA1->CD4"
python -m prepare.receptor_rules

# figures
python -m figures.figure1_data
python -m figures.figure1
python -m figures.figure3
python -m figures.figure4
python -m figures.field_comparison
python - <<'EOF'
import pandas as pd
for kind, parts in (("sec", ["dose_response_top20_v2.csv", "dose_response_sec_rank21_50_v2.csv"]),
                    ("contact", ["dose_response_contact_top10_v2.csv", "dose_response_contact_rank11_50_v2.csv"])):
    pd.concat([pd.read_csv(f"results/{p}") for p in parts]).to_csv(f"results/dose_response_{kind}_top50_v2.csv", index=False)
EOF
F=figures/cases; mkdir -p $F
for c in IL1B-SIGIRR CXCL9-CXCR3 TIMP3-CD44; do
  python -m figures.case_figure results/case_events_secreted.csv "${c/-/->}" $F/case_${c/-/_} secreted results/dose_response_sec_top50_v2.csv
done
python -m figures.case_figure results/case_events_contact.csv "HLA-DQA1->CD4" $F/case_HLA-DQA1_CD4 contact results/dose_response_contact_top50_v2.csv
