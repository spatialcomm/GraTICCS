#!/usr/bin/env bash
# Calibration, tables and figures from the finished runs.
set -e
cd "$(dirname "$0")/.."
B=${GRATICCS_BREAST:-data/breast}; X=${GRATICCS_XENIUM:-data/xenium_kidney}
C=$B/gt_channels.json; S=$B/gt_sec_channels.json; XC=$X/gt_channels.json
O=results/calibration; mkdir -p $O
cal() { local out=$1; shift; python calibrate.py "$@" > $O/$out.txt; }
s3() { echo "$1_s0,$1_s1,$1_s2"; }

for b in 0.05 0.1 0.2 0.4; do cal contact_spike_b${b}_3seed $C - $(s3 G_spike_b$b) $B/spike_meta.json; done
cal contact_real_3seed $C - $(s3 G_real)
cal contact_real_niche_3seed $C - $(s3 G_real_niche)
cal contact_pathway_real_3seed $C - $(s3 G_pw_contact)
cal contact_pathway_spike_b0.2 $C - G_pw_contact_spike_b0.2 $B/spike_pw_meta.json
for f in gate hill decay; do for b in 0.2 0.4; do
  r=$(s3 G_mis_${f}_b$b); [ $f = gate ] && [ $b = 0.2 ] && r=$(s3 arch_graticcs_gate_b0.2)
  cal contact_misspec_${f}_b$b $C - $r $B/spike_meta.json
done; done
for r in 0.3 0.5 0.7; do for b in 0.2 0.4; do cal contact_lag_r${r}_b$b $C - G_lag_r${r}_b$b $B/spike_meta.json; done; done
for b in 0.1 0.4; do cal contact_spikeRNA_b$b $C - G_spikeRNA_b$b $B/spike_rna_meta.json; done
for n in arch_graticcs arch_graticcs_local; do
  for b in 0.05 0.1; do cal ${n}_std_b${b}_3seed $C - $(s3 ${n}_std_b$b) $B/spike_meta.json; done
  cal ${n}_gate_b0.2_3seed $C - $(s3 ${n}_gate_b0.2) $B/spike_meta.json
  cal ${n}_real_3seed $C - $(s3 ${n}_real)
done
cal xenium_real_3seed $XC - $(s3 G_xen_real)
for b in 0.1 0.4; do cal xenium_spike_b$b $XC - G_xen_spike_b$b $X/spike_meta.json; done
for b in 0.05 0.1 0.2 0.4; do cal secreted_spike_b$b $S - G2_sec_spike_b$b $B/spike_sec_meta.json; done
cal secreted_real_3seed $S - $(s3 G2_sec_real)
cal secreted_pathway_real_3seed $S - $(s3 G2_pw_sec)
for pp in prior_programs prior_programs_836; do for t in lit nn; do
  cal secreted_${pp}_${t}_2seed $S $B/$pp/meta_$t.json G2_${pp}_${t}_s0,G2_${pp}_${t}_s1
  python -m analysis.prior_rate $S $B/$pp/mask_$t.npy G2_${pp}_${t}_s0,G2_${pp}_${t}_s1 >> $O/prior_discovery_rates.txt
done; done

python -m analysis.summary
python -m analysis.local_vs_graph
python -m analysis.baselines
python -m analysis.pair_level
python -m analysis.nichenet
python -m analysis.nichenet_graticcs
python -m analysis.holonet_matched        # analysis.holonet needs a separate environment with HoloNet 0.1.0
python -m analysis.case_events contact $(s3 G_real) results/cases_contact_real.csv
python -m analysis.case_events secreted $(s3 G2_sec_real) results/cases_secreted_real_G2.csv
python -m analysis.literature_test results/cases_contact_real.csv contact results/lit_enrichment_contact_real.csv
python -m analysis.literature_test results/cases_secreted_real_G2.csv secreted results/lit_enrichment_secreted_real_G2.csv
python -m prepare.receptor_rules

python -m figures.figure1_data
python -m figures.figure1
python -m figures.main_figures
mkdir -p figures/lit_cases
python -m figures.case_figure results/cases_secreted_real_G2.csv "FN1->ITGA3_ITGB1" figures/lit_cases/Fig_main_FN1_ITGA3_ITGB1.png 0 secreted
python -m figures.case_figure results/cases_secreted_real_G2.csv "SPP1->ITGAV_ITGB1" figures/lit_cases/Fig_S_SPP1_ITGAV_ITGB1.png 0 secreted
python -m figures.case_figure results/cases_contact_real.csv "TNFSF9->TNFRSF9" figures/lit_cases/Fig_S_TNFSF9_TNFRSF9.png 0 contact
python -m figures.case_figure results/cases_contact_real.csv "TNFSF18->TNFRSF18" figures/lit_cases/Fig_S_TNFSF18_TNFRSF18.png 0 contact
python -m figures.supplementary_tables
