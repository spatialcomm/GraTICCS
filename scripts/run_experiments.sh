#!/usr/bin/env bash
# All training runs of the paper, three seeds each. Each run writes results/<name>/fold{0..4}_mo.json, .pt and attention;
# finished runs are skipped.
set -e
cd "$(dirname "$0")/.."
B=${GRATICCS_BREAST:-data/breast}
X=${GRATICCS_XENIUM:-data/xenium_kidney}
done5() { [ "$(ls results/$1/fold*_mo.json 2>/dev/null | wc -l)" -ge 5 ]; }
contact() { local name=$1; shift; done5 "$name" || python train_contact.py --out "$name" "$@"; }
secreted() { local name=$1; shift; done5 "$name" || python train_secreted.py --out "$name" --accum 2 --batch-fovs 2 "$@"; }

for s in 0 1 2; do
  # original tissue
  contact G_real_s$s --seed $s
  contact L_real_s$s --seed $s --arch local
  contact G_real_niche_s$s --seed $s --niche
  contact G_pw_contact_s$s --seed $s --pathway-readouts
  GRATICCS_DATASET=$X contact G_xen_real_s$s --seed $s
  secreted G2_sec_real_s$s --seed $s
  secreted G2_pw_sec_s$s --seed $s --pathway-readouts

  # planted contact signals: four strengths, GraTICCS and the matched local model
  for b in 0.05 0.1 0.2 0.4; do
    contact G_crand_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand_beta
    contact L_crand_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand_beta --arch local
  done
  for b in 0.1 0.4; do
    contact G_crand_rna_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand_rna_beta
    GRATICCS_DATASET=$X contact G_xen_crand_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand_beta
    GRATICCS_DATASET=$X contact G_xen_crand2_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand2_beta
  done
  contact G_crand_pw_b0.2_s$s --seed $s --spike 0.2 --spike-prefix spike_crand_pw_beta --pathway-readouts
  for f in hill gate decay; do for b in 0.2 0.4; do
    contact G_crand_mis_${f}_b${b}_s$s --seed $s --spike $b --spike-prefix spike_crand_mis_${f}_b
  done; done
  contact L_crand_mis_gate_b0.2_s$s --seed $s --spike 0.2 --spike-prefix spike_crand_mis_gate_b --arch local
  for r in 0.7 0.5 0.3; do for b in 0.2 0.4; do
    contact G_crand_lag_r${r}_b${b}_s$s --seed $s --spike r${r}_b$b --spike-prefix spike_crand_lag_
  done; done

  # contact tests with selective senders: GraTICCS, local, local + sender-type rule, local + contact-extent rule
  for e in ctype carea; do
    P="--seed $s --spike 0.2 --spike-prefix spike_${e}_beta"
    contact G_${e}_b0.2_s$s $P
    contact L_${e}_b0.2_s$s $P --arch local
    contact T_${e}_b0.2_s$s $P --arch local_type
    contact W_${e}_b0.2_s$s $P --arch local_area
  done

  # planted secreted signals
  for b in 0.05 0.1 0.2 0.4; do
    secreted G2_sec_rand_b${b}_s$s --seed $s --targets $B/spike_sec_rand_beta$b.npz
    secreted L_sec_rand_b${b}_s$s --seed $s --targets $B/spike_sec_rand_beta$b.npz --arch local
  done
  for r in 0.7 0.5 0.3; do for b in 0.2 0.4; do
    secreted G2_sec_lag_r${r}_b${b}_s$s --seed $s --targets $B/spike_sec_lag_r${r}_b$b.npz
  done; done
  CT=$B/spike_sec_ctype_beta0.2.npz; RD=$B/spike_sec_rand_beta0.2.npz
  secreted G2_sec_ctype_b0.2_s$s --seed $s --targets $CT
  secreted L_sec_ctype_b0.2_s$s --seed $s --targets $CT --arch local
  secreted T_sec_ctype_b0.2_s$s --seed $s --targets $CT --arch local_type
  secreted W_sec_ctype_b0.2_s$s --seed $s --targets $CT --arch local_dist
  secreted T_sec_rand_b0.2_s$s --seed $s --targets $RD --arch local_type
  secreted W_sec_rand_b0.2_s$s --seed $s --targets $RD --arch local_dist

  # NicheNet-consistent planted programs
  ND=$B/nichenet_spike_crand
  for b in 0.0 0.2 0.4; do
    secreted G2_nn_crand_b${b}_s$s --seed $s --targets $ND/Y_beta$b.npz --readout-names $ND/meta.json --alt-rna-pc $ND/alt_rna_pc.npy
  done
done
