#!/usr/bin/env bash
# All training runs of the paper. Each run writes results/<name>/fold{0..4}_mo.json (+ .pt, attention); finished runs are skipped.
set -e
cd "$(dirname "$0")/.."
B=${GRATICCS_BREAST:-data/breast}
done5() { [ "$(ls results/$1/fold*_mo.json 2>/dev/null | wc -l)" -ge 5 ]; }
contact() { local name=$1; shift; done5 "$name" || python train_contact.py --out "$name" "$@"; }
secreted() { local name=$1; shift; done5 "$name" || python train_secreted.py --out "$name" --accum 2 --batch-fovs 2 "$@"; }

# contact channels, breast: unspiked, planted strengths, niche covariate, pathway/TF readouts
for s in 0 1 2; do
  contact G_real_s$s --seed $s
  contact G_real_niche_s$s --seed $s --niche
  contact G_pw_contact_s$s --seed $s --pathway-readouts
  for b in 0.05 0.1 0.2 0.4; do contact G_spike_b${b}_s$s --seed $s --spike $b; done
done
contact G_pw_contact_spike_b0.2 --pathway-readouts --spike 0.2 --spike-prefix spike_pw_beta
for b in 0.1 0.4; do contact G_spikeRNA_b$b --spike $b --spike-prefix spike_rna_beta; done
for r in 0.3 0.5 0.7; do for b in 0.2 0.4; do contact G_lag_r${r}_b$b --spike r${r}_b$b --spike-prefix spike_lag_; done; done
for f in gate hill decay; do for b in 0.2 0.4; do for s in 0 1 2; do
  contact G_mis_${f}_b${b}_s$s --seed $s --spike $b --spike-prefix spike_mis_${f}_b
done; done; done

# graph transformer versus matched local model
for a in graticcs local; do n=arch_graticcs; [ $a = local ] && n=arch_graticcs_local
  for s in 0 1 2; do
    contact ${n}_real_s$s --arch $a --seed $s
    for b in 0.05 0.1; do contact ${n}_std_b${b}_s$s --arch $a --seed $s --spike $b; done
    contact ${n}_gate_b0.2_s$s --arch $a --seed $s --spike 0.2 --spike-prefix spike_mis_gate_b
  done
done

# Xenium renal cell carcinoma
for s in 0 1 2; do GRATICCS_DATASET=${GRATICCS_XENIUM:-data/xenium_kidney} contact G_xen_real_s$s --seed $s; done
for b in 0.1 0.4; do GRATICCS_DATASET=${GRATICCS_XENIUM:-data/xenium_kidney} contact G_xen_spike_b$b --spike $b; done

# secreted channels
for s in 0 1 2; do
  secreted G2_sec_real_s$s --seed $s
  secreted G2_pw_sec_s$s --seed $s --pathway-readouts
done
for b in 0.05 0.1 0.2 0.4; do secreted G2_sec_spike_b$b --targets $B/spike_sec_beta$b.npz; done
for b in 0.0 0.2 0.4; do
  secreted G2_nn_spike_b$b --targets $B/nichenet_spike/Y_beta$b.npz --readout-names $B/nichenet_spike/meta.json \
    --alt-rna-pc $B/nichenet_spike/alt_rna_pc.npy
done
for pp in prior_programs prior_programs_836; do for t in lit nn; do for s in 0 1; do
  secreted G2_${pp}_${t}_s$s --seed $s --targets $B/$pp/Y_$t.npz --readout-names $B/$pp/meta_$t.json --mask $B/$pp/mask_$t.npy
done; done; done
