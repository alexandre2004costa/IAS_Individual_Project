import biasondemand

# The Unbiased, Clean Baseline Dataset (D_base)
biasondemand.generate_dataset(
    path="/baseline",
    dim=10000,  # ← FIXED: Same size as others
)

# Better range: 0.0 to 1.0 in reasonable steps
diffs = [0.00, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00]

def generate_bias_series(bias_levels):
    for l_q_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/bias_lq_{l_q_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=l_q_val,  # ✅ 0.0 to 1.0 range
            l_r_q=0.0,
            thr_supp=1.0
        )

generate_bias_series(diffs)

def generate_noise_series(noise_levels):
    for sy_val in noise_levels:
        biasondemand.generate_dataset(
            path=f"/noise_sy_{sy_val:.2f}",
            dim=10000,
            sy=sy_val,  # Label noise
            l_q=0.0,
            l_r_q=0.0,
            thr_supp=1.0
        )

generate_noise_series(diffs)

def generate_imbalance_series(undersampling_percentages):
    for p_u_val in undersampling_percentages:
        biasondemand.generate_dataset(
            path=f"/imbalance_pu_{p_u_val:.2f}",
            dim=15000,
            sy=0.0,
            l_q=0.0,
            p_u=p_u_val,  # ✅ 0.0 (no imbalance) to 1.0 (extreme)
            l_r_q=0.0,
            thr_supp=1.0
        )

# ✅ FIXED: Use normal order (0.0 → 1.0)
# p_u=0.0 means keep all minority samples (balanced)
# p_u=0.5 means remove 50% of minority samples
# p_u=0.9 means remove 90% of minority samples (severe imbalance)
generate_imbalance_series(diffs)