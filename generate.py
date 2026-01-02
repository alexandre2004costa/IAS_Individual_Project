import biasondemand

# =============================================================================
# CRITICAL: DEFAULT VALUES FROM SOURCE  
# =============================================================================
# Default parameters values:
# l_y = 4      ← NOT 0! (historical bias on Y)
# l_h_r = 1.5  ← NOT 0! (historical bias on R)
# l_h_q = 1    ← NOT 0! (historical bias on Q)
# l_m = 1      ← NOT 0! (measurement bias on R)
# p_u = 1      ← NOT 0! (100% undersampling)
# l_q = 2      ← NOT 0! (importance of Q for Y)
# sy = 5       ← NOT 0! (label noise)
# l_m_y = 0    ✓ Zero
# l_r_q = 0    ✓ Zero
# l_y_b = 0    ✓ Zero
# l_r = False  ✓ False
# l_o = False  ✓ False
# thr_supp = 1 ✓ Correct
# l_m_y_non_linear=False


biasondemand.generate_dataset(
    path="/baseline_",
    dim=10000,
    l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
    l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
    thr_supp=1, l_m_y_non_linear=False
)

# Historical Bias
def generate_historical_bias_Y(bias_levels):
    for l_y_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_Y_ly_{l_y_val:.2f}",
            dim=10000,
            l_y=l_y_val,  # ← VARYING THIS
            l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

def generate_historical_bias_Q(bias_levels):
    for l_h_q_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_Q_lhq_{l_h_q_val:.2f}",
            dim=10000,
            l_h_q=l_h_q_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

def generate_historical_bias_R(bias_levels):
    for l_h_r_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_R_lhr_{l_h_r_val:.2f}",
            dim=10000,
            l_h_r=l_h_r_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

def generate_interaction_proxy_bias(bias_levels):
    for l_y_b_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/interaction_bias_lyb_{l_y_b_val:.2f}",
            dim=10000,
            l_y_b=l_y_b_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

# Measurement Bias
def generate_measurement_bias_Y_linear(bias_levels):
    for l_m_y_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/meas_bias_Y_linear_lmy_{l_m_y_val:.2f}",
            dim=10000,
            l_m_y=l_m_y_val,  # ← VARYING THIS
            l_m_y_non_linear=False,  # Linear measurement bias
            l_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1
        )

def generate_measurement_bias_Y_nonlinear(bias_levels):
    for l_m_y_val in bias_levels:
        print(f"  Generating l_m_y={l_m_y_val:.2f} (non-linear)")
        biasondemand.generate_dataset(
            path=f"/meas_bias_Y_nonlinear_lmy_{l_m_y_val:.2f}",
            dim=10000,
            l_m_y=l_m_y_val,  # ← VARYING THIS
            l_m_y_non_linear=True,  # Non-linear measurement bias
            l_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1
        )

def generate_measurement_bias_R(bias_levels):
    for l_m_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/meas_bias_R_lm_{l_m_val:.2f}",
            dim=10000,
            l_m=l_m_val,  # ← VARYING THIS (R → P proxy)
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

# Representation Bias / Imbalance
def generate_undersampling(undersampling_levels):
    for p_u_val in undersampling_levels[1:]: # p_u = 0 actually makes balance again, so our range is from 0.1 to the default value 1
        print(f"  Generating p_u={p_u_val:.2f} (removes {p_u_val*100:.0f}% of A=1)")
        biasondemand.generate_dataset(
            path=f"/undersample_pu_{(1-p_u_val):.2f}", # We do this because p_u = 0.1 actually means 90% of imbalance, so we switch to show the differences growing in the graphs
            dim=15000,  # Larger to have samples left
            p_u=p_u_val,  # ← VARYING THIS (0=fair, 0.9=severe)
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

def generate_representation_bias(levels):
    for l in levels[1:]: # p_u = 0 actually makes balance again, so our range is from 0.1 to the default value 1
        biasondemand.generate_dataset(
            path=f"/representation_bias_lr_true{(1-l):.2f}",
            dim=15000,
            l_r=True,  # ← ENABLING THIS
            p_u=l,  # Need some undersampling for l_r to matter
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,
            l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

# Label Noise
def generate_label_noise(noise_levels):
    for sy_val in noise_levels:
        print(f"  Generating sy={sy_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/label_noise_sy_{sy_val:.2f}",
            dim=10000,
            sy=sy_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    standard_levels = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    
    generate_historical_bias_Y(standard_levels)     
    generate_historical_bias_Q(standard_levels)      
    generate_historical_bias_R(standard_levels)      
    generate_interaction_proxy_bias(standard_levels)
    generate_measurement_bias_Y_linear(standard_levels)
    generate_measurement_bias_Y_nonlinear(standard_levels)
    generate_measurement_bias_R(standard_levels)
    generate_undersampling(standard_levels)
    generate_representation_bias(standard_levels)
    generate_label_noise(standard_levels)
    
    print("\n" + "="*80)
    print("✓ ALL DATASETS GENERATED SUCCESSFULLY")
    print("="*80)
    
    print("\n📁 DATASET NAMING CONVENTION:")
    print("  Historical Bias:")
    print("    • hist_bias_Y_ly_*           - Direct label bias (MOST SEVERE)")
    print("    • hist_bias_Q_lhq_*          - Feature Q bias")
    print("    • hist_bias_R_lhr_*          - Feature R bias")
    print("    • interaction_bias_lyb_*     - Interaction bias")
    print("\n  Measurement Bias:")
    print("    • meas_bias_Y_linear_lmy_*   - Y measurement (linear)")
    print("    • meas_bias_Y_nonlinear_lmy_*- Y measurement (non-linear)")
    print("    • meas_bias_R_lm_*           - R measurement (proxy)")
    print("\n  Representation Bias:")
    print("    • undersample_pu_*           - Simple undersampling")
    print("    • representation_bias_lr_*   - Conditional undersampling")
    print("\n  Other:")
    print("    • label_noise_sy_*           - Random label noise")
    print("    • baseline_                   - No bias (fair)")
