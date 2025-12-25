import biasondemand

# =============================================================================
# CRITICAL: DEFAULT VALUES FROM SOURCE CODE
# =============================================================================
# Default parameters values:
# l_y = 4      ← NOT 0! (historical bias on Y)
# l_h_r = 1.5  ← NOT 0! (historical bias on R)
# l_h_q = 1    ← NOT 0! (historical bias on Q)
# l_m = 1      ← NOT 0! (measurement bias on R)
# p_u = 1      ← NOT 0! (100% undersampling = removes ALL A=1!)
# l_q = 2      ← NOT 0! (importance of Q for Y)
# sy = 5       ← NOT 0! (label noise)
# l_m_y = 0    ✓ Zero
# l_r_q = 0    ✓ Zero
# l_y_b = 0    ✓ Zero
# l_r = False  ✓ False
# l_o = False  ✓ False
# thr_supp = 1 ✓ Correct
# l_m_y_non_linear=False

# =============================================================================
# BASELINE - Truly Unbiased Dataset
# =============================================================================
print("Generating BASELINE (truly unbiased)...")
biasondemand.generate_dataset(
    path="/baseline_",
    dim=10000,
    # EXPLICITLY SET EVERYTHING TO ZERO/DEFAULT
    l_y=0,          # No historical bias on Y
    l_m_y=0,        # No measurement bias on Y
    l_h_r=0,        # No historical bias on R
    l_h_q=0,        # No historical bias on Q
    l_m=0,          # No measurement bias on R
    p_u=0,          # No undersampling (keep all A=1 samples)
    l_r=False,      # No representation bias
    l_o=False,      # Don't omit variables
    l_y_b=0,        # No interaction bias
    l_q=0,          # Q doesn't influence Y structurally
    sy=0,           # No label noise
    l_r_q=0,        # R doesn't influence Q
    thr_supp=1,      # Don't suppress features
)

# =============================================================================
# 1. HISTORICAL BIAS ON Q (Feature Bias)
# =============================================================================
def generate_historical_bias_Q(bias_levels):
    """Historical bias on feature Q against A=1"""
    for l_h_q_val in bias_levels:
        print(f"Generating historical bias Q: {l_h_q_val}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_Q_lhq_{l_h_q_val:.2f}",
            dim=10000,
            l_y=0,          # No historical bias on Y
            l_m_y=0,        # No measurement bias on Y
            l_h_r=0,        # No historical bias on R
            l_h_q=l_h_q_val,        # Historical bias on Q
            l_m=0,          # No measurement bias on R
            p_u=0,          # No undersampling (keep all A=1 samples)
            l_r=False,      # No representation bias
            l_o=False,      # Don't omit variables
            l_y_b=0,        # No interaction bias
            l_q=0,          # Q doesn't influence Y structurally
            sy=0,           # No label noise
            l_r_q=0,        # R doesn't influence Q
            thr_supp=1,      # Don't suppress features
        )

# =============================================================================
# 2. HISTORICAL BIAS ON R (Feature Bias)
# =============================================================================
def generate_historical_bias_R(bias_levels):
    """Historical bias on feature R against A=1"""
    for l_h_r_val in bias_levels:
        print(f"Generating historical bias R: {l_h_r_val}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_R_lhr_{l_h_r_val:.2f}",
            dim=10000,
            l_h_r=l_h_r_val,  # VARY THIS
            # Set everything else to ZERO
            l_y=0,
            l_m_y=0,
            l_h_q=0,
            l_m=0,
            p_u=0,
            l_r=False,
            l_o=False,
            l_y_b=0,
            l_q=0,
            sy=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# 3. HISTORICAL BIAS ON Y (Direct Label Bias) - MOST SEVERE
# =============================================================================
def generate_historical_bias_Y(bias_levels):
    """Historical bias directly on target Y (most severe)"""
    for l_y_val in bias_levels:
        print(f"Generating historical bias Y: {l_y_val}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_Y_ly_{l_y_val:.2f}",
            dim=10000,
            l_y=l_y_val,  # VARY THIS
            # Set everything else to ZERO
            l_m_y=0,
            l_h_r=0,
            l_h_q=0,
            l_m=0,
            p_u=0,
            l_r=False,
            l_o=False,
            l_y_b=0,
            l_q=0,
            sy=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# 4. UNDERSAMPLING (Representation Bias)
# =============================================================================
def generate_undersampling(undersampling_levels):
    """Undersampling of A=1 group"""
    for p_u_val in undersampling_levels:
        print(f"Generating undersampling: {p_u_val}")
        biasondemand.generate_dataset(
            path=f"/undersample_pu_{p_u_val:.2f}",
            dim=15000,  # Larger to have samples left after undersampling
            p_u=p_u_val,  # VARY THIS (0.0=no bias, 0.9=severe)
            # Set everything else to ZERO
            l_y=0,
            l_m_y=0,
            l_h_r=0,
            l_h_q=0,
            l_m=0,
            l_r=False,
            l_o=False,
            l_y_b=0,
            l_q=0,
            sy=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# 5. LABEL NOISE (Not bias, but degrades performance)
# =============================================================================
def generate_label_noise(noise_levels):
    """Label noise (affects both groups)"""
    for sy_val in noise_levels:
        print(f"Generating label noise: {sy_val}")
        biasondemand.generate_dataset(
            path=f"/label_noise_sy_{sy_val:.2f}",
            dim=10000,
            sy=sy_val,  # VARY THIS
            # Set everything else to ZERO
            l_y=0,
            l_m_y=0,
            l_h_r=0,
            l_h_q=0,
            l_m=0,
            p_u=0,
            l_r=False,
            l_o=False,
            l_y_b=0,
            l_q=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# 6. MEASUREMENT BIAS ON R
# =============================================================================
def generate_measurement_bias_R(bias_levels):
    """Measurement bias on R (R replaced by proxy P)"""
    for l_m_val in bias_levels:
        print(f"Generating measurement bias R: {l_m_val}")
        biasondemand.generate_dataset(
            path=f"/meas_bias_R_lm_{l_m_val:.2f}",
            dim=10000,
            l_m=l_m_val,  # VARY THIS
            # Set everything else to ZERO
            l_y=0,
            l_m_y=0,
            l_h_r=0,
            l_h_q=0,
            p_u=0,
            l_r=False,
            l_o=False,
            l_y_b=0,
            l_q=0,
            sy=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# 7. INTERACTION PROXY BIAS
# =============================================================================
def generate_interaction_bias(bias_levels):
    """Interaction proxy bias (A=1 with high R get lower Y)"""
    for l_y_b_val in bias_levels:
        print(f"Generating interaction bias: {l_y_b_val}")
        biasondemand.generate_dataset(
            path=f"/interaction_bias_lyb_{l_y_b_val:.2f}",
            dim=10000,
            l_y_b=l_y_b_val,  # VARY THIS
            # Set everything else to ZERO
            l_y=0,
            l_m_y=0,
            l_h_r=0,
            l_h_q=0,
            l_m=0,
            p_u=0,
            l_r=False,
            l_o=False,
            l_q=0,
            sy=0,
            l_r_q=0,
            thr_supp=1
        )

# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("CORRECTED BIASONDEMAND DATASET GENERATION")
    print("="*80)
    print("\nIMPORTANT: All parameters explicitly set to zero except the one being varied!")
    print("This ensures we're testing ONE bias type at a time.\n")
    
    # Define levels
    detailed_levels = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    undersampling_levels = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]  # 0=fair, 0.9=severe
    noise_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    print("\n1. Generating Historical Bias on Q...")
    generate_historical_bias_Q(detailed_levels)
    
    print("\n2. Generating Historical Bias on R...")
    generate_historical_bias_R(detailed_levels)
    
    print("\n3. Generating Historical Bias on Y (Direct Label Bias)...")
    generate_historical_bias_Y(detailed_levels)
    
    print("\n4. Generating Undersampling...")
    generate_undersampling(undersampling_levels)
    
    print("\n5. Generating Label Noise...")
    generate_label_noise(detailed_levels)
    
    print("\n6. Generating Measurement Bias on R...")
    generate_measurement_bias_R(detailed_levels)
    
    print("\n7. Generating Interaction Bias...")
    generate_interaction_bias(detailed_levels)
    
    print("\n" + "="*80)
    print("✓ ALL DATASETS GENERATED WITH CORRECT PARAMETERS")
    print("="*80)
    print("\nDataset naming convention:")
    print("  • hist_bias_Q_lhq_*  - Historical bias on Q only")
    print("  • hist_bias_R_lhr_*  - Historical bias on R only")
    print("  • hist_bias_Y_ly_*   - Historical bias on Y only (most severe)")
    print("  • undersample_pu_*   - Undersampling only")
    print("  • label_noise_sy_*   - Label noise only")
    print("  • meas_bias_R_lm_*   - Measurement bias on R only")
    print("  • interaction_bias_* - Interaction proxy bias only")
    print("\nNow you should see CLEAR trends in fairness metrics!")