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


biasondemand.generate_dataset(
    path="/baseline_",
    dim=10000,
    l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
    l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
    thr_supp=1, l_m_y_non_linear=False
)
print("✓ Baseline generated\n")

# =============================================================================
# CATEGORY 1: HISTORICAL BIAS
# =============================================================================
# Most common and impactful in real-world scenarios

def generate_historical_bias_Y(bias_levels):
    """
    1A. Historical Bias on Target Y (MOST SEVERE)
    
    Direct discrimination in labels.
    Example: Loan officers historically approved fewer loans for minorities.
    
    Effect: Y is directly reduced for A=1, regardless of features.
    """
    print("\n" + "="*80)
    print("1A. HISTORICAL BIAS ON Y (Direct Label Bias) - MOST SEVERE")
    print("="*80)
    
    for l_y_val in bias_levels:
        print(f"  Generating l_y={l_y_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_Y_ly_{l_y_val:.2f}",
            dim=10000,
            l_y=l_y_val,  # ← VARYING THIS
            l_m_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Historical bias Y complete\n")


def generate_historical_bias_Q(bias_levels):
    """
    1B. Historical Bias on Feature Q
    
    Q (e.g., credit score) systematically lower for A=1.
    Example: Credit scores biased against minorities due to historical factors.
    
    Effect: Q values reduced for A=1, model learns to penalize A=1.
    """
    print("\n" + "="*80)
    print("1B. HISTORICAL BIAS ON Q (Feature Bias)")
    print("="*80)
    
    for l_h_q_val in bias_levels:
        print(f"  Generating l_h_q={l_h_q_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_Q_lhq_{l_h_q_val:.2f}",
            dim=10000,
            l_h_q=l_h_q_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Historical bias Q complete\n")


def generate_historical_bias_R(bias_levels):
    """
    1C. Historical Bias on Feature R
    
    R (e.g., income) systematically lower for A=1.
    Example: Gender pay gap, income disparities.
    
    Effect: R values reduced for A=1, legitimate feature now correlates with A.
    """
    print("\n" + "="*80)
    print("1C. HISTORICAL BIAS ON R (Feature Bias)")
    print("="*80)
    
    for l_h_r_val in bias_levels:
        print(f"  Generating l_h_r={l_h_r_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/hist_bias_R_lhr_{l_h_r_val:.2f}",
            dim=10000,
            l_h_r=l_h_r_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Historical bias R complete\n")


def generate_interaction_proxy_bias(bias_levels):
    """
    1D. Interaction Proxy Bias (Complex Historical Bias)
    
    A=1 with high R values get lower Y (non-linear discrimination).
    Example: High-earning minorities face discrimination.
    
    Effect: Bias depends on interaction between A and R.
    """
    print("\n" + "="*80)
    print("1D. INTERACTION PROXY BIAS (Complex Historical)")
    print("="*80)
    
    for l_y_b_val in bias_levels:
        print(f"  Generating l_y_b={l_y_b_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/interaction_bias_lyb_{l_y_b_val:.2f}",
            dim=10000,
            l_y_b=l_y_b_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Interaction proxy bias complete\n")


# =============================================================================
# CATEGORY 2: MEASUREMENT BIAS
# =============================================================================
# Variables measured incorrectly or with error

def generate_measurement_bias_Y_linear(bias_levels):
    """
    2A. Measurement Bias on Y (Linear)
    
    Labels Y measured with systematic error.
    Example: Recidivism measured by arrests, not actual crimes.
    
    Effect: Y labels have systematic errors differing by group.
    """
    print("\n" + "="*80)
    print("2A. MEASUREMENT BIAS ON Y (Linear)")
    print("="*80)
    
    for l_m_y_val in bias_levels:
        print(f"  Generating l_m_y={l_m_y_val:.2f} (linear)")
        biasondemand.generate_dataset(
            path=f"/meas_bias_Y_linear_lmy_{l_m_y_val:.2f}",
            dim=10000,
            l_m_y=l_m_y_val,  # ← VARYING THIS
            l_m_y_non_linear=False,  # Linear measurement bias
            l_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1
        )
    print("✓ Measurement bias Y (linear) complete\n")


def generate_measurement_bias_Y_nonlinear(bias_levels):
    """
    2B. Measurement Bias on Y (Non-Linear)
    
    Labels Y measured with non-linear systematic error.
    Example: Measurement error depends on R values.
    
    Effect: Complex measurement errors conditional on features.
    """
    print("\n" + "="*80)
    print("2B. MEASUREMENT BIAS ON Y (Non-Linear)")
    print("="*80)
    
    for l_m_y_val in bias_levels:
        print(f"  Generating l_m_y={l_m_y_val:.2f} (non-linear)")
        biasondemand.generate_dataset(
            path=f"/meas_bias_Y_nonlinear_lmy_{l_m_y_val:.2f}",
            dim=10000,
            l_m_y=l_m_y_val,  # ← VARYING THIS
            l_m_y_non_linear=True,  # Non-linear measurement bias
            l_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1
        )
    print("✓ Measurement bias Y (non-linear) complete\n")


def generate_measurement_bias_R(bias_levels):
    """
    2C. Measurement Bias on Feature R
    
    R observed through noisy proxy P instead of true R.
    Example: Self-reported income (P) vs actual income (R).
    
    Effect: R column replaced by P with measurement error.
    """
    print("\n" + "="*80)
    print("2C. MEASUREMENT BIAS ON R (Proxy)")
    print("="*80)
    
    for l_m_val in bias_levels:
        print(f"  Generating l_m={l_m_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/meas_bias_R_lm_{l_m_val:.2f}",
            dim=10000,
            l_m=l_m_val,  # ← VARYING THIS (R → P proxy)
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Measurement bias R complete\n")


# =============================================================================
# CATEGORY 3: REPRESENTATION BIAS (Sampling Issues)
# =============================================================================
# Some groups under/over-represented in data

def generate_undersampling(undersampling_levels):
    """
    3A. Simple Undersampling (p_u)
    
    Removes percentage of A=1 samples.
    Example: Only 10% of loan applications from minorities in dataset.
    
    Effect: Model has less data for A=1, worse predictions.
    
    NOTE: Higher p_u = MORE bias (removes more samples)
    """
    print("\n" + "="*80)
    print("3A. UNDERSAMPLING (Simple)")
    print("="*80)
    
    for p_u_val in undersampling_levels:
        print(f"  Generating p_u={p_u_val:.2f} (removes {p_u_val*100:.0f}% of A=1)")
        biasondemand.generate_dataset(
            path=f"/undersample_pu_{p_u_val:.2f}",
            dim=15000,  # Larger to have samples left
            p_u=p_u_val,  # ← VARYING THIS (0=fair, 0.9=severe)
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Undersampling complete\n")


def generate_representation_bias(levels):
    """
    3B. Representation Bias (Conditional Undersampling)
    
    Undersamples A=1 conditional on R values.
    Example: High-income minorities are rare in dataset.
    
    Effect: Model never sees certain (A, R) combinations.
    """
    print("\n" + "="*80)
    print("3B. REPRESENTATION BIAS (Conditional on R)")
    print("="*80)
    
    print("  Generating l_r=True")
    for l in levels:
        biasondemand.generate_dataset(
            path=f"/representation_bias_lr_true{l:.2f}",
            dim=15000,
            l_r=True,  # ← ENABLING THIS
            p_u=l,  # Need some undersampling for l_r to matter
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0,
            l_o=False, l_y_b=0, l_q=0, sy=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Representation bias complete\n")


# =============================================================================
# CATEGORY 5: LABEL NOISE (Not Bias - Performance Degradation)
# =============================================================================
# Random errors in labels (affects both groups)

def generate_label_noise(noise_levels):
    """
    5. Label Noise (sy)
    
    Random noise added to Y labels.
    Example: Random labeling errors in dataset.
    
    Effect: Overall accuracy drops, both groups affected equally.
    
    NOTE: This is NOT bias, but degrades performance!
    """
    print("\n" + "="*80)
    print("5. LABEL NOISE (Not Bias - Degrades Performance)")
    print("="*80)
    
    for sy_val in noise_levels:
        print(f"  Generating sy={sy_val:.2f}")
        biasondemand.generate_dataset(
            path=f"/label_noise_sy_{sy_val:.2f}",
            dim=10000,
            sy=sy_val,  # ← VARYING THIS
            l_y=0, l_m_y=0, l_h_r=0, l_h_q=0, l_m=0, p_u=0,
            l_r=False, l_o=False, l_y_b=0, l_q=0, l_r_q=0,
            thr_supp=1, l_m_y_non_linear=False
        )
    print("✓ Label noise complete\n")


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE BIASONDEMAND DATASET GENERATION")
    print("="*80)
    print("\nGenerating all bias types in logical order:")
    print("  1. Historical Bias (Y, Q, R, Interaction)")
    print("  2. Measurement Bias (Y linear, Y non-linear, R)")
    print("  3. Representation Bias (Undersampling, Conditional)")
    print("  4. Omitted Variable Bias")
    print("  5. Label Noise")
    print("\n" + "="*80 + "\n")
    
    # Define levels for continuous parameters
    standard_levels = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 
                       0.60, 0.70, 0.80, 0.90, 1.00]
    
    undersampling_levels = [0.00, 0.10, 0.30, 0.50, 0.70, 0.90]
    
    noise_levels = [0.00, 0.50, 1.00, 2.00, 3.00, 5.00]
    
    # CATEGORY 1: HISTORICAL BIAS (Most Important)
    generate_historical_bias_Y(standard_levels)      # Most severe
    generate_historical_bias_Q(standard_levels)      # Feature bias
    generate_historical_bias_R(standard_levels)      # Feature bias
    generate_interaction_proxy_bias(standard_levels) # Complex bias
    
    # CATEGORY 2: MEASUREMENT BIAS
    generate_measurement_bias_Y_linear(standard_levels)
    generate_measurement_bias_Y_nonlinear(standard_levels)
    generate_measurement_bias_R(standard_levels)
    
    # CATEGORY 3: REPRESENTATION BIAS
    generate_undersampling(standard_levels)
    generate_representation_bias(standard_levels)
    
    # CATEGORY 5: LABEL NOISE
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
    print("    • omitted_var_bias_lo_*      - Omitted variable R")
    print("    • label_noise_sy_*           - Random label noise")
    print("    • baseline                   - No bias (fair)")
    
    print("\n" + "="*80)
    print("🎯 READY FOR FAIRNESS EVALUATION!")
    print("="*80)
    print("\nExpected fairness degradation:")
    print("  Historical Y > Historical Q/R > Undersampling > Measurement > Noise")
    print("\n" + "="*80 + "\n")