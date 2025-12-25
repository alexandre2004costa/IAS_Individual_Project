import biasondemand

# =============================================================================
# BASELINE - Clean, Unbiased Dataset
# =============================================================================
biasondemand.generate_dataset(
    path="/baseline",
    dim=10000,
    sy=0.0,
    l_q=0.0,
    l_r_q=0.0,
    thr_supp=1.0
)

# =============================================================================
# 1. HISTORICAL BIAS SERIES
# =============================================================================

# 1a. Historical Bias on Q (feature bias)
def generate_historical_bias_Q(bias_levels):
    """l_q: Historical bias where Q is biased against A=1"""
    for l_q_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_Q_lq_{l_q_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=l_q_val,      # Historical bias on Q
            l_r_q=0.0,
            l_h_q=0.0,
            thr_supp=1.0
        )

# 1b. Historical Bias on R (another feature)
def generate_historical_bias_R(bias_levels):
    """l_h_r: Historical bias on feature R against A=1"""
    for l_h_r_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_R_lhr_{l_h_r_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            l_h_r=l_h_r_val,  # Historical bias on R
            thr_supp=1.0
        )

# 1c. Historical Bias on Target Y (label bias)
def generate_historical_bias_Y(bias_levels):
    """l_y: Historical bias directly on target Y against A=1"""
    for l_y_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/hist_bias_Y_ly_{l_y_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            l_y=l_y_val,      # Historical bias on Y
            thr_supp=1.0
        )

# 1d. Interaction Proxy Bias (complex historical bias)
def generate_interaction_bias(bias_levels):
    """l_y_b: Historical bias on Y for A=1 with high R values"""
    for l_y_b_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/interaction_bias_lyb_{l_y_b_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            l_y_b=l_y_b_val,  # Interaction proxy bias
            thr_supp=1.0
        )

# =============================================================================
# 2. MEASUREMENT BIAS SERIES
# =============================================================================

# 2a. Measurement Bias on Feature R
def generate_measurement_bias_R(bias_levels):
    """l_m: Measurement bias on R (R is replaced by proxy P)"""
    for l_m_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/meas_bias_R_lm_{l_m_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            l_m=l_m_val,      # Measurement bias on R
            thr_supp=1.0
        )

# 2b. Measurement Bias on Target Y
def generate_measurement_bias_Y(bias_levels):
    """l_m_y: Measurement bias on target Y"""
    for l_m_y_val in bias_levels:
        biasondemand.generate_dataset(
            path=f"/meas_bias_Y_lmy_{l_m_y_val:.2f}",
            dim=10000,
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            l_m_y=l_m_y_val,  # Measurement bias on Y
            thr_supp=1.0
        )

# =============================================================================
# 3. REPRESENTATION BIAS SERIES (Sampling/Selection Bias)
# =============================================================================

# 3a. Simple Undersampling (removes proportion of A=1 group)
def generate_undersampling(undersampling_levels):
    """p_u: Percentage of A=1 instances to REMOVE (higher = more bias)"""
    for p_u_val in undersampling_levels:
        biasondemand.generate_dataset(
            path=f"/undersample_pu_{p_u_val:.2f}",
            dim=15000,  # Larger to have enough samples after undersampling
            sy=0.0,
            l_q=0.0,
            l_r_q=0.0,
            p_u=p_u_val,      # Undersampling percentage
            thr_supp=1.0
        )

# 3b. Representation Bias (conditional undersampling on R)
def generate_representation_bias():
    """l_r: Conditional undersampling based on feature R"""
    biasondemand.generate_dataset(
        path="/representation_bias_lr_true",
        dim=15000,
        sy=0.0,
        l_q=0.0,
        l_r_q=0.0,
        l_r=True,         # Enable representation bias
        thr_supp=1.0
    )

# =============================================================================
# 4. OMITTED VARIABLE BIAS
# =============================================================================

def generate_omitted_variable_bias():
    """l_o: Excludes important variable R (omitted variable bias)"""
    biasondemand.generate_dataset(
        path="/omitted_var_bias_lo_true",
        dim=10000,
        sy=0.0,
        l_q=0.0,
        l_r_q=0.0,
        l_o=True,         # Enable omitted variable bias
        thr_supp=1.0
    )

# =============================================================================
# 5. LABEL NOISE SERIES
# =============================================================================

def generate_label_noise(noise_levels):
    """sy: Standard deviation of noise in Y labels"""
    for sy_val in noise_levels:
        biasondemand.generate_dataset(
            path=f"/label_noise_sy_{sy_val:.2f}",
            dim=10000,
            sy=sy_val,        # Label noise
            l_q=0.0,
            l_r_q=0.0,
            thr_supp=1.0
        )

# =============================================================================
# EXECUTION - Generate All Datasets
# =============================================================================

# Define levels for continuous parameters
light_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
detailed_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
undersampling_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]  # 0.0=no bias, 0.9=remove 90% of A=1

print("Generating datasets...")
print("\n1. Historical Bias Series:")
generate_historical_bias_Q(detailed_levels)
generate_historical_bias_R(detailed_levels)
generate_historical_bias_Y(detailed_levels)
generate_interaction_bias(detailed_levels)

print("\n2. Measurement Bias Series:")
generate_measurement_bias_R(detailed_levels)
generate_measurement_bias_Y(detailed_levels)

print("\n3. Representation Bias Series:")
generate_undersampling(detailed_levels)
#generate_representation_bias()

print("\n4. Omitted Variable Bias:")
#generate_omitted_variable_bias()

print("\n5. Label Noise Series:")
generate_label_noise(detailed_levels)

print("\n✓ All datasets generated!")
print("\nDataset Categories:")
print("  • hist_bias_Q_lq_*    - Historical bias on feature Q")
print("  • hist_bias_R_lhr_*   - Historical bias on feature R")
print("  • hist_bias_Y_ly_*    - Historical bias on target Y")
print("  • interaction_bias_*   - Interaction proxy bias")
print("  • meas_bias_R_lm_*    - Measurement bias on R")
print("  • meas_bias_Y_lmy_*   - Measurement bias on Y")
print("  • undersample_pu_*    - Simple undersampling of A=1")
print("  • representation_bias - Conditional undersampling")
print("  • omitted_var_bias    - Omitted variable R")
print("  • label_noise_sy_*    - Label noise")