# BiasOnDemand Dataset Generation: Complete Guide
## Understanding Synthetic Data Generation for Bias Research

---

## 📊 Overview

BiasOnDemand is a Python library that generates synthetic datasets with **controllable bias parameters**. It creates data based on a **causal model** that simulates real-world scenarios where bias can enter at different stages of the data generation process.

**Key Features:**
- Generate datasets with precise control over bias types and magnitudes
- Simulate various real-world bias scenarios (hiring, lending, healthcare, etc.)
- Isolate individual bias types for systematic study
- Create baseline (unbiased) datasets for comparison

---

## 🏗️ Core Data Structure

BiasOnDemand generates datasets based on a **structural causal model** with four core variables:

```
Causal Structure:
    A → Q → Y
      ↘ R ↗
```

### **The 4 Main Variables:**

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| **A** | Binary (0, 1) | **Protected/Sensitive Attribute**<br>• A=0: Reference group (unprivileged)<br>• A=1: Protected group (where bias affects) | Gender (0=Female, 1=Male)<br>Race (0=Majority, 1=Minority) |
| **R** | Continuous | **Feature Variable**<br>• Causally influenced by A<br>• Important predictor of Y | Income level<br>Years of experience |
| **Q** | Continuous | **Feature Variable**<br>• Causally influenced by both A and R<br>• Important predictor of Y | Credit score<br>Performance rating |
| **Y** | Binary (0, 1) | **Target/Outcome Variable**<br>• The prediction target<br>• Depends on Q, R, and optionally A | Loan approval<br>Hired/Not hired |

### **Additional Generated Variables:**
- **P**: Proxy for R (only present when measurement bias is applied, `l_m > 0`)
- **Additional features**: Random noise features with varying correlation to A

---

## 🔧 Generation Function Signature

```python
biasondemand.generate_dataset(
    path='my_dataset',           # Output directory
    dim=15000,                    # Number of samples
    l_y=4,                        # Historical bias on Y
    l_m_y=0,                      # Measurement bias on Y (linear)
    thr_supp=1,                   # Feature suppression threshold
    l_h_r=1.5,                    # Historical bias on R
    l_h_q=1,                      # Historical bias on Q
    l_m=1,                        # Measurement bias on R
    p_u=1,                        # Undersampling proportion
    l_r=False,                    # Conditional representation bias
    l_o=False,                    # Omitted variable
    l_y_b=0,                      # Interaction proxy bias
    l_q=2,                        # Importance of Q for Y
    sy=5,                         # Label noise std deviation
    l_r_q=0,                      # R to Q influence
    l_m_y_non_linear=False        # Non-linear measurement bias flag
)
```

---

## 📖 Parameter Definitions

### **Basic Parameters**

#### **`path`** (string)
- **Default**: `'my_new_dataset'`
- **Description**: Directory path where the generated dataset will be saved
- **Creates**: Four CSV files: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

#### **`dim`** (int)
- **Default**: `15000`
- **Description**: Total number of samples to generate (before any undersampling)
- **Range**: Typically 1000-100000
- **Usage**: Larger values provide more statistical power but take longer to generate

---

### **Historical Bias Parameters**
*Bias that exists in the real world and gets captured in data*

#### **`l_h_q`** (float)
- **Default**: `1.0`
- **Parameter Type**: Historical bias coefficient
- **What it controls**: Systematic reduction of Q values for group A=1
- **Range**: `0.0` (no bias) to `2.0+` (severe bias)
- **Effect**: 
  - Q values for A=1 become systematically lower
  - Simulates historical disadvantage in feature Q
- **Example**: Credit scores historically lower for minorities
- **Zero for no bias**: Set to `0.0`

#### **`l_h_r`** (float)
- **Default**: `1.5`
- **Parameter Type**: Historical bias coefficient
- **What it controls**: Systematic reduction of R values for group A=1
- **Range**: `0.0` (no bias) to `2.0+` (severe bias)
- **Effect**:
  - R values for A=1 become systematically lower
  - Creates correlation between A and legitimate feature R
- **Example**: Gender pay gap (income lower for women)
- **Zero for no bias**: Set to `0.0`

#### **`l_y`** (float)
- **Default**: `4.0`
- **Parameter Type**: Historical bias coefficient
- **What it controls**: Direct reduction in Y=1 probability for group A=1
- **Range**: `0.0` (no bias) to `5.0+` (severe bias)
- **Effect**:
  - Y=1 becomes less likely for A=1, **independent of features**
  - This is **direct discrimination** in labels
  - **Most severe** type of bias
- **Example**: Loan officers approving fewer loans for minorities regardless of qualifications
- **Zero for no bias**: Set to `0.0`

#### **`l_y_b`** (float)
- **Default**: `0.0`
- **Parameter Type**: Interaction proxy bias coefficient
- **What it controls**: Non-linear bias where A=1 with **high R values** get lower Y
- **Range**: `0.0` (no bias) to `2.0+` (strong interaction bias)
- **Effect**:
  - Creates complex discrimination pattern
  - Bias depends on interaction between A and R
  - High-R individuals in A=1 are particularly disadvantaged
- **Example**: High-earning minorities face discrimination
- **Zero for no bias**: Keep at `0.0`

---

### **Measurement Bias Parameters**
*Bias introduced through incorrect or noisy measurement*

#### **`l_m_y`** (float)
- **Default**: `0.0`
- **Parameter Type**: Measurement bias coefficient
- **What it controls**: Magnitude of measurement error in Y labels
- **Range**: `0.0` (no error) to `2.0+` (high error)
- **Effect**:
  - Y labels become noisy/mislabeled
  - Error can differ systematically between groups
  - Works with `l_m_y_non_linear` flag
- **Example**: Recidivism measured by arrests (proxy) not actual re-offending
- **Zero for no bias**: Keep at `0.0`

#### **`l_m_y_non_linear`** (boolean)
- **Default**: `False`
- **Parameter Type**: Measurement bias type flag
- **What it controls**: Whether measurement bias on Y is linear or non-linear
- **Values**:
  - `False`: Linear measurement bias (error proportional to value)
  - `True`: Non-linear measurement bias (error conditional on R values)
- **Effect**:
  - Changes how `l_m_y` affects the data
  - Non-linear creates more complex error patterns
- **Usage**: Only matters when `l_m_y > 0`
- **Zero for no bias**: Set to `False`

#### **`l_m`** (float)
- **Default**: `1.0`
- **Parameter Type**: Measurement bias coefficient
- **What it controls**: Magnitude of measurement error in R
- **Range**: `0.0` (no error) to `2.0+` (high error)
- **Effect**:
  - When `l_m > 0`, true variable R is replaced by noisy proxy P
  - P = R + measurement_error
  - Model sees P instead of true R
- **Example**: Self-reported income (P) vs actual income (R)
- **Zero for no bias**: Set to `0.0`

---

### **Representation Bias Parameters**
*Bias from sampling/selection process*

#### **`p_u`** (float)
- **Default**: `1.0`
- **Parameter Type**: Undersampling proportion
- **What it controls**: Percentage of A=1 samples to **remove**
- **Range**: `0.0` (keep all) to `1.0` (remove all)
- **Effect**:
  - Removes proportion `p_u` of A=1 samples from dataset
  - Creates imbalanced dataset
  - Final A=1 count ≈ original_count × (1 - p_u)
- **Example**: 
  - `p_u=0.0`: Keep all A=1 samples (balanced)
  - `p_u=0.5`: Remove 50% of A=1 samples
  - `p_u=0.9`: Remove 90% of A=1 samples (severe imbalance)
- **⚠️ WARNING**: `p_u=1.0` removes **ALL** A=1 samples (extreme)
- **Zero for no bias**: Set to `0.0`

#### **`l_r`** (boolean)
- **Default**: `False`
- **Parameter Type**: Conditional representation bias flag
- **What it controls**: Whether undersampling is conditional on R values
- **Values**:
  - `False`: Simple random undersampling (if `p_u > 0`)
  - `True`: Undersampling conditional on R (non-random)
- **Effect**:
  - When `True`: A=1 samples preferentially removed when R is high/low
  - Creates more complex selection bias
  - Certain (A, R) combinations become rare
- **Example**: High-income minorities underrepresented in dataset
- **Usage**: Only meaningful when `p_u > 0`
- **Zero for no bias**: Set to `False`

---

### **Structural Parameters**
*Control the causal structure and relationships*

#### **`l_q`** (float)
- **Default**: `2.0`
- **Parameter Type**: Structural coefficient
- **What it controls**: How strongly Q influences Y in the causal model
- **Range**: `0.0` (Q doesn't matter) to `5.0+` (Q very important)
- **Effect**:
  - Higher values make Q a stronger predictor of Y
  - Changes the relative importance of Q vs R
  - Structural, not bias-related
- **Example**: How much credit score matters for loan approval
- **Zero for minimal influence**: Set to `0.0`

#### **`l_r_q`** (float)
- **Default**: `0.0`
- **Parameter Type**: Structural coefficient
- **What it controls**: Causal influence from R to Q
- **Range**: `0.0` (no influence) to `2.0+` (strong influence)
- **Effect**:
  - Higher values make Q more dependent on R
  - Creates stronger causal chain A→R→Q→Y
  - Structural, not bias-related
- **Example**: Income affecting credit score calculation
- **Zero for no influence**: Keep at `0.0`

---

### **Other Parameters**

#### **`l_o`** (boolean)
- **Default**: `False`
- **Parameter Type**: Omitted variable flag
- **What it controls**: Whether important variable R (or P) is excluded from dataset
- **Values**:
  - `False`: R/P included in dataset
  - `True`: R/P completely removed (omitted variable bias)
- **Effect**:
  - When `True`: Dataset lacks important predictor
  - Model forced to use proxies that may correlate with A
  - Simulates missing important information
- **Example**: Dataset lacks income information for loan prediction
- **Zero for no bias**: Set to `False`

#### **`sy`** (float)
- **Default**: `5.0`
- **Parameter Type**: Label noise standard deviation
- **What it controls**: Amount of random noise added to Y labels
- **Range**: `0.0` (no noise) to `10.0+` (very noisy)
- **Effect**:
  - Adds random Gaussian noise to Y generation
  - Affects both groups equally (not systematic bias)
  - Simulates natural variability or errors
  - Higher values → more label flips (0↔1)
- **Example**: Random errors in medical diagnoses
- **Zero for no noise**: Set to `0.0`

#### **`thr_supp`** (float)
- **Default**: `1.0`
- **Parameter Type**: Suppression threshold
- **What it controls**: Removes features too strongly correlated with A
- **Range**: `0.0` to `1.0`
- **Effect**:
  - Features with correlation to A above threshold are removed
  - Higher threshold → more features kept
  - `1.0` effectively disables suppression
- **Example**: Remove features that directly reveal protected attribute
- **Standard setting**: Keep at `1.0`

---

## 🎯 Parameter Combinations for Common Scenarios

### **Scenario 1: Pure Baseline (No Bias)**
```python
biasondemand.generate_dataset(
    path="/baseline",
    dim=10000,
    l_y=0.0,          # No historical bias on Y
    l_h_r=0.0,        # No historical bias on R
    l_h_q=0.0,        # No historical bias on Q
    l_m=0.0,          # No measurement bias on R
    l_m_y=0.0,        # No measurement bias on Y
    p_u=0.0,          # No undersampling
    l_r=False,        # No conditional bias
    l_o=False,        # Include all variables
    l_y_b=0.0,        # No interaction bias
    sy=0.0,           # No label noise
    l_q=0.0,          # Q doesn't influence Y structurally
    l_r_q=0.0,        # R doesn't influence Q
    thr_supp=1.0
)
```
**Result**: Fair dataset where A doesn't influence outcomes

---

### **Scenario 2: Historical Feature Bias Only**
```python
biasondemand.generate_dataset(
    path="/hist_bias_features",
    dim=10000,
    l_h_r=0.5,        # Moderate bias on R
    l_h_q=0.3,        # Moderate bias on Q
    # All other bias parameters = 0
    l_y=0.0, l_m=0.0, l_m_y=0.0, p_u=0.0,
    l_r=False, l_o=False, l_y_b=0.0, sy=0.0,
    l_q=0.0, l_r_q=0.0, thr_supp=1.0
)
```
**Result**: R and Q values systematically lower for A=1, but Y not directly biased

---

### **Scenario 3: Direct Label Discrimination**
```python
biasondemand.generate_dataset(
    path="/label_discrimination",
    dim=10000,
    l_y=1.0,          # Strong direct bias on Y
    # All other bias parameters = 0
    l_h_r=0.0, l_h_q=0.0, l_m=0.0, l_m_y=0.0, p_u=0.0,
    l_r=False, l_o=False, l_y_b=0.0, sy=0.0,
    l_q=0.0, l_r_q=0.0, thr_supp=1.0
)
```
**Result**: Y=1 less likely for A=1 regardless of features (most severe bias)

---

### **Scenario 4: Severe Imbalance (Undersampling)**
```python
biasondemand.generate_dataset(
    path="/severe_imbalance",
    dim=15000,        # Start with more samples
    p_u=0.9,          # Remove 90% of A=1 samples
    # All other bias parameters = 0
    l_y=0.0, l_h_r=0.0, l_h_q=0.0, l_m=0.0, l_m_y=0.0,
    l_r=False, l_o=False, l_y_b=0.0, sy=0.0,
    l_q=0.0, l_r_q=0.0, thr_supp=1.0
)
```
**Result**: Only 10% of original A=1 samples remain (severe imbalance)

---

### **Scenario 5: Measurement Bias**
```python
biasondemand.generate_dataset(
    path="/measurement_bias",
    dim=10000,
    l_m=0.8,          # Strong measurement bias on R
    l_m_y=0.5,        # Moderate measurement bias on Y (linear)
    l_m_y_non_linear=False,
    # All other bias parameters = 0
    l_y=0.0, l_h_r=0.0, l_h_q=0.0, p_u=0.0,
    l_r=False, l_o=False, l_y_b=0.0, sy=0.0,
    l_q=0.0, l_r_q=0.0, thr_supp=1.0
)
```
**Result**: R observed through noisy proxy P, Y labels have errors

---

### **Scenario 6: Label Noise Only**
```python
biasondemand.generate_dataset(
    path="/label_noise",
    dim=10000,
    sy=3.0,           # High label noise
    # All other bias parameters = 0
    l_y=0.0, l_h_r=0.0, l_h_q=0.0, l_m=0.0, l_m_y=0.0, p_u=0.0,
    l_r=False, l_o=False, l_y_b=0.0,
    l_q=0.0, l_r_q=0.0, thr_supp=1.0
)
```
**Result**: Random noise in labels (affects both groups equally)

---


## 🔬 Systematic Bias Study: Varying Single Parameters

### **Template for Isolating One Bias Type:**
```python
import numpy as np

# Define range of bias levels
bias_levels = np.arange(0.0, 1.1, 0.1)  # [0.0, 0.1, 0.2, ..., 1.0]

# Generate datasets with varying bias
for level in bias_levels:
    biasondemand.generate_dataset(
        path=f"/hist_bias_Q_level_{level:.2f}",
        dim=10000,
        l_h_q=level,      # ← VARY THIS
        # Set ALL other bias parameters to 0
        l_y=0.0, l_h_r=0.0, l_m=0.0, l_m_y=0.0, p_u=0.0,
        l_r=False, l_o=False, l_y_b=0.0, sy=0.0,
        l_q=0.0, l_r_q=0.0, thr_supp=1.0
    )
```

This creates 11 datasets with isolated Q bias ranging from 0.0 (none) to 1.0 (severe).


---

## 🎓 Summary

BiasOnDemand provides fine-grained control over **where** and **how much** bias enters your data:

| Bias Entry Point | Parameters | Effect |
|-----------------|------------|--------|
| **Historical (real-world)** | `l_h_q`, `l_h_r`, `l_y`, `l_y_b` | Features/labels systematically different for A=1 |
| **Measurement** | `l_m`, `l_m_y`, `l_m_y_non_linear` | Variables measured incorrectly |
| **Representation** | `p_u`, `l_r` | Some groups over/under-represented |
| **Omission** | `l_o` | Important variables missing |
| **Random** | `sy` | Label noise (not systematic bias) |


