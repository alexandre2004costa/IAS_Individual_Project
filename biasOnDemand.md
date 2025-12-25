# BiasOnDemand Dataset: Complete Explanation
## Understanding the Data Structure and Parameters

## 📊 Core Dataset Structure

BiasOnDemand generates synthetic datasets based on a **causal model** with the following core variables:

### **The 4 Main Variables:**

```
A → Q → Y  (Causal Chain)
  ↘ R ↗
```

| Variable | Name | Type | Description |
|----------|------|------|-------------|
| **A** | Protected Attribute | Binary (0 or 1) | The sensitive/protected group membership<br>• A=0: Reference/unprivileged group<br>• A=1: Group where bias is introduced |
| **R** | Feature Variable | Continuous | An important feature that influences Y<br>• Causally depends on A<br>• Can be biased or measured incorrectly |
| **Q** | Feature Variable | Continuous | Another important feature that influences Y<br>• Causally depends on both A and R<br>• Can be biased historically |
| **Y** | Target/Label | Binary (0 or 1) | The outcome you want to predict<br>• Depends on Q, R, and potentially A<br>• Can be biased or measured incorrectly |

### **Additional Variables (Generated):**
- **P**: Proxy for R when measurement bias is applied (`l_m > 0`)
- **Additional noise features**: Random features with varying correlation to A

---

## 🔍 The Causal Story

Think of it like a **loan application system**:

| Variable | Real-World Example |
|----------|-------------------|
| **A** | Gender (0=Female, 1=Male) or Race (0=White, 1=Black) |
| **R** | Income level |
| **Q** | Credit score (influenced by both gender/race AND income) |
| **Y** | Loan approval (0=Rejected, 1=Approved) |

### **How it works (Unbiased Case):**
1. Your **income (R)** depends partially on random factors
2. Your **credit score (Q)** is calculated based on your income (R)
3. **Loan approval (Y)** depends on your credit score (Q) and income (R)
4. Your **gender/race (A)** shouldn't matter - but with bias, it does!

### **What bias parameters do:**
They **break the fairness** by making A (protected attribute) inappropriately influence R, Q, or Y.

---

## 🎛️ Dataset Generation Parameters

### **Basic Parameters:**

| Parameter | Description | Default | Effect |
|-----------|-------------|---------|--------|
| `dim` | Number of samples | 10000 | Dataset size |
| `sy` | Label noise std dev | 0.0 | Adds random noise to Y (0=clean, higher=noisier) |
| `l_q` | Importance of Q for Y | 0.0 | How much Q influences Y (structural) |
| `l_r_q` | Influence of R on Q | 0.0 | Causal link strength R→Q |
| `thr_supp` | Correlation threshold | 1.0 | Removes features too correlated with A |

---

## 🚨 Bias Parameters - Detailed Explanation

### **1. Historical Bias** (Bias in the Real World)
*These biases existed in reality and got into the data*

#### **`l_h_q` - Historical Bias on Q**
- **What it does**: Makes Q systematically lower for group A=1
- **Real example**: Credit scores are systematically lower for minority groups due to historical discrimination
- **Effect on data**: Q values for A=1 are reduced
- **Fairness impact**: Model learns to penalize A=1 even if Q shouldn't depend on A

```python
# Without bias: Q_A=0 = Q_A=1 (on average)
# With l_h_q=0.5: Q_A=1 < Q_A=0 (A=1 has lower Q values)
```

#### **`l_h_r` - Historical Bias on R**
- **What it does**: Makes R systematically lower for group A=1
- **Real example**: Income (R) is lower for women due to historical pay gaps
- **Effect on data**: R values for A=1 are reduced
- **Fairness impact**: Legitimate feature R now correlates with A

#### **`l_y` - Historical Bias on Y (Direct Label Bias)**
- **What it does**: Makes Y=1 less likely for group A=1, *independent of features*
- **Real example**: Loan officers historically approved fewer loans for minorities even with same qualifications
- **Effect on data**: Y is directly reduced for A=1
- **Fairness impact**: **Most severe** - direct discrimination in labels

#### **`l_y_b` - Interaction Proxy Bias**
- **What it does**: Complex bias where A=1 with high R values get lower Y
- **Real example**: High-earning minorities face discrimination ("they don't need help")
- **Effect on data**: For A=1 AND high R → lower Y
- **Fairness impact**: Non-linear discrimination pattern

---

### **2. Measurement Bias** (Bias in How We Measure)
*Variables are measured incorrectly*

#### **`l_m` - Measurement Bias on R**
- **What it does**: R is observed through a noisy proxy P instead
- **Real example**: Income self-reported (P) vs actual income (R)
- **Effect on data**: Column R is replaced by P which has measurement error
- **Fairness impact**: Model uses inaccurate feature, different errors for different groups

```python
# If l_m > 0: You get P (proxy) instead of R (true value)
# P = R + noise (measurement error)
```

#### **`l_m_y` - Measurement Bias on Y**
- **What it does**: Y labels are measured with error (mislabeled)
- **Real example**: Recidivism prediction where arrests (Y) don't equal actual crimes
- **Effect on data**: Some Y=0 become Y=1 and vice versa
- **Fairness impact**: Training on wrong labels, error rates differ by group

---

### **3. Representation Bias** (Bias in Who's in the Data)
*Some groups are under/over-represented*

#### **`p_u` - Undersampling of A=1**
- **What it does**: Removes percentage p_u of samples where A=1
- **Real example**: Only 10% of loan applications from minorities in dataset
- **Effect on data**: Fewer samples for A=1 group
- **Values**: 
  - `p_u=0.0` → Keep all A=1 samples (balanced)
  - `p_u=0.5` → Remove 50% of A=1 samples
  - `p_u=0.9` → Remove 90% of A=1 samples (severe imbalance)
- **Fairness impact**: Model has less data to learn about A=1, worse predictions

**⚠️ IMPORTANT**: Higher p_u = MORE bias (opposite of other lambda parameters)

#### **`l_r=True` - Representation Bias (Conditional)**
- **What it does**: Undersamples A=1 *conditional on R values*
- **Real example**: High-income minorities are rare in dataset (conditional undersampling)
- **Effect on data**: Removes A=1 samples preferentially when R is high/low
- **Fairness impact**: Model never sees certain (A, R) combinations

---

### **4. Omitted Variable Bias**

#### **`l_o=True` - Omitted Variable**
- **What it does**: Removes R (or P) from the dataset entirely
- **Real example**: Important feature not collected (e.g., no income data)
- **Effect on data**: R column is deleted, can't be used for prediction
- **Fairness impact**: Model relies on proxies of R that may correlate with A

---

## 📈 How Parameters Affect Fairness Metrics

### **Expected Impact on Your Fairness Metrics:**

| Bias Type | ↑ Parameter | DI | SP | EOD | AOD | Effect |
|-----------|------------|----|----|-----|-----|---------|
| **Historical (Q, R, Y)** | ↑ l_h_q, l_h_r, l_y | ↓ | ↓ (more negative) | ↑ | ↓ (more negative) | A=1 disadvantaged |
| **Measurement (R, Y)** | ↑ l_m, l_m_y | ↓ | ↓ | ↑ | Varies | Noisy predictions |
| **Undersampling** | ↑ p_u | ↓ | ↓ | ↑ | ↓ | Less data for A=1 |
| **Label Noise** | ↑ sy | ~1.0 | ~0.0 | ↑ | ~0.0 | Random degradation |

---

## 🎯 What Each Bias Parameter Actually Changes in Your Data

### **Baseline (No Bias):**
```python
biasondemand.generate_dataset(path="/baseline", dim=10000)
# Result: Fair dataset where A doesn't influence outcomes
# DI ≈ 1.0, SP ≈ 0.0, EOD ≈ 0.0
```

### **Historical Bias on Q (`l_h_q`):**
```python
# l_h_q = 0.3
# What happens:
# - Q values for A=1 are reduced by 30%
# - Model sees: A=1 → lower Q → lower Y
# - Even though Q SHOULDN'T depend on A!

# Your metrics:
# DI: 1.0 → 0.75 (25% reduction in selection rate)
# SP: 0.0 → -0.15 (15% lower outcomes for A=1)
```

### **Direct Label Bias (`l_y`):**
```python
# l_y = 0.5
# What happens:
# - Y=1 probability directly reduced for A=1
# - This is pure discrimination - A directly causes Y to change

# Your metrics:
# DI: 1.0 → 0.60 (40% reduction - SEVERE)
# SP: 0.0 → -0.30 (30% disparity)
# EOD: 0.0 → 0.25 (large inequality in error rates)
```

### **Undersampling (`p_u`):**
```python
# p_u = 0.7
# What happens:
# - 70% of A=1 samples are deleted
# - Model trained on mostly A=0 examples
# - Poor performance on A=1 (never seen it much)

# Your metrics:
# DI: 1.0 → 0.70
# EOD: 0.0 → 0.20 (worse predictions for A=1)
```

### **Label Noise (`sy`):**
```python
# sy = 0.3
# What happens:
# - 30% noise added to Y labels (random flips)
# - Affects both groups (not biased, just noisy)
# - Overall accuracy drops

# Your metrics:
# DI: ~1.0 (stays fairish)
# SP: ~0.0 (stays fairish)
# EOD: 0.0 → 0.15 (randomness affects error rates)
# Accuracy: drops significantly
```
