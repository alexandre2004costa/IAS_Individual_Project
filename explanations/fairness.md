# Fairness Evaluation Metrics & Methodology

This document outlines the specific metrics computed by the `FairnessEvaluator` class and explains the methodology used to calculate the **Overall Fairness Score**.

The evaluation compares a specific sub-group (A = 1 -> *Minority*) against a **Reference Group** (A = 0 -> *Majority*).

## 1. Base Group Metrics
Before calculating fairness, the system calculates standard performance metrics for every individual group ($g$).

| Metric | Code Variable | Definition | Interpretation |
| :--- | :--- | :--- | :--- |
| **Success Rate** | `success_rate` | $P(\hat{Y}=1 \| G=g)$ | The percentage of the group predicted as "Positive" (e.g., hired, approved). |
| **TPR** | `tpr` | True Positive Rate (Recall) | Of the people who *should* be approved, what % did we approve? |
| **FPR** | `fpr` | False Positive Rate | Of the people who *should not* be approved, what % did we approve anyway? |
| **FNR** | `fnr` | False Negative Rate | Of the people who *should* be approved, what % did we reject? |
| **Accuracy** | `accuracy` | $(TP+TN) / Total$ | How often is the model correct for this specific group? |

---

## 2. Fairness (Comparative) Metrics
These metrics measure the disparity between the **Comparision Group** and the **Reference Group**.

### A. Selection Disparities (Outcome Fairness)
Based on class slides and pratical activity on Bias and Fairness

*Does the model approve one group more often than the other?*

#### **DI: Disparate Impact**
* **Formula:** $\frac{\text{Success Rate}_{group}}{\text{Success Rate}_{ref}}$
* **Ideal Value:** `1.0`
* **Description:** The ratio of acceptance rates.
    * Values < 0.8 indicate the group is favored less (often used as a legal threshold).
    * Values > 1.25 indicate the group is favored more.

#### **SP: Statistical Parity**
* **Formula:** $\text{Success Rate}_{group} - \text{Success Rate}_{ref}$
* **Ideal Value:** `0.0`
* **Description:** The absolute difference in acceptance rates. A negative value means the comparison group is selected less frequently.

### B. Error Rate Disparities (Equalized Odds)
*Is the model equally accurate/mistaken for both groups?*

#### **EOD: Equalized Odds Difference**
* **Formula:** $\frac{|TPR_{group} - TPR_{ref}| + |FPR_{group} - FPR_{ref}|}{2}$
* **Ideal Value:** `0.0`
* **Description:** The average **absolute** difference between True Positive Rates and False Positive Rates. This ensures the model isn't punishing one group with lower recall or higher false alarms.

#### **AOD: Average Odds Difference**
* **Formula:** $\frac{(FPR_{group} - FPR_{ref}) + (TPR_{group} - TPR_{ref})}{2}$
* **Ideal Value:** `0.0`
* **Description:** Similar to EOD but keeps the **sign** (+/-). This helps identify if a group systematically has higher or lower rates across the board.

#### **PE: Predictive Equality**
* **Formula:** $FPR_{group} - FPR_{ref}$
* **Ideal Value:** `0.0`
* **Description:** The difference in False Positive Rates.
    * Important in punitive contexts (e.g., fraud detection), ensuring one group isn't falsely accused more often than another.

#### **EOR_TPR: Equal Opportunity Ratio (TPR)**
* **Formula:** $\frac{TPR_{group}}{TPR_{ref}}$
* **Ideal Value:** `1.0`
* **Description:** The ratio of Recalls. Ensures qualified candidates in both groups are identified at the same rate.

#### **EOR_FNR: Equal Opportunity Ratio (FNR)**
* **Formula:** $\frac{FNR_{group}}{FNR_{ref}}$ (Includes logic to handle division by zero)
* **Ideal Value:** `1.0`
* **Description:** The ratio of False Negatives. Ensures qualified candidates are not "missed" disproportionately in one group.

#### **ACC: Accuracy Ratio**
* **Formula:** $\frac{\text{Accuracy}_{group}}{\text{Accuracy}_{ref}}$
* **Ideal Value:** `1.0`
* **Description:** Ensures the model performs equally well (overall correctness) for both groups.

---

## 3. The Overall Fairness Score
The `overall_fairness_score` is a composite metric designed to provide a single number summarizing the fairness of the model.

**Evaluation Rule:**
* **Lower is Better.**
* **0.0** indicates a perfectly fair model (all metrics hit their ideals).
* Higher scores indicate greater deviation from fairness standards.

### Calculation Logic
The score is a **weighted mean of absolute deviations**. Each metric is compared to its ideal value, multiplied by a "Tier" weight based on its importance, and scaled by a global factor.

$$\text{Score} = \text{Mean}(\sum (\text{Weight} \times |\text{Actual} - \text{Ideal}|)) \times 1.5$$

### Metric Weights & Tiers

The algorithm prioritizes legal and outcome-based fairness (Tier 1) over technical accuracy metrics (Tier 3).

| Tier | Metric | Weight | Rationale |
| :--- | :--- | :--- | :--- |
| **Tier 1 (Critical)** | **DI** (Disparate Impact) | **1.00** | Primary legal standard for discrimination. |
| **Tier 1 (Critical)** | **SP** (Statistical Parity) | **0.90** | Direct measure of outcome inequality. |
| **Tier 1 (Critical)** | **EOD** (Equalized Odds) | **0.90** | Best overall measure of error balance. |
| **Tier 2 (High)** | **AOD** (Average Odds) | **0.70** | Directional check on error balance. |
| **Tier 2 (High)** | **PE** (Predictive Equality) | **0.60** | Crucial for avoiding false accusations. |
| **Tier 3 (Medium)** | **EOR_TPR** | **0.50** | Ensures qualified people get in. |
| **Tier 3 (Low)** | **ACC** (Accuracy Ratio) | **0.30** | General model stability check. |
| **Tier 3 (Low)** | **EOR_FNR** | **0.20** | Secondary check on missed opportunities. |

### Why this weighting?
1.  **Selection Rates first:** Most fairness audits focus first on *who got selected* (DI/SP).
2.  **Balanced Errors second:** If selection rates are equal, we check if we are making mistakes equally (EOD).
3.  **Specific Errors third:** Finally, we look at specific types of errors (False Positives vs False Negatives).