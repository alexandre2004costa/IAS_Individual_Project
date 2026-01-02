# DoX-Inspired Explainability Metrics

This document outlines the methodology for the **Simplified DoX Evaluator**, a quantitative framework designed to measure the quality of AI explanations (specifically SHAP values). 

Inspired by the *Degree of Explainability (DoX)* framework by Sovrano & Vitali (2023) (https://arxiv.org/abs/2109.05327), this pipeline does not judge *what* the model predicts, but rather **how well it explains itself**. It breaks down "explainability" into three logical dimensions: **Clarity**, **Distinctiveness**, and **Coverage**.

---

## 1. The Core Philosophy
A good explanation should answer three fundamental questions for the user. We calculate a score (0.0 to 1.0) for each question.

| Dimension | Question | Core Concept | Ideal State |
| :--- | :--- | :--- | :--- |
| **Clarity** | **WHY** is this result happening? | **Signal-to-Noise** | A few clear drivers stand out from the background noise. |
| **Distinctiveness** | **HOW** do features compare? | **differentiation** | It is easy to distinguish which features are more important than others. |
| **Coverage** | **WHAT** is the full picture? | **Completeness** | The top features account for the vast majority of the prediction's weight. |



---

## 2. Metric Definitions

### A. Clarity Score (The "WHY")
*Can the user clearly identify the main drivers of the prediction?*

This measures if the explanation is focused or "fuzzy". An explanation where every feature contributes equally ($0.1, 0.1, 0.1...$) is confusing/unclear. An explanation where a few features dominate ($0.8, 0.1, 0.05...$) is clear.

**Calculated using three sub-components:**
1.  **Concentration (Entropy):** Measures the disorder of the importance distribution. Lower entropy (high concentration) = High Score.
2.  **Dominance:** The sum of the probabilities of the top $K$ features.
3.  **The Gap:** The relative distance between the average importance of the top $K$ features and the rest.

### B. Distinctiveness Score (The "HOW")
*Is the hierarchy of importance clear?*

This measures the internal structure of the explanation. Even if we know the top features, can we rank them easily? 
* **Bad Distinctiveness:** Feature A (0.35) and Feature B (0.34) look almost identical.
* **Good Distinctiveness:** Feature A (0.50) is clearly stronger than Feature B (0.20).

**Calculated using:**
1.  **Range:** The distance between the maximum and minimum SHAP values.
2.  **Coefficient of Variation (CV):** The standard deviation divided by the mean. This captures the "spread" of the importance values.

### C. Coverage Score (The "WHAT")
*Do the top features explain enough of the decision?*

This measures completeness. This score penalizes the explanation if the top K features hide too much information. In this case we are only using the 1 feature(K=1) because the dataset has a low number of attributes.

**Calculation:**
* **Ratio:** $\frac{\sum \text{Top K Importance}}{\sum \text{Total Importance}}$
* **Scaling:** Uses a sigmoid-like curve.
    * If the Top $K$ explain < 80% of the decision, the score drops rapidly.
    * If they explain > 80%, the score asymptotically approaches 1.0.

---

## 3. The Overall DoX Score

The single composite score for explainability is a weighted average of the three dimensions. The weights reflect the cognitive priority of human users (we care most about "Why").

$$\text{DoX} = (0.40 \times \text{Clarity}) + (0.35 \times \text{Distinctiveness}) + (0.25 \times \text{Coverage})$$

| Metric | Weight | Rationale |
| :--- | :--- | :--- |
| **Clarity** | **40%** | The most critical aspect. If a user can't isolate the main cause, the explanation fails immediately. |
| **Distinctiveness** | **35%** | Essential for nuance. Users need to know *how much* more important Factor A is than Factor B. |
| **Coverage** | **25%** | A hygiene metric. As long as we aren't hiding critical info, this is secondary to clarity. |

---

**The Hypothesis:**
A model might be accurate for everyone, but *harder to explain* for certain minority groups. This often happens because the model relies on complex, non-linear interactions for outliers or minorities, resulting in "fuzzier" SHAP values.

**PS: This file is AI generated with some human changes.**