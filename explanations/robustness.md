# Model Robustness Metrics

This document outlines the methodology for the **Simplified Robustness Evaluator**, a quantitative framework designed to measure how "tough" and "trustworthy" a model is.

While accuracy measures how often a model is right, **Robustness** measures how easily it can be broken. A robust model should not flip its prediction just because a tiny amount of noise was added to the input, and its stated confidence should match its actual probability of being correct.

---

## 1. The Core Philosophy
We evaluate robustness through three distinct "stress tests," answering three critical questions about the model's behavior.

| Dimension | Question | Core Concept | Ideal State |
| :--- | :--- | :--- | :--- |
| **Stability** | **Does it flicker?** | **Consistency** | The prediction stays the same even if the input data jitters slightly (e.g., sensor noise). |
| **Resilience** | **Does it degrade?** | **Performance Retention** | The model maintains high accuracy even when data quality drops significantly. |
| **Reliability** | **Is it honest?** | **Calibration** | When the model says "I am 80% sure," it is actually correct 80% of the time. |



---

## 2. Metric Definitions

### A. Stability Score (The "Flicker" Test)
*Does the model give consistent answers for virtually identical inputs?*

We take a data point $x$ and generate variations $x'$ by adding small, random Gaussian noise ($\sigma=0.05$). We then check if the model's prediction changes.

* **Low Stability:** A tiny change in input (e.g., income changes by $1) flips the approval decision. This indicates the decision boundary is too "jittery" or overfitted.
* **High Stability:** The model consistently predicts the same class for the neighborhood of $x$.

**Calculation:**
$$\text{Stability} = \frac{1}{N} \sum_{i=1}^{N} (\text{Prediction}(x_i) == \text{Prediction}(x_i + \text{noise}))$$

### B. Resilience Score (The "Stress" Test)
*How much accuracy is lost when data quality drops?*

Unlike Stability (which checks if the *prediction* changes), Resilience checks if the *correctness* changes. We corrupt the data with significant noise ($\sigma=0.1$) and measure the drop in accuracy.

* **Calculation:**
    $$\text{Resilience} = \min \left( \frac{\text{Accuracy}_{\text{corrupted}}}{\text{Accuracy}_{\text{clean}}}, 1.0 \right)$$

* **Interpretation:** A score of $1.0$ means the model is impervious to this level of noise. A score of $0.5$ means half the accuracy was lost.

### C. Reliability Score (The "Honesty" Test)
*Can we trust the confidence scores?*

This utilizes **Expected Calibration Error (ECE)**. If a weather forecaster predicts rain with 70% confidence 100 times, it should rain exactly 70 of those times. If it rains 100 times, they are under-confident; if it rains 0 times, they are over-confident.



* **Calculation:**
    We quantify this using Expected Calibration Error (ECE):

    1. Binning: Predictions are grouped into "confidence bins" (e.g., 0-10%, 10-20%, etc.).

    2. Gap Analysis: For each bin, we calculate the absolute difference between the average confidence (what the model expected) and the actual accuracy (what actually happened).

    3. Weighting: The gaps are averaged, weighted by the number of samples in each bin, to produce the ECE:
    So the ECE.$$ECE = \sum_{i=1}^{B} \frac{n_i}{N} |acc(i) - conf(i)|$$
    4.  $\text{Reliability} = 1.0 - \text{ECE}$
    

---

## 3. The Overall Robustness Score

The single composite score is a weighted average of the three dimensions. Stability is weighted slightly higher because an unstable model is dangerous in deployment (users lose trust if answers "flicker").

$$\text{Robustness} = (0.40 \times \text{Stability}) + (0.30 \times \text{Resilience}) + (0.30 \times \text{Reliability})$$

| Metric | Weight | Rationale |
| :--- | :--- | :--- |
| **Stability** | **40%** | The foundation of trust. Inconsistent automated decisions are often illegal or highly liable in regulated industries. |
| **Resilience** | **30%** | Important for real-world deployment where data is rarely "clean" like the training set. |
| **Reliability** | **30%** | Critical for "Human-in-the-Loop" systems. Users need to know when to override the AI. |
