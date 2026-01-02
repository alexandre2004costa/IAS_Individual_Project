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
Based on "Certified Adversarial Robustness via Randomized Smoothing"(https://arxiv.org/abs/1902.02918). Idea taken: While the paper focuses on "certifying" deep learning, the underlying mechanism—adding Gaussian noise to an input and checking if the prediction stays consistent—is the standard way to quantify a model's local stability.
*Does the model give consistent answers for virtually identical inputs?*

We take a data point $x$ and generate variations $x'$ by adding small, random Gaussian noise ($\sigma=0.05$). We then check if the model's prediction changes.

* **Low Stability:** A tiny change in input (e.g., income changes by $1) flips the approval decision. This indicates the decision boundary is too "jittery" or overfitted.
* **High Stability:** The model consistently predicts the same class for the neighborhood of $x$.

**Calculation:**
$$\text{Stability} = \frac{1}{N} \sum_{i=1}^{N} (\text{Prediction}(x_i) == \text{Prediction}(x_i + \text{noise}))$$

### B. Resilience Score (The "Stress" Test)
Based on "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations"(https://arxiv.org/abs/1903.12261). Idea taken: This work introduced "mCE" (mean Corruption Error). It argues that a model isn't truly robust unless its accuracy stays high when the data is "smeared" or "jittered" by environmental factors.

*How much accuracy is lost when data quality drops?*

Unlike Stability (which checks if the *prediction* changes), Resilience checks if the *correctness* changes. We corrupt the data with significant noise ($\sigma=0.1$) and measure the drop in accuracy.

* **Calculation:**
    $$\text{Resilience} = \min \left( \frac{\text{Accuracy}_{\text{corrupted}}}{\text{Accuracy}_{\text{clean}}}, 1.0 \right)$$

* **Interpretation:** A score of $1.0$ means the model is impervious to this level of noise. A score of $0.5$ means half the accuracy was lost.

### C. Reliability Score (The "Honesty" Test)
Based on "Understanding Model Calibration - A gentle introduction and visual exploration of calibration and the expected calibration error (ECE)" (https://arxiv.org/html/2501.19047v2#:~:text=This%20definition%20of%20calibration%2C%20ensures,et%20al.%2C%202017). Idea taken: This paper popularized the use of ECE to measure the gap between accuracy and confidence.

*Can we trust the confidence scores?*

This utilizes **Expected Calibration Error (ECE)**. If a weather forecaster predicts rain with 70% confidence 100 times, it should rain exactly 70 of those times. If it rains 100 times, they are under-confident; if it rains 0 times, they are over-confident.



* **Calculation:**
    We quantify this using Expected Calibration Error (ECE):

    1. Binning: Predictions are grouped into "confidence bins" (e.g., 0-10%, 10-20%, etc.).

    2. Gap Analysis: For each bin, we calculate the absolute difference between the average confidence (what the model expected) and the actual accuracy (what actually happened).

    3. Weighting: The gaps are averaged, weighted by the number of samples in each bin, to produce the ECE:
    So the ECE.$$ECE = \sum_{m=1}^{M} \frac{|B_m|}{N} |acc(Bm) - conf(Bm)|$$
        where B is used for representing ”bins” and m for the bin number, while acc and conf are:
            - Average Confidence of Bin $m$:$$\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{j \in B_m} \hat{p}_j$$Where $\hat{p}_j$ is the predicted probability (confidence) for sample $j$.Average Accuracy of Bin $m$:$$\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{j \in B_m} \mathbb{m}(\hat{y}_j = y_j)$$
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


**PS: This file is AI generated with some human changes.**