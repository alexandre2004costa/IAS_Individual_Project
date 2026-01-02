
# ⚖️ BiasOnDemand Evaluation Pipeline  
**Fairness, Explainability, and Robustness under Controlled Bias Conditions**

This project provides a **systematic experimental pipeline** to study how different types of bias affect three fundamental pillars of trustworthy machine learning: **fairness**, **explainability**, and **robustness**.  
It is inspired by the *BiasOnDemand* framework and relies on **synthetically generated datasets with explicit causal structures**.

---

## 📚 About the Project

Machine learning models are often evaluated in isolation with respect to fairness, interpretability, or robustness. This project investigates **how these pillars interact** under controlled bias mechanisms such as:

- Historical bias  
- Measurement bias  
- Proxy and interaction bias  
- Representation bias  
- Undersampling  
- Label noise  

Each bias is introduced with increasing intensity, allowing fine-grained analysis of its effects.

---

## 🧠 Core Idea

The pipeline relies on **causal data generation** using Directed Acyclic Graphs (DAGs), where:

- **A** — Sensitive attribute (binary)
- **R** — Legitimate resource (continuous, causally affected by A)
- **Q** — Performance-related feature
- **Y** — Binary target variable
- **P** — Proxy for R (used under measurement bias)

This structure enables principled reasoning about *why* certain metrics degrade under specific bias conditions.


---

## 🚀 Running the Project

### 1️⃣ Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Full Pipeline (Recommended)
```bash
chmod +x run.sh
./run.sh
```

### This will:

- Generate all biased datasets

- Evaluate fairness metrics

- Evaluate explainability metrics

- Evaluate robustness metrics

- Store results under results/

▶️ Run Individual Steps (Optional)
### Generate datasets
```bash
python generate.py
```
### Fairness evaluation
```bash
python fairness_train.py
```
### Explainability evaluation
```bash
python explanation_train.py
```
### Robustness evaluation
```bash
python robustness_train.py
```
## 📊 Results and Visualization

All figures used in the analysis are generated through the notebooks in:
```bash
graphs/
```

### Each notebook loads precomputed results and produces publication-ready plots:

- Fairness: Disparate Impact, Equalized Odds, ... and Overall Fairness

- Explainability: Clarity, Distinctiveness, Coverage (DoX-inspired)

- Robustness: Stability, Reliability, Resilience

## 📝 Documentation

### The explanations/ folder contains detailed markdown files describing:

- The BiasOnDemand-inspired data generation process

- Causal interpretation of each bias mechanism

- Metric definitions and motivations

- Observed empirical behaviors across pillars

- These files are meant to complement the experimental results and guide interpretation.
