import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SimplifiedRobustnessEvaluator:
    """
    Evaluates model robustness using:
    1. Stability: Prediction consistency under noise.
    2. Resilience: Accuracy retention under input perturbation.
    3. Reliability: Calibration of confidence (1 - ECE).
    """
    
    def __init__(self, model_type='logistic', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.results = {}

    def train_model(self, X_train, y_train):
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.model.fit(X_train, y_train)

    def calculate_stability_score(self, X, n_iter=5, noise_level=0.05):
        """Measures prediction consistency when features are jittered."""
        original_preds = self.model.predict(X)
        stability_runs = []
        for _ in range(n_iter):
            noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0).values
            X_noisy = X + noise
            noisy_preds = self.model.predict(X_noisy)
            stability_runs.append(np.mean(original_preds == noisy_preds))
        return np.mean(stability_runs)

    def calculate_resilience_score(self, X, y, noise_level=0.1):
        """Measures accuracy retention under significant perturbation."""
        clean_acc = accuracy_score(y, self.model.predict(X))
        noise = np.random.normal(0, noise_level, X.shape) * np.std(X, axis=0).values
        X_corrupted = X + noise
        corrupted_acc = accuracy_score(y, self.model.predict(X_corrupted))
        return min(corrupted_acc / (clean_acc + 1e-10), 1.0)

    def calculate_reliability_score(self, X, y, n_bins=10):
        """Measures calibration (1 - Expected Calibration Error)."""
        probs = self.model.predict_proba(X)[:, 1]
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i+1])
            if np.any(mask):
                bin_acc = np.mean(y[mask])
                bin_conf = np.mean(probs[mask])
                ece += np.sum(mask) * np.abs(bin_acc - bin_conf)
        return max(1.0 - (ece / len(y)), 0.0)

    def evaluate(self, X_train, X_test, y_train, y_test, sensitive_attr):
        X_train_f = X_train.drop(columns=[sensitive_attr])
        X_test_f = X_test.drop(columns=[sensitive_attr])
        self.train_model(X_train_f, y_train)
        
        group_results = {}
        for val in sorted(X_test[sensitive_attr].unique()):
            mask = X_test[sensitive_attr] == val
            X_g, y_g = X_test_f[mask], y_test[mask]
            
            s = self.calculate_stability_score(X_g)
            res = self.calculate_resilience_score(X_g, y_g)
            rel = self.calculate_reliability_score(X_g, y_g)
            overall = (0.4 * s) + (0.3 * res) + (0.3 * rel)
            
            group_results[f"group_{val}"] = {
                'group': f"Group_{val}",
                'n_samples': len(X_g),
                'stability': s,
                'resilience': res,
                'reliability': rel,
                'overall_robustness': overall
            }
            
        groups = list(group_results.keys())
        rob_gap = abs(group_results[groups[0]]['overall_robustness'] - 
                      group_results[groups[1]]['overall_robustness']) if len(groups) >= 2 else 0.0

        self.results = {
            'model_type': self.model_type,
            'sensitive_attribute': sensitive_attr,
            'group_results': group_results,
            'robustness_gap': rob_gap
        }
        return self.results

    def results_to_dataframe(self):
        """Convert nested results to a flat DataFrame for CSV storage."""
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        rows = []
        for _, stats in self.results['group_results'].items():
            row = {
                'model_type': self.results['model_type'],
                'sensitive_attribute': self.results['sensitive_attribute'],
                'group': stats['group'],
                'n_samples': stats['n_samples'],
                'stability': stats['stability'],
                'resilience': stats['resilience'],
                'reliability': stats['reliability'],
                'overall_robustness': stats['overall_robustness'],
                'robustness_gap': self.results['robustness_gap']
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def save_results(self, filepath):
        """Save results to CSV, creating the directory if needed."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = self.results_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\n✓ Robustness results saved to {filepath}")

    def print_summary(self):
        print("\n" + "="*80)
        print("ROBUSTNESS EVALUATION SUMMARY")
        print("="*80)
        for _, s in self.results['group_results'].items():
            print(f"\n{s['group']} (n={s['n_samples']}):")
            print(f"  Overall Robustness: {s['overall_robustness']:.4f}")
            print(f"  Stability:          {s['stability']:.4f}")
            print(f"  Resilience:         {s['resilience']:.4f}")
            print(f"  Reliability:        {s['reliability']:.4f}")
        print(f"\nRobustness Gap: {self.results['robustness_gap']:.4f}")
        print("="*80)

def robustness_pipeline_presplit(
    X_train: pd.DataFrame, X_test: pd.DataFrame,
    y_train: pd.Series, y_test: pd.Series,
    sensitive_attribute: str,
    output_csv: str = "results/robustness/robustness_results.csv",
    model_type: str = 'logistic'
) -> Tuple:
    """Complete Robustness evaluation pipeline."""
    evaluator = SimplifiedRobustnessEvaluator(model_type=model_type)
    evaluator.evaluate(X_train, X_test, y_train, y_test, sensitive_attribute)
    evaluator.print_summary()
    evaluator.save_results(output_csv)
    return evaluator, evaluator.results_to_dataframe()

    
    