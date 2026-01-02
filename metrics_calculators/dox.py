import numpy as np
import pandas as pd
import shap
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')


class SimplifiedDoXEvaluator:
    """
    Evaluates explanation quality using simplified DoX-inspired metrics.
    
    Measures three key aspects:
    1. Clarity: Can we identify important features? (WHY)
    2. Distinctiveness: Is relative importance clear? (HOW)
    3. Coverage: Do top features explain enough? (WHAT)
    """
    
    def __init__(self, model_type='logistic', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.results = {}
    
    def train_model(self, X_train, y_train):
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        self.model.fit(X_train, y_train)
        
        if self.model_type == 'logistic':
            self.explainer = shap.LinearExplainer(self.model, X_train)
        else:
            self.explainer = shap.TreeExplainer(self.model)
    
    def calculate_shap_values(self, X):
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
    def calculate_clarity_score(self, shap_values, top_k=3):
        abs_shap = np.abs(shap_values)
        
        if abs_shap.sum() == 0:
            return 0.0
        
        # Normalize to probabilities
        shap_probs = abs_shap / abs_shap.sum()
        
        # Get top k features
        top_indices = np.argsort(abs_shap)[-top_k:]
        nonzero_probs = shap_probs[shap_probs > 0]
        entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs + 1e-10))
        max_entropy = np.log2(len(abs_shap))  # Maximum possible entropy
        concentration_score = 1.0 - (entropy / max_entropy) 
        
        top_dominance = shap_probs[top_indices].sum()
        
        # Relative gap between top-k and rest
        if len(abs_shap) > top_k:
            remaining_indices = np.setdiff1d(np.arange(len(abs_shap)), top_indices)
            top_mean = abs_shap[top_indices].mean()
            remaining_mean = abs_shap[remaining_indices].mean()
            
            if top_mean > 0:
                gap_ratio = (top_mean - remaining_mean) / top_mean
                gap_score = min(max(gap_ratio, 0), 1.0)  # Clip to [0,1]
            else:
                gap_score = 0.0
        else:
            gap_score = 1.0
        
        # Combined clarity score
        clarity = (0.3 * concentration_score + 
                   0.4 * top_dominance + 
                   0.3 * gap_score)
        
        return clarity
    
    def calculate_distinctiveness_score(self, shap_values):
        abs_shap = np.abs(shap_values)
        
        # Range of importance values
        if abs_shap.max() > 0:
            importance_range = abs_shap.max() - abs_shap.min()
            range_score = min(importance_range / 0.5, 1.0)  # Normalize
        else:
            range_score = 0.0
        
        # Coefficient of variation (relative spread)
        if abs_shap.mean() > 0:
            cv = abs_shap.std() / abs_shap.mean()
            cv_score = min(cv, 1.0)  # Cap at 1.0
        else:
            cv_score = 0.0
        
        distinctiveness = 0.6 * range_score + 0.4 * cv_score
        
        return distinctiveness
    
    def calculate_coverage_score(self, shap_values, top_k=3, target_coverage=0.8):
        abs_shap = np.abs(shap_values)
        total_importance = abs_shap.sum()
        
        if total_importance == 0:
            return 0.0
        
        # Get top k features
        top_indices = np.argsort(abs_shap)[-top_k:]
        top_importance = abs_shap[top_indices].sum()
        
        # Coverage ratio
        coverage_ratio = top_importance / total_importance
        
        # Use sigmoid-like scaling for smoother variation
        # This makes the score more sensitive to changes around 0.5-0.9
        if coverage_ratio < target_coverage:
            # Linear scaling below target
            coverage_score = coverage_ratio / target_coverage
        else:
            # Slower growth above target
            excess = coverage_ratio - target_coverage
            coverage_score = 1.0 - (0.2 * np.exp(-5 * excess))  # Asymptotic to 1.0
        
        return min(coverage_score, 1.0)
    
    def calculate_dox_inspired_score(self, shap_values, top_k=3):
        clarity = self.calculate_clarity_score(shap_values, top_k)
        distinctiveness = self.calculate_distinctiveness_score(shap_values)
        coverage = self.calculate_coverage_score(shap_values, top_k)
        
        # Weighted average (inspired by DoX importance)
        weights = {
            'clarity': 0.4,      
            'distinctiveness': 0.35, 
            'coverage': 0.25    
        }
        
        overall_dox = (
            weights['clarity'] * clarity +
            weights['distinctiveness'] * distinctiveness +
            weights['coverage'] * coverage
        )
        
        return {
            'clarity': clarity,
            'distinctiveness': distinctiveness,
            'coverage': coverage,
            'overall_dox': overall_dox
        }
    
    def evaluate_group(self, X, y, y_pred, group_name, top_k=3):
        print(f"  Evaluating {group_name} ({len(X)} samples)...")
        shap_values = self.calculate_shap_values(X)
        
        dox_scores = []
        clarity_scores = []
        distinctiveness_scores = []
        coverage_scores = []
        
        for i in range(len(X)):
            scores = self.calculate_dox_inspired_score(shap_values[i], top_k)
            dox_scores.append(scores['overall_dox'])
            clarity_scores.append(scores['clarity'])
            distinctiveness_scores.append(scores['distinctiveness'])
            coverage_scores.append(scores['coverage'])

        return {
            'group': group_name,
            'n_samples': len(X),
            'mean_dox': np.mean(dox_scores),
            'std_dox': np.std(dox_scores),
            'median_dox': np.median(dox_scores),
            'min_dox': np.min(dox_scores),
            'max_dox': np.max(dox_scores),
            'mean_clarity': np.mean(clarity_scores),
            'mean_distinctiveness': np.mean(distinctiveness_scores),
            'mean_coverage': np.mean(coverage_scores),
            'dox_scores': dox_scores,
            'clarity_scores': clarity_scores,
            'distinctiveness_scores': distinctiveness_scores,
            'coverage_scores': coverage_scores
        }
    
    def evaluate(self, X_train, X_test, y_train, y_test, sensitive_attr, top_k=3):
        print("="*80)
        print("SIMPLIFIED DOX-INSPIRED EVALUATION")
        print("="*80)
        
        # Remove sensitive attribute for training
        X_train_features = X_train.drop(columns=[sensitive_attr])
        X_test_features = X_test.drop(columns=[sensitive_attr])
        
        # Train model
        print("\n1. Training model...")
        self.train_model(X_train_features, y_train)
        
        # Get predictions
        y_pred_test = self.model.predict(X_test_features)
        accuracy = (y_pred_test == y_test).mean()
        print(f"   Model accuracy: {accuracy:.3f}")
        
        # Evaluate DoX for each group
        print("\n2. Evaluating DoX-inspired metrics by group...")
        
        group_results = {}
        for group_value in sorted(X_test[sensitive_attr].unique()):
            # Filter to group
            mask = X_test[sensitive_attr] == group_value
            X_group = X_test_features[mask]
            y_group = y_test[mask] if isinstance(y_test, pd.Series) else y_test[mask]
            y_pred_group = y_pred_test[mask]
            
            # Evaluate DoX
            group_stats = self.evaluate_group(
                X_group, y_group, y_pred_group,
                group_name=f"Group_{group_value}",
                top_k=top_k
            )
            
            group_results[f"group_{group_value}"] = group_stats
        
        # Calculate gaps between groups
        print("\n3. Calculating gaps between groups...")
        groups = list(group_results.keys())
        if len(groups) >= 2:
            dox_gap = abs(
                group_results[groups[0]]['mean_dox'] - 
                group_results[groups[1]]['mean_dox']
            )
            clarity_gap = abs(
                group_results[groups[0]]['mean_clarity'] - 
                group_results[groups[1]]['mean_clarity']
            )
            distinctiveness_gap = abs(
                group_results[groups[0]]['mean_distinctiveness'] - 
                group_results[groups[1]]['mean_distinctiveness']
            )
            coverage_gap = abs(
                group_results[groups[0]]['mean_coverage'] - 
                group_results[groups[1]]['mean_coverage']
            )
            
            print(f"   DoX Gap: {dox_gap:.4f}")
            print(f"   Clarity Gap: {clarity_gap:.4f}")
            print(f"   Distinctiveness Gap: {distinctiveness_gap:.4f}")
            print(f"   Coverage Gap: {coverage_gap:.4f}")
        else:
            dox_gap = clarity_gap = distinctiveness_gap = coverage_gap = 0.0
        
        # Compile results
        results = {
            'model_type': self.model_type,
            'model_accuracy': accuracy,
            'sensitive_attribute': sensitive_attr,
            'n_features': len(X_test_features.columns),
            'top_k': top_k,
            'group_results': group_results,
            'dox_gap': dox_gap,
            'clarity_gap': clarity_gap,
            'distinctiveness_gap': distinctiveness_gap,
            'coverage_gap': coverage_gap
        }
        
        self.results = results
        return results
    
    def results_to_dataframe(self):
        if not self.results:
            raise ValueError("No results available. Run evaluate() first.")
        
        rows = []
        for group_key, stats in self.results['group_results'].items():
            row = {
                'sensitive_attribute': self.results['sensitive_attribute'],
                'group': stats['group'],
                'n_samples': stats['n_samples'],
                'mean_dox': stats['mean_dox'],
                'std_dox': stats['std_dox'],
                'median_dox': stats['median_dox'],
                'min_dox': stats['min_dox'],
                'max_dox': stats['max_dox'],
                'mean_clarity': stats['mean_clarity'],
                'mean_distinctiveness': stats['mean_distinctiveness'],
                'mean_coverage': stats['mean_coverage'],
                'dox_gap': self.results['dox_gap'],
                'clarity_gap': self.results['clarity_gap'],
                'distinctiveness_gap': self.results['distinctiveness_gap'],
                'coverage_gap': self.results['coverage_gap']
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def save_results(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = self.results_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        if not self.results:
            print("No results available.")
            return
        
        print("\n" + "="*80)
        print("SIMPLIFIED DOX-INSPIRED EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nModel: {self.results['model_type']}")
        print(f"Accuracy: {self.results['model_accuracy']:.3f}")
        print(f"Sensitive Attribute: {self.results['sensitive_attribute']}")
        print(f"Number of Features: {self.results['n_features']}")
        print(f"Top K Features: {self.results['top_k']}")
        
        print("\n" + "-"*80)
        print("GROUP-SPECIFIC SCORES")
        print("-"*80)
        
        for group_key, stats in self.results['group_results'].items():
            print(f"\n{stats['group']} (n={stats['n_samples']}):")
            print(f"  Overall DoX:      {stats['mean_dox']:.4f} ± {stats['std_dox']:.4f}")
            print(f"  Clarity (WHY):    {stats['mean_clarity']:.4f}")
            print(f"  Distinctiveness (HOW): {stats['mean_distinctiveness']:.4f}")
            print(f"  Coverage (WHAT):  {stats['mean_coverage']:.4f}")
        
        print("\n" + "-"*80)
        print("GAPS BETWEEN GROUPS")
        print("-"*80)
        print(f"Overall DoX Gap:      {self.results['dox_gap']:.4f}")
        print(f"Clarity Gap:          {self.results['clarity_gap']:.4f}")
        print(f"Distinctiveness Gap:  {self.results['distinctiveness_gap']:.4f}")
        print(f"Coverage Gap:         {self.results['coverage_gap']:.4f}")
        
        # Interpretation
        print("\n" + "-"*80)
        print("INTERPRETATION")
        print("-"*80)
        if self.results['dox_gap'] < 0.05:
            print("✓ Very similar explanation quality across groups")
        elif self.results['dox_gap'] < 0.10:
            print("⚠ Moderate difference in explanation quality")
        else:
            print("⚠️ Significant difference in explanation quality")
        
        print("\n" + "="*80)


def dox_inspired_pipeline_presplit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_attribute: str,
    output_csv: str = "dox_inspired_results.csv",
    model_type: str = 'logistic',
    top_k: int = 3,
    random_state: int = 42
) -> Tuple:

    if sensitive_attribute not in X_train.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in X_train")
    
    # Initialize evaluator
    evaluator = SimplifiedDoXEvaluator(model_type=model_type, random_state=random_state)
    
    # Run evaluation
    evaluator.evaluate(X_train, X_test, y_train, y_test, sensitive_attribute, top_k=top_k)
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(output_csv)
    
    # Get results dataframe
    results_df = evaluator.results_to_dataframe()
    
    return evaluator, results_df
