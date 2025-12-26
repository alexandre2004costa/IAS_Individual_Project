"""
Simplified DoX-Inspired Explainability Pipeline
================================================

A practical explainability metric inspired by DoX (Sovrano & Vitali, 2023)
that measures if explanations can answer key questions:
- WHY: What features drove the prediction?
- HOW: What's their relative importance?
- WHAT: How complete is the explanation?

This simplified approach captures DoX's philosophical foundation
without the computational complexity of full NLP processing.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import spearmanr
from typing import Dict, List, Tuple, Optional
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
        """
        Initialize evaluator.
        
        Parameters:
        -----------
        model_type : str
            'logistic' or 'random_forest'
        random_state : int
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.results = {}
    
    def train_model(self, X_train, y_train):
        """Train model and create SHAP explainer."""
        if self.model_type == 'logistic':
            self.model = LogisticRegression(max_iter=1000, random_state=self.random_state)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Create SHAP explainer
        if self.model_type == 'logistic':
            self.explainer = shap.LinearExplainer(self.model, X_train)
        else:
            self.explainer = shap.TreeExplainer(self.model)
    
    def calculate_shap_values(self, X):
        """Calculate SHAP values for dataset."""
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
        
        return shap_values
    
    def calculate_clarity_score(self, shap_values, top_k=3):
        """
        Q1: WHY - Clarity Score
        
        Can we clearly identify the most important features?
        
        Measures:
        - Are there distinct top features?
        - Is their importance above threshold?
        
        Returns:
        --------
        float : Score 0-1 (higher = clearer important features)
        """
        abs_shap = np.abs(shap_values)
        
        # Get top k features
        top_indices = np.argsort(abs_shap)[-top_k:]
        top_values = abs_shap[top_indices]
        
        # Score 1: Are top features significant? (above mean)
        mean_importance = abs_shap.mean()
        significant_count = np.sum(top_values > mean_importance)
        significance_score = significant_count / top_k
        
        # Score 2: Are top features well-separated from others?
        if len(abs_shap) > top_k:
            next_best = np.sort(abs_shap)[-(top_k+1)]
            min_top = top_values.min()
            separation = (min_top - next_best) / (min_top + 1e-10)
            separation_score = min(separation * 2, 1.0)  # Scale to 0-1
        else:
            separation_score = 1.0
        
        # Combined clarity score
        clarity = 0.7 * significance_score + 0.3 * separation_score
        
        return clarity
    
    def calculate_distinctiveness_score(self, shap_values):
        """
        Q2: HOW - Distinctiveness Score
        
        Is the relative importance of features clear?
        
        Measures:
        - Range of SHAP values (spread)
        - Coefficient of variation
        
        Returns:
        --------
        float : Score 0-1 (higher = clearer relative importance)
        """
        abs_shap = np.abs(shap_values)
        
        # Score 1: Range of importance values
        if abs_shap.max() > 0:
            importance_range = abs_shap.max() - abs_shap.min()
            range_score = min(importance_range / 0.5, 1.0)  # Normalize
        else:
            range_score = 0.0
        
        # Score 2: Coefficient of variation (relative spread)
        if abs_shap.mean() > 0:
            cv = abs_shap.std() / abs_shap.mean()
            cv_score = min(cv, 1.0)  # Cap at 1.0
        else:
            cv_score = 0.0
        
        # Combined distinctiveness score
        distinctiveness = 0.6 * range_score + 0.4 * cv_score
        
        return distinctiveness
    
    def calculate_coverage_score(self, shap_values, top_k=3, target_coverage=0.7):
        """
        Q3: WHAT - Coverage Score
        
        Do the top features explain enough of the prediction?
        
        Measures:
        - Proportion of total importance in top k features
        
        Returns:
        --------
        float : Score 0-1 (higher = better coverage)
        """
        abs_shap = np.abs(shap_values)
        total_importance = abs_shap.sum()
        
        if total_importance == 0:
            return 0.0
        
        # Get top k features
        top_indices = np.argsort(abs_shap)[-top_k:]
        top_importance = abs_shap[top_indices].sum()
        
        # Coverage ratio
        coverage_ratio = top_importance / total_importance
        
        # Score based on target coverage
        coverage_score = min(coverage_ratio / target_coverage, 1.0)
        
        return coverage_score
    
    def calculate_dox_inspired_score(self, shap_values, top_k=3):
        """
        Calculate overall DoX-inspired score for single instance.
        
        Combines three aspects:
        - Clarity (WHY)
        - Distinctiveness (HOW)
        - Coverage (WHAT)
        
        Parameters:
        -----------
        shap_values : np.ndarray
            SHAP values for single instance
        top_k : int
            Number of top features to consider
            
        Returns:
        --------
        dict : Individual scores and overall DoX
        """
        clarity = self.calculate_clarity_score(shap_values, top_k)
        distinctiveness = self.calculate_distinctiveness_score(shap_values)
        coverage = self.calculate_coverage_score(shap_values, top_k)
        
        # Weighted average (inspired by DoX importance)
        weights = {
            'clarity': 0.4,      # WHY is most important
            'distinctiveness': 0.35,  # HOW is very important
            'coverage': 0.25     # WHAT is supporting
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
        """
        Evaluate DoX-inspired scores for a group.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features (without sensitive attribute)
        y : pd.Series or np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        group_name : str
            Name of group
        top_k : int
            Number of top features
            
        Returns:
        --------
        dict : Group statistics
        """
        print(f"  Evaluating {group_name} ({len(X)} samples)...")
        
        # Calculate SHAP values
        shap_values = self.calculate_shap_values(X)
        
        # Calculate DoX for each instance
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
        
        # Calculate statistics
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
        """
        Complete DoX-inspired evaluation.
        
        Parameters:
        -----------
        X_train, X_test : pd.DataFrame
            Features (including sensitive attribute)
        y_train, y_test : pd.Series or np.ndarray
            Labels
        sensitive_attr : str
            Name of sensitive attribute
        top_k : int
            Number of top features
            
        Returns:
        --------
        dict : Evaluation results
        """
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
        """Convert results to DataFrame."""
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
        """Save results to CSV."""
        df = self.results_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        """Print summary of evaluation."""
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
    """
    Complete DoX-inspired evaluation pipeline.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Features including sensitive attribute
    y_train, y_test : pd.Series or np.ndarray
        Labels
    sensitive_attribute : str
        Name of sensitive attribute column
    output_csv : str
        Path to save results
    model_type : str
        'logistic' or 'random_forest'
    top_k : int
        Number of top features to consider
    random_state : int
        
    Returns:
    --------
    tuple : (evaluator, results_df)
    """
    # Validate inputs
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


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SIMPLIFIED DOX-INSPIRED EVALUATION - EXAMPLE")
    print("="*80)
    
    # Example: Load BiasOnDemand dataset
    print("\nLoading baseline dataset...")
    try:
        X_train = pd.read_csv('datasets/baseline_/X_train.csv', index_col=0)
        X_test = pd.read_csv('datasets/baseline_/X_test.csv', index_col=0)
        y_train = pd.read_csv('datasets/baseline_/y_train.csv', index_col=0).squeeze()
        y_test = pd.read_csv('datasets/baseline_/y_test.csv', index_col=0).squeeze()
        
        print("✓ Dataset loaded")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Run evaluation
        print("\nRunning DoX-inspired evaluation...")
        evaluator, results = dox_inspired_pipeline_presplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            sensitive_attribute='A',
            output_csv='dox_inspired_baseline.csv',
            top_k=3
        )
        
        print("\n✓ Evaluation complete!")
        print("\nResults preview:")
        print(results)
        
    except FileNotFoundError:
        print("\n⚠️  Dataset not found!")
        print("Generate datasets first using: python generate_all_datasets.py")