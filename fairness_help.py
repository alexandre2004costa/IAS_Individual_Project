"""
Fairness Evaluation Pipeline
=============================
A comprehensive pipeline for evaluating fairness metrics on classification models.

Metrics Implemented:
- Disparate Impact Ratio (DIR)
- Statistical Parity (SP)
- True Positive Rate (TPR) per group
- False Positive Rate (FPR) per group
- Average Odds Difference (AOD)
- Equalized Odds (via TPR and FPR equality)

Author: Fairness Analysis Project
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class FairnessEvaluator:
    """
    A comprehensive fairness evaluation toolkit for binary classification models.
    
    Implements various fairness metrics including:
    - Equality of Outcome metrics (DIR, SP)
    - Equality of Opportunity metrics (TPR equality)
    - Equalized Odds metrics (TPR and FPR equality)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the FairnessEvaluator.
        
        Parameters:
        -----------
        random_state : int, default=42
            Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.results = {}
    
    @staticmethod
    def _to_numpy(array):
        """
        Safely convert pandas Series/DataFrame or numpy array to numpy array.
        
        Parameters:
        -----------
        array : pd.Series, pd.DataFrame, or np.ndarray
            Input array to convert
            
        Returns:
        --------
        np.ndarray : Converted numpy array
        """
        if hasattr(array, 'values'):
            return array.values
        return np.asarray(array)
        
    def calculate_group_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                group: np.ndarray, group_value) -> Dict:
        """
        Calculate performance metrics for a specific group.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        group : array-like
            Group membership array
        group_value : 
            Value identifying the group
            
        Returns:
        --------
        dict : Dictionary containing group metrics
        """
        # Filter data for this group
        mask = group == group_value
        y_true_group = y_true[mask]
        y_pred_group = y_pred[mask]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true_group, y_pred_group, 
                                         labels=[0, 1]).ravel()
        
        # Calculate metrics
        metrics = {
            'group': group_value,
            'n_samples': len(y_true_group),
            'positive_predictions': np.sum(y_pred_group == 1),
            'negative_predictions': np.sum(y_pred_group == 0),
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn,
            'success_rate': np.mean(y_pred_group == 1),  # P(C=1|G=g)
            'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,  # True Positive Rate
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,  # False Positive Rate
            'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,  # True Negative Rate
            'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0,  # False Negative Rate
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0
        }
        
        return metrics
    
    def calculate_fairness_metrics(self, group_metrics: List[Dict], 
                                   reference_group: Optional[str] = None) -> Dict:
        """
        Calculate all fairness metrics between groups.
        
        Metrics calculated:
        - ACC: Accuracy ratio (acc_g1 / acc_g2)
        - PE: Predictive Equality (FPR_g1 - FPR_g2)
        - EOR_FNR: Equal Opportunity Ratio based on FNR (FNR_g1 / FNR_g2)
        - SP: Statistical Parity (SR_g1 - SR_g2)
        - DI: Disparate Impact (SR_g1 / SR_g2)
        - EOD: Equalized Odds Difference (|TPR_g1 - TPR_g2| + |FPR_g1 - FPR_g2|) / 2
        - EOR_TPR: Equal Opportunity Ratio based on TPR (TPR_g1 / TPR_g2)
        - AOD: Average Odds Difference (|FPR_g1 - FPR_g2| + |TPR_g1 - TPR_g2|) / 2
        
        Parameters:
        -----------
        group_metrics : list of dict
            Metrics for each group
        reference_group : str, optional
            Reference group for comparison (if None, uses first group)
            
        Returns:
        --------
        dict : All fairness metrics for each group comparison
        """
        if reference_group is None:
            reference_group = group_metrics[0]['group']
        
        ref_group = next((g for g in group_metrics if g['group'] == reference_group), None)
        
        if ref_group is None:
            return {'error': 'Reference group not found'}
        
        results = {}
        for group in group_metrics:
            if group['group'] != reference_group:
                comparison_key = f"{group['group']}_vs_{reference_group}"
                
                # ACC: Accuracy ratio
                acc_ratio = group['accuracy'] / ref_group['accuracy'] if ref_group['accuracy'] > 0 else np.inf
                
                # PE: Predictive Equality (FPR difference)
                pe = group['fpr'] - ref_group['fpr']
                
                # EOR_FNR: Equal Opportunity Ratio based on FNR
                eor_fnr = group['fnr'] / ref_group['fnr'] if ref_group['fnr'] > 0 else np.inf
                
                # SP: Statistical Parity (selection rate difference)
                sp = group['success_rate'] - ref_group['success_rate']
                
                # DI: Disparate Impact (selection rate ratio)
                di = group['success_rate'] / ref_group['success_rate'] if ref_group['success_rate'] > 0 else np.inf
                
                # EOD: Equalized Odds Difference
                eod = (abs(group['tpr'] - ref_group['tpr']) + abs(group['fpr'] - ref_group['fpr'])) / 2
                
                # EOR_TPR: Equal Opportunity Ratio based on TPR
                eor_tpr = group['tpr'] / ref_group['tpr'] if ref_group['tpr'] > 0 else np.inf
                
                # AOD: Average Odds Difference (signed version)
                aod = ((group['fpr'] - ref_group['fpr']) + (group['tpr'] - ref_group['tpr'])) / 2
                
                results[comparison_key] = {
                    'ACC': acc_ratio,
                    'PE': pe,
                    'EOR_FNR': eor_fnr,
                    'SP': sp,
                    'DI': di,
                    'EOD': eod,
                    'EOR_TPR': eor_tpr,
                    'AOD': aod
                }
        
        return results
    
    def train_and_evaluate_presplit(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                                   y_train: pd.Series, y_test: pd.Series,
                                   sensitive_attribute: str,
                                   model_params: Optional[Dict] = None) -> Dict:
        """
        Train a logistic regression model and evaluate fairness metrics on pre-split data.
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix (including sensitive attribute)
        X_test : pd.DataFrame
            Test feature matrix (including sensitive attribute)
        y_train : pd.Series
            Training target variable
        y_test : pd.Series
            Test target variable
        sensitive_attribute : str
            Column name of the sensitive attribute
        model_params : dict, optional
            Parameters for LogisticRegression
            
        Returns:
        --------
        dict : Comprehensive results including model performance and fairness metrics
        """
        from sklearn.preprocessing import LabelEncoder
        
        # Separate sensitive attribute
        if sensitive_attribute not in X_train.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in X_train")
        if sensitive_attribute not in X_test.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in X_test")
        
        s_train = X_train[sensitive_attribute].copy()
        s_test = X_test[sensitive_attribute].copy()
        X_train_features = X_train.drop(columns=[sensitive_attribute]).copy()
        X_test_features = X_test.drop(columns=[sensitive_attribute]).copy()
        
        # Encode categorical variables in features
        self.label_encoders_ = {}
        for col in X_train_features.columns:
            if X_train_features[col].dtype == 'object' or X_train_features[col].dtype.name == 'category':
                le = LabelEncoder()
                X_train_features[col] = le.fit_transform(X_train_features[col].astype(str))
                X_test_features[col] = le.transform(X_test_features[col].astype(str))
                self.label_encoders_[col] = le
        
        # Train model
        if model_params is None:
            model_params = {'max_iter': 1000, 'random_state': self.random_state}
        
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train_features, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train_features)
        y_pred_test = self.model.predict(X_test_features)
        
        # Overall performance
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate metrics for each group
        # Convert to numpy arrays for compatibility
        y_test_array = self._to_numpy(y_test)
        s_test_array = self._to_numpy(s_test)
        unique_groups = sorted(np.unique(s_test_array))
        group_metrics = []
        
        for group_value in unique_groups:
            metrics = self.calculate_group_metrics(
                y_test_array, y_pred_test, s_test_array, group_value
            )
            group_metrics.append(metrics)
        
        # Calculate all fairness metrics
        reference_group = unique_groups[0]
        fairness_results = self.calculate_fairness_metrics(group_metrics, reference_group)
        
        # Compile results
        results = {
            'model_type': 'LogisticRegression',
            'model_params': model_params,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(X_train_features),
            'n_test_samples': len(X_test_features),
            'sensitive_attribute': sensitive_attribute,
            'reference_group': reference_group,
            'unique_groups': unique_groups,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_results
        }
        
        self.results = results
        return results
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, 
                          sensitive_attribute: str,
                          test_size: float = 0.3,
                          model_params: Optional[Dict] = None) -> Dict:
        """
        Train a logistic regression model and evaluate fairness metrics.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Feature matrix (including sensitive attribute)
        y : pd.Series
            Target variable
        sensitive_attribute : str
            Column name of the sensitive attribute
        test_size : float, default=0.3
            Proportion of dataset to use as test set
        model_params : dict, optional
            Parameters for LogisticRegression
            
        Returns:
        --------
        dict : Comprehensive results including model performance and fairness metrics
        """
        from sklearn.preprocessing import LabelEncoder
        
        # Separate sensitive attribute
        if sensitive_attribute not in X.columns:
            raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in features")
        
        sensitive_col = X[sensitive_attribute].copy()
        X_features = X.drop(columns=[sensitive_attribute]).copy()
        
        # Encode categorical variables in features
        self.label_encoders_ = {}
        for col in X_features.columns:
            if X_features[col].dtype == 'object' or X_features[col].dtype.name == 'category':
                le = LabelEncoder()
                X_features[col] = le.fit_transform(X_features[col].astype(str))
                self.label_encoders_[col] = le
        
        # Train-test split
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X_features, y, sensitive_col, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Train model
        if model_params is None:
            model_params = {'max_iter': 1000, 'random_state': self.random_state}
        
        self.model = LogisticRegression(**model_params)
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Overall performance
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        # Calculate metrics for each group
        # Convert to numpy arrays for compatibility
        y_test_array = self._to_numpy(y_test)
        s_test_array = self._to_numpy(s_test)
        unique_groups = sorted(np.unique(s_test_array))
        group_metrics = []
        
        for group_value in unique_groups:
            metrics = self.calculate_group_metrics(
                y_test_array, y_pred_test, s_test_array, group_value
            )
            group_metrics.append(metrics)
        
        # Calculate all fairness metrics
        reference_group = unique_groups[0]
        fairness_results = self.calculate_fairness_metrics(group_metrics, reference_group)
        
        # Compile results
        results = {
            'model_type': 'LogisticRegression',
            'model_params': model_params,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'sensitive_attribute': sensitive_attribute,
            'reference_group': reference_group,
            'unique_groups': unique_groups,
            'group_metrics': group_metrics,
            'fairness_metrics': fairness_results
        }
        
        self.results = results
        return results
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for easy analysis and export.
        
        Returns:
        --------
        pd.DataFrame : Results in tabular format
        """
        if not self.results:
            raise ValueError("No results available. Run train_and_evaluate first.")
        
        rows = []
        
        # Overall metrics
        base_row = {
            'model_type': self.results['model_type'],
            'train_accuracy': self.results['train_accuracy'],
            'test_accuracy': self.results['test_accuracy'],
            'sensitive_attribute': self.results['sensitive_attribute'],
            'reference_group': self.results['reference_group']
        }
        
        # Add group-specific metrics
        for group_metric in self.results['group_metrics']:
            row = base_row.copy()
            row.update({
                'group': group_metric['group'],
                'n_samples': group_metric['n_samples'],
                'tpr': group_metric['tpr'],
                'fpr': group_metric['fpr'],
                'fnr': group_metric['fnr'],
                'accuracy': group_metric['accuracy']
            })
            
            # Add fairness metrics (only for non-reference groups)
            if group_metric['group'] != self.results['reference_group']:
                comparison_key = f"{group_metric['group']}_vs_{self.results['reference_group']}"
                fairness = self.results['fairness_metrics'].get(comparison_key, {})
                row['ACC'] = fairness.get('ACC', np.nan)
                row['PE'] = fairness.get('PE', np.nan)
                row['EOR_FNR'] = fairness.get('EOR_FNR', np.nan)
                row['SP'] = fairness.get('SP', np.nan)
                row['DI'] = fairness.get('DI', np.nan)
                row['EOD'] = fairness.get('EOD', np.nan)
                row['EOR_TPR'] = fairness.get('EOR_TPR', np.nan)
                row['AOD'] = fairness.get('AOD', np.nan)
            else:
                # Reference group has baseline values
                row['ACC'] = 1.0
                row['PE'] = 0.0
                row['EOR_FNR'] = 1.0
                row['SP'] = 0.0
                row['DI'] = 1.0
                row['EOD'] = 0.0
                row['EOR_TPR'] = 1.0
                row['AOD'] = 0.0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns
        column_order = [
            'model_type', 'train_accuracy', 'test_accuracy', 'sensitive_attribute',
            'reference_group', 'group', 'n_samples',
            'tpr', 'fpr', 'fnr', 'accuracy',
            'ACC', 'PE', 'EOR_FNR', 'SP', 'DI', 'EOD', 'EOR_TPR', 'AOD'
        ]
        
        return df[column_order]
    
    def save_results(self, filepath: str):
        """
        Save results to CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the CSV file
        """
        df = self.results_to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
    
    def print_summary(self):
        """Print a summary of the fairness evaluation."""
        if not self.results:
            print("No results available. Run train_and_evaluate first.")
            return
        
        print("=" * 80)
        print("FAIRNESS EVALUATION SUMMARY")
        print("=" * 80)
        print(f"\nModel Type: {self.results['model_type']}")
        print(f"Train Accuracy: {self.results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print(f"Sensitive Attribute: {self.results['sensitive_attribute']}")
        print(f"Reference Group: {self.results['reference_group']}")
        
        print("\n" + "-" * 80)
        print("GROUP-SPECIFIC METRICS")
        print("-" * 80)
        for group in self.results['group_metrics']:
            print(f"\nGroup: {group['group']}")
            print(f"  Samples: {group['n_samples']}")
            print(f"  TPR: {group['tpr']:.4f}")
            print(f"  FPR: {group['fpr']:.4f}")
            print(f"  FNR: {group['fnr']:.4f}")
            print(f"  Accuracy: {group['accuracy']:.4f}")
        
        print("\n" + "-" * 80)
        print("FAIRNESS METRICS (vs Reference Group)")
        print("-" * 80)
        
        for comparison_key, metrics in self.results['fairness_metrics'].items():
            print(f"\n{comparison_key}:")
            print(f"  ACC (Accuracy Ratio):            {metrics['ACC']:.4f}  [Ideal: 1.0]")
            print(f"  PE (Predictive Equality):        {metrics['PE']:.4f}  [Ideal: 0.0]")
            print(f"  EOR_FNR (EO Ratio - FNR):        {metrics['EOR_FNR']:.4f}  [Ideal: 1.0]")
            print(f"  SP (Statistical Parity):         {metrics['SP']:.4f}  [Ideal: 0.0]")
            print(f"  DI (Disparate Impact):           {metrics['DI']:.4f}  [Ideal: 1.0]")
            print(f"  EOD (Equalized Odds Diff):       {metrics['EOD']:.4f}  [Ideal: 0.0]")
            print(f"  EOR_TPR (EO Ratio - TPR):        {metrics['EOR_TPR']:.4f}  [Ideal: 1.0]")
            print(f"  AOD (Average Odds Diff):         {metrics['AOD']:.4f}  [Ideal: 0.0]")
        
        print("\n" + "=" * 80)


def fairness_pipeline_presplit(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_attribute: str,
    output_csv: str = "fairness_results.csv",
    model_params: Optional[Dict] = None,
    random_state: int = 42
) -> Tuple[FairnessEvaluator, pd.DataFrame]:
    """
    Complete pipeline for fairness evaluation with pre-split data.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features (including sensitive attribute)
    X_test : pd.DataFrame
        Test features (including sensitive attribute)
    y_train : pd.Series
        Training target variable
    y_test : pd.Series
        Test target variable
    sensitive_attribute : str
        Name of the sensitive attribute column (e.g., 'race', 'gender')
    output_csv : str, default='fairness_results.csv'
        Path to save results CSV
    model_params : dict, optional
        Parameters for LogisticRegression model
    random_state : int, default=42
        Random state for reproducibility
        
    Returns:
    --------
    tuple : (FairnessEvaluator object, Results DataFrame)
    
    Example:
    --------
    >>> evaluator, results_df = fairness_pipeline_presplit(
    ...     X_train=X_train,
    ...     X_test=X_test,
    ...     y_train=y_train,
    ...     y_test=y_test,
    ...     sensitive_attribute='gender',
    ...     output_csv='my_fairness_results.csv'
    ... )
    """
    
    # Validate inputs
    if sensitive_attribute not in X_train.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in X_train")
    if sensitive_attribute not in X_test.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in X_test")
    
    # Initialize evaluator
    evaluator = FairnessEvaluator(random_state=random_state)
    
    # Train and evaluate
    print("Training model and evaluating fairness metrics...")
    evaluator.train_and_evaluate_presplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        sensitive_attribute=sensitive_attribute,
        model_params=model_params
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(output_csv)
    
    # Get results dataframe
    results_df = evaluator.results_to_dataframe()
    
    return evaluator, results_df


def fairness_pipeline(data: pd.DataFrame, 
                     target_column: str,
                     sensitive_attribute: str,
                     feature_columns: Optional[List[str]] = None,
                     output_csv: str = "fairness_results.csv",
                     test_size: float = 0.3,
                     model_params: Optional[Dict] = None,
                     random_state: int = 42) -> Tuple[FairnessEvaluator, pd.DataFrame]:
    # Validate inputs
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    if sensitive_attribute not in data.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in data")
    
    # Prepare features
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
    else:
        if sensitive_attribute not in feature_columns:
            feature_columns = feature_columns + [sensitive_attribute]
    
    X = data[feature_columns]
    y = data[target_column]
    
    # Initialize evaluator
    evaluator = FairnessEvaluator(random_state=random_state)
    
    # Train and evaluate
    print("Training model and evaluating fairness metrics...")
    evaluator.train_and_evaluate(
        X=X, 
        y=y, 
        sensitive_attribute=sensitive_attribute,
        test_size=test_size,
        model_params=model_params
    )
    
    # Print summary
    evaluator.print_summary()
    
    # Save results
    evaluator.save_results(output_csv)
    
    # Get results dataframe
    results_df = evaluator.results_to_dataframe()
    
    return evaluator, results_df


def calculate_overall_fairness_score(results_row):
    """
    Calculate overall fairness score (lower is better).
    
    Ideal values:
    - ACC, DI, EOR_FNR, EOR_TPR = 1.0
    - PE, SP, EOD, AOD = 0.0
    """
    # Absolute deviations from ideal
    deviations = [
        abs(results_row['ACC'] - 1.0)**2,
        abs(results_row['PE'] - 0.0)**2,
        abs(results_row['EOR_FNR'] - 1.0)**2,
        abs(results_row['SP'] - 0.0)**2,
        abs(results_row['DI'] - 1.0)**2,
        abs(results_row['EOD'] - 0.0)**2,
        abs(results_row['EOR_TPR'] - 1.0)**2,
        abs(results_row['AOD'] - 0.0)**2
    ]
    
    # Mean absolute deviation
    overall_fairness = np.mean(deviations)
    return overall_fairness