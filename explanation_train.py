import pandas as pd
import numpy as np
from pathlib import Path
from metrics_calculators.dox import dox_inspired_pipeline_presplit
import time


def evaluate_all_datasets_simplified_dox(dataset_paths, base_dir='datasets', output_dir='dox_inspired_results', top_k=3):

    Path(output_dir).mkdir(exist_ok=True)
    all_results = []
    times = []
    
    print("="*80)
    print(f"BATCH SIMPLIFIED DOX EVALUATION - {len(dataset_paths)} DATASETS")
    print("="*80)
    print(f"\nTop-k features: {top_k}")
    print(f"Estimated time: {len(dataset_paths) * 0.5}-{len(dataset_paths) * 2} minutes")
    print("\n✓ This is much faster than full DoX!\n")
    
    for i, dataset_name in enumerate(dataset_paths, 1):
        dataset_path = Path(base_dir) / dataset_name
        if not dataset_path.exists():
            print(f"\n[{i}/{len(dataset_paths)}] ⚠️  Skipping {dataset_name} - not found")
            continue
        
        print(f"\n{'='*80}")
        print(f"[{i}/{len(dataset_paths)}] EVALUATING: {dataset_name}")
        print('='*80)
        
        start_time = time.time()
        
        try:
            # Load data
            X_train = pd.read_csv(dataset_path / 'X_train.csv', index_col=0)
            X_test = pd.read_csv(dataset_path / 'X_test.csv', index_col=0)
            y_train = pd.read_csv(dataset_path / 'y_train.csv', index_col=0).squeeze()
            y_test = pd.read_csv(dataset_path / 'y_test.csv', index_col=0).squeeze()
            
            # Run evaluation
            output_csv = Path(output_dir) / f'{dataset_name}.csv'
            evaluator, results = dox_inspired_pipeline_presplit(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                sensitive_attribute='A',
                output_csv=str(output_csv),
                top_k=top_k
            )
            
            # Add dataset name and timing
            results['dataset_name'] = dataset_name
            elapsed = time.time() - start_time
            results['evaluation_time_seconds'] = elapsed
            times.append(elapsed)
            
            all_results.append(results)
            
            print(f"\n✓ Completed: {dataset_name} ({elapsed:.1f}s)")
            
            # Estimated remaining time
            if len(times) > 0:
                avg_time = np.mean(times)
                remaining = (len(dataset_paths) - i) * avg_time
                print(f"  Estimated remaining time: {remaining/60:.1f} minutes")
            
        except Exception as e:
            print(f"\n✗ Error with {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_results:
        print("\n" + "="*80)
        print("COMBINING RESULTS")
        print("="*80)
        
        combined = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['dataset_name'] + [col for col in combined.columns if col != 'dataset_name']
        combined = combined[cols]
        
        # Save combined results
        combined_path = Path(output_dir) / 'combined_dox_inspired_results.csv'
        combined.to_csv(combined_path, index=False)
        
        print(f"\n✓ Combined results saved to: {combined_path}")
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        print("\nDoX Scores by Dataset:")
        print(f"{'Dataset':<50} {'Mean DoX':>10} {'DoX Gap':>10}")
        print("-"*80)
        for dataset in combined['dataset_name'].unique():
            subset = combined[combined['dataset_name'] == dataset]
            mean_dox = subset['mean_dox'].mean()
            dox_gap = subset['dox_gap'].iloc[0]
            print(f"{dataset:<50} {mean_dox:>10.4f} {dox_gap:>10.4f}")
        
        print(f"\nTotal evaluation time: {sum(times)/60:.1f} minutes")
        print(f"Average time per dataset: {np.mean(times):.1f} seconds")
        
        return combined
    
    return None


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    detailed_levels = ['0.00', '0.10', '0.20', '0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90', '1.00']
    datasets = [
        'hist_bias_Y_ly_',
        'hist_bias_Q_lhq_',
        'hist_bias_R_lhr_',
        'interaction_bias_lyb_',
        'meas_bias_Y_linear_lmy_',
        'meas_bias_Y_nonlinear_lmy_',
        'meas_bias_R_lm_',
        'undersample_pu_',
        'representation_bias_lr_true',
        'label_noise_sy_',
    ]
    
    all_datasets = []
    for i in detailed_levels:
        for d in datasets:
            if (d == 'undersample_pu_' or d == 'representation_bias_lr_true') and i == '1.00':
                continue
            all_datasets.append(f'{d}{i}')

    results = evaluate_all_datasets_simplified_dox(
        all_datasets,
        base_dir='datasets',
        output_dir='results/explanation/',
        top_k=1  # Consider top feature only
    )
    