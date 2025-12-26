from metrics_calculators.robustness import *

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

for i in detailed_levels:
    for d in datasets:
        dataset = 'datasets/' + d + i
        X_train = pd.read_csv(f'{dataset}/X_train.csv', index_col=0)
        X_test = pd.read_csv(f'{dataset}/X_test.csv', index_col=0)
        y_train = pd.read_csv(f'{dataset}/y_train.csv', index_col=0).squeeze()
        y_test = pd.read_csv(f'{dataset}/y_test.csv', index_col=0).squeeze()
        
        robustness_pipeline_presplit(X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            sensitive_attribute='A',
            model_type='random_forest',
            output_csv="results/robustness/" + d + i
        )