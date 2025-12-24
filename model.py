import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from fairness_help import *

def load_data(directory_path):
    try:
        X_train = pd.read_csv(f"{directory_path}/X_train.csv")
        X_test = pd.read_csv(f"{directory_path}/X_test.csv")
        y_train = pd.read_csv(f"{directory_path}/y_train.csv", header=None, names=['Y'], skiprows=1)                
        y_test = pd.read_csv(f"{directory_path}/y_test.csv", header=None, names=['Y'], skiprows=1)

        y_train['Y'] = pd.to_numeric(y_train['Y'].squeeze(), errors='coerce') 
        y_test['Y'] = pd.to_numeric(y_test['Y'].squeeze(), errors='coerce') 

        if len(X_train) != len(y_train) or len(X_test) != len(y_test):
             print(f"ERRO: Desalinhamento de tamanho. Treino: {len(X_train)} vs {len(y_train)}. Teste: {len(X_test)} vs {len(y_test)}.")
             return None, None, None, None # Retorna None se falhar

        return X_train, y_train, X_test, y_test
    
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos no caminho {directory_path}: {e}")
        return None, None, None, None
    

def fairness_run(sensitive_attribute, diffs = ['0.00','0.05', '0.10', '0.15', '0.20', '0.25', '0.30', '0.35', '0.40', '0.45', '0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '1.00', '2.00']):
    experiment_directories = []
    for i in diffs:
        experiment_directories.append(f'datasets/bias_lq_{i}')
        experiment_directories.append(f'datasets/noise_sy_{i}')
        experiment_directories.append(f'datasets/imbalance_pu_{i}')


    for i, dir_path in enumerate(experiment_directories):
        print(f"\Processing: {dir_path.split('/')[-1]}")
        X_train, y_train, X_test, y_test = load_data(dir_path)
        print(f"\nPre-split data received:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        _, results = fairness_pipeline_presplit(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            sensitive_attribute=sensitive_attribute,
            output_csv=f'results/{dir_path.split("/")[-1]}.csv',
            random_state=42
        )
        ress = calculate_overall_fairness_score(results)
        print(f"Overall Fairness Score: {ress:.4f} (lower is better)")

if __name__ == "__main__":
    fairness_run(sensitive_attribute='A')

    