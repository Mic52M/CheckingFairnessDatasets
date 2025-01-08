import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt

def statistical_parity_check(df, sensitive_column, target_column, target_value):
    """
    Verifica la fairness di un dataset rispetto allo Statistical Parity.

    Parameters:
    - df: pandas DataFrame contenente il dataset
    - sensitive_column: colonna sensibile (es. 'applicant_gender', 'applicant_age')
    - target_column: colonna target (es. 'labels')
    - target_value: il valore della classe target da confrontare (es. 'positivo', '1')

    Returns:
    - disparities: dizionario con le proporzioni per ciascun gruppo sensibile
    - fairness: True se le proporzioni sono simili, False altrimenti (soglia di default: 0.1)
    """
    proportions = (
        df[df[target_column] == target_value]
        .groupby(sensitive_column)
        .size() / df.groupby(sensitive_column).size()
    ).fillna(0)

    disparities = proportions.to_dict()
    max_disparity = max(disparities.values()) - min(disparities.values())
    threshold = 0.1
    fairness = max_disparity <= threshold

    return disparities, fairness

def preprocess_target_column(df, target_column):
    """
    Preprocessa la colonna target per assicurarsi che contenga valori scalari.
    Se i valori sono liste, verrà estratto il primo elemento o convertiti in scalari.
    """
    if df[target_column].apply(lambda x: isinstance(x, list)).any():
        df[target_column] = df[target_column].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    return df

def explore_dataset(df):
    """
    Esplora il dataset e restituisce tutte le colonne disponibili.
    """
    print("\nColonne disponibili nel dataset:")
    print(df.columns.tolist())

def analyze_fairness(df, sensitive_columns, target_column, target_value):
    """
    Verifica la fairness su un dataset.

    Parameters:
    - df: pandas DataFrame contenente il dataset
    - sensitive_columns: elenco delle colonne sensibili
    - target_column: colonna target (es. 'labels')
    - target_value: valore della classe target da verificare (es. 'positivo', '1')
    """
    for sensitive_column in sensitive_columns:
        if sensitive_column in df.columns:
            print(f"\nAnalisi per la colonna sensibile: {sensitive_column}")
            disparities, fairness = statistical_parity_check(df, sensitive_column, target_column, target_value)
            print("Disparità per gruppo:", disparities)
            print("Il dataset è fair?", fairness)
        else:
            print(f"Colonna sensibile '{sensitive_column}' non trovata nel dataset.")

# Esempio di utilizzo
if __name__ == "__main__":
    # Specifica il percorso del dataset o il nome del dataset su Hugging Face
    dataset_path = 'coastalcph/fairlex'
    config_name = 'ecthr'  # Specifica la configurazione per il dataset Hugging Face

    # Caricamento del dataset
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    else:
        dataset = load_dataset(dataset_path, config_name)
        df = pd.DataFrame(dataset['train'])

    # Esplora il dataset
    explore_dataset(df)

    # Seleziona manualmente le colonne sensibili
    sensitive_columns = input("Inserisci le colonne sensibili separate da virgola: ").split(',')
    sensitive_columns = [col.strip() for col in sensitive_columns]

    # Specifica la colonna target e il valore target
    target_column = input("Inserisci la colonna target: ").strip()
    target_value = input("Inserisci il valore della classe target (es. 1 per esito positivo): ").strip()

    # Preprocessamento della colonna target
    df = preprocess_target_column(df, target_column)

    # Analizza la fairness
    analyze_fairness(df, sensitive_columns, target_column, target_value)
