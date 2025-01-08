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

def conditional_statistical_parity(df, sensitive_column, target_column, target_value, control_column):
    """
    Verifica la fairness condizionale rispetto a un attributo di controllo.
    """
    print(f"\nAnalisi condizionale rispetto a {control_column}:")
    grouped = df.groupby([sensitive_column, control_column])[target_column].mean().unstack()
    print(grouped)
    return grouped

def evaluate_model_performance(df, sensitive_column, target_column, prediction_column):
    """
    Valuta la fairness del modello tramite Equality of Opportunity e Equalized Odds.
    """
    metrics = {}
    for group in df[sensitive_column].unique():
        group_data = df[df[sensitive_column] == group]
        tpr = group_data[(group_data[target_column] == 1) & (group_data[prediction_column] == 1)].shape[0] / group_data[group_data[target_column] == 1].shape[0]
        fpr = group_data[(group_data[target_column] == 0) & (group_data[prediction_column] == 1)].shape[0] / group_data[group_data[target_column] == 0].shape[0]
        metrics[group] = {"TPR": tpr, "FPR": fpr}
    return metrics

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
    print("\nValori unici e conteggi nella colonna target:")
    print(df['labels'].value_counts())

def plot_disparities(disparities, sensitive_column):
    """
    Mostra un grafico delle disparità per ciascun gruppo sensibile.
    """
    if all(value == 0 for value in disparities.values()):
        print(f"Grafico non generato: tutte le proporzioni per {sensitive_column} sono uguali a 0.")
        return
    plt.bar(disparities.keys(), disparities.values())
    plt.title(f"Disparità per {sensitive_column}")
    plt.xlabel("Gruppo")
    plt.ylabel("Proporzione di outcome positivi")
    plt.show()

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
            plot_disparities(disparities, sensitive_column)
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

    # Analisi condizionale opzionale
    control_column = input("Inserisci una colonna di controllo per l'analisi condizionale (premi invio per saltare): ").strip()
    if control_column and control_column in df.columns:
        conditional_statistical_parity(df, sensitive_columns[0], target_column, target_value, control_column)

    # Analisi delle metriche di performance del modello
    prediction_column = input("Inserisci la colonna delle predizioni del modello (premi invio per saltare): ").strip()
    if prediction_column and prediction_column in df.columns:
        performance_metrics = evaluate_model_performance(df, sensitive_columns[0], target_column, prediction_column)
        print("Metriche di performance per gruppo:", performance_metrics)
