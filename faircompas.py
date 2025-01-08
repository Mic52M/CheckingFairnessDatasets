import pandas as pd
from datasets import load_dataset
import matplotlib.pyplot as plt
from aif360.datasets import BinaryLabelDataset

def load_compas_dataset():
    """
    Carica il dataset COMPAS e lo converte in un DataFrame pandas.
    """
    # Carica il dataset COMPAS
    df = pd.read_csv('compas-scores-two-years.csv')

    # Rimuove righe con valori NA
    df = df.dropna()

    # Definizione delle colonne richieste
    label_names = ['is_recid']  # Colonna target che indica se è recidivo
    protected_attribute_names = ['race', 'sex']  # Attributi sensibili

    # Verifica che la colonna target esista e non sia vuota
    if 'is_recid' not in df.columns or df['is_recid'].isnull().all():
        print("Avviso: La colonna target 'is_recid' è vuota o non esiste.")
        print("Ecco le colonne disponibili nel dataset:")
        print(df.columns.tolist())
        print("\nPuoi selezionare una colonna target alternativa quando richiesto.")
        return df  # Restituisce il dataset senza interrompere l'esecuzione

    # Crea un dataset BinaryLabelDataset
    dataset = BinaryLabelDataset(
        favorable_label=0, 
        unfavorable_label=1, 
        df=df, 
        label_names=label_names, 
        protected_attribute_names=protected_attribute_names
    )

    # Converte in DataFrame per ulteriori elaborazioni
    df['target'] = dataset.labels.ravel()
    return df

def explore_dataset(df):
    """
    Esplora il dataset e restituisce tutte le colonne disponibili.
    """
    print("\nColonne disponibili nel dataset:")
    print(df.columns.tolist())
    print("\nValori unici e conteggi nella colonna target:")
    if 'target' in df.columns:
        print(df['target'].value_counts())
    else:
        print("La colonna 'target' non è presente nel dataset.")

def statistical_parity_check(df, sensitive_column, target_column, target_value):
    """
    Verifica la fairness di un dataset rispetto allo Statistical Parity.

    Parameters:
    - df: pandas DataFrame contenente il dataset
    - sensitive_column: colonna sensibile (es. 'race', 'sex')
    - target_column: colonna target (es. 'target')
    - target_value: il valore della classe target da confrontare (es. 1 per recidivo)

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
    if not disparities:
        print(f"Errore: Nessun dato valido per la colonna sensibile '{sensitive_column}'.")
        return {}, True

    max_disparity = max(disparities.values()) - min(disparities.values())
    threshold = 0.1
    fairness = max_disparity <= threshold

    return disparities, fairness

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
    - target_column: colonna target (es. 'target')
    - target_value: valore della classe target da verificare (es. 1 per recidivo)
    """
    for sensitive_column in sensitive_columns:
        if sensitive_column in df.columns:
            print(f"\nAnalisi per la colonna sensibile: {sensitive_column}")
            disparities, fairness = statistical_parity_check(df, sensitive_column, target_column, target_value)
            if disparities:
                print("Disparità per gruppo:", disparities)
                print("Il dataset è fair?", fairness)
                plot_disparities(disparities, sensitive_column)
        else:
            print(f"Colonna sensibile '{sensitive_column}' non trovata nel dataset.")

# Esempio di utilizzo
if __name__ == "__main__":
    # Caricamento del dataset COMPAS
    df = load_compas_dataset()

    # Esplora il dataset
    explore_dataset(df)

    # Specifica la colonna target
    if 'target' not in df.columns:
        target_column = input("Inserisci manualmente una colonna target dal dataset: ").strip()
    else:
        target_column = 'target'

    # Seleziona manualmente le colonne sensibili
    sensitive_columns = input("Inserisci le colonne sensibili separate da virgola: ").split(',')
    sensitive_columns = [col.strip() for col in sensitive_columns]

    # Specifica il valore target
    target_value = int(input("Inserisci il valore della classe target (es. 1 per recidivo): ").strip())

    # Analizza la fairness
    analyze_fairness(df, sensitive_columns, target_column, target_value)
