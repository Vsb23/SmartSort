# FILE: stats.py (MODIFICATO PER GESTIRE DUE TEST SET)

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler

# --- 1. FUNZIONE PER GRAFICO K-FOLD (Invariata) ---
def plot_kfold_validation_summary(kfold_summary_path, output_dir):
    """
    Genera un grafico a barre che riassume le performance K-Fold (Media e Dev. Std).
    Questo grafico è relativo al training/validazione, non ai test set finali.
    """
    print("\n--- 1. Generazione Grafico K-Fold (Media & Dev. Std) ---")
    try:
        metrics_df = pd.read_csv(kfold_summary_path, index_col=0)
    except FileNotFoundError:
        print(f"❌ ERRORE: File delle metriche K-Fold '{kfold_summary_path}' non trovato.")
        print("   Assicurati di aver eseguito 'dataset_ricorsivo.py' (versione aggiornata).")
        return

    models = metrics_df.index
    means = metrics_df['F1_Mean']
    std_devs = metrics_df['F1_StdDev']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, means, yerr=std_devs, capsize=5, color='royalblue', alpha=0.7, ecolor='black')
    
    ax.set_title('Performance K-Fold (K=5) - F1-Score Medio con Deviazione Standard', fontsize=15)
    ax.set_ylabel('F1-Score (Weighted Mean)', fontsize=12)
    ax.set_xlabel('Pipeline di Modelli', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        std = std_devs.iloc[i] 
        ax.annotate(f'{height:.3f}\n(± {std:.3f})', 
                    (bar.get_x() + bar.get_width() / 2., height), 
                    ha='center', va='bottom', xytext=(0, 8), 
                    textcoords='offset points', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "1_kfold_performance_summary.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"✅ Grafico K-Fold (con Dev. Std) salvato in: {filename}")


# --- 2. FUNZIONE PER GRAFICO SINGOLO TEST SET (MODIFICATA) ---
def plot_test_set_performance(metrics_csv_path, output_dir, suffix=''):
    """
    Genera un grafico a barre che riassume le performance sul SINGOLO TEST SET.
    'suffix' viene usato per differenziare i nomi dei file (es. '_primo', '_secondo')
    """
    # Pulisce il suffisso per un titolo più leggibile
    title_suffix = suffix.replace('_', ' ').strip().title()
    print(f"\n--- 2. Generazione Grafico Riassuntivo (Test Set: {title_suffix}) ---")
    
    try:
        metrics_df = pd.read_csv(metrics_csv_path, index_col=0)
    except FileNotFoundError:
        print(f"❌ ERRORE: File delle metriche '{metrics_csv_path}' non trovato.")
        print(f"   Assicurati che 'dataset_ricorsivo.py' abbia generato questo file.")
        return
        
    metrics_to_plot = metrics_df[['Precision_Weighted', 'Recall_Weighted', 'F1_Score_Weighted', 'Accuracy']].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics_to_plot.plot(kind='bar', ax=ax, colormap='viridis')
    
    ax.set_title(f'Confronto Performance (Test Set: {title_suffix})', fontsize=16)
    ax.set_ylabel('Punteggio', fontsize=12)
    ax.set_xlabel('Pipeline di Modelli', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.legend(title='Metriche')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, f"2_test_set_performance{suffix}.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"✅ Grafico riassuntivo (Test Set {title_suffix}) salvato in: {filename}")


# # --- 3. FUNZIONE PER MATRICI DI CONFUSIONE (MODIFICATA) ---
# def plot_confusion_matrices(predictions_csv_path, output_dir, suffix=''):
#     """
#     Genera e salva le matrici di confusione per ogni pipeline.
#     'suffix' viene usato per differenziare i nomi dei file (es. '_primo', '_secondo')
#     """
#     title_suffix = suffix.replace('_', ' ').strip().title()
#     print(f"\n--- 3. Generazione Matrici di Confusione (Test Set: {title_suffix}) ---")
    
#     try:
#         df = pd.read_csv(predictions_csv_path)
#     except FileNotFoundError:
#         print(f"❌ ERRORE: File delle previsioni '{predictions_csv_path}' non trovato.")
#         return
    
#     y_true = df['ground_truth_category']
#     models = ['LR', 'RF', 'SVM', 'NB']
    
#     for model in models:
#         pred_col = f'{model}_L3_pred'
#         if pred_col not in df.columns: 
#             print(f"   INFO: Saltato modello '{model}' (colonna '{pred_col}' non trovata).")
#             continue
            
#         y_pred = df[pred_col]
#         labels = sorted(list(set(y_true) | set(y_pred)))
#         cm = confusion_matrix(y_true, y_pred, labels=labels)
#         fig_size = max(8, len(labels) / 2.5) # Calcolo dinamico della dimensione
        
#         fig, ax = plt.subplots(figsize=(fig_size, fig_size))
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
#         disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
#         ax.set_title(f"Matrice di Confusione - '{model}' (Test Set: {title_suffix})")
#         plt.tight_layout()
        
#         filename = os.path.join(output_dir, f"3_confusion_matrix_L3_{model}{suffix}.png")
#         plt.savefig(filename, dpi=150)
#         plt.close()
#         print(f"✅ Matrice di confusione per '{model}' (Test Set {title_suffix}) salvata.")


# --- 4. FUNZIONE CURVA DI LOSS (Invariata) ---
def plot_loss_curve(X_train, y_train, output_dir):
    """
    Genera una curva di apprendimento (train vs validation loss) usando un modello SGD.
    """
    print("\n--- 4. Generazione Curva di Loss (Train vs Validation) ---")
    try:
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"❌ ERRORE durante lo split per la curva di loss: {e}")
        return

    scaler = StandardScaler(with_mean=False)
    X_train_part_scaled = scaler.fit_transform(X_train_part)
    X_val_scaled = scaler.transform(X_val)
    
    model = SGDClassifier(
        loss='log_loss',
        random_state=42,
        learning_rate='adaptive',
        eta0=0.01,
        alpha=0.001,
        max_iter=1,
        tol=None,
        warm_start=True
    )
    
    n_epochs = 150
    train_losses, val_losses = [], []
    classes = np.unique(y_train)
    
    print("Addestramento iterativo per la curva di loss...")
    for epoch in range(n_epochs):
        model.partial_fit(X_train_part_scaled, y_train_part, classes=classes)
        train_prob = model.predict_proba(X_train_part_scaled)
        val_prob = model.predict_proba(X_val_scaled)
        train_losses.append(log_loss(y_train_part, train_prob, labels=model.classes_))
        val_losses.append(log_loss(y_val, val_prob, labels=model.classes_))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title("Curva di Apprendimento (Train vs Validation Loss)")
    plt.xlabel("Epoche")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(output_dir, "4_learning_curve_SGD_L3.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Curva di Loss salvata.")

# --- 5. FUNZIONE TOP TF-IDF (Invariata) ---
def plot_top_tfidf_features(vectorizer_path, output_dir, top_n=25):
    """Visualizza le parole con i punteggi IDF più alti."""
    print("\n--- 5. Analisi delle Parole più Importanti (TF-IDF) ---")
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE: File vectorizer '{vectorizer_path}' non trovato.")
        return
        
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = vectorizer.idf_
    
    top_indices = scores.argsort()[-top_n:]
    top_scores = scores[top_indices]
    top_features = feature_names[top_indices]

    plt.figure(figsize=(10, 8))
    plt.barh(top_features, top_scores, color='skyblue')
    plt.title(f'Le {top_n} Parole più Rilevanti (punteggio IDF)')
    plt.xlabel('Punteggio IDF (più alto = più raro e specifico)')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    filename = os.path.join(output_dir, "5_top_tfidf_features.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Grafico TF-IDF salvato.")


if __name__ == '__main__':
    # --- IMPOSTAZIONI DEI PERCORSI (MODIFICATO) ---
    
    output_visualization_dir = 'metrics/'
    os.makedirs(output_visualization_dir, exist_ok=True)
    
    # 1. Percorso K-Fold (relativo al training)
    kfold_summary_csv = 'metrics/kfold_metrics_summary.csv'
    
    # 2. Percorsi per Test Set 1 (Generale)
    test_set_metrics_csv_1 = 'metrics/performance_metrics_summary_primo.csv'
    test_set_predictions_csv_1 = 'metrics/predictions_and_evaluation_results_primo.csv'
    suffix_1 = '_primo'
    
    # 3. Percorsi per Test Set 2 (Calcio)
    test_set_metrics_csv_2 = 'metrics/performance_metrics_summary_secondo.csv'
    test_set_predictions_csv_2 = 'metrics/predictions_and_evaluation_results_secondo.csv'
    suffix_2 = '_secondo'

    test_set_metrics_csv_3 = 'metrics/performance_metrics_summary_terzo.csv'
    test_set_predictions_csv_3 = 'metrics/predictions_and_evaluation_results_terzo.csv'
    suffix_2 = '_terzo'
    
    # 4. Percorsi per grafici di training (loss, tfidf)
    vectorizer_file = 'saved_models/vectorizer.pkl'
    X_train_file = 'processed_data/X_train_combined.npz'
    y_train_file = 'processed_data/y_train_L3.pkl'

    print("🚀 Avvio generazione metriche visive... 🚀")
    
    # --- ESECUZIONE DELLE FUNZIONI DI PLOTTING ---
    
    # 1. Grafico K-Fold (1 solo grafico, relativo al training)
    plot_kfold_validation_summary(kfold_summary_csv, output_visualization_dir)
    
    # 2. Grafici Performance (2 grafici, uno per test set)
    plot_test_set_performance(test_set_metrics_csv_1, output_visualization_dir, suffix=suffix_1)
    plot_test_set_performance(test_set_metrics_csv_2, output_visualization_dir, suffix=suffix_2)
    
    # # 3. Matrici di confusione (2 set di matrici)
    # plot_confusion_matrices(test_set_predictions_csv_1, output_visualization_dir, suffix=suffix_1)
    # plot_confusion_matrices(test_set_predictions_csv_2, output_visualization_dir, suffix=suffix_2)
    
    # 4. Curva di Loss (1 solo grafico, relativo al training)
    try:
        print("\nCarico i dati pre-processati per la curva di loss...")
        X_train = load_npz(X_train_file)
        y_train = pd.read_pickle(y_train_file)
        print("✅ Dati caricati con successo.")
        
        plot_loss_curve(X_train, y_train, output_visualization_dir)
    except FileNotFoundError:
        print(f"⚠️  Impossibile generare la curva di loss: file '{X_train_file}' o '{y_train_file}' non trovato.")
    except Exception as e:
        print(f"⚠️  Impossibile generare la curva di loss: {e}")

    # 5. Grafico TF-IDF (1 solo grafico, relativo al training)
    plot_top_tfidf_features(vectorizer_file, output_visualization_dir)
    
    print("\n🎉 Processo di visualizzazione completato! Controlla la cartella 'metrics/'. 🎉")