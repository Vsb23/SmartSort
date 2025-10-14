# FILE: stats.py (MODIFICATO E SEMPLIFICATO)

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler

# --- NESSUNA FUNZIONE DI SUPPORTO NECESSARIA QUI ---
# Le features vengono caricate direttamente, non ricalcolate.

# --- FUNZIONI DI PLOTTING E VISUALIZZAZIONE (INVARIATE) ---

def plot_performance_summary(metrics_csv_path, output_dir):
    """Genera un grafico a barre che riassume le performance dei modelli."""
    print("\n--- 1. Generazione Grafico Riassuntivo delle Performance ---")
    try:
        metrics_df = pd.read_csv(metrics_csv_path, index_col=0)
    except FileNotFoundError:
        print(f"❌ ERRORE: File delle metriche '{metrics_csv_path}' non trovato. Esegui prima 'dataset_ricorsivo.py'.")
        return
    # ... (il resto della funzione è identico)
    metrics_to_plot = metrics_df[['Precision_Weighted', 'Recall_Weighted', 'F1_Score_Weighted']].copy()
    fig, ax = plt.subplots(figsize=(12, 7))
    metrics_to_plot.plot(kind='bar', ax=ax, colormap='viridis')
    ax.set_title('Confronto Performance Pipeline Modelli (Weighted Avg)', fontsize=16)
    ax.set_ylabel('Punteggio', fontsize=12)
    ax.set_xlabel('Pipeline di Modelli', fontsize=12)
    ax.tick_params(axis='x', rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 1.1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.legend(title='Metriche')
    plt.tight_layout()
    filename = os.path.join(output_dir, "performance_summary_barchart.png")
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"✅ Grafico riassuntivo salvato in: {filename}")


def plot_confusion_matrices(predictions_csv_path, output_dir):
    """Genera e salva le matrici di confusione per ogni pipeline di modelli."""
    print("\n--- 2. Generazione Matrici di Confusione ---")
    try:
        df = pd.read_csv(predictions_csv_path)
    except FileNotFoundError:
        print(f"❌ ERRORE: File delle previsioni '{predictions_csv_path}' non trovato.")
        return
    # ... (il resto della funzione è identico)
    y_true = df['ground_truth_category']
    models = ['LR', 'RF', 'SVM', 'NB']
    
    for model in models:
        pred_col = f'{model}_L3_pred'
        if pred_col not in df.columns: continue
        y_pred = df[pred_col]
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_size = max(8, len(labels) / 2.0)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
        ax.set_title(f"Matrice di Confusione - Pipeline '{model}' (Livello L3)")
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"confusion_matrix_L3_{model}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Matrice di confusione per '{model}' salvata.")


def plot_loss_curve(X_train, y_train, output_dir):
    """
    Genera una curva di apprendimento (train vs validation loss) usando un modello SGD,
    applicando lo scaling e un learning rate adattivo per massima stabilità.
    """
    print("\n--- 3. Generazione Curva di Loss (Train vs Validation) ---")
    try:
        X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"❌ ERRORE durante lo split per la curva di loss: {e}")
        return

    scaler = StandardScaler(with_mean=False)
    X_train_part_scaled = scaler.fit_transform(X_train_part)
    X_val_scaled = scaler.transform(X_val)
    
# In stats.py
    model = SGDClassifier(
        loss='log_loss',
        random_state=42,
        learning_rate='adaptive',
        eta0=0.01,
        alpha=0.001,  # PROVA AD AGGIUNGERE QUESTO (aumentalo se necessario a 0.01)
        max_iter=1,
        tol=None,
        warm_start=True
    )
    
    n_epochs = 50
    train_losses, val_losses = [], []
    classes = np.unique(y_train)
    
    print("Addestramento iterativo per la curva di loss (con learning rate adattivo)...")
    for epoch in range(n_epochs):
        model.partial_fit(X_train_part_scaled, y_train_part, classes=classes)
        train_prob = model.predict_proba(X_train_part_scaled)
        val_prob = model.predict_proba(X_val_scaled)
        train_losses.append(log_loss(y_train_part, train_prob, labels=model.classes_))
        val_losses.append(log_loss(y_val, val_prob, labels=model.classes_))
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss', linestyle='--')
    plt.title("Curva di Apprendimento (Train vs Validation Loss)")
    plt.xlabel("Epoche")
    plt.ylabel("Log Loss")
    plt.legend()
    plt.grid(True)
    
    filename = os.path.join(output_dir, "learning_curve_SGD_L3.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Curva di Loss salvata.")

def plot_top_tfidf_features(vectorizer_path, output_dir, top_n=25):
    """Visualizza le parole con i punteggi IDF più alti."""
    print("\n--- 4. Analisi delle Parole più Importanti (TF-IDF) ---")
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE: File vectorizer '{vectorizer_path}' non trovato.")
        return
    # ... (il resto della funzione è identico)
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

    filename = os.path.join(output_dir, "top_tfidf_features.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Grafico TF-IDF salvato.")


if __name__ == '__main__':
    # --- IMPOSTAZIONI DEI PERCORSI ---
    metrics_summary_csv = 'metrics/performance_metrics_summary.csv'
    predictions_csv = 'metrics/predictions_and_evaluation_results.csv'
    vectorizer_file = 'saved_models/vectorizer.pkl'
    
    # --- NUOVI PERCORSI PER I DATI PROCESSATI ---
    X_train_file = 'processed_data/X_train_combined.npz'
    y_train_file = 'processed_data/y_train_L3.pkl'
    
    output_visualization_dir = 'metrics/'
    os.makedirs(output_visualization_dir, exist_ok=True)

    print("🚀 Avvio generazione metriche visive... 🚀")
    
    plot_performance_summary(metrics_summary_csv, output_visualization_dir)
    plot_confusion_matrices(predictions_csv, output_visualization_dir)
    
    # --- BLOCCO PER CURVA DI LOSS (SEMPLIFICATO) ---
    try:
        print("\nCarico i dati pre-processati per la curva di loss...")
        X_train = load_npz(X_train_file)
        y_train = pd.read_pickle(y_train_file)
        print("✅ Dati caricati con successo.")
        
        plot_loss_curve(X_train, y_train, output_visualization_dir)
    except FileNotFoundError:
        print(f"⚠️  Impossibile generare la curva di loss: file '{X_train_file}' o '{y_train_file}' non trovato.")
        print("   Assicurati di aver eseguito 'dataset_ricorsivo.py' almeno una volta.")
    except Exception as e:
        print(f"⚠️  Impossibile generare la curva di loss: {e}")

    plot_top_tfidf_features(vectorizer_file, output_visualization_dir)
    
    print("\n🎉 Processo di visualizzazione completato! Controlla la cartella 'metrics/visualizations'. 🎉")