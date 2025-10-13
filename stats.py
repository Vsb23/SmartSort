import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from scipy.sparse import hstack, csr_matrix

# Assicurati che le funzioni di supporto necessarie siano definite o importate
# Queste sono necessarie per ricreare le feature per la curva di loss
from rdflib import Graph, Namespace, RDFS
from collections import defaultdict
import re

# (Qui incolliamo le funzioni di supporto necessarie prese dall'altro script)
def estrai_category_keywords_da_ontologia(ontology_path):
    # ... (codice della funzione)
    pass
# ... (incolla qui anche create_enhanced_features, load_training_data, ecc.)


def plot_confusion_matrices(predictions_csv_path, output_dir):
    """
    Genera e salva le matrici di confusione per ogni pipeline di modelli.
    """
    print("--- 1. Generazione Matrici di Confusione ---")
    try:
        df = pd.read_csv(predictions_csv_path)
    except FileNotFoundError:
        print(f"❌ ERRORE: File delle previsioni '{predictions_csv_path}' non trovato. Esegui prima lo script principale.")
        return

    y_true = df['ground_truth_category']
    models = ['LR', 'RF', 'SVM', 'NB']
    
    for model in models:
        pred_col = f'{model}_L3_pred'
        if pred_col not in df.columns: continue
        y_pred = df[pred_col]
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        
        plt.title(f"Matrice di Confusione - Pipeline '{model}' (Livello L3)")
        plt.ylabel('Categoria Vera (Ground Truth)')
        plt.xlabel('Categoria Predetta')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"confusion_matrix_L3_{model}.png")
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"✅ Matrice di confusione per '{model}' salvata.")

def plot_loss_curve(X_train, y_train, output_dir):
    """
    Genera una curva di apprendimento (loss) usando un modello iterativo.
    """
    print("\n--- 2. Generazione Curva di Loss (Esempio con SGD) ---")
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    model = SGDClassifier(loss='log_loss', max_iter=1, tol=None, random_state=42, warm_start=True)
    n_epochs = 50
    train_losses, val_losses = [], []
    classes = np.unique(y_train)
    
    print("Addestramento iterativo per la curva di loss...")
    for epoch in range(n_epochs):
        model.partial_fit(X_train_part, y_train_part, classes=classes)
        train_prob = model.predict_proba(X_train_part)
        val_prob = model.predict_proba(X_val)
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
    
    filename = os.path.join(output_dir, "learning_curve_SGD_L3.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Curva di Loss salvata.")

def plot_top_tfidf_features(vectorizer_path, output_dir, top_n=25):
    """
    Visualizza le parole con i punteggi TF-IDF più alti.
    """
    print("\n--- 3. Analisi delle Parole più Importanti (TF-IDF) ---")
    try:
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE: File vectorizer '{vectorizer_path}' non trovato. Esegui prima lo script di addestramento.")
        return

    feature_names = np.array(vectorizer.get_feature_names_out())
    # Calcoliamo un punteggio medio per ogni parola (idf è una buona proxy)
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
    # --- IMPOSTAZIONI ---
    predictions_csv = 'metrics/predictions_and_evaluation_results.csv'
    training_csv = 'training_result/training_set_categorized.csv' # Necessario per la curva di loss
    vectorizer_file = 'saved_models/vectorizer.pkl' # Necessario per loss e tf-idf
    ontology_file = 'Ontology.owx'
    
    output_visualization_dir = 'metrics/visualizations'
    os.makedirs(output_visualization_dir, exist_ok=True)

    print("🚀 Avvio generazione metriche visive... 🚀")
    
    # 1. Matrici di Confusione
    plot_confusion_matrices(predictions_csv, output_visualization_dir)
    
    # 2. Curva di Loss
    # (Questa parte richiede di caricare i dati di training e ricreare le feature)
    try:
        df_train = load_training_data(training_csv) # Assicurati che load_training_data sia definita
        df_train['l3_label'] = df_train['single_label']
        with open(vectorizer_file, 'rb') as f: vectorizer_for_loss = pickle.load(f)
        X_train_tfidf_loss = vectorizer_for_loss.transform(df_train['clean_text'])
        X_train_combined_loss = hstack([X_train_tfidf_loss, csr_matrix(create_enhanced_features(df_train, estrai_category_keywords_da_ontologia(ontology_file)).values)]) # Assicurati che le funzioni siano definite
        plot_loss_curve(X_train_combined_loss, df_train['l3_label'], output_visualization_dir)
    except Exception as e:
        print(f"⚠️  Impossibile generare la curva di loss: {e}")

    # 3. Grafico TF-IDF
    plot_top_tfidf_features(vectorizer_file, output_visualization_dir)
    
    print("\n🎉 Processo di visualizzazione completato! Controlla la cartella 'metrics/visualizations'. 🎉")