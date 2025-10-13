import pandas as pd
import numpy as np
import re
from rdflib import Graph, Namespace, RDFS
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.base import clone
from constraint import Problem
import itertools
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay, log_loss
from sklearn.model_selection import train_test_split
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- FUNZIONI DI SUPPORTO: ONTOLOGIA E DATI ---

def estrai_category_keywords_da_ontologia(ontology_path):
    g = Graph()
    g.parse(ontology_path, format="xml")
    ns = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    hasKeyword = ns.hasKeyword
    category_keywords = defaultdict(list)
    for s, p, o in g.triples((None, hasKeyword, None)):
        cat_name = str(s).split("#")[-1]
        category_keywords[cat_name].append(str(o))
    return dict(category_keywords)

def get_parents(g, ns, child_class_name):
    parents = set()
    if child_class_name == "Altro" or not child_class_name: return parents
    child_uri = ns[child_class_name]
    for s, p, o in g.triples((child_uri, RDFS.subClassOf, None)):
        if '#' in o: parents.add(o.split('#')[-1])
    return parents

def parse_labels(label_str):
    if pd.isna(label_str) or not label_str.strip(): return []
    return list(dict.fromkeys(lbl.strip() for lbl in str(label_str).split() if lbl.strip()))

def load_training_data(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    df['clean_text'] = df['clean_text'].fillna('')
    df['single_label'] = df['category'].apply(lambda x: parse_labels(x)[0] if parse_labels(x) else "Altro")
    return df[df['single_label'].notna()].copy()

def create_enhanced_features(df, ontology_keywords):
    keyword_features = []
    for _, row in df.iterrows():
        text = str(row.get('clean_text', '')).lower()
        doc_features = {}
        for category, keywords in ontology_keywords.items():
            keyword_count = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', text)) for kw in keywords)
            doc_features[f"{category}_keywords"] = keyword_count
        keyword_features.append(doc_features)
    return pd.DataFrame(keyword_features, index=df.index).fillna(0)

# --- FUNZIONI PER CREAZIONE GROUND TRUTH (Logica a Regole) ---

categories_by_specificity = {
    "very_specific": ["AI_ML", "Web_development", "System_programming", "Data_analysis", "Database", "Security","Ambiente", "Ecologia", "Energia", "Spazio", "Alimentazione", "Cardiologia", "Oncologia","Archeologia", "Antica", "Moderna", "Contemporanea", "Preistoria", "Comunicazione", "Animale","Botanica", "Culturale", "Umana"],
    "specific": ["Informatica", "Biologia", "Fisica", "Medicina", "Chimica", "Antropologia", "Filosofia", "Paleontologia", "Storia"],
    "general": ["Scienza", "Studi_umanistici"], "fallback": ["Altro"]
}
extension_categories = {".html": ["Web_development"], ".css": ["Web_development"], ".js": ["Web_development"], ".php": ["Web_development"], ".py": ["AI_ML"], ".java": ["System_programming"], ".cpp": ["System_programming"], ".c": ["System_programming"], ".sql": ["Database"], ".csv": ["Data_analysis"], ".json": ["Data_analysis"], ".unknown": ["Altro"]}

def find_most_specific_category(row, category_keywords):
    full_text = f"{str(row.get('titolo', '')).lower()} {str(row.get('filename', '')).lower()} {str(row.get('clean_text', '')).lower()}"
    category_scores = defaultdict(int)
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            matches = len(re.findall(pattern, full_text))
            if matches > 0: category_scores[category] += len(keyword.split()) * 2 * matches
    best_category, best_score = None, 0
    for level in ['very_specific', 'specific']:
        if best_category is None:
            for category in categories_by_specificity[level]:
                if category_scores.get(category, 0) > best_score:
                    best_score, best_category = category_scores[category], category
    if best_category is None and row.get('extension', '') in extension_categories:
        ext_cats = extension_categories[row.get('extension', '')]
        if ext_cats and ext_cats[0] != 'Altro': best_category = ext_cats[0]
    if best_category is None:
        for category in categories_by_specificity['general']:
            if category_scores.get(category, 0) > 3: best_category = category; break
    return best_category if best_category else "Altro"

# --- FUNZIONI PER PREDIZIONE CON VINCOLI (CSP) ---

def setup_csp_problem(doc_probs, ontology_graph, ns):
    problem = Problem()
    prob_l1 = {k: v for k, v in doc_probs['L1'].items() if v > 0.01}
    prob_l2 = {k: v for k, v in doc_probs['L2'].items() if v > 0.01}
    prob_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0.01}
    if not prob_l1 or not prob_l2 or not prob_l3: return None
    problem.addVariable("L1Category", list(prob_l1.keys()))
    problem.addVariable("L2Category", list(prob_l2.keys()))
    problem.addVariable("L3Category", list(prob_l3.keys()))
    def hierarchical_constraint(l2_cat, l3_cat): return l2_cat in get_parents(ontology_graph, ns, l3_cat)
    def hierarchical_constraint_l1(l1_cat, l2_cat): return l1_cat in get_parents(ontology_graph, ns, l2_cat)
    problem.addConstraint(hierarchical_constraint, ("L2Category", "L3Category"))
    problem.addConstraint(hierarchical_constraint_l1, ("L1Category", "L2Category"))
    return problem

def find_best_csp_solution(problem, doc_probs):
    if problem is None: return "N/A", "N/A", "N/A", {}
    solutions = problem.getSolutions()
    if not solutions: return "Incoerente", "Incoerente", "Incoerente", {}
    best_solution, max_prob = None, -1
    for sol in solutions:
        l1, l2, l3 = sol["L1Category"], sol["L2Category"], sol["L3Category"]
        prob = doc_probs['L1'].get(l1, 0) * doc_probs['L2'].get(l2, 0) * doc_probs['L3'].get(l3, 0)
        if prob > max_prob:
            max_prob, best_solution = prob, sol
    if best_solution:
        l1, l2, l3 = best_solution["L1Category"], best_solution["L2Category"], best_solution["L3Category"]
        probs_dict = {'L1_pred': l1, 'L2_pred': l2, 'L3_pred': l3, 'Combined_prob': max_prob}
        return l1, l2, l3, probs_dict
    return "No_Solution", "No_Solution", "No_Solution", {}

# --- FUNZIONI PER VALUTAZIONE E VISUALIZZAZIONE ---

def evaluate_and_get_metrics(df_predictions, df_test_with_ground_truth):
    print("\n" + "--- VALUTAZIONE PERFORMANCE (ACCURACY, PRECISION, RECALL, F1) ---".center(90, "="))
    df_merged = pd.merge(df_predictions, df_test_with_ground_truth[['filename', 'ground_truth_category']], on='filename')
    y_true = df_merged['ground_truth_category']
    models = ['LR', 'RF', 'SVM', 'NB']
    all_metrics = {}
    for model in models:
        pred_col = f'{model}_L3_pred'
        if pred_col not in df_merged.columns: continue
        y_pred = df_merged[pred_col]
        print("\n" + f"--- Performance della Pipeline '{model}' ---".center(90, "-"))
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        all_metrics[model] = {'Accuracy': accuracy, 'Precision_Weighted': precision, 'Recall_Weighted': recall, 'F1_Score_Weighted': f1}
        print(f"Accuratezza Generale: {accuracy:.4f}")
        labels = sorted(list(set(y_true) | set(y_pred)))
        print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print("\n" + "="*90)
    return all_metrics

def plot_performance_summary(metrics_df, output_dir):
    print("\n--- Generazione Grafico Riassuntivo delle Performance ---")
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

def plot_and_save_confusion_matrices(df_predictions, df_test_with_ground_truth, output_dir):
    print("\n--- Generazione Matrici di Confusione ---")
    df_merged = pd.merge(df_predictions, df_test_with_ground_truth[['filename', 'ground_truth_category']], on='filename')
    y_true = df_merged['ground_truth_category']
    models = ['LR', 'RF', 'SVM', 'NB']
    for model in models:
        pred_col = f'{model}_L3_pred'
        if pred_col not in df_merged.columns: continue
        y_pred = df_merged[pred_col]
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_size = max(6, len(labels) / 2.5)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues', values_format='d')
        ax.set_title(f"Matrice di Confusione - Pipeline '{model}' (Livello L3)")
        plt.tight_layout()
        filename = os.path.join(output_dir, f"confusion_matrix_L3_{model}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"✅ Matrice di confusione per '{model}' salvata.")

def plot_loss_curve(X_train, y_train, output_dir):
    print("\n--- Generazione Curva di Loss (Esempio con SGD) ---")
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
    print("\n--- Analisi delle Parole più Importanti (TF-IDF) ---")
    try:
        with open(vectorizer_path, 'rb') as f: vectorizer = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ ERRORE: File vectorizer '{vectorizer_path}' non trovato.")
        return
    feature_names = np.array(vectorizer.get_feature_names_out())
    scores = vectorizer.idf_
    top_indices = scores.argsort()[-top_n:]
    top_scores, top_features = scores[top_indices], feature_names[top_indices]
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


# =================================================================================
# --- BLOCCO DI ESECUZIONE PRINCIPALE ---
# =================================================================================
if __name__ == "__main__":
    # --- 1. SETUP E CONFIGURAZIONE ---
    ontology_path = "Ontology.owx"
    train_csv_file = "training_result/training_set_categorized.csv"
    test_csv_file = "test_result/test_data_with_text.csv"
    output_model_dir = "saved_models"
    output_metrics_dir = "metrics"
    force_retrain = False

    print("🚀 AVVIO PIPELINE AVANZATA: ADDESTRAMENTO, SALVATAGGIO, PREDIZIONE E VALUTAZIONE 🚀")
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)
    
    model_templates = {'LR': make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, random_state=42)), 'RF': RandomForestClassifier(n_estimators=100, random_state=42), 'SVM': make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, random_state=42)), 'NB': MultinomialNB()}
    levels = {'L1': None, 'L2': None, 'L3': None}

    # --- 2. CONTROLLO ESISTENZA MODELLI E ADDESTRAMENTO CONDIZIONALE ---
    training_needed = False
    expected_files = ['vectorizer.pkl'] + [f"model_{level}_{model}.pkl" for level in levels for model in model_templates]
    
    for filename in expected_files:
        if not os.path.exists(os.path.join(output_model_dir, filename)):
            training_needed = True
            print(f"⚠️  File modello mancante: {filename}. L'addestramento è necessario.")
            break
            
    if not training_needed and not force_retrain:
        print("\n✅ Tutti i modelli sono già presenti. Salto la fase di addestramento.")
    else:
        print("\n" + "--- FASE 1: Addestramento e Salvataggio Modelli ---".center(80, "="))
        if force_retrain: print("ℹ️  'force_retrain' è True. I modelli verranno riaddestrati.")
        g = Graph(); g.parse(ontology_path, format="xml")
        NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
        df_train = load_training_data(train_csv_file)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(df_train['clean_text'])
        with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'wb') as f: pickle.dump(vectorizer, f)
        print("✅ Vectorizer addestrato e salvato.")
        X_train_combined = hstack([X_train_tfidf, csr_matrix(create_enhanced_features(df_train, estrai_category_keywords_da_ontologia(ontology_path)).values)])
        df_train['l3_label'] = df_train['single_label']
        df_train['l2_label'] = df_train['l3_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else "Altro")
        df_train['l1_label'] = df_train['l2_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if x != "Altro" and get_parents(g, NS, x) else "Altro")
        levels_with_data = {'L1': df_train['l1_label'], 'L2': df_train['l2_label'], 'L3': df_train['l3_label']}
        for level_name, y_train in levels_with_data.items():
            for model_name, model_template in model_templates.items():
                model_to_fit = clone(model_template)
                model_to_fit.fit(X_train_combined, y_train)
                with open(os.path.join(output_model_dir, f"model_{level_name}_{model_name}.pkl"), 'wb') as f: pickle.dump(model_to_fit, f)
        print("✅ Tutti i modelli sono stati addestrati e salvati.")

    # --- 3. PREDIZIONE E VALUTAZIONE ---
    print("\n" + "--- FASE 2: Predizione, Creazione Ground Truth e Valutazione ---".center(80, "="))
    g = Graph(); g.parse(ontology_path, format="xml")
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = estrai_category_keywords_da_ontologia(ontology_path)
    df_test = pd.read_csv(test_csv_file).fillna('')
    df_test['ground_truth_category'] = df_test.apply(lambda row: find_most_specific_category(row, ONTOLOGY_KEYWORDS), axis=1)
    print("✅ Ground Truth per il test set generato.")
    with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'rb') as f: vectorizer = pickle.load(f)
    trained_models = {'L1': {}, 'L2': {}, 'L3': {}}
    for level in levels.keys():
        for model in model_templates.keys():
            with open(os.path.join(output_model_dir, f"model_{level}_{model}.pkl"), 'rb') as f: trained_models[level][model] = pickle.load(f)
    print("✅ Modelli e vectorizer caricati per la predizione.")
    X_test_tfidf = vectorizer.transform(df_test['clean_text'])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(create_enhanced_features(df_test, ONTOLOGY_KEYWORDS).values)])
    results = []
    for i in range(len(df_test)):
        row_result = {'filename': df_test.iloc[i]['filename']}
        for model_name in model_templates.keys():
            doc_probs = {level_name: dict(zip(models[model_name].classes_, models[model_name].predict_proba(X_test_combined[i])[0])) for level_name, models in trained_models.items()}
            problem = setup_csp_problem(doc_probs, g, NS)
            l1, l2, l3, probs_dict = find_best_csp_solution(problem, doc_probs)
            row_result[f'{model_name}_L1_pred'] = l1
            row_result[f'{model_name}_L2_pred'] = l2
            row_result[f'{model_name}_L3_pred'] = l3
            row_result[f'{model_name}_Combined_prob'] = probs_dict.get('Combined_prob', 0.0)
        results.append(row_result)
    df_results = pd.DataFrame(results)
    
    # --- 4. VALUTAZIONE E SALVATAGGIO FINALE ---
    performance_metrics = evaluate_and_get_metrics(df_results, df_test)
    metrics_df = pd.DataFrame(performance_metrics).T
    print("\n--- RIEPILOGO METRICHE AGGREGATE ---")
    print(metrics_df.round(4).to_string())
    output_predictions_filename = os.path.join(output_metrics_dir, "predictions_and_evaluation_results.csv")
    output_metrics_filename = os.path.join(output_metrics_dir, "performance_metrics_summary.csv")
    df_final_output = pd.merge(df_results, df_test[['filename', 'ground_truth_category']], on='filename')
    df_final_output.to_csv(output_predictions_filename, index=False, float_format='%.4f')
    print(f"\n✅ Risultati finali (previsioni ML + ground truth) salvati in '{output_predictions_filename}'.")
    metrics_df.to_csv(output_metrics_filename, float_format='%.4f')
    print(f"✅ Riepilogo metriche di performance salvato in '{output_metrics_filename}'.")
    
    # --- 5. GENERAZIONE GRAFICI ---
    plot_and_save_confusion_matrices(df_results, df_test, output_metrics_dir)
    try:
        df_train_for_loss = load_training_data(train_csv_file)
        df_train_for_loss['l3_label'] = df_train_for_loss['single_label']
        with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'rb') as f: 
            vectorizer_for_loss = pickle.load(f)
        X_train_tfidf_loss = vectorizer_for_loss.transform(df_train_for_loss['clean_text'])
        X_train_combined_loss = hstack([X_train_tfidf_loss, csr_matrix(create_enhanced_features(df_train_for_loss, estrai_category_keywords_da_ontologia(ontology_path)).values)])
        plot_loss_curve(X_train_combined_loss, df_train_for_loss['l3_label'], output_metrics_dir)
    except Exception as e:
        print(f"⚠️  Impossibile generare la curva di loss: {e}")
    plot_top_tfidf_features(os.path.join(output_model_dir, 'vectorizer.pkl'), output_metrics_dir)
    plot_performance_summary(metrics_df, output_metrics_dir)
    
    print("\n🎉 PROCESSO COMPLETO TERMINATO! 🎉")