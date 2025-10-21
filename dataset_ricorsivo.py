# FILE: dataset_ricorsivo.py (MODIFICATO CON FILTRAGGIO E LOGICA DI FALLBACK)

import pandas as pd
import numpy as np
import re
from rdflib import Graph, Namespace, RDFS
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix, save_npz
from sklearn.base import clone
from constraint import Problem
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pickle
import os

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

def load_training_data(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    df['clean_text'] = df['clean_text'].fillna('')
    df['single_label'] = df['category']
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

# --- FUNZIONI PER PREDIZIONE CON VINCOLI (CSP) ---

# --- MODIFICA FALLBACK ---
# setup_csp_problem ora restuisce una tupla: (problema, dizionario_fallback_l3)
def setup_csp_problem(doc_probs, ontology_graph, ns):
    problem = Problem()
    
    # 1. Filtriamo L3 per le categorie più probabili per ridurre il rumore
    prob_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0.05} # Soglia 5%
    if not prob_l3:
        # Se nessuna L3 è probabile, prepariamo per il fallback
        return None, doc_probs['L3'] 

    # 2. Troviamo tutti i genitori L2 e L1 validi basati sulle L3 probabili
    possible_l2 = set()
    for l3_cat in prob_l3:
        possible_l2.update(get_parents(ontology_graph, ns, l3_cat))
    
    possible_l1 = set()
    for l2_cat in possible_l2:
        possible_l1.update(get_parents(ontology_graph, ns, l2_cat))

    # 3. Se non possiamo costruire una gerarchia (es. L3 è "Altro"), prepariamo per il fallback
    if not possible_l1 or not possible_l2:
        return None, doc_probs['L3']

    # 4. Creiamo il problema CSP solo con le categorie gerarchicamente valide
    #    Questo riduce drasticamente lo spazio di ricerca
    valid_l1 = list(possible_l1 & doc_probs['L1'].keys())
    valid_l2 = list(possible_l2 & doc_probs['L2'].keys())
    valid_l3 = list(prob_l3.keys()) # Usiamo le L3 filtrate

    if not valid_l1 or not valid_l2 or not valid_l3:
        # Se l'intersezione è vuota, c'è un conflitto. Prepariamo per il fallback
        return None, doc_probs['L3']

    problem.addVariable("L1Category", valid_l1)
    problem.addVariable("L2Category", valid_l2)
    problem.addVariable("L3Category", valid_l3)
    
    def hierarchical_constraint(l2_cat, l3_cat): return l2_cat in get_parents(ontology_graph, ns, l3_cat)
    def hierarchical_constraint_l1(l1_cat, l2_cat): return l1_cat in get_parents(ontology_graph, ns, l2_cat)
    
    problem.addConstraint(hierarchical_constraint, ("L2Category", "L3Category"))
    problem.addConstraint(hierarchical_constraint_l1, ("L1Category", "L2Category"))
    
    # Restituiamo il problema CSP e None per il fallback (perché useremo il Piano A)
    return problem, None

# --- MODIFICA FALLBACK ---
# find_best_csp_solution ora accetta la tupla, g, e ns per gestire il Piano B
def find_best_csp_solution(problem_tuple, doc_probs, g, ns):
    problem, fallback_l3_probs = problem_tuple

    # --- PIANO A: Prova a risolvere il CSP ---
    if problem is not None:
        solutions = problem.getSolutions()
        if solutions:
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

    # --- PIANO B (FALLBACK): Se il Piano A fallisce (problem=None o solutions=[]) ---
    # Questo codice viene eseguito se "Incoerente" o "No_Solution" stavano per accadere.
    
    # Se il dizionario di fallback non è stato passato, usa quello di L3 completo
    if fallback_l3_probs is None:
        fallback_l3_probs = doc_probs['L3']
        
    if not fallback_l3_probs:
         return "N/A", "N/A", "N/A", {}
         
    # 1. Trova la L3 più probabile (la nostra "ancora")
    l3_pred = max(fallback_l3_probs, key=fallback_l3_probs.get)
    
    # 2. Ricostruisci L2 (genitore di L3)
    l2_parents = list(get_parents(g, ns, l3_pred))
    l2_pred = l2_parents[0] if l2_parents else "Altro" # Prende il primo genitore L2, o "Altro"
    
    # 3. Ricostruisci L1 (genitore di L2)
    l1_parents = list(get_parents(g, ns, l2_pred))
    l1_pred = l1_parents[0] if l1_parents else "Altro" # Prende il primo genitore L1, o "Altro"

    # Calcola la probabilità di questa catena "forzata"
    prob = doc_probs['L1'].get(l1_pred, 0) * doc_probs['L2'].get(l2_pred, 0) * doc_probs['L3'].get(l3_pred, 0)
    probs_dict = {'L1_pred': l1_pred, 'L2_pred': l2_pred, 'L3_pred': l3_pred, 'Combined_prob': prob}
    
    return l1_pred, l2_pred, l3_pred, probs_dict
    
# --- FUNZIONE PER VALUTAZIONE ---

def evaluate_and_get_metrics(df_predictions, df_test_with_ground_truth):
    print("\n" + "--- VALUTAZIONE PERFORMANCE (ACCURACY, PRECISION, RECALL, F1) ---".center(90, "="))
    df_test_with_ground_truth['ground_truth_category'] = df_test_with_ground_truth['category']
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

# =================================================================================
# --- BLOCCO DI ESECUZIONE PRINCIPALE ---
# =================================================================================
if __name__ == "__main__":
    # --- 1. SETUP E CONFIGURAZIONE ---
    ontology_path = "Ontology.owx"
    train_csv_file = "training_result/training_set_categorized.csv"
    test_csv_file = "test_result/test_data_with_text.csv"
    test_labels_csv_file = "test_result/test_set_categorized.csv"
    output_model_dir = "saved_models"
    output_metrics_dir = "metrics"
    force_retrain = False
    min_samples_per_class = 5 
    
    # --- IMPORTANTE: Rivedi questo parametro! ---

    max_tfidf_features = 30 

    print("🚀 AVVIO PIPELINE ML: ADDESTRAMENTO, PREDIZIONE E VALUTAZIONE 🚀")
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)
    
    model_templates = {
        'LR': make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=0.1,max_iter=1000, random_state=42)), 
        'RF': RandomForestClassifier(n_estimators=100, random_state=42), 
        'SVM': make_pipeline(StandardScaler(with_mean=False), SVC(probability=True, random_state=42)), 
        'NB': MultinomialNB()
    }
    levels = {'L1': None, 'L2': None, 'L3': None}

    # --- 2. ADDESTRAMENTO CONDIZIONALE ---
    training_needed = force_retrain or not all(os.path.exists(os.path.join(output_model_dir, f)) for f in ['vectorizer.pkl'] + [f"model_{lvl}_{mdl}.pkl" for lvl in levels for mdl in model_templates])
            
    if not training_needed:
        print("\n✅ Tutti i modelli e i dati processati sono già presenti. Salto la fase di addestramento.")
    else:
        print("\n" + "--- FASE 1: Addestramento e Salvataggio Modelli ---".center(80, "="))
        if force_retrain: print("ℹ️  'force_retrain' è True. I modelli verranno riaddestrati.")
        
        output_processed_data_dir = "processed_data"
        os.makedirs(output_processed_data_dir, exist_ok=True)
        
        g = Graph(); g.parse(ontology_path, format="xml")
        NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
        df_train = load_training_data(train_csv_file)
        
        print(f"\n--- Filtro delle classi con meno di {min_samples_per_class} campioni ---")
        original_doc_count = len(df_train)
        category_counts = df_train['single_label'].value_counts()
        categories_to_keep = category_counts[category_counts >= min_samples_per_class].index.tolist()
        df_train = df_train[df_train['single_label'].isin(categories_to_keep)]
        filtered_doc_count = len(df_train)
        print(f"Numero di categorie originali: {len(category_counts)}")
        print(f"Numero di categorie mantenute: {len(categories_to_keep)}")
        print(f"Documenti rimossi: {original_doc_count - filtered_doc_count}")
        print(f"Dataset di training ridotto a {filtered_doc_count} documenti.")

        # --- MODIFICA FALLBACK ---
        # Usa la variabile max_tfidf_features
        vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english') 
        print(f"\nAddestramento Vectorizer con max_features={max_tfidf_features}...")
        
        X_train_tfidf = vectorizer.fit_transform(df_train['clean_text'])
        with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'wb') as f: pickle.dump(vectorizer, f)
        print("✅ Vectorizer addestrato e salvato.")
        
        X_train_combined = hstack([X_train_tfidf, csr_matrix(create_enhanced_features(df_train, estrai_category_keywords_da_ontologia(ontology_path)).values)])
        df_train['l3_label'] = df_train['single_label']
        df_train['l2_label'] = df_train['l3_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else "Altro")
        df_train['l1_label'] = df_train['l2_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if x != "Altro" and get_parents(g, NS, x) else "Altro")
        
        levels_with_data = {'L1': df_train['l1_label'], 'L2': df_train['l2_label'], 'L3': df_train['l3_label']}
        
        X_train_path = os.path.join(output_processed_data_dir, 'X_train_combined.npz')
        y_train_path = os.path.join(output_processed_data_dir, 'y_train_L3.pkl')
        save_npz(X_train_path, X_train_combined)
        levels_with_data['L3'].to_pickle(y_train_path)
        print(f"✅ Features di training salvate in '{X_train_path}'.")
        print(f"✅ Etichette di training salvate in '{y_train_path}'.")
        
        for level_name, y_train in levels_with_data.items():
            for model_name, model_template in model_templates.items():
                print(f"Addestramento modello: {model_name} per Livello: {level_name}...")
                model_to_fit = clone(model_template)
                model_to_fit.fit(X_train_combined, y_train)
                with open(os.path.join(output_model_dir, f"model_{level_name}_{model_name}.pkl"), 'wb') as f: pickle.dump(model_to_fit, f)
        print("✅ Tutti i modelli sono stati addestrati e salvati.")

    # --- 3. PREDIZIONE E VALUTAZIONE ---
    print("\n" + "--- FASE 2: Predizione e Valutazione su Test Set ---".center(80, "="))
    g = Graph(); g.parse(ontology_path, format="xml")
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = estrai_category_keywords_da_ontologia(ontology_path)
    
    df_test_data = pd.read_csv(test_csv_file).fillna('')
    df_test_labels = pd.read_csv(test_labels_csv_file).fillna('')
    print("✅ File di test (dati e etichette) caricati.")

    with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'rb') as f: vectorizer = pickle.load(f)
    trained_models = {'L1': {}, 'L2': {}, 'L3': {}}
    for level in levels.keys():
        for model in model_templates.keys():
            with open(os.path.join(output_model_dir, f"model_{level}_{model}.pkl"), 'rb') as f: trained_models[level][model] = pickle.load(f)
    print("✅ Modelli e vectorizer caricati per la predizione.")
    
    X_test_tfidf = vectorizer.transform(df_test_data['clean_text'])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(create_enhanced_features(df_test_data, ONTOLOGY_KEYWORDS).values)])
    
    results = []
    print(f"Inizio predizioni su {len(df_test_data)} documenti di test...")
    for i in range(len(df_test_data)):
        row_result = {'filename': df_test_data.iloc[i]['filename']}
        for model_name in model_templates.keys():
            doc_probs = {level_name: dict(zip(models[model_name].classes_, models[model_name].predict_proba(X_test_combined[i])[0])) for level_name, models in trained_models.items()}
            
            # --- MODIFICA FALLBACK ---
            # Aggiornamento delle chiamate alle funzioni modificate
            problem_tuple = setup_csp_problem(doc_probs, g, NS)
            l1, l2, l3, probs_dict = find_best_csp_solution(problem_tuple, doc_probs, g, NS)
            
            row_result[f'{model_name}_L1_pred'] = l1
            row_result[f'{model_name}_L2_pred'] = l2
            row_result[f'{model_name}_L3_pred'] = l3
        results.append(row_result)
    print("✅ Predizioni completate.")
    
    df_results = pd.DataFrame(results)
    
    # --- 4. VALUTAZIONE E SALVATAGGIO FINALE ---
    performance_metrics = evaluate_and_get_metrics(df_results, df_test_labels)
    metrics_df = pd.DataFrame(performance_metrics).T
    print("\n--- RIEPILOGO METRICHE AGGREGATE ---")
    print(metrics_df.round(4).to_string())
    
    output_predictions_filename = os.path.join(output_metrics_dir, "predictions_and_evaluation_results.csv")
    output_metrics_filename = os.path.join(output_metrics_dir, "performance_metrics_summary.csv")
    
    df_test_labels['ground_truth_category'] = df_test_labels['category']
    df_final_output = pd.merge(df_results, df_test_labels[['filename', 'ground_truth_category']], on='filename')
    df_final_output.to_csv(output_predictions_filename, index=False)
    print(f"\n✅ Risultati finali (previsioni + ground truth) salvati in '{output_predictions_filename}'.")
    
    metrics_df.to_csv(output_metrics_filename, float_format='%.4f')
    print(f"✅ Riepilogo metriche di performance salvato in '{output_metrics_filename}'.")
    
    print("\n🎉 PROCESSO DI TRAINING E PREDIZIONE TERMINATO! 🎉")