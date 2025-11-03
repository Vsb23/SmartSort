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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
import pickle
import os

# --- FUNZIONI DI SUPPORTO: ONTOLOGIA E DATI (INVARIATE) ---

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

# --- FUNZIONI CSP (MODIFICATE PER CONTEGGIO FALLBACK) ---

def setup_csp_problem(doc_probs, ontology_graph, ns):
    problem = Problem()
    prob_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0.05}
    if not prob_l3:
        return None, doc_probs['L3'], True 

    possible_l2 = set()
    for l3_cat in prob_l3:
        possible_l2.update(get_parents(ontology_graph, ns, l3_cat))
    possible_l1 = set()
    for l2_cat in possible_l2:
        possible_l1.update(get_parents(ontology_graph, ns, l2_cat))

    if not possible_l1 or not possible_l2:
        return None, doc_probs['L3'], True

    valid_l1 = list(possible_l1 & doc_probs['L1'].keys())
    valid_l2 = list(possible_l2 & doc_probs['L2'].keys())
    valid_l3 = list(prob_l3.keys()) 

    if not valid_l1 or not valid_l2 or not valid_l3:
        return None, doc_probs['L3'], True

    problem.addVariable("L1Category", valid_l1)
    problem.addVariable("L2Category", valid_l2)
    problem.addVariable("L3Category", valid_l3)
    
    def hierarchical_constraint(l2_cat, l3_cat): return l2_cat in get_parents(ontology_graph, ns, l3_cat)
    def hierarchical_constraint_l1(l1_cat, l2_cat): return l1_cat in get_parents(ontology_graph, ns, l2_cat)
    
    problem.addConstraint(hierarchical_constraint, ("L2Category", "L3Category"))
    problem.addConstraint(hierarchical_constraint_l1, ("L1Category", "L2Category"))
    
    return problem, None, False

def find_best_csp_solution(problem_tuple, doc_probs, g, ns):
    problem, fallback_l3_probs, used_fallback_in_setup = problem_tuple

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
                return l1, l2, l3, probs_dict, False

    if fallback_l3_probs is None:
        fallback_l3_probs = doc_probs['L3']
    if not fallback_l3_probs:
         return "N/A", "N/A", "N/A", {}, True
         
    l3_pred = max(fallback_l3_probs, key=fallback_l3_probs.get)
    l2_parents = list(get_parents(g, ns, l3_pred))
    l2_pred = l2_parents[0] if l2_parents else "Altro"
    l1_parents = list(get_parents(g, ns, l2_pred))
    l1_pred = l1_parents[0] if l1_parents else "Altro"

    prob = doc_probs['L1'].get(l1_pred, 0) * doc_probs['L2'].get(l2_pred, 0) * doc_probs['L3'].get(l3_pred, 0)
    probs_dict = {'L1_pred': l1_pred, 'L2_pred': l2_pred, 'L3_pred': l3_pred, 'Combined_prob': prob}
    
    return l1_pred, l2_pred, l3_pred, probs_dict, True
    
# --- FUNZIONE PER VALUTAZIONE (USATA SOLO ALLA FINE) ---
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


def run_evaluation_on_test_set(test_data_path, test_labels_path, trained_models, vectorizer, ontology_keywords, g, ns, output_metrics_dir, suffix):
    """
    Esegue l'intera pipeline di predizione e valutazione su un dato test set.
    'suffix' (es. '_primo' o '_secondo') viene usato per salvare i file di output.
    """
    
    print("\n" + f"--- FASE DI VALUTAZIONE (Test Set: {suffix}) ---".center(80, "="))
    
    # --- 1. Caricamento Dati di Test ---
    try:
        df_test_data = pd.read_csv(test_data_path).fillna('')
        df_test_labels = pd.read_csv(test_labels_path).fillna('')
    except FileNotFoundError as e:
        print(f"❌ ERRORE: File di test non trovato: {e}.")
        print(f"   Impossibile valutare il set '{suffix}'. Salto...")
        return
    
    print(f"✅ File di test ({suffix}) caricati. Numero documenti: {len(df_test_data)}")

    # --- 2. Feature Engineering sul Test Set ---
    X_test_tfidf = vectorizer.transform(df_test_data['clean_text'])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(create_enhanced_features(df_test_data, ONTOLOGY_KEYWORDS).values)])
    
    results = []
    fallback_counts = {model_name: 0 for model_name in trained_models['L3'].keys()}
    total_test_samples = len(df_test_data)
    model_names = trained_models['L3'].keys()

    print(f"Inizio predizioni su {total_test_samples} documenti...")
    for i in range(total_test_samples):
        row_result = {'filename': df_test_data.iloc[i]['filename']}
        for model_name in model_names:
            
            # --- NOTA: Corretto un bug dal file originale ---
            # Il codice originale usava 'models' che non è definito qui.
            # Usiamo 'trained_models[level_name][model_name]'
            doc_probs = {
                level_name: dict(zip(
                    trained_models[level_name][model_name].classes_,
                    trained_models[level_name][model_name].predict_proba(X_test_combined[i])[0]
                )) for level_name in trained_models.keys()
            }
            
            problem_tuple = setup_csp_problem(doc_probs, g, ns)
            l1, l2, l3, _, used_fallback = find_best_csp_solution(problem_tuple, doc_probs, g, ns)
            
            if used_fallback:
                fallback_counts[model_name] += 1
            
            row_result[f'{model_name}_L1_pred'] = l1
            row_result[f'{model_name}_L2_pred'] = l2
            row_result[f'{model_name}_L3_pred'] = l3
        results.append(row_result)
    print("✅ Predizioni completate.")
    
    df_results = pd.DataFrame(results)
    
    # --- 3. Valutazione e Salvataggio Risultati ---
    performance_metrics = evaluate_and_get_metrics(df_results, df_test_labels)
    metrics_df = pd.DataFrame(performance_metrics).T
    print(f"\n--- RIEPILOGO METRICHE AGGREGATE (Test Set: {suffix}) ---")
    print(metrics_df.round(4).to_string())

    print(f"\n--- Statistiche Fallback (Test Set: {suffix}) ---")
    for model_name in model_names:
        perc = (fallback_counts[model_name] / total_test_samples) * 100
        print(f"Modello: {model_name:<5} | Fallback usati: {fallback_counts[model_name]}/{total_test_samples} ({perc:.2f}%)")
    
    # Salva i file usando il suffisso
    output_predictions_filename = os.path.join(output_metrics_dir, f"predictions_and_evaluation_results{suffix}.csv")
    output_metrics_filename = os.path.join(output_metrics_dir, f"performance_metrics_summary{suffix}.csv")
    
    df_test_labels['ground_truth_category'] = df_test_labels['category']
    df_final_output = pd.merge(df_results, df_test_labels[['filename', 'ground_truth_category']], on='filename')
    df_final_output.to_csv(output_predictions_filename, index=False)
    print(f"\n✅ Risultati finali (Test Set {suffix}) salvati in '{output_predictions_filename}'.")
    
    metrics_df.to_csv(output_metrics_filename, float_format='%.4f')
    print(f"✅ Riepilogo metriche (Test Set {suffix}) salvato in '{output_metrics_filename}'.")

# =================================================================================
# --- BLOCCO DI ESECUZIONE PRINCIPALE (CORRETTO) ---
# =================================================================================
if __name__ == "__main__":
    # --- 1. SETUP E CONFIGURAZIONE ---
    ontology_path = "Ontology.owx"
    
    # Questo percorso è CORRETTO e punta all'output di categorize_files.py
    train_csv_file = "training_result/training_set_categorized.csv"
    
    output_model_dir = "saved_models"
    output_metrics_dir = "metrics"
    force_retrain = False 
    min_samples_per_class = 5 
    max_tfidf_features = 30 
    N_SPLITS_KFOLD = 5

    print("🚀 AVVIO PIPELINE ML: ADDESTRAMENTO, PREDIZIONE E VALUTAZIONE 🚀")
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_metrics_dir, exist_ok=True)
    
    model_templates = {
        'LR': make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=0.1,max_iter=1000 )), 
        'RF': RandomForestClassifier(n_estimators=100  ), 
        'SVM': make_pipeline(StandardScaler(with_mean=False), SVC(probability=True)), 
        'NB': MultinomialNB()
    }
    levels = {'L1': None, 'L2': None, 'L3': None}
    
    g = Graph(); g.parse(ontology_path, format="xml")
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = estrai_category_keywords_da_ontologia(ontology_path)

    # --- 2. CARICAMENTO E PRE-PROCESSING DATI DI TRAINING ---
    print("\n" + "--- FASE 1: Caricamento e Processing Dati di Training ---".center(80, "="))
    df_train = load_training_data(train_csv_file)
    
    print(f"\n--- Filtro delle classi con meno di {min_samples_per_class} campioni ---")
    original_doc_count = len(df_train)
    category_counts = df_train['single_label'].value_counts()
    categories_to_keep = category_counts[category_counts >= min_samples_per_class].index.tolist()
    df_train = df_train[df_train['single_label'].isin(categories_to_keep)]
    print(f"Dataset di training ridotto a {len(df_train)} documenti.")

    vectorizer = TfidfVectorizer(max_features=max_tfidf_features, stop_words='english') 
    X_train_tfidf = vectorizer.fit_transform(df_train['clean_text'])
    X_train_combined = hstack([X_train_tfidf, csr_matrix(create_enhanced_features(df_train, ONTOLOGY_KEYWORDS).values)])
    
    df_train['l3_label'] = df_train['single_label']
    df_train['l2_label'] = df_train['l3_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else "Altro")
    df_train['l1_label'] = df_train['l2_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if x != "Altro" and get_parents(g, NS, x) else "Altro")
    
    y_train_l1 = df_train['l1_label'].values
    y_train_l2 = df_train['l2_label'].values
    y_train_l3 = df_train['l3_label'].values
    
    print("\nSalvataggio dati processati per stats.py (curva di loss)...")
    output_processed_data_dir = "processed_data"
    os.makedirs(output_processed_data_dir, exist_ok=True)
    X_train_path = os.path.join(output_processed_data_dir, 'X_train_combined.npz')
    y_train_path = os.path.join(output_processed_data_dir, 'y_train_L3.pkl')
    save_npz(X_train_path, X_train_combined)
    df_train['l3_label'].to_pickle(y_train_path) 
    print(f"✅ Dati processati (X e y) salvati in '{output_processed_data_dir}'.")

    with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'wb') as f: pickle.dump(vectorizer, f)
    print("✅ Vectorizer (basato sul training set completo) salvato.")


    # --- 3. K-FOLD CROSS-VALIDATION ---
    print("\n" + f"--- FASE 2: Esecuzione K-Fold Cross-Validation (K={N_SPLITS_KFOLD}) ---".center(80, "="))
    
    skf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True,  )
    fold_metrics = {model_name: [] for model_name in model_templates.keys()}
    fallback_counts = {model_name: 0 for model_name in model_templates.keys()}
    total_val_samples = 0
    
    for fold, (train_index, val_index) in enumerate(skf.split(X_train_combined, y_train_l3)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS_KFOLD} ---")
        X_train_fold, X_val_fold = X_train_combined[train_index], X_train_combined[val_index]
        y_val_true_l3 = y_train_l3[val_index]
        total_val_samples += len(y_val_true_l3)
        fold_models = {'L1': {}, 'L2': {}, 'L3': {}}
        
        for level_name, y_train in [('L1', y_train_l1), ('L2', y_train_l2), ('L3', y_train_l3)]:
            for model_name, model_template in model_templates.items():
                model_to_fit = clone(model_template)
                model_to_fit.fit(X_train_fold, y_train[train_index])
                fold_models[level_name][model_name] = model_to_fit
        
        print("Predizione su fold di validazione...")
        y_val_preds = {model_name: [] for model_name in model_templates.keys()}

        for i in range(X_val_fold.shape[0]):
            for model_name in model_templates.keys():
                doc_probs = {
                    level_name: dict(zip(
                        fold_models[level_name][model_name].classes_,
                        fold_models[level_name][model_name].predict_proba(X_val_fold[i])[0]
                    )) for level_name in levels.keys()
                }
                problem_tuple = setup_csp_problem(doc_probs, g, NS)
                l1, l2, l3, _, used_fallback = find_best_csp_solution(problem_tuple, doc_probs, g, NS)
                y_val_preds[model_name].append(l3)
                if used_fallback:
                    fallback_counts[model_name] += 1
        
        print("Metriche del Fold:")
        for model_name in model_templates.keys():
            f1 = f1_score(y_val_true_l3, y_val_preds[model_name], average='weighted', zero_division=0)
            fold_metrics[model_name].append(f1)
            print(f"  - {model_name} (F1-Weighted): {f1:.4f}")

    print("\n" + "--- Risultati K-Fold Cross-Validation (Medie) ---".center(80, "="))
    print("\nSalvataggio risultati K-Fold su file...")
    df_kfold_metrics = pd.DataFrame(fold_metrics)
    kfold_output_path = os.path.join(output_metrics_dir, "kfold_metrics_raw.csv")
    df_kfold_metrics.to_csv(kfold_output_path, index=False, float_format='%.4f')
    
    df_kfold_summary = pd.DataFrame(index=model_templates.keys())
    df_kfold_summary['F1_Mean'] = df_kfold_metrics.mean()
    df_kfold_summary['F1_StdDev'] = df_kfold_metrics.std()
    
    kfold_summary_path = os.path.join(output_metrics_dir, "kfold_metrics_summary.csv")
    df_kfold_summary.to_csv(kfold_summary_path, float_format='%.4f')
    print(f"✅ Riepilogo K-Fold (Media e Dev. Std) salvato in: {kfold_summary_path}")
    print("=" * 80)


    # --- 4. ADDESTRAMENTO FINALE SUL TRAINING SET COMPLETO ---
    print("\n" + "--- FASE 3: Addestramento Modelli Finali (su 100% Training Set) ---".center(80, "="))
    
    final_training_needed = force_retrain or not all(os.path.exists(os.path.join(output_model_dir, f)) for f in [f"model_{lvl}_{mdl}.pkl" for lvl in levels for mdl in model_templates])

    if not final_training_needed:
         print("✅ Modelli finali già presenti. Salto addestramento.")
         trained_models = {'L1': {}, 'L2': {}, 'L3': {}}
         for level in levels.keys():
             for model in model_templates.keys():
                 with open(os.path.join(output_model_dir, f"model_{level}_{model}.pkl"), 'rb') as f: trained_models[level][model] = pickle.load(f)
    else:
         print("Avvio addestramento modelli finali...")
         trained_models = {'L1': {}, 'L2': {}, 'L3': {}}
         for level_name, y_train in [('L1', y_train_l1), ('L2', y_train_l2), ('L3', y_train_l3)]:
             for model_name, model_template in model_templates.items():
                 print(f"Addestramento modello: {model_name} per Livello: {level_name}...")
                 model_to_fit = clone(model_template)
                 model_to_fit.fit(X_train_combined, y_train)
                 trained_models[level_name][model_name] = model_to_fit
                 with open(os.path.join(output_model_dir, f"model_{level_name}_{model_name}.pkl"), 'wb') as f: pickle.dump(model_to_fit, f)
         print("✅ Tutti i modelli finali sono stati addestrati e salvati.")

    # --- 5. PREDIZIONE E VALUTAZIONE SUI TEST SET ESTERNI ---
    
    print("\n" + "--- FASE 4&5: Avvio Valutazione sui Test Set Esterni ---".center(80, "="))
    
    print("\nCaricamento vectorizer finale...")
    try:
        with open(os.path.join(output_model_dir, 'vectorizer.pkl'), 'rb') as f: 
            vectorizer_final = pickle.load(f)
        print("✅ Vectorizer caricato.")
    except FileNotFoundError:
        print(f"❌ ERRORE CRITICO: Vectorizer non trovato in '{output_model_dir}'.")
        print("   Impossibile procedere con la valutazione dei test set.")
        trained_models = None 
    
    if trained_models:
        # --- Test Set 1 (Generale / test_result) ---
        run_evaluation_on_test_set(
              test_data_path="test_result/test_data_with_text.csv", 
              test_labels_path="test_result/test_set_categorized.csv", 
            
             trained_models=trained_models,
             vectorizer=vectorizer_final,
             ontology_keywords=ONTOLOGY_KEYWORDS,
             g=g, # <-- Passa 'g'
             ns=NS, # <-- Passa 'NS'
             output_metrics_dir=output_metrics_dir,
             suffix="_primo" # Questo crea i file ..._primo.csv
        )
        
         # --- Test Set 2 (Calcio / test_result_2) ---
        run_evaluation_on_test_set(
             test_data_path="test_result_2/test_data_2_with_text.csv", 
             test_labels_path="test_result_2/test_set_2_categorized.csv",
            
             trained_models=trained_models,
             vectorizer=vectorizer_final,
             ontology_keywords=ONTOLOGY_KEYWORDS,
             g=g, # <-- Passa 'g'
             ns=NS, # <-- Passa 'NS'
             output_metrics_dir=output_metrics_dir,
             suffix="_secondo" # Questo crea i file ..._secondo.csv
        )

        run_evaluation_on_test_set(
            test_data_path="test_result_3/test_data_3_with_text.csv", 
            test_labels_path="test_result_3/test_set_3_categorized.csv",
            
            trained_models=trained_models,
            vectorizer=vectorizer_final,
            ontology_keywords=ONTOLOGY_KEYWORDS,
            g=g, # <-- Passa 'g'
            ns=NS, # <-- Passa 'NS'
            output_metrics_dir=output_metrics_dir,
            suffix="_terzo" # Questo crea i file ... _terzo.csv
        )
    
    print("\n🎉 PROCESSO DI TRAINING E PREDIZIONE TERMINATO! 🎉")