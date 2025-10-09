import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from rdflib import Graph, Namespace, RDFS
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from constraint import Problem, BacktrackingSolver
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# (Le funzioni da 'load_keywords_from_ontology' a 'create_enhanced_features' rimangono invariate)
def load_keywords_from_ontology(ontology_path, ns):
    g = Graph()
    g.parse(ontology_path, format='xml')
    keywords_dict = defaultdict(list)
    hasKeyword_predicate = ns.hasKeyword
    for subject, _, obj in g.triples((None, hasKeyword_predicate, None)):
        class_name = str(subject).split('#')[-1]
        keywords_dict[class_name].append(str(obj))
    return dict(keywords_dict)

def get_parents(g, ns, child_name):
    parents = set()
    try:
        child_uri = ns[child_name]
        for _, _, o in g.triples((child_uri, RDFS.subClassOf, None)):
            parent_name = str(o).split('#')[-1]
            if parent_name != child_name and "Thing" not in parent_name:
                parents.add(parent_name)
    except KeyError: pass
    return parents

def parse_labels(label_str):
    if pd.isna(label_str) or label_str.strip() == '': return []
    return list(dict.fromkeys([lbl.strip() for lbl in str(label_str).split(';') if lbl.strip()]))

def load_training_data(csv_file, g, ns):
    df = pd.read_csv(csv_file, low_memory=False)
    df['text'] = df['clean_text'].fillna("")
    df['single_label'] = df['category'].apply(lambda x: parse_labels(x)[0] if parse_labels(x) else 'Altro')
    return df[df['single_label'].notna()].copy()

def create_enhanced_features(df, ontology_keywords):
    keyword_features = []
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        doc_features = {}
        for category, keywords in ontology_keywords.items():
            keyword_count = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', text)) for kw in keywords if kw)
            doc_features[f'{category}_keywords'] = keyword_count
        keyword_features.append(doc_features)
    return pd.DataFrame(keyword_features, index=df.index)

def setup_csp_problem(doc_probs, ontology_graph, ns):
    problem = Problem(BacktrackingSolver())
    probs_l1 = {k: v for k, v in doc_probs['L1'].items() if v > 0.01}
    probs_l2 = {k: v for k, v in doc_probs['L2'].items() if v > 0.01}
    probs_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0.01}
    if not probs_l1 or not probs_l2 or not probs_l3: return None
    problem.addVariable("L1_Category", list(probs_l1.keys()))
    problem.addVariable("L2_Category", list(probs_l2.keys()))
    problem.addVariable("L3_Category", list(probs_l3.keys()))
    def hierarchical_constraint(l1_cat, l2_cat, l3_cat):
        return l1_cat in get_parents(ontology_graph, ns, l2_cat) and \
               l2_cat in get_parents(ontology_graph, ns, l3_cat)
    problem.addConstraint(hierarchical_constraint, ("L1_Category", "L2_Category", "L3_Category"))
    return problem

def find_best_csp_solution(problem, doc_probs):
    if problem is None: return None, None, None, {}
    solutions = problem.getSolutions()
    if not solutions: return None, None, None, {}
    best_solution, max_prob = None, -1
    for sol in solutions:
        l1, l2, l3 = sol["L1_Category"], sol["L2_Category"], sol["L3_Category"]
        prob = doc_probs['L1'].get(l1, 0) * doc_probs['L2'].get(l2, 0) * doc_probs['L3'].get(l3, 0)
        if prob > max_prob:
            max_prob = prob
            best_solution = sol
    if best_solution:
        l1, l2, l3 = best_solution['L1_Category'], best_solution['L2_Category'], best_solution['L3_Category']
        probs_dict = {l1: doc_probs['L1'].get(l1, 0), l2: doc_probs['L2'].get(l2, 0), l3: doc_probs['L3'].get(l3, 0)}
        return l1, l2, l3, probs_dict
    return None, None, None, {}

def predict_for_validation(data_to_predict, models, vectorizer, g, ns):
    """Funzione di predizione semplificata, usata solo per la validazione k-fold."""
    keyword_features_df = create_enhanced_features(data_to_predict, ONTOLOGY_KEYWORDS)
    tfidf_features = vectorizer.transform(data_to_predict['text'])
    combined_features = hstack([tfidf_features, csr_matrix(keyword_features_df.values)])
    
    model_l1, model_l2, model_l3 = models['L1'], models['L2'], models['L3']
    
    probs_l1 = model_l1.predict_proba(combined_features)
    probs_l2 = model_l2.predict_proba(combined_features)
    probs_l3 = model_l3.predict_proba(combined_features)
    
    probs_dict_list = [
        {'L1': dict(zip(model_l1.classes_, p1)), 'L2': dict(zip(model_l2.classes_, p2)), 'L3': dict(zip(model_l3.classes_, p3))}
        for p1, p2, p3 in zip(probs_l1, probs_l2, probs_l3)
    ]
    
    final_predictions = []
    for doc_probs in probs_dict_list:
        problem = setup_csp_problem(doc_probs, g, NS)
        _, _, l3, _ = find_best_csp_solution(problem, doc_probs)
        final_predictions.append(l3 if l3 is not None else 'Altro')
        
    return final_predictions

if __name__ == "__main__":
    ontology_path = "./Ontology.owx"
    train_csv_file = "./training_data/training_set_85percent.csv"
    test_csv_file = "./test_data/test_set_15percent.csv"
    
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = load_keywords_from_ontology(ontology_path, NS)
    
    df_train = load_training_data(train_csv_file, g, NS)
    df_test = pd.read_csv(test_csv_file, low_memory=False)
    df_test['text'] = df_test['clean_text'].fillna("")

    train_keyword_features_df = create_enhanced_features(df_train, ONTOLOGY_KEYWORDS)
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(df_train['text'])
    X_train_combined = hstack([X_train_tfidf, csr_matrix(train_keyword_features_df.values)])
    y_train = df_train['single_label'].values
    
    df_train['l2_parent'] = df_train['single_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else None)
    df_train['l1_parent'] = df_train['l2_parent'].apply(lambda x: list(get_parents(g, NS, x))[0] if pd.notna(x) and get_parents(g, NS, x) else None)

    # --- K-FOLD CROSS-VALIDATION (per valutazione) ---
    print("\n--- 📊 ESECUZIONE K-FOLD CROSS-VALIDATION (K=5) SUL TRAINING SET ---")
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_index, val_index) in enumerate(skf.split(X_train_combined, y_train)):
        # ... (Questa sezione rimane identica, usa predict_for_validation) ...
        X_train_fold, X_val_fold = X_train_combined[train_index], X_train_combined[val_index]
        df_train_fold, df_val_fold = df_train.iloc[train_index], df_train.iloc[val_index]
        y_train_fold, y_val_fold_true = df_train_fold['single_label'].values, df_val_fold['single_label'].values
        
        y_train_l1_fold = df_train_fold.loc[df_train_fold['l1_parent'].notna(), 'l1_parent'].values
        X_train_l1_fold = X_train_fold[df_train_fold['l1_parent'].notna().values]
        y_train_l2_fold = df_train_fold.loc[df_train_fold['l2_parent'].notna(), 'l2_parent'].values
        X_train_l2_fold = X_train_fold[df_train_fold['l2_parent'].notna().values]
        
        model_l1 = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='lbfgs'))
        model_l2 = SVC(probability=True, random_state=42, class_weight='balanced')
        model_l3 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        if len(y_train_l1_fold) > 0: model_l1.fit(X_train_l1_fold, y_train_l1_fold)
        if len(y_train_l2_fold) > 0: model_l2.fit(X_train_l2_fold, y_train_l2_fold)
        model_l3.fit(X_train_fold, y_train_fold)

        models_fold = {'L1': model_l1, 'L2': model_l2, 'L3': model_l3}
        y_val_pred = predict_for_validation(df_val_fold, models_fold, vectorizer, g, NS)
        accuracy = accuracy_score(y_val_fold_true, y_val_pred)
        fold_accuracies.append(accuracy)
        print(f"Accuracy del Fold {fold + 1}/{n_splits}: {accuracy:.4f}")

    print(f"\n🎯 Accuracy Media sui {n_splits} fold: {np.mean(fold_accuracies):.4f} +/- {np.std(fold_accuracies):.4f}")

    # --- ADDESTRAMENTO FINALE SUL 100% DEI DATI ---
    print("\n\n--- 🚀 ADDESTRAMENTO MODELLO FINALE SUL 100% DEL TRAINING SET ---")
    l1_mask = df_train['l1_parent'].notna()
    X_train_l1, y_train_l1 = X_train_combined[l1_mask.values], df_train.loc[l1_mask, 'l1_parent'].values
    l2_mask = df_train['l2_parent'].notna()
    X_train_l2, y_train_l2 = X_train_combined[l2_mask.values], df_train.loc[l2_mask, 'l2_parent'].values
    
    model_l1_final = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, solver='lbfgs'))
    model_l2_final = SVC(probability=True, random_state=42, class_weight='balanced')
    model_l3_final = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    if len(y_train_l1) > 0: model_l1_final.fit(X_train_l1, y_train_l1)
    if len(y_train_l2) > 0: model_l2_final.fit(X_train_l2, y_train_l2)
    model_l3_final.fit(X_train_combined, y_train)

    # --- RISTABILITA LOGICA DI PREDIZIONE DETTAGLIATA PER IL TEST SET FINALE ---
    print("\n--- ⚙️ ESECUZIONE PREDIZIONI DETTAGLIATE SUL TEST SET FINALE ---")
    test_keyword_features_df = create_enhanced_features(df_test, ONTOLOGY_KEYWORDS)
    X_test_tfidf = vectorizer.transform(df_test['text'])
    X_test_combined = hstack([X_test_tfidf, csr_matrix(test_keyword_features_df.values)])
    
    probs_l1 = model_l1_final.predict_proba(X_test_combined)
    probs_l2 = model_l2_final.predict_proba(X_test_combined)
    probs_l3 = model_l3_final.predict_proba(X_test_combined)
    
    df_result = df_test[['filename']].copy()
    df_result['probs_l1'] = [dict(zip(model_l1_final.classes_, p)) for p in probs_l1]
    df_result['probs_l2'] = [dict(zip(model_l2_final.classes_, p)) for p in probs_l2]
    df_result['probs_l3'] = [dict(zip(model_l3_final.classes_, p)) for p in probs_l3]

    final_predictions, final_probs_list = [], []
    for _, row in df_result.iterrows():
        doc_probs = {'L1': row['probs_l1'], 'L2': row['probs_l2'], 'L3': row['probs_l3']}
        l1, l2, l3, final_probs = find_best_csp_solution(setup_csp_problem(doc_probs, g, NS), doc_probs)
        final_predictions.append({'L1_pred': l1, 'L2_pred': l2, 'L3_pred': l3})
        final_probs_list.append(final_probs)

    predictions_df = pd.DataFrame(final_predictions, index=df_result.index)
    df_result = pd.concat([df_result, predictions_df], axis=1)
    df_result['final_probs'] = final_probs_list
    
    df_result['L1_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L1_pred'], 0) * 100, axis=1)
    df_result['L2_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L2_pred'], 0) * 100, axis=1)
    df_result['L3_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L3_pred'], 0) * 100, axis=1)
    
    output_cols = ['filename', 'L1_pred', 'L1_prob', 'L2_pred', 'L2_prob', 'L3_pred', 'L3_prob']
    
    output_filename = 'predictions_on_test_set_detailed.csv'
    df_result.to_csv(output_filename, index=False, columns=output_cols)
    print(f"\n✅ Risultati finali dettagliati salvati in '{output_filename}'")
    print(df_result[output_cols].head().to_string(index=False))
    print("\n🎊 PROCESSO COMPLETATO!")