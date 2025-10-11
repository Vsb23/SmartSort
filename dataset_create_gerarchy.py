import pandas as pd
import numpy as np
import re
from rdflib import Graph, Namespace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack, csr_matrix
from constraint import Problem, BacktrackingSolver
from sklearn.base import BaseEstimator, ClassifierMixin

# (Le funzioni di supporto come load_keywords_from_ontology, get_parents, etc. rimangono invariate)
def load_keywords_from_ontology(ontology_path, ns):
    g = Graph()
    g.parse(ontology_path, format="xml")
    ontology_keywords = {}
    for s, p, o in g.triples((None, ns['hasKeyword'], None)):
        category_name = s.split('#')[-1]
        if category_name not in ontology_keywords:
            ontology_keywords[category_name] = []
        ontology_keywords[category_name].append(str(o))
    return ontology_keywords

def get_parents(g, ns, child_class_name):
    parents = set()
    if child_class_name == "Altro" or not child_class_name:
        return parents
    child_uri = ns[child_class_name]
    from rdflib.namespace import RDFS
    for s, p, o in g.triples((child_uri, RDFS.subClassOf, None)):
        if '#' in o:
            parents.add(o.split('#')[-1])
    return parents

def parse_labels(label_str):
    if pd.isna(label_str) or not label_str.strip():
        return []
    return list(dict.fromkeys(lbl.strip() for lbl in str(label_str).split() if lbl.strip()))

def load_training_data(csv_file):
    df = pd.read_csv(csv_file, low_memory=False)
    df['clean_text'] = df['clean_text'].fillna('')
    df['single_label'] = df['category'].apply(lambda x: parse_labels(x)[0] if parse_labels(x) else "Altro")
    return df[df['single_label'].notna()].copy()

def create_enhanced_features(df, ontology_keywords):
    keyword_features = []
    for _, row in df.iterrows():
        text = str(row['clean_text']).lower()
        doc_features = {}
        for category, keywords in ontology_keywords.items():
            keyword_count = sum(len(re.findall(r'\b' + re.escape(kw.lower()) + r'\b', text)) for kw in keywords)
            doc_features[f"{category}_keywords"] = keyword_count
        keyword_features.append(doc_features)
    return pd.DataFrame(keyword_features, index=df.index).fillna(0)

def setup_csp_problem(doc_probs, ontology_graph, ns):
    problem = Problem(BacktrackingSolver())
    prob_l1 = {k: v for k, v in doc_probs['L1'].items() if v > 0.01}
    prob_l2 = {k: v for k, v in doc_probs['L2'].items() if v > 0.01}
    prob_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0.01}
    if not prob_l1 or not prob_l2 or not prob_l3: return None
    problem.addVariable("L1Category", list(prob_l1.keys()))
    problem.addVariable("L2Category", list(prob_l2.keys()))
    problem.addVariable("L3Category", list(prob_l3.keys()))
    def hierarchical_constraint(l1_cat, l2_cat, l3_cat):
        l2_parents = get_parents(ontology_graph, ns, l2_cat)
        l3_parents = get_parents(ontology_graph, ns, l3_cat)
        return l1_cat in l2_parents and l2_cat in l3_parents
    problem.addConstraint(hierarchical_constraint, ("L1Category", "L2Category", "L3Category"))
    return problem

def find_best_csp_solution(problem, doc_probs):
    if problem is None: return None, None, None, {}
    solutions = problem.getSolutions()
    if not solutions: return None, None, None, {}
    best_solution = None
    max_prob = -1
    for sol in solutions:
        l1, l2, l3 = sol["L1Category"], sol["L2Category"], sol["L3Category"]
        prob = doc_probs['L1'].get(l1, 0) * doc_probs['L2'].get(l2, 0) * doc_probs['L3'].get(l3, 0)
        if prob > max_prob:
            max_prob = prob
            best_solution = sol
    if best_solution:
        l1, l2, l3 = best_solution["L1Category"], best_solution["L2Category"], best_solution["L3Category"]
        probs_dict = {
            'L1_pred': l1, 'L1_prob': doc_probs['L1'].get(l1, 0),
            'L2_pred': l2, 'L2_prob': doc_probs['L2'].get(l2, 0),
            'L3_pred': l3, 'L3_prob': doc_probs['L3'].get(l3, 0)
        }
        return l1, l2, l3, probs_dict
    return None, None, None, {}

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper per classificatori che applica vincoli gerarchici durante il training
    """
    def __init__(self, base_classifier, ontology_graph, namespace, level):
        self.base_classifier = base_classifier
        self.ontology_graph = ontology_graph
        self.namespace = namespace
        self.level = level
    
    def fit(self, X, y, parent_predictions=None):
        if parent_predictions is not None and self.level > 1:
            # Filtra training set per mantenere solo esempi coerenti
            consistent_mask = self._get_consistent_samples(y, parent_predictions)
            if np.sum(consistent_mask) > 0:  # Se ci sono campioni coerenti
                X_filtered = X[consistent_mask]
                y_filtered = y[consistent_mask]
                print(f"Training L{self.level}: {len(y)} -> {len(y_filtered)} campioni (filtrati per coerenza)")
                self.base_classifier.fit(X_filtered, y_filtered)
            else:
                print(f"Training L{self.level}: Nessun campione coerente trovato, training normale")
                self.base_classifier.fit(X, y)
        else:
            self.base_classifier.fit(X, y)
        return self
    
    def predict(self, X):
        return self.base_classifier.predict(X)
    
    def predict_proba(self, X):
        return self.base_classifier.predict_proba(X)
    
    def _get_consistent_samples(self, y_current, parent_predictions):
        consistent_mask = np.ones(len(y_current), dtype=bool)
        
        for i, (current_label, parent_label) in enumerate(zip(y_current, parent_predictions)):
            if not self._is_child_of(current_label, parent_label):
                consistent_mask[i] = False
        return consistent_mask
    
    def _is_child_of(self, child_category, parent_category):
        if child_category == "Altro" or parent_category == "Altro":
            return True  # Permetti "Altro" 
        parents = get_parents(self.ontology_graph, self.namespace, child_category)
        return parent_category in parents
    @property
    def classes_(self):
        """Fornisce accesso alle classi del classificatore base"""
        return self.base_classifier.classes_

class HierarchicalTrainingPipeline:
    """
    Pipeline per training gerarchico con vincoli progressivi - SVM + NB per L3
    """
    def __init__(self, ontology_graph, namespace):
        self.ontology_graph = ontology_graph
        self.namespace = namespace
        self.models = {}
        
    def fit_hierarchical(self, X_train, y_train_l1, y_train_l2, y_train_l3, 
                        base_classifier_l1, base_classifier_l2, 
                        base_classifier_l3_svm, base_classifier_l3_nb):
        
        print("=== TRAINING GERARCHICO CON VINCOLI (SVM + NB per L3) ===")
        
        # 1. Addestra L1 normalmente
        print("Training L1...")
        self.models['L1'] = HierarchicalClassifier(
            base_classifier_l1, self.ontology_graph, self.namespace, level=1
        )
        self.models['L1'].fit(X_train, y_train_l1)
        
        # 2. Predici L1 per guidare L2
        l1_predictions = self.models['L1'].predict(X_train)
        
        # 3. Addestra L2 con vincoli L1
        print("Training L2 con vincoli L1...")
        self.models['L2'] = HierarchicalClassifier(
            base_classifier_l2, self.ontology_graph, self.namespace, level=2
        )
        self.models['L2'].fit(X_train, y_train_l2, parent_predictions=l1_predictions)
        
        # 4. Predici L2 per guidare L3
        l2_predictions = self.models['L2'].predict(X_train)
        
        # 5. Addestra ENTRAMBI i modelli L3 con vincoli L2
        print("Training L3 SVM con vincoli L2...")
        self.models['L3_svm'] = HierarchicalClassifier(
            base_classifier_l3_svm, self.ontology_graph, self.namespace, level=3
        )
        self.models['L3_svm'].fit(X_train, y_train_l3, parent_predictions=l2_predictions)
        
        print("Training L3 Naive Bayes con vincoli L2...")
        self.models['L3_nb'] = HierarchicalClassifier(
            base_classifier_l3_nb, self.ontology_graph, self.namespace, level=3
        )
        self.models['L3_nb'].fit(X_train, y_train_l3, parent_predictions=l2_predictions)
        
        return self




def train_hierarchical_models(X_train_combined, y_train_l1, y_train_l2, y_train_l3, g, NS):
    """
    Training gerarchico migliorato - SVM + NB entrambi con vincoli
    """
    
    # Training gerarchico completo con SVM + NB per L3
    pipeline = HierarchicalTrainingPipeline(g, NS)
    
    # Definisci i classificatori base
    base_l1 = make_pipeline(StandardScaler(with_mean=False), 
                           LogisticRegression(max_iter=1000, random_state=42))
    base_l2 = RandomForestClassifier(n_estimators=100, random_state=42)
    base_l3_svm = SVC(probability=True, random_state=42)
    base_l3_nb = MultinomialNB()  # NB non ha bisogno di scaling
    
    # Training gerarchico con entrambi i modelli per L3
    pipeline.fit_hierarchical(X_train_combined, y_train_l1, y_train_l2, y_train_l3,
                             base_l1, base_l2, base_l3_svm, base_l3_nb)
    
    return pipeline

if __name__ == "__main__":
    
    ontology_path = "Ontology.owx"
    train_csv_file = "training_result/training_set_categorized.csv"
    test_csv_file = "test_result/test_data_with_text.csv"

    
    print("Avvio del processo...")
    g = Graph()
    g.parse(ontology_path, format="xml")
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = load_keywords_from_ontology(ontology_path, NS)
    
    df_train = load_training_data(train_csv_file)
    df_test = pd.read_csv(test_csv_file)
    df_test['clean_text'] = df_test['clean_text'].fillna('')

    print("Creazione delle feature...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(df_train['clean_text'])
    train_keyword_features_df = create_enhanced_features(df_train, ONTOLOGY_KEYWORDS)
    X_train_combined = hstack([X_train_tfidf, csr_matrix(train_keyword_features_df.values)])
    
    df_train['l3_label'] = df_train['single_label']
    df_train['l2_label'] = df_train['l3_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else "Altro")
    df_train['l1_label'] = df_train['l2_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if x != "Altro" and get_parents(g, NS, x) else "Altro")
    
    y_train_l1 = df_train['l1_label'].values
    y_train_l2 = df_train['l2_label'].values
    y_train_l3 = df_train['l3_label'].values


    # Addestramento modelli gerarchici
    print("--- Addestramento Modelli Finali con Vincoli Gerarchici ---")
    hierarchical_pipeline = train_hierarchical_models(
        X_train_combined, y_train_l1, y_train_l2, y_train_l3, g, NS
    )

    # Estrazione modelli finali
    model_l1_final = hierarchical_pipeline.models['L1']
    model_l2_final = hierarchical_pipeline.models['L2'] 
    model_l3_svm_final = hierarchical_pipeline.models['L3_svm']
    model_l3_nb_final = hierarchical_pipeline.models['L3_nb']  # NB ora con vincoli

    print("Tutti i modelli finali sono stati addestrati con vincoli gerarchici (incluso NB per L3).")

    print("\n--- Predizioni sul Test Set Finale (Comparazione + Ensemble) ---")
    X_test_tfidf = vectorizer.transform(df_test['clean_text'])
    test_keyword_features_df = create_enhanced_features(df_test, ONTOLOGY_KEYWORDS)
    X_test_combined = hstack([X_test_tfidf, csr_matrix(test_keyword_features_df.values)])
    
    probs_l1 = model_l1_final.predict_proba(X_test_combined)
    probs_l2 = model_l2_final.predict_proba(X_test_combined)
    probs_l3_svm = model_l3_svm_final.predict_proba(X_test_combined)
    probs_l3_nb = model_l3_nb_final.predict_proba(X_test_combined)
    
    final_results = []
    
    assert np.array_equal(model_l3_svm_final.classes_, model_l3_nb_final.classes_), "Le classi dei modelli L3 non corrispondono!"

    for i in range(len(df_test)):
        # Pipeline 1: SVM
        doc_probs_svm = {
            'L1': dict(zip(model_l1_final.classes_, probs_l1[i])), 
            'L2': dict(zip(model_l2_final.classes_, probs_l2[i])), 
            'L3': dict(zip(model_l3_svm_final.classes_, probs_l3_svm[i]))
        }
        _, _, _, final_probs_svm = find_best_csp_solution(setup_csp_problem(doc_probs_svm, g, NS), doc_probs_svm)

        # Pipeline 2: Naive Bayes
        doc_probs_nb = {
            'L1': dict(zip(model_l1_final.classes_, probs_l1[i])), 
            'L2': dict(zip(model_l2_final.classes_, probs_l2[i])), 
            'L3': dict(zip(model_l3_nb_final.classes_, probs_l3_nb[i]))
        }
        _, _, _, final_probs_nb = find_best_csp_solution(setup_csp_problem(doc_probs_nb, g, NS), doc_probs_nb)

        # Pipeline Ensemble
        probs_l3_ensemble = (probs_l3_svm[i] + probs_l3_nb[i]) / 2.0
        doc_probs_ensemble = {
            'L1': dict(zip(model_l1_final.classes_, probs_l1[i])), 
            'L2': dict(zip(model_l2_final.classes_, probs_l2[i])), 
            'L3': dict(zip(model_l3_svm_final.classes_, probs_l3_ensemble))
        }
        _, _, _, final_probs_ensemble = find_best_csp_solution(setup_csp_problem(doc_probs_ensemble, g, NS), doc_probs_ensemble)

        result_row = {
            'filename': df_test.iloc[i]['filename'],
            'L1_pred': final_probs_ensemble.get('L1_pred', "N/A"),
            'L2_pred': final_probs_ensemble.get('L2_pred', "N/A"),
            'L3_pred_svm': final_probs_svm.get('L3_pred', "Altro"),
            'L3_prob_svm': final_probs_svm.get('L3_prob', 0.0),
            'L3_pred_nb': final_probs_nb.get('L3_pred', "Altro"),
            'L3_prob_nb': final_probs_nb.get('L3_prob', 0.0),
            'L3_pred_ensemble': final_probs_ensemble.get('L3_pred', "Altro"),
            'L3_prob_ensemble': final_probs_ensemble.get('L3_prob', 0.0),
        }
        final_results.append(result_row)

    df_result = pd.DataFrame(final_results)

    from sklearn.metrics import precision_recall_fscore_support

    # Estrai etichette vere dal test set (se presenti)
    y_true_l1 = df_test['l1_label'].values if 'l1_label' in df_test.columns else None
    y_true_l2 = df_test['l2_label'].values if 'l2_label' in df_test.columns else None
    y_true_l3 = df_test['single_label'].values if 'single_label' in df_test.columns else None

    # Ricava le etichette predette da final_results
    y_pred_l1 = df_result['L1_pred'].values
    y_pred_l2 = df_result['L2_pred'].values
    y_pred_l3_svm = df_result['L3_pred_svm'].values
    y_pred_l3_nb = df_result['L3_pred_nb'].values
    y_pred_l3_ensemble = df_result['L3_pred_ensemble'].values

    def evaluate_classification(y_true, y_pred):
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return precision, recall, f1

    output_cols = ['filename', 'L1_pred', 'L2_pred',
                   'L3_pred_svm', 'L3_prob_svm',
                   'L3_pred_nb', 'L3_prob_nb',
                   'L3_pred_ensemble', 'L3_prob_ensemble']
    output_filename = "predictions_on_testset_full_comparison.csv"

    df_result.to_csv(output_filename, index=False,
                     columns=output_cols, float_format='%.4f')

    print(f"\nRisultati finali dettagliati salvati in {output_filename}")
    print("--- Esempio di output ---")
    print(df_result[output_cols].head().to_string(index=False))
    print("\nPROCESSO COMPLETATO!")
