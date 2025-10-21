import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from rdflib import Graph, Namespace
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from constraint import Problem, BacktrackingSolver
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from collections import Counter




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



def plot_kfold_accuracies(folds, accuracy_svm, accuracy_nb, accuracy_ensemble):
    # Grafico a barre per confrontare le accuratezze dei modelli L3 su ogni fold
    x = np.array(folds)
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))

    bars1 = ax.bar(x - width, accuracy_svm, width, label='Accuracy L3 SVM')
    bars2 = ax.bar(x, accuracy_nb, width, label='Accuracy L3 NB')
    bars3 = ax.bar(x + width, accuracy_ensemble, width, label='Accuracy L3 Ensemble')

    ax.set_xlabel('Fold Number', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title('K-Fold Cross Validation Accuracy Scores Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Funzione per aggiungere valori sopra le barre
    def add_values(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', 
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=10)

    add_values(bars1)
    add_values(bars2)
    add_values(bars3)

    plt.tight_layout()
    
    # Salvataggio immagine
    output_folder = "test_result"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "kfold_accuracies_comparison.png")
    plt.savefig(output_path, dpi=150)
    print(f"Grafico salvato in: {output_path}")

def plot_loss_curve(X_train, y_train, output_dir):
    counter = Counter(y_train)
    rare_classes = {cls for cls, count in counter.items() if count < 2}
    y_train = np.array([cls if cls not in rare_classes else "Altro" for cls in y_train])
    
    X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    scaler = StandardScaler(with_mean=False)
    X_train_part_scaled = scaler.fit_transform(X_train_part)
    X_val_scaled = scaler.transform(X_val)

    model = SGDClassifier(
    loss='log_loss',
    random_state=42,
    learning_rate='adaptive',
    eta0=0.01,
    alpha=0.001,  # o puoi provare 0.01 se necessario
    max_iter=1,
    tol=None,
    warm_start=True
    )

    n_epochs = 50
    train_losses, val_losses = [] , []
    classes = np.unique(y_train)

    print("Addestramento iterativo per la curva di loss...")

    for epoch in range(n_epochs):
        model.partial_fit(X_train_part_scaled, y_train_part, classes=classes)
        train_prob = model.predict_proba(X_train_part_scaled)
        val_prob = model.predict_proba(X_val_scaled)
        # Sostituisce eventuali NaN e valori infiniti nelle probabilità con un valore molto piccolo per evitare errori nel calcolo della log loss e garantire stabilità numerica
        train_prob = np.nan_to_num(train_prob, nan=1e-15, posinf=1e-15, neginf=1e-15)
        val_prob = np.nan_to_num(val_prob, nan=1e-15, posinf=1e-15, neginf=1e-15)
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

    print(f"✅ Curva di Loss salvata in {filename}")


if __name__ == "__main__":

    # Percorsi file
    ontology_path = "Ontology.owx"
    train_csv_file = "training_result/training_set_categorized.csv"
    test_csv_file = "test_result/test_set_categorized.csv"

    print("Avvio del processo...")
    g = Graph()
    g.parse(ontology_path, format="xml")
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    ONTOLOGY_KEYWORDS = load_keywords_from_ontology(ontology_path, NS)

    # Caricamento e preparazione dati di training
    df_train = load_training_data(train_csv_file)
    df_train['l3_label'] = df_train['single_label']
    df_train['l2_label'] = df_train['l3_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else "Altro")
    df_train['l1_label'] = df_train['l2_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if x != "Altro" and get_parents(g, NS, x) else "Altro")

    # Creazione feature e matrice combinata
    vectorizer = TfidfVectorizer(max_features=45, stop_words='english')
    X_tfidf = vectorizer.fit_transform(df_train['clean_text'])
    keyword_features_df = create_enhanced_features(df_train, ONTOLOGY_KEYWORDS)
    X_combined = hstack([X_tfidf, csr_matrix(keyword_features_df.values)])

    # Variabili target
    y_l1 = df_train['l1_label'].values
    y_l2 = df_train['l2_label'].values
    y_l3 = df_train['l3_label'].values

    # Plot curva di apprendimento per L3
    output_dir = "test_result/images"
    os.makedirs(output_dir, exist_ok=True)
    plot_loss_curve(X_combined, y_l3, output_dir)

    # --- FASE 1: K-FOLD CROSS VALIDATION ---

    print("\n--- Inizio K-Fold Cross Validation su Training Set ---")
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    accuracies_l3_svm, accuracies_l3_nb, accuracies_l3_ensemble = [], [], []
    fold_no = 1
    for train_index, val_index in skf.split(X_combined, y_l3):

        print(f"\n-- Fold {fold_no} --")

        X_train, X_val = X_combined[train_index], X_combined[val_index]
        y_train_l1, y_val_l1 = y_l1[train_index], y_l1[val_index]
        y_train_l2, y_val_l2 = y_l2[train_index], y_l2[val_index]
        y_train_l3, y_val_l3 = y_l3[train_index], y_l3[val_index]

        pipeline = train_hierarchical_models(X_train, y_train_l1, y_train_l2, y_train_l3, g, NS)

        model_l1 = pipeline.models['L1']
        model_l2 = pipeline.models['L2']
        model_l3_svm = pipeline.models['L3_svm']
        model_l3_nb = pipeline.models['L3_nb']

        probs_l1 = model_l1.predict_proba(X_val)
        probs_l2 = model_l2.predict_proba(X_val)
        probs_l3_svm = model_l3_svm.predict_proba(X_val)
        probs_l3_nb = model_l3_nb.predict_proba(X_val)

        preds_svm, preds_nb, preds_ensemble = [], [], []

        for i in range(X_val.shape[0]):
            doc_probs_svm = {'L1': dict(zip(model_l1.classes_, probs_l1[i])),
                             'L2': dict(zip(model_l2.classes_, probs_l2[i])),
                             'L3': dict(zip(model_l3_svm.classes_, probs_l3_svm[i]))}
            _, _, _, final_probs_svm = find_best_csp_solution(setup_csp_problem(doc_probs_svm, g, NS), doc_probs_svm)

            doc_probs_nb = {'L1': dict(zip(model_l1.classes_, probs_l1[i])),
                            'L2': dict(zip(model_l2.classes_, probs_l2[i])),
                            'L3': dict(zip(model_l3_nb.classes_, probs_l3_nb[i]))}
            _, _, _, final_probs_nb = find_best_csp_solution(setup_csp_problem(doc_probs_nb, g, NS), doc_probs_nb)

            probs_l3_ensemble = (probs_l3_svm[i] + probs_l3_nb[i]) / 2
            doc_probs_ensemble = {'L1': dict(zip(model_l1.classes_, probs_l1[i])),
                                  'L2': dict(zip(model_l2.classes_, probs_l2[i])),
                                  'L3': dict(zip(model_l3_svm.classes_, probs_l3_ensemble))}
            _, _, _, final_probs_ensemble = find_best_csp_solution(setup_csp_problem(doc_probs_ensemble, g, NS), doc_probs_ensemble)

            preds_svm.append(final_probs_svm.get('L3_pred', "Altro"))
            preds_nb.append(final_probs_nb.get('L3_pred', "Altro"))
            preds_ensemble.append(final_probs_ensemble.get('L3_pred', "Altro"))

        accuracy_svm = accuracy_score(y_val_l3, preds_svm)
        accuracy_nb = accuracy_score(y_val_l3, preds_nb)
        accuracy_ensemble = accuracy_score(y_val_l3, preds_ensemble)

        print(f"Accuracy L3 SVM: {accuracy_svm:.4f}")
        print(f"Accuracy L3 NB: {accuracy_nb:.4f}")
        print(f"Accuracy L3 Ensemble: {accuracy_ensemble:.4f}")

        accuracies_l3_svm.append(accuracy_svm)
        accuracies_l3_nb.append(accuracy_nb)
        accuracies_l3_ensemble.append(accuracy_ensemble)

        fold_no += 1

    print("\n--- Media accuratezza K-Fold ---")
    print(f"Media Accuracy L3 SVM: {np.mean(accuracies_l3_svm):.4f}")
    print(f"Media Accuracy L3 NB: {np.mean(accuracies_l3_nb):.4f}")
    print(f"Media Accuracy L3 Ensemble: {np.mean(accuracies_l3_ensemble):.4f}")

    # --- FASE 2: TRAINING FINALE su tutto il training set e PREDIZIONE su test set esterno ---

    print("\n--- Training finale su tutto il training set ---")
    pipeline_finale = train_hierarchical_models(X_combined, y_l1, y_l2, y_l3, g, NS)

    model_l1_final = pipeline_finale.models['L1']
    model_l2_final = pipeline_finale.models['L2']
    model_l3_svm_final = pipeline_finale.models['L3_svm']
    model_l3_nb_final = pipeline_finale.models['L3_nb']

    print("\n--- Caricamento test set esterno ---")
    df_test = pd.read_csv(test_csv_file)
    df_test['clean_text'] = df_test['clean_text'].fillna('')

    X_test_tfidf = vectorizer.transform(df_test['clean_text'])
    test_keyword_features_df = create_enhanced_features(df_test, ONTOLOGY_KEYWORDS)
    X_test_combined = hstack([X_test_tfidf, csr_matrix(test_keyword_features_df.values)])

    probs_l1_test = model_l1_final.predict_proba(X_test_combined)
    probs_l2_test = model_l2_final.predict_proba(X_test_combined)
    probs_l3_svm_test = model_l3_svm_final.predict_proba(X_test_combined)
    probs_l3_nb_test = model_l3_nb_final.predict_proba(X_test_combined)

    final_results = []

    assert np.array_equal(model_l3_svm_final.classes_, model_l3_nb_final.classes_), "Le classi L3 dei modelli non corrispondono!"

    for i in range(len(df_test)):
        doc_probs_svm = {'L1': dict(zip(model_l1_final.classes_, probs_l1_test[i])),
                         'L2': dict(zip(model_l2_final.classes_, probs_l2_test[i])),
                         'L3': dict(zip(model_l3_svm_final.classes_, probs_l3_svm_test[i]))}
        _, _, _, final_probs_svm = find_best_csp_solution(setup_csp_problem(doc_probs_svm, g, NS), doc_probs_svm)

        doc_probs_nb = {'L1': dict(zip(model_l1_final.classes_, probs_l1_test[i])),
                        'L2': dict(zip(model_l2_final.classes_, probs_l2_test[i])),
                        'L3': dict(zip(model_l3_nb_final.classes_, probs_l3_nb_test[i]))}
        _, _, _, final_probs_nb = find_best_csp_solution(setup_csp_problem(doc_probs_nb, g, NS), doc_probs_nb)

        probs_l3_ensemble_test = (probs_l3_svm_test[i] + probs_l3_nb_test[i]) / 2
        doc_probs_ensemble = {'L1': dict(zip(model_l1_final.classes_, probs_l1_test[i])),
                              'L2': dict(zip(model_l2_final.classes_, probs_l2_test[i])),
                              'L3': dict(zip(model_l3_svm_final.classes_, probs_l3_ensemble_test))}
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
    output_cols = [
        'filename', 'L1_pred', 'L2_pred',
        'L3_pred_svm', 'L3_prob_svm',
        'L3_pred_nb', 'L3_prob_nb',
        'L3_pred_ensemble', 'L3_prob_ensemble'
    ]

    output_folder = "test_result"
    os.makedirs(output_folder, exist_ok=True)
    output_filename = "predictions_on_testset_full_comparison_gerarchy.csv"
    output_path = os.path.join(output_folder, output_filename)

    df_result.to_csv(output_path, index=False, columns=output_cols, float_format='%.4f')

    print(f"\nRisultati finali dettagliati salvati in {output_path}")
    print("--- Esempio di output ---")
    print(df_result[output_cols].head().to_string(index=False))

    print("\nPROCESSO COMPLETATO!")

    plot_kfold_accuracies([1, 2, 3], accuracies_l3_svm, accuracies_l3_nb, accuracies_l3_ensemble)