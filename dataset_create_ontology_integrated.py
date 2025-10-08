import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, accuracy_score
from rdflib import Graph, Namespace, RDFS
import numpy as np
from collections import Counter, defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from constraint import Problem, BacktrackingSolver

def load_keywords_from_ontology(ontology_path, ns):
    """
    Carica keywords direttamente dall'ontologia OWL
    """
    print("📖 Caricamento keywords dall'ontologia...")
    
    g = Graph()
    g.parse(ontology_path, format='xml')
    
    query = """
    PREFIX ns: <http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#>
    
    SELECT ?class ?keyword WHERE {
        ?class ns:hasKeyword ?keyword .
    }
    """
    
    keywords_dict = defaultdict(list)
    
    try:
        results = g.query(query)
        
        for row in results:
            class_uri = str(row['class'])
            keyword = str(row['keyword'])
            
            class_name = class_uri.split('#')[-1]
            keywords_dict[class_name].append(keyword)
        
        print(f"✅ Keywords caricate per {len(keywords_dict)} categorie")
        
        for category, keywords in list(keywords_dict.items())[:5]:
            print(f"  {category}: {len(keywords)} keywords")
        
    except Exception as e:
        print(f"⚠️ Errore nell'estrazione SPARQL: {e}")
        print("📝 Usando metodo alternativo...")
        
        for subject, predicate, obj in g:
            if str(predicate).endswith('#hasKeyword'):
                class_name = str(subject).split('#')[-1]
                keyword = str(obj)
                keywords_dict[class_name].append(keyword)
        
        print(f"✅ Keywords caricate (metodo alternativo) per {len(keywords_dict)} categorie")
    
    keywords_dict = dict(keywords_dict)
    if 'Altro' not in keywords_dict:
        keywords_dict['Altro'] = []
    
    return keywords_dict

def get_all_subclasses(g, base_class_uri):
    """Ottiene tutte le sottoclassi di una classe base"""
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))
    
    for subclass in direct_subclasses:
        subclass_name = str(subclass).split('#')[-1]
        subclasses.add(subclass_name)
        subclasses |= get_all_subclasses(g, subclass)
    
    subclasses.add('Altro')
    return subclasses

def get_subclasses_by_level(g, base_class_uri):
    """
    Ottiene tutte le sottoclassi di una classe base,
    organizzate per livello gerarchico.
    """
    levels = defaultdict(set)
    queue = [(base_class_uri, 0)]
    visited = set()
    
    while queue:
        current_uri, level = queue.pop(0)
        
        if current_uri in visited:
            continue
        visited.add(current_uri)
        
        levels[level].add(str(current_uri).split('#')[-1])
        
        direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, current_uri)))
        for subclass in direct_subclasses:
            queue.append((subclass, level + 1))
            
    return levels

def get_parents(g, ns, child_name):
    """Ottiene i genitori di una classe nell'ontologia."""
    parents = set()
    try:
        child_uri = ns[child_name]
        for s, p, o in g.triples((child_uri, RDFS.subClassOf, None)):
            parent_name = str(o).split('#')[-1]
            if parent_name != child_name:
                parents.add(parent_name)
    except:
        pass
    return parents

def get_most_specific_category(labels, g, ns):
    """Restituisce SOLO la categoria più specifica"""
    if not labels:
        return ['Altro']
    
    if len(labels) == 1 and labels[0] == 'Altro':
        return ['Altro']
    
    filtered_labels = [label for label in labels if label != 'Altro']
    if not filtered_labels:
        return ['Altro']
    
    most_specific = []
    
    for label in filtered_labels:
        try:
            subclasses = get_all_subclasses(g, ns[label])
            has_subclass_in_labels = any(sub in filtered_labels for sub in subclasses if sub != label and sub != 'Altro')
            
            if not has_subclass_in_labels:
                most_specific.append(label)
        except:
            most_specific.append(label)
    
    return most_specific[:1] if most_specific else [filtered_labels[0]]

def parse_labels(label_str):
    """Parsa le etichette da stringa a lista"""
    if pd.isna(label_str):
        return []
    labels = [lbl.strip() for lbl in str(label_str).split(';')]
    seen = set()
    unique_labels = []
    for lbl in labels:
        if lbl and lbl not in seen:
            seen.add(lbl)
            unique_labels.append(lbl)
    return unique_labels

def load_dataset_with_single_category(csv_file, g, ns):
    """
    Carica dataset con UNA SOLA categoria per documento,
    usando la colonna 'clean_text' già presente nel CSV.
    """
    df = pd.read_csv(csv_file, low_memory=False)

    df['text'] = df['clean_text'].fillna("")

    single_labels = []
    for idx, row in df.iterrows():
        raw_labels = parse_labels(row['category'])
        specific_label = get_most_specific_category(raw_labels, g, ns)
        single_labels.append(specific_label[0] if specific_label else None)

    df['single_label'] = single_labels
    df_train = df[df['single_label'].notna()].copy()

    return df_train

def train_eval_single_label_model(name, clf, X_train, y_train, X_val, y_val, all_labels):
    """Addestra e valuta modello per classificazione SINGLE-LABEL"""
    print(f"---- Training {name} (Single-Label) ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 weighted: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=None, zero_division=0))
    return clf

def balance_single_label_dataset(df, label_col='single_label', max_samples_per_class=100):
    """Bilancia dataset per classificazione single-label"""
    counter = Counter(df[label_col])
    print(f"📊 Distribuzione originale:")
    for label, count in counter.items():
        print(f"  {label}: {count}")
    
    balanced_indices = []
    class_counts = Counter()
    
    sorted_indices = df.index.tolist()
    np.random.shuffle(sorted_indices)
    
    for idx in sorted_indices:
        label = df.loc[idx, label_col]
        if class_counts[label] < max_samples_per_class:
            balanced_indices.append(idx)
            class_counts[label] += 1
    
    df_balanced = df.loc[balanced_indices].copy()
    
    print(f"📊 Distribuzione bilanciata:")
    counter_balanced = Counter(df_balanced[label_col])
    for label, count in counter_balanced.items():
        print(f"  {label}: {count}")
    
    return df_balanced

def create_enhanced_features(df, ontology_keywords):
    """Crea features avanzate basate su keywords ontologiche"""
    print("🔧 Creazione features avanzate basate su ontologia...")
    
    keyword_features = []
    
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        doc_features = {}
        
        for category, keywords in ontology_keywords.items():
            keyword_count = 0
            for keyword in keywords:
                if keyword:
                    if ' ' in keyword:
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = len(re.findall(pattern, text))
                        keyword_count += matches * len(keyword.split())
                    else:
                        pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        matches = len(re.findall(pattern, text))
                        keyword_count += matches
            
            doc_features[f'{category}_keywords'] = keyword_count
        
        keyword_features.append(doc_features)
    
    return pd.DataFrame(keyword_features)

def setup_csp_problem(doc_probs, ontology_graph, ns):
    """
    Imposta il problema CSP per la post-elaborazione.
    """
    problem = Problem(BacktrackingSolver())
    
    probs_l1 = {k: v for k, v in doc_probs['L1'].items() if v > 0}
    probs_l2 = {k: v for k, v in doc_probs['L2'].items() if v > 0}
    probs_l3 = {k: v for k, v in doc_probs['L3'].items() if v > 0}

    problem.addVariable("L1_Category", list(probs_l1.keys()))
    problem.addVariable("L2_Category", list(probs_l2.keys()))
    problem.addVariable("L3_Category", list(probs_l3.keys()))
    
    def hierarchical_constraint(l1_cat, l2_cat, l3_cat):
        parents_l2 = get_parents(ontology_graph, ns, l2_cat)
        if l1_cat not in parents_l2:
            return False
            
        parents_l3 = get_parents(ontology_graph, ns, l3_cat)
        if l2_cat not in parents_l3:
            return False
            
        return True

    problem.addConstraint(hierarchical_constraint, ("L1_Category", "L2_Category", "L3_Category"))
    
    def probability_constraint(l1_cat, l2_cat, l3_cat):
        prob_l1 = probs_l1.get(l1_cat, 0)
        prob_l2 = probs_l2.get(l2_cat, 0)
        prob_l3 = probs_l3.get(l3_cat, 0)
        
        return (prob_l1 * prob_l2 * prob_l3) > 0

    problem.addConstraint(probability_constraint, ("L1_Category", "L2_Category", "L3_Category"))
    
    return problem

def find_best_csp_solution(problem, doc_probs):
    """
    Risolve il problema CSP per trovare la soluzione ottimale.
    """
    solutions = problem.getSolutions()
    if not solutions:
        return None, None, None, {}
    
    best_solution = None
    max_prob = -1
    
    for sol in solutions:
        l1_cat = sol["L1_Category"]
        l2_cat = sol["L2_Category"]
        l3_cat = sol["L3_Category"]
        
        prob = doc_probs['L1'].get(l1_cat, 0) * doc_probs['L2'].get(l2_cat, 0) * doc_probs['L3'].get(l3_cat, 0)
        
        if prob > max_prob:
            max_prob = prob
            best_solution = sol
            
    if best_solution:
        l1 = best_solution['L1_Category']
        l2 = best_solution['L2_Category']
        l3 = best_solution['L3_Category']
        
        probs_dict = {
            f'{l1}': doc_probs['L1'].get(l1, 0),
            f'{l2}': doc_probs['L2'].get(l2, 0),
            f'{l3}': doc_probs['L3'].get(l3, 0)
        }
        
        return l1, l2, l3, probs_dict
        
    return None, None, None, {}

if __name__ == "__main__":
    # Configurazione
    ontology_path = "./Ontology.owx"
    csv_file = "./training_set_single_category_85percent.csv"
    
    print("🚀 DATASET CREATION CON KEYWORDS DALL'ONTOLOGIA")
    print("=" * 60)
    
    # Carica ontologia
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    
    ONTOLOGY_KEYWORDS = load_keywords_from_ontology(ontology_path, NS)
    
    if not ONTOLOGY_KEYWORDS:
        print("⚠️ Nessuna keyword trovata nell'ontologia!")
        print("📝 Verifica che l'ontologia contenga le hasKeyword properties")
        print("🔧 Usando fallback keywords minimali...")
        ONTOLOGY_KEYWORDS = {
            'Informatica': ['computer', 'software', 'algorithm'],
            'AI_ML': ['machine learning', 'artificial intelligence'],
            'Altro': []
        }
        
    # Ottieni tutte le categorie dall'ontologia e definisci i livelli
    levels = get_subclasses_by_level(g, NS['Scienza'])
    levels_humanities = get_subclasses_by_level(g, NS['Studi_umanistici'])
    for level, classes in levels_humanities.items():
        levels[level] |= classes
        
    all_l1_labels = sorted(list(levels.get(0, set())))
    all_l2_labels = sorted(list(levels.get(1, set())))
    all_l3_labels = sorted(list(levels.get(2, set())))

    print(f"📊 Categorie L1: {all_l1_labels}")
    print(f"📊 Categorie L2: {all_l2_labels}")
    print(f"📊 Categorie L3: {all_l3_labels}")
    
    df_train = load_dataset_with_single_category(csv_file, g, NS)
    print(f"📁 Dataset caricato: {len(df_train)} documenti")
    
    df_train_balanced = balance_single_label_dataset(df_train, 'single_label', max_samples_per_class=50)
    df_train_balanced = df_train_balanced.reset_index(drop=True)
    
    keyword_features_df = create_enhanced_features(df_train_balanced, ONTOLOGY_KEYWORDS)
    keyword_features_df = keyword_features_df.reset_index(drop=True)
    
    print(f"⚖️ Dataset bilanciato: {len(df_train_balanced)} documenti")
    print(f"🎯 Features ontologiche create: {keyword_features_df.shape[1]} features")
    
    # Split train/validation - SINGLE LABEL
    if len(df_train_balanced) >= 10:
        unique_labels = df_train_balanced['single_label'].unique()
        can_stratify = all(sum(df_train_balanced['single_label'] == label) >= 2 for label in unique_labels)
        
        if can_stratify and len(unique_labels) > 1:
            train, val = train_test_split(df_train_balanced, test_size=0.2, random_state=42,
                                         stratify=df_train_balanced['single_label'])
        else:
            train, val = train_test_split(df_train_balanced, test_size=0.2, random_state=42)
    else:
        print("⚠️ Dataset troppo piccolo per il training")
        train, val = df_train_balanced, df_train_balanced.sample(frac=0.1)
    
    # Reset degli indici per i DataFrame di training e validazione
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    # Vettorizzazione TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(train['text'])
    X_val_tfidf = vectorizer.transform(val['text'])
    
    # Seleziona features ontologiche
    X_train_keywords = keyword_features_df.iloc[train.index].values
    X_val_keywords = keyword_features_df.iloc[val.index].values
    
    # Combina TF-IDF + Keywords ontologiche
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_keywords)])
    X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_keywords)])
    
    # Labels
    y_train = train['single_label'].values
    y_val = val['single_label'].values
    
    print("\n🚀 ADDESTRAMENTO MODELLO PER LIVELLO DI CLASSIFICAZIONE:")
    print("=" * 60)
    
    # Addestra modelli separati per ogni livello
    model_l1 = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(max_iter=1000, solver='lbfgs')
    )
    
    model_l2 = SVC(probability=True, random_state=42)
    model_l3 = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Prepara le etichette per l'addestramento dei modelli di livello superiore
    # CORREZIONE: Assegna le etichette genitore corrette per ogni livello
    train['l1_parent'] = train['single_label'].apply(lambda x: get_parents(g, NS, list(get_parents(g, NS, x))[0]).pop() if get_parents(g, NS, x) and get_parents(g, NS, list(get_parents(g, NS, x))[0]) else None)
    train['l2_parent'] = train['single_label'].apply(lambda x: list(get_parents(g, NS, x))[0] if get_parents(g, NS, x) else None)
    
    # Filtra il training set per rimuovere i valori 'None' nelle etichette genitore
    
    # Filtro per il modello L1
    l1_mask = train['l1_parent'].notna()
    X_train_l1 = X_train_combined[l1_mask.values]
    y_train_l1 = train[l1_mask]['l1_parent'].values
    
    # Filtro per il modello L2
    l2_mask = train['l2_parent'].notna()
    X_train_l2 = X_train_combined[l2_mask.values]
    y_train_l2 = train[l2_mask]['l2_parent'].values
    
    # Per il modello L3, usiamo tutti i dati di training
    X_train_l3 = X_train_combined
    y_train_l3 = y_train
    
    print("📚 Training Model L1...")
    if len(y_train_l1) > 0:
        model_l1.fit(X_train_l1, y_train_l1)
    else:
        print("⚠️ Impossibile addestrare il modello L1: nessun campione valido.")
    
    print("📚 Training Model L2 (SVM)...")
    if len(y_train_l2) > 0:
        model_l2.fit(X_train_l2, y_train_l2)
    else:
        print("⚠️ Impossibile addestrare il modello L2: nessun campione valido.")
    
    print("📚 Training Model L3 (Random Forest)...")
    if len(y_train_l3) > 0:
        model_l3.fit(X_train_l3, y_train_l3)
    else:
        print("⚠️ Impossibile addestrare il modello L3: nessun campione valido.")
    
    # Predizioni per L'INTERO DATASET BILANCIATO
    X_all_tfidf = vectorizer.transform(df_train_balanced['text'])
    X_all_keywords = keyword_features_df.values
    X_all_combined = hstack([X_all_tfidf, csr_matrix(X_all_keywords)])

    # Prepara il DataFrame per i risultati
    df_result = df_train_balanced.copy()
    
    # Predizioni di probabilità per ogni livello
    probs_l1 = model_l1.predict_proba(X_all_combined)
    probs_l2 = model_l2.predict_proba(X_all_combined)
    probs_l3 = model_l3.predict_proba(X_all_combined)
    
    df_result['probs_l1'] = [dict(zip(model_l1.classes_, p)) for p in probs_l1]
    df_result['probs_l2'] = [dict(zip(model_l2.classes_, p)) for p in probs_l2]
    df_result['probs_l3'] = [dict(zip(model_l3.classes_, p)) for p in probs_l3]

    print("\n⚖️ ESECUZIONE DEL CSP PER LA POST-ELABORAZIONE:")
    print("=" * 60)
    
    final_predictions = []
    final_probs = []
    
    for idx, row in df_result.iterrows():
        doc_probs = {
            'L1': row['probs_l1'],
            'L2': row['probs_l2'],
            'L3': row['probs_l3'],
        }

        problem = setup_csp_problem(doc_probs, g, NS)
        l1, l2, l3, probs = find_best_csp_solution(problem, doc_probs)
        
        final_predictions.append({
            'L1_pred': l1,
            'L2_pred': l2,
            'L3_pred': l3,
        })
        final_probs.append(probs)

    predictions_df = pd.DataFrame(final_predictions)
    df_result = pd.concat([df_result.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)
    df_result['final_probs'] = final_probs
    
    # Aggiungi le probabilità finali in colonne separate
    df_result['L1_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L1_pred'], 0), axis=1)
    df_result['L2_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L2_pred'], 0), axis=1)
    df_result['L3_prob'] = df_result.apply(lambda row: row['final_probs'].get(row['L3_pred'], 0), axis=1)

    output_cols = ['filename', 'single_label', 'L1_pred', 'L1_prob', 'L2_pred', 'L2_prob', 'L3_pred', 'L3_prob']
    
    print("\n📋 ESEMPIO DI RISULTATI POST-CSP:")
    print(df_result[output_cols].head().to_string(index=False))
    
    df_result.to_csv('predictions_with_csp_postprocessing.csv', index=False)
    print("\n✅ Risultati salvati in 'predictions_with_csp_postprocessing.csv'")
    
    print("\n🎊 PROCESSO COMPLETATO!")