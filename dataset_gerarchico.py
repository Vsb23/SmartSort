import pandas as pd
import re
import numpy as np
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, RDFS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix



def load_keywords_from_ontology(ontology_path, ns):
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
    except Exception as e:
        print(f"⚠️ Errore nell'estrazione SPARQL: {e}")
        print("📝 Usando metodo alternativo...")
        for s, p, o in g:
            if str(p).endswith('#hasKeyword'):
                class_name = str(s).split('#')[-1]
                keyword = str(o)
                keywords_dict[class_name].append(keyword)
        print(f"✅ Keywords caricate (metodo alternativo) per {len(keywords_dict)} categorie")
    keywords_dict = dict(keywords_dict)
    if 'Altro' not in keywords_dict:
        keywords_dict['Altro'] = []
    return keywords_dict


def get_all_subclasses(g, base_class_uri):
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))
    for subclass in direct_subclasses:
        subclass_name = str(subclass).split('#')[-1]
        subclasses.add(subclass_name)
        subclasses |= get_all_subclasses(g, subclass)
    return subclasses


def extract_hierarchical_levels(g, ns):
    level_1 = set()
    level_2 = set()
    level_3 = set()

    root_1 = ns['Scienza']
    root_2 = ns['Studi_umanistici']

    level_1.add('Scienza')
    level_1.add('Studi_umanistici')

    children_1 = set(s for s, p, o in g.triples((None, RDFS.subClassOf, root_1)))
    children_2 = set(s for s, p, o in g.triples((None, RDFS.subClassOf, root_2)))
    for c in children_1:
        level_2.add(str(c).split('#')[-1])
    for c in children_2:
        level_2.add(str(c).split('#')[-1])

    for parent in children_1 | children_2:
        grandchildren = set(s for s, p, o in g.triples((None, RDFS.subClassOf, parent)))
        for gc in grandchildren:
            level_3.add(str(gc).split('#')[-1])

    level_1.add('Altro')

    return list(level_1), list(level_2), list(level_3)


def get_all_superclasses_recursive(g, cls_uri):
    superclasses = set()
    for s, p, o in g.triples((cls_uri, RDFS.subClassOf, None)):
        if o not in superclasses and o != cls_uri:
            superclasses.add(o)
            superclasses |= get_all_superclasses_recursive(g, o)
    return superclasses


def add_hierarchical_labels_from_specific(df, g, ns):
    lvl1_roots = {'Scienza', 'Studi_umanistici'}
    level_1_labels = []
    level_2_labels = []
    level_3_labels = []

    # Prepare a mapping from URI string to local name for faster access
    uri_to_local = {}
    for s in g.subjects():
        uri_to_local[str(s)] = str(s).split('#')[-1]

    for cat in df['category']:
        if not isinstance(cat, str):
            cat = 'Altro'
        # find the corresponding URI for the category name
        cls_uri = None
        for s in g.subjects():
            if uri_to_local[str(s)] == cat:
                cls_uri = s
                break
        if cls_uri is None:
            # category not in ontology, fallback
            level_3_labels.append(cat)
            level_2_labels.append('Altro')
            level_1_labels.append('Altro')
            continue

        superclasses = get_all_superclasses_recursive(g, cls_uri)
        superclass_names = {uri_to_local.get(str(uri), 'Altro') for uri in superclasses}

        # Level 1: look for roots
        l1 = next((root for root in lvl1_roots if root in superclass_names), None)
        # Level 2: any superclass not in lvl1 roots and not 'Altro'
        l2 = next((sc for sc in superclass_names if sc not in lvl1_roots and sc != 'Altro'), 'Altro')

        level_1_labels.append(l1 if l1 else 'Altro')
        level_2_labels.append(l2 if l2 else 'Altro')
        level_3_labels.append(cat)

    df['category_level_1'] = level_1_labels
    df['category_level_2'] = level_2_labels
    df['category_level_3'] = level_3_labels
    return df


def map_labels_to_level(df, level):
    col = f'category_level_{level}'
    valid_labels = set(all_labels_per_level[level])

    def mapper(label):
        if label in valid_labels:
            return label
        else:
            return None

    df_level = df[df[col].isin(valid_labels)].copy()
    df_level['label_level'] = df_level[col].apply(mapper)
    return df_level[df_level['label_level'].notna()]


def create_enhanced_features(df, ontology_keywords):
    print("🔧 Creazione features avanzate basate su ontologia...")
    keyword_features = []
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        doc_features = {}
        for category, keywords in ontology_keywords.items():
            keyword_count = 0
            for keyword in keywords:
                if keyword:
                    pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                    matches = len(re.findall(pattern, text))
                    keyword_count += matches * len(keyword.split()) if ' ' in keyword else matches
            doc_features[f'{category}_keywords'] = keyword_count
        keyword_features.append(doc_features)
    return pd.DataFrame(keyword_features)


def train_eval_single_label_model(name, clf, X_train, y_train, X_val, y_val):
    print(f"---- Training {name} (Single-Label) ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 weighted: {f1:.4f}")

    # y_val = vere etichette
    # y_pred = predizioni del modello

    mask = ~((y_val == 'Altro') & (y_pred != 'Altro'))

    y_val_filtered = y_val[mask]
    y_pred_filtered = y_pred[mask]
    print(classification_report(y_val, y_pred, target_names=None, zero_division=0))
    return clf


def build_parent_child_map(g, ns, all_labels_per_level):
    # Costruisce mappe padre -> set figli per i livelli 1->2 e 2->3

    # URI mapping per velocizzare
    uri_to_local = {str(s): str(s).split("#")[-1] for s in g.subjects()}

    # Livello 1 -> livello 2
    parent_to_children_1_2 = {lvl1: set() for lvl1 in all_labels_per_level[1]}
    for lvl2_cat in all_labels_per_level[2]:
        # Trova URI categoria figlia
        lvl2_uri = None
        for s in g.subjects():
            if uri_to_local.get(str(s), None) == lvl2_cat:
                lvl2_uri = s
                break
        if lvl2_uri is None:
            continue
        # Superclassi
        superclasses = get_all_superclasses_recursive(g, lvl2_uri)
        superclass_names = {uri_to_local.get(str(uri), None) for uri in superclasses}
        for cls_name in parent_to_children_1_2.keys():
            if cls_name in superclass_names:
                parent_to_children_1_2[cls_name].add(lvl2_cat)

    # Livello 2 -> livello 3
    parent_to_children_2_3 = {lvl2: set() for lvl2 in all_labels_per_level[2]}
    for lvl3_cat in all_labels_per_level[3]:
        lvl3_uri = None
        for s in g.subjects():
            if uri_to_local.get(str(s), None) == lvl3_cat:
                lvl3_uri = s
                break
        if lvl3_uri is None:
            continue
        superclasses = get_all_superclasses_recursive(g, lvl3_uri)
        superclass_names = {uri_to_local.get(str(uri), None) for uri in superclasses}
        for cls_name in parent_to_children_2_3.keys():
            if cls_name in superclass_names:
                parent_to_children_2_3[cls_name].add(lvl3_cat)

    return parent_to_children_1_2, parent_to_children_2_3


def hierarchical_predict(df, models_per_level, vectorizers, keyword_features_per_level, all_labels_per_level, g, ns):
    df_result = df.copy()
    df_result['predicted_level_1'] = None
    df_result['prob_level_1'] = None
    df_result['predicted_level_2'] = None
    df_result['prob_level_2'] = None
    df_result['predicted_level_3'] = None
    df_result['prob_level_3'] = None

    # Costruisci le mappe padre Figlio per la gerarchia
    parent_to_children_1_2, parent_to_children_2_3 = build_parent_child_map(g, ns, all_labels_per_level)

    for idx, row in df.iterrows():
        probs_lvl_1, pred_lvl_1 = predict_single_level(row,
                                                      models_per_level[1], vectorizers[1], keyword_features_per_level[1], all_labels_per_level[1], row.name)
        df_result.at[idx, 'predicted_level_1'] = pred_lvl_1
        df_result.at[idx, 'prob_level_1'] = max(probs_lvl_1.values()) if probs_lvl_1 else None

        # Se la categoria di livello 1 ha figli a livello 2, predici livello 2
        if pred_lvl_1 in parent_to_children_1_2 and parent_to_children_1_2[pred_lvl_1]:
            probs_lvl_2, pred_lvl_2 = predict_single_level(row,
                                                          models_per_level[2], vectorizers[2], keyword_features_per_level[2], all_labels_per_level[2], row.name)
            df_result.at[idx, 'predicted_level_2'] = pred_lvl_2
            df_result.at[idx, 'prob_level_2'] = max(probs_lvl_2.values()) if probs_lvl_2 else None

            # Se categoria livello 2 ha figli a livello 3, predici livello 3
            if pred_lvl_2 in parent_to_children_2_3 and parent_to_children_2_3[pred_lvl_2]:
                probs_lvl_3, pred_lvl_3 = predict_single_level(row,
                                                              models_per_level[3], vectorizers[3], keyword_features_per_level[3], all_labels_per_level[3], row.name)
                df_result.at[idx, 'predicted_level_3'] = pred_lvl_3
                df_result.at[idx, 'prob_level_3'] = max(probs_lvl_3.values()) if probs_lvl_3 else None

    return df_result


def predict_single_level(row, model, vectorizer, keyword_features_df, labels, idx):
    X_tfidf = vectorizer.transform([row['text']])
    X_keywords = csr_matrix(keyword_features_df.iloc[[idx]].values)
    X_combined = hstack([X_tfidf, X_keywords])
    probas = model.predict_proba(X_combined)
    pred_idx = np.argmax(probas[0])
    pred_label = labels[pred_idx]
    probs_dict = {label: prob for label, prob in zip(labels, probas[0])}
    return probs_dict, pred_label


if __name__ == "__main__":
    ontology_path = "./Ontology.owx"
    csv_file = "./training_set_single_category_85percent.csv"

    print("🚀 Inizio estrazione gerarchica e addestramento modelli")

    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")

    lvl1_cats, lvl2_cats, lvl3_cats = extract_hierarchical_levels(g, NS)
    print(f"Livello 1: {lvl1_cats}")
    print(f"Livello 2: {lvl2_cats}")
    print(f"Livello 3: {lvl3_cats}")

    ontology_keywords = load_keywords_from_ontology(ontology_path, NS)
    if not ontology_keywords:
        print("⚠️ Nessuna keyword trovata nell'ontologia!")
        ontology_keywords = {'Altro': []}

    df = pd.read_csv(csv_file, low_memory=False)
    df['text'] = df['clean_text'].fillna("")

    df = add_hierarchical_labels_from_specific(df, g, NS)

    models_per_level = {}
    vectorizers = {}
    keyword_features_per_level = {}
    all_labels_per_level = {1: lvl1_cats, 2: lvl2_cats, 3: lvl3_cats}

    for level in [1, 2, 3]:
        print(f"\n--- Preparazione dati livello {level} ---")
        level_labels = all_labels_per_level[level]
        df_level = map_labels_to_level(df, level)

        print(f"Documenti livello {level}: {len(df_level)}")
        print(f"Distribuzione classi:\n{df_level['label_level'].value_counts()}")

        keyword_feat_df = create_enhanced_features(df_level, ontology_keywords)
        keyword_features_per_level[level] = keyword_feat_df.reset_index(drop=True)

        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df_level['text'])
        vectorizers[level] = vectorizer

        X_combined = hstack([X_tfidf, csr_matrix(keyword_feat_df.values)])

        y = df_level['label_level'].values
        counter = Counter(y)
        min_count = min(counter.values())
        if min_count < 2:
            print(f"⚠️ Classe con meno di 2 esempi trovata, salto stratificazione livello {level}")
            train, val = train_test_split(df_level, test_size=0.2, random_state=42)
        else:
            train, val = train_test_split(df_level, test_size=0.2, random_state=42, stratify=y)

        train = train.reset_index(drop=True)
        val = val.reset_index(drop=True)

        X_train_tfidf = vectorizers[level].transform(train['text'])
        X_val_tfidf = vectorizers[level].transform(val['text'])
        X_train_keywords = keyword_feat_df.iloc[train.index].values
        X_val_keywords = keyword_feat_df.iloc[val.index].values
        X_train_combined = hstack([X_train_tfidf, csr_matrix(X_train_keywords)])
        X_val_combined = hstack([X_val_tfidf, csr_matrix(X_val_keywords)])
        y_train = train['label_level'].values
        y_val = val['label_level'].values

        print(f"Addestrando modelli livello {level}...")

        if len(np.unique(y_train)) < 2:
            print(f"⚠️ Livello {level} ha meno di 2 classi in train, salto addestramento")
            continue

        clf_lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
        lr_model = train_eval_single_label_model("Logistic Regression", clf_lr, X_train_combined, y_train, X_val_combined, y_val)

        clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model = train_eval_single_label_model("Random Forest", clf_rf, X_train_combined, y_train, X_val_combined, y_val)

        clf_svc = SVC(probability=True, random_state=42)
        svc_model = train_eval_single_label_model("SVM", clf_svc, X_train_combined, y_train, X_val_combined, y_val)

        clf_nb = MultinomialNB()
        nb_model = train_eval_single_label_model("Naive Bayes", clf_nb, X_train_combined, y_train, X_val_combined, y_val)

        models_per_level[level] = rf_model

    print("\n--- Predizioni Gerarchiche ---")
    df_predictions = hierarchical_predict(df, models_per_level, vectorizers, keyword_features_per_level, all_labels_per_level, g, NS)

    print("Predizioni finali con probabilità gerarchiche:")
    print(df_predictions.head())

    df_predictions.to_csv('predictions_hierarchical.csv', index=False)
    print("\n✅ Salvataggio predizioni gerarchiche completato: predictions_hierarchical.csv")

