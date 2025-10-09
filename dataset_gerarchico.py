import pandas as pd
import re
import numpy as np
from collections import defaultdict, Counter
from rdflib import Graph, Namespace, RDFS
from sklearn.model_selection import StratifiedKFold
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
    uri_to_local = {}
    for s in g.subjects():
        uri_to_local[str(s)] = str(s).split('#')[-1]
    for cat in df['category']:
        if not isinstance(cat, str):
            cat = 'Altro'
        cls_uri = None
        for s in g.subjects():
            if uri_to_local[str(s)] == cat:
                cls_uri = s
                break
        if cls_uri is None:
            level_3_labels.append(cat)
            level_2_labels.append('Altro')
            level_1_labels.append('Altro')
            continue
        superclasses = get_all_superclasses_recursive(g, cls_uri)
        superclass_names = {uri_to_local.get(str(uri), 'Altro') for uri in superclasses}
        l1 = next((root for root in lvl1_roots if root in superclass_names), None)
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
    return pd.DataFrame(keyword_features, index=df.index)

def build_parent_child_map(g, ns, all_labels_per_level):
    uri_to_local = {str(s): str(s).split("#")[-1] for s in g.subjects()}
    parent_to_children_1_2 = {lvl1: set() for lvl1 in all_labels_per_level[1]}
    for lvl2_cat in all_labels_per_level[2]:
        lvl2_uri = None
        for s in g.subjects():
            if uri_to_local.get(str(s), None) == lvl2_cat:
                lvl2_uri = s
                break
        if lvl2_uri is None:
            continue
        superclasses = get_all_superclasses_recursive(g, lvl2_uri)
        superclass_names = {uri_to_local.get(str(uri), None) for uri in superclasses}
        for cls_name in parent_to_children_1_2.keys():
            if cls_name in superclass_names:
                parent_to_children_1_2[cls_name].add(lvl2_cat)
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

def hierarchical_predict_ensemble(df, models_per_level, vectorizers, keyword_features_per_level, all_labels_per_level, g, ns):
    df_result = df.copy()
    df_result['predicted_level_1'] = None
    df_result['prob_level_1'] = None
    df_result['predicted_level_2'] = None
    df_result['prob_level_2'] = None
    df_result['predicted_level_3'] = None
    df_result['prob_level_3'] = None

    parent_to_children_1_2, parent_to_children_2_3 = build_parent_child_map(g, ns, all_labels_per_level)

    for idx, row in df.iterrows():
        # --- Livello 1 ---
        models = models_per_level[1]
        vectorizer = vectorizers[1]
        keyword_feats = keyword_features_per_level[1]
        labels = all_labels_per_level[1]

        probas_sum = None
        X_tfidf = vectorizer.transform([row['text']])
        X_keywords = csr_matrix(keyword_feats.iloc[[row.name]].values)
        X_combined = hstack([X_tfidf, X_keywords])
        for model in models:
            probas = model.predict_proba(X_combined)[0]
            if probas_sum is None:
                probas_sum = probas
            else:
                probas_sum += probas
        probas_avg = probas_sum / len(models)
        pred_idx1 = np.argmax(probas_avg)
        pred_label1 = labels[pred_idx1]
        df_result.at[idx, 'predicted_level_1'] = pred_label1
        df_result.at[idx, 'prob_level_1'] = probas_avg[pred_idx1]

        # --- Livello 2: limitazione delle categorie alle sole figlie ---
        valid_lvl2 = set(parent_to_children_1_2.get(pred_label1, []))
        if not valid_lvl2:
            continue  # Nessuna categoria figlia, passa oltre

        models = models_per_level[2]
        vectorizer = vectorizers[2]
        keyword_feats = keyword_features_per_level[2]
        labels = all_labels_per_level[2]

        probas_sum = None
        X_tfidf = vectorizer.transform([row['text']])
        X_keywords = csr_matrix(keyword_feats.iloc[[row.name]].values)
        X_combined = hstack([X_tfidf, X_keywords])
        for model in models:
            probas = model.predict_proba(X_combined)[0]
            if probas_sum is None:
                probas_sum = probas
            else:
                probas_sum += probas
        probas_avg_lvl2 = probas_sum / len(models)
        # Setta a zero la probabilità delle label non figlie
        for i, label in enumerate(labels):
            if label not in valid_lvl2:
                probas_avg_lvl2[i] = 0
        pred_idx2 = np.argmax(probas_avg_lvl2)
        pred_label2 = labels[pred_idx2]
        df_result.at[idx, 'predicted_level_2'] = pred_label2
        df_result.at[idx, 'prob_level_2'] = probas_avg_lvl2[pred_idx2]

        # --- Livello 3: limitazione delle categorie alle sole figlie ---
        valid_lvl3 = set(parent_to_children_2_3.get(pred_label2, []))
        if not valid_lvl3:
            continue  # Nessuna categoria figlia, passa oltre

        models = models_per_level[3]
        vectorizer = vectorizers[3]
        keyword_feats = keyword_features_per_level[3]
        labels = all_labels_per_level[3]

        probas_sum = None
        X_tfidf = vectorizer.transform([row['text']])
        X_keywords = csr_matrix(keyword_feats.iloc[[row.name]].values)
        X_combined = hstack([X_tfidf, X_keywords])
        for model in models:
            probas = model.predict_proba(X_combined)[0]
            if probas_sum is None:
                probas_sum = probas
            else:
                probas_sum += probas
        probas_avg_lvl3 = probas_sum / len(models)
        for i, label in enumerate(labels):
            if i >= len(probas_avg_lvl3):
                break
            if label not in valid_lvl3:
                probas_avg_lvl3[i] = 0
                
        pred_idx3 = np.argmax(probas_avg_lvl3)
        pred_label3 = labels[pred_idx3]
        df_result.at[idx, 'predicted_level_3'] = pred_label3
        df_result.at[idx, 'prob_level_3'] = probas_avg_lvl3[pred_idx3]
    return df_result



def predict_single_level(row, model, vectorizer, ontology_keywords, keyword_columns, labels):
    X_tfidf = vectorizer.transform([row['text']])

    # --- Inizio generazione feature on-the-fly ---
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
    
    # Ordina i valori delle feature usando la lista di colonne salvata
    ordered_keyword_values = [doc_features[col] for col in keyword_columns]
    X_keywords = csr_matrix([ordered_keyword_values])
    # --- Fine generazione feature ---

    X_combined = hstack([X_tfidf, X_keywords])
    probas = model.predict_proba(X_combined)
    pred_idx = np.argmax(probas[0])
    pred_label = labels[pred_idx]
    probs_dict = {label: prob for label, prob in zip(labels, probas[0])}
    return probs_dict, pred_label

if __name__ == "__main__":
    ontology_path = "./Ontology.owx"
    train_csv_file = "./train_set_85percent.csv"
    test_csv_file = "./test_set_15percent.csv"

    print("🚀 Inizio estrazione gerarchica e addestramento modelli")
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")

    lvl1_cats, lvl2_cats, lvl3_cats = extract_hierarchical_levels(g, NS)

    ontology_keywords = load_keywords_from_ontology(ontology_path, NS)
    if not ontology_keywords:
        print("⚠️ Nessuna keyword trovata nell'ontologia!")
        ontology_keywords = {'Altro': []}

    # CARICAMENTO TRAINING SET:
    df_train = pd.read_csv(train_csv_file, low_memory=False)
    df_train['text'] = df_train['clean_text'].fillna("")
    df_train = add_hierarchical_labels_from_specific(df_train, g, NS)

    # CARICAMENTO TEST SET:
    df_test = pd.read_csv(test_csv_file, low_memory=False)
    df_test['text'] = df_test['clean_text'].fillna("")
    # NON serve aggiungere le label nel test set (puoi aggiungerle se disponibili)

    all_labels_per_level = {1: lvl1_cats, 2: lvl2_cats, 3: lvl3_cats}

    models_per_level = {}
    vectorizers = {}
    keyword_features_per_level = {}

    # Addestramento con Cross Validation sul training set
    for level in [1, 2, 3]:
        print(f"\n--- Preparazione dati livello {level} ---")
        level_labels = all_labels_per_level[level]
        df_level = map_labels_to_level(df_train, level)

        print(f"Documenti livello {level}: {len(df_level)}")
        keyword_feat_df = create_enhanced_features(df_level, ontology_keywords).reset_index(drop=True)
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_tfidf = vectorizer.fit_transform(df_level['text'])
        X_keywords = keyword_feat_df.values
        X_combined = hstack([X_tfidf, csr_matrix(X_keywords)])
        y = df_level['label_level'].values

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # rf_metrics = []
        # lr_metrics = []
        svc_metrics = []
        # nb_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_combined, y)):
            print(f"Fold {fold+1}")
            X_train, X_val = X_combined[train_idx], X_combined[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # # Random Forest
            # clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            # clf_rf.fit(X_train, y_train)
            # y_pred_rf = clf_rf.predict(X_val)
            # acc_rf = accuracy_score(y_val, y_pred_rf)
            # f1_rf = f1_score(y_val, y_pred_rf, average='weighted')
            # print(f"Random Forest Fold {fold+1} Acc: {acc_rf:.4f} F1: {f1_rf:.4f}")
            # rf_metrics.append((acc_rf, f1_rf))

            # # Logistic Regression
            # clf_lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
            # clf_lr.fit(X_train, y_train)
            # y_pred_lr = clf_lr.predict(X_val)
            # acc_lr = accuracy_score(y_val, y_pred_lr)
            # f1_lr = f1_score(y_val, y_pred_lr, average='weighted')
            # print(f"LogReg Fold {fold+1} Acc: {acc_lr:.4f} F1: {f1_lr:.4f}")
            # lr_metrics.append((acc_lr, f1_lr))

            # SVM
            clf_svc = SVC(probability=True, random_state=42)
            clf_svc.fit(X_train, y_train)
            y_pred_svc = clf_svc.predict(X_val)
            acc_svc = accuracy_score(y_val, y_pred_svc)
            f1_svc = f1_score(y_val, y_pred_svc, average='weighted')
            print(f"SVM Fold {fold+1} Acc: {acc_svc:.4f} F1: {f1_svc:.4f}")
            svc_metrics.append((acc_svc, f1_svc))

            # # Naive Bayes
            # clf_nb = MultinomialNB()
            # clf_nb.fit(X_train, y_train)
            # y_pred_nb = clf_nb.predict(X_val)
            # acc_nb = accuracy_score(y_val, y_pred_nb)
            # f1_nb = f1_score(y_val, y_pred_nb, average='weighted')
            # print(f"Naive Bayes Fold {fold+1} Acc: {acc_nb:.4f} F1: {f1_nb:.4f}")
            # nb_metrics.append((acc_nb, f1_nb))

        # print(f"Media Accuracy RF CV: {np.mean([m[0] for m in rf_metrics]):.4f}, Media F1 RF CV: {np.mean([m[1] for m in rf_metrics]):.4f}")
        # print(f"Media Accuracy LR CV: {np.mean([m[0] for m in lr_metrics]):.4f}, Media F1 LR CV: {np.mean([m[1] for m in lr_metrics]):.4f}")
        print(f"Media Accuracy SVM CV: {np.mean([m[0] for m in svc_metrics]):.4f}, Media F1 SVM CV: {np.mean([m[1] for m in svc_metrics]):.4f}")
        # print(f"Media Accuracy NB CV: {np.mean([m[0] for m in nb_metrics]):.4f}, Media F1 NB CV: {np.mean([m[1] for m in nb_metrics]):.4f}")
        
        
        # Allena modello finale su tutto il training set per uso in predizioni
        # clf_rf_final = RandomForestClassifier(n_estimators=100, random_state=42)
        # clf_rf_final.fit(X_combined, y)

        # clf_lr_final = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs'))
        # clf_lr_final.fit(X_combined, y)

        clf_svc_final = SVC(probability=True, random_state=42)
        clf_svc_final.fit(X_combined, y)

        # clf_nb_final = MultinomialNB()
        # clf_nb_final.fit(X_combined, y)
        models_per_level[level] = [
            # clf_rf_final, clf_lr_final,
            clf_svc_final, 
            # clf_nb_final 
            ]

        vectorizers[level] = vectorizer
        keyword_features_per_level[level] = keyword_feat_df.reset_index(drop=True)

    # Predizione e valutazione sul test set, tramite gerarchico
    df_predictions = hierarchical_predict_ensemble(df_test, models_per_level, vectorizers, keyword_features_per_level, all_labels_per_level, g, NS)
    print(df_predictions.head())
    df_predictions.to_csv("predictions_test_set.csv", index=False)

    print("✅ Salvataggio predizioni su test set completato: predictions_test_set.csv")
