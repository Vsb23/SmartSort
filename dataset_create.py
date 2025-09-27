import os
import pandas as pd
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from rdflib import Graph, Namespace, RDFS
import numpy as np
from collections import Counter
 
def get_all_subclasses(g, base_class_uri):
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))
    for subclass in direct_subclasses:
        subclasses.add(str(subclass).split('#')[-1])
        subclasses |= get_all_subclasses(g, subclass)
    return subclasses
 
def get_superclasses(g, cls_uri):
    superclasses = set()
    for s, p, o in g.triples((cls_uri, RDFS.subClassOf, None)):
        name = str(o).split('#')[-1]
        if name != str(cls_uri).split('#')[-1]:  # evita loop
            superclasses.add(name)
            superclasses |= get_superclasses(g, o)
    return superclasses
 
def enrich_labels(labels, g, ns):
    enriched = set(labels)
    for label in labels:
        try:
            enriched |= get_superclasses(g, ns[label])
        except:
            pass  # Ignora se label non è in ontologia
    return list(enriched)
 
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Errore estrazione testo PDF {pdf_path}: {e}")
    return text
 
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
 
def parse_labels(label_str):
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
 
def load_dataset_with_category_labels(csv_file, base_folder, g, ns):
    df = pd.read_csv(csv_file, low_memory=False)
    texts = []
    labels = []
    for idx, row in df.iterrows():
        if row['type'] == 'file' and row['extension'] == '.pdf':
            pdf_file_path = os.path.join(base_folder, row['filename'])
            text = extract_text_from_pdf(pdf_file_path)
            text = preprocess_text(text)
        else:
            text = ""
        texts.append(text)
        raw_labels = parse_labels(row['category'])
        enriched_labels = enrich_labels(raw_labels, g, ns)
        labels.append(enriched_labels)
    df['text'] = texts
    df['labels'] = labels
    df_train = df[df['labels'].map(len) > 0].copy()  # Solo con almeno una label
    return df_train
 
def encode_labels(labels_list, all_labels):
    arr = np.zeros(len(all_labels))
    for label in labels_list:
        if label in all_labels:
            idx = all_labels.index(label)
            arr[idx] = 1
    return arr
 
def train_eval_model(name, clf, X_train, y_train, X_val, y_val, all_labels):
    print(f"---- Training {name} ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="micro")
    print(f"{name} F1 micro: {f1}")
    print(classification_report(y_val, y_pred, target_names=all_labels, zero_division=0))
    return clf
 
def filter_general_superclasses(predicted_labels, g, ns):
    filtered = set(predicted_labels)
    for label in predicted_labels:
        try:
            superclasses = get_superclasses(g, ns[label])
            filtered -= superclasses
        except:
            pass
    return list(filtered)
 
def predict_top_categories(df, clf, vectorizer, all_labels):
    X = vectorizer.transform(df['text'])
    probs = clf.predict_proba(X)
    top_categories = []
    for prob in probs:
        top_idx = np.argsort(prob)[::-1][:3]
        top_cats = [all_labels[i] for i in top_idx]
        top_categories.append(top_cats)
    df['predicted_category'] = top_categories
    return df

def undersample_general_classes(df, label_col='labels', general_labels=None, threshold=0.5):

    if general_labels is None:
        general_labels = ['Scienza', 'Studi_umanistici']
    
    counter = Counter()
    for labels in df[label_col]:
        counter.update(labels)

    max_counts = {}
    for label in counter:
        if label in general_labels:
            max_counts[label] = int(threshold * counter[label])
        else:
            max_counts[label] = counter[label]

    current_counts = Counter()
    selected_indices = []
    
    # CORREZIONE CRITICA: usa df.index invece di enumerate
    for idx in df.index:  
        labels = df.loc[idx, label_col]
        keep = True
        for lab in labels:
            if current_counts[lab] >= max_counts.get(lab, 0):
                keep = False
                break
        if keep:
            selected_indices.append(idx)
            current_counts.update(labels)

    return df.loc[selected_indices]

 
if __name__ == "__main__":
    ontology_path = "./Ontology.owx"
    base_folder = "./References"
    csv_file = "training_set.csv"
 
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
 
    categories = set(['Scienza'])
    categories |= get_all_subclasses(g, NS['Scienza'])
    categories.add('Studi_umanistici')
    categories |= get_all_subclasses(g, NS['Studi_umanistici'])
    all_labels = sorted(categories)
 
    df_train = load_dataset_with_category_labels(csv_file, base_folder, g, NS)
    general_labels = ['Scienza', 'Studi_umanistici']
    df_train_balanced = undersample_general_classes(df_train, 'labels', general_labels, threshold=0.5)


    class_counts = Counter(df_train['labels'].apply(lambda x: x[0]))
    can_stratify = all(count >= 2 for count in class_counts.values()) and len(df_train) >= 2
    if can_stratify:
        train, val = train_test_split(df_train, test_size=0.2, random_state=42,
                                     stratify=df_train['labels'].apply(lambda x: x[0]))
    else:
        train, val = train_test_split(df_train, test_size=0.2, random_state=42)
 
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
 
    y_train = np.array(train['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())
    y_val = np.array(val['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())
 
    # Logistic Regression
    clf_lr = OneVsRestClassifier(LogisticRegression(max_iter=200))
    model_lr = train_eval_model("Logistic Regression", clf_lr, X_train, y_train, X_val, y_val, all_labels)
 
    # SVM
    clf_svm = OneVsRestClassifier(SVC(kernel="linear", probability=True))
    model_svm = train_eval_model("SVM (linear kernel)", clf_svm, X_train, y_train, X_val, y_val, all_labels)
 
    # Random Forest
    clf_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model_rf = train_eval_model("Random Forest", clf_rf, X_train, y_train, X_val, y_val, all_labels)
 
    # Predizioni filtrate per Logistic Regression
    df_pred_lr = predict_top_categories(df_train, model_lr, vectorizer, all_labels)

    df_pred_lr['filtered_predicted_category'] = df_pred_lr['predicted_category'].apply(  # Non 'predicted_category'
        lambda labels: filter_general_superclasses(labels, g, NS))
    print(df_pred_lr[['filename', 'filtered_predicted_category']].head())  # Non 'filtered_predicted_category'

 
    # Predizioni filtrate per SVM
    df_pred_svm = predict_top_categories(df_train, model_svm, vectorizer, all_labels)
    df_pred_svm['filtered_predicted_labels'] = df_pred_svm['predicted_labels'].apply(
        lambda labels: filter_general_superclasses(labels, g, NS))
 
    # Predizioni filtrate per Random Forest
    df_pred_rf = predict_top_categories(df_train, model_rf, vectorizer, all_labels)
    df_pred_rf['filtered_predicted_labels'] = df_pred_rf['predicted_labels'].apply(
        lambda labels: filter_general_superclasses(labels, g, NS))

    print("Indice etichette e nomi corrispondenti:")
    for idx, label in enumerate(all_labels):
        print(f"{idx}: {label}")


    # Conta le occorrenze di ogni label su df_train['labels']
    counter = Counter()
    for lablist in df_train['labels']:
        counter.update(lablist)

    for label, count in counter.items():
        print(f"{label}: {count} ({count/len(df_train)*100:.2f}%)")