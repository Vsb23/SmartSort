import os
import pandas as pd
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from rdflib import Graph, Namespace, RDFS
import numpy as np
import ast

def load_ontology_categories(ontology_path):
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    categories = set()
    categories.add('Scienza')
    for s, p, o in g.triples((None, RDFS.subClassOf, NS['Scienza'])):
        class_name = str(s).split('#')[-1]
        categories.add(class_name)
    return sorted(categories)

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

def load_dataset_with_category_labels(csv_file, base_folder):
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

    df = pd.read_csv(csv_file)
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

        labels.append(parse_labels(row['category']))

    df['text'] = texts
    df['labels'] = labels
    df_train = df[df['labels'].map(len) > 0].copy()
    return df_train

def encode_labels(labels_list, all_labels):
    arr = np.zeros(len(all_labels))
    for label in labels_list:
        if label in all_labels:
            idx = all_labels.index(label)
            arr[idx] = 1
    return arr

def train_model(train, val, all_labels):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    y_train = np.array(train['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())
    y_val = np.array(val['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())

    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    print("F1 score micro:", f1_score(y_val, y_pred, average='micro'))
    print(classification_report(y_val, y_pred, target_names=all_labels))

    return clf, vectorizer

def predict_top_categories(df, clf, vectorizer, all_labels):
    X = vectorizer.transform(df['text'])
    probs = clf.predict_proba(X)
    top_categories = []
    for prob in probs:
        top_idx = np.argsort(prob)[::-1][:3]
        top_cats = [all_labels[i] for i in top_idx]
        top_categories.append(top_cats)
    df['predicted_labels'] = top_categories
    return df

if __name__ == "__main__":
    ontology_path = "./Ontology.owx"  # Cambia con il percorso reale
    base_folder = "./References"
    csv_file = "training_set.csv"  # File CSV con colonna 'category' compilata manualmente

    ontology_categories = load_ontology_categories(ontology_path)

    df_train = load_dataset_with_category_labels(csv_file, base_folder)

    train, val = train_test_split(df_train, test_size=0.2, random_state=42,
                                  stratify=df_train['labels'].apply(lambda x: x[0]))

    model, vectorizer = train_model(train, val, ontology_categories)

    df_pred = predict_top_categories(df_train, model, vectorizer, ontology_categories)

    print(df_pred[['filename', 'predicted_labels']].head())

    df_pred.to_csv("output_with_predicted_categories.csv", index=False)
