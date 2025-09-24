import os
import pandas as pd
import PyPDF2
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# 1) Carica CSV e estrai testi PDF
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
    # pulizia base: rimuovi numeri, punteggiatura, minuscola
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 2) Carica dataset e estrai testo completo
def load_and_prepare_dataset(csv_file, base_folder):
    df = pd.read_csv(csv_file)
    
    # Assumendo il CSV ha colonna 'relative_path' e 'titolo' = nome file
    texts = []
    for idx, row in df.iterrows():
        if row['type'] == 'file' and row['extension'] == '.pdf':
            pdf_file_path = os.path.join(base_folder, row['relative_path'], row['titolo'] + row['extension'])
            text = extract_text_from_pdf(pdf_file_path)
            text = preprocess_text(text)
        else:
            text = ""
        texts.append(text)
    df['text'] = texts
    return df

# 3) Esempio semplificato di etichette multi-label (sostituire con label reali)
def dummy_labeling(df):
    # Ad esempio usa la colonna 'category' come unica label (da espandere)
    df['labels'] = df['category'].apply(lambda x: [x])
    return df

# 4) Split dati
def split_dataset(df):
    train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['category'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['category'])
    return train, val, test

# 5) Feature extraction base + modello
def train_model(train, val):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    
    # Ricodifica labels numeriche (dummy example)
    all_labels = sorted(set(label for labels in train['labels'] for label in labels))
    label_index = {label: i for i, label in enumerate(all_labels)}
    
    import numpy as np
    def encode_labels(labels):
        arr = np.zeros(len(all_labels))
        for l in labels:
            arr[label_index[l]] = 1
        return arr

    y_train = np.array(train['labels'].apply(encode_labels).tolist())
    y_val = np.array(val['labels'].apply(encode_labels).tolist())
    
    clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_val)
    print("F1 score micro:", f1_score(y_val, y_pred, average='micro'))
    print(classification_report(y_val, y_pred, target_names=all_labels))
    return clf, vectorizer, all_labels

# === MAIN ===
base_folder = "./References"
csv_file = "output.csv"

df = load_and_prepare_dataset(csv_file, base_folder)
df = dummy_labeling(df)
train, val, test = split_dataset(df)

model, vectorizer, labels = train_model(train, val)

# Da qui puoi salvare modello, fare test, implementare output top3 categorie, etc.

