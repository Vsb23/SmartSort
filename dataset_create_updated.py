
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
    """Ottiene tutte le sottoclassi di una classe base - AGGIORNATO per includere 'Altro'"""
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))

    for subclass in direct_subclasses:
        subclass_name = str(subclass).split('#')[-1]
        subclasses.add(subclass_name)
        subclasses |= get_all_subclasses(g, subclass)

    # AGGIUNTA: Includi sempre "Altro" come categoria valida per ogni ramo
    subclasses.add('Altro')
    return subclasses

def get_superclasses(g, cls_uri):
    """Ottiene tutte le superclassi di una classe"""
    superclasses = set()
    for s, p, o in g.triples((cls_uri, RDFS.subClassOf, None)):
        name = str(o).split('#')[-1]
        if name != str(cls_uri).split('#')[-1]:  # evita loop
            superclasses.add(name)
            superclasses |= get_superclasses(g, o)
    return superclasses

def enrich_labels(labels, g, ns):
    """Arricchisce le etichette con informazioni gerarchiche - AGGIORNATO per 'Altro'"""
    enriched = set(labels)

    for label in labels:
        try:
            # Se è "Altro", non aggiungere superclassi (è già una categoria terminale)
            if label != 'Altro':
                enriched |= get_superclasses(g, ns[label])
        except:
            pass  # Ignora se label non è in ontologia

    return list(enriched)

def extract_text_from_pdf(pdf_path):
    """Estrae testo da PDF"""
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
    """Preprocessa il testo per l'analisi"""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

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

def load_dataset_with_category_labels(csv_file, base_folder, g, ns):
    """Carica dataset con etichette di categoria - AGGIORNATO per supportare 'Altro'"""
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

        # AGGIORNAMENTO: Gestisci "Altro" in modo speciale
        if 'Altro' in raw_labels:
            # "Altro" non ha superclassi specifiche, mantieni come categoria terminale
            enriched_labels = raw_labels
        else:
            enriched_labels = enrich_labels(raw_labels, g, ns)

        labels.append(enriched_labels)

    df['text'] = texts
    df['labels'] = labels
    df_train = df[df['labels'].map(len) > 0].copy()  # Solo con almeno una label

    return df_train

def encode_labels(labels_list, all_labels):
    """Codifica etichette in formato binario per classificazione multi-label"""
    arr = np.zeros(len(all_labels))
    for label in labels_list:
        if label in all_labels:
            idx = all_labels.index(label)
            arr[idx] = 1
    return arr

def train_eval_model(name, clf, X_train, y_train, X_val, y_val, all_labels):
    """Addestra e valuta un modello di classificazione"""
    print(f"---- Training {name} ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    f1 = f1_score(y_val, y_pred, average="micro")
    print(f"{name} F1 micro: {f1}")
    print(classification_report(y_val, y_pred, target_names=all_labels, zero_division=0))
    return clf

def filter_general_superclasses(predicted_labels, g, ns):
    """Filtra superclassi generali mantenendo solo le più specifiche"""
    filtered = set(predicted_labels)

    for label in predicted_labels:
        try:
            # Non filtrare "Altro" - è sempre valido
            if label == 'Altro':
                continue

            superclasses = get_superclasses(g, ns[label])
            filtered -= superclasses
        except:
            pass  # Mantieni se non in ontologia

    return list(filtered)

def predict_top_categories(df, clf, vectorizer, all_labels):
    """Predice le top categorie per i documenti"""
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
    """Sottocampiona classi generali per bilanciare il dataset - AGGIORNATO per 'Altro'"""
    if general_labels is None:
        # AGGIORNAMENTO: Includi "Altro" nelle categorie da sottocampionare
        general_labels = ['Scienza', 'Studi_umanistici', 'Altro']

    counter = Counter()
    for labels in df[label_col]:
        counter.update(labels)

    max_counts = {}
    for label in counter:
        if label in general_labels:
            # Per "Altro", usiamo una soglia più bassa per evitare over-representation
            if label == 'Altro':
                max_counts[label] = int(threshold * 0.5 * counter[label])
            else:
                max_counts[label] = int(threshold * counter[label])
        else:
            max_counts[label] = counter[label]

    current_counts = Counter()
    selected_indices = []

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

def create_hierarchical_features(df, ontology_keywords):
    """Crea features gerarchiche basate su keywords ontologiche"""
    hierarchical_features = []

    for _, row in df.iterrows():
        features = {}
        text = str(row['text']).lower()

        # Features per ogni categoria
        for category, keywords in ontology_keywords.items():
            keyword_count = sum(text.count(keyword.lower()) for keyword in keywords)
            features[f'{category}_keywords'] = keyword_count

        hierarchical_features.append(features)

    return pd.DataFrame(hierarchical_features)

# AGGIORNAMENTO: Keywords per categoria "Altro"
ONTOLOGY_KEYWORDS = {
   'Biologia': ['biology', 'biological', 'organism', 'cell', 'genetic', 'dna', 'rna', 'protein', 'enzyme', 'evolution', 'molecular', 'bio'],
        'Ambiente': ['environment', 'environmental', 'climate', 'pollution', 'sustainability', 'green', 'ecosystem', 'conservation', 'eco'],
        'Ecologia': ['ecology', 'ecological', 'ecosystem', 'biodiversity', 'habitat', 'species', 'wildlife', 'conservation'],
        'Chimica': ['chemistry', 'chemical', 'compound', 'molecule', 'reaction', 'synthesis', 'catalyst', 'polymer', 'organic'],
        'Fisica': ['physics', 'physical', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetic', 'optics', 'particle'],
        'Energia': ['energy', 'power', 'renewable', 'solar', 'wind', 'nuclear', 'battery', 'fuel', 'electricity'],
        'Spazio': ['space', 'satellite', 'orbit', 'planetary', 'astronomy', 'astrophysics', 'cosmic', 'rocket', 'aerospace'],
        'Informatica': ['computer', 'computing', 'algorithm', 'programming', 'software', 'hardware', 'technology', 'digital', 'it'],
        'AI_ML': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'ai', 'ml', 'classification', 'prediction'],
        'Web_development': ['web', 'website', 'html', 'css', 'javascript', 'frontend', 'backend', 'server', 'browser', 'http'],
        'System_programming': ['system', 'operating system', 'kernel', 'linux', 'unix', 'driver', 'embedded', 'real-time'],
        'Comunicazione': ['communication', 'media', 'social', 'network', 'information', 'signal', 'broadcast', 'telecom'],
        'Data_analysis': ['data', 'analysis', 'statistics', 'analytics', 'visualization', 'mining', 'big data', 'dataset'],
        'Database': ['database', 'sql', 'nosql', 'storage', 'dbms', 'query', 'indexing', 'data management'],
        'Security': ['security', 'cybersecurity', 'encryption', 'authentication', 'firewall', 'forensic', 'cryptography'],
        'Medicina': ['medicine', 'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment', 'diagnosis'],
        'Alimentazione': ['nutrition', 'food', 'diet', 'meal', 'dietary', 'eating', 'vitamin', 'dietitian'],
        'Cardiologia': ['cardiology', 'heart', 'cardiac', 'cardiovascular', 'coronary', 'artery'],
        'Oncologia': ['oncology', 'cancer', 'tumor', 'malignant', 'chemotherapy', 'radiation'],
        'Antropologia': ['anthropology', 'anthropological', 'human', 'culture', 'society', 'social'],
        'Archeologia': ['archaeology', 'archaeological', 'artifact', 'excavation', 'ancient'],
        'Linguistica': ['linguistic', 'language', 'linguistics', 'sociolinguistics'],
        'Culturale': ['cultural', 'culture', 'ethnography', 'ritual', 'tradition'],
        'Filosofia': ['philosophy', 'philosophical', 'ethics', 'metaphysics', 'logic'],
        'Paleontologia': ['paleontology', 'fossil', 'prehistoric', 'evolution', 'extinct'],
        'Animale': ['animal', 'fossil animal', 'vertebrate', 'mammal'],
        'Botanica': ['plant', 'fossil plant', 'botanical', 'flora'],
        'Umana': ['human evolution', 'hominid', 'ancestor', 'primitive human'],
        'Storia': ['history', 'historical', 'past', 'chronology', 'period', 'era'],
        'antica': ['ancient', 'antiquity', 'classical', 'roman', 'greek'],
        'moderna': ['modern', 'renaissance', 'enlightenment', 'industrial revolution'],
        'contemporanea': ['contemporary', 'modern', '19th', '20th', '21st', 'world war'],
        'Preistoria': ['prehistory', 'prehistoric', 'stone age', 'bronze age', 'iron age'],
        'Altro': []

}

if __name__ == "__main__":
    # Configurazione
    ontology_path = "./Ontology.owx"
    base_folder = "./References"
    csv_file = "training_set_all_files_with_altro_85percent.csv"  # AGGIORNATO

    # Carica ontologia
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")

    # Ottieni tutte le categorie - AGGIORNATO per includere "Altro"
    categories = set(['Scienza'])
    categories |= get_all_subclasses(g, NS['Scienza'])
    categories.add('Studi_umanistici')
    categories |= get_all_subclasses(g, NS['Studi_umanistici'])

    # Assicurati che "Altro" sia incluso
    categories.add('Altro')
    all_labels = sorted(categories)

    print(f"📊 Categorie totali (incluso 'Altro'): {len(all_labels)}")
    print(f"Categoria 'Altro' inclusa: {'Altro' in all_labels}")

    # Carica dataset
    df_train = load_dataset_with_category_labels(csv_file, base_folder, g, NS)
    print(f"📁 Dataset caricato: {len(df_train)} documenti")

    # Sottocampiona classi generali (incluso "Altro")
    general_labels = ['Scienza', 'Studi_umanistici', 'Altro']
    df_train_balanced = undersample_general_classes(df_train, 'labels', general_labels, threshold=0.5)
    print(f"⚖️ Dataset bilanciato: {len(df_train_balanced)} documenti")

    # Split train/validation
    class_counts = Counter(df_train['labels'].apply(lambda x: x[0] if x else 'Altro'))
    can_stratify = all(count >= 2 for count in class_counts.values()) and len(df_train) >= 2

    if can_stratify:
        train, val = train_test_split(df_train, test_size=0.2, random_state=42,
                                    stratify=df_train['labels'].apply(lambda x: x[0] if x else 'Altro'))
    else:
        train, val = train_test_split(df_train, test_size=0.2, random_state=42)

    # Vettorizzazione
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])

    y_train = np.array(train['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())
    y_val = np.array(val['labels'].apply(lambda labels: encode_labels(labels, all_labels)).tolist())

    print(f"🔧 Features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}")

    # Addestramento modelli
    print("\n🚀 ADDESTRAMENTO MODELLI:")
    print("=" * 50)

    # Logistic Regression
    clf_lr = OneVsRestClassifier(LogisticRegression(max_iter=200))
    model_lr = train_eval_model("Logistic Regression", clf_lr, X_train, y_train, X_val, y_val, all_labels)

    # SVM
    clf_svm = OneVsRestClassifier(SVC(kernel="linear", probability=True))
    model_svm = train_eval_model("SVM (linear kernel)", clf_svm, X_train, y_train, X_val, y_val, all_labels)

    # Random Forest
    clf_rf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
    model_rf = train_eval_model("Random Forest", clf_rf, X_train, y_train, X_val, y_val, all_labels)

    # Predizioni e filtraggio
    print("\n🎯 PREDIZIONI E FILTRAGGIO:")
    print("=" * 50)

    # Predizioni per Logistic Regression
    df_pred_lr = predict_top_categories(df_train, model_lr, vectorizer, all_labels)
    df_pred_lr['filtered_predicted_category'] = df_pred_lr['predicted_category'].apply(
        lambda labels: filter_general_superclasses(labels, g, NS))

    print("📋 Sample predictions (Logistic Regression):")
    print(df_pred_lr[['filename', 'filtered_predicted_category']].head())

    # Statistiche finali
    print("\n📊 STATISTICHE FINALI:")
    print("=" * 50)

    counter = Counter()
    for lablist in df_train['labels']:
        counter.update(lablist)

    print(f"Distribuzione categorie (Top 10):")
    for label, count in counter.most_common(10):
        print(f"{label:20}: {count:4d} ({count/len(df_train)*100:.1f}%)")

    # Statistiche specifiche per "Altro"
    altro_count = counter.get('Altro', 0)
    print(f"\n🎯 Categoria 'ALTRO': {altro_count} occorrenze ({altro_count/len(df_train)*100:.1f}% del dataset)")

    print("\n✅ Addestramento completato!")
    print(f"Modelli addestrati su {len(all_labels)} categorie (incluso 'Altro')")
