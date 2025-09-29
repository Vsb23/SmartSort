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
from sklearn.metrics import f1_score, classification_report, accuracy_score
from rdflib import Graph, Namespace, RDFS
import numpy as np
from collections import Counter


def get_all_subclasses(g, base_class_uri):
    """Ottiene tutte le sottoclassi di una classe base"""
    subclasses = set()
    direct_subclasses = set(s for s, p, o in g.triples((None, RDFS.subClassOf, base_class_uri)))
    
    for subclass in direct_subclasses:
        subclass_name = str(subclass).split('#')[-1]
        subclasses.add(subclass_name)
        subclasses |= get_all_subclasses(g, subclass)
    
    # Includi "Altro" come categoria valida
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


def get_most_specific_category(labels, g, ns):
    """
    NUOVA FUNZIONE: Restituisce SOLO la categoria più specifica
    Invece di aggiungere superclassi, trova la foglia più specifica
    """
    if not labels:
        return ['Altro']  # Default fallback
    
    # Se c'è "Altro", usalo solo se è l'unico
    if len(labels) == 1 and labels[0] == 'Altro':
        return ['Altro']
    
    # Rimuovi "Altro" se ci sono categorie più specifiche
    filtered_labels = [label for label in labels if label != 'Altro']
    if not filtered_labels:
        return ['Altro']
    
    # Trova la categoria più specifica (quella senza sottoclassi)
    most_specific = []
    
    for label in filtered_labels:
        try:
            # Controlla se questa categoria ha sottoclassi
            subclasses = get_all_subclasses(g, ns[label])
            # Se non ha sottoclassi tra le etichette correnti, è specifica
            has_subclass_in_labels = any(sub in filtered_labels for sub in subclasses if sub != label and sub != 'Altro')
            
            if not has_subclass_in_labels:
                most_specific.append(label)
        except:
            # Se non è nell'ontologia, considerala specifica
            most_specific.append(label)
    
    # Se non troviamo nulla di specifico, prendi la prima categoria
    return most_specific[:1] if most_specific else [filtered_labels[0]]


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


def load_dataset_with_single_category(csv_file, base_folder, g, ns):
    """
    MODIFICATA: Carica dataset con UNA SOLA categoria per documento
    """
    df = pd.read_csv(csv_file, low_memory=False)
    texts = []
    single_labels = []
    
    for idx, row in df.iterrows():
        if row['type'] == 'file' and row['extension'] == '.pdf':
            pdf_file_path = os.path.join(base_folder, row['filename'])
            text = extract_text_from_pdf(pdf_file_path)
            text = preprocess_text(text)
        else:
            text = ""
        
        texts.append(text)
        raw_labels = parse_labels(row['category'])
        
        # CHIAVE: Ottieni SOLO la categoria più specifica
        specific_label = get_most_specific_category(raw_labels, g, ns)
        single_labels.append(specific_label[0])  # Solo UNA categoria
    
    df['text'] = texts
    df['single_label'] = single_labels
    df_train = df[df['single_label'].notna()].copy()  # Solo con label valida
    
    return df_train


def train_eval_single_label_model(name, clf, X_train, y_train, X_val, y_val, all_labels):
    """
    MODIFICATA: Addestra e valuta modello per classificazione SINGLE-LABEL
    """
    print(f"---- Training {name} (Single-Label) ----")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    # Metriche per single-label
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")
    
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(f"{name} F1 weighted: {f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=None, zero_division=0))
    return clf


def predict_single_category(df, clf, vectorizer, all_labels):
    """
    NUOVA: Predice UNA SOLA categoria per documento
    """
    X = vectorizer.transform(df['text'])
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)
    
    # Ottieni confidenze
    confidences = []
    for i, pred in enumerate(predictions):
        pred_idx = all_labels.index(pred) if pred in all_labels else 0
        confidence = probabilities[i][pred_idx]
        confidences.append(confidence)
    
    df_result = df.copy()
    df_result['predicted_category'] = predictions
    df_result['confidence'] = confidences
    return df_result


def balance_single_label_dataset(df, label_col='single_label', max_samples_per_class=100):
    """
    NUOVA: Bilancia dataset per classificazione single-label
    """
    counter = Counter(df[label_col])
    print(f"📊 Distribuzione originale:")
    for label, count in counter.most_common(10):
        print(f"  {label}: {count}")
    
    # Sottocampiona classi con troppi esempi
    balanced_indices = []
    class_counts = Counter()
    
    # Priorità: prima le classi con meno esempi
    sorted_indices = df.index.tolist()
    np.random.shuffle(sorted_indices)  # Randomizza per evitare bias
    
    for idx in sorted_indices:
        label = df.loc[idx, label_col]
        if class_counts[label] < max_samples_per_class:
            balanced_indices.append(idx)
            class_counts[label] += 1
    
    df_balanced = df.loc[balanced_indices].copy()
    
    print(f"📊 Distribuzione bilanciata:")
    counter_balanced = Counter(df_balanced[label_col])
    for label, count in counter_balanced.most_common(10):
        print(f"  {label}: {count}")
    
    return df_balanced


# Keywords ontologiche (stesse di prima)
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
    'Altro': []  # Categoria catch-all
}


if __name__ == "__main__":
    # Configurazione
    ontology_path = "./Ontology.owx"
    base_folder = "./References"
    csv_file = "./training_set_single_category_85percent.csv"
    
    # Carica ontologia
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
    
    # Ottieni tutte le categorie
    categories = set(['Scienza'])
    categories |= get_all_subclasses(g, NS['Scienza'])
    categories.add('Studi_umanistici')
    categories |= get_all_subclasses(g, NS['Studi_umanistici'])
    categories.add('Altro')
    all_labels = sorted(list(categories))
    
    print(f"📊 Categorie totali: {len(all_labels)}")
    print("🔍 Prime 10 categorie:", all_labels[:10])
    
    # Carica dataset con UNA categoria per documento
    df_train = load_dataset_with_single_category(csv_file, base_folder, g, NS)
    print(f"📁 Dataset caricato: {len(df_train)} documenti")
    
    # Bilancia dataset
    df_train_balanced = balance_single_label_dataset(df_train, 'single_label', max_samples_per_class=50)
    print(f"⚖️ Dataset bilanciato: {len(df_train_balanced)} documenti")
    
    # Split train/validation - SINGLE LABEL
    if len(df_train_balanced) >= 10:  # Controllo minimo
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
    
    # Vettorizzazione
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    
    # Labels SINGLE - non più binary encoding
    y_train = train['single_label'].values
    y_val = val['single_label'].values
    
    print(f"🔧 Features: {X_train.shape[1]}, Training samples: {X_train.shape[0]}")
    print(f"📋 Unique labels in training: {len(np.unique(y_train))}")
    
    # Addestramento modelli SINGLE-LABEL
    print("\n🚀 ADDESTRAMENTO MODELLI (SINGLE-LABEL):")
    print("=" * 60)
    
    # Logistic Regression - NON più OneVsRestClassifier
    clf_lr = LogisticRegression(max_iter=200, multi_class='ovr')
    model_lr = train_eval_single_label_model("Logistic Regression", clf_lr, X_train, y_train, X_val, y_val, all_labels)
    
    # SVM
    clf_svm = SVC(kernel="linear", probability=True)
    model_svm = train_eval_single_label_model("SVM (linear)", clf_svm, X_train, y_train, X_val, y_val, all_labels)
    
    # Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    model_rf = train_eval_single_label_model("Random Forest", clf_rf, X_train, y_train, X_val, y_val, all_labels)
    
    # Predizioni SINGLE CATEGORY
    print("\n🎯 PREDIZIONI SINGLE-CATEGORY:")
    print("=" * 60)
    
    df_pred_lr = predict_single_category(df_train_balanced, model_lr, vectorizer, all_labels)
    
    print("📋 Sample predictions (Logistic Regression):")
    sample_preds = df_pred_lr[['filename', 'single_label', 'predicted_category', 'confidence']].head(10)
    print(sample_preds.to_string(index=False))
    
    # Statistiche finali
    print("\n📊 STATISTICHE FINALI (SINGLE-LABEL):")
    print("=" * 60)
    
    # Distribuzione categorie reali
    counter_real = Counter(df_train_balanced['single_label'])
    print("🏷️ Distribuzione categorie reali (Top 10):")
    for label, count in counter_real.most_common(10):
        print(f"{label:20}: {count:4d} ({count/len(df_train_balanced)*100:.1f}%)")
    
    # Distribuzione predizioni
    counter_pred = Counter(df_pred_lr['predicted_category'])
    print("\n🤖 Distribuzione predizioni (Top 10):")
    for label, count in counter_pred.most_common(10):
        print(f"{label:20}: {count:4d} ({count/len(df_pred_lr)*100:.1f}%)")
    
    # Accuracy complessiva
    correct_predictions = (df_pred_lr['single_label'] == df_pred_lr['predicted_category']).sum()
    overall_accuracy = correct_predictions / len(df_pred_lr)
    print(f"\n🎯 Accuracy complessiva: {overall_accuracy:.4f} ({correct_predictions}/{len(df_pred_lr)})")
    
    # Statistiche "Altro"
    altro_real = counter_real.get('Altro', 0)
    altro_pred = counter_pred.get('Altro', 0)
    print(f"\n🎯 Categoria 'ALTRO':")
    print(f"   Reale: {altro_real} ({altro_real/len(df_train_balanced)*100:.1f}%)")
    print(f"   Predetta: {altro_pred} ({altro_pred/len(df_pred_lr)*100:.1f}%)")
    
    print("\n✅ ADDESTRAMENTO SINGLE-LABEL COMPLETATO!")
    print("Ora ogni documento ha esattamente UNA categoria (la più specifica)")
          