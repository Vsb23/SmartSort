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
from sklearn.utils.class_weight import compute_class_weight
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
            # CORREZIONE: Usa relative_path per ricostruire il percorso completo
            rel = row.get('relative_path', '')
            pdf_file_path = os.path.join(base_folder, rel, row['filename'])
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

def compute_class_weights(y_train):
    """
    NUOVA: Calcola i pesi delle classi per bilanciare il dataset
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_train
    )
    weights_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    print("🔢 Pesi calcolati per le classi:")
    for cls, weight in sorted(weights_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {weight:.3f}")
    
    return weights_dict

def train_eval_single_label_model_weighted(name, clf, X_train, y_train, X_val, y_val, all_labels, class_weights=None):
    """
    MODIFICATA: Addestra e valuta modello con pesi delle classi
    """
    print(f"---- Training {name} (Single-Label con Class Weights) ----")
    
    # Se il modello supporta class_weight, lo impostiamo
    if hasattr(clf, 'estimator') and hasattr(clf.estimator, 'class_weight'):
        clf.estimator.class_weight = class_weights
    elif hasattr(clf, 'class_weight'):
        clf.class_weight = class_weights
    
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
        pred_idx = list(clf.classes_).index(pred) if pred in clf.classes_ else 0
        confidence = probabilities[i][pred_idx]
        confidences.append(confidence)
    
    df_result = df.copy()
    df_result['predicted_category'] = predictions
    df_result['confidence'] = confidences
    return df_result

def analyze_class_distribution(df, label_col='single_label'):
    """
    NUOVA: Analizza la distribuzione delle classi
    """
    counter = Counter(df[label_col])
    total = len(df)
    
    print(f"📊 Distribuzione delle classi (totale: {total}):")
    for label, count in counter.most_common():
        percentage = (count / total) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    return counter

# Keywords ontologiche (esempio - completa secondo le tue necessità)
ONTOLOGY_KEYWORDS = {
    'Biologia': ['biology', 'biological', 'organism', 'cellular', 'genetic', 'genome', 'ribonucleic', 'protein', 'enzyme', 'biotechnology'],
    
    'Ambiente': ['environment', 'environmental', 'climate', 'pollution', 'sustainability','green', 'ecosystem', 'conservation', 'biodiversity', 'renewable'],
    
    'Ecologia': ['ecology', 'ecological', 'habitat', 'species', 'wildlife', 'biome', 'population', 'community', 'niche', 'predator'],
    
    'Chimica': ['chemistry', 'chemical', 'compound', 'molecule', 'reaction', 'synthesis', 'catalyst', 'polymer', 'organic', 'inorganic'],
    
    'Fisica': ['physics', 'physical', 'quantum', 'mechanics', 'thermodynamics', 'electromagnetic', 'optics', 'particle', 'relativity', 'gravity'],
    
    'Energia': ['energy', 'power', 'solar', 'wind', 'nuclear', 'battery', 'fuel', 'electricity', 'turbine', 'generator'],
    
    'Spazio': ['space', 'satellite', 'orbit', 'planetary', 'astronomy', 'astrophysics', 'cosmic', 'rocket', 'aerospace', 'telescope'],
    
    'Informatica': ['computer', 'computing', 'algorithm', 'programming', 'software','hardware', 'digital', 'coding', 'processor', 'binary'],
    
    'AI_ML': ['artificial intelligence', 'machine learning', 'neural network', 'deep learning', 'classification', 'prediction', 'supervised', 'unsupervised', 'regression', 'clustering'],
    
    'Web_development': ['website', 'html', 'stylesheet', 'javascript', 'frontend', 'backend', 'server', 'browser', 'http', 'responsive'],
    
    'System_programming': ['system', 'operating system', 'kernel', 'linux', 'unix', 'driver', 'embedded', 'real-time', 'firmware', 'compiler'],
    
    'Comunicazione': ['communication', 'media', 'broadcast', 'information', 'messaging', 'signal', 'telecom', 'wireless', 'protocol', 'transmission'],
    
    'Data_analysis': ['data', 'analysis', 'statistics', 'analytics', 'visualization', 'mining', 'dataset', 'metrics', 'correlation', 'trend'],
    
    'Database': ['database', 'query', 'nosql', 'storage', 'dbms', 'indexing', 'table', 'schema', 'transaction', 'relational'],
    
    'Security': ['security', 'cybersecurity', 'encryption', 'authentication', 'firewall','forensic', 'cryptography', 'vulnerability', 'malware', 'intrusion'],
    
    'Medicina': ['medicine', 'medical', 'health', 'healthcare', 'clinical', 'patient', 'treatment', 'diagnosis', 'therapy', 'pharmaceutical'],
    
    'Alimentazione': ['nutrition', 'food', 'diet', 'meal', 'dietary','eating', 'vitamin', 'dietitian', 'calorie', 'nutrient'],
    
    'Cardiologia': ['cardiology', 'heart', 'cardiac', 'cardiovascular', 'coronary', 'artery', 'blood pressure', 'valve', 'rhythm', 'circulation'],
    
    'Oncologia': ['oncology', 'cancer', 'tumor', 'malignant', 'chemotherapy','radiation', 'metastasis', 'biopsy', 'carcinogen', 'remission'],
    
    'Antropologia': ['anthropology', 'anthropological', 'human', 'society', 'kinship','tribal', 'primitive', 'fieldwork', 'ethnology', 'cultural anthropology'],
    
    'Archeologia': ['archaeology', 'archaeological', 'artifact', 'excavation', 'site','pottery', 'burial', 'stratigraphy', 'dating', 'ruins'],
    
    'Linguistica': ['linguistic', 'language', 'linguistics', 'sociolinguistics', 'phonetics','grammar', 'syntax', 'semantics', 'morphology', 'dialect'],
    
    'Culturale': ['cultural', 'folklore', 'tradition', 'custom', 'belief', 'identity', 'heritage', 'ceremonial', 'symbolic', 'intercultural'],
    
    'Filosofia': ['philosophy', 'philosophical', 'ethics', 'metaphysics', 'logic','epistemology', 'ontology', 'moral', 'reason', 'wisdom'],
    
    'Paleontologia': ['paleontology', 'fossil', 'evolution', 'extinct', 'dinosaur','paleozoic', 'mesozoic', 'cenozoic', 'sedimentary', 'trilobite'],
    
    'Animale': ['animal', 'vertebrate', 'mammal', 'reptile', 'amphibian','bird', 'fish', 'skeleton', 'bone', 'spine'],
    
    'Botanica': ['plant', 'botanical', 'flora', 'leaf', 'root','stem', 'flower', 'seed', 'photosynthesis', 'chlorophyll'],
    
    'Umana': ['human evolution', 'hominid', 'ancestor', 'primitive human', 'homo sapiens','neanderthal', 'bipedal', 'cranium', 'primates', 'australopithecus'],
    
    'Storia': ['history', 'historical', 'past', 'chronology', 'period','epoch', 'civilization', 'empire', 'dynasty', 'chronicle'],
    
    'antica': ['antiquity', 'classical', 'roman', 'greek', 'egypt','mesopotamia', 'babylon', 'pharaoh', 'gladiator', 'colosseum'
    ],
    
    'moderna': ['modern', 'renaissance', 'enlightenment', 'industrial revolution', 'reformation','capitalism', 'colonialism', 'nationalism', 'democracy', 'monarchy'],
    
    'contemporanea': ['contemporary', '19th', '20th', '21st', 'world war','globalization', 'digitalization', 'internet age', 'terrorism', 'pandemic'],
    
    'Preistoria': ['prehistory', 'stone age', 'bronze age', 'iron age', 'neolithic','paleolithic', 'hunter gatherer', 'cave painting', 'megalith', 'dolmen'],
    'Altro': []}

if __name__ == "__main__":
    ontology_path = "./Ontology.owx"
    base_folder = "./References"
    csv_file = "training_set_single_category_85percent.csv"  # Assumo che il CSV si chiami output.csv dal create_csv.py
 
    # Carica ontologia
    g = Graph()
    g.parse(ontology_path, format='xml')
    NS = Namespace("http://www.semanticweb.org/vsb/ontologies/2025/8/untitled-ontology-11#")
 
    # Ottieni tutte le categorie dall'ontologia
    categories = set(['Scienza'])
    categories |= get_all_subclasses(g, NS['Scienza'])
    categories.add('Studi_umanistici')
    categories |= get_all_subclasses(g, NS['Studi_umanistici'])
    all_labels = sorted(categories)
 
    # Carica dataset
    df_train = load_dataset_with_single_category(csv_file, base_folder, g, NS)
    
    # DIAGNOSTICA: Analizza la distribuzione
    print(f"Dataset caricato: {len(df_train)} esempi")
    class_distribution = analyze_class_distribution(df_train)
    
    X = df_train['text']
    y = df_train['single_label']

    class_counts = Counter(y)
    classes_with_min_2 = [cls for cls, count in class_counts.items() if count >= 2]

    if len(classes_with_min_2) < 2:
        print("❌ Troppo poche classi con >=2 esempi. Split senza stratificazione.")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        # Filtra il dataset per includere solo le classi con >=2 esempi
        mask = y.isin(classes_with_min_2)
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
        )
    
    # Split train/validation
    X = df_train['text']
    y = df_train['single_label']
    
    # Controlla se abbiamo abbastanza dati per lo split
    if len(df_train) < 10:
        print("ATTENZIONE: Dataset troppo piccolo per split. Usando tutto per training.")
        X_train, X_val = X, X.iloc[:]  # Usa primi 2 per validazione fittizia
        y_train, y_val = y, y.iloc[:2]
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    print(f"Training set: {len(X_train)} esempi")
    print(f"Validation set: {len(X_val)} esempi")
    
    # Calcola pesi delle classi
    class_weights = compute_class_weights(y_train)
    
    # Vectorizzazione
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # MODELLI CON CLASS WEIGHTS
    print("\n" + "="*50)
    print("TRAINING MODELLI CON CLASS WEIGHTS")
    print("="*50)
    
    # Logistic Regression con pesi bilanciati
    clf_lr = LogisticRegression(
        max_iter=1000, 
        class_weight=class_weights,  # AGGIUNTO: pesi delle classi
        random_state=42
    )
    model_lr = train_eval_single_label_model_weighted(
        "Logistic Regression", clf_lr, X_train_vec, y_train, X_val_vec, y_val, all_labels, class_weights
    )
    
    # SVM con pesi bilanciati
    clf_svm = SVC(
        kernel="linear", 
        probability=True,
        class_weight=class_weights,  # AGGIUNTO: pesi delle classi
        random_state=42
    )
    model_svm = train_eval_single_label_model_weighted(
        "SVM (linear)", clf_svm, X_train_vec, y_train, X_val_vec, y_val, all_labels, class_weights
    )
    
    # Random Forest con pesi bilanciati
    clf_rf = RandomForestClassifier(
        n_estimators=100, 
        class_weight=class_weights,  # AGGIUNTO: pesi delle classi
        random_state=42
    )
    model_rf = train_eval_single_label_model_weighted(
        "Random Forest", clf_rf, X_train_vec, y_train, X_val_vec, y_val, all_labels, class_weights
    )
    
    # Predizioni con modello migliore (esempio con Logistic Regression)
    print("\n" + "="*50)
    print("PREDIZIONI SU TUTTO IL DATASET")
    print("="*50)
    
    df_predictions = predict_single_category(df_train, model_lr, vectorizer, all_labels)
    
    # Mostra alcune predizioni
    print("\nEsempi di predizioni:")
    sample_predictions = df_predictions[['filename', 'single_label', 'predicted_category', 'confidence']].head(10)
    for _, row in sample_predictions.iterrows():
        print(f"File: {row['filename']}")
        print(f"  Vero: {row['single_label']} | Predetto: {row['predicted_category']} | Confidenza: {row['confidence']:.3f}")
    
    # Statistiche finali
    correct_predictions = (df_predictions['single_label'] == df_predictions['predicted_category']).sum()
    total_predictions = len(df_predictions)
    accuracy_full = correct_predictions / total_predictions
    
    print(f"\nAccuracy su tutto il dataset: {accuracy_full:.3f} ({correct_predictions}/{total_predictions})")
    
    # Analisi errori per classe
    print("\nAnalisi errori per classe:")
    error_analysis = df_predictions[df_predictions['single_label'] != df_predictions['predicted_category']]
    if len(error_analysis) > 0:
        error_counter = Counter(error_analysis['single_label'])
        for label, errors in error_counter.most_common(5):
            total_for_class = (df_predictions['single_label'] == label).sum()
            error_rate = errors / total_for_class if total_for_class > 0 else 0
            print(f"  {label}: {errors}/{total_for_class} errori (tasso errore: {error_rate:.2f})")
    
    print("\n🎯 Training completato con class weighting!")